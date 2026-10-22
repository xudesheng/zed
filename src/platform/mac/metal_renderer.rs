use super::metal_atlas::MetalAtlas;
use crate::{
    AtlasTextureId, Background, Bounds, ContentMask, DevicePixels, MonochromeSprite, PaintSurface,
    Path, Point, PolychromeSprite, PrimitiveBatch, Quad, ScaledPixels, Scene, Shadow, Size,
    Surface, Underline, point, size,
};
use anyhow::Result;
use block::ConcreteBlock;
use cocoa::{
    base::{NO, YES},
    foundation::{NSSize, NSUInteger},
    quartzcore::AutoresizingMask,
};

use core_foundation::base::TCFType;
use core_video::{
    metal_texture::CVMetalTextureGetTexture, metal_texture_cache::CVMetalTextureCache,
    pixel_buffer::kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
};
use foreign_types::{ForeignType, ForeignTypeRef};
use metal::{
    CAMetalLayer, CommandQueue, MTLPixelFormat, MTLResourceOptions, NSRange,
    RenderPassColorAttachmentDescriptorRef,
};
use objc::{self, msg_send, sel, sel_impl};
use parking_lot::Mutex;

use std::{cell::Cell, ffi::c_void, mem, ptr, sync::Arc};

// Exported to metal
pub(crate) type PointF = crate::Point<f32>;

#[cfg(not(feature = "runtime_shaders"))]
const SHADERS_METALLIB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shaders.metallib"));
#[cfg(feature = "runtime_shaders")]
const SHADERS_SOURCE_FILE: &str = include_str!(concat!(env!("OUT_DIR"), "/stitched_shaders.metal"));
// Use 4x MSAA, all devices support it.
// https://developer.apple.com/documentation/metal/mtldevice/1433355-supportstexturesamplecount
const PATH_SAMPLE_COUNT: u32 = 4;

pub type Context = Arc<Mutex<InstanceBufferPool>>;
pub type Renderer = MetalRenderer;

pub unsafe fn new_renderer(
    context: self::Context,
    _native_window: *mut c_void,
    _native_view: *mut c_void,
    _bounds: crate::Size<f32>,
    _transparent: bool,
) -> Renderer {
    MetalRenderer::new(context)
}

pub(crate) struct InstanceBufferPool {
    buffer_size: usize,
    buffers: Vec<metal::Buffer>,
}

impl Default for InstanceBufferPool {
    fn default() -> Self {
        Self {
            buffer_size: 2 * 1024 * 1024,
            buffers: Vec::new(),
        }
    }
}

pub(crate) struct InstanceBuffer {
    metal_buffer: metal::Buffer,
    size: usize,
}

impl InstanceBufferPool {
    pub(crate) fn reset(&mut self, buffer_size: usize) {
        self.buffer_size = buffer_size;
        self.buffers.clear();
    }

    pub(crate) fn acquire(&mut self, device: &metal::Device) -> InstanceBuffer {
        let buffer = self.buffers.pop().unwrap_or_else(|| {
            device.new_buffer(
                self.buffer_size as u64,
                MTLResourceOptions::StorageModeManaged,
            )
        });
        InstanceBuffer {
            metal_buffer: buffer,
            size: self.buffer_size,
        }
    }

    pub(crate) fn release(&mut self, buffer: InstanceBuffer) {
        if buffer.size == self.buffer_size {
            self.buffers.push(buffer.metal_buffer)
        }
    }
}

pub(crate) struct MetalRenderer {
    device: metal::Device,
    layer: metal::MetalLayer,
    presents_with_transaction: bool,
    command_queue: CommandQueue,
    paths_rasterization_pipeline_state: metal::RenderPipelineState,
    path_sprites_pipeline_state: metal::RenderPipelineState,
    shadows_pipeline_state: metal::RenderPipelineState,
    quads_pipeline_state: metal::RenderPipelineState,
    underlines_pipeline_state: metal::RenderPipelineState,
    monochrome_sprites_pipeline_state: metal::RenderPipelineState,
    polychrome_sprites_pipeline_state: metal::RenderPipelineState,
    surfaces_pipeline_state: metal::RenderPipelineState,
    unit_vertices: metal::Buffer,
    #[allow(clippy::arc_with_non_send_sync)]
    instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>,
    sprite_atlas: Arc<MetalAtlas>,
    core_video_texture_cache: core_video::metal_texture_cache::CVMetalTextureCache,
    path_intermediate_texture: Option<metal::Texture>,
    path_intermediate_msaa_texture: Option<metal::Texture>,
    path_sample_count: u32,
    isolated_pool: crate::isolated::TargetPool,
    isolated_pipelines: Vec<metal::RenderPipelineState>,
    isolated_depth: usize,
    composite_pipeline: metal::RenderPipelineState,
}

#[repr(C)]
pub struct PathRasterizationVertex {
    pub xy_position: Point<ScaledPixels>,
    pub st_position: Point<f32>,
    pub color: Background,
    pub bounds: Bounds<ScaledPixels>,
}

impl MetalRenderer {
    pub fn new(instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>) -> Self {
        // Prefer low‐power integrated GPUs on Intel Mac. On Apple
        // Silicon, there is only ever one GPU, so this is equivalent to
        // `metal::Device::system_default()`.
        let mut devices = metal::Device::all();
        devices.sort_by_key(|device| (device.is_removable(), device.is_low_power()));
        let Some(device) = devices.pop() else {
            log::error!("unable to access a compatible graphics device");
            std::process::exit(1);
        };

        let layer = metal::MetalLayer::new();
        layer.set_device(&device);
        layer.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        layer.set_opaque(false);
        layer.set_maximum_drawable_count(3);
        unsafe {
            let _: () = msg_send![&*layer, setAllowsNextDrawableTimeout: NO];
            let _: () = msg_send![&*layer, setNeedsDisplayOnBoundsChange: YES];
            let _: () = msg_send![
                &*layer,
                setAutoresizingMask: AutoresizingMask::WIDTH_SIZABLE
                    | AutoresizingMask::HEIGHT_SIZABLE
            ];
        }
        #[cfg(feature = "runtime_shaders")]
        let library = device
            .new_library_with_source(&SHADERS_SOURCE_FILE, &metal::CompileOptions::new())
            .expect("error building metal library");
        #[cfg(not(feature = "runtime_shaders"))]
        let library = device
            .new_library_with_data(SHADERS_METALLIB)
            .expect("error building metal library");

        fn to_float2_bits(point: PointF) -> u64 {
            let mut output = point.y.to_bits() as u64;
            output <<= 32;
            output |= point.x.to_bits() as u64;
            output
        }

        let unit_vertices = [
            to_float2_bits(point(0., 0.)),
            to_float2_bits(point(1., 0.)),
            to_float2_bits(point(0., 1.)),
            to_float2_bits(point(0., 1.)),
            to_float2_bits(point(1., 0.)),
            to_float2_bits(point(1., 1.)),
        ];
        let unit_vertices = device.new_buffer_with_data(
            unit_vertices.as_ptr() as *const c_void,
            mem::size_of_val(&unit_vertices) as u64,
            MTLResourceOptions::StorageModeManaged,
        );

        let paths_rasterization_pipeline_state = build_path_rasterization_pipeline_state(
            &device,
            &library,
            "paths_rasterization",
            "path_rasterization_vertex",
            "path_rasterization_fragment",
            MTLPixelFormat::BGRA8Unorm,
            PATH_SAMPLE_COUNT,
        );
        let path_sprites_pipeline_state = build_path_sprite_pipeline_state(
            &device,
            &library,
            "path_sprites",
            "path_sprite_vertex",
            "path_sprite_fragment",
            MTLPixelFormat::BGRA8Unorm,
            false,
        );
        let shadows_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "shadows",
            "shadow_vertex",
            "shadow_fragment",
            MTLPixelFormat::BGRA8Unorm,
            false,
        );
        let quads_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "quads",
            "quad_vertex",
            "quad_fragment",
            MTLPixelFormat::BGRA8Unorm,
            false,
        );
        let underlines_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "underlines",
            "underline_vertex",
            "underline_fragment",
            MTLPixelFormat::BGRA8Unorm,
            false,
        );
        let monochrome_sprites_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "monochrome_sprites",
            "monochrome_sprite_vertex",
            "monochrome_sprite_fragment",
            MTLPixelFormat::BGRA8Unorm,
            false,
        );
        let polychrome_sprites_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "polychrome_sprites",
            "polychrome_sprite_vertex",
            "polychrome_sprite_fragment",
            MTLPixelFormat::BGRA8Unorm,
            false,
        );
        let surfaces_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "surfaces",
            "surface_vertex",
            "surface_fragment",
            MTLPixelFormat::BGRA8Unorm,
            false,
        );

        let command_queue = device.new_command_queue();
        let sprite_atlas = Arc::new(MetalAtlas::new(device.clone()));
        let core_video_texture_cache =
            CVMetalTextureCache::new(None, device.clone(), None).unwrap();

        let isolated_pipelines = vec![
            build_pipeline_state(
                &device,
                &library,
                "isolated_shadows",
                "shadow_vertex",
                "shadow_fragment",
                MTLPixelFormat::BGRA8Unorm,
                true,
            ),
            build_pipeline_state(
                &device,
                &library,
                "isolated_quads",
                "quad_vertex",
                "quad_fragment",
                MTLPixelFormat::BGRA8Unorm,
                true,
            ),
            build_pipeline_state(
                &device,
                &library,
                "isolated_underlines",
                "underline_vertex",
                "underline_fragment",
                MTLPixelFormat::BGRA8Unorm,
                true,
            ),
            build_pipeline_state(
                &device,
                &library,
                "isolated_monochrome_sprites",
                "monochrome_sprite_vertex",
                "monochrome_sprite_fragment",
                MTLPixelFormat::BGRA8Unorm,
                true,
            ),
            build_pipeline_state(
                &device,
                &library,
                "isolated_polychrome_sprites",
                "polychrome_sprite_vertex",
                "polychrome_sprite_fragment",
                MTLPixelFormat::BGRA8Unorm,
                true,
            ),
            build_pipeline_state(
                &device,
                &library,
                "isolated_surfaces",
                "surface_vertex",
                "surface_fragment",
                MTLPixelFormat::BGRA8Unorm,
                true,
            ),
            build_path_sprite_pipeline_state(
                &device,
                &library,
                "isolated_paths",
                "path_sprite_vertex",
                "path_sprite_fragment",
                MTLPixelFormat::BGRA8Unorm,
                true,
            ),
        ];
        let composite_pipeline = build_path_sprite_pipeline_state(
            &device,
            &library,
            "isolated_composite",
            "isolated_vertex",
            "isolated_fragment",
            MTLPixelFormat::BGRA8Unorm,
            true,
        );
        Self {
            device,
            layer,
            presents_with_transaction: false,
            command_queue,
            paths_rasterization_pipeline_state,
            path_sprites_pipeline_state,
            shadows_pipeline_state,
            quads_pipeline_state,
            underlines_pipeline_state,
            monochrome_sprites_pipeline_state,
            polychrome_sprites_pipeline_state,
            surfaces_pipeline_state,
            unit_vertices,
            instance_buffer_pool,
            sprite_atlas,
            core_video_texture_cache,
            path_intermediate_texture: None,
            path_intermediate_msaa_texture: None,
            path_sample_count: PATH_SAMPLE_COUNT,
            isolated_pool: Default::default(),
            isolated_pipelines,
            isolated_depth: 0,
            composite_pipeline,
        }
    }

    pub fn layer(&self) -> &metal::MetalLayerRef {
        &self.layer
    }

    pub fn layer_ptr(&self) -> *mut CAMetalLayer {
        self.layer.as_ptr()
    }

    pub fn sprite_atlas(&self) -> &Arc<MetalAtlas> {
        &self.sprite_atlas
    }

    pub fn set_presents_with_transaction(&mut self, presents_with_transaction: bool) {
        self.presents_with_transaction = presents_with_transaction;
        self.layer
            .set_presents_with_transaction(presents_with_transaction);
    }

    pub fn update_drawable_size(&mut self, size: Size<DevicePixels>) {
        let size = NSSize {
            width: size.width.0 as f64,
            height: size.height.0 as f64,
        };
        unsafe {
            let _: () = msg_send![
                self.layer(),
                setDrawableSize: size
            ];
        }
        let device_pixels_size = Size {
            width: DevicePixels(size.width as i32),
            height: DevicePixels(size.height as i32),
        };
        self.update_path_intermediate_textures(device_pixels_size);
    }

    fn update_path_intermediate_textures(&mut self, size: Size<DevicePixels>) {
        // We are uncertain when this happens, but sometimes size can be 0 here. Most likely before
        // the layout pass on window creation. Zero-sized texture creation causes SIGABRT.
        // https://github.com/zed-industries/zed/issues/36229
        if size.width.0 <= 0 || size.height.0 <= 0 {
            self.path_intermediate_texture = None;
            self.path_intermediate_msaa_texture = None;
            return;
        }

        let texture_descriptor = metal::TextureDescriptor::new();
        texture_descriptor.set_width(size.width.0 as u64);
        texture_descriptor.set_height(size.height.0 as u64);
        texture_descriptor.set_pixel_format(metal::MTLPixelFormat::BGRA8Unorm);
        texture_descriptor
            .set_usage(metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead);
        self.path_intermediate_texture = Some(self.device.new_texture(&texture_descriptor));

        if self.path_sample_count > 1 {
            let mut msaa_descriptor = texture_descriptor;
            msaa_descriptor.set_texture_type(metal::MTLTextureType::D2Multisample);
            msaa_descriptor.set_storage_mode(metal::MTLStorageMode::Private);
            msaa_descriptor.set_sample_count(self.path_sample_count as _);
            self.path_intermediate_msaa_texture = Some(self.device.new_texture(&msaa_descriptor));
        } else {
            self.path_intermediate_msaa_texture = None;
        }
    }

    pub fn update_transparency(&self, _transparent: bool) {
        // todo(mac)?
    }

    pub fn destroy(&self) {
        // nothing to do
    }

    pub fn draw(&mut self, scene: &Scene) {
        let layer = self.layer.clone();
        let viewport_size = layer.drawable_size();
        let viewport_size: Size<DevicePixels> = size(
            (viewport_size.width.ceil() as i32).into(),
            (viewport_size.height.ceil() as i32).into(),
        );
        let drawable = if let Some(drawable) = layer.next_drawable() {
            drawable
        } else {
            // Drawable unavailability is a transient frame skip.
            log::error!(
                "failed to retrieve next drawable, drawable size: {:?}",
                viewport_size
            );
            return;
        };

        loop {
            let mut instance_buffer = self.instance_buffer_pool.lock().acquire(&self.device);

            let command_buffer =
                self.draw_primitives(scene, &mut instance_buffer, drawable, viewport_size);

            match command_buffer {
                Ok(command_buffer) => {
                    let instance_buffer_pool = self.instance_buffer_pool.clone();
                    let instance_buffer = Cell::new(Some(instance_buffer));
                    let block = ConcreteBlock::new(move |_| {
                        if let Some(instance_buffer) = instance_buffer.take() {
                            instance_buffer_pool.lock().release(instance_buffer);
                        }
                    });
                    let block = block.copy();
                    command_buffer.add_completed_handler(&block);

                    if self.presents_with_transaction {
                        command_buffer.commit();
                        command_buffer.wait_until_scheduled();
                        drawable.present();
                    } else {
                        command_buffer.present_drawable(drawable);
                        command_buffer.commit();
                    }
                    return;
                }
                Err(err) => {
                    log::error!(
                        "failed to render: {}. retrying with larger instance buffer size",
                        err
                    );
                    let mut instance_buffer_pool = self.instance_buffer_pool.lock();
                    let buffer_size = instance_buffer_pool.buffer_size;
                    if buffer_size >= 256 * 1024 * 1024 {
                        // Instance-buffer pressure skips this frame; the
                        // prepared target remains valid for a later retry.
                        log::error!("instance buffer size grew too large: {}", buffer_size);
                        break;
                    }
                    instance_buffer_pool.reset(buffer_size * 2);
                    log::info!(
                        "increased instance buffer size to {}",
                        instance_buffer_pool.buffer_size
                    );
                }
            }
        }
    }

    fn draw_primitives(
        &mut self,
        scene: &Scene,
        instance_buffer: &mut InstanceBuffer,
        drawable: &metal::MetalDrawableRef,
        viewport_size: Size<DevicePixels>,
    ) -> Result<metal::CommandBuffer> {
        let command_queue = self.command_queue.clone();
        let command_buffer = command_queue.new_command_buffer();
        retain_isolated_resources(scene, command_buffer);
        let mut instance_offset = 0;
        self.encode_scene(
            scene,
            instance_buffer,
            &mut instance_offset,
            command_buffer,
            drawable.texture(),
            viewport_size,
            if self.layer.is_opaque() { 1. } else { 0. },
        )?;
        instance_buffer.metal_buffer.did_modify_range(NSRange {
            location: 0,
            length: instance_offset as NSUInteger,
        });
        Ok(command_buffer.to_owned())
    }

    fn encode_scene(
        &mut self,
        scene: &Scene,
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        command_buffer: &metal::CommandBufferRef,
        texture: &metal::TextureRef,
        viewport_size: Size<DevicePixels>,
        alpha: f64,
    ) -> Result<()> {
        self.encode_batches(
            scene,
            instance_buffer,
            instance_offset,
            command_buffer,
            texture,
            viewport_size,
            alpha,
        )
    }

    fn encode_isolated_scene(
        &mut self,
        scene: &Scene,
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        command_buffer: &metal::CommandBufferRef,
        texture: &metal::TextureRef,
        viewport_size: Size<DevicePixels>,
    ) -> Result<()> {
        if self.isolated_depth == 0 {
            self.swap_isolated_pipelines();
        }
        self.isolated_depth += 1;
        let result = self.encode_batches(
            scene,
            instance_buffer,
            instance_offset,
            command_buffer,
            texture,
            viewport_size,
            0.,
        );
        self.isolated_depth -= 1;
        if self.isolated_depth == 0 {
            self.swap_isolated_pipelines();
        }
        result
    }
    fn swap_isolated_pipelines(&mut self) {
        std::mem::swap(
            &mut self.shadows_pipeline_state,
            &mut self.isolated_pipelines[0],
        );
        std::mem::swap(
            &mut self.quads_pipeline_state,
            &mut self.isolated_pipelines[1],
        );
        std::mem::swap(
            &mut self.underlines_pipeline_state,
            &mut self.isolated_pipelines[2],
        );
        std::mem::swap(
            &mut self.monochrome_sprites_pipeline_state,
            &mut self.isolated_pipelines[3],
        );
        std::mem::swap(
            &mut self.polychrome_sprites_pipeline_state,
            &mut self.isolated_pipelines[4],
        );
        std::mem::swap(
            &mut self.surfaces_pipeline_state,
            &mut self.isolated_pipelines[5],
        );
        std::mem::swap(
            &mut self.path_sprites_pipeline_state,
            &mut self.isolated_pipelines[6],
        );
    }
    fn encode_batches(
        &mut self,
        scene: &Scene,
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        command_buffer: &metal::CommandBufferRef,
        texture: &metal::TextureRef,
        viewport_size: Size<DevicePixels>,
        alpha: f64,
    ) -> Result<()> {
        let mut command_encoder =
            new_command_encoder(command_buffer, texture, viewport_size, |color_attachment| {
                color_attachment.set_load_action(metal::MTLLoadAction::Clear);
                color_attachment.set_clear_color(metal::MTLClearColor::new(0., 0., 0., alpha));
            });

        for batch in scene.batches() {
            let ok = match batch {
                PrimitiveBatch::Isolated(layer) => {
                    // Cached paint can replay a target failed after Codi's
                    // preflight. Omit only that composite, not the window or
                    // unrelated siblings. The owner is woken by failure().
                    if layer.target.is_failed() {
                        continue;
                    }
                    // Render and consume this placement before another placement
                    // reuses the same prepared target. Descendants still finish
                    // before their parent composite; tracked heap hazards order
                    // every write/sample across these encoder boundaries.
                    command_encoder.end_encoding();
                    let textures = &layer.target.metal;
                    let old = self
                        .path_intermediate_texture
                        .replace(textures.paths.clone());
                    let old_msaa = self
                        .path_intermediate_msaa_texture
                        .replace(textures.msaa.clone());
                    let result = self.encode_isolated_scene(
                        &layer.scene,
                        instance_buffer,
                        instance_offset,
                        command_buffer,
                        &textures.color,
                        textures.pixel_size.map(|v| DevicePixels(v as i32)),
                    );
                    self.path_intermediate_texture = old;
                    self.path_intermediate_msaa_texture = old_msaa;
                    command_encoder =
                        new_command_encoder(command_buffer, texture, viewport_size, |attachment| {
                            attachment.set_load_action(metal::MTLLoadAction::Load);
                        });
                    result.is_ok() && self.draw_isolated(layer, viewport_size, command_encoder)
                }
                PrimitiveBatch::Shadows(shadows) => self.draw_shadows(
                    shadows,
                    instance_buffer,
                    instance_offset,
                    viewport_size,
                    command_encoder,
                ),
                PrimitiveBatch::Quads(quads) => self.draw_quads(
                    quads,
                    instance_buffer,
                    instance_offset,
                    viewport_size,
                    command_encoder,
                ),
                PrimitiveBatch::Paths(paths) => {
                    command_encoder.end_encoding();

                    let did_draw = self.draw_paths_to_intermediate(
                        paths,
                        instance_buffer,
                        instance_offset,
                        viewport_size,
                        command_buffer,
                    );

                    command_encoder = new_command_encoder(
                        command_buffer,
                        texture,
                        viewport_size,
                        |color_attachment| {
                            color_attachment.set_load_action(metal::MTLLoadAction::Load);
                        },
                    );

                    if did_draw {
                        self.draw_paths_from_intermediate(
                            paths,
                            instance_buffer,
                            instance_offset,
                            viewport_size,
                            command_encoder,
                        )
                    } else {
                        false
                    }
                }
                PrimitiveBatch::Underlines(underlines) => self.draw_underlines(
                    underlines,
                    instance_buffer,
                    instance_offset,
                    viewport_size,
                    command_encoder,
                ),
                PrimitiveBatch::MonochromeSprites {
                    texture_id,
                    sprites,
                } => self.draw_monochrome_sprites(
                    texture_id,
                    sprites,
                    instance_buffer,
                    instance_offset,
                    viewport_size,
                    command_encoder,
                ),
                PrimitiveBatch::PolychromeSprites {
                    texture_id,
                    sprites,
                } => self.draw_polychrome_sprites(
                    texture_id,
                    sprites,
                    instance_buffer,
                    instance_offset,
                    viewport_size,
                    command_encoder,
                ),
                PrimitiveBatch::Surfaces(surfaces) => self.draw_surfaces(
                    surfaces,
                    instance_buffer,
                    instance_offset,
                    viewport_size,
                    command_encoder,
                ),
            };
            if !ok {
                command_encoder.end_encoding();
                anyhow::bail!(
                    "scene too large: {} paths, {} shadows, {} quads, {} underlines, {} mono, {} poly, {} surfaces",
                    scene.paths.len(),
                    scene.shadows.len(),
                    scene.quads.len(),
                    scene.underlines.len(),
                    scene.monochrome_sprites.len(),
                    scene.polychrome_sprites.len(),
                    scene.surfaces.len(),
                );
            }
        }

        command_encoder.end_encoding();

        Ok(())
    }

    fn draw_paths_to_intermediate(
        &self,
        paths: &[Path<ScaledPixels>],
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        viewport_size: Size<DevicePixels>,
        command_buffer: &metal::CommandBufferRef,
    ) -> bool {
        if paths.is_empty() {
            return true;
        }
        let Some(intermediate_texture) = &self.path_intermediate_texture else {
            return false;
        };

        let render_pass_descriptor = metal::RenderPassDescriptor::new();
        let color_attachment = render_pass_descriptor
            .color_attachments()
            .object_at(0)
            .unwrap();
        color_attachment.set_load_action(metal::MTLLoadAction::Clear);
        color_attachment.set_clear_color(metal::MTLClearColor::new(0., 0., 0., 0.));

        if let Some(msaa_texture) = &self.path_intermediate_msaa_texture {
            color_attachment.set_texture(Some(msaa_texture));
            color_attachment.set_resolve_texture(Some(intermediate_texture));
            color_attachment.set_store_action(metal::MTLStoreAction::MultisampleResolve);
        } else {
            color_attachment.set_texture(Some(intermediate_texture));
            color_attachment.set_store_action(metal::MTLStoreAction::Store);
        }

        let command_encoder = command_buffer.new_render_command_encoder(render_pass_descriptor);
        command_encoder.set_render_pipeline_state(&self.paths_rasterization_pipeline_state);

        align_offset(instance_offset);
        let mut vertices = Vec::new();
        for path in paths {
            vertices.extend(path.vertices.iter().map(|v| PathRasterizationVertex {
                xy_position: v.xy_position,
                st_position: v.st_position,
                color: path.color,
                bounds: path.bounds.intersect(&path.content_mask.bounds),
            }));
        }
        let vertices_bytes_len = mem::size_of_val(vertices.as_slice());
        let next_offset = *instance_offset + vertices_bytes_len;
        if next_offset > instance_buffer.size {
            command_encoder.end_encoding();
            return false;
        }
        command_encoder.set_vertex_buffer(
            PathRasterizationInputIndex::Vertices as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );
        command_encoder.set_vertex_bytes(
            PathRasterizationInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_fragment_buffer(
            PathRasterizationInputIndex::Vertices as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );
        let buffer_contents =
            unsafe { (instance_buffer.metal_buffer.contents() as *mut u8).add(*instance_offset) };
        unsafe {
            ptr::copy_nonoverlapping(
                vertices.as_ptr() as *const u8,
                buffer_contents,
                vertices_bytes_len,
            );
        }
        command_encoder.draw_primitives(
            metal::MTLPrimitiveType::Triangle,
            0,
            vertices.len() as u64,
        );
        *instance_offset = next_offset;

        command_encoder.end_encoding();
        true
    }

    fn draw_shadows(
        &self,
        shadows: &[Shadow],
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) -> bool {
        if shadows.is_empty() {
            return true;
        }
        align_offset(instance_offset);

        command_encoder.set_render_pipeline_state(&self.shadows_pipeline_state);
        command_encoder.set_vertex_buffer(
            ShadowInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            ShadowInputIndex::Shadows as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );
        command_encoder.set_fragment_buffer(
            ShadowInputIndex::Shadows as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );

        command_encoder.set_vertex_bytes(
            ShadowInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        let shadow_bytes_len = mem::size_of_val(shadows);
        let buffer_contents =
            unsafe { (instance_buffer.metal_buffer.contents() as *mut u8).add(*instance_offset) };

        let next_offset = *instance_offset + shadow_bytes_len;
        if next_offset > instance_buffer.size {
            return false;
        }

        unsafe {
            ptr::copy_nonoverlapping(
                shadows.as_ptr() as *const u8,
                buffer_contents,
                shadow_bytes_len,
            );
        }

        command_encoder.draw_primitives_instanced(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            shadows.len() as u64,
        );
        *instance_offset = next_offset;
        true
    }

    fn draw_quads(
        &self,
        quads: &[Quad],
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) -> bool {
        if quads.is_empty() {
            return true;
        }
        align_offset(instance_offset);

        command_encoder.set_render_pipeline_state(&self.quads_pipeline_state);
        command_encoder.set_vertex_buffer(
            QuadInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            QuadInputIndex::Quads as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );
        command_encoder.set_fragment_buffer(
            QuadInputIndex::Quads as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );

        command_encoder.set_vertex_bytes(
            QuadInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        let quad_bytes_len = mem::size_of_val(quads);
        let buffer_contents =
            unsafe { (instance_buffer.metal_buffer.contents() as *mut u8).add(*instance_offset) };

        let next_offset = *instance_offset + quad_bytes_len;
        if next_offset > instance_buffer.size {
            return false;
        }

        unsafe {
            ptr::copy_nonoverlapping(quads.as_ptr() as *const u8, buffer_contents, quad_bytes_len);
        }

        command_encoder.draw_primitives_instanced(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            quads.len() as u64,
        );
        *instance_offset = next_offset;
        true
    }

    fn draw_paths_from_intermediate(
        &self,
        paths: &[Path<ScaledPixels>],
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) -> bool {
        let Some(first_path) = paths.first() else {
            return true;
        };

        let Some(ref intermediate_texture) = self.path_intermediate_texture else {
            return false;
        };

        command_encoder.set_render_pipeline_state(&self.path_sprites_pipeline_state);
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        command_encoder.set_fragment_texture(
            SpriteInputIndex::AtlasTexture as u64,
            Some(intermediate_texture),
        );

        // When copying paths from the intermediate texture to the drawable,
        // each pixel must only be copied once, in case of transparent paths.
        //
        // If all paths have the same draw order, then their bounds are all
        // disjoint, so we can copy each path's bounds individually. If this
        // batch combines different draw orders, we perform a single copy
        // for a minimal spanning rect.
        let sprites;
        if paths.last().unwrap().order == first_path.order {
            sprites = paths
                .iter()
                .map(|path| PathSprite {
                    bounds: path.clipped_bounds(),
                })
                .collect();
        } else {
            let mut bounds = first_path.clipped_bounds();
            for path in paths.iter().skip(1) {
                bounds = bounds.union(&path.clipped_bounds());
            }
            sprites = vec![PathSprite { bounds }];
        }

        align_offset(instance_offset);
        let sprite_bytes_len = mem::size_of_val(sprites.as_slice());
        let next_offset = *instance_offset + sprite_bytes_len;
        if next_offset > instance_buffer.size {
            return false;
        }

        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );

        let buffer_contents =
            unsafe { (instance_buffer.metal_buffer.contents() as *mut u8).add(*instance_offset) };
        unsafe {
            ptr::copy_nonoverlapping(
                sprites.as_ptr() as *const u8,
                buffer_contents,
                sprite_bytes_len,
            );
        }

        command_encoder.draw_primitives_instanced(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            sprites.len() as u64,
        );
        *instance_offset = next_offset;

        true
    }

    fn draw_underlines(
        &self,
        underlines: &[Underline],
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) -> bool {
        if underlines.is_empty() {
            return true;
        }
        align_offset(instance_offset);

        command_encoder.set_render_pipeline_state(&self.underlines_pipeline_state);
        command_encoder.set_vertex_buffer(
            UnderlineInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            UnderlineInputIndex::Underlines as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );
        command_encoder.set_fragment_buffer(
            UnderlineInputIndex::Underlines as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );

        command_encoder.set_vertex_bytes(
            UnderlineInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        let underline_bytes_len = mem::size_of_val(underlines);
        let buffer_contents =
            unsafe { (instance_buffer.metal_buffer.contents() as *mut u8).add(*instance_offset) };

        let next_offset = *instance_offset + underline_bytes_len;
        if next_offset > instance_buffer.size {
            return false;
        }

        unsafe {
            ptr::copy_nonoverlapping(
                underlines.as_ptr() as *const u8,
                buffer_contents,
                underline_bytes_len,
            );
        }

        command_encoder.draw_primitives_instanced(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            underlines.len() as u64,
        );
        *instance_offset = next_offset;
        true
    }

    fn draw_monochrome_sprites(
        &self,
        texture_id: AtlasTextureId,
        sprites: &[MonochromeSprite],
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) -> bool {
        if sprites.is_empty() {
            return true;
        }
        align_offset(instance_offset);

        let sprite_bytes_len = mem::size_of_val(sprites);
        let buffer_contents =
            unsafe { (instance_buffer.metal_buffer.contents() as *mut u8).add(*instance_offset) };

        let next_offset = *instance_offset + sprite_bytes_len;
        if next_offset > instance_buffer.size {
            return false;
        }

        let texture = self.sprite_atlas.metal_texture(texture_id);
        let texture_size = size(
            DevicePixels(texture.width() as i32),
            DevicePixels(texture.height() as i32),
        );
        command_encoder.set_render_pipeline_state(&self.monochrome_sprites_pipeline_state);
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::AtlasTextureSize as u64,
            mem::size_of_val(&texture_size) as u64,
            &texture_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_fragment_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );
        command_encoder.set_fragment_texture(SpriteInputIndex::AtlasTexture as u64, Some(&texture));

        unsafe {
            ptr::copy_nonoverlapping(
                sprites.as_ptr() as *const u8,
                buffer_contents,
                sprite_bytes_len,
            );
        }

        command_encoder.draw_primitives_instanced(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            sprites.len() as u64,
        );
        *instance_offset = next_offset;
        true
    }

    fn draw_polychrome_sprites(
        &self,
        texture_id: AtlasTextureId,
        sprites: &[PolychromeSprite],
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) -> bool {
        if sprites.is_empty() {
            return true;
        }
        align_offset(instance_offset);

        let texture = self.sprite_atlas.metal_texture(texture_id);
        let texture_size = size(
            DevicePixels(texture.width() as i32),
            DevicePixels(texture.height() as i32),
        );
        command_encoder.set_render_pipeline_state(&self.polychrome_sprites_pipeline_state);
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::AtlasTextureSize as u64,
            mem::size_of_val(&texture_size) as u64,
            &texture_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_fragment_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );
        command_encoder.set_fragment_texture(SpriteInputIndex::AtlasTexture as u64, Some(&texture));

        let sprite_bytes_len = mem::size_of_val(sprites);
        let buffer_contents =
            unsafe { (instance_buffer.metal_buffer.contents() as *mut u8).add(*instance_offset) };

        let next_offset = *instance_offset + sprite_bytes_len;
        if next_offset > instance_buffer.size {
            return false;
        }

        unsafe {
            ptr::copy_nonoverlapping(
                sprites.as_ptr() as *const u8,
                buffer_contents,
                sprite_bytes_len,
            );
        }

        command_encoder.draw_primitives_instanced(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            sprites.len() as u64,
        );
        *instance_offset = next_offset;
        true
    }

    fn draw_surfaces(
        &mut self,
        surfaces: &[PaintSurface],
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) -> bool {
        command_encoder.set_render_pipeline_state(&self.surfaces_pipeline_state);
        command_encoder.set_vertex_buffer(
            SurfaceInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_bytes(
            SurfaceInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        for surface in surfaces {
            let texture_size = size(
                DevicePixels::from(surface.image_buffer.get_width() as i32),
                DevicePixels::from(surface.image_buffer.get_height() as i32),
            );

            assert_eq!(
                surface.image_buffer.get_pixel_format(),
                kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
            );

            let y_texture = self
                .core_video_texture_cache
                .create_texture_from_image(
                    surface.image_buffer.as_concrete_TypeRef(),
                    None,
                    MTLPixelFormat::R8Unorm,
                    surface.image_buffer.get_width_of_plane(0),
                    surface.image_buffer.get_height_of_plane(0),
                    0,
                )
                .unwrap();
            let cb_cr_texture = self
                .core_video_texture_cache
                .create_texture_from_image(
                    surface.image_buffer.as_concrete_TypeRef(),
                    None,
                    MTLPixelFormat::RG8Unorm,
                    surface.image_buffer.get_width_of_plane(1),
                    surface.image_buffer.get_height_of_plane(1),
                    1,
                )
                .unwrap();

            align_offset(instance_offset);
            let next_offset = *instance_offset + mem::size_of::<Surface>();
            if next_offset > instance_buffer.size {
                return false;
            }

            command_encoder.set_vertex_buffer(
                SurfaceInputIndex::Surfaces as u64,
                Some(&instance_buffer.metal_buffer),
                *instance_offset as u64,
            );
            command_encoder.set_vertex_bytes(
                SurfaceInputIndex::TextureSize as u64,
                mem::size_of_val(&texture_size) as u64,
                &texture_size as *const Size<DevicePixels> as *const _,
            );
            // let y_texture = y_texture.get_texture().unwrap().
            command_encoder.set_fragment_texture(SurfaceInputIndex::YTexture as u64, unsafe {
                let texture = CVMetalTextureGetTexture(y_texture.as_concrete_TypeRef());
                Some(metal::TextureRef::from_ptr(texture as *mut _))
            });
            command_encoder.set_fragment_texture(SurfaceInputIndex::CbCrTexture as u64, unsafe {
                let texture = CVMetalTextureGetTexture(cb_cr_texture.as_concrete_TypeRef());
                Some(metal::TextureRef::from_ptr(texture as *mut _))
            });

            unsafe {
                let buffer_contents = (instance_buffer.metal_buffer.contents() as *mut u8)
                    .add(*instance_offset)
                    as *mut SurfaceBounds;
                ptr::write(
                    buffer_contents,
                    SurfaceBounds {
                        bounds: surface.bounds,
                        content_mask: surface.content_mask.clone(),
                    },
                );
            }

            command_encoder.draw_primitives(metal::MTLPrimitiveType::Triangle, 0, 6);
            *instance_offset = next_offset;
        }
        true
    }
}

fn new_command_encoder<'a>(
    command_buffer: &'a metal::CommandBufferRef,
    texture: &'a metal::TextureRef,
    viewport_size: Size<DevicePixels>,
    configure_color_attachment: impl Fn(&RenderPassColorAttachmentDescriptorRef),
) -> &'a metal::RenderCommandEncoderRef {
    let render_pass_descriptor = metal::RenderPassDescriptor::new();
    let color_attachment = render_pass_descriptor
        .color_attachments()
        .object_at(0)
        .unwrap();
    color_attachment.set_texture(Some(texture));
    color_attachment.set_store_action(metal::MTLStoreAction::Store);
    configure_color_attachment(color_attachment);

    let command_encoder = command_buffer.new_render_command_encoder(render_pass_descriptor);
    command_encoder.set_viewport(metal::MTLViewport {
        originX: 0.0,
        originY: 0.0,
        width: i32::from(viewport_size.width) as f64,
        height: i32::from(viewport_size.height) as f64,
        znear: 0.0,
        zfar: 1.0,
    });
    command_encoder
}

fn build_pipeline_state(
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
    label: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
    pixel_format: metal::MTLPixelFormat,
    isolated: bool,
) -> metal::RenderPipelineState {
    let vertex_fn = library
        .get_function(vertex_fn_name, None)
        .expect("error locating vertex function");
    let fragment_fn = library
        .get_function(fragment_fn_name, None)
        .expect("error locating fragment function");

    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label(label);
    descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
    descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
    let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(pixel_format);
    color_attachment.set_blending_enabled(true);
    color_attachment.set_rgb_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_alpha_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_source_rgb_blend_factor(metal::MTLBlendFactor::SourceAlpha);
    color_attachment.set_source_alpha_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_destination_rgb_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);
    color_attachment.set_destination_alpha_blend_factor(if isolated {
        metal::MTLBlendFactor::OneMinusSourceAlpha
    } else {
        metal::MTLBlendFactor::One
    });

    device
        .new_render_pipeline_state(&descriptor)
        .expect("could not create render pipeline state")
}

fn build_path_sprite_pipeline_state(
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
    label: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
    pixel_format: metal::MTLPixelFormat,
    isolated: bool,
) -> metal::RenderPipelineState {
    let vertex_fn = library
        .get_function(vertex_fn_name, None)
        .expect("error locating vertex function");
    let fragment_fn = library
        .get_function(fragment_fn_name, None)
        .expect("error locating fragment function");

    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label(label);
    descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
    descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
    let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(pixel_format);
    color_attachment.set_blending_enabled(true);
    color_attachment.set_rgb_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_alpha_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_source_rgb_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_source_alpha_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_destination_rgb_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);
    color_attachment.set_destination_alpha_blend_factor(if isolated {
        metal::MTLBlendFactor::OneMinusSourceAlpha
    } else {
        metal::MTLBlendFactor::One
    });

    device
        .new_render_pipeline_state(&descriptor)
        .expect("could not create render pipeline state")
}

fn build_path_rasterization_pipeline_state(
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
    label: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
    pixel_format: metal::MTLPixelFormat,
    path_sample_count: u32,
) -> metal::RenderPipelineState {
    let vertex_fn = library
        .get_function(vertex_fn_name, None)
        .expect("error locating vertex function");
    let fragment_fn = library
        .get_function(fragment_fn_name, None)
        .expect("error locating fragment function");

    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label(label);
    descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
    descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
    if path_sample_count > 1 {
        descriptor.set_raster_sample_count(path_sample_count as _);
        descriptor.set_alpha_to_coverage_enabled(false);
    }
    let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(pixel_format);
    color_attachment.set_blending_enabled(true);
    color_attachment.set_rgb_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_alpha_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_source_rgb_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_source_alpha_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_destination_rgb_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);
    color_attachment.set_destination_alpha_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);

    device
        .new_render_pipeline_state(&descriptor)
        .expect("could not create render pipeline state")
}

// Align to multiples of 256 make Metal happy.
fn align_offset(offset: &mut usize) {
    *offset = (*offset).div_ceil(256) * 256;
}

#[repr(C)]
enum ShadowInputIndex {
    Vertices = 0,
    Shadows = 1,
    ViewportSize = 2,
}

#[repr(C)]
enum QuadInputIndex {
    Vertices = 0,
    Quads = 1,
    ViewportSize = 2,
}

#[repr(C)]
enum UnderlineInputIndex {
    Vertices = 0,
    Underlines = 1,
    ViewportSize = 2,
}

#[repr(C)]
enum SpriteInputIndex {
    Vertices = 0,
    Sprites = 1,
    ViewportSize = 2,
    AtlasTextureSize = 3,
    AtlasTexture = 4,
}

#[repr(C)]
enum SurfaceInputIndex {
    Vertices = 0,
    Surfaces = 1,
    ViewportSize = 2,
    TextureSize = 3,
    YTexture = 4,
    CbCrTexture = 5,
}

#[repr(C)]
enum PathRasterizationInputIndex {
    Vertices = 0,
    ViewportSize = 1,
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct PathSprite {
    pub bounds: Bounds<ScaledPixels>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct SurfaceBounds {
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
}

pub(crate) struct IsolatedTextures {
    pub(crate) pixel_size: Size<u32>,
    color: metal::Texture,
    paths: metal::Texture,
    msaa: metal::Texture,
    _heap: metal::Heap,
    _reservation: crate::isolated::TargetReservation,
}

fn retain_isolated_resources(scene: &Scene, command_buffer: &metal::CommandBufferRef) {
    // Retain reservations even for an encoding failure: the uncommitted
    // command buffer may retain textures until its autorelease pool drains.
    let mut targets = Vec::new();
    collect_isolated_targets(scene, &mut targets);
    let completed = ConcreteBlock::new(move |buffer: &metal::CommandBufferRef| {
        if buffer.status() == metal::MTLCommandBufferStatus::Error {
            for target in &targets {
                target.fail();
            }
        }
    })
    .copy();
    command_buffer.add_completed_handler(&completed);
}
fn collect_isolated_targets(scene: &Scene, out: &mut Vec<crate::PreparedIsolatedLayer>) {
    for layer in &scene.isolated {
        out.push(layer.target.clone());
        collect_isolated_targets(&layer.scene, out);
    }
}
impl MetalRenderer {
    pub(crate) fn prepare_isolated_layer(
        &self,
        window: crate::WindowId,
        scale: f32,
        logical: Size<crate::Pixels>,
    ) -> Result<crate::PreparedIsolatedLayer, crate::IsolatedLayerError> {
        use crate::IsolatedLayerError as E;
        let dimension = |v: crate::Pixels| {
            let v = (f64::from(f32::from(v)) * f64::from(scale)).ceil() + 1.;
            if !v.is_finite() || !(1.0..=16384.0).contains(&v) {
                Err(E::Limit)
            } else {
                Ok(v as u32)
            }
        };
        let pixel_size = size(dimension(logical.width)?, dimension(logical.height)?);
        let descriptor = metal::TextureDescriptor::new();
        descriptor.set_width(pixel_size.width as u64);
        descriptor.set_height(pixel_size.height as u64);
        descriptor.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        descriptor.set_storage_mode(metal::MTLStorageMode::Private);
        descriptor
            .set_usage(metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead);
        let msaa_descriptor = metal::TextureDescriptor::new();
        msaa_descriptor.set_width(pixel_size.width as u64);
        msaa_descriptor.set_height(pixel_size.height as u64);
        msaa_descriptor.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        msaa_descriptor.set_storage_mode(metal::MTLStorageMode::Private);
        msaa_descriptor.set_usage(metal::MTLTextureUsage::RenderTarget);
        msaa_descriptor.set_texture_type(metal::MTLTextureType::D2Multisample);
        msaa_descriptor.set_sample_count(self.path_sample_count as u64);
        let mut bytes = 0usize;
        for desc in [&*descriptor, &*descriptor, &*msaa_descriptor] {
            let required = self.device.heap_texture_size_and_align(desc);
            let alignment = required.align as usize;
            if alignment == 0 {
                return Err(E::Device);
            }
            bytes = bytes.checked_add(alignment - 1).ok_or(E::Limit)? / alignment * alignment;
            bytes = bytes.checked_add(required.size as usize).ok_or(E::Limit)?;
        }
        // Metal rounds heaps to the host VM page size. Reserve that padding
        // before allocation as well as each attachment's queried alignment.
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if page <= 0 {
            return Err(E::Device);
        }
        let page = page as usize;
        bytes = bytes.checked_add(page - 1).ok_or(E::Limit)? / page * page;
        let reservation = self.isolated_pool.reserve(bytes)?;
        let heap_descriptor = metal::HeapDescriptor::new();
        heap_descriptor.set_size(bytes as u64);
        heap_descriptor.set_storage_mode(metal::MTLStorageMode::Private);
        // Heap resources default to untracked hazards. Child writes must be
        // visible to the following composite sampler on the same command buffer.
        heap_descriptor.set_hazard_tracking_mode(metal::MTLHazardTrackingMode::Tracked);
        // metal-rs' Device::new_heap assumes non-null. Preserve allocation
        // failure as a Result instead of constructing an invalid foreign ref.
        let raw: *mut metal::MTLHeap =
            unsafe { msg_send![&*self.device, newHeapWithDescriptor: &*heap_descriptor] };
        if raw.is_null() {
            return Err(E::Device);
        }
        let heap = unsafe { metal::Heap::from_ptr(raw) };
        if heap.size() as usize > bytes {
            return Err(E::Limit);
        }
        let color = heap.new_texture(&descriptor).ok_or(E::Device)?;
        let paths = heap.new_texture(&descriptor).ok_or(E::Device)?;
        let msaa = heap.new_texture(&msaa_descriptor).ok_or(E::Device)?;
        Ok(crate::PreparedIsolatedLayer {
            window,
            scale,
            size: logical,
            failure: Arc::new(crate::isolated::FailureSignal::default()),
            recording: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            metal: Arc::new(IsolatedTextures {
                pixel_size,
                color,
                paths,
                msaa,
                _heap: heap,
                _reservation: reservation,
            }),
        })
    }
    fn draw_isolated(
        &self,
        layer: &crate::IsolatedPrimitive,
        viewport: Size<DevicePixels>,
        encoder: &metal::RenderCommandEncoderRef,
    ) -> bool {
        let b = layer.bounds;
        let c = layer.content_mask.bounds;
        let uniforms = [
            b.origin.x.0,
            b.origin.y.0,
            b.size.width.0,
            b.size.height.0,
            c.origin.x.0,
            c.origin.y.0,
            c.size.width.0,
            c.size.height.0,
            layer.opacity,
            viewport.width.0 as f32,
            viewport.height.0 as f32,
            0.,
        ];
        encoder.set_render_pipeline_state(&self.composite_pipeline);
        encoder.set_vertex_buffer(0, Some(&self.unit_vertices), 0);
        encoder.set_vertex_bytes(
            1,
            mem::size_of_val(&uniforms) as u64,
            uniforms.as_ptr() as *const _,
        );
        encoder.set_fragment_bytes(
            1,
            mem::size_of_val(&uniforms) as u64,
            uniforms.as_ptr() as *const _,
        );
        encoder.set_fragment_texture(0, Some(&layer.target.metal.color));
        encoder.draw_primitives(metal::MTLPrimitiveType::Triangle, 0, 6);
        true
    }
}

#[cfg(test)]
mod isolated_tests {
    use super::*;
    use crate::{
        Bounds, ContentMask, IsolatedPrimitive, PreparedIsolatedLayer, ScaledPixels as S, WindowId,
        rgba,
    };
    fn bounds(x: f32, y: f32, w: f32, h: f32) -> Bounds<S> {
        Bounds::new(point(S(x), S(y)), size(S(w), S(h)))
    }
    fn quad(scene: &mut Scene, b: Bounds<S>, color: u32) {
        scene.insert_primitive(Quad {
            bounds: b,
            content_mask: ContentMask {
                bounds: bounds(0., 0., 64., 64.),
            },
            background: rgba(color).into(),
            ..Default::default()
        });
    }
    fn layer(renderer: &MetalRenderer, scene: Scene, opacity: f32) -> IsolatedPrimitive {
        let target = renderer
            .prepare_isolated_layer(
                WindowId::default(),
                1.,
                size(crate::px(16.), crate::px(16.)),
            )
            .unwrap();
        IsolatedPrimitive {
            order: 0,
            bounds: bounds(0., 0., 17., 17.),
            content_mask: ContentMask {
                bounds: bounds(0., 0., 16., 16.),
            },
            opacity,
            target,
            scene: Arc::new(scene),
        }
    }
    fn pixels(renderer: &mut MetalRenderer, mut scene: Scene) -> Vec<u8> {
        scene.finish();
        let descriptor = metal::TextureDescriptor::new();
        descriptor.set_width(16);
        descriptor.set_height(16);
        descriptor.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        descriptor.set_storage_mode(metal::MTLStorageMode::Shared);
        descriptor.set_usage(metal::MTLTextureUsage::RenderTarget);
        let texture = renderer.device.new_texture(&descriptor);
        let queue = renderer.command_queue.clone();
        let commands = queue.new_command_buffer();
        let mut instance = renderer
            .instance_buffer_pool
            .lock()
            .acquire(&renderer.device);
        let mut offset = 0;
        renderer
            .encode_scene(
                &scene,
                &mut instance,
                &mut offset,
                commands,
                &texture,
                size(DevicePixels(16), DevicePixels(16)),
                0.,
            )
            .unwrap();
        instance.metal_buffer.did_modify_range(NSRange {
            location: 0,
            length: offset as u64,
        });
        commands.commit();
        commands.wait_until_completed();
        assert_ne!(commands.status(), metal::MTLCommandBufferStatus::Error);
        let mut pixels = vec![0u8; 16 * 16 * 4];
        texture.get_bytes(
            pixels.as_mut_ptr() as *mut _,
            64,
            metal::MTLRegion::new_2d(0, 0, 16, 16),
            0,
        );
        pixels
    }
    fn pixel(image: &[u8], x: usize, y: usize) -> [u8; 4] {
        image[(y * 16 + x) * 4..(y * 16 + x) * 4 + 4]
            .try_into()
            .unwrap()
    }
    #[test]
    fn native_isolation_composites_overlaps_nested_alpha_and_replay() {
        let mut renderer = MetalRenderer::new(Arc::new(Mutex::new(Default::default())));
        for opacity in [0., 0.5, 1.] {
            let mut child = Scene::default();
            quad(&mut child, bounds(0., 0., 12., 12.), 0xff0000ff);
            quad(&mut child, bounds(4., 4., 12., 12.), 0x0000ffff);
            child.finish();
            let group = layer(&renderer, child, opacity);
            let mut scene = Scene::default();
            quad(&mut scene, bounds(0., 0., 16., 16.), 0xffffffff);
            scene.insert_primitive(crate::Primitive::Isolated(group));
            let mut replay = Scene::default();
            replay.replay(0..scene.len(), &scene);
            drop(scene);
            let image = pixels(&mut renderer, replay);
            let faded = ((1. - opacity) * 255.).round() as u8;
            assert_eq!(pixel(&image, 6, 6), [255, faded, faded, 255]);
            assert_eq!(pixel(&image, 2, 2), [faded, faded, 255, 255]);
        }
        let mut leaf = Scene::default();
        quad(&mut leaf, bounds(0., 0., 16., 16.), 0xff000080);
        quad(&mut leaf, bounds(0., 0., 16., 16.), 0x0000ff80);
        leaf.finish();
        let inner = layer(&renderer, leaf, 0.5);
        let mut child = Scene::default();
        child.insert_primitive(crate::Primitive::Isolated(inner));
        child.finish();
        let outer = layer(&renderer, child, 0.5);
        let mut scene = Scene::default();
        scene.insert_primitive(crate::Primitive::Isolated(outer));
        let image = pixels(&mut renderer, scene);
        let actual = pixel(&image, 8, 8);
        // BGRA premultiplied: blue .5*.25, red .5*.5*.25,
        // alpha (.5 + .5*(1-.5))*.25. One unit allows UNORM rounding.
        for (actual, expected) in actual.into_iter().zip([32i16, 0, 16, 48]) {
            assert!(
                (i16::from(actual) - expected).abs() <= 1,
                "{actual} {expected}"
            );
        }
    }
    #[test]
    fn repeated_target_placements_are_consumed_before_reuse() {
        let mut renderer = MetalRenderer::new(Arc::new(Mutex::new(Default::default())));
        let mut red = Scene::default();
        quad(&mut red, bounds(0., 0., 16., 16.), 0xff0000ff);
        red.finish();
        let first = layer(&renderer, red, 1.);
        let mut blue = Scene::default();
        quad(&mut blue, bounds(0., 0., 16., 16.), 0x0000ffff);
        blue.finish();
        let mut second = first.clone();
        second.scene = Arc::new(blue);
        second.content_mask.bounds = bounds(8., 0., 8., 16.);
        let mut scene = Scene::default();
        scene.insert_primitive(crate::Primitive::Isolated(first));
        scene.insert_primitive(crate::Primitive::Isolated(second));
        let image = pixels(&mut renderer, scene);
        assert_eq!(pixel(&image, 2, 8), [0, 0, 255, 255]);
        assert_eq!(pixel(&image, 12, 8), [255, 0, 0, 255]);
    }
    #[test]
    fn poisoned_replayed_layer_does_not_suppress_window_or_siblings() {
        let mut renderer = MetalRenderer::new(Arc::new(Mutex::new(Default::default())));
        let mut red = Scene::default();
        quad(&mut red, bounds(0., 0., 16., 16.), 0xff0000ff);
        red.finish();
        let failed = layer(&renderer, red, 1.);
        let owner = failed.target.clone();
        let mut blue = Scene::default();
        quad(&mut blue, bounds(8., 0., 8., 16.), 0x0000ffff);
        blue.finish();
        let healthy = layer(&renderer, blue, 1.);
        let healthy_owner = healthy.target.clone();
        let mut cached = Scene::default();
        quad(&mut cached, bounds(0., 0., 16., 16.), 0xffffffff);
        cached.insert_primitive(crate::Primitive::Isolated(failed));
        cached.insert_primitive(crate::Primitive::Isolated(healthy));
        let mut replay = Scene::default();
        replay.replay(0..cached.len(), &cached);
        owner.fail(); // completion after view recording, before cached replay
        let image = pixels(&mut renderer, replay);
        assert_eq!(pixel(&image, 2, 8), [255, 255, 255, 255]);
        assert_eq!(pixel(&image, 12, 8), [255, 0, 0, 255]);
        assert!(!healthy_owner.is_failed());
    }
    #[test]
    fn command_buffer_retains_target_after_scene_drop_until_retirement() {
        let renderer = MetalRenderer::new(Arc::new(Mutex::new(Default::default())));
        let weak = objc::rc::autoreleasepool(|| {
            let group = layer(&renderer, Scene::default(), 0.5);
            let weak = Arc::downgrade(&group.target.metal);
            let mut scene = Scene::default();
            scene.insert_primitive(crate::Primitive::Isolated(group));
            let commands = renderer.command_queue.new_command_buffer();
            retain_isolated_resources(&scene, commands);
            drop(scene); // simulates prepared/replay owners closing before GPU completion
            assert!(weak.upgrade().is_some());
            commands.commit();
            commands.wait_until_completed();
            weak
        });
        assert!(weak.upgrade().is_none());
    }
    #[test]
    fn encoding_failure_restores_renderer_and_next_generation_draws() {
        let mut renderer = MetalRenderer::new(Arc::new(Mutex::new(Default::default())));
        let mut child = Scene::default();
        quad(&mut child, bounds(0., 0., 12., 12.), 0xff0000ff);
        child.finish();
        let group = layer(&renderer, child, 0.5);
        let owner = group.target.clone();
        let texture = group.target.metal.color.clone();
        let mut scene = Scene::default();
        scene.insert_primitive(crate::Primitive::Isolated(group));
        scene.finish();
        let commands = renderer.command_queue.new_command_buffer().to_owned();
        let mut exhausted = InstanceBuffer {
            metal_buffer: renderer
                .device
                .new_buffer(256, MTLResourceOptions::StorageModeManaged),
            size: 0,
        };
        let pipeline = renderer.quads_pipeline_state.as_ptr();
        assert!(
            renderer
                .encode_scene(
                    &scene,
                    &mut exhausted,
                    &mut 0,
                    &commands,
                    &texture,
                    size(DevicePixels(16), DevicePixels(16)),
                    0.
                )
                .is_err()
        );
        assert!(renderer.path_intermediate_texture.is_none());
        assert!(renderer.path_intermediate_msaa_texture.is_none());
        assert_eq!(pipeline, renderer.quads_pipeline_state.as_ptr());
        // Buffer exhaustion is transient: the same retained generation retries.
        assert!(!owner.is_failed());
        let image = pixels(&mut renderer, scene);
        assert_eq!(pixel(&image, 2, 2), [0, 0, 128, 128]);
        assert!(!owner.is_failed());
        let mut next = Scene::default();
        quad(&mut next, bounds(0., 0., 16., 16.), 0x0000ffff);
        assert_eq!(pixel(&pixels(&mut renderer, next), 8, 8), [255, 0, 0, 255]);
    }
    #[test]
    fn all_attachments_share_window_quota_and_failure_releases_it() {
        let renderer = MetalRenderer::new(Arc::new(Mutex::new(Default::default())));
        assert!(matches!(
            renderer.prepare_isolated_layer(
                WindowId::default(),
                2.,
                size(crate::px(1600.), crate::px(1000.))
            ),
            Err(crate::IsolatedLayerError::Limit)
        ));
        let mut targets = Vec::<PreparedIsolatedLayer>::new();
        loop {
            match renderer.prepare_isolated_layer(
                WindowId::default(),
                2.,
                size(crate::px(300.), crate::px(200.)),
            ) {
                Ok(t) => targets.push(t),
                Err(crate::IsolatedLayerError::Limit) => break,
                Err(error) => panic!("{error}"),
            }
        }
        assert!(!targets.is_empty());
        assert!(targets.len() < 10);
        let replay = targets.last().unwrap().clone();
        targets.clear();
        assert!(
            renderer
                .prepare_isolated_layer(
                    WindowId::default(),
                    2.,
                    size(crate::px(300.), crate::px(200.))
                )
                .is_ok()
        );
        replay.fail();
        assert!(replay.is_failed());
        drop(replay);
        assert!(
            renderer
                .prepare_isolated_layer(
                    WindowId::default(),
                    2.,
                    size(crate::px(300.), crate::px(200.))
                )
                .is_ok()
        );
    }
}
