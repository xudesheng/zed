//! Window-local prepared native text. This prototype deliberately retains the
//! loose outline font box, but bitmap/unknown technologies use actual phase
//! allocations. No font file bytes, shaping, or raster images are retained here.
use crate::{
    Bounds, DevicePixels, Hsla, IsZero, LineLayout, Pixels, Point, RenderGlyphParams,
    SUBPIXEL_VARIANTS_X, SUBPIXEL_VARIANTS_Y, Size, Window, WindowId, point, px, size,
};
use anyhow::{Result, ensure};
use std::sync::Arc;

#[derive(Clone, Debug)]
struct Phase {
    params: RenderGlyphParams,
    raster: Bounds<DevicePixels>,
    bitmap_size: Size<DevicePixels>,
}
#[derive(Clone, Debug)]
struct PreparedGlyph {
    offset: Point<Pixels>,
    phases: Arc<Vec<Phase>>,
}

/// A baseline-relative, placement-independent envelope plus the exact glyphs
/// and phase metrics used by paint. Displayed logical pixels; retain base-size
/// shaping and apply view zoom only to offsets and the raster transform.
#[derive(Clone, Debug)]
pub struct PreparedGlyphLine {
    window: WindowId,
    device_scale: f32,
    view_scale: f32,
    envelope: Option<Bounds<Pixels>>,
    glyphs: Vec<PreparedGlyph>,
    query_envelope_glyphs: usize,
}
impl PreparedGlyphLine {
    /// Prepare outside paint. Errors are fatal; never substitute another font
    /// or switch to estimates when the native allocation query fails.
    pub fn new(layout: &LineLayout, view_scale: f32, window: &Window) -> Result<Self> {
        let ds = window.scale_factor();
        ensure!(
            ds.is_finite() && ds > 0.0 && view_scale.is_finite() && view_scale > 0.0,
            "invalid prepared-line scale"
        );
        let mut result = Self {
            window: window.window_handle().window_id(),
            device_scale: ds,
            view_scale,
            envelope: None,
            glyphs: Vec::new(),
            query_envelope_glyphs: 0,
        };
        let system = window.text_system();
        let mut cache = std::collections::HashMap::new();
        for run in &layout.runs {
            let outline = system.has_outline_only_bounds(run.font_id)?;
            for glyph in &run.glyphs {
                let offset = point(
                    glyph.position.x * view_scale,
                    -glyph.position.y * view_scale,
                );
                ensure!(
                    offset.x.0.is_finite() && offset.y.0.is_finite(),
                    "invalid shaped offset"
                );
                let key = (run.font_id, glyph.id, glyph.is_emoji);
                let phases = if let Some(phases) = cache.get(&key) {
                    Arc::clone(phases)
                } else {
                    let mut phases = Vec::new();
                    for y in 0..if glyph.is_emoji {
                        1
                    } else {
                        SUBPIXEL_VARIANTS_Y
                    } {
                        for x in 0..if glyph.is_emoji {
                            1
                        } else {
                            SUBPIXEL_VARIANTS_X
                        } {
                            let mut params = window.affine_glyph_params(
                                Point::default(),
                                run.font_id,
                                glyph.id,
                                layout.font_size,
                                view_scale,
                                glyph.is_emoji,
                            )?;
                            params.subpixel_variant = point(x as u8, y as u8);
                            let raster = system.raster_bounds(&params)?;
                            let bitmap_size = if raster.is_zero() {
                                Size::default()
                            } else {
                                system.glyph_bitmap_size(&params, raster)?
                            };
                            phases.push(Phase {
                                params,
                                raster,
                                bitmap_size,
                            });
                        }
                    }
                    let phases = Arc::new(phases);
                    cache.insert(key, Arc::clone(&phases));
                    phases
                };
                // For any baseline B and glyph offset Q in physical pixels:
                // Q - 1 < floor(B + Q) - B <= Q. Union every legal phase's
                // actual allocation, including nonzero-phase size extension.
                // The closed interval below safely contains both extrema.
                for phase in phases.iter().filter(|p| !p.raster.is_zero()) {
                    let b = phase_envelope(offset, phase.raster, phase.bitmap_size, ds);
                    result.union(b);
                }
                if outline {
                    // Keep the documented loose outline policy. The queried
                    // union above also guards against incorrect font metadata.
                    let b = system.bounding_box(run.font_id, layout.font_size);
                    let rs = ds * view_scale;
                    let left = (b.origin.x.0 * rs).floor() - 1.0;
                    let top = (-(b.origin.y.0 + b.size.height.0) * rs).floor() - 1.0;
                    let right = ((b.origin.x.0 + b.size.width.0) * rs).ceil() + 1.0;
                    let bottom = (-b.origin.y.0 * rs).ceil() + 1.0;
                    ensure!(
                        [left, top, right, bottom].iter().all(|n| n.is_finite())
                            && right >= left
                            && bottom >= top,
                        "invalid outline font box"
                    );
                    result.union(Bounds::new(
                        offset + point(px(left / ds), px(top / ds)),
                        size(px((right - left) / ds), px((bottom - top) / ds)),
                    ));
                } else {
                    result.query_envelope_glyphs += 1;
                }
                result.glyphs.push(PreparedGlyph { offset, phases });
            }
        }
        Ok(result)
    }
    fn union(&mut self, b: Bounds<Pixels>) {
        self.envelope = Some(self.envelope.map_or(b, |old| old.union(&b)));
    }
    /// Baseline-relative placement envelope in displayed logical pixels.
    pub fn envelope(&self) -> Option<Bounds<Pixels>> {
        self.envelope
    }
    /// Number of glyphs whose font metadata cannot bound native allocations.
    pub fn query_envelope_glyph_count(&self) -> usize {
        self.query_envelope_glyphs
    }
    fn validate(&self, baseline: Point<Pixels>, window: &Window) -> Result<()> {
        ensure!(
            self.window == window.window_handle().window_id()
                && self.device_scale == window.scale_factor(),
            "stale prepared line window/device"
        );
        ensure!(
            baseline.x.0.is_finite() && baseline.y.0.is_finite(),
            "invalid baseline"
        );
        Ok(())
    }
    fn phase<'a>(
        &self,
        glyph: &'a PreparedGlyph,
        baseline: Point<Pixels>,
        window: &Window,
    ) -> Result<(Point<Pixels>, &'a Phase)> {
        let origin = baseline + glyph.offset;
        let p = &glyph.phases[0].params;
        // Shared phase selection preserves the ordinary glyph painter's
        // handling of negative origins, including its saturating phase cast.
        let params = window.affine_glyph_params(
            origin,
            p.font_id,
            p.glyph_id,
            p.font_size,
            self.view_scale,
            p.is_emoji,
        )?;
        let index = if p.is_emoji {
            0
        } else {
            params.subpixel_variant.y as usize * SUBPIXEL_VARIANTS_X as usize
                + params.subpixel_variant.x as usize
        };
        Ok((origin, &glyph.phases[index]))
    }
    /// Exact positioned allocation union, from retained phase metrics only.
    pub fn positioned_bounds(
        &self,
        baseline: Point<Pixels>,
        window: &Window,
    ) -> Result<Option<Bounds<Pixels>>> {
        self.validate(baseline, window)?;
        let mut result: Option<Bounds<Pixels>> = None;
        for glyph in &self.glyphs {
            let (origin, phase) = self.phase(glyph, baseline, window)?;
            if phase.raster.is_zero() {
                continue;
            }
            let ds = self.device_scale;
            let physical = origin.scale(ds);
            let b = Bounds::new(
                point(
                    px((physical.x.0.floor() + phase.raster.origin.x.0 as f32) / ds),
                    px((physical.y.0.floor() + phase.raster.origin.y.0 as f32) / ds),
                ),
                size(
                    px(phase.bitmap_size.width.0 as f32 / ds),
                    px(phase.bitmap_size.height.0 as f32 / ds),
                ),
            );
            result = Some(result.map_or(b, |old| old.union(&b)));
        }
        Ok(result)
    }
    /// Paint translates retained glyphs and selects cached phases. Caller owns
    /// allocation/decoration clips; no nominal ShapedLine culling is used.
    pub fn paint(&self, baseline: Point<Pixels>, color: Hsla, window: &mut Window) -> Result<()> {
        self.validate(baseline, window)?;
        for glyph in &self.glyphs {
            let (origin, phase) = self.phase(glyph, baseline, window)?;
            window.paint_prepared_glyph(
                origin,
                &phase.params,
                phase.raster,
                Some(phase.bitmap_size),
                color,
            )?;
        }
        Ok(())
    }
}

fn phase_envelope(
    offset: Point<Pixels>,
    raster: Bounds<DevicePixels>,
    bitmap: Size<DevicePixels>,
    ds: f32,
) -> Bounds<Pixels> {
    Bounds::new(
        offset
            + point(
                px((raster.origin.x.0 as f32 - 1.0) / ds),
                px((raster.origin.y.0 as f32 - 1.0) / ds),
            ),
        size(
            px((bitmap.width.0 as f32 + 1.0) / ds),
            px((bitmap.height.0 as f32 + 1.0) / ds),
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mermaid_affine_phase_envelope_contains_translated_allocations() {
        // Negative and positive shaped offsets, physical baselines around each
        // 1/4 phase boundary, two device scales and four view zooms.
        for ds in [1.0, 2.0] {
            for zoom in [0.5, 1.0, 1.5, 2.0] {
                for q in [-7.31, -0.01, 0.0, 0.17, 15.91] {
                    let offset = point(px(q * zoom), px(-q * zoom));
                    let raster = Bounds::new(
                        point(DevicePixels(-3), DevicePixels(-19)),
                        size(DevicePixels(21), DevicePixels(23)),
                    );
                    for extra in [0, 1] {
                        let bitmap = size(DevicePixels(21 + extra), DevicePixels(23 + extra));
                        let envelope = phase_envelope(offset, raster, bitmap, ds);
                        for b in [
                            -23.001_f32,
                            -0.999,
                            -0.751,
                            -0.5,
                            -0.001,
                            0.0,
                            0.249,
                            0.25,
                            0.251,
                            0.499,
                            0.5,
                            0.749,
                            0.75,
                            0.999,
                            23.001,
                        ] {
                            for (offset, min, len, r, n) in [
                                (
                                    offset.x.0,
                                    envelope.origin.x.0,
                                    envelope.size.width.0,
                                    -3.0,
                                    bitmap.width.0,
                                ),
                                (
                                    offset.y.0,
                                    envelope.origin.y.0,
                                    envelope.size.height.0,
                                    -19.0,
                                    bitmap.height.0,
                                ),
                            ] {
                                let positioned = ((b + offset * ds).floor() + r - b) / ds;
                                assert!(
                                    positioned >= min - 1e-5
                                        && positioned + n as f32 / ds <= min + len + 1e-5
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
