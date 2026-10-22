//! Bounded native isolated layers. Preparation owns resources; frame/replay and
//! GPU completion retain them independently of the preparing element.
use crate::{Bounds, Pixels, Size, Window, WindowId};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

/// A preparation/recording failure. Never substitute per-primitive opacity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IsolatedLayerError {
    /// The backend does not implement isolated native layers.
    Unsupported,
    /// Invalid dimensions, opacity, placement or window/scale identity.
    Invalid,
    /// The window's aggregate live/in-flight target quota was exceeded.
    Limit,
    /// Native allocation, encoding or GPU completion failed.
    Device,
}
impl std::fmt::Display for IsolatedLayerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "isolated layer: {self:?}")
    }
}
impl std::error::Error for IsolatedLayerError {}

/// An opaque target prepared outside paint. Clones share both reservation and
/// failure state; a failed generation can never be painted again.
#[derive(Clone)]
pub struct PreparedIsolatedLayer {
    pub(crate) window: WindowId,
    pub(crate) scale: f32,
    pub(crate) size: Size<Pixels>,
    pub(crate) failed: Arc<AtomicBool>,
    pub(crate) recording: Arc<AtomicBool>,
    #[cfg(all(target_os = "macos", not(feature = "macos-blade")))]
    pub(crate) metal: Arc<crate::metal_renderer::IsolatedTextures>,
}
impl std::fmt::Debug for PreparedIsolatedLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedIsolatedLayer")
            .field("window", &self.window)
            .field("size", &self.size)
            .field("failed", &self.is_failed())
            .finish()
    }
}
impl PreparedIsolatedLayer {
    /// Check before publishing/painting a retained generation, including errors
    /// reported asynchronously by the GPU completion handler.
    pub fn is_failed(&self) -> bool {
        self.failed.load(Ordering::Acquire)
    }
    pub(crate) fn fail(&self) {
        self.failed.store(true, Ordering::Release);
    }
}

impl Window {
    /// Reserve all target and path-AA attachments before any new display list is
    /// published. The 32 MiB quota covers all diagrams in this window.
    pub fn prepare_isolated_layer(
        &self,
        size: Size<Pixels>,
    ) -> Result<PreparedIsolatedLayer, IsolatedLayerError> {
        if !f32::from(size.width).is_finite()
            || !f32::from(size.height).is_finite()
            || size.width <= crate::px(0.)
            || size.height <= crate::px(0.)
        {
            return Err(IsolatedLayerError::Invalid);
        }
        self.platform_window.prepare_isolated_layer(
            self.window_handle().window_id(),
            self.scale_factor(),
            size,
        )
    }

    /// Record children into a separate native Scene, then composite once in the
    /// parent order. The closure should only paint. Recording restores Scene,
    /// clip and opacity on success, error and unwinding panic.
    pub fn paint_isolated_layer<E>(
        &mut self,
        target: &PreparedIsolatedLayer,
        bounds: Bounds<Pixels>,
        opacity: f32,
        paint: impl FnOnce(&mut Window) -> Result<(), E>,
    ) -> Result<Result<(), E>, IsolatedLayerError> {
        self.invalidator.debug_assert_paint();
        if target.is_failed() {
            return Err(IsolatedLayerError::Device);
        }
        if target.window != self.window_handle().window_id()
            || target.scale != self.scale_factor()
            || bounds.size != target.size
            || !opacity.is_finite()
            || !(0.0..=1.0).contains(&opacity)
            || !f32::from(bounds.origin.x).is_finite()
            || !f32::from(bounds.origin.y).is_finite()
        {
            return Err(IsolatedLayerError::Invalid);
        }
        #[cfg(all(target_os = "macos", not(feature = "macos-blade")))]
        {
            // Reusing the target in separate placements is supported, but a
            // target cannot be sampled while it is also an ancestor attachment.
            if target.recording.swap(true, Ordering::AcqRel) {
                return Err(IsolatedLayerError::Invalid);
            }
            let inherited = self.content_mask();
            let mask = crate::ContentMask {
                bounds: inherited.bounds.intersect(&bounds),
            };
            let old_scene = std::mem::take(&mut self.next_frame.scene);
            let masks = self.content_mask_stack.clone();
            let old_opacity = self.element_opacity;
            self.element_opacity = 1.;
            self.content_mask_stack.push(mask.clone());
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| paint(self)));
            let mut child = std::mem::replace(&mut self.next_frame.scene, old_scene);
            self.content_mask_stack = masks;
            self.element_opacity = old_opacity;
            target.recording.store(false, Ordering::Release);
            match result {
                Err(payload) => std::panic::resume_unwind(payload),
                Ok(Err(error)) => Ok(Err(error)),
                Ok(Ok(())) => {
                    let physical = bounds.scale(target.scale);
                    let origin = physical.origin.map(|v| crate::ScaledPixels(v.0.floor()));
                    child.translate_for_isolation(origin);
                    child.finish();
                    let composite = crate::IsolatedPrimitive {
                        order: 0,
                        bounds: Bounds {
                            origin,
                            size: target
                                .metal
                                .pixel_size
                                .map(|v| crate::ScaledPixels(v as f32)),
                        },
                        content_mask: mask.scale(target.scale),
                        opacity: opacity * old_opacity,
                        target: target.clone(),
                        scene: Arc::new(child),
                    };
                    // Always request a distinct layer, even inside a parent's
                    // non-overlapping paint_layer declaration.
                    self.next_frame.scene.push_layer(composite.bounds);
                    self.next_frame
                        .scene
                        .insert_primitive(crate::Primitive::Isolated(composite));
                    self.next_frame.scene.pop_layer();
                    Ok(Ok(()))
                }
            }
        }
        #[cfg(not(all(target_os = "macos", not(feature = "macos-blade"))))]
        {
            let _ = paint;
            Err(IsolatedLayerError::Unsupported)
        }
    }
}

pub(crate) const TARGET_POOL_BYTES: usize = 32 * 1024 * 1024;
#[derive(Default, Clone)]
pub(crate) struct TargetPool(Arc<parking_lot::Mutex<usize>>);
pub(crate) struct TargetReservation {
    pool: TargetPool,
    bytes: usize,
}
impl TargetPool {
    pub(crate) fn reserve(&self, bytes: usize) -> Result<TargetReservation, IsolatedLayerError> {
        let mut used = self.0.lock();
        if bytes > TARGET_POOL_BYTES.saturating_sub(*used) {
            return Err(IsolatedLayerError::Limit);
        }
        *used += bytes;
        Ok(TargetReservation {
            pool: self.clone(),
            bytes,
        })
    }
}
impl Drop for TargetReservation {
    fn drop(&mut self) {
        *self.pool.0.lock() -= self.bytes;
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn quota_counts_retained_and_inflight_owners_until_last_release() {
        let pool = TargetPool::default();
        let prepared = Arc::new(pool.reserve(TARGET_POOL_BYTES).unwrap());
        let replay = prepared.clone();
        let inflight = replay.clone();
        drop(prepared);
        drop(replay);
        assert!(matches!(pool.reserve(1), Err(IsolatedLayerError::Limit)));
        drop(inflight);
        assert!(pool.reserve(TARGET_POOL_BYTES).is_ok());
    }
    #[test]
    fn nested_and_failed_reservations_are_atomic() {
        let pool = TargetPool::default();
        let parent = pool.reserve(TARGET_POOL_BYTES / 2).unwrap();
        assert!(matches!(
            pool.reserve(TARGET_POOL_BYTES),
            Err(IsolatedLayerError::Limit)
        ));
        let child = pool.reserve(TARGET_POOL_BYTES / 2).unwrap();
        assert!(pool.reserve(1).is_err());
        drop(child);
        drop(parent);
        assert!(pool.reserve(TARGET_POOL_BYTES).is_ok());
    }
}
