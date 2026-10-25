//! Bounded native isolated layers. Preparation owns resources; frame/replay and
//! GPU completion retain them independently of the preparing element.
use crate::{Bounds, Pixels, Size, Window, WindowId};
use futures::{
    FutureExt,
    channel::oneshot,
    future::{BoxFuture, Shared},
};
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
    pub(crate) pixel_size: Size<u32>,
    pub(crate) failure: Arc<FailureSignal>,
    pub(crate) recording: Arc<AtomicBool>,
    #[cfg(all(target_os = "macos", not(feature = "macos-blade")))]
    pub(crate) metal: Arc<crate::metal_renderer::IsolatedTextures>,
    #[cfg(target_os = "windows")]
    pub(crate) directx: Arc<crate::platform::windows::IsolatedTextures>,
    #[cfg(any(target_os = "linux", target_os = "freebsd"))]
    pub(crate) blade: Arc<crate::platform::blade::IsolatedTextures>,
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
        self.failure.failed.load(Ordering::Acquire)
    }
    /// Resolves once this target suffers a terminal GPU failure, including if it
    /// already failed. Each caller is notified; cancelling a waiter is harmless.
    /// The future does not retain the target's textures or window reservation.
    pub fn failure(&self) -> impl std::future::Future<Output = ()> + Send + 'static {
        self.failure.wait.clone()
    }
    pub(crate) fn fail(&self) {
        self.failure.fail();
    }
}

// One shared completion instead of an AtomicWaker: replay/preparation owners may
// subscribe independently. Registration and failure cannot lose a wakeup.
pub(crate) struct FailureSignal {
    failed: AtomicBool,
    sender: parking_lot::Mutex<Option<oneshot::Sender<()>>>,
    wait: Shared<BoxFuture<'static, ()>>,
}
impl Default for FailureSignal {
    fn default() -> Self {
        let (sender, receiver) = oneshot::channel();
        Self {
            failed: AtomicBool::new(false),
            sender: parking_lot::Mutex::new(Some(sender)),
            // Sender drop is cancellation, not a device failure. A cancelled
            // owner drops its future; no completion should be synthesized.
            wait: async move {
                if receiver.await.is_err() {
                    futures::future::pending::<()>().await;
                }
            }
            .boxed()
            .shared(),
        }
    }
}
impl FailureSignal {
    fn fail(&self) {
        self.failed.store(true, Ordering::Release);
        let sender = self.sender.lock().take();
        if let Some(sender) = sender {
            let _ = sender.send(());
        }
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
    /// clip and opacity on success, error and unwinding panic. Hitboxes, cursor
    /// styles and deferred element draws are not isolated: do not record them here.
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
                            size: target.pixel_size.map(|v| crate::ScaledPixels(v as f32)),
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

#[cfg(test)]
mod notification_tests {
    use super::*;
    #[test]
    fn failure_wakes_all_waiters_including_late_subscribers() {
        futures::executor::block_on(async {
            let signal = Arc::new(FailureSignal::default());
            let mut first = Box::pin(signal.wait.clone());
            let mut second = Box::pin(signal.wait.clone());
            assert!(futures::poll!(&mut first).is_pending());
            assert!(futures::poll!(&mut second).is_pending());
            drop(signal.wait.clone()); // cancelling another subscriber changes nothing
            let worker = signal.clone();
            std::thread::spawn(move || worker.fail()).join().unwrap();
            assert!(signal.failed.load(Ordering::Acquire));
            assert!(futures::poll!(&mut first).is_ready());
            assert!(futures::poll!(&mut second).is_ready());
            assert!(signal.wait.clone().now_or_never().is_some());
            signal.fail(); // idempotent
        });
    }
}

// A single recursive walk retains nested targets as well as siblings through
// actual native completion, independently of window/Scene ownership.
pub(crate) fn collect_isolated_targets(scene: &crate::Scene, out: &mut Vec<PreparedIsolatedLayer>) {
    for layer in &scene.isolated {
        out.push(layer.target.clone());
        collect_isolated_targets(&layer.scene, out);
    }
}
#[cfg(not(target_os = "macos"))]
pub(crate) fn target_pixel_size(
    logical: Size<Pixels>,
    scale: f32,
) -> Result<Size<u32>, IsolatedLayerError> {
    let dimension = |n: Pixels| {
        let n = (f32::from(n) * scale).ceil() + 1.;
        if !n.is_finite() || n < 1. || n > 16384. {
            Err(IsolatedLayerError::Limit)
        } else {
            Ok(n as u32)
        }
    };
    Ok(crate::size(
        dimension(logical.width)?,
        dimension(logical.height)?,
    ))
}
