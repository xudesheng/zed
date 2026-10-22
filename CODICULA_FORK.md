# Codicula GPUI release branch

This branch of `xudesheng/zed` carries the standalone published GPUI 0.2.2 package, followed by Codicula's native affine/prepared-line changes. It does not change the fork's main branch or import a newer Zed editor.

Base: crates.io `gpui` 0.2.2, archive SHA-256 `979b45cfa6ec723b6f42330915a1b3769b930d02b2d505f9697f8ca602bee707`. The package records upstream commit `69e2130295c2649963eb639fc70b4f2ee8ea1624` with `dirty: true`; the published bytes, including the normalized Cargo manifest and registry support-crate dependencies, are the actual base. This independent release branch preserves that exact package instead of treating the upstream commit as an identical source tree. `.cargo-ok` is registry-local state and is omitted.

The original Apache-2.0 license and notices are retained. Codicula consumers pin an immutable commit, not this moving branch name. Ordinary editor glyph APIs retain zoom 1 behavior. New prepared/affine behavior is validated on macOS; other platform conformance is not claimed. Rebase the native changes onto later published packages only with native/editor regression checks.

## Native isolated layers

The release branch also supplies `Window::prepare_isolated_layer` and
`paint_isolated_layer` for macOS/Metal. Other backends reject preparation with
`Unsupported`. Ordinary `paint_layer`, glyph APIs and ordinary-window blend
pipelines retain their prior behavior.

Each target owns a private **tracked-hazard** Metal heap containing BGRA8 color,
BGRA8 path intermediate and 4× MSAA attachments. The 32 MiB per-window pool reserves
queried heap texture sizes/alignments and host VM page padding before allocating.
All diagrams, nested groups, retained/replayed records and command-buffer owners
share that quota. A target that does not fit fails the whole new diagram; there is
no tiling or silent alpha substitution. A 1600×1000 logical target at scale 2 needs
at least 153,600,000 attachment bytes before padding and is deliberately rejected.

Recording restores Scene/clip/opacity state on errors and unwind. Child records
translate by the integer physical target origin, preserving fractional raster
phase. A separate ordered composite primitive retains the child Scene and target
through cached replay. Each placement is rendered and consumed before target
reuse; nested children complete before their parent composite. Separate offscreen
blend pipelines use source-over alpha, and the composite multiplies premultiplied
RGBA once. GPU completion retains reservations and poisons shared failure state on
error. Owners must inspect `is_failed()` before publishing or repainting a retained
generation; application scheduling/font-generation invalidation remains the
consumer's responsibility. The closure is for native primitive recording, not
nested element lifecycle/cache management. Recursive use of the same attachment
is rejected; distinct placements of one prepared target are supported.

Validation includes real Metal readback for overlapping/nested alpha and repeated
target placements, replay, quota exhaustion/recovery, encoding-failure state
restoration, command-buffer retention after Scene drop, and window probes for
mixed paths/glyphs/clips at four zooms. This is macOS evidence, not other-backend
conformance. Keep all generated captures/logs outside this repository.

Isolated targets additionally expose `failure()`, a multi-subscriber completion
future. Terminal GPU command-buffer failure wakes current owners without a new
input/frame; late subscribers observe the same completion. Waiters retain no
textures or quota reservations. Cancelling a waiter is harmless. Cached Scene
replay skips only a failed composite so unrelated window content still presents.
Missing drawables and the instance-buffer growth cap are transient frame skips,
not permanent target failures. Only genuine terminal GPU errors poison retained
targets. Recording closures must not create hitboxes, cursor styles or deferred
element draws; only Scene, clip and opacity state are isolated/restored.
