# Codicula GPUI release branch

This branch of `xudesheng/zed` carries the standalone published GPUI 0.2.2 package, followed by Codicula's native affine/prepared-line changes. It does not change the fork's main branch or import a newer Zed editor.

Base: crates.io `gpui` 0.2.2, archive SHA-256 `979b45cfa6ec723b6f42330915a1b3769b930d02b2d505f9697f8ca602bee707`. The package records upstream commit `69e2130295c2649963eb639fc70b4f2ee8ea1624` with `dirty: true`; the published bytes, including the normalized Cargo manifest and registry support-crate dependencies, are the actual base. This independent release branch preserves that exact package instead of treating the upstream commit as an identical source tree. `.cargo-ok` is registry-local state and is omitted.

The original Apache-2.0 license and notices are retained. Codicula consumers pin an immutable commit, not this moving branch name. Ordinary editor glyph APIs retain zoom 1 behavior. New prepared/affine behavior is validated on macOS; other platform conformance is not claimed. Rebase the native changes onto later published packages only with native/editor regression checks.
