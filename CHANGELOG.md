# Changelog

Please also check out the [`wgpu-3dgs-editor` changelog](https://github.com/LioQing/wgpu-3dgs-editor/blob/master/CHANGELOG.md) and the [`wgpu-3dgs-core` changelog](https://github.com/LioQing/wgpu-3dgs-core/blob/master/CHANGELOG.md).

## [0.6.0](https://crates.io/crates/wgpu-3dgs-viewer/0.6.0) - 2026-01-11

### Added

- 🤖 CI workflow. [#17](https://github.com/LioQing/wgpu-3dgs-viewer/pull/17)

### Changed

- ⚡ Upgrade `wgpu` to 28.0, `wesl` to 0.3, `half` to 2.7, and `bytemuck` to 1.24. [#14](https://github.com/LioQing/wgpu-3dgs-viewer/pull/14)

## [0.5.0](https://crates.io/crates/wgpu-3dgs-viewer/0.5.0) - 2025-12-30

🎅 Merry Christmas, and in advance Happy New Year! 🎉

This release doesn't have signficant new features to this crate or `wgpu-3dgs-editor`, but `wgpu-3dgs-core` has major updates including [SPZ](https://github.com/nianticlabs/spz) support!

While `wgpu` and `wesl` versions are lagging behind, I will try to keep them up-to-date in the next releases hopefully in early 2026.

### Added

- 🎨 Add `ViewerCreateOptions` for more flexible viewer creation. [#12](https://github.com/LioQing/wgpu-3dgs-viewer/pull/12)

### Changed

- ⚡ Upgrade `wgpu` to 27.0 and `bitflags` to 2.10. [#13](https://github.com/LioQing/wgpu-3dgs-viewer/pull/13)
- 🔍 Update usage of `gaussian_unpack_sh` to zero-based indexing. [#10](https://github.com/LioQing/wgpu-3dgs-viewer/pull/10)

### Breaking Changes

- Rename `Viewer::new_with` and `MultiModelViewer::new_with` to `new_with_options`. [#12](https://github.com/LioQing/wgpu-3dgs-viewer/pull/12)

## [0.4.1](https://crates.io/crates/wgpu-3dgs-viewer/0.4.1) - 2025-10-01

### Added

- 📑 Add example modules documentations.

### Changed

- 🩹 Fix compilation error on viewport selection.

## [0.4.0](https://crates.io/crates/wgpu-3dgs-viewer/0.4.0) - 2025-09-20

This is a big rework update! We are splitting the project into multiple crates to make it more modular and easier to use.

[`wgpu-3dgs-editor`](https://github.com/LioQing/wgpu-3dgs-editor) and [`wgpu-3dgs-core`](https://github.com/LioQing/wgpu-3dgs-core) are now available for editing and core functionalities respectively. You may also access them via `wgpu-3dgs-viewer::editor` (requires `editor` feature) and `wgpu-3dgs-viewer::core`.

### Added

- 🔦 Shaders are now [WESL](https://wesl-lang.dev/) which is more modular.
- 📜 The source code can now also be licensed under Apache 2.0, just like Rust's source code.
- 🔪 Improved frustum culling, Gaussians will not pop in and out at the edge now.
- 🏔️ Added option to use different maximum standard deviation in `GaussianTransform`.

### Removed

- ✈️ All the masking, editing, and selection features. These features are available in the new [`wgpu-3dgs-editor`](https://github.com/LioQing/wgpu-3dgs-editor) crate.
- ❌ Query and selection gizmo features are removed (may be added back in the future).

### Changed

- 🔄 Update `wgpu` to 26.0.
- 🪛 Make `Preprocessor` and `Renderer` take `GaussianPod` as a generic parameter to enforce buffer safety.
- 🫥 Viewport related selection (brush and rectangle selections) is now available via the `selection` feature and module.
- 🏃‍➡️ Binaries of the crate are now examples, since they were not very complete anyway.
- 👓 Fixed blurry rendering due to wrong focal and standard deviation calculation.

## [0.3.0](https://crates.io/crates/wgpu-3dgs-viewer/0.3.0) - 2025-05-14

### Changed

- 🔄 Update `wgpu` to 25.0 and `glam` to 0.30.

## [0.2.0](https://crates.io/crates/wgpu-3dgs-viewer/0.2.0) - 2025-04-03

Some major new features and improvements have been added to the viewer.

### Added

- 🔢 Multi-model viewer to see multiple Gaussian models.
- 🎭 Masking with composite shapes, i.e. boxes and ellipsoids.
- 🎥 Depth stencil as an option to render with.
- ⏬ Download for Gaussian edits and masks.

### Changed

- ✅ Update to Rust 2024 edition.
- ⏫ Increase the capability of the viewer to handle larger models.

## [0.1.0](https://crates.io/crates/wgpu-3dgs-viewer/0.1.0) - 2025-02-27

The first version of this project.

### Added

- ⭐ Everything!
