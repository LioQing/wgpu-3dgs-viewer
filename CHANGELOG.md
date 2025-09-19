# Changelog

## 0.4.0 - Unreleased

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
