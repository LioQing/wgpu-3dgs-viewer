# Changelog

## 0.4.0 - Unreleased

This is a big update! We are splitting the project into multiple crates to make it more modular and easier to use.

### Added

- 🔦 Shaders are now [WESL](https://wesl-lang.dev/) which is more modular.

### Removed

- ❌ All the masking, editing, querying, and selection features. These features are available in the new `wgpu-3dgs-editor` crate with a different API.

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
