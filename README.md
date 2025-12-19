# 3D Gaussian Splatting Viewer

...written in Rust using [wgpu](https://wgpu.rs/).

[![Crates.io](https://img.shields.io/crates/v/wgpu-3dgs-viewer)](https://crates.io/crates/wgpu-3dgs-viewer)
[![Docs.rs](https://img.shields.io/docsrs/wgpu-3dgs-viewer)](https://docs.rs/wgpu-3dgs-viewer/latest/wgpu_3dgs_viewer)
[![Coverage](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2FLioQing%2Fwgpu-3dgs-viewer%2Frefs%2Fheads%2Fmaster%2Fcoverage%2Fbadge.json)](https://github.com/LioQing/wgpu-3dgs-viewer/tree/master/coverage)
[![License](https://img.shields.io/crates/l/wgpu-3dgs-viewer)](https://crates.io/crates/wgpu-3dgs-viewer)

![Cover.gif](https://raw.githubusercontent.com/LioQing/wgpu-3dgs-viewer/27a35021a67224b59eb6ed737ac4cfa33af91901/media/Cover.gif)

## Overview

> [!WARNING]
>
> This library is under active development, breaking API changes between versions may occur frequently.
>
> Use at your own risk.

This library displays 3D Gaussian Splatting models with wgpu. It includes a ready‑to‑use pipeline and modular pieces you can swap out.

- Rendering pipeline
  - Preprocess: cull off‑screen points and set up indirect draw data.
  - Sort and draw: sort by depth and draw the Gaussians.
  - Modes: Gaussians may be displayed as splat, ellipse, or point.
  - Transforms: apply model or per-Gaussian transforms.
- Abstraction for renderer and buffers
  - Viewer: one type that manages the buffers and pipelines.
  - Low-level access: preprocessor, sorter, renderer, and their buffers can be used separately.
- Optional features
  - Multi-model: render many models with custom draw orders.
  - Selection: viewport selection (e.g. rectangle, brush) that marks Gaussians for editing.
- Shaders
  - WGSL shaders packaged with WESL, you can extend or replace them.

## Usage

You may read the documentation of the following types for more details:

- [`Viewer`]: Manages buffers and renders a model.
  - [`Preprocessor`]: Culls Gaussians and fills indirect args and depths.
  - [`RadixSorter`]: Sorts Gaussians by depth on the GPU.
  - [`Renderer`]: Draws Gaussians with the selected display mode.
- [`MultiModelViewer`]: [`Viewer`] equivalent for multiple models. Requires `multi-model` feature.
- [`selection`]: Select Gaussians based on viewport interactions, e.g. rectangle or brush. Requires `selection` feature.

> [!TIP]
>
> The design principles of this crate are to provide modularity and flexibility to the end user of the API, which means exposing low-level WebGPU APIs. However, this means that you have to take care of your code when accessing low-level components. You risk breaking things at run-time if you don't handle them properly.
>
> If you do not want to take the risk, consider using the higher-level wrappers and avoid any instances of passing `wgpu` types into functions.

### Simple Viewer

You can use [`Viewer`] to render a single 3D Gaussian Splatting model:

```rust ignore
use wgpu_3dgs_viewer as gs;
use glam::UVec2;

// Setup wgpu...

// Read the Gaussians from the .ply file
let gaussians = gs::core::Gaussians::read_from_file(model_path, gs::core::GaussiansSource::Ply)
    .expect("gaussians");

// Create the camera
let camera = gs::Camera::new(0.1..1e4, 60f32.to_radians());

// Create the viewer
let mut viewer = gs::Viewer::new(&device, config.view_formats[0], &gaussians).expect("viewer");

// Setup camera parameters...

// Update the viewer's camera buffer
viewer.update_camera(
    &queue,
    &camera,
    UVec2::new(config.width, config.height),
);

// Create wgpu command encoder...

// Render the model
viewer.render(&mut encoder, &texture_view);
```

## Examples

See the [examples](https://github.com/LioQing/wgpu-3dgs-viewer/tree/master/examples) directory for usage examples.

## Dependencies

This crate depends on the following crates:

| `wgpu-3dgs-viewer` | `wgpu` | `glam` | `wesl` |
| ------------------ | ------ | ------ | ------ |
| 0.4                | 26.0   | 0.30   | 0.2    |
| 0.3                | 25.0   | 0.30   | N/A    |
| 0.1 - 0.2          | 24.0   | 0.29   | N/A    |

## Related Crates

- [wgpu-3dgs-editor](https://crates.io/crates/wgpu-3dgs-editor)
- [wgpu-3dgs-core](https://crates.io/crates/wgpu-3dgs-core)

## Acknowledgements

This crate uses modified code from [KeKsBoTer's wgpu_sort](https://crates.io/crates/wgpu_sort).

References are also taken from other 3D Gaussian splatting renderer implemntations, including [antimatter15's splat](https://github.com/antimatter15/splat), [KeKsBoTer's web-splat](https://github.com/KeKsBoTer/web-splat), and [Aras' Unity Gaussian Splatting](https://github.com/aras-p/UnityGaussianSplatting).
