# 3D Gaussian Splatting Viewer

...written in Rust using [wgpu](https://wgpu.rs/).

[![Crates.io](https://img.shields.io/crates/v/wgpu-3dgs-viewer)](https://crates.io/crates/wgpu-3dgs-viewer) [![Docs.rs](https://img.shields.io/docsrs/wgpu-3dgs-viewer)](https://docs.rs/wgpu-3dgs-viewer/latest/wgpu_3dgs_viewer) [![License](https://img.shields.io/github/license/lioqing/wgpu-3dgs-viewer)](./LICENSE)

## Overview

### Introduction

This crate provides a low-level interface to render 3D Gaussian splatting using the [wgpu](https://wgpu.rs/) graphics API. It is designed to provide as much flexibility and extensibility as possible, while still being easy to use. It also provides function to load models very quickly from output of the original paper [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/).

### Features

- 🎨 **WebGPU**: [wgpu](https://wgpu.rs/), the Rust implementation of [WebGPU](https://www.w3.org/TR/webgpu/), provides safe, portable, and efficient rendering.
- 🤖 **Low-level**: Very close to the underlying WebGPU API, so you can use it as a low-level graphics API if you want, such as directly writing to the buffers and textures.
- 📦 **Compression**: Optionally compress the Gaussian splatting to reduce GPU memory usage and improve performance.
- 🔎 **Selection & Editing**: Support for selecting and editing the Gaussians to hide, override the color, adjust contrast, etc. for better visualization.
- 🏙️ **Multi-model**: Support for loading multiple models at once with customized rendering order.
- 🎭 **Masking**: Support for masking Gaussians with composite shapes, defined by complex set operations (union, intersection, difference, etc.).

### Demo

Simple (real-time rendering):

![simple](https://github.com/LioQing/wgpu-3dgs-viewer/blob/fe8f7093dfe8cfed2a9bace723d174b75a3e5a1c/media/simple.gif?raw=true)

Selection & Editing (multi-model viewer, custom centroid based rendering order, Gaussian removal):

![selection](https://github.com/LioQing/wgpu-3dgs-viewer/blob/fe8f7093dfe8cfed2a9bace723d174b75a3e5a1c/media/selection.gif?raw=true)

Masking (box and ellipsoid masks, depth testing):

![mask](https://github.com/LioQing/wgpu-3dgs-viewer/blob/fe8f7093dfe8cfed2a9bace723d174b75a3e5a1c/media/mask.gif?raw=true)

While there are examples provided, you can more directly see the viewer in action by going to my [3D Gaussian Splatting Viewer App](https://github.com/lioqing/wgpu-3dgs-viewer-app) which builds on this crate and provides a more user-friendly interface.

## Dependencies

This crate depends on the following crates:

| `wgpu-3dgs-viewer` | `wgpu` | `glam` |
| ------------------ | ------ | ------ |
| 0.1 - 0.2          | 24.0   | 0.29   |
| 0.3                | 25.0   | 0.30   |

## Usage

### Library

You can use it as a library to render 3D Gaussian splatting in your own application. 

Generally, the [`Viewer`] is sufficient for most use cases. However, you may directly use the individual components from the fields of [`Viewer`] if you want more control.

Example:

```rust
use wgpu_3dgs_viewer::{Camera, Gaussians, Viewer};
use glam::uvec2;

// ...

// Read the Gaussians from the .ply file
let f = std::fs::File::open(model_path).expect("ply file");
let mut reader = std::io::BufReader::new(f);
let gaussians = Gaussians::read_ply(&mut reader).expect("gaussians");

// Create the camera
let camera = Camera::new(0.1..1e4, 60f32.to_radians());

// Create the viewer
let mut viewer =
    Viewer::new(&device, config.view_formats[0], &gaussians).expect("viewer");

// ...

// Update the viewer's camera buffer
viewer.update_camera(
    &queue,
    &camera,
    uvec2(config.width, config.height),
);

// ...

// Render the viewer
viewer.render(
    &mut encoder,
    &texture_view,
    gaussians.gaussians.len() as u32,
);
```

You may also take a look at some binary examples:

- [`simple-wgpu-3dgs-viewer`](./src/bin/simple.rs): a simple example
- [`selection-wgpu-3dgs-viewer`](./src/bin/selection.rs): a selection and multi-model example
- [`mask-wgpu-3dgs-viewer`](./src/bin/mask.rs): a masking and depth testing example

### Standalone Examples

To run the examples, you can use the following commands:

```sh
cargo run --example simple -- "path/to/model.ply"
```

Usage:

```text
A 3D Gaussian splatting viewer written in Rust using wgpu.

Use W, A, S, D, Space, Shift to move, use mouse to rotate.


Usage: simple.exe --model <MODEL>

Options:
  -m, --model <MODEL>
          Path to the .ply file

  -h, --help
          Print help (see a summary with '-h')

  -V, --version
          Print version
```

Or try the multi-model example:

```sh
cargo run --example multi-model --features multi-model -- -m "path/to/model1.ply" -m "path/to/model2.ply" .. -o offsetx offsety offsetz
```

Usage:

```text
A 3D Gaussian splatting viewer written in Rust using wgpu.

Use W, A, S, D, Space, Shift to move, use mouse to rotate.


Usage: multi-model.exe [OPTIONS]

Options:
  -m, --models <MODELS>...
          Path to the .ply file

  -o, --offset <OFFSET> <OFFSET> <OFFSET>
          The offset of each model

          [default: 10.0,0.0,0.0]

  -h, --help
          Print help (see a summary with '-h')

  -V, --version
          Print version
```

## Acknowledgements

This crate uses modified code from [KeKsBoTer's wgpu_sort](https://crates.io/crates/wgpu_sort).

References are also taken from other 3D Gaussian splatting renderer implemntations, including [antimatter15's splat](https://github.com/antimatter15/splat), [KeKsBoTer's web-splat](https://github.com/KeKsBoTer/web-splat), and [Aras' Unity Gaussian Splatting](https://github.com/aras-p/UnityGaussianSplatting).
