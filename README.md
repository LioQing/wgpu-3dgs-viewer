# 3D Gaussian Splatting Viewer

...written in Rust using [wgpu](https://wgpu.rs/).

This crate is built for [3D Gaussian Splatting Viewer App](https://lioqing.com/wgpu-3dgs-viewer-app) which is also made by me.

## Usage

There are two ways to use this viewer:

1. As a standalone application

2. As a library

### Standalone Application

To run the standalone application, use the following command:

```sh
cargo run --bin simple-wgpu-3dgs-viewer --features="bin-simple" -- -m "path/to/model.ply"
```

Usage:

```
A 3D Gaussian splatting viewer written in Rust using wgpu.

In default mode, move the camera with W, A, S, D, Space, Shift, and rotate with mouse.
In selectio mode, click anywhere on the model to select the nearest Gaussian.
Use C to toggle between default and selection mode.

Usage: simple-wgpu-3dgs-viewer.exe --model <MODEL>

Options:
  -m, --model <MODEL>
          Path to the .ply file

  -h, --help
          Print help (see a summary with '-h')

  -V, --version
          Print version
```

### Library

Generally, the `Viewer` is sufficient for most use cases. However, you may directly use the individual components from the fields of `Viewer` if you want more control.

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

You may also take a look at [the `simple-wgpu-3dgs-viewer` binary](./src/bin/simple.rs) for a simple example, and [the `selection-wgpu-3dgs-viewer` binary](./src/bin/selection.rs) for an example with the selection related features enabled.
