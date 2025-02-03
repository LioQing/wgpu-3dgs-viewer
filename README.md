# 3D Gaussian Splatting Viewer

...written in Rust using [wgpu](https://wgpu.rs/).

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
let camera = Camera::new(1e-4..1e4, 60f32.to_radians());

// Create the viewer
let viewer = Viewer::new(&device, surface_texture_format, &gaussians);

// ...

// Update the viewer's buffers each frame
viewer.update(
    &queue,
    &camera,
    uvec2(surface_config.width, surface_config.height),
);

// ...

// Render the viewer each frame
viewer.render(
    &mut encoder,
    &surface_texture_view,
    gaussians.gaussians.len() as u32,
);
```

You may also take a look at [the `simple-wgpu-3dgs-viewer` binary](./src/bin/simple.rs) for an example.
