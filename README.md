# 3D Gaussian Splatting Viewer

...written in Rust using [wgpu](https://wgpu.rs/).

## Usage

There are two ways to use this viewer:

1. As a standalone application

2. As a library

### Standalone Application

To run the standalone application, use the following command:

```sh
cargo run --bin wgpu-3dgs-viewer --features="bin" -- -m "path/to/model.ply"
```

Usage:

```
A 3D Gaussian splatting viewer written in Rust using wgpu.

Usage: wgpu-3dgs-viewer.exe --model <MODEL>

Options:
  -m, --model <MODEL>  Path to the .ply file
  -h, --help           Print help
  -V, --version        Print version
```

### Library

To use this viewer as a library, take a look at the [`wgpu-3dgs-viewer` binary](./src/bin.rs) for an example.