# 3D Gaussian Splatting Viewer

...written in Rust using [wgpu](https://wgpu.rs/).

[![Crates.io](https://img.shields.io/crates/v/wgpu-3dgs-viewer)](https://crates.io/crates/wgpu-3dgs-viewer) [![Docs.rs](https://img.shields.io/docsrs/wgpu-3dgs-viewer)](https://docs.rs/wgpu-3dgs-viewer/latest/wgpu_3dgs_viewer) [![License](https://img.shields.io/github/license/lioqing/wgpu-3dgs-viewer)](./LICENSE)

## Overview

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

## Demo

Simple (real-time rendering):

![simple](https://github.com/LioQing/wgpu-3dgs-viewer/blob/fe8f7093dfe8cfed2a9bace723d174b75a3e5a1c/media/simple.gif?raw=true)

Selection & Editing (multi-model viewer, custom centroid based rendering order, Gaussian removal):

![selection](https://github.com/LioQing/wgpu-3dgs-viewer/blob/fe8f7093dfe8cfed2a9bace723d174b75a3e5a1c/media/selection.gif?raw=true)

Masking (box and ellipsoid masks, depth testing):

![mask](https://github.com/LioQing/wgpu-3dgs-viewer/blob/fe8f7093dfe8cfed2a9bace723d174b75a3e5a1c/media/mask.gif?raw=true)

While there are examples provided, you can more directly see the viewer in action by going to my [3D Gaussian Splatting Viewer App](https://github.com/lioqing/wgpu-3dgs-viewer-app) which builds on this crate and provides a more user-friendly interface.

## Usage

You may read the documentation of the following types for more details:
- [`Viewer`]: Manages buffers and renders a model.
    - [`Preprocessor`]: Culls Gaussians and fills indirect args and depths.
    - [`RadixSorter`]: Sorts Gaussians by depth on the GPU.
    - [`Renderer`]: Draws Gaussians with the selected display mode.
- [`MultiModelViewer`]: [`Viewer`] equivalent for multiple models. Requires `multi-model` feature.
- [`selection`]: Select Gaussians based on viewport interactions, e.g. rectangle or brush. Requires `selection` feature.

## Dependencies

This crate depends on the following crates:

| `wgpu-3dgs-viewer` | `wgpu` | `glam` | `wesl` |
| ------------------ | ------ | ------ | ------ |
| 0.4                | 26.0   | 0.30   | 0.2    |
| 0.3                | 25.0   | 0.30   | N/A    |
| 0.1 - 0.2          | 24.0   | 0.29   | N/A    |

## Acknowledgements

This crate uses modified code from [KeKsBoTer's wgpu_sort](https://crates.io/crates/wgpu_sort).

References are also taken from other 3D Gaussian splatting renderer implemntations, including [antimatter15's splat](https://github.com/antimatter15/splat), [KeKsBoTer's web-splat](https://github.com/KeKsBoTer/web-splat), and [Aras' Unity Gaussian Splatting](https://github.com/aras-p/UnityGaussianSplatting).
