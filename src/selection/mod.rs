//! Selection of Gaussians via viewport interactions.
//!
//! This module provides tools to select Gaussians using viewport-based interactions,
//! such as rectangle selection and brush selection.
//!
//! To get started, take a look at [`ViewportSelector`], which provides the highest level of
//! abstraction for viewport selection. It manages user interaction states and holds a
//! [`ViewportTexture`] to store the selection mask.
//!
//! The [`core::ComputeBundle`](crate::core::ComputeBundle) created by [`create_viewport_bundle`] is
//! used to evaluate [`ViewportTexture`] on Gaussians. You should create a
//! [`editor::SelectionBundle`](crate::editor::SelectionBundle) to select the Gaussians, or use the
//! [`editor::BasicSelectionModifier`](crate::editor::BasicSelectionModifier) to select and modify
//! basic attributes of the selected Gaussians.
//!
//! ```rust
//! # use pollster::FutureExt;
//! #
//! # async {
//! # use wgpu_3dgs_viewer::{
//! #     Viewer,
//! #     core::{self, glam::*},
//! #     editor, selection,
//! # };
//! #
//! # type GaussianPod = core::GaussianPodWithShSingleCov3dSingleConfigs;
//! #
//! # let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
//! #
//! # let adapter = instance
//! #     .request_adapter(&wgpu::RequestAdapterOptions::default())
//! #     .await
//! #     .expect("adapter");
//! #
//! # let (device, _queue) = adapter
//! #     .request_device(&wgpu::DeviceDescriptor {
//! #         label: Some("Device"),
//! #         required_features: wgpu::Features::empty(),
//! #         required_limits: adapter.limits(),
//! #         memory_hints: wgpu::MemoryHints::default(),
//! #         trace: wgpu::Trace::Off,
//! #     })
//! #     .await
//! #     .expect("device");
//! #
//! # let viewer = Viewer::<GaussianPod>::new(
//! #     &device,
//! #     wgpu::TextureFormat::Rgba8UnormSrgb,
//! #     &core::Gaussians {
//! #         gaussians: vec![core::Gaussian {
//! #             rot: Quat::IDENTITY,
//! #             pos: Vec3::ZERO,
//! #             color: U8Vec4::ZERO,
//! #             sh: [Vec3::ZERO; 15],
//! #             scale: Vec3::ONE,
//! #         }],
//! #     },
//! # )
//! # .unwrap();
//! #
//! // Create a selection bundle
//! editor::SelectionBundle::<GaussianPod>::new(
//!     &device,
//!     vec![selection::create_viewport_bundle::<GaussianPod>(&device)],
//! );
//!
//! // Create a basic selection modifier
//! editor::SelectionModifier::new_with_basic_modifier(
//!     &device,
//!     &viewer.gaussians_buffer,
//!     &viewer.model_transform_buffer,
//!     &viewer.gaussian_transform_buffer,
//!     vec![selection::create_viewport_bundle::<GaussianPod>(&device)],
//! );
//! # }.block_on();
//! ```
//!
//! If you wish to use other editor features, consider using the re-exported
//! [`editor`](crate::editor) module, and read through its documentation.

mod buffer;
mod viewport;
mod viewport_selector;
mod viewport_texture_brush;
mod viewport_texture_rectangle;

pub use buffer::*;
pub use viewport::*;
pub use viewport_selector::*;
pub use viewport_texture_brush::*;
pub use viewport_texture_rectangle::*;
