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
//! // Create a selection bundle
//! editor::SelectionBundle::new::<GaussianPod>(
//!     &device,
//!     vec![selection::create_viewport_bundle::<GaussianPod>(&device)],
//! )
//! .unwrap();
//!
//! // Create a basic selection modifier
//! editor::BasicSelectionModifier::new(
//!     &device,
//!     &viewer.gaussians_buffer,
//!     &viewer.model_transform_buffer,
//!     &viewer.gaussian_transform_buffer,
//!     vec![selection::create_viewport_bundle::<GaussianPod>(&device)],
//! )
//! .unwrap();
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
