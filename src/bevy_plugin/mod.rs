//! Bevy plugin for 3D Gaussian Splatting.
//!
//! # Version Compatibility
//!
//! This module requires that Bevy's internal `wgpu` version matches the `wgpu`
//! version used by this crate (28.0). Since Bevy 0.18 ships with wgpu 27 and
//! Bevy 0.19-dev uses wgpu 29, you may need to use `[patch.crates-io]` in your
//! workspace `Cargo.toml` to align the wgpu versions.
//!
//! # Usage
//!
//! ```rust,no_run
//! use bevy::prelude::*;
//! use wgpu_3dgs_viewer::bevy_plugin::{
//!     GaussianSplattingPlugin, GaussianCloud, GaussianSplatSettings,
//! };
//!
//! fn main() {
//!     App::new()
//!         .add_plugins((DefaultPlugins, GaussianSplattingPlugin::default()))
//!         .add_systems(Startup, setup)
//!         .run();
//! }
//!
//! fn setup(mut commands: Commands) {
//!     commands.spawn((
//!         Camera3d::default(),
//!         Transform::from_xyz(0.0, 0.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
//!     ));
//!
//!     commands.spawn((
//!         GaussianCloud::from_file("path/to/model.ply"),
//!         GaussianSplatSettings::default(),
//!         Transform::default(),
//!     ));
//! }
//! ```

mod render;

use std::sync::Arc;

use bevy::prelude::*;
use bevy::render::render_graph::{RenderGraphExt, ViewNodeRunner};
use bevy::render::{ExtractSchedule, Render, RenderApp, RenderSet};

use crate::core::{
    GaussianDisplayMode, GaussianMaxStdDev, GaussianShDegree, Gaussians, GaussiansSource,
};

pub use render::GaussianSplattingNode;

/// Bevy plugin for rendering 3D Gaussian Splats.
///
/// Add this to your Bevy app alongside `DefaultPlugins`:
///
/// ```rust,no_run
/// # use bevy::prelude::*;
/// # use wgpu_3dgs_viewer::bevy_plugin::GaussianSplattingPlugin;
/// App::new()
///     .add_plugins((DefaultPlugins, GaussianSplattingPlugin::default()))
///     .run();
/// ```
pub struct GaussianSplattingPlugin {
    /// The texture format to use for the Gaussian splat render pipeline.
    ///
    /// This must match the format of the camera's render target.
    /// Defaults to `Rgba16Float` (Bevy's HDR format for `Camera3d`).
    pub texture_format: wgpu::TextureFormat,
}

impl Default for GaussianSplattingPlugin {
    fn default() -> Self {
        Self {
            texture_format: wgpu::TextureFormat::Rgba16Float,
        }
    }
}

impl Plugin for GaussianSplattingPlugin {
    fn build(&self, app: &mut App) {
        let Ok(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .insert_resource(render::GaussianSplattingPipeline::new(self.texture_format))
            .init_resource::<render::ExtractedGaussianData>()
            .add_systems(ExtractSchedule, render::extract_gaussian_clouds)
            .add_systems(
                Render,
                render::prepare_gaussian_clouds.in_set(RenderSet::Prepare),
            );

        use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};

        render_app
            .add_render_graph_node::<ViewNodeRunner<GaussianSplattingNode>>(
                Core3d,
                render::GaussianSplattingLabel,
            )
            .add_render_graph_edges(
                Core3d,
                (Node3d::MainTransparentPass, render::GaussianSplattingLabel),
            );
    }
}

/// Component that marks an entity as a 3D Gaussian splat cloud.
///
/// Attach this to an entity along with a [`Transform`] and optionally
/// [`GaussianSplatSettings`] to render gaussian splats in your scene.
///
/// # Example
///
/// ```rust,no_run
/// # use bevy::prelude::*;
/// # use wgpu_3dgs_viewer::bevy_plugin::{GaussianCloud, GaussianSplatSettings};
/// # fn setup(mut commands: Commands) {
/// commands.spawn((
///     GaussianCloud::from_file("model.ply"),
///     GaussianSplatSettings::default(),
///     Transform::from_rotation(Quat::from_rotation_z(std::f32::consts::PI)),
/// ));
/// # }
/// ```
#[derive(Component)]
pub struct GaussianCloud {
    /// The loaded gaussian data (shared via Arc for cheap extraction).
    pub(crate) gaussians: Arc<Gaussians>,
}

impl GaussianCloud {
    /// Load a gaussian cloud from a `.ply` or `.spz` file.
    ///
    /// # Panics
    ///
    /// Panics if the file cannot be loaded as either PLY or SPZ format.
    pub fn from_file(path: &str) -> Self {
        let gaussians = [GaussiansSource::Ply, GaussiansSource::Spz]
            .into_iter()
            .find_map(|source| Gaussians::read_from_file(path, source).ok())
            .unwrap_or_else(|| panic!("Failed to load gaussian cloud from: {path}"));

        Self {
            gaussians: Arc::new(gaussians),
        }
    }

    /// Create a gaussian cloud from pre-loaded gaussian data.
    pub fn from_gaussians(gaussians: Gaussians) -> Self {
        Self {
            gaussians: Arc::new(gaussians),
        }
    }
}

/// Display settings for gaussian splats.
///
/// Controls how the gaussian splats are rendered (size, display mode, SH degree, etc.).
#[derive(Component, Clone)]
pub struct GaussianSplatSettings {
    /// Size multiplier for the gaussians.
    pub size: f32,
    /// Display mode: `Splat`, `Ellipse`, or `Point`.
    pub display_mode: GaussianDisplayMode,
    /// Spherical harmonics degree (0-3).
    pub sh_degree: GaussianShDegree,
    /// Whether to hide the SH0 (DC) component.
    pub no_sh0: bool,
    /// Maximum standard deviation for gaussian rendering.
    pub max_std_dev: GaussianMaxStdDev,
}

impl Default for GaussianSplatSettings {
    fn default() -> Self {
        Self {
            size: 1.0,
            display_mode: GaussianDisplayMode::Splat,
            sh_degree: GaussianShDegree::new(3).expect("valid sh degree"),
            no_sh0: false,
            max_std_dev: GaussianMaxStdDev::new(3.0).expect("valid max std dev"),
        }
    }
}
