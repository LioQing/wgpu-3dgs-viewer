//! Level of Detail (LOD) for 3D Gaussian Splatting.
//!
//! LOD works by sorting gaussians by visual importance (size × opacity) and
//! limiting how many are dispatched to the GPU preprocessor. Since the
//! preprocessor already accepts a `gaussian_count` parameter, no shader
//! modifications are required.
//!
//! # Usage
//!
//! Gaussians should be sorted by importance **before** creating the viewer
//! (i.e. before uploading to the GPU buffer). Then at runtime, adjust the
//! budget via [`LodConfig`] to control quality vs performance.
//!
//! ```rust,ignore
//! use wgpu_3dgs_viewer::lod::{LodConfig, sort_gaussians_by_importance};
//!
//! // Sort once before creating the Viewer.
//! let mut gaussians: Vec<Gaussian> = load_gaussians();
//! sort_gaussians_by_importance(&mut gaussians);
//!
//! // At runtime, toggle LOD on/off and adjust budget.
//! let mut lod = LodConfig::default();
//! lod.enabled = true;
//! lod.budget = 500_000; // render at most 500k gaussians
//! ```

use crate::core::IterGaussian;
use glam::Vec3;

/// Configuration for Level of Detail rendering.
///
/// When enabled, the viewer will render at most `budget` gaussians per frame.
/// Gaussians should be pre-sorted by importance (most important first) so that
/// reducing the budget drops the least important splats.
#[derive(Debug, Clone)]
pub struct LodConfig {
    /// Whether LOD is enabled. When `false`, all gaussians are rendered.
    pub enabled: bool,
    /// Maximum number of gaussians to render when LOD is enabled.
    pub budget: u32,
}

impl Default for LodConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            budget: u32::MAX,
        }
    }
}

impl LodConfig {
    /// Create a new LOD config with the given budget.
    pub fn with_budget(budget: u32) -> Self {
        Self {
            enabled: true,
            budget,
        }
    }

    /// Return the effective gaussian count, clamped to the budget if enabled.
    pub fn effective_count(&self, total: u32) -> u32 {
        if self.enabled {
            total.min(self.budget)
        } else {
            total
        }
    }
}

/// Compute the visual importance of a gaussian.
///
/// Importance = volume (product of scales) × opacity (alpha channel).
/// Gaussians with higher importance contribute more to the final image.
pub fn gaussian_importance(scale: Vec3, opacity: u8) -> f32 {
    (scale.x * scale.y * scale.z).abs() * (opacity as f32)
}

/// Sort gaussians by importance in descending order (most important first).
///
/// Call this **before** creating the [`Viewer`](crate::Viewer) so that the GPU
/// buffer contains gaussians ordered by importance. Then the LOD budget can
/// simply truncate the dispatch count.
pub fn sort_gaussians_by_importance(gaussians: &impl IterGaussian) -> Vec<crate::core::Gaussian> {
    let mut sorted: Vec<_> = gaussians.iter_gaussian().collect();
    sorted.sort_by(|a, b| {
        let ia = gaussian_importance(a.scale, a.color.w);
        let ib = gaussian_importance(b.scale, b.color.w);
        ib.partial_cmp(&ia).unwrap_or(std::cmp::Ordering::Equal)
    });
    sorted
}

/// Axis-aligned bounding box.
#[derive(Debug, Clone, Copy)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    /// Compute the AABB of a set of gaussians.
    pub fn from_gaussians(gaussians: &impl IterGaussian) -> Self {
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        for g in gaussians.iter_gaussian() {
            min = min.min(g.pos);
            max = max.max(g.pos);
        }
        Aabb { min, max }
    }

    /// Center of the bounding box.
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// Half-extents of the bounding box.
    pub fn half_extents(&self) -> Vec3 {
        (self.max - self.min) * 0.5
    }

    /// Check if a point is inside the AABB.
    pub fn contains(&self, point: Vec3) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
            && point.z >= self.min.z
            && point.z <= self.max.z
    }
}
