//! Render world integration for Bevy's render graph.
//!
//! This module implements the extract → prepare → render pipeline for
//! 3D Gaussian Splatting within Bevy's rendering architecture.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use bevy::prelude::*;
use bevy::render::camera::ExtractedCamera;
use bevy::render::render_graph::{NodeRunError, RenderGraphContext, RenderLabel, ViewNode};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue};
use bevy::render::view::{ExtractedView, ViewTarget};
use bevy::render::Extract;

use crate::core::{Gaussians, IterGaussian};
use crate::{CameraPod, Viewer};

use super::{GaussianCloud, GaussianSplatSettings};

// ---------------------------------------------------------------------------
// Render graph label
// ---------------------------------------------------------------------------

/// Render graph label for the Gaussian Splatting pass.
#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub(crate) struct GaussianSplattingLabel;

// ---------------------------------------------------------------------------
// Extracted data (main world → render world)
// ---------------------------------------------------------------------------

/// Data extracted from the main world each frame.
#[derive(Resource, Default)]
pub(crate) struct ExtractedGaussianData {
    pub clouds: Vec<ExtractedCloud>,
}

/// A single gaussian cloud extracted from the main world.
pub(crate) struct ExtractedCloud {
    pub entity: Entity,
    pub gaussians: Arc<Gaussians>,
    pub transform: GlobalTransform,
    pub settings: GaussianSplatSettings,
}

// ---------------------------------------------------------------------------
// Pipeline resource (render world)
// ---------------------------------------------------------------------------

/// Holds the per-entity [`Viewer`] instances and the target texture format.
#[derive(Resource)]
pub(crate) struct GaussianSplattingPipeline {
    viewers: HashMap<Entity, ViewerState>,
    texture_format: wgpu::TextureFormat,
}

impl GaussianSplattingPipeline {
    pub fn new(texture_format: wgpu::TextureFormat) -> Self {
        Self {
            viewers: HashMap::new(),
            texture_format,
        }
    }
}

/// Per-entity viewer state holding the GPU pipeline for one gaussian cloud.
struct ViewerState {
    viewer: Viewer,
}

// ---------------------------------------------------------------------------
// Extract system
// ---------------------------------------------------------------------------

/// Copies gaussian cloud data from the main world into the render world.
///
/// The [`Gaussians`] data is behind an [`Arc`], so extraction is cheap
/// (just an Arc clone) even for large point clouds.
pub(crate) fn extract_gaussian_clouds(
    mut commands: Commands,
    clouds: Extract<
        Query<(
            Entity,
            &GaussianCloud,
            &GlobalTransform,
            Option<&GaussianSplatSettings>,
        )>,
    >,
) {
    let mut data = ExtractedGaussianData::default();

    for (entity, cloud, transform, settings) in clouds.iter() {
        data.clouds.push(ExtractedCloud {
            entity,
            gaussians: cloud.gaussians.clone(),
            transform: *transform,
            settings: settings.cloned().unwrap_or_default(),
        });
    }

    commands.insert_resource(data);
}

// ---------------------------------------------------------------------------
// Prepare system
// ---------------------------------------------------------------------------

/// Creates / updates [`Viewer`] instances for each gaussian cloud entity.
///
/// - New entities get a fresh `Viewer` created on the GPU.
/// - Existing entities have their model transform and display settings updated.
/// - Entities that no longer exist are cleaned up.
pub(crate) fn prepare_gaussian_clouds(
    mut pipeline: ResMut<GaussianSplattingPipeline>,
    extracted: Res<ExtractedGaussianData>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    let device = render_device.wgpu_device();
    let queue: &wgpu::Queue = &**render_queue;
    let format = pipeline.texture_format;

    // Remove viewers whose entities no longer exist.
    let active: HashSet<Entity> = extracted.clouds.iter().map(|c| c.entity).collect();
    pipeline.viewers.retain(|e, _| active.contains(e));

    // Create viewers for new entities.
    for cloud in &extracted.clouds {
        if !pipeline.viewers.contains_key(&cloud.entity) {
            match Viewer::new(device, format, cloud.gaussians.as_ref()) {
                Ok(viewer) => {
                    pipeline.viewers.insert(
                        cloud.entity,
                        ViewerState { viewer },
                    );
                }
                Err(e) => {
                    error!("Failed to create gaussian splatting viewer: {e}");
                }
            }
        }
    }

    // Update model transforms and display settings.
    for cloud in &extracted.clouds {
        if let Some(state) = pipeline.viewers.get_mut(&cloud.entity) {
            let (scale, rot, pos) = cloud.transform.to_scale_rotation_translation();
            state
                .viewer
                .update_model_transform(queue, pos, rot, scale);
            state.viewer.update_gaussian_transform(
                queue,
                cloud.settings.size,
                cloud.settings.display_mode,
                cloud.settings.sh_degree,
                cloud.settings.no_sh0,
                cloud.settings.max_std_dev,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Render graph node
// ---------------------------------------------------------------------------

/// Render graph node that runs the 3D Gaussian Splatting pipeline.
///
/// For each active [`Viewer`]:
/// 1. Updates the camera uniform buffer from Bevy's extracted camera.
/// 2. Dispatches the preprocess compute pass (frustum culling + depth).
/// 3. Dispatches the radix sort compute pass (depth ordering).
/// 4. Renders the sorted gaussian splats into the view target with alpha
///    blending, compositing over whatever was previously rendered.
#[derive(Default)]
pub struct GaussianSplattingNode;

impl ViewNode for GaussianSplattingNode {
    type ViewQuery = (
        &'static ViewTarget,
        &'static ExtractedView,
        &'static ExtractedCamera,
    );

    fn run<'w>(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        (view_target, extracted_view, extracted_camera): bevy::ecs::query::QueryItem<
            'w,
            '_,
            Self::ViewQuery,
        >,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let pipeline = world.resource::<GaussianSplattingPipeline>();
        let render_queue = world.resource::<RenderQueue>();
        let queue: &wgpu::Queue = &**render_queue;

        if pipeline.viewers.is_empty() {
            return Ok(());
        }

        // Build the camera uniform from Bevy's extracted camera data.
        let view_matrix = extracted_view.world_from_view.compute_matrix().inverse();
        let proj_matrix = extracted_view.clip_from_view;
        let viewport_size = extracted_camera
            .physical_viewport_size
            .unwrap_or(UVec2::new(1920, 1080));

        let camera_pod = CameraPod {
            view: view_matrix,
            proj: proj_matrix,
            size: viewport_size.as_vec2(),
            _padding: [0; 2],
        };

        // Update camera buffer for every viewer.
        for state in pipeline.viewers.values() {
            state
                .viewer
                .camera_buffer
                .update_with_pod(queue, &camera_pod);
        }

        let encoder = render_context.command_encoder();

        // ---- Compute passes (preprocess + radix sort) for all viewers ----
        for state in pipeline.viewers.values() {
            state
                .viewer
                .preprocessor
                .preprocess(encoder, state.viewer.effective_gaussian_count());
            state
                .viewer
                .radix_sorter
                .sort(encoder, &state.viewer.radix_sort_indirect_args_buffer);
        }

        // ---- Render pass: draw all viewers into the view target ----
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Gaussian Splatting Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: view_target.main_texture_view(),
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // Load (not Clear) so we composite over existing content.
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                ..Default::default()
            });

            for state in pipeline.viewers.values() {
                state.viewer.renderer.render_with_pass(
                    &mut render_pass,
                    &state.viewer.indirect_args_buffer,
                );
            }
        }

        Ok(())
    }
}
