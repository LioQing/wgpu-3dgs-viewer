#![doc = include_str!("../README.md")]

mod buffer;
mod camera;
mod error;
mod preprocessor;
mod radix_sorter;
mod renderer;
pub mod shader;
mod wesl_utils;

#[cfg(feature = "multi-model")]
mod multi_model;

#[cfg(feature = "selection")]
pub mod selection;

use glam::*;
use wgpu_3dgs_core::{
    GaussianDisplayMode, GaussianPod, GaussianPodWithShNorm8Cov3dHalfConfigs, GaussianShDegree,
    GaussianTransformBuffer, GaussianTransformPod, Gaussians, GaussiansBuffer,
    ModelTransformBuffer, ModelTransformPod,
};

pub use buffer::*;
pub use camera::*;
pub use error::*;
pub use preprocessor::*;
pub use radix_sorter::*;
pub use renderer::*;

#[cfg(feature = "multi-model")]
pub use multi_model::*;

pub use wgpu_3dgs_core as core;

#[cfg(feature = "editor")]
pub use wgpu_3dgs_editor as editor;

/// The default viewer [`GaussianPod`] type.
pub type DefaultGaussianPod = GaussianPodWithShNorm8Cov3dHalfConfigs;

/// The 3D Gaussian splatting viewer.
#[derive(Debug)]
pub struct Viewer<G: GaussianPod = DefaultGaussianPod> {
    pub camera_buffer: CameraBuffer,
    pub model_transform_buffer: ModelTransformBuffer,
    pub gaussian_transform_buffer: GaussianTransformBuffer,
    pub gaussians_buffer: GaussiansBuffer<G>,
    pub indirect_args_buffer: IndirectArgsBuffer,
    pub radix_sort_indirect_args_buffer: RadixSortIndirectArgsBuffer,
    pub indirect_indices_buffer: IndirectIndicesBuffer,
    pub gaussians_depth_buffer: GaussiansDepthBuffer,

    pub preprocessor: Preprocessor,
    pub radix_sorter: RadixSorter,
    pub renderer: Renderer,
}

impl<G: GaussianPod> Viewer<G> {
    /// Create a new viewer.
    pub fn new(
        device: &wgpu::Device,
        texture_format: wgpu::TextureFormat,
        gaussians: &Gaussians,
    ) -> Result<Self, Error> {
        Self::new_with(
            device,
            texture_format,
            None,
            GaussiansBuffer::<G>::DEFAULT_USAGE,
            gaussians,
        )
    }

    /// Create a new viewer with all options.
    pub fn new_with(
        device: &wgpu::Device,
        texture_format: wgpu::TextureFormat,
        depth_stencil: Option<wgpu::DepthStencilState>,
        gaussians_buffer_usage: wgpu::BufferUsages,
        gaussians: &Gaussians,
    ) -> Result<Self, Error> {
        log::debug!("Creating camera buffer");
        let camera_buffer = CameraBuffer::new(device);

        log::debug!("Creating model transform buffer");
        let model_transform_buffer = ModelTransformBuffer::new(device);

        log::debug!("Creating gaussian transform buffer");
        let gaussian_transform_buffer = GaussianTransformBuffer::new(device);

        log::debug!("Creating gaussians buffer");
        let gaussians_buffer =
            GaussiansBuffer::new_with_usage(device, &gaussians.gaussians, gaussians_buffer_usage);

        log::debug!("Creating indirect args buffer");
        let indirect_args_buffer = IndirectArgsBuffer::new(device);

        log::debug!("Creating radix sort indirect args buffer");
        let radix_sort_indirect_args_buffer = RadixSortIndirectArgsBuffer::new(device);

        log::debug!("Creating indirect indices buffer");
        let indirect_indices_buffer =
            IndirectIndicesBuffer::new(device, gaussians.gaussians.len() as u32);

        log::debug!("Creating gaussians depth buffer");
        let gaussians_depth_buffer =
            GaussiansDepthBuffer::new(device, gaussians.gaussians.len() as u32);

        log::debug!("Creating preprocessor");
        let preprocessor = Preprocessor::new(
            device,
            &camera_buffer,
            &model_transform_buffer,
            &gaussians_buffer,
            &indirect_args_buffer,
            &radix_sort_indirect_args_buffer,
            &indirect_indices_buffer,
            &gaussians_depth_buffer,
        )?;

        log::debug!("Creating radix sorter");
        let radix_sorter =
            RadixSorter::new(device, &gaussians_depth_buffer, &indirect_indices_buffer);

        log::debug!("Creating renderer");
        let renderer = Renderer::new(
            device,
            texture_format,
            depth_stencil,
            &camera_buffer,
            &model_transform_buffer,
            &gaussian_transform_buffer,
            &gaussians_buffer,
            &indirect_indices_buffer,
        )?;

        log::info!("Viewer created");

        Ok(Self {
            camera_buffer,
            model_transform_buffer,
            gaussian_transform_buffer,
            gaussians_buffer,
            indirect_args_buffer,
            radix_sort_indirect_args_buffer,
            indirect_indices_buffer,
            gaussians_depth_buffer,

            preprocessor,
            radix_sorter,
            renderer,
        })
    }

    /// Update the camera.
    pub fn update_camera(
        &mut self,
        queue: &wgpu::Queue,
        camera: &impl CameraTrait,
        texture_size: UVec2,
    ) {
        self.camera_buffer.update(queue, camera, texture_size);
    }

    /// Update the camera with [`CameraPod`].
    pub fn update_camera_with_pod(&mut self, queue: &wgpu::Queue, pod: &CameraPod) {
        self.camera_buffer.update_with_pod(queue, pod);
    }

    /// Update the model transform.
    pub fn update_model_transform(
        &mut self,
        queue: &wgpu::Queue,
        pos: Vec3,
        rot: Quat,
        scale: Vec3,
    ) {
        self.model_transform_buffer.update(queue, pos, rot, scale);
    }

    /// Update the model transform with [`ModelTransformPod`].
    pub fn update_model_transform_with_pod(
        &mut self,
        queue: &wgpu::Queue,
        pod: &ModelTransformPod,
    ) {
        self.model_transform_buffer.update_with_pod(queue, pod);
    }

    /// Update the Gaussian transform.
    pub fn update_gaussian_transform(
        &mut self,
        queue: &wgpu::Queue,
        size: f32,
        display_mode: GaussianDisplayMode,
        sh_deg: GaussianShDegree,
        no_sh0: bool,
    ) {
        self.gaussian_transform_buffer
            .update(queue, size, display_mode, sh_deg, no_sh0);
    }

    /// Update the Gaussian transform with [`GaussianTransformPod`].
    pub fn update_gaussian_transform_with_pod(
        &mut self,
        queue: &wgpu::Queue,
        pod: &GaussianTransformPod,
    ) {
        self.gaussian_transform_buffer.update_with_pod(queue, pod);
    }

    /// Render the viewer.
    pub fn render(&self, encoder: &mut wgpu::CommandEncoder, texture_view: &wgpu::TextureView) {
        self.preprocessor
            .preprocess(encoder, self.gaussians_buffer.len() as u32);

        self.radix_sorter
            .sort(encoder, &self.radix_sort_indirect_args_buffer);

        self.renderer
            .render(encoder, texture_view, &self.indirect_args_buffer);
    }
}
