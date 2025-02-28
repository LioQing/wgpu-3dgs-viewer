#![doc = include_str!("../README.md")]

mod buffer;
mod camera;
mod error;
mod gaussian;
mod postprocessor;
mod preprocessor;
pub mod query;
mod radix_sorter;
mod renderer;

#[cfg(feature = "query-texture-tool")]
mod query_texture_tool;

#[cfg(feature = "query-texture-overlay")]
mod query_texture_overlay;

#[cfg(feature = "query-tool")]
mod query_tool;

#[cfg(feature = "query-toolset")]
pub mod query_toolset;

#[cfg(feature = "query-cursor")]
pub mod query_cursor;

use glam::*;

pub use buffer::*;
pub use camera::*;
pub use error::*;
pub use gaussian::*;
pub use postprocessor::*;
pub use preprocessor::*;
pub use radix_sorter::*;
pub use renderer::*;

#[cfg(feature = "query-texture-tool")]
pub use query_texture_tool::*;

#[cfg(feature = "query-texture-overlay")]
pub use query_texture_overlay::*;

#[cfg(feature = "query-tool")]
pub use query_tool::*;

#[cfg(feature = "query-toolset")]
pub use query_toolset::*;

#[cfg(feature = "query-cursor")]
pub use query_cursor::*;

/// The 3D Gaussian splatting viewer.
#[derive(Debug)]
pub struct Viewer<G: GaussianPod = GaussianPodWithShNorm8Cov3dHalfConfigs> {
    pub camera_buffer: CameraBuffer,
    pub model_transform_buffer: ModelTransformBuffer,
    pub gaussian_transform_buffer: GaussianTransformBuffer,
    pub gaussians_buffer: GaussiansBuffer<G>,
    pub indirect_args_buffer: IndirectArgsBuffer,
    pub radix_sort_indirect_args_buffer: RadixSortIndirectArgsBuffer,
    pub indirect_indices_buffer: IndirectIndicesBuffer,
    pub gaussians_depth_buffer: GaussiansDepthBuffer,
    pub query_buffer: QueryBuffer,
    pub query_result_count_buffer: QueryResultCountBuffer,
    pub query_results_buffer: QueryResultsBuffer,
    pub postprocess_indirect_args_buffer: PostprocessIndirectArgsBuffer,
    pub selection_highlight_buffer: SelectionHighlightBuffer,
    pub selection_buffer: SelectionBuffer,
    pub gaussians_edit_buffer: GaussiansEditBuffer,
    pub selection_edit_buffer: SelectionEditBuffer,

    #[cfg(feature = "query-texture")]
    pub query_texture: QueryTexture,

    pub preprocessor: Preprocessor,
    pub radix_sorter: RadixSorter,
    pub renderer: Renderer,
    pub postprocessor: Postprocessor,
}

impl<G: GaussianPod> Viewer<G> {
    /// Create a new viewer.
    pub fn new(
        device: &wgpu::Device,
        texture_format: wgpu::TextureFormat,
        #[cfg(feature = "query-texture")] texture_size: UVec2,
        gaussians: &Gaussians,
    ) -> Result<Self, Error> {
        log::debug!("Creating camera buffer");
        let camera_buffer = CameraBuffer::new(device);

        log::debug!("Creating model transform buffer");
        let model_transform_buffer = ModelTransformBuffer::new(device);

        log::debug!("Creating gaussian transform buffer");
        let gaussian_transform_buffer = GaussianTransformBuffer::new(device);

        log::debug!("Creating gaussians buffer");
        let gaussians_buffer = GaussiansBuffer::new(device, &gaussians.gaussians);

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

        log::debug!("Creating query buffer");
        let query_buffer = QueryBuffer::new(device);

        log::debug!("Creating query result count buffer");
        let query_result_count_buffer = QueryResultCountBuffer::new(device);

        log::debug!("Creating query results buffer");
        let query_results_buffer =
            QueryResultsBuffer::new(device, gaussians.gaussians.len() as u32);

        log::debug!("Creating postprocess indirect args buffer");
        let postprocess_indirect_args_buffer = PostprocessIndirectArgsBuffer::new(device);

        log::debug!("Creating selection highlight buffer");
        let selection_highlight_buffer = SelectionHighlightBuffer::new(device);

        log::debug!("Creating selection buffer");
        let selection_buffer = SelectionBuffer::new(device, gaussians.gaussians.len() as u32);

        log::debug!("Creating gaussians edit buffer");
        let gaussians_edit_buffer =
            GaussiansEditBuffer::new(device, gaussians.gaussians.len() as u32);

        log::debug!("Creating selection edit buffer");
        let selection_edit_buffer = SelectionEditBuffer::new(device);

        #[cfg(feature = "query-texture")]
        let query_texture = {
            log::debug!("Creating query texture");
            QueryTexture::new(device, texture_size)
        };

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
            &query_buffer,
            &query_result_count_buffer,
            &query_results_buffer,
            &gaussians_edit_buffer,
            &selection_buffer,
            &selection_edit_buffer,
            #[cfg(feature = "query-texture")]
            &query_texture,
        )?;

        log::debug!("Creating radix sorter");
        let radix_sorter =
            RadixSorter::new(device, &gaussians_depth_buffer, &indirect_indices_buffer);

        log::debug!("Creating renderer");
        let renderer = Renderer::new(
            device,
            texture_format,
            &camera_buffer,
            &model_transform_buffer,
            &gaussian_transform_buffer,
            &gaussians_buffer,
            &indirect_indices_buffer,
            &query_buffer,
            &query_result_count_buffer,
            &query_results_buffer,
            &selection_highlight_buffer,
            &selection_buffer,
            &gaussians_edit_buffer,
        )?;

        log::debug!("Creating postprocessor");
        let postprocessor = Postprocessor::new(
            device,
            &postprocess_indirect_args_buffer,
            &query_buffer,
            &query_result_count_buffer,
            &query_results_buffer,
            &selection_buffer,
        );

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
            query_buffer,
            query_result_count_buffer,
            query_results_buffer,
            postprocess_indirect_args_buffer,
            selection_highlight_buffer,
            selection_buffer,
            gaussians_edit_buffer,
            selection_edit_buffer,

            #[cfg(feature = "query-texture")]
            query_texture,

            preprocessor,
            radix_sorter,
            renderer,
            postprocessor,
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

    /// Update the query.
    ///
    /// There can only be one query at a time.
    pub fn update_query(&mut self, queue: &wgpu::Queue, query: &QueryPod) {
        self.query_buffer.update(queue, query);
    }

    /// Update the model transform.
    pub fn update_model_transform(
        &mut self,
        queue: &wgpu::Queue,
        pos: Vec3,
        quat: Quat,
        scale: Vec3,
    ) {
        self.model_transform_buffer.update(queue, pos, quat, scale);
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

    /// Update the selection highlight.
    pub fn update_selection_highlight(&mut self, queue: &wgpu::Queue, color: Vec4) {
        self.selection_highlight_buffer.update(queue, color);
    }

    /// Update the selection highlight with [`SelectionHighlightPod`].
    pub fn update_selection_highlight_with_pod(
        &mut self,
        queue: &wgpu::Queue,
        pod: &SelectionHighlightPod,
    ) {
        self.selection_highlight_buffer.update_with_pod(queue, pod);
    }

    /// Update the selection edit.
    ///
    /// Set [`GaussianEditFlag::ENABLED`] to apply the edits on the selected Gaussians.
    #[allow(clippy::too_many_arguments)]
    pub fn update_selection_edit(
        &mut self,
        queue: &wgpu::Queue,
        flag: GaussianEditFlag,
        hsv: Vec3,
        contrast: f32,
        exposure: f32,
        gamma: f32,
        alpha: f32,
    ) {
        self.selection_edit_buffer
            .update(queue, flag, hsv, contrast, exposure, gamma, alpha);
    }

    /// Update the selection edit with [`GaussianEditPod`].
    pub fn update_selection_edit_with_pod(&mut self, queue: &wgpu::Queue, pod: &GaussianEditPod) {
        self.selection_edit_buffer.update_with_pod(queue, pod);
    }

    /// Update the query texture size.
    ///
    /// This requires the `query-texture` feature.
    #[cfg(feature = "query-texture")]
    pub fn update_query_texture_size(&mut self, device: &wgpu::Device, size: UVec2) {
        self.query_texture.update_size(device, size);
        self.preprocessor.update_bind_group(
            device,
            &self.camera_buffer,
            &self.model_transform_buffer,
            &self.gaussians_buffer,
            &self.indirect_args_buffer,
            &self.radix_sort_indirect_args_buffer,
            &self.indirect_indices_buffer,
            &self.gaussians_depth_buffer,
            &self.query_buffer,
            &self.query_result_count_buffer,
            &self.query_results_buffer,
            &self.gaussians_edit_buffer,
            &self.selection_buffer,
            &self.selection_edit_buffer,
            &self.query_texture,
        );
    }

    /// Render the viewer.
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        texture_view: &wgpu::TextureView,
        gaussian_count: u32,
    ) {
        self.preprocessor.preprocess(encoder, gaussian_count);

        self.radix_sorter
            .sort(encoder, &self.radix_sort_indirect_args_buffer);

        self.renderer
            .render(encoder, texture_view, &self.indirect_args_buffer);

        self.postprocessor.postprocess(
            encoder,
            gaussian_count,
            &self.postprocess_indirect_args_buffer,
        );
    }

    /// Download the query results from the GPU.
    pub async fn download_query_results(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<QueryResultPod>, Error> {
        query::download(
            device,
            queue,
            &self.query_result_count_buffer,
            &self.query_results_buffer,
        )
        .await
    }
}
