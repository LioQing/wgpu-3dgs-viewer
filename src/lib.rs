mod buffer;
mod camera;
mod error;
mod gaussian;
mod preprocessor;
pub mod query;
mod radix_sorter;
mod renderer;

use glam::*;

pub use buffer::*;
pub use camera::*;
pub use error::*;
pub use gaussian::*;
pub use preprocessor::*;
pub use radix_sorter::*;
pub use renderer::*;

/// The 3D Gaussian splatting viewer.
#[derive(Debug)]
pub struct Viewer {
    pub camera_buffer: CameraBuffer,
    pub model_transform_buffer: ModelTransformBuffer,
    pub gaussian_transform_buffer: GaussianTransformBuffer,
    pub gaussians_buffer: GaussiansBuffer,
    pub indirect_args_buffer: IndirectArgsBuffer,
    pub radix_sort_indirect_args_buffer: RadixSortIndirectArgsBuffer,
    pub indirect_indices_buffer: IndirectIndicesBuffer,
    pub gaussians_depth_buffer: GaussiansDepthBuffer,
    pub query_buffer: QueryBuffer,
    pub query_result_count_buffer: QueryResultCountBuffer,
    pub query_results_buffer: QueryResultsBuffer,

    pub preprocessor: Preprocessor,
    pub radix_sorter: RadixSorter,
    pub renderer: Renderer,
}

impl Viewer {
    /// Create a new viewer.
    pub fn new(
        device: &wgpu::Device,
        texture_format: wgpu::TextureFormat,
        gaussians: &Gaussians,
    ) -> Self {
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
        );

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
        );

        log::info!("Viewer created");

        Self {
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

            preprocessor,
            radix_sorter,
            renderer,
        }
    }

    /// Update the camera.
    pub fn update_camera(&mut self, queue: &wgpu::Queue, camera: &Camera, texture_size: UVec2) {
        self.camera_buffer.update(queue, camera, texture_size);
    }

    /// Update the query.
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

    /// Update the Gaussian transform.
    pub fn update_gaussian_transform(
        &mut self,
        queue: &wgpu::Queue,
        size: f32,
        display_mode: GaussianDisplayMode,
    ) {
        self.gaussian_transform_buffer
            .update(queue, size, display_mode);
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
