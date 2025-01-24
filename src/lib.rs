mod buffer;
mod camera;
mod error;
mod gaussian;
mod preprocessor;
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
    #[allow(dead_code)]
    camera_buffer: CameraBuffer,
    #[allow(dead_code)]
    gaussians_buffer: GaussiansBuffer,
    #[allow(dead_code)]
    indirect_args_buffer: IndirectArgsBuffer,
    #[allow(dead_code)]
    radix_sort_indirect_args_buffer: RadixSortIndirectArgsBuffer,
    #[allow(dead_code)]
    indirect_indices_buffer: IndirectIndicesBuffer,
    #[allow(dead_code)]
    gaussians_depth_buffer: GaussiansDepthBuffer,

    preprocessor: Preprocessor,
    radix_sorter: RadixSorter,
    renderer: Renderer,
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

        log::debug!("Creating gaussians buffer");
        let gaussians_buffer = GaussiansBuffer::new(device, gaussians.gaussians());

        log::debug!("Creating indirect args buffer");
        let indirect_args_buffer = IndirectArgsBuffer::new(device);

        log::debug!("Creating radix sort indirect args buffer");
        let radix_sort_indirect_args_buffer = RadixSortIndirectArgsBuffer::new(device);

        log::debug!("Creating indirect indices buffer");
        let indirect_indices_buffer =
            IndirectIndicesBuffer::new(device, gaussians.gaussians().len() as u32);

        log::debug!("Creating gaussians depth buffer");
        let gaussians_depth_buffer =
            GaussiansDepthBuffer::new(device, gaussians.gaussians().len() as u32);

        log::debug!("Creating preprocessor");
        let preprocessor = Preprocessor::new(
            device,
            &camera_buffer,
            &gaussians_buffer,
            &indirect_args_buffer,
            &radix_sort_indirect_args_buffer,
            &indirect_indices_buffer,
            &gaussians_depth_buffer,
        );

        log::debug!("Creating radix sorter");
        let radix_sorter =
            RadixSorter::new(device, &gaussians_depth_buffer, &indirect_indices_buffer);

        log::debug!("Creating renderer");
        let renderer = Renderer::new(
            device,
            texture_format,
            &camera_buffer,
            &gaussians_buffer,
            &indirect_indices_buffer,
        );

        log::info!("Viewer created");

        Self {
            camera_buffer,
            gaussians_buffer,
            indirect_args_buffer,
            radix_sort_indirect_args_buffer,
            indirect_indices_buffer,
            gaussians_depth_buffer,

            preprocessor,
            radix_sorter,
            renderer,
        }
    }

    /// Update the viewer.
    ///
    /// Call this function before every render, the buffers are updated from the CPU to the GPU.
    pub fn update(&mut self, queue: &wgpu::Queue, camera: &Camera, texture_size: UVec2) {
        self.camera_buffer.update(queue, camera, texture_size);
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
}
