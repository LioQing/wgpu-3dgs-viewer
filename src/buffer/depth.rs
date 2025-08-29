use crate::{core::BufferWrapper, wgpu_sort};

/// The Gaussians depth storage buffer.
#[derive(Debug, Clone)]
pub struct GaussiansDepthBuffer(wgpu::Buffer);

impl GaussiansDepthBuffer {
    /// Create a new Gaussians depth buffer.
    pub fn new(device: &wgpu::Device, gaussian_count: u32) -> Self {
        // Must correspond to [`crate::radix_sorter::wgpu_sort::GPUSorter::create_keyval_buffers`].
        let size = wgpu_sort::keys_buffer_size_bytes(gaussian_count);

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gaussians Depth Buffer"),
            size: size as wgpu::BufferAddress,
            usage: Self::DEFAULT_USAGES,
            mapped_at_creation: false,
        });

        Self(buffer)
    }
}

impl BufferWrapper for GaussiansDepthBuffer {
    const DEFAULT_USAGES: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE;

    fn buffer(&self) -> &wgpu::Buffer {
        &self.0
    }
}

impl From<GaussiansDepthBuffer> for wgpu::Buffer {
    fn from(wrapper: GaussiansDepthBuffer) -> Self {
        wrapper.0
    }
}
