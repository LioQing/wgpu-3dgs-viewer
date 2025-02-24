use glam::*;

use wgpu::util::DeviceExt;

use super::{GaussianEditFlag, GaussianEditPod};

/// The selection highlight uniform buffer for storing selection highlight data.
#[derive(Debug)]
pub struct SelectionHighlightBuffer(wgpu::Buffer);

impl SelectionHighlightBuffer {
    /// Create a new selection highlight buffer.
    pub fn new(device: &wgpu::Device) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Selection Highlight Buffer"),
            contents: bytemuck::cast_slice(&[SelectionHighlightPod::default()]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Self(buffer)
    }

    /// Update the selection highlight buffer.
    pub fn update(&self, queue: &wgpu::Queue, color: Vec4) {
        self.update_with_pod(queue, &SelectionHighlightPod::new(color));
    }

    /// Update the selection highlight buffer with [`SelectionHighlightPod`].
    pub fn update_with_pod(&self, queue: &wgpu::Queue, pod: &SelectionHighlightPod) {
        queue.write_buffer(&self.0, 0, bytemuck::bytes_of(pod));
    }

    /// Get the buffer.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.0
    }
}

/// The POD representation of the selection highlight.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SelectionHighlightPod {
    /// The selection color.
    ///
    /// The alpha value is for the selection highlight intensity, not the opacity.
    pub color: Vec4,
}

impl SelectionHighlightPod {
    /// Create a new selection highlight.
    pub fn new(color: Vec4) -> Self {
        Self { color }
    }
}

impl Default for SelectionHighlightPod {
    fn default() -> Self {
        Self::new(vec4(1.0, 0.0, 1.0, 1.0))
    }
}

/// The selection storage buffer for storing selected Gaussians as a bitvec.
#[derive(Debug)]
pub struct SelectionBuffer(wgpu::Buffer);

impl SelectionBuffer {
    /// Create a new selection buffer.
    pub fn new(device: &wgpu::Device, gaussian_count: u32) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Selection Buffer"),
            size: (gaussian_count.div_ceil(32) * std::mem::size_of::<u32>() as u32)
                as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        Self(buffer)
    }

    /// Get the buffer.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.0
    }
}

/// The selection edit uniform buffer for editing selected Gaussians.
#[derive(Debug)]
pub struct SelectionEditBuffer(wgpu::Buffer);

impl SelectionEditBuffer {
    /// Create a new selection edit buffer.
    pub fn new(device: &wgpu::Device) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Selection Edit Buffer"),
            contents: bytemuck::cast_slice(&[GaussianEditPod::default()]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Self(buffer)
    }

    /// Update the selection edit buffer.
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &self,
        queue: &wgpu::Queue,
        flag: GaussianEditFlag,
        hsv: Vec3,
        contrast: f32,
        exposure: f32,
        gamma: f32,
        alpha: f32,
    ) {
        self.update_with_pod(
            queue,
            &GaussianEditPod::new(flag, hsv, contrast, exposure, gamma, alpha),
        );
    }

    /// Update the selection edit buffer with [`GaussianEditPod`].
    pub fn update_with_pod(&self, queue: &wgpu::Queue, pod: &GaussianEditPod) {
        queue.write_buffer(&self.0, 0, bytemuck::bytes_of(pod));
    }

    /// Get the buffer.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.0
    }
}
