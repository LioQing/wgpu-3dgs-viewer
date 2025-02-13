use glam::*;

use wgpu::util::DeviceExt;

/// The selection highlight uniform buffer for storing selection highlight data.
#[derive(Debug)]
pub struct SelectionHighlightBuffer(wgpu::Buffer);

impl SelectionHighlightBuffer {
    /// Create a new selection highlight buffer.
    pub fn new(device: &wgpu::Device) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Selection Highlight Buffer"),
            contents: bytemuck::cast_slice(&[SelectionHighlight::default()]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Self(buffer)
    }

    /// Update the selection highlight buffer.
    pub fn update(&self, queue: &wgpu::Queue, color: Vec4) {
        queue.write_buffer(
            &self.0,
            0,
            bytemuck::cast_slice(&[SelectionHighlight::new(color)]),
        );
    }

    /// Get the buffer.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.0
    }
}

/// The POD representation of the selection highlight.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SelectionHighlight {
    /// The selection color.
    ///
    /// The alpha value is for the selection highlight intensity, not the opacity.
    pub color: Vec4,
}

impl SelectionHighlight {
    /// Create a new selection highlight.
    pub fn new(color: Vec4) -> Self {
        Self { color }
    }
}

impl Default for SelectionHighlight {
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
