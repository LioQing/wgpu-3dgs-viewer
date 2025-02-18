use glam::*;

use wgpu::util::DeviceExt;

/// The indirect args storage buffer for [`Renderer`](crate::Renderer).
#[derive(Debug)]
pub struct IndirectArgsBuffer(wgpu::Buffer);

impl IndirectArgsBuffer {
    /// Create a new indirect args buffer.
    pub fn new(device: &wgpu::Device) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Indirect Args Buffer"),
            contents: wgpu::util::DrawIndirectArgs {
                vertex_count: 6,
                instance_count: 0,
                first_vertex: 0,
                first_instance: 0,
            }
            .as_bytes(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
        });

        Self(buffer)
    }

    /// Get the buffer.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.0
    }
}

/// The dispatch indirect args storage buffer for [`RadixSorter`](crate::RadixSorter).
#[derive(Debug)]
pub struct RadixSortIndirectArgsBuffer(wgpu::Buffer);

impl RadixSortIndirectArgsBuffer {
    /// Create a new dispatch indirect args buffer.
    pub fn new(device: &wgpu::Device) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Radix Sort Indirect Args Buffer"),
            contents: wgpu::util::DispatchIndirectArgs { x: 1, y: 1, z: 1 }.as_bytes(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
        });

        Self(buffer)
    }

    /// Get the buffer.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.0
    }
}

/// The indirect indices storage buffer for [`Renderer`](crate::Renderer).
#[derive(Debug)]
pub struct IndirectIndicesBuffer(wgpu::Buffer);

impl IndirectIndicesBuffer {
    /// Create a new indirect indices buffer.
    pub fn new(device: &wgpu::Device, gaussian_count: u32) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Indirect Indices Buffer"),
            size: (gaussian_count * std::mem::size_of::<u32>() as u32) as wgpu::BufferAddress,
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

/// The dipsatch indirect args storage buffer for [`Postprocessor`](crate::Postprocessor).
#[derive(Debug)]
pub struct PostprocessIndirectArgsBuffer(wgpu::Buffer);

impl PostprocessIndirectArgsBuffer {
    /// Create a new postprocessor indirect args buffer.
    pub fn new(device: &wgpu::Device) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Postprocessor Indirect Args Buffer"),
            size: std::mem::size_of::<wgpu::util::DispatchIndirectArgs>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
            mapped_at_creation: false,
        });

        Self(buffer)
    }

    /// Get the buffer.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.0
    }
}

/// The query cursor buffer for [`QueryCursor`](crate::QueryCursor).
///
/// This requires the `query-cursor` feature.
#[cfg(feature = "query-cursor")]
#[derive(Debug)]
pub struct QueryCursorBuffer(wgpu::Buffer);

#[cfg(feature = "query-cursor")]
impl QueryCursorBuffer {
    /// Create a new query cursor buffer.
    pub fn new(device: &wgpu::Device) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Query Cursor Buffer"),
            contents: bytemuck::cast_slice(&[QueryCursorPod::new(vec4(1.0, 1.0, 1.0, 1.0), 1.0)]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Self(buffer)
    }

    /// Update the query cursor buffer.
    pub fn update(&self, queue: &wgpu::Queue, outline_color: Vec4, outline_width: f32) {
        queue.write_buffer(
            &self.0,
            0,
            bytemuck::cast_slice(&[QueryCursorPod::new(outline_color, outline_width)]),
        );
    }

    /// Get the buffer.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.0
    }
}

/// The POD representation of a [`QueryCursor`].
///
/// This requires the `query-cursor` feature.
#[cfg(feature = "query-cursor")]
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct QueryCursorPod {
    /// The outline color.
    pub outline_color: Vec4,
    /// The outline width.
    pub outline_width: f32,
    /// Padding.
    _padding: [f32; 3],
}

#[cfg(feature = "query-cursor")]
impl QueryCursorPod {
    /// Create a new query cursor POD.
    pub fn new(outline_color: Vec4, outline_width: f32) -> Self {
        Self {
            outline_color,
            outline_width,
            _padding: [0.0; 3],
        }
    }
}
