use glam::*;
use wgpu::util::DeviceExt;

use crate::core::{self, BufferWrapper, FixedSizeBufferWrapper};

/// The indirect args storage buffer for [`Renderer`](crate::Renderer).
#[derive(Debug, Clone)]
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
            usage: Self::DEFAULT_USAGES,
        });

        Self(buffer)
    }
}

impl BufferWrapper for IndirectArgsBuffer {
    const DEFAULT_USAGES: wgpu::BufferUsages = wgpu::BufferUsages::from_bits_retain(
        wgpu::BufferUsages::STORAGE.bits() | wgpu::BufferUsages::INDIRECT.bits(),
    );

    fn buffer(&self) -> &wgpu::Buffer {
        &self.0
    }
}

impl From<IndirectArgsBuffer> for wgpu::Buffer {
    fn from(wrapper: IndirectArgsBuffer) -> Self {
        wrapper.0
    }
}

impl TryFrom<wgpu::Buffer> for IndirectArgsBuffer {
    type Error = core::Error;

    fn try_from(buffer: wgpu::Buffer) -> Result<Self, Self::Error> {
        Self::verify_buffer_size(&buffer).map(|()| Self(buffer))
    }
}

impl FixedSizeBufferWrapper for IndirectArgsBuffer {
    type Pod = wgpu::util::DrawIndirectArgs;
}

/// The dispatch indirect args storage buffer for [`RadixSorter`](crate::RadixSorter).
#[derive(Debug, Clone)]
pub struct RadixSortIndirectArgsBuffer(wgpu::Buffer);

impl RadixSortIndirectArgsBuffer {
    /// Create a new dispatch indirect args buffer.
    pub fn new(device: &wgpu::Device) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Radix Sort Indirect Args Buffer"),
            contents: wgpu::util::DispatchIndirectArgs { x: 1, y: 1, z: 1 }.as_bytes(),
            usage: Self::DEFAULT_USAGES,
        });

        Self(buffer)
    }
}

impl BufferWrapper for RadixSortIndirectArgsBuffer {
    const DEFAULT_USAGES: wgpu::BufferUsages = wgpu::BufferUsages::from_bits_retain(
        wgpu::BufferUsages::STORAGE.bits() | wgpu::BufferUsages::INDIRECT.bits(),
    );

    fn buffer(&self) -> &wgpu::Buffer {
        &self.0
    }
}

impl From<RadixSortIndirectArgsBuffer> for wgpu::Buffer {
    fn from(wrapper: RadixSortIndirectArgsBuffer) -> Self {
        wrapper.0
    }
}

impl TryFrom<wgpu::Buffer> for RadixSortIndirectArgsBuffer {
    type Error = core::Error;

    fn try_from(buffer: wgpu::Buffer) -> Result<Self, Self::Error> {
        Self::verify_buffer_size(&buffer).map(|()| Self(buffer))
    }
}

impl FixedSizeBufferWrapper for RadixSortIndirectArgsBuffer {
    type Pod = wgpu::util::DispatchIndirectArgs;
}

/// The indirect indices storage buffer for [`Renderer`](crate::Renderer).
#[derive(Debug, Clone)]
pub struct IndirectIndicesBuffer(wgpu::Buffer);

impl IndirectIndicesBuffer {
    /// Create a new indirect indices buffer.
    pub fn new(device: &wgpu::Device, gaussian_count: u32) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Indirect Indices Buffer"),
            size: (gaussian_count * std::mem::size_of::<u32>() as u32) as wgpu::BufferAddress,
            usage: Self::DEFAULT_USAGES,
            mapped_at_creation: false,
        });

        Self(buffer)
    }
}

impl BufferWrapper for IndirectIndicesBuffer {
    const DEFAULT_USAGES: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE;

    fn buffer(&self) -> &wgpu::Buffer {
        &self.0
    }
}

impl From<IndirectIndicesBuffer> for wgpu::Buffer {
    fn from(wrapper: IndirectIndicesBuffer) -> Self {
        wrapper.0
    }
}
