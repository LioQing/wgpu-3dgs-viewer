use glam::*;

use wgpu::util::DeviceExt;

use crate::{wgpu_sort, Camera, Gaussian};

/// The Gaussians storage buffer.
#[derive(Debug)]
pub struct GaussiansBuffer(wgpu::Buffer);

impl GaussiansBuffer {
    /// Create a new Gaussians buffer.
    pub fn new(device: &wgpu::Device, gaussians: &[Gaussian]) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Gaussians Buffer"),
            contents: bytemuck::cast_slice(
                gaussians
                    .iter()
                    .map(GaussianPod::from_gaussian)
                    .collect::<Vec<_>>()
                    .as_slice(),
            ),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        Self(buffer)
    }

    /// Get the buffer.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.0
    }

    /// Get the number of Gaussians.
    pub fn len(&self) -> usize {
        self.0.size() as usize / std::mem::size_of::<Gaussian>()
    }

    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Update the buffer.
    pub fn update(&self, queue: &wgpu::Queue, gaussians: &[Gaussian]) {
        if gaussians.len() != self.len() {
            log::error!(
                "Gaussians count mismatch, buffer has {}, but {} were provided",
                self.len(),
                gaussians.len()
            );
            return;
        }

        queue.write_buffer(
            &self.0,
            0,
            bytemuck::cast_slice(
                gaussians
                    .iter()
                    .map(GaussianPod::from_gaussian)
                    .collect::<Vec<_>>()
                    .as_slice(),
            ),
        );
    }
}

/// The POD representation of Gaussian.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GaussianPod {
    pub pos: Vec3,
    pub color: U8Vec4,
    pub cov3d: [f32; 6],
    _padding: [f32; 2],
}

impl GaussianPod {
    /// Convert from Gaussian to Gaussian POD.
    pub fn from_gaussian(gaussian: &Gaussian) -> Self {
        // Covariance
        let r = Mat3::from_quat(gaussian.rotation);
        let s = Mat3::from_diagonal(gaussian.scale);
        let m = r * s;
        let sigma = m * m.transpose();
        let cov3d = [
            sigma.x_axis.x,
            sigma.x_axis.y,
            sigma.x_axis.z,
            sigma.y_axis.y,
            sigma.y_axis.z,
            sigma.z_axis.z,
        ];

        // Color
        let color = gaussian.color;

        // Position
        let pos = gaussian.pos;

        Self {
            pos,
            color,
            cov3d,
            _padding: [0.0; 2],
        }
    }
}

impl From<Gaussian> for GaussianPod {
    fn from(gaussian: Gaussian) -> Self {
        Self::from_gaussian(&gaussian)
    }
}

impl From<&Gaussian> for GaussianPod {
    fn from(gaussian: &Gaussian) -> Self {
        Self::from_gaussian(gaussian)
    }
}

/// The POD representation of Gaussian in PLY format.
///
/// Fields are stored as arrays because using glam types would add padding
/// according to C alignment rules.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PlyGaussianPod {
    pub pos: [f32; 3],
    pub n: [f32; 3],
    pub color: [f32; 3 * 16],
    pub alpha: f32,
    pub scale: [f32; 3],
    pub rotation: [f32; 4],
}

/// The camera buffer.
#[derive(Debug)]
pub struct CameraBuffer(wgpu::Buffer);

impl CameraBuffer {
    /// Create a new camera buffer.
    pub fn new(device: &wgpu::Device) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Camera Buffer"),
            size: std::mem::size_of::<CameraPod>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self(buffer)
    }

    /// Update the camera buffer.
    pub fn update(&self, queue: &wgpu::Queue, camera: &Camera, size: UVec2) {
        queue.write_buffer(
            &self.0,
            0,
            bytemuck::bytes_of(&CameraPod::new(camera, size)),
        );
    }

    /// Get the buffer.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.0
    }
}

/// The POD representation of camera.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraPod {
    pub view: Mat4,
    pub proj: Mat4,
    pub size: Vec2,
    _padding_0: [u32; 2],
}

impl CameraPod {
    /// Create a new camera.
    pub fn new(camera: &Camera, size: UVec2) -> Self {
        Self {
            view: camera.view(),
            proj: camera.projection(size.x as f32 / size.y as f32),
            size: size.as_vec2(),
            _padding_0: [0; 2],
        }
    }
}

/// The transformation buffer.
#[derive(Debug)]
pub struct TransformBuffer(wgpu::Buffer);

impl TransformBuffer {
    /// Create a new transformation buffer.
    pub fn new(device: &wgpu::Device) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Transform Buffer"),
            contents: bytemuck::bytes_of(&TransformPod::IDENTITY),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Self(buffer)
    }

    /// Update the transformation buffer.
    pub fn update(&self, queue: &wgpu::Queue, pos: Vec3, quat: Quat, scale: Vec3) {
        queue.write_buffer(
            &self.0,
            0,
            bytemuck::bytes_of(&TransformPod::new(pos, quat, scale)),
        );
    }

    /// Get the buffer.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.0
    }
}

/// The POD representation of a transformation.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TransformPod {
    pub pos: Vec3,
    _padding_0: f32,
    pub quat: Quat,
    pub scale: Vec3,
    _padding_1: f32,
}

impl TransformPod {
    /// The identity transformation.
    pub const IDENTITY: Self = Self::new(Vec3::ZERO, Quat::IDENTITY, Vec3::ONE);

    /// Create a new transformation.
    pub const fn new(pos: Vec3, quat: Quat, scale: Vec3) -> Self {
        Self {
            pos,
            _padding_0: 0.0,
            quat,
            scale,
            _padding_1: 0.0,
        }
    }
}

/// The Gaussians depth storage buffer.
#[derive(Debug)]
pub struct GaussiansDepthBuffer(wgpu::Buffer);

impl GaussiansDepthBuffer {
    /// Create a new Gaussians depth buffer.
    pub fn new(device: &wgpu::Device, gaussian_count: u32) -> Self {
        // Must correspond to [`crate::radix_sorter::wgpu_sort::GPUSorter::create_keyval_buffers`].
        let size = wgpu_sort::keys_buffer_size_bytes(gaussian_count);

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gaussians Depth Buffer"),
            size: size as wgpu::BufferAddress,
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
