use glam::*;
use wgpu_3dgs_core::BufferWrapper;

use crate::CameraTrait;

/// The camera buffer.
#[derive(Debug, Clone)]
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
    pub fn update(&self, queue: &wgpu::Queue, camera: &impl CameraTrait, size: UVec2) {
        self.update_with_pod(queue, &CameraPod::new(camera, size));
    }

    /// Update the camera buffer with [`CameraPod`].
    pub fn update_with_pod(&self, queue: &wgpu::Queue, pod: &CameraPod) {
        queue.write_buffer(&self.0, 0, bytemuck::bytes_of(pod));
    }
}

impl BufferWrapper for CameraBuffer {
    fn buffer(&self) -> &wgpu::Buffer {
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
    pub _padding: [u32; 2],
}

impl CameraPod {
    /// Create a new camera.
    pub fn new(camera: &impl CameraTrait, size: UVec2) -> Self {
        Self {
            view: camera.view(),
            proj: camera.projection(size.x as f32 / size.y as f32),
            size: size.as_vec2(),
            _padding: [0; 2],
        }
    }
}
