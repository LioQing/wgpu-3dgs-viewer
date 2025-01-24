use std::ops::Range;

use glam::*;

/// A camera.
#[derive(Debug)]
pub struct Camera {
    /// The position of the camera.
    pos: Vec3,
    /// The z range of the camera.
    z: Range<f32>,
    /// The vertical FOV.
    vertical_fov: f32,
    /// The pitch.
    pitch: f32,
    /// The yaw.
    yaw: f32,
}

impl Camera {
    /// Up direction.
    pub const UP: Vec3 = Vec3::Y;

    /// The pitch limit.
    pub const PITCH_LIMIT: Range<f32> =
        -std::f32::consts::FRAC_PI_2 + 1e-6..std::f32::consts::FRAC_PI_2 - 1e-6;

    /// Create a new camera.
    pub fn new(z: Range<f32>, vertical_fov: f32) -> Self {
        Self {
            pos: Vec3::ZERO,
            z,
            vertical_fov,
            pitch: 0.0,
            yaw: 0.0,
        }
    }

    /// Get the position.
    pub fn pos(&self) -> Vec3 {
        self.pos
    }

    /// Get the z range.
    pub fn z(&self) -> &Range<f32> {
        &self.z
    }

    /// Get the vertical FOV.
    pub fn vertical_fov(&self) -> f32 {
        self.vertical_fov
    }

    /// Get the current pitch.
    pub fn pitch(&self) -> f32 {
        self.pitch
    }

    /// Get the current yaw.
    pub fn yaw(&self) -> f32 {
        self.yaw
    }

    /// Move the camera.
    pub fn move_by(&mut self, forward: f32, right: f32) {
        self.pos += self.get_forward() * forward + self.get_right() * right;
    }

    /// Move the camera forward.
    pub fn move_up(&mut self, up: f32) {
        self.pos += Self::UP * up;
    }

    /// Apply pitch.
    pub fn pitch_by(&mut self, delta: f32) {
        self.pitch = (self.pitch + delta).clamp(Self::PITCH_LIMIT.start, Self::PITCH_LIMIT.end);
    }

    /// Apply yaw.
    pub fn yaw_by(&mut self, delta: f32) {
        self.yaw = (self.yaw + delta).rem_euclid(2.0 * std::f32::consts::PI);
    }

    /// Get the forward vector.
    pub fn get_forward(&self) -> Vec3 {
        Vec3::new(
            self.pitch.cos() * self.yaw.sin(),
            self.pitch.sin(),
            self.pitch.cos() * self.yaw.cos(),
        )
    }

    /// Get the right vector.
    pub fn get_right(&self) -> Vec3 {
        self.get_forward().cross(Self::UP).normalize()
    }

    /// Get the view matrix.
    pub fn view(&self) -> Mat4 {
        Mat4::look_to_rh(self.pos(), self.get_forward(), Self::UP)
    }

    /// Get the projection matrix.
    pub fn projection(&self, aspect_ratio: f32) -> Mat4 {
        Mat4::perspective_rh(
            self.vertical_fov(),
            aspect_ratio,
            self.z().start,
            self.z().end,
        )
    }
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
    _padding_1: [u32; 2],
}

impl CameraPod {
    /// Create a new camera.
    pub fn new(camera: &Camera, size: UVec2) -> Self {
        Self {
            view: camera.view(),
            proj: camera.projection(size.x as f32 / size.y as f32),
            size: size.as_vec2(),
            _padding_1: [0; 2],
        }
    }
}
