use std::ops::Range;

use glam::*;

/// A camera.
#[derive(Debug)]
pub struct Camera {
    /// The position of the camera.
    pub pos: Vec3,
    /// The z range of the camera.
    pub z: Range<f32>,
    /// The vertical FOV.
    pub vertical_fov: f32,
    /// The pitch.
    pub pitch: f32,
    /// The yaw.
    pub yaw: f32,
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
        Mat4::look_to_rh(self.pos, self.get_forward(), Self::UP)
    }

    /// Get the projection matrix.
    pub fn projection(&self, aspect_ratio: f32) -> Mat4 {
        Mat4::perspective_rh(self.vertical_fov, aspect_ratio, self.z.start, self.z.end)
    }
}
