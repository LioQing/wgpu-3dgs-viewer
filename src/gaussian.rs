use std::io::BufRead;

use bytemuck::Zeroable;
use glam::*;

use crate::{Error, PlyGaussianPod};

/// A scene containing Gaussians.
#[derive(Debug)]
pub struct Gaussians {
    /// The Gaussians.
    gaussians: Vec<Gaussian>,
}

impl Gaussians {
    /// Read a splat PLY file.
    pub fn read_ply(reader: &mut impl BufRead) -> Result<Self, Error> {
        let count = Self::read_ply_header(reader)?;
        let gaussians = Self::read_ply_gaussians(reader, count)?;

        Ok(Self { gaussians })
    }

    /// Read a splat PLY header.
    fn read_ply_header(reader: &mut impl BufRead) -> Result<usize, Error> {
        let mut count = 0;
        loop {
            let mut line = String::new();
            reader.read_line(&mut line)?;

            if line.starts_with("end_header") {
                break Ok(count);
            } else if line.starts_with("element vertex") {
                count = line
                    .split_whitespace()
                    .nth(2)
                    .ok_or(Error::PlyVertexCountNotFound)?
                    .parse()?;
            }
        }
    }

    /// Read the splat PLY Gaussians into [`pod::SplatGaussian`].
    fn read_ply_gaussians(reader: &mut impl BufRead, count: usize) -> Result<Vec<Gaussian>, Error> {
        std::iter::repeat_n(PlyGaussianPod::zeroed(), count)
            .map(|mut gaussian| {
                reader.read_exact(bytemuck::bytes_of_mut(&mut gaussian))?;
                Ok(gaussian.into())
            })
            .collect()
    }

    /// Get the Gaussians.
    pub fn gaussians(&self) -> &[Gaussian] {
        &self.gaussians
    }

    /// Sort the Gaussians by distance to the camera.
    pub fn sort(&mut self, camera_pos: Vec3) {
        self.gaussians.sort_by(|a, b| {
            let a_dist = (a.pos - camera_pos).length_squared();
            let b_dist = (b.pos - camera_pos).length_squared();
            a_dist.partial_cmp(&b_dist).unwrap()
        });
    }
}

/// The Gaussian.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gaussian {
    pub rotation: Quat,
    pub pos: Vec3,
    pub color: U8Vec4,
    pub scale: Vec3,
}

impl Gaussian {
    /// Convert from PLY Gaussian to Gaussian.
    pub fn from_ply(ply: &PlyGaussianPod) -> Self {
        let adjust = Quat::from_axis_angle(Vec3::Z, 180f32.to_radians());

        // Position
        let pos = Mat3::from_quat(adjust) * Vec3::from_array(ply.pos);

        // Rotation
        let rotation = (adjust
            * Quat::from_xyzw(
                ply.rotation[1],
                ply.rotation[2],
                ply.rotation[3],
                ply.rotation[0],
            ))
        .normalize();

        // Scale
        let scale = Vec3::from_array(ply.scale).exp();

        // Color
        const SH: f32 = 0.2820948;
        let color = ((Vec3::splat(0.5) + Vec3::from_slice(&ply.color[..3]) * SH) * 255.0)
            .extend((1.0 / (1.0 + (-ply.alpha).exp())) * 255.0)
            .clamp(Vec4::splat(0.0), Vec4::splat(255.0))
            .as_u8vec4();

        Self {
            rotation,
            pos,
            color,
            scale,
        }
    }
}

impl From<PlyGaussianPod> for Gaussian {
    fn from(ply: PlyGaussianPod) -> Self {
        Self::from_ply(&ply)
    }
}

impl From<&PlyGaussianPod> for Gaussian {
    fn from(ply: &PlyGaussianPod) -> Self {
        Self::from_ply(ply)
    }
}
