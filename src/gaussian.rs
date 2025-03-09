use std::io::{BufRead, Write};

use bytemuck::Zeroable;
use glam::*;

use crate::{Error, GaussianEditFlag, GaussianEditPod, PlyGaussianPod};

/// A scene containing Gaussians.
#[derive(Debug, Clone)]
pub struct Gaussians {
    /// The Gaussians.
    pub gaussians: Vec<Gaussian>,
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
        let mut line = String::new();
        reader.read_line(&mut line)?;
        if line.as_str().trim().to_lowercase() != "ply" {
            return Err(Error::NotPly);
        }

        let mut count = 0;
        loop {
            let mut line = String::new();
            if reader.read_line(&mut line)? == 0 {
                return Err(Error::PlyHeaderNotFound);
            }

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

    /// Read the splat PLY Gaussians into [`PlyGaussianPod`].
    fn read_ply_gaussians(reader: &mut impl BufRead, count: usize) -> Result<Vec<Gaussian>, Error> {
        let mut gaussians = Vec::with_capacity(count);
        for _ in 0..count {
            let mut gaussian = PlyGaussianPod::zeroed();
            reader.read_exact(bytemuck::bytes_of_mut(&mut gaussian))?;
            gaussians.push(gaussian.into());
        }

        Ok(gaussians)
    }

    /// Write the Gaussians to a PLY file.
    pub fn write_ply<'a>(
        &self,
        writer: &mut impl Write,
        edits: Option<impl IntoIterator<Item = &'a GaussianEditPod>>,
    ) -> Result<(), Error> {
        writeln!(writer, "ply")?;
        writeln!(writer, "format binary_little_endian 1.0")?;
        writeln!(writer, "element vertex 597903")?;
        writeln!(writer, "property float x")?;
        writeln!(writer, "property float y")?;
        writeln!(writer, "property float z")?;
        writeln!(writer, "property float nx")?;
        writeln!(writer, "property float ny")?;
        writeln!(writer, "property float nz")?;
        writeln!(writer, "property float f_dc_0")?;
        writeln!(writer, "property float f_dc_1")?;
        writeln!(writer, "property float f_dc_2")?;
        writeln!(writer, "property float f_rest_0")?;
        writeln!(writer, "property float f_rest_1")?;
        writeln!(writer, "property float f_rest_2")?;
        writeln!(writer, "property float f_rest_3")?;
        writeln!(writer, "property float f_rest_4")?;
        writeln!(writer, "property float f_rest_5")?;
        writeln!(writer, "property float f_rest_6")?;
        writeln!(writer, "property float f_rest_7")?;
        writeln!(writer, "property float f_rest_8")?;
        writeln!(writer, "property float f_rest_9")?;
        writeln!(writer, "property float f_rest_10")?;
        writeln!(writer, "property float f_rest_11")?;
        writeln!(writer, "property float f_rest_12")?;
        writeln!(writer, "property float f_rest_13")?;
        writeln!(writer, "property float f_rest_14")?;
        writeln!(writer, "property float f_rest_15")?;
        writeln!(writer, "property float f_rest_16")?;
        writeln!(writer, "property float f_rest_17")?;
        writeln!(writer, "property float f_rest_18")?;
        writeln!(writer, "property float f_rest_19")?;
        writeln!(writer, "property float f_rest_20")?;
        writeln!(writer, "property float f_rest_21")?;
        writeln!(writer, "property float f_rest_22")?;
        writeln!(writer, "property float f_rest_23")?;
        writeln!(writer, "property float f_rest_24")?;
        writeln!(writer, "property float f_rest_25")?;
        writeln!(writer, "property float f_rest_26")?;
        writeln!(writer, "property float f_rest_27")?;
        writeln!(writer, "property float f_rest_28")?;
        writeln!(writer, "property float f_rest_29")?;
        writeln!(writer, "property float f_rest_30")?;
        writeln!(writer, "property float f_rest_31")?;
        writeln!(writer, "property float f_rest_32")?;
        writeln!(writer, "property float f_rest_33")?;
        writeln!(writer, "property float f_rest_34")?;
        writeln!(writer, "property float f_rest_35")?;
        writeln!(writer, "property float f_rest_36")?;
        writeln!(writer, "property float f_rest_37")?;
        writeln!(writer, "property float f_rest_38")?;
        writeln!(writer, "property float f_rest_39")?;
        writeln!(writer, "property float f_rest_40")?;
        writeln!(writer, "property float f_rest_41")?;
        writeln!(writer, "property float f_rest_42")?;
        writeln!(writer, "property float f_rest_43")?;
        writeln!(writer, "property float f_rest_44")?;
        writeln!(writer, "property float opacity")?;
        writeln!(writer, "property float scale_0")?;
        writeln!(writer, "property float scale_1")?;
        writeln!(writer, "property float scale_2")?;
        writeln!(writer, "property float rot_0")?;
        writeln!(writer, "property float rot_1")?;
        writeln!(writer, "property float rot_2")?;
        writeln!(writer, "property float rot_3")?;
        writeln!(writer, "end_header")?;

        match edits {
            Some(edits) => self
                .gaussians
                .iter()
                .zip(edits.into_iter())
                .filter_map(|(gaussian, edit)| gaussian.with_edit(edit))
                .map(|gaussian| gaussian.to_ply())
                .try_for_each(|gaussian| writer.write_all(bytemuck::bytes_of(&gaussian)))?,
            None => self
                .gaussians
                .iter()
                .map(|gaussian| gaussian.to_ply())
                .try_for_each(|gaussian| writer.write_all(bytemuck::bytes_of(&gaussian)))?,
        }

        Ok(())
    }
}

/// The Gaussian.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gaussian {
    pub rotation: Quat,
    pub pos: Vec3,
    pub color: U8Vec4,
    pub sh: [Vec3; 15],
    pub scale: Vec3,
}

impl Gaussian {
    /// Convert from PLY Gaussian to Gaussian.
    pub fn from_ply(ply: &PlyGaussianPod) -> Self {
        // Position
        let pos = Vec3::from_array(ply.pos);

        // Rotation
        let rotation = Quat::from_xyzw(
            ply.rotation[1],
            ply.rotation[2],
            ply.rotation[3],
            ply.rotation[0],
        )
        .normalize();

        // Scale
        let scale = Vec3::from_array(ply.scale).exp();

        // Color
        const SH_C0: f32 = 0.2820948;
        let color = ((Vec3::splat(0.5) + Vec3::from_array(ply.color) * SH_C0) * 255.0)
            .extend((1.0 / (1.0 + (-ply.alpha).exp())) * 255.0)
            .clamp(Vec4::splat(0.0), Vec4::splat(255.0))
            .as_u8vec4();

        // Spherical harmonics
        let sh = std::array::from_fn(|i| Vec3::new(ply.sh[i], ply.sh[i + 15], ply.sh[i + 30]));

        Self {
            rotation,
            pos,
            color,
            sh,
            scale,
        }
    }

    /// Convert to PLY Gaussian.
    pub fn to_ply(&self) -> PlyGaussianPod {
        // Position
        let pos = self.pos.to_array();

        // Rotation
        let rotation = [
            self.rotation.w,
            self.rotation.x,
            self.rotation.y,
            self.rotation.z,
        ];

        // Scale
        let scale = self.scale.map(|x| x.ln()).to_array();

        // Color
        const SH_C0: f32 = 0.2820948;
        let rgba = self.color.as_vec4() / 255.0;
        let color = ((rgba.xyz() / SH_C0) - Vec3::splat(0.5 / SH_C0)).to_array();

        // Alpha
        let alpha = -(1.0 / rgba.w - 1.0).ln();

        // Spherical harmonics
        let mut sh = [0.0; 3 * 15];
        for i in 0..15 {
            sh[i] = self.sh[i].x;
            sh[i + 15] = self.sh[i].y;
            sh[i + 30] = self.sh[i].z;
        }

        let normal = [0.0, 0.0, 1.0];

        PlyGaussianPod {
            pos,
            normal,
            color,
            sh,
            alpha,
            scale,
            rotation,
        }
    }

    /// With a [`GaussianEditPod`] applied to the PLY Gaussian.
    pub fn with_edit(mut self, edit: &GaussianEditPod) -> Option<Self> {
        // None
        if !edit.flag().contains(GaussianEditFlag::ENABLED) {
            return Some(self);
        }

        // Hide
        if edit.flag().contains(GaussianEditFlag::HIDDEN) {
            return None;
        }

        let color = self.color.as_vec4() / 255.0;
        let mut rgb = color.xyz();

        // RGB
        if edit.flag().contains(GaussianEditFlag::OVERRIDE_COLOR) {
            rgb = edit.rgb();
        } else {
            fn rgb_to_hsv(c: Vec3) -> Vec3 {
                const K: Vec4 = Vec4::new(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
                let p = if c.z < c.y {
                    vec4(c.y, c.z, K.x, K.y)
                } else {
                    vec4(c.z, c.y, K.w, K.z)
                };
                let q = if p.x < c.x {
                    vec4(c.x, p.y, p.z, p.x)
                } else {
                    vec4(p.x, p.y, p.w, c.x)
                };

                let d = q.x - q.w.min(q.y);
                const E: f32 = 1.0e-10;
                vec3(
                    (q.z + (q.w - q.y) / (6.0 * d + E)).abs(),
                    d / (q.x + E),
                    q.x,
                )
            }

            fn hsv_to_rgb(c: Vec3) -> Vec3 {
                let k = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
                let p = ((c.xxx() + k.xyz()).fract() * 6.0 - k.www()).abs();
                c.z * k
                    .xxx()
                    .lerp((p - k.xxx()).clamp(Vec3::ZERO, Vec3::ONE), c.y)
            }

            let hsv = rgb_to_hsv(rgb);
            let hsv_edited = vec3(
                (hsv.x + edit.hue()).fract(),
                hsv.y * edit.saturation(),
                hsv.z * edit.brightness(),
            )
            .clamp(Vec3::ZERO, Vec3::ONE);
            rgb = hsv_to_rgb(hsv_edited);
        }

        // Contrast
        const CONTRAST_CONST: f32 = 259.0 / 255.0;
        let contrast = edit.contrast();
        let contrast_factor = CONTRAST_CONST * (contrast + 1.0) / (CONTRAST_CONST - contrast);
        rgb = (rgb - 0.5) * contrast_factor + 0.5;

        // Exposure
        rgb *= 2.0f32.powf(edit.exposure());

        // Gamma
        rgb = rgb.powf(edit.gamma());

        // Alpha
        let a = color.w * edit.alpha();

        self.color = (rgb.extend(a) * 255.0).as_u8vec4();

        Some(self)
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
