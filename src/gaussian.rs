use std::io::{BufRead, Write};

use bytemuck::Zeroable;
use glam::*;

use crate::{Error, GaussianEditFlag, GaussianEditPod, PlyGaussianPod};

/// Types of PLY files.
#[derive(Debug, Clone)]
enum PlyType {
    /// The Inria PLY format.
    ///
    /// This can be directly loaded into [`PlyGaussianPod`].
    Inria(usize),

    /// Custom PLY format.
    Custom(ply_rs::ply::Header),
}

/// A scene containing Gaussians.
#[derive(Debug, Clone)]
pub struct Gaussians {
    /// The Gaussians.
    pub gaussians: Vec<Gaussian>,
}

impl Gaussians {
    /// Read a splat PLY file.
    pub fn read_ply(reader: &mut impl BufRead) -> Result<Self, Error> {
        let ply_type = Self::read_ply_header(reader)?;
        let gaussians = Self::read_ply_gaussians(reader, ply_type)?;

        Ok(Self { gaussians })
    }

    /// Read a splat PLY header.
    fn read_ply_header(reader: &mut impl BufRead) -> Result<PlyType, Error> {
        let parser = ply_rs::parser::Parser::<ply_rs::ply::DefaultElement>::new();
        let header = parser.read_header(reader)?;
        let vertex = header
            .elements
            .get("vertex")
            .ok_or(Error::PlyVertexNotFound)?;

        let ply_type = match vertex
            .properties
            .iter()
            .map(|(name, _)| name.as_str())
            .zip(PLY_PROPERTIES.iter())
            .all(|(a, b)| a == *b)
            && header.encoding == ply_rs::ply::Encoding::BinaryLittleEndian
        {
            true => PlyType::Inria(vertex.count),
            false => PlyType::Custom(header),
        };

        Ok(ply_type)
    }

    /// Read the splat PLY Gaussians into [`PlyGaussianPod`].
    fn read_ply_gaussians(
        reader: &mut impl BufRead,
        ply_type: PlyType,
    ) -> Result<Vec<Gaussian>, Error> {
        match ply_type {
            PlyType::Inria(count) => {
                log::info!("Reading Inria PLY format with {} Gaussians", count);

                let mut gaussians = Vec::with_capacity(count);
                for _ in 0..count {
                    let mut gaussian = PlyGaussianPod::zeroed();
                    reader.read_exact(bytemuck::bytes_of_mut(&mut gaussian))?;
                    gaussians.push(gaussian.into());
                }

                Ok(gaussians)
            }
            PlyType::Custom(header) => {
                log::info!("Reading custom PLY format");

                let vertex = header
                    .elements
                    .get("vertex")
                    .ok_or(Error::PlyVertexNotFound)?;

                let parser = ply_rs::parser::Parser::<PlyGaussianPod>::new();
                let mut gaussians = Vec::with_capacity(vertex.count);
                for _ in 0..vertex.count {
                    let gaussian = match header.encoding {
                        ply_rs::ply::Encoding::Ascii => {
                            let mut line = String::new();
                            reader.read_line(&mut line)?;

                            let mut gaussian = PlyGaussianPod::zeroed();
                            line.split(' ')
                                .map(|s| s.parse::<f32>())
                                .zip(vertex.properties.keys())
                                .try_for_each(|(value, name)| match value {
                                    Ok(value) => {
                                        gaussian.set_value(name, value);
                                        Ok(())
                                    }
                                    Err(_) => Err(Error::PlyVertexPropertyNotFound(name.clone())),
                                })?;

                            gaussian
                        }
                        ply_rs::ply::Encoding::BinaryLittleEndian => {
                            parser.read_little_endian_element(reader, vertex)?
                        }
                        ply_rs::ply::Encoding::BinaryBigEndian => {
                            parser.read_big_endian_element(reader, vertex)?
                        }
                    };

                    gaussians.push(gaussian.into());
                }

                Ok(gaussians)
            }
        }
    }

    /// Write the Gaussians to a PLY file.
    pub fn write_ply<'a>(
        &self,
        writer: &mut impl Write,
        edits: Option<impl IntoIterator<Item = &'a GaussianEditPod>>,
    ) -> Result<(), Error> {
        writeln!(writer, "ply")?;
        writeln!(writer, "format binary_little_endian 1.0")?;
        writeln!(writer, "element vertex {}", self.gaussians.len())?;
        for property in PLY_PROPERTIES {
            writeln!(writer, "property float {property}")?;
        }
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

const PLY_PROPERTIES: &[&str] = &[
    "x",
    "y",
    "z",
    "nx",
    "ny",
    "nz",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
    "f_rest_0",
    "f_rest_1",
    "f_rest_2",
    "f_rest_3",
    "f_rest_4",
    "f_rest_5",
    "f_rest_6",
    "f_rest_7",
    "f_rest_8",
    "f_rest_9",
    "f_rest_10",
    "f_rest_11",
    "f_rest_12",
    "f_rest_13",
    "f_rest_14",
    "f_rest_15",
    "f_rest_16",
    "f_rest_17",
    "f_rest_18",
    "f_rest_19",
    "f_rest_20",
    "f_rest_21",
    "f_rest_22",
    "f_rest_23",
    "f_rest_24",
    "f_rest_25",
    "f_rest_26",
    "f_rest_27",
    "f_rest_28",
    "f_rest_29",
    "f_rest_30",
    "f_rest_31",
    "f_rest_32",
    "f_rest_33",
    "f_rest_34",
    "f_rest_35",
    "f_rest_36",
    "f_rest_37",
    "f_rest_38",
    "f_rest_39",
    "f_rest_40",
    "f_rest_41",
    "f_rest_42",
    "f_rest_43",
    "f_rest_44",
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
];
