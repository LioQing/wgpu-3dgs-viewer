use bytemuck::Zeroable;
use glam::*;

use half::f16;
use wgpu::util::DeviceExt;

use crate::{CameraTrait, Gaussian, wgpu_sort};

/// The Gaussians storage buffer.
#[derive(Debug)]
pub struct GaussiansBuffer<G: GaussianPod>(wgpu::Buffer, std::marker::PhantomData<G>);

impl<G: GaussianPod> GaussiansBuffer<G> {
    /// Create a new Gaussians buffer.
    pub fn new(device: &wgpu::Device, gaussians: &[Gaussian]) -> Self {
        Self::new_with_pods(
            device,
            gaussians
                .iter()
                .map(G::from_gaussian)
                .collect::<Vec<_>>()
                .as_slice(),
        )
    }

    /// Create a new Gaussians buffer with [`GaussianPod`].
    pub fn new_with_pods(device: &wgpu::Device, gaussians: &[G]) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Gaussians Buffer"),
            contents: bytemuck::cast_slice(gaussians),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        Self(buffer, std::marker::PhantomData)
    }

    /// Create a new Gaussians buffer with the specified size.
    pub fn new_empty(device: &wgpu::Device, len: usize) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gaussians Buffer"),
            size: (len * std::mem::size_of::<G>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self(buffer, std::marker::PhantomData)
    }

    /// Get the buffer.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.0
    }

    /// Get the number of Gaussians.
    pub fn len(&self) -> usize {
        self.0.size() as usize / std::mem::size_of::<G>()
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

        self.update_with_pod(
            queue,
            gaussians
                .iter()
                .map(G::from_gaussian)
                .collect::<Vec<_>>()
                .as_slice(),
        );
    }

    /// Update the buffer with [`GaussianPod`].
    pub fn update_with_pod(&self, queue: &wgpu::Queue, pods: &[G]) {
        if pods.len() != self.len() {
            log::error!(
                "Gaussians count mismatch, buffer has {}, but {} were provided",
                self.len(),
                pods.len()
            );
            return;
        }

        queue.write_buffer(&self.0, 0, bytemuck::cast_slice(pods));
    }

    /// Update a range of the buffer.
    pub fn update_range(&self, queue: &wgpu::Queue, start: usize, gaussians: &[Gaussian]) {
        if start + gaussians.len() > self.len() {
            log::error!(
                "Gaussians count mismatch, buffer has {}, but {} were provided starting at {}",
                self.len(),
                gaussians.len(),
                start
            );
            return;
        }

        self.update_range_with_pod(
            queue,
            start,
            gaussians
                .iter()
                .map(G::from_gaussian)
                .collect::<Vec<_>>()
                .as_slice(),
        );
    }

    /// Update a range of the buffer with [`GaussianPod`].
    pub fn update_range_with_pod(&self, queue: &wgpu::Queue, start: usize, pods: &[G]) {
        if start + pods.len() > self.len() {
            log::error!(
                "Gaussians count mismatch, buffer has {}, but {} were provided starting at {}",
                self.len(),
                pods.len(),
                start
            );
            return;
        }

        queue.write_buffer(
            &self.0,
            (start * std::mem::size_of::<G>()) as wgpu::BufferAddress,
            bytemuck::cast_slice(pods),
        );
    }
}

/// The spherical harmonics configuration of Gaussian.
pub trait GaussianShConfig {
    /// The feature name of the configuration.
    ///
    /// Must match the feature name in the shader.
    const FEATURE: &'static str;

    /// The [`GaussianPod`] field type.
    type Field: bytemuck::Pod + bytemuck::Zeroable;

    /// Create from [`Gaussian.sh`].
    fn from_sh(sh: &[Vec3; 15]) -> Self::Field;
}

/// The single precision SH configuration of Gaussian.
pub struct GaussianShSingleConfig;

impl GaussianShConfig for GaussianShSingleConfig {
    const FEATURE: &'static str = "sh_single";

    type Field = [Vec3; 15];

    fn from_sh(sh: &[Vec3; 15]) -> Self::Field {
        *sh
    }
}

/// The half precision SH configuration of Gaussian.
pub struct GaussianShHalfConfig;

impl GaussianShConfig for GaussianShHalfConfig {
    const FEATURE: &'static str = "sh_half";

    type Field = [f16; 3 * 15 + 1];

    fn from_sh(sh: &[Vec3; 15]) -> Self::Field {
        sh.iter()
            .flat_map(|sh| sh.to_array())
            .map(f16::from_f32)
            .chain(std::iter::once(f16::from_f32(0.0)))
            .collect::<Vec<_>>()
            .try_into()
            .expect("SH half")
    }
}

/// The min max 8 bit normalized SH configuration of Gaussian.
pub struct GaussianShNorm8Config;

impl GaussianShConfig for GaussianShNorm8Config {
    const FEATURE: &'static str = "sh_norm8";

    type Field = [u8; 4 + (3 * 15 + 3)]; // ([f16; 2], [U8Vec4; (3 * 15 + 3) / 4])

    fn from_sh(sh: &[Vec3; 15]) -> Self::Field {
        let mut sh_pod = [0; 4 + (3 * 15 + 3)];

        let sh = sh.iter().flat_map(|sh| sh.to_array()).collect::<Vec<_>>();
        let (min, max) = sh.iter().fold((f32::MAX, f32::MIN), |(min, max), &x| {
            (min.min(x), max.max(x))
        });

        sh_pod[0..2].copy_from_slice(&f16::from_f32(min).to_ne_bytes());
        sh_pod[2..4].copy_from_slice(&f16::from_f32(max).to_ne_bytes());
        sh_pod[4..].copy_from_slice(
            &sh.iter()
                .map(|&x| ((x - min) / (max - min) * 255.0).round() as u8)
                .chain(std::iter::repeat_n(0, 3))
                .collect::<Vec<_>>(),
        );

        sh_pod
    }
}

/// The none SH configuration of Gaussian.
pub struct GaussianShNoneConfig;

impl GaussianShConfig for GaussianShNoneConfig {
    const FEATURE: &'static str = "sh_none";

    type Field = ();

    fn from_sh(_sh: &[Vec3; 15]) -> Self::Field {}
}

/// The covariance 3D configuration of Gaussian.
pub trait GaussianCov3dConfig {
    /// The name of the configuration.
    ///
    /// Must match the name in the shader.
    const FEATURE: &'static str;

    /// The [`GaussianPod`] field type.
    type Field: bytemuck::Pod + bytemuck::Zeroable;

    /// Create from a single precision cov3d.
    fn from_cov3d(cov3d: [f32; 6]) -> Self::Field;
}

/// The single precision covariance 3D configuration of Gaussian.
pub struct GaussianCov3dSingleConfig;

impl GaussianCov3dConfig for GaussianCov3dSingleConfig {
    const FEATURE: &'static str = "cov3d_single";

    type Field = [f32; 6];

    fn from_cov3d(cov3d: [f32; 6]) -> Self::Field {
        cov3d
    }
}

/// The half precision covariance 3D configuration of Gaussian.
pub struct GaussianCov3dHalfConfig;

impl GaussianCov3dConfig for GaussianCov3dHalfConfig {
    const FEATURE: &'static str = "cov3d_half";

    type Field = [f16; 6];

    fn from_cov3d(cov3d: [f32; 6]) -> Self::Field {
        cov3d.map(f16::from_f32)
    }
}

/// The Gaussian POD trait.
pub trait GaussianPod: for<'a> From<&'a Gaussian> + bytemuck::NoUninit {
    /// The SH configuration.
    type ShConfig: GaussianShConfig;

    /// The covariance 3D configuration.
    type Cov3dConfig: GaussianCov3dConfig;

    /// Create a new Gaussian POD from the Gaussian.
    fn from_gaussian(gaussian: &Gaussian) -> Self {
        Self::from(gaussian)
    }

    /// Create the features for [`Wesl`](wesl::Wesl) compilation.
    fn features() -> [(&'static str, bool); 6] {
        [
            GaussianShSingleConfig::FEATURE,
            GaussianShHalfConfig::FEATURE,
            GaussianShNorm8Config::FEATURE,
            GaussianShNoneConfig::FEATURE,
            GaussianCov3dSingleConfig::FEATURE,
            GaussianCov3dHalfConfig::FEATURE,
        ]
        .map(|name| {
            (
                name,
                name == Self::ShConfig::FEATURE || name == Self::Cov3dConfig::FEATURE,
            )
        })
    }
}

/// Macro to create the POD representation of Gaussian given the configurations.
macro_rules! gaussian_pod {
    (sh = $sh:ident, cov3d = $cov3d:ident, padding_size = $padding:expr) => {
        paste::paste! {
            /// The POD representation of Gaussian.
            #[repr(C)]
            #[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
            pub struct [< GaussianPodWith Sh $sh Cov3d $cov3d Configs >] {
                pub pos: Vec3,
                pub color: U8Vec4,
                pub sh: <[< GaussianSh $sh Config >] as GaussianShConfig>::Field,
                pub cov3d: <[< GaussianCov3d $cov3d Config >] as GaussianCov3dConfig>::Field,
                _padding: [f32; $padding],
            }

            impl From<&Gaussian> for [< GaussianPodWith Sh $sh Cov3d $cov3d Configs >] {
                fn from(gaussian: &Gaussian) -> Self {
                    // Covariance
                    let r = Mat3::from_quat(gaussian.rotation);
                    let s = Mat3::from_diagonal(gaussian.scale);
                    let m = r * s;
                    let sigma = m * m.transpose();
                    let cov3d = [< GaussianCov3d $cov3d Config >]::from_cov3d([
                        sigma.x_axis.x,
                        sigma.x_axis.y,
                        sigma.x_axis.z,
                        sigma.y_axis.y,
                        sigma.y_axis.z,
                        sigma.z_axis.z,
                    ]);

                    // Color
                    let color = gaussian.color;

                    // Spherical harmonics
                    let sh = [< GaussianSh $sh Config >]::from_sh(&gaussian.sh);

                    // Position
                    let pos = gaussian.pos;

                    Self {
                        pos,
                        color,
                        sh,
                        cov3d,
                        _padding: [0.0; $padding],
                    }
                }
            }

            impl GaussianPod for [< GaussianPodWith Sh $sh Cov3d $cov3d Configs >] {
                type ShConfig = [< GaussianSh $sh Config >];
                type Cov3dConfig = [< GaussianCov3d $cov3d Config >];
            }
        }
    };
}

gaussian_pod!(sh = Single, cov3d = Single, padding_size = 1);
gaussian_pod!(sh = Single, cov3d = Half, padding_size = 0);
gaussian_pod!(sh = Half, cov3d = Single, padding_size = 3);
gaussian_pod!(sh = Half, cov3d = Half, padding_size = 2);
gaussian_pod!(sh = Norm8, cov3d = Single, padding_size = 1);
gaussian_pod!(sh = Norm8, cov3d = Half, padding_size = 0);
gaussian_pod!(sh = None, cov3d = Single, padding_size = 2);
gaussian_pod!(sh = None, cov3d = Half, padding_size = 1);

/// The POD representation of Gaussian in PLY format.
///
/// Fields are stored as arrays because using glam types would add padding
/// according to C alignment rules.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PlyGaussianPod {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
    pub sh: [f32; 3 * 15],
    pub alpha: f32,
    pub scale: [f32; 3],
    pub rotation: [f32; 4],
}

impl PlyGaussianPod {
    pub fn set_value(&mut self, name: &str, value: f32) {
        macro_rules! set_prop {
            ($name:expr, $field:expr) => {
                $field = value
            };
        }

        match name {
            "x" => set_prop!("x", self.pos[0]),
            "y" => set_prop!("y", self.pos[1]),
            "z" => set_prop!("z", self.pos[2]),
            "nx" => set_prop!("nx", self.normal[0]),
            "ny" => set_prop!("ny", self.normal[1]),
            "nz" => set_prop!("nz", self.normal[2]),
            "f_dc_0" => set_prop!("f_dc_0", self.color[0]),
            "f_dc_1" => set_prop!("f_dc_1", self.color[1]),
            "f_dc_2" => set_prop!("f_dc_2", self.color[2]),
            "f_rest_0" => set_prop!("f_rest_0", self.sh[0]),
            "f_rest_1" => set_prop!("f_rest_1", self.sh[1]),
            "f_rest_2" => set_prop!("f_rest_2", self.sh[2]),
            "f_rest_3" => set_prop!("f_rest_3", self.sh[3]),
            "f_rest_4" => set_prop!("f_rest_4", self.sh[4]),
            "f_rest_5" => set_prop!("f_rest_5", self.sh[5]),
            "f_rest_6" => set_prop!("f_rest_6", self.sh[6]),
            "f_rest_7" => set_prop!("f_rest_7", self.sh[7]),
            "f_rest_8" => set_prop!("f_rest_8", self.sh[8]),
            "f_rest_9" => set_prop!("f_rest_9", self.sh[9]),
            "f_rest_10" => set_prop!("f_rest_10", self.sh[10]),
            "f_rest_11" => set_prop!("f_rest_11", self.sh[11]),
            "f_rest_12" => set_prop!("f_rest_12", self.sh[12]),
            "f_rest_13" => set_prop!("f_rest_13", self.sh[13]),
            "f_rest_14" => set_prop!("f_rest_14", self.sh[14]),
            "f_rest_15" => set_prop!("f_rest_15", self.sh[15]),
            "f_rest_16" => set_prop!("f_rest_16", self.sh[16]),
            "f_rest_17" => set_prop!("f_rest_17", self.sh[17]),
            "f_rest_18" => set_prop!("f_rest_18", self.sh[18]),
            "f_rest_19" => set_prop!("f_rest_19", self.sh[19]),
            "f_rest_20" => set_prop!("f_rest_20", self.sh[20]),
            "f_rest_21" => set_prop!("f_rest_21", self.sh[21]),
            "f_rest_22" => set_prop!("f_rest_22", self.sh[22]),
            "f_rest_23" => set_prop!("f_rest_23", self.sh[23]),
            "f_rest_24" => set_prop!("f_rest_24", self.sh[24]),
            "f_rest_25" => set_prop!("f_rest_25", self.sh[25]),
            "f_rest_26" => set_prop!("f_rest_26", self.sh[26]),
            "f_rest_27" => set_prop!("f_rest_27", self.sh[27]),
            "f_rest_28" => set_prop!("f_rest_28", self.sh[28]),
            "f_rest_29" => set_prop!("f_rest_29", self.sh[29]),
            "f_rest_30" => set_prop!("f_rest_30", self.sh[30]),
            "f_rest_31" => set_prop!("f_rest_31", self.sh[31]),
            "f_rest_32" => set_prop!("f_rest_32", self.sh[32]),
            "f_rest_33" => set_prop!("f_rest_33", self.sh[33]),
            "f_rest_34" => set_prop!("f_rest_34", self.sh[34]),
            "f_rest_35" => set_prop!("f_rest_35", self.sh[35]),
            "f_rest_36" => set_prop!("f_rest_36", self.sh[36]),
            "f_rest_37" => set_prop!("f_rest_37", self.sh[37]),
            "f_rest_38" => set_prop!("f_rest_38", self.sh[38]),
            "f_rest_39" => set_prop!("f_rest_39", self.sh[39]),
            "f_rest_40" => set_prop!("f_rest_40", self.sh[40]),
            "f_rest_41" => set_prop!("f_rest_41", self.sh[41]),
            "f_rest_42" => set_prop!("f_rest_42", self.sh[42]),
            "f_rest_43" => set_prop!("f_rest_43", self.sh[43]),
            "f_rest_44" => set_prop!("f_rest_44", self.sh[44]),
            "opacity" => set_prop!("opacity", self.alpha),
            "scale_0" => set_prop!("scale_0", self.scale[0]),
            "scale_1" => set_prop!("scale_1", self.scale[1]),
            "scale_2" => set_prop!("scale_2", self.scale[2]),
            "rot_0" => set_prop!("rot_0", self.rotation[0]),
            "rot_1" => set_prop!("rot_1", self.rotation[1]),
            "rot_2" => set_prop!("rot_2", self.rotation[2]),
            "rot_3" => set_prop!("rot_3", self.rotation[3]),
            _ => {
                log::warn!("Unknown property: {name}");
            }
        }
    }
}

impl ply_rs::ply::PropertyAccess for PlyGaussianPod {
    fn new() -> Self {
        PlyGaussianPod::zeroed()
    }

    fn set_property(&mut self, property_name: String, property: ply_rs::ply::Property) {
        let ply_rs::ply::Property::Float(value) = property else {
            panic!("Expected float property");
        };

        self.set_value(&property_name, value);
    }
}

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
    pub fn new(camera: &impl CameraTrait, size: UVec2) -> Self {
        Self {
            view: camera.view(),
            proj: camera.projection(size.x as f32 / size.y as f32),
            size: size.as_vec2(),
            _padding_0: [0; 2],
        }
    }
}

/// The model transformation buffer.
#[derive(Debug, Clone)]
pub struct ModelTransformBuffer(wgpu::Buffer);

impl ModelTransformBuffer {
    /// Create a new model transformation buffer.
    pub fn new(device: &wgpu::Device) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Model transform Buffer"),
            contents: bytemuck::bytes_of(&ModelTransformPod::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Self(buffer)
    }

    /// Update the model transformation buffer.
    pub fn update(&self, queue: &wgpu::Queue, pos: Vec3, quat: Quat, scale: Vec3) {
        self.update_with_pod(queue, &ModelTransformPod::new(pos, quat, scale));
    }

    /// Update the model transformation buffer with [`ModelTransformPod`].
    pub fn update_with_pod(&self, queue: &wgpu::Queue, pod: &ModelTransformPod) {
        queue.write_buffer(&self.0, 0, bytemuck::bytes_of(pod));
    }

    /// Get the buffer.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.0
    }
}

/// The POD representation of a model transformation.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ModelTransformPod {
    pub pos: Vec3,
    _padding_0: f32,
    pub quat: Quat,
    pub scale: Vec3,
    _padding_1: f32,
}

impl ModelTransformPod {
    /// Create a new model transformation.
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

impl Default for ModelTransformPod {
    fn default() -> Self {
        Self::new(Vec3::ZERO, Quat::IDENTITY, Vec3::ONE)
    }
}

/// The Gaussian display modes.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GaussianDisplayMode {
    Splat = 0,
    Ellipse = 1,
    Point = 2,
}

/// The Gaussian spherical harmonics degrees.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaussianShDegree(u8);

impl GaussianShDegree {
    /// Create a new Gaussian SH degree.
    ///
    /// Returns [`None`] if the degree is not in the range of \[0, 3\].
    pub const fn new(sh_deg: u8) -> Option<Self> {
        match sh_deg {
            0..=3 => Some(Self(sh_deg)),
            _ => None,
        }
    }

    /// Create a new Gaussian SH degree without checking.
    pub const fn new_unchecked(sh_deg: u8) -> Self {
        Self(sh_deg)
    }

    /// Get the degree.
    pub const fn degree(&self) -> u8 {
        self.0
    }
}

/// The Gaussian transform buffer.
#[derive(Debug, Clone)]
pub struct GaussianTransformBuffer(wgpu::Buffer);

impl GaussianTransformBuffer {
    /// Create a new Gaussian transform buffer.
    pub fn new(device: &wgpu::Device) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Gaussian transform Buffer"),
            contents: bytemuck::bytes_of(&GaussianTransformPod::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Self(buffer)
    }

    /// Update the Gaussian transformation buffer.
    pub fn update(
        &self,
        queue: &wgpu::Queue,
        size: f32,
        display_mode: GaussianDisplayMode,
        sh_deg: GaussianShDegree,
        no_sh0: bool,
    ) {
        self.update_with_pod(
            queue,
            &GaussianTransformPod::new(size, display_mode, sh_deg, no_sh0),
        );
    }

    /// Update the Gaussian transformation buffer with [`GaussianTransformPod`].
    pub fn update_with_pod(&self, queue: &wgpu::Queue, transform: &GaussianTransformPod) {
        queue.write_buffer(&self.0, 0, bytemuck::bytes_of(transform));
    }

    /// Get the buffer.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.0
    }
}

/// The POD representation of a Gaussian transformation.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GaussianTransformPod {
    pub size: f32,

    /// \[display_mode, sh_deg, no_sh0, padding\]
    pub flags: U8Vec4,
}

impl GaussianTransformPod {
    /// Create a new Gaussian transformation.
    pub const fn new(
        size: f32,
        display_mode: GaussianDisplayMode,
        sh_deg: GaussianShDegree,
        no_sh0: bool,
    ) -> Self {
        let display_mode = display_mode as u8;
        let sh_deg = sh_deg.0;
        let no_sh0 = no_sh0 as u8;

        Self {
            size,
            flags: u8vec4(display_mode, sh_deg, no_sh0, 0),
        }
    }
}

impl Default for GaussianTransformPod {
    fn default() -> Self {
        Self::new(
            1.0,
            GaussianDisplayMode::Splat,
            GaussianShDegree::new_unchecked(3),
            false,
        )
    }
}

/// The Gaussians depth storage buffer.
#[derive(Debug, Clone)]
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
