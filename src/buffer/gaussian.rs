use glam::*;

use half::f16;
use wgpu::util::DeviceExt;

use crate::{CameraTrait, Error, Gaussian, wgpu_sort};

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
}

/// The spherical harmonics configuration of Gaussian.
pub trait GaussianShConfig {
    /// The name of the configuration.
    ///
    /// Must match the name in the shader.
    const NAME: &'static str;

    /// The WGSL shader.
    const WGSL: &'static str = include_str!("../shader/gaussian_configs.wgsl");

    /// The [`GaussianPod`] field type.
    type Field: bytemuck::Pod + bytemuck::Zeroable;

    /// The spherical harmonics field definition.
    fn sh_field() -> &'static str {
        Self::WGSL
            .split(format!("sh field - {}", Self::NAME).as_str())
            .nth(1)
            .expect("SH field")
            .trim_matches('\n')
    }

    /// The spherical harmonics unpack definition.
    fn sh_unpack() -> &'static str {
        Self::WGSL
            .split(format!("sh unpack - {}", Self::NAME).as_str())
            .nth(1)
            .expect("SH unpack")
            .trim_matches('\n')
    }

    /// Create from [`Gaussian.sh`].
    fn from_sh(sh: &[Vec3; 15]) -> Self::Field;
}

/// The single precision SH configuration of Gaussian.
pub struct GaussianShSingleConfig;

impl GaussianShConfig for GaussianShSingleConfig {
    const NAME: &'static str = "single";

    type Field = [Vec3; 15];

    fn from_sh(sh: &[Vec3; 15]) -> Self::Field {
        *sh
    }
}

/// The half precision SH configuration of Gaussian.
pub struct GaussianShHalfConfig;

impl GaussianShConfig for GaussianShHalfConfig {
    const NAME: &'static str = "half";

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
    const NAME: &'static str = "norm 8";

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
    const NAME: &'static str = "none";

    type Field = ();

    fn from_sh(_sh: &[Vec3; 15]) -> Self::Field {}
}

/// The covariance 3D configuration of Gaussian.
pub trait GaussianCov3dConfig {
    /// The name of the configuration.
    ///
    /// Must match the name in the shader.
    const NAME: &'static str;

    /// The WGSL shader.
    const WGSL: &'static str = include_str!("../shader/gaussian_configs.wgsl");

    /// The [`GaussianPod`] field type.
    type Field: bytemuck::Pod + bytemuck::Zeroable;

    /// The covariance 3D field definition.
    fn cov3d_field() -> &'static str {
        Self::WGSL
            .split(format!("cov3d field - {}", Self::NAME).as_str())
            .nth(1)
            .expect("Cov3d field")
            .trim_matches('\n')
    }

    /// The covariance 3D unpack definition.
    fn cov3d_unpack() -> &'static str {
        Self::WGSL
            .split(format!("cov3d unpack - {}", Self::NAME).as_str())
            .nth(1)
            .expect("Cov3d unpack")
            .trim_matches('\n')
    }

    /// Create from a single precision cov3d.
    fn from_cov3d(cov3d: [f32; 6]) -> Self::Field;
}

/// The single precision covariance 3D configuration of Gaussian.
pub struct GaussianCov3dSingleConfig;

impl GaussianCov3dConfig for GaussianCov3dSingleConfig {
    const NAME: &'static str = "single";

    type Field = [f32; 6];

    fn from_cov3d(cov3d: [f32; 6]) -> Self::Field {
        cov3d
    }
}

/// The half precision covariance 3D configuration of Gaussian.
pub struct GaussianCov3dHalfConfig;

impl GaussianCov3dConfig for GaussianCov3dHalfConfig {
    const NAME: &'static str = "half";

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

/// The Gaussians edit storage buffer.
#[derive(Debug, Clone)]
pub struct GaussiansEditBuffer {
    data: wgpu::Buffer,
    download: wgpu::Buffer,
}

impl GaussiansEditBuffer {
    /// Create a new Gaussians edit buffer.
    pub fn new(device: &wgpu::Device, gaussian_count: u32) -> Self {
        let size = gaussian_count * std::mem::size_of::<GaussianEditPod>() as u32;

        let data = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gaussians Edit Buffer"),
            size: size as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let download = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gaussians Edit Download Buffer"),
            size: size as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self { data, download }
    }

    /// Download the Gaussian edit.
    pub async fn download(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<GaussianEditPod>, Error> {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Gaussian Edit Download Encoder"),
        });
        self.prepare_download(&mut encoder);
        queue.submit(Some(encoder.finish()));

        self.map_download(device).await
    }

    /// Prepare for downloading the Gaussian edit.
    pub fn prepare_download(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.copy_buffer_to_buffer(self.buffer(), 0, &self.download, 0, self.download.size());
    }

    /// Map the download buffer to read the Gaussian edit.
    pub async fn map_download(&self, device: &wgpu::Device) -> Result<Vec<GaussianEditPod>, Error> {
        let (tx, rx) = oneshot::channel();
        let buffer_slice = self.download.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            if let Err(e) = tx.send(result) {
                log::error!("Error occurred while sending Gaussian edit: {e:?}");
            }
        });
        device.poll(wgpu::Maintain::Wait);
        rx.await??;

        let edits = bytemuck::allocation::pod_collect_to_vec(&buffer_slice.get_mapped_range());
        self.download.unmap();

        Ok(edits)
    }

    /// Get the buffer.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.data
    }

    /// Get the download buffer.
    pub fn download_buffer(&self) -> &wgpu::Buffer {
        &self.download
    }
}

bitflags::bitflags! {
    /// The flags for [`GaussianEditPod`].
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct GaussianEditFlag: u8 {
        /// No flag.
        const NONE = 0;

        /// Is the edit enabled.
        const ENABLED = 1 << 0;

        /// Hide the Gaussian.
        const HIDDEN = 1 << 1;

        /// Override the base color.
        const OVERRIDE_COLOR = 1 << 2;
    }
}

/// The POD representation of a Gaussian edit.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GaussianEditPod {
    /// \[Flag (8), HSB, or RGB if [`GaussianEditFlag::OVERRIDE_COLOR`] (24)\]
    pub flag_hsv: U8Vec4,

    /// \[Contrast (8), Exposure (8), Gamma (8), Alpha (8)\]
    pub contr_expo_gamma_alpha: U8Vec4,
}

impl GaussianEditPod {
    /// Create a new Gaussian edit.
    ///
    /// The hue is in the range of \[0, 1\].
    /// The saturation is in the range of \[0, 2\].
    /// The brightness is in the range of \[0, 2\].
    /// The RGB is in the range of \[0, 1\] (if [`GaussianEditFlag::OVERRIDE_COLOR`]).
    /// The contrast is in the range of \[-1, 1\].
    /// The exposure is in the range of \[-5, 5\].
    /// The gamma is in the range of \[0, 5\].
    /// The alpha is in the range of \[0, 2\].
    pub fn new(
        flag: GaussianEditFlag,
        hsv_or_rgb: Vec3,
        contrast: f32,
        exposure: f32,
        gamma: f32,
        alpha: f32,
    ) -> Self {
        let hsv_or_rgb = match flag.contains(GaussianEditFlag::OVERRIDE_COLOR) {
            true => hsv_or_rgb.clamp(Vec3::ZERO, Vec3::ONE) * 255.0,
            false => hsv_or_rgb.clamp(Vec3::ZERO, vec3(1.0, 2.0, 2.0)) * vec3(255.0, 127.5, 127.5),
        }
        .as_u8vec3();
        let contr_expo_gamma_alpha = vec4(
            (contrast.clamp(-1.0, 1.0) + 1.0) * 127.5,
            (exposure.clamp(-5.0, 5.0) + 5.0) * 25.5,
            gamma.clamp(0.0, 5.0) * 51.0,
            alpha.clamp(0.0, 2.0) * 127.5,
        )
        .as_u8vec4();
        let flag = flag.bits();

        Self {
            flag_hsv: u8vec4(flag, hsv_or_rgb.x, hsv_or_rgb.y, hsv_or_rgb.z),
            contr_expo_gamma_alpha,
        }
    }

    /// Get the flag.
    pub fn flag(&self) -> GaussianEditFlag {
        GaussianEditFlag::from_bits_truncate(self.flag_hsv.x)
    }

    /// Get the hue in the range of \[0, 1\].
    pub fn hue(&self) -> f32 {
        self.flag_hsv.y as f32 / 255.0
    }

    /// Get the saturation in the range of \[0, 2\].
    pub fn saturation(&self) -> f32 {
        self.flag_hsv.z as f32 / 127.5
    }

    /// Get the brightness in the range of \[0, 2\].
    pub fn brightness(&self) -> f32 {
        self.flag_hsv.w as f32 / 127.5
    }

    /// Get the RGB in the range of \[0, 1\].
    pub fn rgb(&self) -> Vec3 {
        self.flag_hsv.yzw().as_vec3().map(|x| x / 255.0)
    }

    /// Get the contrast in the range of \[-1, 1\].
    pub fn contrast(&self) -> f32 {
        self.contr_expo_gamma_alpha.x as f32 / 127.5 - 1.0
    }

    /// Get the exposure in the range of \[-5, 5\].
    pub fn exposure(&self) -> f32 {
        self.contr_expo_gamma_alpha.y as f32 / 25.5 - 5.0
    }

    /// Get the gamma in the range of \[0, 5\].
    pub fn gamma(&self) -> f32 {
        self.contr_expo_gamma_alpha.z as f32 / 51.0
    }

    /// Get the alpha in the range of \[0, 2\].
    pub fn alpha(&self) -> f32 {
        self.contr_expo_gamma_alpha.w as f32 / 127.5
    }
}

impl Default for GaussianEditPod {
    fn default() -> Self {
        Self::new(
            GaussianEditFlag::NONE,
            vec3(0.0, 0.0, 0.0),
            0.0,
            0.0,
            1.0,
            1.0,
        )
    }
}
