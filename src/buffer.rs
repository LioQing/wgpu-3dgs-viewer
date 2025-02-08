use glam::*;

use half::f16;
use wgpu::util::DeviceExt;

use crate::{wgpu_sort, CameraTrait, Error, Gaussian};

/// The Gaussians storage buffer.
#[derive(Debug)]
pub struct GaussiansBuffer<G: GaussianPod>(wgpu::Buffer, std::marker::PhantomData<G>);

impl<G: GaussianPod> GaussiansBuffer<G> {
    /// Create a new Gaussians buffer.
    pub fn new(device: &wgpu::Device, gaussians: &[Gaussian]) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Gaussians Buffer"),
            contents: bytemuck::cast_slice(
                gaussians
                    .iter()
                    .map(G::from_gaussian)
                    .collect::<Vec<_>>()
                    .as_slice(),
            ),
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
                    .map(G::from_gaussian)
                    .collect::<Vec<_>>()
                    .as_slice(),
            ),
        );
    }
}

/// The spherical harmonics configuration of Gaussian.
pub trait GaussianShConfig {
    /// The name of the configuration.
    ///
    /// Must match the name in the shader.
    const NAME: &'static str;

    /// The WGSL shader.
    const WGSL: &'static str = include_str!("shader/gaussian_configs.wgsl");

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

/// The min max normalized SH configuration of Gaussian.
pub struct GaussianShMinMaxNormConfig;

impl GaussianShConfig for GaussianShMinMaxNormConfig {
    const NAME: &'static str = "min max norm";

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
    const WGSL: &'static str = include_str!("shader/gaussian_configs.wgsl");

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
gaussian_pod!(sh = MinMaxNorm, cov3d = Single, padding_size = 1);
gaussian_pod!(sh = MinMaxNorm, cov3d = Half, padding_size = 0);
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
    pub fn update(&self, queue: &wgpu::Queue, camera: &impl CameraTrait, size: UVec2) {
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
#[derive(Debug)]
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
        queue.write_buffer(
            &self.0,
            0,
            bytemuck::bytes_of(&ModelTransformPod::new(pos, quat, scale)),
        );
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
    /// Returns `None` if the degree is not in the range of \[0, 3\].
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
}

/// The Gaussian transform buffer.
#[derive(Debug)]
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
        queue.write_buffer(
            &self.0,
            0,
            bytemuck::bytes_of(&GaussianTransformPod::new(
                size,
                display_mode,
                sh_deg,
                no_sh0,
            )),
        );
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

    /// (display_mode, sh_deg, no_sh0, padding)
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

/// The query uniform buffer for [`Renderer`](crate::Renderer).
#[derive(Debug)]
pub struct QueryBuffer(wgpu::Buffer);

impl QueryBuffer {
    /// Create a new query buffer.
    pub fn new(device: &wgpu::Device) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Query Buffer"),
            contents: bytemuck::bytes_of(&QueryPod::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Self(buffer)
    }

    /// Update the query buffer.
    pub fn update(&self, queue: &wgpu::Queue, query: &QueryPod) {
        queue.write_buffer(&self.0, 0, bytemuck::bytes_of(query));
    }

    /// Get the buffer.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.0
    }
}

/// The types of queries.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QueryType {
    None = 0,
    Hit = 1,
}

/// The POD representation of a query.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct QueryPod {
    pub content_u32: UVec4,
    pub content_f32: Vec4,
}

impl QueryPod {
    /// Create a new query.
    pub const fn new(content_u32: UVec4, content_f32: Vec4) -> Self {
        Self {
            content_u32,
            content_f32,
        }
    }

    /// Get the type of the query.
    pub fn query_type(&self) -> QueryType {
        match self.content_u32.x {
            x if x == QueryType::None as u32 => QueryType::None,
            x if x == QueryType::Hit as u32 => QueryType::Hit,
            _ => panic!("Unknown query type"),
        }
    }

    /// Create a new [`QueryType::None`] query.
    pub const fn none() -> Self {
        Self::new(
            uvec4(QueryType::None as u32, 0, 0, 0),
            vec4(0.0, 0.0, 0.0, 0.0),
        )
    }

    /// Get as a reference of [`QueryType::None`] query.
    pub fn as_none(&self) -> &QueryNonePod {
        bytemuck::cast_ref(self)
    }

    /// Get as a mutable reference of [`QueryType::None`] query.
    pub fn as_none_mut(&mut self) -> &mut QueryNonePod {
        bytemuck::cast_mut(self)
    }

    /// Create a new [`QueryType::Hit`] query.
    ///
    /// `coords` are the surface texture coordinates.
    pub const fn hit(coords: Vec2) -> Self {
        Self::new(
            uvec4(QueryType::Hit as u32, 0, 0, 0),
            vec4(coords.x, coords.y, 0.0, 0.0),
        )
    }

    /// Get as a reference of [`QueryType::Hit`] query.
    pub fn as_hit(&self) -> &QueryHitPod {
        bytemuck::cast_ref(self)
    }

    /// Get as a mutable reference of [`QueryType::Hit`] query.
    pub fn as_hit_mut(&mut self) -> &mut QueryHitPod {
        bytemuck::cast_mut(self)
    }
}

impl Default for QueryPod {
    fn default() -> Self {
        Self::none()
    }
}

impl From<QueryNonePod> for QueryPod {
    fn from(query: QueryNonePod) -> Self {
        query.0
    }
}

impl From<QueryHitPod> for QueryPod {
    fn from(query: QueryHitPod) -> Self {
        query.0
    }
}

/// The POD representation of the [`QueryType::None`].
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct QueryNonePod(QueryPod);

impl QueryNonePod {
    /// Create a new query.
    pub const fn new() -> Self {
        Self(QueryPod::none())
    }

    /// Get a reference to the query.
    pub fn as_query(&self) -> &QueryPod {
        &self.0
    }
}

impl Default for QueryNonePod {
    fn default() -> Self {
        Self::new()
    }
}

impl From<QueryPod> for QueryNonePod {
    fn from(query: QueryPod) -> Self {
        Self(query)
    }
}

/// The POD representation of the [`QueryType::Hit`].
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct QueryHitPod(QueryPod);

impl QueryHitPod {
    /// Create a new query.
    pub const fn new(coords: Vec2) -> Self {
        Self(QueryPod::hit(coords))
    }

    /// Get the coordinates.
    pub fn coords(&self) -> Vec2 {
        self.0.content_f32.xy()
    }

    /// Get a reference to the query.
    pub fn as_query(&self) -> &QueryPod {
        &self.0
    }
}

impl From<QueryPod> for QueryHitPod {
    fn from(query: QueryPod) -> Self {
        Self(query)
    }
}

/// The query result count storage buffer for [`Renderer`](crate::Renderer).
#[derive(Debug)]
pub struct QueryResultCountBuffer {
    data: wgpu::Buffer,
    download: wgpu::Buffer,
}

impl QueryResultCountBuffer {
    /// Create a new query result count buffer.
    pub fn new(device: &wgpu::Device) -> Self {
        let data = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Query Result Count Buffer"),
            size: std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let download = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Query Result Count Download buffer"),
            size: data.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self { data, download }
    }

    /// Download the query result count.
    pub async fn download(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<u32, Error> {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Query Result Count Download Encoder"),
        });
        self.prepare_download(&mut encoder);
        queue.submit(Some(encoder.finish()));

        self.map_download(device).await
    }

    /// Prepare for downloading the query result count.
    pub fn prepare_download(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.copy_buffer_to_buffer(self.buffer(), 0, &self.download, 0, self.buffer().size());
    }

    /// Map the download buffer to read the query result count.
    pub async fn map_download(&self, device: &wgpu::Device) -> Result<u32, Error> {
        let (tx, rx) = oneshot::channel();
        let buffer_slice = self.download.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            if let Err(e) = tx.send(result) {
                log::error!("Error occurred while sending query result count: {e:?}");
            }
        });
        device.poll(wgpu::Maintain::Wait);
        rx.await??;

        let count = bytemuck::pod_read_unaligned(&buffer_slice.get_mapped_range());
        self.download.unmap();

        Ok(count)
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

/// The query results storage buffer for [`Renderer`](crate::Renderer).
#[derive(Debug)]
pub struct QueryResultsBuffer(wgpu::Buffer);

impl QueryResultsBuffer {
    /// Create a new query results buffer.
    pub fn new(device: &wgpu::Device, gaussian_count: u32) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Query Results Buffer"),
            size: (gaussian_count * std::mem::size_of::<QueryPod>() as u32) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        Self(buffer)
    }

    /// Download the query results.
    pub async fn download(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        count: u32,
    ) -> Result<Vec<QueryResultPod>, Error> {
        if count == 0 {
            return Ok(Vec::new());
        }

        let download = self.create_download_buffer(device, count);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Query Results Download Encoder"),
        });
        self.prepare_download(&mut encoder, &download);
        queue.submit(Some(encoder.finish()));

        self.map_download(device, &download).await
    }

    /// Create the download buffer.
    pub fn create_download_buffer(&self, device: &wgpu::Device, count: u32) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Query Results Download buffer"),
            size: (count * std::mem::size_of::<QueryResultPod>() as u32) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Prepare for downloading the query results.
    pub fn prepare_download(&self, encoder: &mut wgpu::CommandEncoder, download: &wgpu::Buffer) {
        encoder.copy_buffer_to_buffer(self.buffer(), 0, download, 0, download.size());
    }

    /// Map the download buffer to read the query results.
    pub async fn map_download(
        &self,
        device: &wgpu::Device,
        download: &wgpu::Buffer,
    ) -> Result<Vec<QueryResultPod>, Error> {
        let (tx, rx) = oneshot::channel();
        let buffer_slice = download.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            if let Err(e) = tx.send(result) {
                log::error!("Error occurred while sending query results: {e:?}");
            }
        });
        device.poll(wgpu::Maintain::Wait);
        rx.await??;

        Ok(bytemuck::allocation::pod_collect_to_vec(
            &buffer_slice.get_mapped_range(),
        ))
    }

    /// Get the buffer.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.0
    }
}

/// The POD representation of a query result.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct QueryResultPod {
    pub content_u32: UVec4,
    pub content_f32: Vec4,
}

impl QueryResultPod {
    /// Get as a reference of [`QueryType::Hit`] query result.
    pub fn as_hit(&self) -> &QueryHitResultPod {
        bytemuck::cast_ref(self)
    }

    /// Get as a mutable reference of [`QueryType::Hit`] query result.
    pub fn as_hit_mut(&mut self) -> &mut QueryHitResultPod {
        bytemuck::cast_mut(self)
    }
}

/// The POD representation of the query result of [`QueryType::Hit`].
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct QueryHitResultPod(QueryResultPod);

impl QueryHitResultPod {
    /// Get the index of the hit Gaussian.
    pub const fn index(&self) -> u32 {
        self.0.content_u32.x
    }

    /// Get the index of the hit Gaussian mutably.
    pub fn index_mut(&mut self) -> &mut u32 {
        &mut self.0.content_u32.x
    }

    /// Get the normalized depth of the hit Gaussian.
    pub fn depth(&self) -> f32 {
        self.0.content_f32.x
    }

    /// Get the normalized depth of the hit Gaussian mutably.
    pub fn depth_mut(&mut self) -> &mut f32 {
        &mut self.0.content_f32.x
    }

    /// Get the alpha of the hit Gaussian.
    pub fn alpha(&self) -> f32 {
        self.0.content_f32.y
    }

    /// Get the alpha of the hit Gaussian mutably.
    pub fn alpha_mut(&mut self) -> &mut f32 {
        &mut self.0.content_f32.y
    }
}

impl From<QueryResultPod> for QueryHitResultPod {
    fn from(result: QueryResultPod) -> Self {
        Self(result)
    }
}
