use glam::*;

use wgpu::util::DeviceExt;

use crate::{wgpu_sort, CameraTrait, Error, Gaussian};

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
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GaussianDisplayMode {
    Splat = 0,
    Ellipse = 1,
    Point = 2,
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
    pub fn update(&self, queue: &wgpu::Queue, size: f32, display_mode: GaussianDisplayMode) {
        queue.write_buffer(
            &self.0,
            0,
            bytemuck::bytes_of(&GaussianTransformPod::new(size, display_mode)),
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
    pub display_mode: u32,
}

impl GaussianTransformPod {
    /// Create a new Gaussian transformation.
    pub const fn new(size: f32, display_mode: GaussianDisplayMode) -> Self {
        let display_mode = display_mode as u32;
        Self { size, display_mode }
    }
}

impl Default for GaussianTransformPod {
    fn default() -> Self {
        Self::new(1.0, GaussianDisplayMode::Splat)
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

    /// Get the difference from the coordinates to the hit Gaussian.
    pub fn diff(&self) -> Vec2 {
        self.0.content_f32.zw()
    }
}

impl From<QueryResultPod> for QueryHitResultPod {
    fn from(result: QueryResultPod) -> Self {
        Self(result)
    }
}
