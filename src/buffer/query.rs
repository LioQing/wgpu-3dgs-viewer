use glam::*;

use wgpu::util::DeviceExt;

use crate::Error;

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
    ///
    /// There can only be one query at a time.
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
    /// Do not query anything.
    None = 0 << 24,

    /// Query a single pixel hit. Done by render shader.
    Hit = 1 << 24,

    /// Query centroids by a rectangle. Done by preprocess shader.
    Rect = 2 << 24,

    /// Query centroids by a brush. Done by preprocess shader.
    Brush = 3 << 24,
}

/// The selection operations.
///
/// It will operate on the selection buffer using the query result.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuerySelectionOp {
    /// Do nothing.
    None = 0 << 16,

    /// Discards previous selection and selects the new one.
    Set = 1 << 16,

    /// Removes the new selection from the previous selection.
    Remove = 2 << 16,

    /// Adds the new selection to the previous selection.
    Add = 3 << 16,
}

/// The POD representation of a query.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct QueryPod {
    /// X: \[Type (8), SelectOp (8), Padding (16), ...\].
    /// Y-W: Additional content.
    pub content_u32: UVec4,

    /// Any additional content.
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
        match self.content_u32.x & 0xFF00_0000 {
            x if x == QueryType::None as u32 => QueryType::None,
            x if x == QueryType::Hit as u32 => QueryType::Hit,
            x if x == QueryType::Rect as u32 => QueryType::Rect,
            x if x == QueryType::Brush as u32 => QueryType::Brush,
            _ => panic!("Unknown query type"),
        }
    }

    /// Get the selection operation of the query.
    pub fn query_selection_op(&self) -> QuerySelectionOp {
        match self.content_u32.x & 0x00FF_0000 {
            x if x == QuerySelectionOp::None as u32 => QuerySelectionOp::None,
            x if x == QuerySelectionOp::Set as u32 => QuerySelectionOp::Set,
            x if x == QuerySelectionOp::Add as u32 => QuerySelectionOp::Add,
            x if x == QuerySelectionOp::Remove as u32 => QuerySelectionOp::Remove,
            _ => panic!("Unknown query selection operation"),
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
    /// - `coords` are the surface texture coordinates.
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

    /// Create a new [`QueryType::Rect`] query.
    ///
    /// - `top_left` are the top left coordinates of the selection rectangle.
    /// - `bottom_right` are the bottom right coordinates of the selection rectangle.
    ///
    /// *Note*: No result if `top_left.x < bottom_right.x` or `top_left.y < bottom_right.y`.
    pub const fn rect(top_left: Vec2, bottom_right: Vec2) -> Self {
        Self::new(
            uvec4(QueryType::Rect as u32, 0, 0, 0),
            vec4(top_left.x, top_left.y, bottom_right.x, bottom_right.y),
        )
    }

    /// Get as a reference of [`QueryType::Rect`] query.
    pub fn as_rect(&self) -> &QueryRectPod {
        bytemuck::cast_ref(self)
    }

    /// Get as a mutable reference of [`QueryType::Rect`] query.
    pub fn as_rect_mut(&mut self) -> &mut QueryRectPod {
        bytemuck::cast_mut(self)
    }

    /// Create a new [`QueryType::Brush`] query.
    ///
    /// - `radius` is the radius of the brush.
    /// - `start` is the starting point of the brush path.
    /// - `end` is the ending point of the brush path.
    pub const fn brush(radius: u32, start: Vec2, end: Vec2) -> Self {
        Self::new(
            uvec4(QueryType::Brush as u32, radius, 0, 0),
            vec4(start.x, start.y, end.x, end.y),
        )
    }

    /// Get as a reference of [`QueryType::Brush`] query.
    pub fn as_brush(&self) -> &QueryBrushPod {
        bytemuck::cast_ref(self)
    }

    /// Get as a mutable reference of [`QueryType::Brush`] query.
    pub fn as_brush_mut(&mut self) -> &mut QueryBrushPod {
        bytemuck::cast_mut(self)
    }

    /// Set the selection operation.
    pub fn with_selection_op(mut self, selection_op: QuerySelectionOp) -> Self {
        self.content_u32.x |= selection_op as u32;
        self
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

impl From<QueryRectPod> for QueryPod {
    fn from(query: QueryRectPod) -> Self {
        query.0
    }
}

impl From<QueryBrushPod> for QueryPod {
    fn from(query: QueryBrushPod) -> Self {
        query.0
    }
}

/// The POD representation of the [`QueryType::None`].
#[repr(C)]
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

    /// Get a mutable reference to the query.
    pub fn as_query_mut(&mut self) -> &mut QueryPod {
        &mut self.0
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

    /// Get a mutable reference to the query.
    pub fn as_query_mut(&mut self) -> &mut QueryPod {
        &mut self.0
    }
}

impl From<QueryPod> for QueryHitPod {
    fn from(query: QueryPod) -> Self {
        Self(query)
    }
}

/// The POD representation of the [`QueryType::Rect`].
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct QueryRectPod(QueryPod);

impl QueryRectPod {
    /// Create a new query.
    pub const fn new(top_left: Vec2, bottom_right: Vec2) -> Self {
        Self(QueryPod::rect(top_left, bottom_right))
    }

    /// Get the top left coordinates of the selection rectangle.
    pub fn top_left(&self) -> Vec2 {
        self.0.content_f32.xy()
    }

    /// Get the bottom right coordinates of the selection rectangle.
    pub fn bottom_right(&self) -> Vec2 {
        self.0.content_f32.zw()
    }

    /// Get a reference to the query.
    pub fn as_query(&self) -> &QueryPod {
        &self.0
    }

    /// Get a mutable reference to the query.
    pub fn as_query_mut(&mut self) -> &mut QueryPod {
        &mut self.0
    }
}

impl From<QueryPod> for QueryRectPod {
    fn from(query: QueryPod) -> Self {
        Self(query)
    }
}

/// The POD representation of the [`QueryType::Brush`].
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct QueryBrushPod(QueryPod);

impl QueryBrushPod {
    /// Create a new query.
    pub const fn new(radius: u32, start: Vec2, end: Vec2) -> Self {
        Self(QueryPod::brush(radius, start, end))
    }

    /// Get the radius of the brush.
    pub fn radius(&self) -> u32 {
        self.0.content_u32.y
    }

    /// Get the starting point of the brush path.
    pub fn start(&self) -> Vec2 {
        self.0.content_f32.xy()
    }

    /// Get the ending point of the brush path.
    pub fn end(&self) -> Vec2 {
        self.0.content_f32.zw()
    }

    /// Get a reference to the query.
    pub fn as_query(&self) -> &QueryPod {
        &self.0
    }

    /// Get a mutable reference to the query.
    pub fn as_query_mut(&mut self) -> &mut QueryPod {
        &mut self.0
    }
}

impl From<QueryPod> for QueryBrushPod {
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
            size: std::mem::size_of::<u32>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let download = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Query Result Count Download buffer"),
            size: std::mem::size_of::<u32>() as wgpu::BufferAddress,
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
        encoder.copy_buffer_to_buffer(self.buffer(), 0, &self.download, 0, self.download.size());
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
