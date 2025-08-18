use glam::*;
use wgpu_3dgs_core::BufferWrapper;

/// A viewport selection texture for the compute bundle created by
/// [`selection::create_viewport_bundle`](crate::selection::create_viewport_bundle).
#[derive(Debug, Clone)]
pub struct ViewportTexture {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}

impl ViewportTexture {
    /// Create a new viewport texture.
    pub fn new(device: &wgpu::Device, size: UVec2) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Viewport Selection Texture"),
            size: wgpu::Extent3d {
                width: size.x,
                height: size.y,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            view_formats: &[],
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        Self { texture, view }
    }

    /// Get the texture.
    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    /// Get the texture view.
    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }
}

/// The top left coordinate buffer for [`ViewportTextureRectangle`](crate::selection::ViewportTextureRectangle).
#[derive(Debug, Clone)]
pub struct ViewportTextureRectangleTopLeftBuffer(wgpu::Buffer);

impl ViewportTextureRectangleTopLeftBuffer {
    /// Create a new top left coordinate buffer.
    pub fn new(device: &wgpu::Device) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Viewport Selection Texture Rectangle Top Left Buffer"),
            size: std::mem::size_of::<Vec2>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self(buffer)
    }

    /// Update the top left coordinate buffer.
    pub fn update(&self, queue: &wgpu::Queue, top_left: Vec2) {
        queue.write_buffer(&self.0, 0, bytemuck::bytes_of(&top_left));
    }
}

impl BufferWrapper for ViewportTextureRectangleTopLeftBuffer {
    fn buffer(&self) -> &wgpu::Buffer {
        &self.0
    }
}

/// The bottom right coordinate buffer for [`ViewportTextureRectangle`](crate::selection::ViewportTextureRectangle).
#[derive(Debug, Clone)]
pub struct ViewportTextureRectangleBottomRightBuffer(wgpu::Buffer);

impl ViewportTextureRectangleBottomRightBuffer {
    /// Create a new bottom right coordinate buffer.
    pub fn new(device: &wgpu::Device) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Viewport Selection Texture Rectangle Bottom Right Buffer"),
            size: std::mem::size_of::<Vec2>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self(buffer)
    }

    /// Update the bottom right coordinate buffer.
    pub fn update(&self, queue: &wgpu::Queue, bottom_right: Vec2) {
        queue.write_buffer(&self.0, 0, bytemuck::bytes_of(&bottom_right));
    }
}

impl BufferWrapper for ViewportTextureRectangleBottomRightBuffer {
    fn buffer(&self) -> &wgpu::Buffer {
        &self.0
    }
}
