use glam::*;
use wgpu::util::DeviceExt;

use crate::core::{self, BufferWrapper, FixedSizeBufferWrapper};

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

/// The position buffer for [`ViewportTexture`].
///
/// This is used for [`ViewportTextureRectangleRenderer`](crate::selection::ViewportTextureRectangleRenderer)
/// and [`ViewportTextureBrushRenderer`](crate::selection::ViewportTextureBrushRenderer).
#[derive(Debug, Clone)]
pub struct ViewportTexturePosBuffer(wgpu::Buffer);

impl ViewportTexturePosBuffer {
    /// Create a new position buffer.
    pub fn new(device: &wgpu::Device) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Viewport Selection Texture Pos Buffer"),
            size: std::mem::size_of::<Vec2>() as wgpu::BufferAddress,
            usage: Self::DEFAULT_USAGES,
            mapped_at_creation: false,
        });

        Self(buffer)
    }

    /// Update the position buffer.
    pub fn update(&self, queue: &wgpu::Queue, pos: Vec2) {
        queue.write_buffer(&self.0, 0, bytemuck::bytes_of(&pos));
    }
}

impl BufferWrapper for ViewportTexturePosBuffer {
    fn buffer(&self) -> &wgpu::Buffer {
        &self.0
    }
}

impl From<ViewportTexturePosBuffer> for wgpu::Buffer {
    fn from(wrapper: ViewportTexturePosBuffer) -> Self {
        wrapper.0
    }
}

impl TryFrom<wgpu::Buffer> for ViewportTexturePosBuffer {
    type Error = core::FixedSizeBufferWrapperError;

    fn try_from(buffer: wgpu::Buffer) -> Result<Self, Self::Error> {
        Self::verify_buffer_size(&buffer).map(|()| Self(buffer))
    }
}

impl FixedSizeBufferWrapper for ViewportTexturePosBuffer {
    type Pod = Vec2;
}

/// The f32 buffer for [`ViewportTexture`].
///
/// This is used for [`ViewportTextureBrushRenderer`](crate::selection::ViewportTextureBrushRenderer).
#[derive(Debug, Clone)]
pub struct ViewportTextureF32Buffer(wgpu::Buffer);

impl ViewportTextureF32Buffer {
    /// Create a new position buffer.
    pub fn new(device: &wgpu::Device) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Viewport Selection Texture Pos Buffer"),
            size: std::mem::size_of::<f32>() as wgpu::BufferAddress,
            usage: Self::DEFAULT_USAGES,
            mapped_at_creation: false,
        });

        Self(buffer)
    }

    /// Update the f32 buffer.
    pub fn update(&self, queue: &wgpu::Queue, value: f32) {
        queue.write_buffer(&self.0, 0, bytemuck::bytes_of(&value));
    }
}

impl BufferWrapper for ViewportTextureF32Buffer {
    fn buffer(&self) -> &wgpu::Buffer {
        &self.0
    }
}

impl From<ViewportTextureF32Buffer> for wgpu::Buffer {
    fn from(wrapper: ViewportTextureF32Buffer) -> Self {
        wrapper.0
    }
}

impl TryFrom<wgpu::Buffer> for ViewportTextureF32Buffer {
    type Error = core::FixedSizeBufferWrapperError;

    fn try_from(buffer: wgpu::Buffer) -> Result<Self, Self::Error> {
        Self::verify_buffer_size(&buffer).map(|()| Self(buffer))
    }
}

impl FixedSizeBufferWrapper for ViewportTextureF32Buffer {
    type Pod = f32;
}

/// The invert selection buffer for [`Preprocessor`](crate::Preprocessor).
///
/// This is used for inverting the selection in the preprocessor, it is essentially just a boolean.
#[derive(Debug, Clone)]
pub struct PreprocessorInvertSelectionBuffer(wgpu::Buffer);

impl PreprocessorInvertSelectionBuffer {
    /// Create a new invert selection buffer.
    ///
    /// Note: the initial value is true.
    pub fn new(device: &wgpu::Device) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Preprocessor Invert Selection Buffer"),
            contents: bytemuck::bytes_of(&1u32),
            usage: Self::DEFAULT_USAGES,
        });

        Self(buffer)
    }

    /// Update the invert selection buffer.
    pub fn update(&self, queue: &wgpu::Queue, invert: bool) {
        let value: u32 = if invert { 1 } else { 0 };
        queue.write_buffer(&self.0, 0, bytemuck::bytes_of(&value));
    }
}

impl BufferWrapper for PreprocessorInvertSelectionBuffer {
    fn buffer(&self) -> &wgpu::Buffer {
        &self.0
    }
}

impl From<PreprocessorInvertSelectionBuffer> for wgpu::Buffer {
    fn from(wrapper: PreprocessorInvertSelectionBuffer) -> Self {
        wrapper.0
    }
}

impl TryFrom<wgpu::Buffer> for PreprocessorInvertSelectionBuffer {
    type Error = core::FixedSizeBufferWrapperError;

    fn try_from(buffer: wgpu::Buffer) -> Result<Self, Self::Error> {
        Self::verify_buffer_size(&buffer).map(|()| Self(buffer))
    }
}

impl FixedSizeBufferWrapper for PreprocessorInvertSelectionBuffer {
    type Pod = u32;
}
