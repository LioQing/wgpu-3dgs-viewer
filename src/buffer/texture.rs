#[cfg(feature = "query-texture")]
use glam::*;

/// A texture, including its view.
pub trait Texture {
    /// The texture.
    fn texture(&self) -> &wgpu::Texture;

    /// The view of the texture.
    fn view(&self) -> &wgpu::TextureView;
}

/// A query texture.
///
/// This requires the `query-texture` feature.
#[cfg(feature = "query-texture")]
#[derive(Debug)]
pub struct QueryTexture {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}

#[cfg(feature = "query-texture")]
impl QueryTexture {
    /// Create a new query texture.
    pub fn new(device: &wgpu::Device, size: UVec2) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Query Texture"),
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

    /// Update the size of the query texture.
    ///
    /// You should also want to call any update bind group in pipelines that use this texture.
    pub fn update_size(&mut self, device: &wgpu::Device, size: UVec2) {
        self.texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Query Texture"),
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

        self.view = self
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
    }
}

#[cfg(feature = "query-texture")]
impl Texture for QueryTexture {
    fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    fn view(&self) -> &wgpu::TextureView {
        &self.view
    }
}
