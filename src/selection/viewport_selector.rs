use glam::*;

use crate::{
    CameraBuffer, Error,
    selection::{
        ViewportTexture, ViewportTextureBrushRenderer, ViewportTextureF32Buffer,
        ViewportTexturePosBuffer, ViewportTextureRectangleRenderer,
    },
};

/// The viewport selector type.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewportSelectorType {
    /// Rectangle selection.
    #[default]
    Rectangle,
    /// Brush selection.
    Brush,
}

/// A selector to handle viewport selections.
#[derive(Debug)]
pub struct ViewportSelector {
    /// The start position of the selection.
    ///
    /// - In rectangle, this is the top left corner.
    /// - In brush, this is the previoous brush position.
    start_pos: Option<Vec2>,

    /// The end position of the selection.
    ///
    /// - In rectangle, this is the bottom right corner.
    /// - In brush, this is the current brush position.
    end_pos: Option<Vec2>,

    /// The radius of the brush selection.
    brush_radius: f32,

    /// The buffer for [`ViewportSelector::start_pos`].
    start_buffer: ViewportTexturePosBuffer,

    /// The buffer for [`ViewportSelector::end_pos`].
    end_buffer: ViewportTexturePosBuffer,

    /// The buffer for [`ViewportSelector::brush_radius`].
    radius_buffer: ViewportTextureF32Buffer,

    /// The viewport texture holding the selection.
    viewport_texture: ViewportTexture,

    /// The rectangle renderer for viewport selection.
    rectangle_renderer: ViewportTextureRectangleRenderer,

    /// The brush renderer for viewport selection.
    brush_renderer: ViewportTextureBrushRenderer,

    /// The selector type.
    pub selector_type: ViewportSelectorType,
}

impl ViewportSelector {
    /// The default brush radius.
    pub const DEFAULT_BRUSH_RADIUS: f32 = 50.0;

    /// Create a new viewport selector.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        viewport_size: UVec2,
        camera: &CameraBuffer,
    ) -> Result<Self, Error> {
        let start_buffer = ViewportTexturePosBuffer::new(device);
        let end_buffer = ViewportTexturePosBuffer::new(device);
        let radius_buffer = ViewportTextureF32Buffer::new(device);
        radius_buffer.update(queue, Self::DEFAULT_BRUSH_RADIUS);
        let viewport_texture = ViewportTexture::new(device, viewport_size);
        let rectangle_renderer = ViewportTextureRectangleRenderer::new(
            device,
            &viewport_texture,
            camera,
            &start_buffer,
            &end_buffer,
        )?;
        let brush_renderer = ViewportTextureBrushRenderer::new(
            device,
            &viewport_texture,
            camera,
            &start_buffer,
            &end_buffer,
            &radius_buffer,
        )?;

        Ok(Self {
            start_pos: None,
            end_pos: None,
            brush_radius: Self::DEFAULT_BRUSH_RADIUS,

            start_buffer,
            end_buffer,
            radius_buffer,

            viewport_texture,

            rectangle_renderer,
            brush_renderer,

            selector_type: ViewportSelectorType::default(),
        })
    }

    /// Start the selection at the given position.
    pub fn start(&mut self, queue: &wgpu::Queue, pos: Vec2) {
        self.start_pos = Some(pos);
        self.start_buffer.update(queue, pos);
        self.end_pos = Some(pos);
        self.end_buffer.update(queue, pos);
    }

    /// Update the end position of the selection.
    pub fn update(&mut self, queue: &wgpu::Queue, pos: Vec2) {
        match self.selector_type {
            ViewportSelectorType::Rectangle => {
                self.end_pos = Some(pos);
                self.end_buffer.update(queue, pos);
            }
            ViewportSelectorType::Brush => {
                self.start_pos = self.end_pos;
                self.start_buffer
                    .update(queue, self.start_pos.unwrap_or(pos));
                self.end_pos = Some(pos);
                self.end_buffer.update(queue, pos);
            }
        }
    }

    /// Clear the selection viewport texture.
    pub fn clear(&mut self, encoder: &mut wgpu::CommandEncoder) {
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Viewport Selection Clear Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: self.viewport_texture.view(),
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
    }

    /// Render the selection rectangle.
    pub fn render(&self, encoder: &mut wgpu::CommandEncoder) {
        match self.selector_type {
            ViewportSelectorType::Rectangle => self
                .rectangle_renderer
                .render(encoder, &self.viewport_texture),
            ViewportSelectorType::Brush => {
                self.brush_renderer.render(encoder, &self.viewport_texture)
            }
        }
    }

    /// Get the viewport texture.
    pub fn texture(&self) -> &ViewportTexture {
        &self.viewport_texture
    }

    /// Set the brush radius.
    pub fn set_brush_radius(&mut self, queue: &wgpu::Queue, radius: f32) {
        self.brush_radius = radius;
        self.radius_buffer.update(queue, radius);
    }

    /// Update the viewport size.
    ///
    /// After calling this method, you need to update bind groups that uses this texture.
    pub fn resize(&mut self, device: &wgpu::Device, new_size: UVec2) {
        self.viewport_texture = ViewportTexture::new(device, new_size);
    }
}
