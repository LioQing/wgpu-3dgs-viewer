use glam::*;

use crate::{
    CameraBuffer, Error,
    selection::{ViewportTexture, ViewportTexturePosBuffer, ViewportTextureRectangleRenderer},
};

/// A selector to handle viewport selections.
#[derive(Debug)]
pub struct ViewportSelector {
    /// The start position of the selection.
    ///
    /// - In rectangle, this is the top left corner.
    /// - In brush, this is the previoous brush position.
    pub start_pos: Option<Vec2>,

    /// The end position of the selection.
    ///
    /// - In rectangle, this is the bottom right corner.
    /// - In brush, this is the current brush position.
    pub end_pos: Option<Vec2>,

    /// The buffer for [`ViewportSelector::start_pos`].
    pub start_buffer: ViewportTexturePosBuffer,

    /// The buffer for [`ViewportSelector::end_pos`].
    pub end_buffer: ViewportTexturePosBuffer,

    /// The viewport texture holding the selection.
    pub viewport_texture: ViewportTexture,

    /// The rectangle renderer for viewport selection.
    pub rectangle_renderer: ViewportTextureRectangleRenderer,
}

impl ViewportSelector {
    /// Create a new viewport selector.
    pub fn new(
        device: &wgpu::Device,
        viewport_size: UVec2,
        camera: &CameraBuffer,
    ) -> Result<Self, Error> {
        let start_buffer = ViewportTexturePosBuffer::new(device);
        let end_buffer = ViewportTexturePosBuffer::new(device);
        let viewport_texture = ViewportTexture::new(device, viewport_size);
        let rectangle_renderer = ViewportTextureRectangleRenderer::new(
            device,
            &viewport_texture,
            camera,
            &start_buffer,
            &end_buffer,
        )?;

        Ok(Self {
            start_pos: None,
            end_pos: None,
            start_buffer,
            end_buffer,
            viewport_texture,
            rectangle_renderer,
        })
    }

    /// Start the selection at the given position.
    pub fn start(&mut self, queue: &wgpu::Queue, pos: Vec2) {
        self.start_pos = Some(pos);
        self.start_buffer.update(queue, pos);
    }

    /// Update the end position of the selection.
    pub fn update(&mut self, queue: &wgpu::Queue, pos: Vec2) {
        self.end_pos = Some(pos);
        self.end_buffer.update(queue, pos);
    }

    /// Clear the selection.
    pub fn clear(&mut self, queue: &wgpu::Queue) {
        self.start_pos = None;
        self.end_pos = None;
        self.start_buffer.update(queue, Vec2::ZERO);
        self.end_buffer.update(queue, Vec2::ZERO);
    }

    /// Render the selection rectangle.
    pub fn render(&self, encoder: &mut wgpu::CommandEncoder) {
        self.rectangle_renderer
            .render(encoder, &self.viewport_texture);
    }
}
