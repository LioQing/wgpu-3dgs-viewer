use glam::*;

use crate::{
    CameraBuffer, RendererCreateError,
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
///
/// ## Overview
///
/// This is used to handle viewport selections, including rectangle and brush selections.
///
/// It manages user interaction by storing the start and end states of the selections.
///
/// ## Usage
///
/// This selector is used in conjunction with the compute bundle created by
/// [`selection::create_viewport_bundle`](crate::selection::create_viewport_bundle).
///
/// Each function of this selector should reflects user's action:
/// - [`start`](Self::start): called when the user starts the selection (e.g., mouse button down).
/// - [`update`](Self::update): called when the user updates the selection (e.g., mouse move with button held).
/// - [`clear`](Self::clear): called to clear the selection texture (e.g., mouse button up).
/// - [`render`](Self::render): called to render the selection to the viewport texture (e.g., every frame).
///
/// Also, don't forget to evaluate and apply the selection using either a
/// [`editor::SelectionBundle`](crate::editor::SelectionBundle) or a selection modifier like
/// [`editor::BasicSelectionModifier`](crate::editor::BasicSelectionModifier) after the selection is
/// done (e.g., mouse button up).
///
/// Here is an example that uses this selector with a [`editor::BasicSelectionModifier`](crate::editor::BasicSelectionModifier),
/// which modifies some basic attributes of the selected Gaussians:
///
/// ```rust
/// # use pollster::FutureExt;
/// #
/// # async {
/// # use wgpu_3dgs_viewer::{
/// #     Viewer,
/// #     core::{self, BufferWrapper, glam::*},
/// #     editor::{self, Modifier},
/// #     selection,
/// # };
/// #
/// # type GaussianPod = wgpu_3dgs_viewer::core::GaussianPodWithShSingleCov3dSingleConfigs;
/// #
/// # let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
/// #
/// # let adapter = instance
/// #     .request_adapter(&wgpu::RequestAdapterOptions::default())
/// #     .await
/// #     .expect("adapter");
/// #
/// # let (device, queue) = adapter
/// #     .request_device(&wgpu::DeviceDescriptor {
/// #         label: Some("Device"),
/// #         required_features: wgpu::Features::empty(),
/// #         required_limits: adapter.limits(),
/// #         memory_hints: wgpu::MemoryHints::default(),
/// #         trace: wgpu::Trace::Off,
/// #     })
/// #     .await
/// #     .expect("device");
/// #
/// # let viewer = Viewer::<GaussianPod>::new(
/// #     &device,
/// #     wgpu::TextureFormat::Rgba8UnormSrgb,
/// #     &core::Gaussians {
/// #         gaussians: vec![core::Gaussian {
/// #             rot: Quat::IDENTITY,
/// #             pos: Vec3::ZERO,
/// #             color: U8Vec4::ZERO,
/// #             sh: [Vec3::ZERO; 15],
/// #             scale: Vec3::ONE,
/// #         }],
/// #     },
/// # )
/// # .unwrap();
/// #
/// # let viewport_size = UVec2::new(800, 600);
/// #
/// // Create the selector
/// let mut selector = selection::ViewportSelector::new(
///     &device,
///     &queue,
///     viewport_size,
///     &viewer.camera_buffer,
/// ).unwrap();
///
/// // Create the selection modifier
/// let mut selection_modifier = editor::SelectionModifier::new_with_basic_modifier(
///     &device,
///     &viewer.gaussians_buffer,
///     &viewer.model_transform_buffer,
///     &viewer.gaussian_transform_buffer,
///     vec![selection::create_viewport_bundle::<GaussianPod>(&device)],
/// );
///
/// // Create the bind group for the selector
/// let bind_group = selection_modifier.selection.bundles[0]
///     .create_bind_group(
///         &device,
///         1, // index 0 is the Gaussians buffer, so we use 1,
///         // see documentation of create_viewport_bundle
///         [
///             viewer.camera_buffer.buffer().as_entire_binding(),
///             wgpu::BindingResource::TextureView(selector.texture().view()), // Supply the selection texture to write to
///         ],
///     )
///     .unwrap();
///
/// // Set the selection expression to just use the selector
/// selection_modifier.selection_expr = editor::SelectionExpr::Selection(0, vec![bind_group]);
///
/// // In the event loop, handle user input to start, update, end, and apply the selection
/// let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
///     label: Some("Test Encoder"),
/// });
/// let left_mouse_button_just_pressed = true;
/// let left_mouse_button_held = false;
/// let left_mouse_button_just_released = false;
/// let mouse_pos = Vec2::ZERO;
///
/// if left_mouse_button_just_pressed {
///     selector.start(&queue, mouse_pos);
/// }
///
/// if left_mouse_button_held {
///     selector.update(&queue, mouse_pos);
/// }
///
/// if left_mouse_button_just_released {
///     selection_modifier.apply(
///         // Evaluate selection and apply modifiers
///         &device,
///         &mut encoder,
///         &viewer.gaussians_buffer,
///         &viewer.model_transform_buffer,
///         &viewer.gaussian_transform_buffer,
///     );
///
///     selector.clear(&mut encoder);
/// }
///
/// // Render the selector
/// selector.render(&mut encoder);
/// # }
/// # .block_on();
/// ```
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
    ) -> Result<Self, RendererCreateError> {
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
                depth_slice: None,
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
