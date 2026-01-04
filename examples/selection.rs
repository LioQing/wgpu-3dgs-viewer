//! This example enables viewport-based selsection of Gaussians using the `viewer-selection` feature.
//!
//! For example, to filter the selected Gaussians:
//!
//! ```sh
//! cargo run --example selection --features="viewer-selection" -- --model path/to/model.ply --filter
//! ```
//!
//! To view more options and the controls, run with `--help`:
//!
//! ```sh
//! cargo run --example selection --features="viewer-selection" -- --help
//! ```

use std::sync::Arc;

use async_trait::async_trait;
use clap::Parser;
use glam::*;
use winit::{error::EventLoopError, event_loop::EventLoop, keyboard::KeyCode, window::Window};

use wgpu_3dgs_viewer::{
    self as gs,
    core::{BufferWrapper, GaussiansSource},
    editor::{BasicColorRgbOverrideOrHsvModifiersPod, Modifier},
};

mod utils;
use utils::core;

/// The command line arguments.
#[derive(Parser, Debug)]
#[command(
    version,
    about,
    long_about = "\
    A 3D Gaussian splatting viewer written in Rust using wgpu.\n\
    \n\
    Use W, A, S, D, Space, Shift to move, use mouse to rotate.\n\
    Use N to disable selection mode.\n\
    Use B to toggle brush selection mode.\n\
    Use R to toggle rectangle selection mode.\n\
    Use I to invert selection, has immediate effect in filter mode.\n\
    Use Left Click to use the current selector.\n\
    "
)]
struct Args {
    /// Path to the .ply file.
    #[arg(short, long)]
    model: String,

    /// Enable filter mode, where instead of modifying the color, it filters the selected Gaussians.
    #[arg(short, long)]
    filter: bool,

    /// Enable immediate mode, where the selection is applied while still selecting.
    #[arg(short, long)]
    immediate: bool,
}

fn main() -> Result<(), EventLoopError> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let event_loop = EventLoop::new()?;
    event_loop.run_app(&mut core::App::<System>::new(Args::parse()))?;
    Ok(())
}

/// The application system.
#[allow(dead_code)]
struct System {
    surface: wgpu::Surface<'static>,
    queue: wgpu::Queue,
    device: wgpu::Device,
    config: wgpu::SurfaceConfiguration,

    filter: bool,
    immediate: bool,
    inverted: bool,
    selector_type: Option<gs::selection::ViewportSelectorType>,

    camera: gs::Camera,
    gaussians: gs::core::Gaussians,
    viewer: gs::Viewer,
    selector: gs::selection::ViewportSelector,

    viewport_selection_modifier: gs::editor::NonDestructiveModifier<
        gs::DefaultGaussianPod,
        gs::editor::BasicSelectionModifier<gs::DefaultGaussianPod>,
    >,
    viewport_texture_overlay_renderer: utils::selection::ViewportTextureOverlayRenderer,
}

#[async_trait]
impl core::System for System {
    type Args = Args;

    async fn init(window: Arc<Window>, args: &Args) -> Self {
        let model_path = &args.model;
        let filter = args.filter;
        let immediate = args.immediate;
        let size = window.inner_size();

        log::debug!("Creating wgpu instance");
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

        log::debug!("Creating window surface");
        let surface = instance.create_surface(window.clone()).expect("surface");

        log::debug!("Requesting adapter");
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("adapter");

        log::debug!("Requesting device");
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Device"),
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .expect("device");

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps.formats[0];
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![surface_format.remove_srgb_suffix()],
            desired_maximum_frame_latency: 2,
        };

        log::debug!("Configuring surface");
        surface.configure(&device, &config);

        log::debug!("Creating gaussians");
        let gaussians = [GaussiansSource::Ply, GaussiansSource::Spz]
            .into_iter()
            .find_map(|source| gs::core::Gaussians::read_from_file(model_path, source).ok())
            .expect("gaussians");

        log::debug!("Creating camera");
        let camera = gs::Camera::new(0.1..1e4, 60f32.to_radians());

        log::debug!("Creating viewer");
        let mut viewer = gs::Viewer::new_with_options(
            &device,
            config.view_formats[0],
            &gaussians,
            gs::ViewerCreateOptions {
                gaussians_buffer_usage:
                    gs::core::GaussiansBuffer::<gs::DefaultGaussianPod>::DEFAULT_USAGES
                        | wgpu::BufferUsages::COPY_SRC,
                ..Default::default()
            },
        )
        .expect("viewer");
        viewer.update_model_transform(
            &queue,
            Vec3::ZERO,
            Quat::from_axis_angle(Vec3::Z, 180f32.to_radians()),
            Vec3::ONE,
        );

        log::debug!("Creating selector");
        let mut selector = gs::selection::ViewportSelector::new(
            &device,
            &queue,
            UVec2::new(size.width, size.height),
            &viewer.camera_buffer,
        )
        .expect("selector");
        selector.selector_type = gs::selection::ViewportSelectorType::Brush;

        log::debug!("Creating selection viewport selection modifier");
        let mut viewport_selection_modifier = gs::editor::NonDestructiveModifier::new(
            &device,
            &queue,
            gs::editor::BasicSelectionModifier::new_with_basic_modifier(
                &device,
                &viewer.gaussians_buffer,
                &viewer.model_transform_buffer,
                &viewer.gaussian_transform_buffer,
                vec![gs::selection::create_viewport_bundle::<
                    gs::DefaultGaussianPod,
                >(&device)],
            ),
            &viewer.gaussians_buffer,
        )
        .expect("modifier");

        let viewport_selection_bind_group = viewport_selection_modifier.modifier.selection.bundles
            [0]
        .create_bind_group(
            &device,
            // index 0 is the Gaussians buffer, so we use 1,
            // see docs of create_viewport_bundle
            1,
            [
                viewer.camera_buffer.buffer().as_entire_binding(),
                wgpu::BindingResource::TextureView(selector.texture().view()),
            ],
        )
        .expect("bind group");

        viewport_selection_modifier.modifier.selection_expr =
            gs::editor::SelectionExpr::Selection(0, vec![viewport_selection_bind_group]);

        viewport_selection_modifier // Non destructive modifier
            .modifier // Selection modifier
            .modifier // Basic modifier
            .basic_color_modifiers_buffer
            .update_with_pod(
                &queue,
                &gs::editor::BasicColorModifiersPod {
                    rgb_or_hsv: BasicColorRgbOverrideOrHsvModifiersPod::new_rgb_override(
                        Vec3::new(1.0, 1.0, 0.0),
                    ),
                    ..Default::default()
                },
            );

        log::debug!("Creating selection viewport texture overlay renderer");
        let viewport_texture_overlay_renderer =
            utils::selection::ViewportTextureOverlayRenderer::new(
                &device,
                config.view_formats[0],
                selector.texture(),
            );

        log::info!("System initialized");

        Self {
            surface,
            device,
            queue,
            config,

            filter,
            immediate,
            inverted: filter,
            selector_type: None,

            camera,
            gaussians,
            viewer,
            selector,

            viewport_selection_modifier,
            viewport_texture_overlay_renderer,
        }
    }

    fn update(&mut self, input: &core::Input, delta_time: f32) {
        // Toggle selection mode
        if input.pressed_keys.contains(&KeyCode::KeyN) {
            self.selector_type = None;
            log::info!("Selector: None");
        }
        if input.pressed_keys.contains(&KeyCode::KeyR) {
            self.selector_type = Some(gs::selection::ViewportSelectorType::Rectangle);
            log::info!("Selector: Rectangle");
            self.selector.selector_type = gs::selection::ViewportSelectorType::Rectangle;
        }
        if input.pressed_keys.contains(&KeyCode::KeyB) {
            self.selector_type = Some(gs::selection::ViewportSelectorType::Brush);
            log::info!("Selector: Brush");
            self.selector.selector_type = gs::selection::ViewportSelectorType::Brush;
        }
        if input.pressed_keys.contains(&KeyCode::KeyI) {
            self.inverted = !self.inverted;
            log::info!("Inverted: {}", self.inverted);
            if self.filter {
                self.viewer
                    .invert_selection_buffer
                    .update(&self.queue, self.inverted);
            }
        }

        if self.selector_type.is_some() {
            self.update_selection(input, delta_time);
        } else {
            self.update_movement(input, delta_time);
        }

        // Update the viewer
        self.viewer.update_camera(
            &self.queue,
            &self.camera,
            uvec2(self.config.width, self.config.height),
        );
    }

    fn render(&mut self) {
        let texture = match self.surface.get_current_texture() {
            Ok(texture) => texture,
            Err(e) => {
                log::error!("Failed to get current texture: {e:?}");
                return;
            }
        };
        let texture_view = texture.texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Texture View"),
            format: Some(self.config.view_formats[0]),
            ..Default::default()
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Command Encoder"),
            });

        self.viewer.render(&mut encoder, &texture_view);

        if !self.immediate && self.selector_type.is_some() {
            self.viewport_texture_overlay_renderer
                .render(&mut encoder, &texture_view);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        if let Err(e) = self.device.poll(wgpu::PollType::wait_indefinitely()) {
            log::error!("Failed to poll device: {e:?}");
        }
        texture.present();
    }

    fn resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        if size.width > 0 && size.height > 0 {
            self.config.width = size.width;
            self.config.height = size.height;
            self.surface.configure(&self.device, &self.config);

            // Update selector viewport texture
            self.selector
                .resize(&self.device, UVec2::new(size.width, size.height));

            // Update viewport selection bundle
            let viewport_selection_bind_group =
                self.viewport_selection_modifier.modifier.selection.bundles[0]
                    .create_bind_group(
                        &self.device,
                        // index 0 is the Gaussians buffer, so we use 1,
                        // see docs of create_viewport_bundle
                        1,
                        [
                            self.viewer.camera_buffer.buffer().as_entire_binding(),
                            wgpu::BindingResource::TextureView(self.selector.texture().view()),
                        ],
                    )
                    .expect("bind group");

            // Update viewport selection modifier selection expr
            self.viewport_selection_modifier.modifier.selection_expr =
                gs::editor::SelectionExpr::Selection(0, vec![viewport_selection_bind_group]);

            // Update viewport texture overlay renderer
            self.viewport_texture_overlay_renderer
                .update_bind_group(&self.device, self.selector.texture());
        }
    }
}

impl System {
    fn update_selection(&mut self, input: &core::Input, _delta_time: f32) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Command Encoder"),
            });

        if input
            .pressed_mouse
            .contains(&winit::event::MouseButton::Left)
        {
            self.selector.start(&self.queue, input.mouse_pos);
        }

        if input.held_mouse.contains(&winit::event::MouseButton::Left) {
            self.selector.update(&self.queue, input.mouse_pos);

            if self.immediate {
                self.apply_selection(&mut encoder);
            }
        }

        if input
            .released_mouse
            .contains(&winit::event::MouseButton::Left)
        {
            self.apply_selection(&mut encoder);
            self.selector.clear(&mut encoder);
        }

        if input.held_mouse.contains(&winit::event::MouseButton::Left) {
            self.selector.render(&mut encoder);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        if let Err(e) = self.device.poll(wgpu::PollType::wait_indefinitely()) {
            log::error!("Failed to poll device: {e:?}");
        }
    }

    fn apply_selection(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if self.filter {
            // In filter mode, we only evaluate the selection to the viewer's selection buffer
            // and do not modify anything.
            self.viewport_selection_modifier
                .try_apply_with(
                    encoder,
                    &self.viewer.gaussians_buffer,
                    |encoder, modifier, gaussians| {
                        modifier.selection.evaluate(
                            &self.device,
                            encoder,
                            &modifier.selection_expr,
                            &self.viewer.selection_buffer,
                            &self.viewer.model_transform_buffer,
                            &self.viewer.gaussian_transform_buffer,
                            gaussians,
                        );
                    },
                )
                .expect("apply selection modifier");
        } else {
            if self.inverted {
                self.viewport_selection_modifier
                    .modifier
                    .selection_expr
                    .update_with(gs::editor::SelectionExpr::complement);
            }

            self.viewport_selection_modifier.apply(
                &self.device,
                encoder,
                &self.viewer.gaussians_buffer,
                &self.viewer.model_transform_buffer,
                &self.viewer.gaussian_transform_buffer,
            );
        }
    }

    fn update_movement(&mut self, input: &core::Input, delta_time: f32) {
        // Camera movement
        const SPEED: f32 = 1.0;

        let mut forward = 0.0;
        if input.held_keys.contains(&KeyCode::KeyW) {
            forward += SPEED * delta_time;
        }
        if input.held_keys.contains(&KeyCode::KeyS) {
            forward -= SPEED * delta_time;
        }

        let mut right = 0.0;
        if input.held_keys.contains(&KeyCode::KeyD) {
            right += SPEED * delta_time;
        }
        if input.held_keys.contains(&KeyCode::KeyA) {
            right -= SPEED * delta_time;
        }

        self.camera.move_by(forward, right);

        let mut up = 0.0;
        if input.held_keys.contains(&KeyCode::Space) {
            up += SPEED * delta_time;
        }
        if input.held_keys.contains(&KeyCode::ShiftLeft) {
            up -= SPEED * delta_time;
        }

        self.camera.move_up(up);

        // Camera rotation
        const SENSITIVITY: f32 = 0.15;

        let yaw = input.mouse_diff.x * SENSITIVITY * delta_time;
        let pitch = input.mouse_diff.y * SENSITIVITY * delta_time;

        self.camera.pitch_by(-pitch);
        self.camera.yaw_by(-yaw);
    }
}
