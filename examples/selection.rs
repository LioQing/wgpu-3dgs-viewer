use std::sync::Arc;

use clap::Parser;
use glam::*;
use winit::{error::EventLoopError, event_loop::EventLoop, keyboard::KeyCode, window::Window};

use wgpu_3dgs_viewer::{self as gs, core::BufferWrapper, editor::Modifier};

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
    Use C to toggle selection mode.\n\
    Use Left Click to draw rectangle selection.\n\
    Use Right Click to draw brush selection.\n\
    "
)]
struct Args {
    /// Path to the .ply file.
    #[arg(short, long)]
    model: String,
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

    selection_mode: bool,

    camera: gs::Camera,
    gaussians: gs::core::Gaussians,
    viewer: gs::Viewer,
    selector: gs::selection::ViewportSelector,

    viewport_selection_modifier: gs::editor::NonDestructiveModifier<
        gs::DefaultGaussianPod,
        gs::editor::BasicSelectionModifier,
    >,
    viewport_texture_overlay_renderer: utils::selection::ViewportTextureOverlayRenderer,
}

impl core::System for System {
    type Args = Args;

    async fn init(window: Arc<Window>, args: &Args) -> Self {
        let model_path = &args.model;
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
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
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
        let f = std::fs::File::open(model_path).expect("ply file");
        let mut reader = std::io::BufReader::new(f);
        let gaussians = gs::core::Gaussians::read_ply(&mut reader).expect("gaussians");

        log::debug!("Creating camera");
        let adjust_quat = Quat::from_axis_angle(Vec3::Z, 180f32.to_radians());
        let mut camera = gs::Camera::new(0.1..1e4, 60f32.to_radians());
        camera.pos = gaussians
            .gaussians
            .iter()
            .map(|g| adjust_quat * g.pos)
            .sum::<Vec3>()
            / gaussians.gaussians.len() as f32;
        camera.pos.z -= 1.0;

        log::debug!("Creating viewer");
        let mut viewer = gs::Viewer::new_with(
            &device,
            config.view_formats[0],
            None,
            gs::core::GaussiansBuffer::<gs::DefaultGaussianPod>::DEFAULT_USAGE
                | wgpu::BufferUsages::COPY_SRC,
            &gaussians,
        )
        .expect("viewer");
        viewer.update_model_transform(&queue, Vec3::ZERO, adjust_quat, Vec3::ONE);
        viewer.update_gaussian_transform(
            &queue,
            1.0,
            gs::core::GaussianDisplayMode::Splat,
            gs::core::GaussianShDegree::new(3).expect("SH degree"),
            false,
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

        log::debug!("Creating selection viewport selection compute bundle");
        let viewport_selection_compute_bundle =
            gs::selection::create_viewport_bundle::<gs::DefaultGaussianPod>(&device);

        log::debug!("Creating selection viewport selection modifier");
        let mut viewport_selection_modifier = gs::editor::NonDestructiveModifier::new(
            &device,
            &queue,
            gs::editor::BasicSelectionModifier::new(
                &device,
                &viewer.gaussians_buffer,
                &viewer.model_transform_buffer,
                &viewer.gaussian_transform_buffer,
                vec![viewport_selection_compute_bundle],
            ),
            &viewer.gaussians_buffer,
        )
        .expect("modifier");

        let viewport_selection_bind_group = viewport_selection_modifier.modifier.selection.bundles
            [0]
        .create_bind_group(
            &device,
            // index 0 is the Gaussians buffer, so we use 1,
            // see docs of create_sphere_bundle or create_box_bundle
            1,
            [
                viewer.camera_buffer.buffer().as_entire_binding(),
                wgpu::BindingResource::TextureView(selector.texture().view()),
            ],
        )
        .expect("bind group");

        viewport_selection_modifier.modifier.selection_expr =
            gs::editor::SelectionExpr::Selection(0, vec![viewport_selection_bind_group]);

        viewport_selection_modifier
            .modifier
            .basic_color_modifiers_buffer
            .update_with_pod(
                &queue,
                &gs::editor::BasicColorModifiersPod {
                    alpha: 0.0,
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

            selection_mode: false,

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
        if input.pressed_keys.contains(&KeyCode::KeyC) {
            self.selection_mode = !self.selection_mode;
            log::info!(
                "Selection mode {}",
                if self.selection_mode {
                    "enabled"
                } else {
                    "disabled"
                }
            );
        }

        if self.selection_mode {
            self.update_selection(input, delta_time);
        } else {
            self.update_movement(input, delta_time);
        }
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

        if self.selection_mode {
            self.viewport_texture_overlay_renderer
                .render(&mut encoder, &texture_view);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        if let Err(e) = self.device.poll(wgpu::PollType::Wait) {
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
                        // see docs of create_sphere_bundle or create_box_bundle
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
        }

        if input
            .released_mouse
            .contains(&winit::event::MouseButton::Left)
        {
            self.viewport_selection_modifier.apply(
                &self.device,
                &mut encoder,
                &self.viewer.gaussians_buffer,
                &self.viewer.model_transform_buffer,
                &self.viewer.gaussian_transform_buffer,
            );

            self.selector.clear(&mut encoder);
        }

        if input.held_mouse.contains(&winit::event::MouseButton::Left) {
            self.selector.render(&mut encoder);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        if let Err(e) = self.device.poll(wgpu::PollType::Wait) {
            log::error!("Failed to poll device: {e:?}");
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

        // Update the viewer
        self.viewer.update_camera(
            &self.queue,
            &self.camera,
            uvec2(self.config.width, self.config.height),
        );
    }
}
