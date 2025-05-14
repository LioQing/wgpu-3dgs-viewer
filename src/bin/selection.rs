use std::sync::Arc;

use clap::Parser;
use glam::*;
use winit::{
    error::EventLoopError, event::MouseButton, event_loop::EventLoop, keyboard::KeyCode,
    window::Window,
};

use wgpu_3dgs_viewer as gs;

/// The command line arguments.
#[derive(Parser, Debug)]
#[command(
    version,
    about,
    long_about = "\
    A 3D Gaussian splatting viewer written in Rust using wgpu.\n\
    \n\
    In default mode, use W, A, S, D, Space, Shift to move, use mouse to rotate.\n\
    In selection mode, use left mouse button to brush select, \
    use right mouse button to box select, \
    hold space to use immediate selection, \
    use delete to detele selected Gaussians.\n\
    Use C to toggle between default and selection mode.\
    "
)]
struct Args {
    /// Path to the .ply file.
    #[arg(short, long, num_args = 1..)]
    models: Vec<String>,

    /// The offset of each model.
    #[arg(
        short,
        long,
        num_args = 3,
        value_delimiter = ',',
        default_value = "10.0,0.0,0.0"
    )]
    offset: Vec<f32>,
}

fn main() -> Result<(), EventLoopError> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let event_loop = EventLoop::new()?;
    event_loop.run_app(&mut gs::bin_core::App::<System>::new(Args::parse()))?;
    Ok(())
}

/// The application system.
#[allow(dead_code)]
struct System {
    surface: wgpu::Surface<'static>,
    queue: wgpu::Queue,
    device: wgpu::Device,
    config: wgpu::SurfaceConfiguration,

    camera: gs::Camera,
    gaussians: Vec<gs::Gaussians>,
    gaussian_centroids: Vec<Vec3>,
    viewer: gs::MultiModelViewer<gs::DefaultGaussianPod, usize>,

    query_cursor: gs::QueryCursor,
    query_toolset: gs::QueryToolset,
    query_texture_overlay: gs::QueryTextureOverlay,

    is_selecting: bool,
}

impl gs::bin_core::System for System {
    type Args = Args;

    async fn init(window: Arc<Window>, args: &Args) -> Self {
        let model_paths = &args.models;
        let model_offset = Vec3::from_slice(&args.offset);
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
        let gaussians = model_paths
            .iter()
            .map(|model_path| {
                log::debug!("Reading model from {model_path}");
                let f = std::fs::File::open(model_path).expect("ply file");
                let mut reader = std::io::BufReader::new(f);
                gs::Gaussians::read_ply(&mut reader).expect("gaussians")
            })
            .collect::<Vec<_>>();

        log::debug!("Computing gaussian centroids");
        let mut gaussian_centroids = gaussians
            .iter()
            .map(|g| {
                let mut centroid = Vec3::ZERO;
                for gaussian in &g.gaussians {
                    centroid += gaussian.pos;
                }
                centroid / g.gaussians.len() as f32
            })
            .collect::<Vec<_>>();

        log::debug!("Creating camera");
        let camera = gs::Camera::new(0.1..1e4, 60f32.to_radians());

        log::debug!("Creating viewer");
        let mut viewer = gs::MultiModelViewer::new(
            &device,
            config.view_formats[0],
            uvec2(config.width, config.height),
        );
        viewer.update_gaussian_transform(
            &queue,
            1.0,
            gs::GaussianDisplayMode::Splat,
            gs::GaussianShDegree::new(3).expect("SH degree"),
            false,
        );
        viewer.update_selection_highlight(&queue, vec4(1.0, 1.0, 0.0, 0.5));

        let adjust_quat = Quat::from_axis_angle(Vec3::Z, 180f32.to_radians());
        for (i, gaussians) in gaussians.iter().enumerate() {
            let offset = model_offset * i as f32;

            log::debug!("Pushing model {i}");

            viewer.insert_model(&device, i, gaussians);
            viewer.update_model_transform(&queue, &i, offset, adjust_quat, Vec3::ONE);

            gaussian_centroids[i] = adjust_quat.mul_vec3(gaussian_centroids[i]) + offset;
        }

        log::debug!("Creating query cursor");
        let query_cursor = gs::QueryCursor::new(
            &device,
            config.view_formats[0],
            &viewer.world_buffers.camera_buffer,
        );

        log::debug!("Creating query toolset");
        let query_toolset = gs::QueryToolset::new(
            &device,
            &viewer.world_buffers.query_texture,
            &viewer.world_buffers.camera_buffer,
        );

        log::debug!("Creating query texture overlay");
        let query_texture_overlay = gs::QueryTextureOverlay::new(
            &device,
            config.view_formats[0],
            &viewer.world_buffers.query_texture,
        );

        log::info!("System initialized");

        Self {
            surface,
            device,
            queue,
            config,

            camera,
            gaussians,
            gaussian_centroids,
            viewer,

            query_cursor,
            query_toolset,
            query_texture_overlay,

            is_selecting: false,
        }
    }

    fn update(&mut self, input: &gs::bin_core::Input, delta_time: f32) {
        if input.pressed_keys.contains(&KeyCode::KeyC) {
            self.is_selecting = !self.is_selecting;
            self.viewer.update_query(&self.queue, &gs::QueryPod::none());
        }

        if self.is_selecting {
            self.query_toolset
                .set_use_texture(!input.held_keys.contains(&KeyCode::Space));

            let selection_op = if input.held_keys.contains(&KeyCode::ShiftLeft) {
                gs::QuerySelectionOp::Add
            } else if input.held_keys.contains(&KeyCode::ControlLeft) {
                gs::QuerySelectionOp::Remove
            } else {
                gs::QuerySelectionOp::Set
            };

            if input.pressed_mouse.contains(&MouseButton::Left) {
                self.query_toolset.start(
                    gs::QueryToolsetTool::Brush,
                    selection_op,
                    input.mouse_pos,
                );
            } else if input.pressed_mouse.contains(&MouseButton::Right) {
                self.query_toolset
                    .start(gs::QueryToolsetTool::Rect, selection_op, input.mouse_pos);
            } else if input.released_mouse.contains(&MouseButton::Left)
                || input.released_mouse.contains(&MouseButton::Right)
            {
                self.query_toolset.end();
            } else {
                self.query_toolset.update_pos(input.mouse_pos);
            }

            if input.scroll_diff != 0.0 {
                let new_brush_radius = (self.query_toolset.brush_radius() as i32
                    + input.scroll_diff as i32 * 5)
                    .max(1) as u32;
                self.query_toolset.update_brush_radius(new_brush_radius);
            }

            self.viewer
                .update_query(&self.queue, self.query_toolset.query());

            self.query_cursor.update_query_toolset(
                &self.queue,
                &self.query_toolset,
                input.mouse_pos,
            );
        } else {
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

        // Selection edit
        if input.pressed_keys.contains(&KeyCode::Delete) {
            self.viewer.update_selection_edit(
                &self.queue,
                gs::GaussianEditFlag::ENABLED | gs::GaussianEditFlag::HIDDEN,
                Vec3::ZERO,
                0.0,
                0.0,
                0.0,
                0.0,
            );
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

        self.query_toolset.render(
            &self.queue,
            &mut encoder,
            &self.viewer.world_buffers.query_texture,
        );

        let mut render_keys = self
            .gaussian_centroids
            .iter()
            .enumerate()
            .map(|(i, centroid)| (i, centroid - self.camera.pos))
            .collect::<Vec<_>>();

        render_keys.sort_by(|(_, a), (_, b)| {
            a.length()
                .partial_cmp(&b.length())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let render_keys = render_keys
            .into_iter()
            .rev()
            .map(|(i, _)| i)
            .collect::<Vec<_>>();

        self.viewer
            .render(
                &mut encoder,
                &texture_view,
                render_keys.iter().collect::<Vec<_>>().as_slice(),
            )
            .expect("render");

        if self.is_selecting {
            if let Some((gs::QueryToolsetUsedTool::QueryTextureTool { .. }, ..)) =
                self.query_toolset.state()
            {
                self.query_texture_overlay
                    .render(&mut encoder, &texture_view);
            } else {
                self.query_cursor.render(&mut encoder, &texture_view);
            }
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
            self.viewer
                .update_query_texture_size(&self.device, uvec2(size.width, size.height));
            self.query_texture_overlay
                .update_bind_group(&self.device, &self.viewer.world_buffers.query_texture);
        }
    }
}
