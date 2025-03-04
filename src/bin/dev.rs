use std::{collections::HashSet, sync::Arc};

use clap::Parser;
use glam::*;
use winit::{
    application::ApplicationHandler,
    error::EventLoopError,
    event::{DeviceEvent, DeviceId, ElementState, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
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
        default_value = "2.0,0.0,0.0"
    )]
    offset: Vec<f32>,
}

fn main() -> Result<(), EventLoopError> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let event_loop = EventLoop::new()?;
    event_loop.run_app(&mut App::new(Args::parse()))?;
    Ok(())
}

/// The input state.
struct Input {
    pub pressed_keys: HashSet<KeyCode>,
    pub held_keys: HashSet<KeyCode>,
    pub pressed_mouse: HashSet<MouseButton>,
    pub held_mouse: HashSet<MouseButton>,
    pub released_mouse: HashSet<MouseButton>,
    pub scroll_diff: f32,
    pub mouse_diff: Vec2,
    pub mouse_pos: Vec2,
}

impl Input {
    fn new() -> Self {
        Self {
            pressed_keys: HashSet::new(),
            held_keys: HashSet::new(),
            pressed_mouse: HashSet::new(),
            held_mouse: HashSet::new(),
            released_mouse: HashSet::new(),
            scroll_diff: 0.0,
            mouse_diff: Vec2::ZERO,
            mouse_pos: Vec2::ZERO,
        }
    }

    /// Update the input state based on [`DeviceEvent`].
    fn device_event(&mut self, event: &DeviceEvent) {
        match event {
            DeviceEvent::Key(input) => match input.physical_key {
                PhysicalKey::Unidentified(..) => {}
                PhysicalKey::Code(key) => match input.state {
                    ElementState::Pressed => {
                        if !self.held_keys.contains(&key) {
                            self.pressed_keys.insert(key);
                            self.held_keys.insert(key);
                        }
                    }
                    ElementState::Released => {
                        self.held_keys.remove(&key);
                    }
                },
            },
            DeviceEvent::MouseMotion { delta: (x, y) } => {
                self.mouse_diff += vec2(*x as f32, *y as f32);
            }
            _ => {}
        }
    }

    /// Update the input state based on [`WindowEvent`].
    fn window_event(&mut self, event: &WindowEvent) {
        match event {
            WindowEvent::MouseInput { state, button, .. } => match *state {
                ElementState::Pressed => {
                    self.pressed_mouse.insert(*button);
                    self.held_mouse.insert(*button);
                }
                ElementState::Released => {
                    self.released_mouse.insert(*button);
                    self.held_mouse.remove(button);
                }
            },
            WindowEvent::MouseWheel { delta, .. } => match delta {
                winit::event::MouseScrollDelta::LineDelta(_, y) => {
                    self.scroll_diff += y;
                }
                winit::event::MouseScrollDelta::PixelDelta(pos) => {
                    self.scroll_diff += pos.y as f32;
                }
            },
            WindowEvent::CursorMoved { position, .. } => {
                self.mouse_pos = vec2(position.x as f32, position.y as f32);
            }
            _ => {}
        }
    }

    /// Clear states for a new frame.
    fn new_frame(&mut self) {
        self.pressed_keys.clear();
        self.scroll_diff = 0.0;
        self.pressed_mouse.clear();
        self.released_mouse.clear();
        self.mouse_diff = Vec2::ZERO;
    }
}

/// The application.
struct App {
    input: Input,
    system: Option<System>,
    window: Option<Arc<Window>>,
    args: Args,
    timer: std::time::SystemTime,
}

impl App {
    fn new(args: Args) -> Self {
        Self {
            input: Input::new(),
            system: None,
            window: None,
            args,
            timer: std::time::SystemTime::now(),
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        log::debug!("Creating window");
        self.window = Some(Arc::new(
            event_loop
                .create_window(Window::default_attributes())
                .expect("window"),
        ));

        log::debug!("Creating system");
        self.system = Some(futures::executor::block_on(System::new(
            self.window.as_ref().expect("window").clone(),
            &self.args.models,
            Vec3::from_slice(&self.args.offset),
        )));

        event_loop.set_control_flow(ControlFlow::Poll);

        self.timer = std::time::SystemTime::now();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        self.input.window_event(&event);

        match event {
            WindowEvent::RedrawRequested => {
                // Capture frame rate
                let delta_time = self.timer.elapsed().expect("elapsed time").as_secs_f32();
                self.timer = std::time::SystemTime::now();

                // Update system
                self.system
                    .as_mut()
                    .expect("system")
                    .update(&self.input, delta_time);

                self.system.as_mut().expect("system").render();

                // Request redraw
                self.window.as_mut().expect("window").request_redraw();

                // Clear input states
                self.input.new_frame();
            }
            WindowEvent::Resized(size) => {
                log::info!("Window resized to {size:?}");
                self.system.as_mut().expect("system").resize(size);
            }
            WindowEvent::CloseRequested | WindowEvent::Destroyed => {
                log::info!("The application was requested to close, quitting the application");
                event_loop.exit();
            }
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        self.input.device_event(&event);
    }
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
    viewer: gs::MultiModelViewer<gs::GaussianPodWithShNorm8Cov3dHalfConfigs, usize>,

    query_cursor: gs::QueryCursor,
    query_toolset: gs::QueryToolset,
    query_texture_overlay: gs::QueryTextureOverlay,

    is_selecting: bool,
}

impl System {
    pub async fn new(window: Arc<Window>, model_paths: &[String], model_offset: Vec3) -> Self {
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
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: adapter.limits(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
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

    pub fn update(&mut self, input: &Input, delta_time: f32) {
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
            const SENSITIVITY: f32 = 0.3;

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

    pub fn render(&mut self) {
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
        self.device.poll(wgpu::Maintain::Wait);
        texture.present();
    }

    pub fn resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
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
