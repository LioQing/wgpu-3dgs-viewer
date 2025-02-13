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
    In default mode, move the camera with W, A, S, D, Space, Shift, and rotate with mouse.\n\
    In selection mode, use left mouse button to brush select, \
    use right mouse button to box select.\n\
    Use C to toggle between default and selection mode.\
    "
)]
struct Args {
    /// Path to the .ply file.
    #[arg(short, long)]
    model: String,
}

fn main() -> Result<(), EventLoopError> {
    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "info");
    }
    env_logger::init();

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
            &self.args.model,
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
struct System {
    surface: wgpu::Surface<'static>,
    queue: wgpu::Queue,
    device: wgpu::Device,
    config: wgpu::SurfaceConfiguration,

    camera: gs::Camera,
    gaussians: gs::Gaussians,
    viewer: gs::Viewer,

    is_selecting: bool,
    query: gs::QueryPod,
    selection_rect_start: Vec2,
    selection_brush_radius: u32,
    selection_bind_group: wgpu::BindGroup,
    selection_pipeline: wgpu::RenderPipeline,
}

impl System {
    pub async fn new(window: Arc<Window>, model_path: &str) -> Self {
        let size = window.inner_size();

        log::debug!("Creating wgpu instance");
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());

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
        let f = std::fs::File::open(model_path).expect("ply file");
        let mut reader = std::io::BufReader::new(f);
        let gaussians = gs::Gaussians::read_ply(&mut reader).expect("gaussians");

        log::debug!("Creating camera");
        let adjust_quat = Quat::from_axis_angle(Vec3::Z, 180f32.to_radians());
        let mut camera = gs::Camera::new(0.1..1e4, 60f32.to_radians());
        camera.pos = gaussians
            .gaussians
            .iter()
            .map(|g| adjust_quat * g.pos)
            .sum::<Vec3>()
            / gaussians.gaussians.len() as f32;
        camera.pos.z += gaussians
            .gaussians
            .iter()
            .map(|g| (adjust_quat * g.pos).z - camera.pos.z)
            .fold(f32::INFINITY, |a, b| a.min(b));

        log::debug!("Creating viewer");
        let mut viewer =
            gs::Viewer::new(&device, config.view_formats[0], &gaussians).expect("viewer");
        viewer.update_model_transform(&queue, Vec3::ZERO, adjust_quat, Vec3::ONE);
        viewer.update_gaussian_transform(
            &queue,
            1.0,
            gs::GaussianDisplayMode::Splat,
            gs::GaussianShDegree::new(3).expect("SH degree"),
            false,
        );
        viewer.update_selection_highlight(&queue, vec4(1.0, 1.0, 0.0, 0.5));

        log::debug!("Creating selection bind group layout");
        let selection_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Selection Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        log::debug!("Creating selection bind group");
        let selection_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Selection Bind Group"),
            layout: &selection_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: viewer.query_buffer.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: viewer.camera_buffer.buffer().as_entire_binding(),
                },
            ],
        });

        log::debug!("Creating selection pipeline");
        let selection_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Selection Pipeline Layout"),
                bind_group_layouts: &[&selection_bind_group_layout],
                push_constant_ranges: &[],
            });

        let selection_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Selection Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("selection.wgsl").into()),
        });

        let selection_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Selection Pipeline"),
            layout: Some(&selection_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &selection_shader,
                entry_point: Some("vert_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &selection_shader,
                entry_point: Some("frag_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.view_formats[0],
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        log::info!("System initialized");

        Self {
            surface,
            device,
            queue,
            config,

            camera,
            gaussians,
            viewer,

            is_selecting: false,
            query: gs::QueryPod::none(),
            selection_rect_start: Vec2::ZERO,
            selection_brush_radius: 40,
            selection_bind_group,
            selection_pipeline,
        }
    }

    pub fn update(&mut self, input: &Input, delta_time: f32) {
        if input.pressed_keys.contains(&KeyCode::KeyC) {
            self.is_selecting = !self.is_selecting;
            self.query = gs::QueryPod::none();
        }

        if self.is_selecting {
            // Selection brush
            if self.query.query_type() == gs::QueryType::None
                && input.pressed_mouse.contains(&MouseButton::Left)
            {
                let selection_op = if input.held_keys.contains(&KeyCode::ShiftLeft) {
                    gs::QuerySelectionOp::Add
                } else if input.held_keys.contains(&KeyCode::ControlLeft) {
                    gs::QuerySelectionOp::Remove
                } else {
                    gs::QuerySelectionOp::Set
                };

                self.query = gs::QueryPod::brush(
                    self.selection_brush_radius,
                    input.mouse_pos,
                    input.mouse_pos,
                )
                .with_selection_op(selection_op);
            } else if self.query.query_type() == gs::QueryType::Brush
                && input.held_mouse.contains(&MouseButton::Left)
            {
                let selection_op = match self.query.query_selection_op() {
                    gs::QuerySelectionOp::Add | gs::QuerySelectionOp::Set => {
                        gs::QuerySelectionOp::Add
                    }
                    gs::QuerySelectionOp::Remove => gs::QuerySelectionOp::Remove,
                    gs::QuerySelectionOp::None => gs::QuerySelectionOp::Set,
                };

                self.query = gs::QueryPod::brush(
                    self.selection_brush_radius,
                    self.query.as_brush().end(),
                    input.mouse_pos,
                )
                .with_selection_op(selection_op);
            }

            if input.scroll_diff != 0.0 {
                self.selection_brush_radius = (self.selection_brush_radius as i32
                    + input.scroll_diff as i32 * 5)
                    .max(1) as u32;
            }

            // Selection rect
            if self.query.query_type() == gs::QueryType::None
                && input.pressed_mouse.contains(&MouseButton::Right)
            {
                let selection_op = if input.held_keys.contains(&KeyCode::ShiftLeft) {
                    gs::QuerySelectionOp::Add
                } else if input.held_keys.contains(&KeyCode::ControlLeft) {
                    gs::QuerySelectionOp::Remove
                } else {
                    gs::QuerySelectionOp::Set
                };

                self.selection_rect_start = input.mouse_pos;
                self.query =
                    gs::QueryPod::rect(self.selection_rect_start, self.selection_rect_start)
                        .with_selection_op(selection_op);
            } else if self.query.query_type() == gs::QueryType::Rect
                && input.held_mouse.contains(&MouseButton::Right)
            {
                let selection_op = match self.query.query_selection_op() {
                    gs::QuerySelectionOp::Add | gs::QuerySelectionOp::Set => {
                        gs::QuerySelectionOp::Add
                    }
                    gs::QuerySelectionOp::Remove => gs::QuerySelectionOp::Remove,
                    gs::QuerySelectionOp::None => gs::QuerySelectionOp::Set,
                };

                let top_left = self.selection_rect_start.min(input.mouse_pos);
                let bottom_right = self.selection_rect_start.max(input.mouse_pos);

                self.query =
                    gs::QueryPod::rect(top_left, bottom_right).with_selection_op(selection_op);
            }

            // Clear query
            if input.held_mouse.is_empty() {
                self.query = gs::QueryPod::none();

                // For selection cursor
                self.query.content_u32.y = self.selection_brush_radius;
                self.query.content_f32.z = input.mouse_pos.x;
                self.query.content_f32.w = input.mouse_pos.y;
            }
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

        // Update the viewer
        self.viewer.update_camera(
            &self.queue,
            &self.camera,
            uvec2(self.config.width, self.config.height),
        );
        self.viewer.update_query(&self.queue, &self.query);
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

        self.viewer.render(
            &mut encoder,
            &texture_view,
            self.gaussians.gaussians.len() as u32,
        );

        if self.is_selecting {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Selection Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &texture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_pipeline(&self.selection_pipeline);
            render_pass.set_bind_group(0, &self.selection_bind_group, &[]);
            render_pass.draw(0..6, 0..1);
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
        }
    }
}
