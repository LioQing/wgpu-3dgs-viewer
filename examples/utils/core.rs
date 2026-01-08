use std::{collections::HashSet, sync::Arc};

use clap::Parser;
use glam::*;
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, ElementState, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

/// The input state.
pub struct Input {
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
    pub fn new() -> Self {
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

impl Default for Input {
    fn default() -> Self {
        Self::new()
    }
}

/// The system for application.
pub trait System {
    /// The arguments.
    type Args: Parser + Send + Sync + 'static;

    /// Initialize the system.
    #[allow(async_fn_in_trait)]
    async fn init(window: Arc<Window>, args: &Self::Args) -> Self
    where
        Self: Sized;

    /// Update the system.
    fn update(&mut self, input: &Input, delta_time: f32);

    /// Render the system.
    fn render(&mut self);

    /// Resize the system.
    fn resize(&mut self, _size: winit::dpi::PhysicalSize<u32>) {
        log::warn!("Resize not implemented for this system");
    }
}

/// The application.
pub struct App<S: System> {
    input: Input,
    system: Option<S>,
    window: Option<Arc<Window>>,
    args: S::Args,
    timer: std::time::SystemTime,
    startup_time: std::time::SystemTime,
}

impl<S: System> App<S> {
    pub fn new(args: S::Args) -> Self {
        Self {
            input: Input::new(),
            system: None,
            window: None,
            args,
            timer: std::time::SystemTime::now(),
            startup_time: std::time::SystemTime::now(),
        }
    }
}

impl<S: System> ApplicationHandler for App<S> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        log::debug!("Creating window");
        self.window = Some(Arc::new(
            event_loop
                .create_window(Window::default_attributes())
                .expect("window"),
        ));

        log::debug!("Creating system");
        self.system = Some(pollster::block_on(S::init(
            self.window.as_ref().expect("window").clone(),
            &self.args,
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

                // Override input in coverage mode.
                if cfg!(coverage) {
                    self.input.pressed_keys.clear();
                    self.input.held_keys.clear();
                    self.input.held_keys.insert(KeyCode::KeyS);
                    self.input.held_keys.insert(KeyCode::ShiftLeft);
                    self.input.scroll_diff = 0.0;
                    self.input.pressed_mouse.clear();
                    self.input.held_mouse.clear();
                    self.input.held_mouse.insert(MouseButton::Left);
                    self.input.released_mouse.clear();
                    self.input.mouse_diff = Vec2::new(0.0, -1.0);
                    if self.input.mouse_pos == Vec2::ZERO {
                        self.input.mouse_pos = vec2(400.0, 300.0);
                    }
                }

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

                // Exit after 1 second in coverage mode
                if cfg!(coverage) && self.startup_time.elapsed().expect("elapsed").as_secs() > 1 {
                    log::info!(
                        "The application has been running for more than 1 second, exiting for coverage."
                    );
                    event_loop.exit();
                }
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

pub struct WgpuCore {
    #[cfg(not(coverage))]
    pub surface: wgpu::Surface<'static>,
    pub queue: wgpu::Queue,
    pub device: wgpu::Device,
    pub config: wgpu::SurfaceConfiguration,
}

impl WgpuCore {
    pub async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();

        log::debug!("Creating wgpu instance");
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

        #[cfg(not(coverage))]
        let surface = {
            log::debug!("Creating window surface");
            let surface = instance.create_surface(window.clone()).expect("surface");
            Some(surface)
        };
        #[cfg(coverage)]
        let surface = None;

        log::debug!("Requesting adapter");
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: surface.as_ref(),
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

        let surface_caps = surface
            .as_ref()
            .map(|s| s.get_capabilities(&adapter))
            .unwrap_or_else(|| wgpu::SurfaceCapabilities {
                formats: vec![wgpu::TextureFormat::Rgba8Unorm],
                present_modes: vec![wgpu::PresentMode::Fifo],
                alpha_modes: vec![wgpu::CompositeAlphaMode::Opaque],
                ..Default::default()
            });
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
        if let Some(surface) = &surface {
            surface.configure(&device, &config);
        }

        Self {
            #[cfg(not(coverage))]
            surface: surface.expect("surface"),
            queue,
            device,
            config,
        }
    }

    #[cfg(coverage)]
    pub fn get_current_texture_for_coverage(&self) -> wgpu::Texture {
        self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Coverage Texture"),
            size: wgpu::Extent3d {
                width: self.config.width,
                height: self.config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        })
    }
}
