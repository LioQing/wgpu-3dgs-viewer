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
}

impl<S: System> App<S> {
    pub fn new(args: S::Args) -> Self {
        Self {
            input: Input::new(),
            system: None,
            window: None,
            args,
            timer: std::time::SystemTime::now(),
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

        self.window.as_ref().unwrap().set_cursor_visible(false);

        log::debug!("Creating system");
        self.system = Some(futures::executor::block_on(S::init(
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
