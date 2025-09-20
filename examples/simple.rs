use std::sync::Arc;

use clap::Parser;
use colored::Colorize;
use glam::*;
use winit::{error::EventLoopError, event_loop::EventLoop, keyboard::KeyCode, window::Window};

use wgpu_3dgs_viewer as gs;

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
    "
)]
struct Args {
    /// Path to the .ply file.
    #[arg(short, long)]
    model: String,

    /// The size of the Gaussians.
    #[arg(long, default_value_t = 1.0)]
    size: f32,

    /// The display mode of the Gaussians.
    #[arg(long, value_enum, default_value_t = DisplayMode::Splat, ignore_case = true)]
    mode: DisplayMode,

    /// The SH degree of the Gaussians.
    #[arg(long, default_value_t = 3, value_parser = clap::value_parser!(u8).range(0..=3))]
    sh_degree: u8,

    /// Whether to hide SH0.
    #[arg(long, default_value_t)]
    no_sh0: bool,

    /// The standard deviation multiplier for the Gaussian size.
    #[arg(long, default_value_t = 3.0)]
    std_dev: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum DisplayMode {
    Splat,
    Ellipse,
    Point,
}

fn main() -> Result<(), EventLoopError> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = Args::parse();

    if !(0.0..=3.0).contains(&args.std_dev) {
        eprintln!(
            "{} invalid value '{}' for '{}': {} is not in 0.0..=3.0\n\nFor more information, try '{}'.",
            "error:".red().bold(),
            args.std_dev.to_string().yellow(),
            "--std-dev <STD_DEV>".bold(),
            args.std_dev,
            "--help".bold()
        );
        std::process::exit(1);
    }

    let event_loop = EventLoop::new()?;
    event_loop.run_app(&mut core::App::<System>::new(args))?;
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
    gaussians: gs::core::Gaussians,
    viewer: gs::Viewer,
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
        let camera = gs::Camera::new(0.1..1e4, 60f32.to_radians());

        log::debug!("Creating viewer");
        let mut viewer =
            gs::Viewer::new(&device, config.view_formats[0], &gaussians).expect("viewer");
        viewer.update_model_transform(
            &queue,
            Vec3::ZERO,
            Quat::from_axis_angle(Vec3::Z, 180f32.to_radians()),
            Vec3::ONE,
        );

        viewer.update_gaussian_transform(
            &queue,
            args.size,
            match args.mode {
                DisplayMode::Splat => gs::core::GaussianDisplayMode::Splat,
                DisplayMode::Ellipse => gs::core::GaussianDisplayMode::Ellipse,
                DisplayMode::Point => gs::core::GaussianDisplayMode::Point,
            },
            gs::core::GaussianShDegree::new(args.sh_degree).expect("sh degree"),
            args.no_sh0,
            args.std_dev,
        );

        log::info!("System initialized");

        Self {
            surface,
            device,
            queue,
            config,

            camera,
            gaussians,
            viewer,
        }
    }

    fn update(&mut self, input: &core::Input, delta_time: f32) {
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
        }
    }
}
