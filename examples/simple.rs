//! This example renders a simple 3D Gaussian splatting model.
//!
//! For example, to render the model with a size of 1.2 and max standard deviation of 2.0, run:
//!
//! ```sh
//! cargo run --example simple -- -m "path/to/model.ply" --size 1.2 --std-dev 2.0
//! ```
//!
//! To view more options and the controls, run with `--help`:
//!
//! ```sh
//! cargo run --example simple -- --help
//! ```

use std::sync::Arc;

use clap::Parser;
use colored::Colorize;
use glam::*;
use winit::{error::EventLoopError, event_loop::EventLoop, keyboard::KeyCode, window::Window};

use wgpu_3dgs_viewer as gs;
use wgpu_3dgs_viewer::core::{GaussianMaxStdDev, GaussiansSource};

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

    /// The max standard deviation for the Gaussians.
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
    core: core::WgpuCore,

    camera: gs::Camera,
    gaussians: gs::core::Gaussians,
    viewer: gs::Viewer,
}

impl core::System for System {
    type Args = Args;

    async fn init(window: Arc<Window>, args: &Args) -> Self {
        let model_path = &args.model;

        log::debug!("Creating wgpu core");
        let core = core::WgpuCore::new(window).await;

        log::debug!("Creating gaussians");
        let gaussians = [GaussiansSource::Ply, GaussiansSource::Spz]
            .into_iter()
            .find_map(|source| gs::core::Gaussians::read_from_file(model_path, source).ok())
            .expect("gaussians");

        log::debug!("Creating camera");
        let camera = gs::Camera::new(0.1..1e4, 60f32.to_radians());

        log::debug!("Creating viewer");
        let mut viewer =
            gs::Viewer::new(&core.device, core.config.view_formats[0], &gaussians).expect("viewer");
        viewer.update_model_transform(
            &core.queue,
            Vec3::ZERO,
            Quat::from_axis_angle(Vec3::Z, 180f32.to_radians()),
            Vec3::ONE,
        );

        viewer.update_gaussian_transform(
            &core.queue,
            args.size,
            match args.mode {
                DisplayMode::Splat => gs::core::GaussianDisplayMode::Splat,
                DisplayMode::Ellipse => gs::core::GaussianDisplayMode::Ellipse,
                DisplayMode::Point => gs::core::GaussianDisplayMode::Point,
            },
            gs::core::GaussianShDegree::new(args.sh_degree).expect("sh degree"),
            args.no_sh0,
            GaussianMaxStdDev::new(args.std_dev).expect("max std dev"),
        );

        log::info!("System initialized");

        Self {
            core,

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
            &self.core.queue,
            &self.camera,
            uvec2(self.core.config.width, self.core.config.height),
        );
    }

    fn render(&mut self) {
        #[cfg(not(coverage))]
        let (surface_texture, texture) = {
            let surface_texture = match self.core.surface.get_current_texture() {
                Ok(texture) => texture,
                Err(e) => {
                    log::error!("Failed to get current texture: {e:?}");
                    return;
                }
            };
            let texture = surface_texture.texture.clone();

            (surface_texture, texture)
        };
        #[cfg(coverage)]
        let texture = self.core.get_current_texture_for_coverage();

        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Texture View"),
            format: Some(self.core.config.view_formats[0]),
            ..Default::default()
        });

        let mut encoder =
            self.core
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Command Encoder"),
                });

        self.viewer.render(&mut encoder, &texture_view);

        self.core.queue.submit(std::iter::once(encoder.finish()));
        if let Err(e) = self.core.device.poll(wgpu::PollType::wait_indefinitely()) {
            log::error!("Failed to poll device: {e:?}");
        }
        #[cfg(not(coverage))]
        surface_texture.present();
    }

    fn resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        self.core.config.width = size.width;
        self.core.config.height = size.height;
        #[cfg(not(coverage))]
        self.core
            .surface
            .configure(&self.core.device, &self.core.config);
    }
}
