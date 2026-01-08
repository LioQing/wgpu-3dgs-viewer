//! This example renders multiple models in the order of their centroids' distance to the camera using the `multi-model` feature.
//!
//! For example, to use an offset of (10, 0, 0) between each model, run:
//!
//! ```sh
//! cargo run --example multi-model --features="multi-model" -- -m "path/to/model1.ply" -m "path/to/model2.ply" --offset 10.0,0.0,0.0
//! ```
//!
//! To view more options and the controls, run with `--help`:
//!
//! ```sh
//! cargo run --example multi-model --features="multi-model" -- --help
//! ```

use std::sync::Arc;

use clap::Parser;
use glam::*;
use winit::{error::EventLoopError, event_loop::EventLoop, keyboard::KeyCode, window::Window};

use wgpu_3dgs_viewer as gs;
use wgpu_3dgs_viewer::core::{GaussiansSource, IterGaussian};

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
    event_loop.run_app(&mut core::App::<System>::new(Args::parse()))?;
    Ok(())
}

/// The application system.
#[allow(dead_code)]
struct System {
    core: core::WgpuCore,

    camera: gs::Camera,
    gaussians: Vec<gs::core::Gaussians>,
    gaussian_centroids: Vec<Vec3>,
    viewer: gs::MultiModelViewer<gs::DefaultGaussianPod, usize>,
}

impl core::System for System {
    type Args = Args;

    async fn init(window: Arc<Window>, args: &Args) -> Self {
        let model_paths = &args.models;
        let model_offset = Vec3::from_slice(&args.offset);

        log::debug!("Creating wgpu core");
        let core = core::WgpuCore::new(window).await;

        log::debug!("Creating gaussians");
        let gaussians = model_paths
            .iter()
            .map(|model_path| {
                log::debug!("Reading model from {model_path}");
                [GaussiansSource::Ply, GaussiansSource::Spz]
                    .into_iter()
                    .find_map(|source| gs::core::Gaussians::read_from_file(model_path, source).ok())
                    .expect("gaussians")
            })
            .collect::<Vec<_>>();

        log::debug!("Computing gaussian centroids");
        let mut gaussian_centroids = gaussians
            .iter()
            .map(|g| {
                let mut centroid = Vec3::ZERO;
                for gaussian in g.iter_gaussian() {
                    centroid += gaussian.pos;
                }
                centroid / g.len() as f32
            })
            .collect::<Vec<_>>();

        log::debug!("Creating camera");
        let camera = gs::Camera::new(0.1..1e4, 60f32.to_radians());

        log::debug!("Creating viewer");
        let mut viewer =
            gs::MultiModelViewer::new(&core.device, core.config.view_formats[0]).expect("viewer");

        let quat = Quat::from_axis_angle(Vec3::Z, 180f32.to_radians());
        for (i, gaussians) in gaussians.iter().enumerate() {
            let offset = model_offset * i as f32;

            log::debug!("Pushing model {i}");

            viewer.insert_model(&core.device, i, gaussians);
            viewer
                .update_model_transform(&core.queue, &i, offset, quat, Vec3::ONE)
                .expect("update model");

            gaussian_centroids[i] = quat.mul_vec3(gaussian_centroids[i]) + offset;
        }

        log::info!("System initialized");

        Self {
            core,

            camera,
            gaussians,
            gaussian_centroids,
            viewer,
        }
    }

    fn update(&mut self, input: &core::Input, delta_time: f32) {
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

        self.core.queue.submit(std::iter::once(encoder.finish()));
        if let Err(e) = self.core.device.poll(wgpu::PollType::wait_indefinitely()) {
            log::error!("Failed to poll device: {e:?}");
        }
        #[cfg(not(coverage))]
        surface_texture.present();
    }

    fn resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        if size.width > 0 && size.height > 0 {
            self.core.config.width = size.width;
            self.core.config.height = size.height;
            #[cfg(not(coverage))]
            self.core
                .surface
                .configure(&self.core.device, &self.core.config);
        }
    }
}
