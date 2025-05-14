use std::sync::Arc;

use clap::Parser;
use glam::*;
use winit::{error::EventLoopError, event_loop::EventLoop, keyboard::KeyCode, window::Window};

use wgpu_3dgs_viewer as gs;

/// The command line arguments.
#[derive(Parser, Debug)]
#[command(
    version,
    about,
    long_about = "\
    A 3D Gaussian splatting viewer written in Rust using wgpu.\n\
    \n\
    Use W, A, S, D, Space, Shift to move, use mouse to rotate, \
    use scroll wheel to change the size of the mask, \
    use C to toggle between box and ellipsoid masks.\n\
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

    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,

    camera: gs::Camera,
    gaussians: gs::Gaussians,
    viewer: gs::Viewer,

    mask_evaluator: gs::MaskEvaluator,
    mask_gizmo: gs::MaskGizmo,
    mask_shape: gs::MaskShape,
}

impl System {
    fn depth_stencil_state() -> wgpu::DepthStencilState {
        wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }
    }
}

impl gs::bin_core::System for System {
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

        log::debug!("Creating depth texture");
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        log::debug!("Creating depth view");
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Depth Texture View"),
            format: Some(wgpu::TextureFormat::Depth32Float),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::DepthOnly,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
            usage: None,
        });

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
        camera.pos.z -= 1.0;

        log::debug!("Creating viewer");
        let mut viewer = gs::Viewer::new_with(
            &device,
            config.view_formats[0],
            Some(Self::depth_stencil_state()),
            &gaussians,
        )
        .expect("viewer");
        viewer.update_model_transform(&queue, Vec3::ZERO, adjust_quat, Vec3::ONE);
        viewer.update_gaussian_transform(
            &queue,
            1.0,
            gs::GaussianDisplayMode::Splat,
            gs::GaussianShDegree::new(3).expect("SH degree"),
            false,
        );
        viewer.update_selection_highlight(&queue, vec4(1.0, 1.0, 0.0, 0.5));

        log::debug!("Creating mask evaluator");
        let mask_evaluator = gs::MaskEvaluator::new::<gs::DefaultGaussianPod>(&device);

        log::debug!("Creating mask gizmo");
        let mut mask_gizmo = gs::MaskGizmo::new_with(
            &device,
            config.view_formats[0],
            &viewer.camera_buffer,
            Some(Self::depth_stencil_state()),
            Some(Self::depth_stencil_state()),
        );

        log::debug!("Creating mask shape");
        let mask_shape = gs::MaskShape::new(gs::MaskShapeKind::Box);

        log::info!("System initialized");

        mask_evaluator.evaluate(
            &device,
            &queue,
            &gs::MaskOpTree::shape(&mask_shape.to_mask_op_shape_pod()),
            &viewer.mask_buffer,
            &viewer.model_transform_buffer,
            &viewer.gaussians_buffer,
        );

        mask_gizmo.update(
            &device,
            &queue,
            &viewer.camera_buffer,
            mask_shape.kind,
            &[mask_shape.to_mask_gizmo_pod()],
        );

        Self {
            surface,
            device,
            queue,
            config,

            depth_texture,
            depth_view,

            camera,
            gaussians,
            viewer,

            mask_evaluator,
            mask_gizmo,
            mask_shape,
        }
    }

    fn update(&mut self, input: &gs::bin_core::Input, delta_time: f32) {
        // Resize mask
        if input.scroll_diff != 0.0 {
            let scale = 1.0 + input.scroll_diff * 0.1;
            self.mask_shape.scale *= scale;

            self.mask_evaluator.evaluate(
                &self.device,
                &self.queue,
                &gs::MaskOpTree::shape(&self.mask_shape.to_mask_op_shape_pod())
                    .symmetric_difference(gs::MaskOpTree::shape(
                        &gs::MaskShape {
                            pos: self.mask_shape.pos + Vec3::new(0.0, 0.0, 10.0),
                            scale: self.mask_shape.scale * 0.5,
                            ..self.mask_shape.clone()
                        }
                        .to_mask_op_shape_pod(),
                    )),
                &self.viewer.mask_buffer,
                &self.viewer.model_transform_buffer,
                &self.viewer.gaussians_buffer,
            );

            self.mask_gizmo.update(
                &self.device,
                &self.queue,
                &self.viewer.camera_buffer,
                self.mask_shape.kind,
                &[
                    self.mask_shape.to_mask_gizmo_pod(),
                    gs::MaskShape {
                        pos: self.mask_shape.pos + Vec3::new(0.0, 0.0, 10.0),
                        scale: self.mask_shape.scale * 0.5,
                        ..self.mask_shape.clone()
                    }
                    .to_mask_gizmo_pod(),
                ],
            );
        }

        // Toggle mask shape
        if input.pressed_keys.contains(&KeyCode::KeyC) {
            self.mask_shape.kind = match self.mask_shape.kind {
                gs::MaskShapeKind::Box => gs::MaskShapeKind::Ellipsoid,
                gs::MaskShapeKind::Ellipsoid => gs::MaskShapeKind::Box,
            };

            self.mask_evaluator.evaluate(
                &self.device,
                &self.queue,
                &gs::MaskOpTree::shape(&self.mask_shape.to_mask_op_shape_pod())
                    .symmetric_difference(gs::MaskOpTree::shape(
                        &gs::MaskShape {
                            pos: self.mask_shape.pos + Vec3::new(0.0, 0.0, 10.0),
                            scale: self.mask_shape.scale * 0.5,
                            ..self.mask_shape.clone()
                        }
                        .to_mask_op_shape_pod(),
                    )),
                &self.viewer.mask_buffer,
                &self.viewer.model_transform_buffer,
                &self.viewer.gaussians_buffer,
            );

            self.mask_gizmo.update(
                &self.device,
                &self.queue,
                &self.viewer.camera_buffer,
                self.mask_shape.kind,
                &[
                    self.mask_shape.to_mask_gizmo_pod(),
                    gs::MaskShape {
                        pos: self.mask_shape.pos + Vec3::new(0.0, 0.0, 10.0),
                        scale: self.mask_shape.scale * 0.5,
                        ..self.mask_shape.clone()
                    }
                    .to_mask_gizmo_pod(),
                ],
            );
        }

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

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Mask Gizmo Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &texture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            match self.mask_shape.kind {
                gs::MaskShapeKind::Box => self.mask_gizmo.render_box_with_pass(&mut render_pass),
                gs::MaskShapeKind::Ellipsoid => {
                    self.mask_gizmo.render_ellipsoid_with_pass(&mut render_pass)
                }
            }
        }

        self.viewer
            .preprocessor
            .preprocess(&mut encoder, self.gaussians.gaussians.len() as u32);

        self.viewer
            .radix_sorter
            .sort(&mut encoder, &self.viewer.radix_sort_indirect_args_buffer);

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Renderer Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &texture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            self.viewer
                .renderer
                .render_with_pass(&mut render_pass, &self.viewer.indirect_args_buffer);
        }

        self.viewer.postprocessor.postprocess(
            &mut encoder,
            self.gaussians.gaussians.len() as u32,
            &self.viewer.postprocess_indirect_args_buffer,
        );

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

            self.depth_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Depth Texture"),
                size: wgpu::Extent3d {
                    width: self.config.width,
                    height: self.config.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            self.depth_view = self
                .depth_texture
                .create_view(&wgpu::TextureViewDescriptor {
                    label: Some("Depth Texture View"),
                    format: Some(wgpu::TextureFormat::Depth32Float),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    aspect: wgpu::TextureAspect::DepthOnly,
                    base_mip_level: 0,
                    mip_level_count: None,
                    base_array_layer: 0,
                    array_layer_count: None,
                    usage: None,
                });
        }
    }
}
