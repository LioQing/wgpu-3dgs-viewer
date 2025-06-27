use crate::{
    CameraBuffer, Error, GaussianPod, GaussianTransformBuffer, GaussiansBuffer, IndirectArgsBuffer,
    IndirectIndicesBuffer, ModelTransformBuffer, wesl_utils,
};

/// A renderer for Gaussians.
#[derive(Debug)]
pub struct Renderer<B = wgpu::BindGroup> {
    /// The bind group layout.
    #[allow(dead_code)]
    bind_group_layout: wgpu::BindGroupLayout,
    /// The bind group.
    bind_group: B,
    /// The render pipeline.
    pipeline: wgpu::RenderPipeline,
}

impl<B> Renderer<B> {
    /// Create the bind group.
    #[allow(clippy::too_many_arguments)]
    pub fn create_bind_group<G: GaussianPod>(
        &self,
        device: &wgpu::Device,
        camera: &CameraBuffer,
        model_transform: &ModelTransformBuffer,
        gaussian_transform: &GaussianTransformBuffer,
        gaussians: &GaussiansBuffer<G>,
        indirect_indices: &IndirectIndicesBuffer,
    ) -> wgpu::BindGroup {
        Renderer::create_bind_group_static(
            device,
            &self.bind_group_layout,
            camera,
            model_transform,
            gaussian_transform,
            gaussians,
            indirect_indices,
        )
    }
}

impl Renderer {
    /// The bind group layout descriptor.
    pub const BIND_GROUP_LAYOUT_DESCRIPTOR: wgpu::BindGroupLayoutDescriptor<'static> =
        wgpu::BindGroupLayoutDescriptor {
            label: Some("Renderer Bind Group Layout"),
            entries: &[
                // Camera uniform buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Model transform uniform buffer
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
                // Gaussian transform uniform buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Gaussian storage buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Indirect indices storage buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        };

    /// Create a new renderer.
    #[allow(clippy::too_many_arguments)]
    pub fn new<G: GaussianPod>(
        device: &wgpu::Device,
        texture_format: wgpu::TextureFormat,
        depth_stencil: Option<wgpu::DepthStencilState>,
        camera: &CameraBuffer,
        model_transform: &ModelTransformBuffer,
        gaussian_transform: &GaussianTransformBuffer,
        gaussians: &GaussiansBuffer<G>,
        indirect_indices: &IndirectIndicesBuffer,
    ) -> Result<Self, Error> {
        if (device.limits().max_storage_buffer_binding_size as u64) < gaussians.buffer().size() {
            return Err(Error::ModelSizeExceedsDeviceLimit {
                model_size: gaussians.buffer().size(),
                device_limit: device.limits().max_storage_buffer_binding_size,
            });
        }

        let this = Renderer::new_without_bind_group::<G>(device, texture_format, depth_stencil);

        log::debug!("Creating renderer bind group");
        let bind_group = this.create_bind_group(
            device,
            camera,
            model_transform,
            gaussian_transform,
            gaussians,
            indirect_indices,
        );

        Ok(Self {
            bind_group_layout: this.bind_group_layout,
            bind_group,
            pipeline: this.pipeline,
        })
    }

    /// Render the scene.
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        indirect_args: &IndirectArgsBuffer,
    ) {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Renderer Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        self.render_with_pass(&mut render_pass, indirect_args);
    }

    /// Render the scene with a [`wgpu::RenderPass`].
    pub fn render_with_pass(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        indirect_args: &IndirectArgsBuffer,
    ) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.draw_indirect(indirect_args.buffer(), 0);
    }

    /// Create the bind group statically.
    #[allow(clippy::too_many_arguments)]
    fn create_bind_group_static<G: GaussianPod>(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        camera: &CameraBuffer,
        model_transform: &ModelTransformBuffer,
        gaussian_transform: &GaussianTransformBuffer,
        gaussians: &GaussiansBuffer<G>,
        indirect_indices: &IndirectIndicesBuffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Renderer Bind Group"),
            layout: bind_group_layout,
            entries: &[
                // Camera uniform buffer
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera.buffer().as_entire_binding(),
                },
                // Model transform uniform buffer
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: model_transform.buffer().as_entire_binding(),
                },
                // Gaussian transform uniform buffer
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: gaussian_transform.buffer().as_entire_binding(),
                },
                // Gaussian storage buffer
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: gaussians.buffer().as_entire_binding(),
                },
                // Indirect indices storage buffer
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: indirect_indices.buffer().as_entire_binding(),
                },
            ],
        })
    }
}

impl Renderer<()> {
    /// Create a new renderer without internally managed bind group.
    ///
    /// To create a bind group with layout matched to this renderer, use the
    /// [`Renderer::create_bind_group`] method.
    pub fn new_without_bind_group<G: GaussianPod>(
        device: &wgpu::Device,
        texture_format: wgpu::TextureFormat,
        depth_stencil: Option<wgpu::DepthStencilState>,
    ) -> Self {
        log::debug!("Creating renderer bind group layout");
        let bind_group_layout =
            device.create_bind_group_layout(&Renderer::BIND_GROUP_LAYOUT_DESCRIPTOR);

        log::debug!("Creating renderer pipeline layout");
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Renderer Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        log::debug!("Creating renderer shader");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Renderer Shader"),
            source: wgpu::ShaderSource::Wgsl(
                wesl_utils::compiler(G::features())
                    .compile("render")
                    .inspect_err(|e| log::error!("{e}"))
                    .unwrap()
                    .to_string()
                    .into(),
            ),
        });

        log::debug!("Creating renderer pipeline");
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Renderer Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vert_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("frag_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: texture_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        log::info!("Renderer created");

        Self {
            bind_group_layout,
            bind_group: (),
            pipeline,
        }
    }

    /// Render the scene.
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        bind_group: &wgpu::BindGroup,
        indirect_args: &IndirectArgsBuffer,
    ) {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Renderer Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        self.render_with_pass(&mut render_pass, bind_group, indirect_args);
    }

    /// Render the scene with a [`wgpu::RenderPass`].
    pub fn render_with_pass(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        bind_group: &wgpu::BindGroup,
        indirect_args: &IndirectArgsBuffer,
    ) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.draw_indirect(indirect_args.buffer(), 0);
    }
}
