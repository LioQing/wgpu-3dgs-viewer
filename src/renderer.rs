use crate::{
    CameraBuffer, Error, GaussianCov3dConfig, GaussianPod, GaussianShConfig,
    GaussianTransformBuffer, GaussiansBuffer, GaussiansEditBuffer, IndirectArgsBuffer,
    IndirectIndicesBuffer, ModelTransformBuffer, QueryBuffer, QueryResultCountBuffer,
    QueryResultsBuffer, SelectionBuffer, SelectionHighlightBuffer,
};

/// A renderer for Gaussians.
#[derive(Debug)]
pub struct Renderer {
    /// The bind group layout.
    #[allow(dead_code)]
    bind_group_layout: wgpu::BindGroupLayout,
    /// The bind group.
    bind_group: wgpu::BindGroup,
    /// The render pipeline.
    pipeline: wgpu::RenderPipeline,
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
                // Query uniform buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Query result count storage buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Query results storage buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Selection highlight uniform buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Selection storage buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Gaussians edit storage buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
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
        camera: &CameraBuffer,
        model_transform: &ModelTransformBuffer,
        gaussian_transform: &GaussianTransformBuffer,
        gaussians: &GaussiansBuffer<G>,
        indirect_indices: &IndirectIndicesBuffer,
        query: &QueryBuffer,
        query_result_count: &QueryResultCountBuffer,
        query_results: &QueryResultsBuffer,
        selection_highlight: &SelectionHighlightBuffer,
        selection: &SelectionBuffer,
        gaussians_edit: &GaussiansEditBuffer,
    ) -> Result<Self, Error> {
        if (device.limits().max_storage_buffer_binding_size as u64) < gaussians.buffer().size() {
            return Err(Error::ModelSizeExceedsDeviceLimit {
                model_size: gaussians.buffer().size(),
                device_limit: device.limits().max_storage_buffer_binding_size,
            });
        }

        log::debug!("Creating renderer bind group layout");
        let bind_group_layout =
            device.create_bind_group_layout(&Self::BIND_GROUP_LAYOUT_DESCRIPTOR);

        log::debug!("Creating renderer bind group");
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Renderer Bind Group"),
            layout: &bind_group_layout,
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
                // Query uniform buffer
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: query.buffer().as_entire_binding(),
                },
                // Query result count storage buffer
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: query_result_count.buffer().as_entire_binding(),
                },
                // Query results storage buffer
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: query_results.buffer().as_entire_binding(),
                },
                // Selection highlight uniform buffer
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: selection_highlight.buffer().as_entire_binding(),
                },
                // Selection storage buffer
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: selection.buffer().as_entire_binding(),
                },
                // Gaussians edit storage buffer
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: gaussians_edit.buffer().as_entire_binding(),
                },
            ],
        });

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
                include_str!("shader/render.wgsl")
                    .replace("{{gaussian_sh_field}}", G::ShConfig::sh_field())
                    .replace("{{gaussian_sh_unpack}}", G::ShConfig::sh_unpack())
                    .replace("{{gaussian_cov3d_field}}", G::Cov3dConfig::cov3d_field())
                    .replace("{{gaussian_cov3d_unpack}}", G::Cov3dConfig::cov3d_unpack())
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
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("frag_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: texture_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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

        log::info!("Renderer created");

        Ok(Self {
            bind_group_layout,
            bind_group,
            pipeline,
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
}
