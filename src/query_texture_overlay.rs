use glam::*;
use wgpu::util::DeviceExt;

use crate::{QueryTexture, Texture};

/// The query texture overlay.
///
/// This allows rendering the query texture as an overlay.
///
/// This requires the `query-texture-overlay` feature.
#[derive(Debug)]
pub struct QueryTextureOverlay {
    /// The sampler.
    sampler: wgpu::Sampler,
    /// The overlay color buffer.
    color_buffer: wgpu::Buffer,
    /// The bind group layout.
    bind_group_layout: wgpu::BindGroupLayout,
    /// The bind group.
    bind_group: wgpu::BindGroup,
    /// The pipeline.
    pipeline: wgpu::RenderPipeline,
}

impl QueryTextureOverlay {
    /// The bind group layout descriptor.
    pub const BIND_GROUP_LAYOUT_DESCRIPTOR: wgpu::BindGroupLayoutDescriptor<'static> =
        wgpu::BindGroupLayoutDescriptor {
            label: Some("Query Texture Overlay Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        };

    /// Create a new query texture overlay.
    pub fn new(
        device: &wgpu::Device,
        texture_format: wgpu::TextureFormat,
        query_texture: &QueryTexture,
    ) -> Self {
        Self::new_with(device, texture_format, None, query_texture)
    }

    /// Create a new query texture overlay with all options.
    pub fn new_with(
        device: &wgpu::Device,
        texture_format: wgpu::TextureFormat,
        depth_stencil: Option<wgpu::DepthStencilState>,
        query_texture: &QueryTexture,
    ) -> Self {
        log::debug!("Creating query texture overlay sampler");
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Query Texture Overlay Sampler"),
            ..Default::default()
        });

        log::debug!("Creating query texture overlay color buffer");
        let color_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Query Texture Overlay Color Buffer"),
            contents: bytemuck::cast_slice(&[vec4(1.0, 1.0, 1.0, 0.5)]),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
        });

        log::debug!("Creating query texture overlay bind group layout");
        let bind_group_layout =
            device.create_bind_group_layout(&Self::BIND_GROUP_LAYOUT_DESCRIPTOR);

        log::debug!("Creating query texture overlay bind group");
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Query Texture Overlay Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(query_texture.view()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: color_buffer.as_entire_binding(),
                },
            ],
        });

        log::debug!("Creating query texture overlay pipeline layout");
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Query Texture Pipeline Overlay Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        log::debug!("Creating query texture overlay shader");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Query Texture Overlay Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shader/query_texture_overlay.wgsl").into(),
            ),
        });

        log::debug!("Creating query texture overlay pipeline");
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Query Texture Overlay Pipeline"),
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
            depth_stencil,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self {
            sampler,
            color_buffer,
            bind_group_layout,
            bind_group,
            pipeline,
        }
    }

    /// Update the color.
    pub fn update_color(&self, queue: &wgpu::Queue, color: Vec4) {
        queue.write_buffer(&self.color_buffer, 0, bytemuck::cast_slice(&[color]));
    }

    /// Update the bind group.
    ///
    /// This is specifically for updating the query texture size.
    pub fn update_bind_group(&mut self, device: &wgpu::Device, query_texture: &QueryTexture) {
        self.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Query Texture Overlay Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(query_texture.view()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.color_buffer.as_entire_binding(),
                },
            ],
        });
    }

    /// Render the query texture overlay.
    pub fn render(&self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView) {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Query Texture Overlay Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
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

        self.render_with_pass(&mut render_pass);
    }

    /// Render the query texture overlay with a [`wgpu::RenderPass`].
    pub fn render_with_pass(&self, pass: &mut wgpu::RenderPass<'_>) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}
