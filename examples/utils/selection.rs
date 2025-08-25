#![allow(dead_code)]

use wgpu_3dgs_viewer::selection::ViewportTexture;

/// Renderer to render the selection viewport texture as an overlay.
#[derive(Debug)]
pub struct ViewportTextureOverlayRenderer {
    /// The sampler.
    sampler: wgpu::Sampler,
    /// The bind group layout.
    bind_group_layout: wgpu::BindGroupLayout,
    /// The bind group.
    bind_group: wgpu::BindGroup,
    /// The pipeline.
    pipeline: wgpu::RenderPipeline,
}

impl ViewportTextureOverlayRenderer {
    /// The bind group layout descriptor.
    pub const BIND_GROUP_LAYOUT_DESCRIPTOR: wgpu::BindGroupLayoutDescriptor<'static> =
        wgpu::BindGroupLayoutDescriptor {
            label: Some("Selection Viewport Texture Overlay Renderer Bind Group Layout"),
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
            ],
        };

    /// Create a new selection viewport texture overlay renderer.
    pub fn new(
        device: &wgpu::Device,
        texture_format: wgpu::TextureFormat,
        viewport_texture: &ViewportTexture,
    ) -> Self {
        log::debug!("Creating selection viewport texture overlay renderer sampler");
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Selection Viewport Texture Overlay Renderer Sampler"),
            ..Default::default()
        });

        log::debug!("Creating selection viewport texture overlay renderer bind group layout");
        let bind_group_layout =
            device.create_bind_group_layout(&Self::BIND_GROUP_LAYOUT_DESCRIPTOR);

        log::debug!("Creating selection viewport texture overlay renderer bind group");
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Selection Viewport Texture Overlay Renderer Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(viewport_texture.view()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        log::debug!("Creating selection viewport texture overlay renderer pipeline layout");
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Selection Viewport Texture Overlay Renderer Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        log::debug!("Creating selection viewport texture overlay renderer shader");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Selection Viewport Texture Overlay Renderer Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shader/selection/viewport_texture_overlay.wgsl").into(),
            ),
        });

        log::debug!("Creating selection viewport texture overlay renderer pipeline");
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Selection Viewport Texture Overlay Renderer Pipeline"),
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

        Self {
            sampler,
            bind_group_layout,
            bind_group,
            pipeline,
        }
    }

    /// Update the bind group.
    ///
    /// This is specifically for updating the selection viewport texture size.
    pub fn update_bind_group(&mut self, device: &wgpu::Device, viewport_texture: &ViewportTexture) {
        self.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Selection Viewport Texture Overlay Renderer Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(viewport_texture.view()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
    }

    /// Render the selection viewport texture overlay.
    pub fn render(&self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView) {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Selection Viewport Texture Overlay Render Pass"),
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

    /// Render the selection viewport texture overlay with a [`wgpu::RenderPass`].
    pub fn render_with_pass(&self, pass: &mut wgpu::RenderPass<'_>) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}
