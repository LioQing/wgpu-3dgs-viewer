use crate::{
    CameraBuffer, RendererCreateError,
    core::BufferWrapper,
    selection::{ViewportTexture, ViewportTextureF32Buffer, ViewportTexturePosBuffer},
    wesl_utils,
};

/// A renderer for applying a brush selection to [`ViewportTexture`].
#[derive(Debug)]
pub struct ViewportTextureBrushRenderer<B = wgpu::BindGroup> {
    /// The bind group layout.
    #[allow(dead_code)]
    bind_group_layout: wgpu::BindGroupLayout,
    /// The bind group.
    bind_group: B,
    /// The render pipeline.
    pipeline: wgpu::RenderPipeline,
}

impl<B> ViewportTextureBrushRenderer<B> {
    /// Create the bind group.
    pub fn create_bind_group(
        &self,
        device: &wgpu::Device,
        camera: &CameraBuffer,
        start: &ViewportTexturePosBuffer,
        end: &ViewportTexturePosBuffer,
        radius: &ViewportTextureF32Buffer,
    ) -> wgpu::BindGroup {
        ViewportTextureBrushRenderer::create_bind_group_static(
            device,
            &self.bind_group_layout,
            camera,
            start,
            end,
            radius,
        )
    }
}

impl ViewportTextureBrushRenderer {
    /// The bind group layout descriptor.
    pub const BIND_GROUP_LAYOUT_DESCRIPTOR: wgpu::BindGroupLayoutDescriptor<'static> =
        wgpu::BindGroupLayoutDescriptor {
            label: Some("Viewport Selection Texture Brush Renderer Bind Group Layout"),
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
                // Start uniform buffer
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
                // End uniform buffer
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
                // Radius uniform buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        };

    /// Create a new renderer.
    pub fn new(
        device: &wgpu::Device,
        texture: &ViewportTexture,
        camera: &CameraBuffer,
        start: &ViewportTexturePosBuffer,
        end: &ViewportTexturePosBuffer,
        radius: &ViewportTextureF32Buffer,
    ) -> Result<Self, RendererCreateError> {
        let this = ViewportTextureBrushRenderer::new_without_bind_group(device, texture)?;

        log::debug!("Creating viewport texture brush renderer bind group");
        let bind_group = this.create_bind_group(device, camera, start, end, radius);

        Ok(Self {
            bind_group_layout: this.bind_group_layout,
            bind_group,
            pipeline: this.pipeline,
        })
    }

    /// Render the brush.
    pub fn render(&self, encoder: &mut wgpu::CommandEncoder, texture: &ViewportTexture) {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Viewport Texture Brush Renderer Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: texture.view(),
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            ..Default::default()
        });

        self.render_with_pass(&mut render_pass);
    }

    /// Render the brush with a [`wgpu::RenderPass`].
    pub fn render_with_pass(&self, pass: &mut wgpu::RenderPass<'_>) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.draw(0..6, 0..3);
    }

    /// Create the bind group statically.
    fn create_bind_group_static(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        camera: &CameraBuffer,
        start: &ViewportTexturePosBuffer,
        end: &ViewportTexturePosBuffer,
        radius: &ViewportTextureF32Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Viewport Texture Brush Renderer Bind Group"),
            layout: bind_group_layout,
            entries: &[
                // Camera uniform buffer
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera.buffer().as_entire_binding(),
                },
                // Start uniform buffer
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: start.buffer().as_entire_binding(),
                },
                // End uniform buffer
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: end.buffer().as_entire_binding(),
                },
                // Radius uniform buffer
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: radius.buffer().as_entire_binding(),
                },
            ],
        })
    }
}

impl ViewportTextureBrushRenderer<()> {
    /// Create a new renderer without internally managed bind group.
    ///
    /// To create a bind group with layout matched to this renderer, use the
    /// [`ViewportTextureBrushRenderer::create_bind_group`] method.
    pub fn new_without_bind_group(
        device: &wgpu::Device,
        texture: &ViewportTexture,
    ) -> Result<Self, RendererCreateError> {
        log::debug!("Creating viewport texture brush renderer bind group layout");
        let bind_group_layout = device
            .create_bind_group_layout(&ViewportTextureBrushRenderer::BIND_GROUP_LAYOUT_DESCRIPTOR);

        log::debug!("Creating viewport texture brush renderer pipeline layout");
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Viewport Texture Brush Renderer Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            ..Default::default()
        });

        log::debug!("Creating viewport texture brush renderer shader");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Viewport Texture Brush Renderer Shader"),
            source: wgpu::ShaderSource::Wgsl(
                wesl::compile_sourcemap(
                    &"wgpu_3dgs_viewer::selection::viewport_texture_brush"
                        .parse()
                        .expect("selection::viewport_texture_brush module path"),
                    &wesl_utils::resolver(),
                    &wesl::NoMangler,
                    &wesl::CompileOptions::default(),
                )?
                .to_string()
                .into(),
            ),
        });

        log::debug!("Creating viewport texture brush renderer pipeline");
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Viewport Texture Brush Renderer Pipeline"),
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
                    format: texture.texture().format(),
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        log::info!("Viewport texture brush renderer created");

        Ok(Self {
            bind_group_layout,
            bind_group: (),
            pipeline,
        })
    }

    /// Render the brush.
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        texture: &ViewportTexture,
        bind_group: &wgpu::BindGroup,
    ) {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Viewport Texture Brush Renderer Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: texture.view(),
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            ..Default::default()
        });

        self.render_with_pass(&mut render_pass, bind_group);
    }

    /// Render the brush with a [`wgpu::RenderPass`].
    pub fn render_with_pass(&self, pass: &mut wgpu::RenderPass<'_>, bind_group: &wgpu::BindGroup) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.draw(0..6, 0..3);
    }
}
