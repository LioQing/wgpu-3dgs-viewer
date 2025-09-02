use crate::{
    CameraBuffer, Error,
    core::BufferWrapper,
    selection::{ViewportTexture, ViewportTexturePosBuffer},
    wesl_utils,
};

/// A renderer for applying a rectangle selection to [`ViewportTexture`].
#[derive(Debug)]
pub struct ViewportTextureRectangleRenderer<B = wgpu::BindGroup> {
    /// The bind group layout.
    #[allow(dead_code)]
    bind_group_layout: wgpu::BindGroupLayout,
    /// The bind group.
    bind_group: B,
    /// The render pipeline.
    pipeline: wgpu::RenderPipeline,
}

impl<B> ViewportTextureRectangleRenderer<B> {
    /// Create the bind group.
    pub fn create_bind_group(
        &self,
        device: &wgpu::Device,
        camera: &CameraBuffer,
        top_left: &ViewportTexturePosBuffer,
        bottom_right: &ViewportTexturePosBuffer,
    ) -> wgpu::BindGroup {
        ViewportTextureRectangleRenderer::create_bind_group_static(
            device,
            &self.bind_group_layout,
            camera,
            top_left,
            bottom_right,
        )
    }
}

impl ViewportTextureRectangleRenderer {
    /// The bind group layout descriptor.
    pub const BIND_GROUP_LAYOUT_DESCRIPTOR: wgpu::BindGroupLayoutDescriptor<'static> =
        wgpu::BindGroupLayoutDescriptor {
            label: Some("Viewport Selection Texture Rectangle Renderer Bind Group Layout"),
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
                // Top left uniform buffer
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
                // Bottom right uniform buffer
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
            ],
        };

    /// Create a new renderer.
    pub fn new(
        device: &wgpu::Device,
        texture: &ViewportTexture,
        camera: &CameraBuffer,
        top_left: &ViewportTexturePosBuffer,
        bottom_right: &ViewportTexturePosBuffer,
    ) -> Result<Self, Error> {
        let this = ViewportTextureRectangleRenderer::new_without_bind_group(device, texture)?;

        log::debug!("Creating viewport texture rectangle renderer bind group");
        let bind_group = this.create_bind_group(device, camera, top_left, bottom_right);

        Ok(Self {
            bind_group_layout: this.bind_group_layout,
            bind_group,
            pipeline: this.pipeline,
        })
    }

    /// Render the rectangle.
    pub fn render(&self, encoder: &mut wgpu::CommandEncoder, texture: &ViewportTexture) {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Viewport Texture Rectangle Renderer Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: texture.view(),
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        self.render_with_pass(&mut render_pass);
    }

    /// Render the rectangle with a [`wgpu::RenderPass`].
    pub fn render_with_pass(&self, pass: &mut wgpu::RenderPass<'_>) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.draw(0..6, 0..1);
    }

    /// Create the bind group statically.
    fn create_bind_group_static(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        camera: &CameraBuffer,
        top_left: &ViewportTexturePosBuffer,
        bottom_right: &ViewportTexturePosBuffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Viewport Texture Rectangle Renderer Bind Group"),
            layout: bind_group_layout,
            entries: &[
                // Camera uniform buffer
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera.buffer().as_entire_binding(),
                },
                // Top left uniform buffer
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: top_left.buffer().as_entire_binding(),
                },
                // Bottom right uniform buffer
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bottom_right.buffer().as_entire_binding(),
                },
            ],
        })
    }
}

impl ViewportTextureRectangleRenderer<()> {
    /// Create a new renderer without internally managed bind group.
    ///
    /// To create a bind group with layout matched to this renderer, use the
    /// [`ViewportTextureRectangleRenderer::create_bind_group`] method.
    pub fn new_without_bind_group(
        device: &wgpu::Device,
        texture: &ViewportTexture,
    ) -> Result<Self, Error> {
        log::debug!("Creating viewport texture rectangle renderer bind group layout");
        let bind_group_layout = device.create_bind_group_layout(
            &ViewportTextureRectangleRenderer::BIND_GROUP_LAYOUT_DESCRIPTOR,
        );

        log::debug!("Creating viewport texture rectangle renderer pipeline layout");
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Viewport Texture Rectangle Renderer Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        log::debug!("Creating viewport texture rectangle renderer shader");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Viewport Texture Rectangle Renderer Shader"),
            source: wgpu::ShaderSource::Wgsl(
                wesl::compile_sourcemap(
                    &"wgpu_3dgs_viewer::selection::viewport_texture_rectangle"
                        .parse()
                        .expect("selection::viewport_texture_rectangle module path"),
                    &wesl_utils::resolver(),
                    &wesl::NoMangler,
                    &wesl::CompileOptions::default(),
                )?
                .to_string()
                .into(),
            ),
        });

        log::debug!("Creating viewport texture rectangle renderer pipeline");
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Viewport Texture Rectangle Renderer Pipeline"),
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
            multiview: None,
            cache: None,
        });

        log::info!("Viewport texture rectangle renderer created");

        Ok(Self {
            bind_group_layout,
            bind_group: (),
            pipeline,
        })
    }

    /// Render the rectangle.
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        texture: &ViewportTexture,
        bind_group: &wgpu::BindGroup,
    ) {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Viewport Texture Rectangle Renderer Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: texture.view(),
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        self.render_with_pass(&mut render_pass, bind_group);
    }

    /// Render the rectangle with a [`wgpu::RenderPass`].
    pub fn render_with_pass(&self, pass: &mut wgpu::RenderPass<'_>, bind_group: &wgpu::BindGroup) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.draw(0..6, 0..1);
    }
}
