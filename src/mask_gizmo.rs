use crate::{CameraBuffer, MaskGizmoPod, MaskGizmosBuffer, MaskShapeKind};

/// A mask gizmo.
#[derive(Debug)]
pub struct MaskGizmo {
    /// The bind group layout.
    bind_group_layout: wgpu::BindGroupLayout,

    /// The box gizmos buffer.
    pub box_gizmos_buffer: MaskGizmosBuffer,
    /// The box bind group.
    box_bind_group: wgpu::BindGroup,
    /// The box pipeline.
    box_pipeline: wgpu::RenderPipeline,

    /// The ellipsoid gizmos buffer.
    pub ellipsoid_gizmos_buffer: MaskGizmosBuffer,
    /// The ellipsoid bind group.
    ellipsoid_bind_group: wgpu::BindGroup,
    /// The ellipsoid pipeline.
    ellipsoid_pipeline: wgpu::RenderPipeline,
}

impl MaskGizmo {
    /// The bind group layout descriptor.
    pub const BIND_GROUP_LAYOUT_DESCRIPTOR: wgpu::BindGroupLayoutDescriptor<'static> =
        wgpu::BindGroupLayoutDescriptor {
            label: Some("Mask Gizmo Bind Group Layout"),
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
                // Gizmo storage buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
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

    /// Create a new mask gizmo.
    pub fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        camera_buffer: &CameraBuffer,
    ) -> Self {
        Self::new_with(device, format, camera_buffer, None, None)
    }

    /// Create a new mask gizmo with all options.
    pub fn new_with(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        camera_buffer: &CameraBuffer,
        box_depth_stencil: Option<wgpu::DepthStencilState>,
        ellipsoid_depth_stencil: Option<wgpu::DepthStencilState>,
    ) -> Self {
        let bind_group_layout =
            device.create_bind_group_layout(&Self::BIND_GROUP_LAYOUT_DESCRIPTOR);

        let box_gizmos_buffer = MaskGizmosBuffer::new_empty(device, 1);
        let ellipsoid_gizmos_buffer = MaskGizmosBuffer::new_empty(device, 1);

        let box_pipeline = Self::create_pipeline(
            device,
            &bind_group_layout,
            "Box",
            include_str!("shader/mask_gizmo_box.wgsl"),
            format,
            box_depth_stencil,
        );
        let ellipsoid_pipeline = Self::create_pipeline(
            device,
            &bind_group_layout,
            "Ellipsoid",
            include_str!("shader/mask_gizmo_ellipsoid.wgsl"),
            format,
            ellipsoid_depth_stencil,
        );

        let box_bind_group = Self::create_bind_group_static(
            device,
            &bind_group_layout,
            camera_buffer,
            &box_gizmos_buffer,
        );
        let ellipsoid_bind_group = Self::create_bind_group_static(
            device,
            &bind_group_layout,
            camera_buffer,
            &ellipsoid_gizmos_buffer,
        );

        Self {
            bind_group_layout,
            box_gizmos_buffer,
            box_bind_group,
            box_pipeline,
            ellipsoid_gizmos_buffer,
            ellipsoid_bind_group,
            ellipsoid_pipeline,
        }
    }

    /// Update gizmos.
    ///
    /// This recreates the buffer and bind group. Alternatively, you could directly update the
    /// buffers then call [`MaskGizmo::update_bind_group`] to potentially get better performance.
    pub fn update(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        camera_buffer: &CameraBuffer,
        shape: MaskShapeKind,
        gizmos: &[MaskGizmoPod],
    ) {
        match shape {
            MaskShapeKind::Box => {
                if self.box_gizmos_buffer.len() != gizmos.len() {
                    self.box_gizmos_buffer = MaskGizmosBuffer::new(device, gizmos);
                } else {
                    self.box_gizmos_buffer.update(queue, gizmos);
                }
            }
            MaskShapeKind::Ellipsoid => {
                if self.ellipsoid_gizmos_buffer.len() != gizmos.len() {
                    self.ellipsoid_gizmos_buffer = MaskGizmosBuffer::new(device, gizmos);
                } else {
                    self.ellipsoid_gizmos_buffer.update(queue, gizmos);
                }
            }
        }

        self.update_bind_group(device, camera_buffer, shape);
    }

    /// Update the bind group.
    pub fn update_bind_group(
        &mut self,
        device: &wgpu::Device,
        camera_buffer: &CameraBuffer,
        shape: MaskShapeKind,
    ) {
        match shape {
            MaskShapeKind::Box => {
                self.box_bind_group =
                    self.create_bind_group(device, camera_buffer, &self.box_gizmos_buffer);
            }
            MaskShapeKind::Ellipsoid => {
                self.ellipsoid_bind_group =
                    self.create_bind_group(device, camera_buffer, &self.ellipsoid_gizmos_buffer);
            }
        }
    }

    /// Render the mask gizmo.
    pub fn render(&self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView) {
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Mask Gizmo Box Render Pass"),
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

            self.render_box_with_pass(&mut render_pass);
        }

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Mask Gizmo Ellipsoid Render Pass"),
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

            self.render_ellipsoid_with_pass(&mut render_pass);
        }
    }

    /// Render the box mask gizmos with a [`wgpu::RenderPass`].
    pub fn render_box_with_pass(&self, pass: &mut wgpu::RenderPass<'_>) {
        pass.set_pipeline(&self.box_pipeline);
        pass.set_bind_group(0, &self.box_bind_group, &[]);
        pass.draw(0..4, 0..4 * self.box_gizmos_buffer.len() as u32);
    }

    /// Render the ellipsoid mask gizmos with a [`wgpu::RenderPass`].
    pub fn render_ellipsoid_with_pass(&self, pass: &mut wgpu::RenderPass<'_>) {
        pass.set_pipeline(&self.ellipsoid_pipeline);
        pass.set_bind_group(0, &self.ellipsoid_bind_group, &[]);
        pass.draw(0..33, 0..3 * self.ellipsoid_gizmos_buffer.len() as u32);
    }

    /// Create the pipeline.
    fn create_pipeline(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        shape: &str,
        shader: &str,
        format: wgpu::TextureFormat,
        depth_stencil: Option<wgpu::DepthStencilState>,
    ) -> wgpu::RenderPipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(format!("Mask Gizmo {shader} Shader").as_str()),
            source: wgpu::ShaderSource::Wgsl(shader.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(format!("Mask Gizmo {shape} Pipeline Layout").as_str()),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        });

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(format!("Mask Gizmo {shape} Pipeline").as_str()),
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
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineStrip,
                ..Default::default()
            },
            depth_stencil,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        })
    }

    /// Create a bind group for the gizmo.
    pub fn create_bind_group(
        &self,
        device: &wgpu::Device,
        camera: &CameraBuffer,
        gizmos: &MaskGizmosBuffer,
    ) -> wgpu::BindGroup {
        Self::create_bind_group_static(device, &self.bind_group_layout, camera, gizmos)
    }

    /// Create a bind group for the gizmo statically.
    fn create_bind_group_static(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        camera: &CameraBuffer,
        gizmos: &MaskGizmosBuffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Mask Gizmo Bind Group"),
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: gizmos.buffer().as_entire_binding(),
                },
            ],
        })
    }
}
