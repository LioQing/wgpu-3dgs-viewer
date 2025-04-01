use glam::*;

use crate::{CameraBuffer, QueryBuffer, QueryCursorBuffer, QueryPod, QueryType};

#[cfg(feature = "query-toolset")]
use crate::QueryToolset;

/// The supplement argument for [`QueryType::None`].
#[derive(Debug, Clone, PartialEq)]
pub struct QueryNoneSupplement {
    /// The position.
    pub position: Vec2,
    /// The radius.
    pub radius: u32,
}

/// The query cursor.
///
/// This displays a cursor according to the current query.
///
/// It maintains a separate [`QueryBuffer`] because even when the query is
/// [`QueryType::None`], the position of the cursor is still tracked.
///
/// This requires the `query-cursor` feature.
#[derive(Debug)]
pub struct QueryCursor {
    /// The query buffer.
    query_buffer: QueryBuffer,
    /// The cursor buffer.
    cursor_buffer: QueryCursorBuffer,
    /// The cursor bind group layout.
    #[allow(dead_code)]
    bind_group_layout: wgpu::BindGroupLayout,
    /// The cursor bind group.
    bind_group: wgpu::BindGroup,
    /// The cursor pipeline.
    pipeline: wgpu::RenderPipeline,
}

impl QueryCursor {
    /// The bind group layout descriptor.
    pub const BIND_GROUP_LAYOUT_DESCRIPTOR: wgpu::BindGroupLayoutDescriptor<'static> =
        wgpu::BindGroupLayoutDescriptor {
            label: Some("Query Cursor Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
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

    /// Create a new query cursor.
    pub fn new(
        device: &wgpu::Device,
        texture_format: wgpu::TextureFormat,
        camera: &CameraBuffer,
    ) -> Self {
        Self::new_with(device, texture_format, None, camera)
    }

    /// Create a new query cursor with all options.
    pub fn new_with(
        device: &wgpu::Device,
        texture_format: wgpu::TextureFormat,
        depth_stencil: Option<wgpu::DepthStencilState>,
        camera: &CameraBuffer,
    ) -> Self {
        log::debug!("Creating query cursor query buffer");
        let query_buffer = QueryBuffer::new(device);

        log::debug!("Creating query cursor cursor buffer");
        let cursor_buffer = QueryCursorBuffer::new(device);

        log::debug!("Creating query cursor bind group layout");
        let bind_group_layout =
            device.create_bind_group_layout(&Self::BIND_GROUP_LAYOUT_DESCRIPTOR);

        log::debug!("Creating query cursor bind group");
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Query Cursor Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: query_buffer.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: camera.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cursor_buffer.buffer().as_entire_binding(),
                },
            ],
        });

        log::debug!("Creating query cursor pipeline");
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Query Cursor Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        log::debug!("Creating query cursor shader");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Query Cursor Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader/query_cursor.wgsl").into()),
        });

        log::debug!("Creating query cursor pipeline");
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Query Cursor Pipeline"),
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
                    blend: None,
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
            query_buffer,
            cursor_buffer,
            bind_group_layout,
            bind_group,
            pipeline,
        }
    }

    /// Update the query.
    ///
    /// **Important**: If query is [`QueryType::None`], the [`QueryNoneSupplement`] must be provided
    /// for it to work correctly. Supplements will have no effect if the query is not
    /// [`QueryType::None`].
    pub fn update_query(
        &mut self,
        queue: &wgpu::Queue,
        query: &QueryPod,
        supplement: Option<QueryNoneSupplement>,
    ) {
        match supplement {
            Some(QueryNoneSupplement { position, radius })
                if query.query_type() == QueryType::None =>
            {
                self.query_buffer.update(
                    queue,
                    &QueryPod {
                        content_u32: uvec4(0, radius, 0, 0),
                        content_f32: vec4(0.0, 0.0, position.x, position.y),
                    },
                );
            }
            _ => self.query_buffer.update(queue, query),
        }
    }

    /// Update the query using query toolset.
    ///
    /// This is more convenient than [`Self::update_query`] because it automatically handles the
    /// [`QueryType::None`] case by displaying the brush radius.
    ///
    /// This requires the `query-toolset` feature.
    #[cfg(feature = "query-toolset")]
    pub fn update_query_toolset(
        &mut self,
        queue: &wgpu::Queue,
        query_toolset: &QueryToolset,
        pos: Vec2,
    ) {
        self.update_query(
            queue,
            query_toolset.query(),
            match query_toolset.query().query_type() {
                QueryType::None => Some(QueryNoneSupplement {
                    position: pos,
                    radius: query_toolset.brush_radius(),
                }),
                _ => None,
            },
        )
    }

    /// Update the cursor.
    pub fn update_cursor(&mut self, queue: &wgpu::Queue, outline_color: Vec4, outline_width: f32) {
        self.cursor_buffer
            .update(queue, outline_color, outline_width);
    }

    /// Render the cursor.
    pub fn render(&self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView) {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Query Cursor Render Pass"),
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

    /// Render the cursor with a [`wgpu::RenderPass`].
    pub fn render_with_pass(&self, pass: &mut wgpu::RenderPass<'_>) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.draw(0..6, 0..1);
    }
}
