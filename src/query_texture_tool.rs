use glam::*;

use crate::{CameraBuffer, Error, QueryBuffer, QueryPod, QueryTexture, Texture};

/// The query texture tool state.
#[derive(Debug)]
pub enum QueryTextureToolState {
    /// The rectangle tool state.
    Rect { start: Vec2 },

    /// The brush tool state.
    Brush { radius: u32, pos: Vec2 },
}

/// The query texture tool.
///
/// It allows updating the query texture using commonly found tools in editing softwares.
/// This includes: rectangle, brush.
///
/// This requires the `query-texture-tool` feature.
#[derive(Debug)]
pub struct QueryTextureTool {
    /// Whether the state just started.
    just_started: bool,
    /// The query texture tool state.
    state: Option<QueryTextureToolState>,

    /// The query for the tool.
    query: QueryPod,
    /// The query buffer.
    query_buffer: QueryBuffer,

    /// The bind group layout.
    #[allow(dead_code)]
    bind_group_layout: wgpu::BindGroupLayout,
    /// The bind group.
    bind_group: wgpu::BindGroup,
    /// The rect pipeline.
    rect_pipeline: wgpu::RenderPipeline,
    /// The brush pipeline.
    brush_pipeline: wgpu::RenderPipeline,
}

impl QueryTextureTool {
    /// The bind group layout descriptor.
    pub const BIND_GROUP_LAYOUT_DESCRIPTOR: wgpu::BindGroupLayoutDescriptor<'static> =
        wgpu::BindGroupLayoutDescriptor {
            label: Some("Query Texture Tool Bind Group Layout"),
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
                // Query uniform buffer
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
            ],
        };

    /// Create a new query texture tool.
    pub fn new(device: &wgpu::Device, query_texture: &QueryTexture, camera: &CameraBuffer) -> Self {
        let query = QueryPod::none();
        let query_buffer = QueryBuffer::new(device);

        let bind_group_layout =
            device.create_bind_group_layout(&Self::BIND_GROUP_LAYOUT_DESCRIPTOR);

        log::debug!("Creating query texture tool bind group");
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Query Texture Tool Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: query_buffer.buffer().as_entire_binding(),
                },
            ],
        });

        log::debug!("Creating query texture tool pipeline layout");
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Query Texture Tool Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        log::debug!("Creating query texture tool rect shader");
        let rect_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Query Texture Tool Rect Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shader/query_texture_tool_rect.wgsl").into(),
            ),
        });

        log::debug!("Creating query texture tool rect pipeline");
        let rect_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Query Texture Tool Rect Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &rect_shader,
                entry_point: Some("vert_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &rect_shader,
                entry_point: Some("frag_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: query_texture.texture().format(),
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

        log::debug!("Creating query texture tool brush shader");
        let brush_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Query Texture Tool Brush Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shader/query_texture_tool_brush.wgsl").into(),
            ),
        });

        log::debug!("Creating query texture tool brush pipeline");
        let brush_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Query Texture Tool Brush Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &brush_shader,
                entry_point: Some("vert_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &brush_shader,
                entry_point: Some("frag_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: query_texture.texture().format(),
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

        Self {
            just_started: false,
            state: None,

            query,
            query_buffer,

            bind_group_layout,
            bind_group,
            rect_pipeline,
            brush_pipeline,
        }
    }

    /// Get the query.
    pub fn query(&self) -> &QueryPod {
        &self.query
    }

    /// Get the query buffer.
    pub fn query_buffer(&self) -> &QueryBuffer {
        &self.query_buffer
    }

    /// Get the state.
    pub fn state(&self) -> Option<&QueryTextureToolState> {
        self.state.as_ref()
    }

    /// Start a new rect query.
    pub fn start_rect(&mut self, start: Vec2) -> Result<&QueryPod, Error> {
        if self.state.is_some() {
            return Err(Error::QueryTextureToolAlreadyInUse);
        }

        self.query = QueryPod::rect(start, start);
        self.just_started = true;
        self.state = Some(QueryTextureToolState::Rect { start });

        Ok(&self.query)
    }

    /// Start a new brush query.
    pub fn start_brush(&mut self, radius: u32, pos: Vec2) -> Result<&QueryPod, Error> {
        if self.state.is_some() {
            return Err(Error::QueryTextureToolAlreadyInUse);
        }

        self.query = QueryPod::brush(radius, pos, pos);
        self.just_started = true;
        self.state = Some(QueryTextureToolState::Brush { radius, pos });

        Ok(&self.query)
    }

    /// Update the current query.
    pub fn update(&mut self, pos: Vec2) -> Result<&QueryPod, Error> {
        match &mut self.state {
            Some(QueryTextureToolState::Rect { start }) => {
                let top_left = start.min(pos);
                let bottom_right = start.max(pos);

                self.query = QueryPod::rect(top_left, bottom_right);

                Ok(&self.query)
            }
            Some(QueryTextureToolState::Brush { radius, pos: start }) => {
                self.query = QueryPod::brush(*radius, *start, pos);
                *start = pos;

                Ok(&self.query)
            }
            None => Err(Error::QueryTextureToolNotInUse),
        }
    }

    /// Update the brush radius.
    pub fn update_brush_radius(&mut self, radius: u32) -> Result<&QueryPod, Error> {
        match &mut self.state {
            Some(QueryTextureToolState::Brush {
                radius: old_radius,
                pos,
            }) => {
                self.query = QueryPod::brush(radius, *pos, *pos);
                *old_radius = radius;

                Ok(&self.query)
            }
            _ => Err(Error::QueryTextureToolNotInUse),
        }
    }

    /// End the current query.
    pub fn end(&mut self) -> Result<&QueryPod, Error> {
        match &self.state {
            Some(..) => {
                self.just_started = false;
                self.state = None;
                Ok(&self.query)
            }
            None => Err(Error::QueryTextureToolNotInUse),
        }
    }

    /// Update the buffer.
    pub fn update_buffer(&mut self, queue: &wgpu::Queue) {
        self.query_buffer.update(queue, &self.query);
    }

    /// Render the tool.
    pub fn render(&mut self, encoder: &mut wgpu::CommandEncoder, query_texture: &QueryTexture) {
        if self.state.is_none() {
            return;
        }

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Query Texture Tool Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: query_texture.view(),
                resolve_target: None,
                ops: wgpu::Operations {
                    load: if self.just_started
                        || matches!(&self.state, Some(QueryTextureToolState::Rect { .. }))
                    {
                        self.just_started = false;
                        wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
                    } else {
                        wgpu::LoadOp::Load
                    },
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        match &self.state {
            Some(QueryTextureToolState::Rect { .. }) => {
                render_pass.set_pipeline(&self.rect_pipeline);
                render_pass.set_bind_group(0, &self.bind_group, &[]);
                render_pass.draw(0..6, 0..1);
            }
            Some(QueryTextureToolState::Brush { .. }) => {
                render_pass.set_pipeline(&self.brush_pipeline);
                render_pass.set_bind_group(0, &self.bind_group, &[]);
                render_pass.draw(0..6, 0..3);
            }
            None => {}
        };
    }
}
