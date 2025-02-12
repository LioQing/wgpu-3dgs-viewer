use crate::{
    PostprocessIndirectArgsBuffer, QueryBuffer, QueryResultCountBuffer, QueryResultsBuffer,
    SelectionBuffer,
};

/// Postprocessor to postprocess the query and selection operations.
///
/// It carry out selection operations based on the query using
/// [`QuerySelectionOp`](crate::QuerySelectionOp).
#[derive(Debug)]
pub struct Postprocessor {
    /// The bind group layout.
    #[allow(dead_code)]
    bind_group_layout: wgpu::BindGroupLayout,
    /// The bind group.
    bind_group: wgpu::BindGroup,
    /// The pre compute pipeline.
    pre_pipeline: wgpu::ComputePipeline,
    /// The compute pipeline.
    pipeline: wgpu::ComputePipeline,
}

impl Postprocessor {
    /// The workgroup size.
    pub const WORKGROUP_SIZE: u32 = 64;

    /// The bind group layout descriptor.
    pub const BIND_GROUP_LAYOUT_DESCRIPTOR: wgpu::BindGroupLayoutDescriptor<'static> =
        wgpu::BindGroupLayoutDescriptor {
            label: Some("Postprocessor Bind Group Layout"),
            entries: &[
                // Postprocess indirect args buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Query uniform buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Query result count storage buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Query results storage buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Selection storage buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        };

    /// Create a new postprocessor.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &wgpu::Device,
        postprocess_indirect_args: &PostprocessIndirectArgsBuffer,
        query: &QueryBuffer,
        query_result_count: &QueryResultCountBuffer,
        query_results: &QueryResultsBuffer,
        selection: &SelectionBuffer,
    ) -> Self {
        let bind_group_layout =
            device.create_bind_group_layout(&Postprocessor::BIND_GROUP_LAYOUT_DESCRIPTOR);

        log::debug!("Creating postprocessor bind group");
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[
                // Postprocess indirect args buffer
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: postprocess_indirect_args.buffer().as_entire_binding(),
                },
                // Query uniform buffer
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: query.buffer().as_entire_binding(),
                },
                // Query result count storage buffer
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: query_result_count.buffer().as_entire_binding(),
                },
                // Query results storage buffer
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: query_results.buffer().as_entire_binding(),
                },
                // Selection storage buffer
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: selection.buffer().as_entire_binding(),
                },
            ],
            label: Some("Postprocessor Bind Group"),
        });

        log::debug!("Creating postprocessor pipeline layout");
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Postprocessor Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        log::debug!("Creating postprocessor shader module");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Postprocessor Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shader/postprocess.wgsl")
                    .replace(
                        "{{workgroup_size}}",
                        Self::WORKGROUP_SIZE.to_string().as_str(),
                    )
                    .into(),
            ),
        });

        log::debug!("Creating postprocessor pre pipeline");
        let pre_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Postprocessor Pre Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("pre_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        log::debug!("Creating postprocessor pipeline");
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Postprocessor Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        log::info!("Postprocessor created");

        Self {
            bind_group_layout,
            bind_group,
            pre_pipeline,
            pipeline,
        }
    }

    /// Postprocess the query and selection.
    pub fn postprocess(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        gaussian_count: u32,
        indirect_args_buffer: &PostprocessIndirectArgsBuffer,
    ) {
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Postprocessor Pre Compute Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.pre_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(
                gaussian_count.div_ceil(32).div_ceil(Self::WORKGROUP_SIZE),
                1,
                1,
            );
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Postprocessor Compute Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups_indirect(indirect_args_buffer.buffer(), 0);
        }
    }
}
