use crate::{
    PostprocessIndirectArgsBuffer, QueryBuffer, QueryResultCountBuffer, QueryResultsBuffer,
    SelectionBuffer,
};

/// Postprocessor to postprocess the query and selection operations.
///
/// It carry out selection operations based on the query using
/// [`QuerySelectionOp`](crate::QuerySelectionOp).
#[derive(Debug)]
pub struct Postprocessor<B = wgpu::BindGroup> {
    /// The pre bind group layout.
    #[allow(dead_code)]
    pre_bind_group_layout: wgpu::BindGroupLayout,
    /// The pre bind group.
    pre_bind_group: B,
    /// The bind group layout.
    #[allow(dead_code)]
    bind_group_layout: wgpu::BindGroupLayout,
    /// The bind group.
    bind_group: B,
    /// The pre compute pipeline.
    pre_pipeline: wgpu::ComputePipeline,
    /// The compute pipeline.
    pipeline: wgpu::ComputePipeline,
}

impl<B> Postprocessor<B> {
    /// Create the bind groups.
    ///
    /// This returns (pre_bind_group, bind_group).
    pub fn create_bind_groups(
        &self,
        device: &wgpu::Device,
        postprocess_indirect_args: &PostprocessIndirectArgsBuffer,
        query: &QueryBuffer,
        query_result_count: &QueryResultCountBuffer,
        query_results: &QueryResultsBuffer,
        selection: &SelectionBuffer,
    ) -> (wgpu::BindGroup, wgpu::BindGroup) {
        Postprocessor::create_bind_groups_static(
            device,
            &self.pre_bind_group_layout,
            &self.bind_group_layout,
            postprocess_indirect_args,
            query,
            query_result_count,
            query_results,
            selection,
        )
    }
}

impl Postprocessor {
    /// The workgroup size.
    pub const WORKGROUP_SIZE: u32 = 64;

    /// The pre bind group layout descriptor.
    pub const PRE_BIND_GROUP_LAYOUT_DESCRIPTOR: wgpu::BindGroupLayoutDescriptor<'static> =
        wgpu::BindGroupLayoutDescriptor {
            label: Some("Postprocessor Pre Bind Group Layout"),
            entries: &[
                // Query uniform buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
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
                    binding: 1,
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
                    binding: 2,
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
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Postprocess indirect args buffer
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

    /// The bind group layout descriptor.
    pub const BIND_GROUP_LAYOUT_DESCRIPTOR: wgpu::BindGroupLayoutDescriptor<'static> =
        wgpu::BindGroupLayoutDescriptor {
            label: Some("Postprocessor Bind Group Layout"),
            entries: &[
                // Query uniform buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
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
                    binding: 1,
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
                    binding: 2,
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
                    binding: 3,
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
        let this = Postprocessor::new_without_bind_groups(device);

        log::debug!("Creating postprocessor bind groups");
        let (pre_bind_group, bind_group) = this.create_bind_groups(
            device,
            postprocess_indirect_args,
            query,
            query_result_count,
            query_results,
            selection,
        );

        Self {
            pre_bind_group_layout: this.pre_bind_group_layout,
            pre_bind_group,
            bind_group_layout: this.bind_group_layout,
            bind_group,
            pre_pipeline: this.pre_pipeline,
            pipeline: this.pipeline,
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
            pass.set_bind_group(0, &self.pre_bind_group, &[]);
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

    /// Create the bind group statically.
    #[allow(clippy::too_many_arguments)]
    fn create_bind_groups_static(
        device: &wgpu::Device,
        pre_bind_group_layout: &wgpu::BindGroupLayout,
        bind_group_layout: &wgpu::BindGroupLayout,
        postprocess_indirect_args: &PostprocessIndirectArgsBuffer,
        query: &QueryBuffer,
        query_result_count: &QueryResultCountBuffer,
        query_results: &QueryResultsBuffer,
        selection: &SelectionBuffer,
    ) -> (wgpu::BindGroup, wgpu::BindGroup) {
        let pre_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Pre Postprocessor Pre Bind Group"),
            layout: pre_bind_group_layout,
            entries: &[
                // Query uniform buffer
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: query.buffer().as_entire_binding(),
                },
                // Query result count storage buffer
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: query_result_count.buffer().as_entire_binding(),
                },
                // Query results storage buffer
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: query_results.buffer().as_entire_binding(),
                },
                // Selection storage buffer
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: selection.buffer().as_entire_binding(),
                },
                // Postprocess indirect args buffer
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: postprocess_indirect_args.buffer().as_entire_binding(),
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Postprocessor Bind Group"),
            layout: bind_group_layout,
            entries: &[
                // Query uniform buffer
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: query.buffer().as_entire_binding(),
                },
                // Query result count storage buffer
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: query_result_count.buffer().as_entire_binding(),
                },
                // Query results storage buffer
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: query_results.buffer().as_entire_binding(),
                },
                // Selection storage buffer
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: selection.buffer().as_entire_binding(),
                },
            ],
        });

        (pre_bind_group, bind_group)
    }
}

impl Postprocessor<()> {
    /// Create a new postprocessor without interally managed bind group.
    ///
    /// To create a bind group with layout matched to this postprocessor, use the
    /// [`Postprocessor::create_bind_groups`] method.
    pub fn new_without_bind_groups(device: &wgpu::Device) -> Self {
        let pre_bind_group_layout =
            device.create_bind_group_layout(&Postprocessor::PRE_BIND_GROUP_LAYOUT_DESCRIPTOR);

        let bind_group_layout =
            device.create_bind_group_layout(&Postprocessor::BIND_GROUP_LAYOUT_DESCRIPTOR);

        log::debug!("Creating postprocessor pre pipeline layout");
        let pre_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Postprocessor Pre Pipeline Layout"),
            bind_group_layouts: &[&pre_bind_group_layout],
            push_constant_ranges: &[],
        });

        log::debug!("Creating postprocessor pre shader module");
        let pre_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Postprocessor Pre Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shader/postprocess.wgsl")
                    .replace(
                        "{{workgroup_size}}",
                        Postprocessor::WORKGROUP_SIZE.to_string().as_str(),
                    )
                    .into(),
            ),
        });

        log::debug!("Creating postprocessor pre pipeline");
        let pre_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Postprocessor Pre Pipeline"),
            layout: Some(&pre_pipeline_layout),
            module: &pre_shader,
            entry_point: Some("pre_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
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
                        Postprocessor::WORKGROUP_SIZE.to_string().as_str(),
                    )
                    .lines()
                    .scan(false, |state, line| {
                        if line.contains("// Pre only begin") {
                            *state = true;
                        } else if line.contains("// Pre only end") {
                            *state = false;
                        }

                        if *state {
                            Some(format!("// {line}\n"))
                        } else {
                            Some(format!("{line}\n"))
                        }
                    })
                    .collect::<String>()
                    .into(),
            ),
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
            pre_bind_group_layout,
            pre_bind_group: (),
            bind_group_layout,
            bind_group: (),
            pre_pipeline,
            pipeline,
        }
    }

    /// Postprocess the query and selection.
    pub fn postprocess(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pre_bind_group: &wgpu::BindGroup,
        bind_group: &wgpu::BindGroup,
        gaussian_count: u32,
        indirect_args_buffer: &PostprocessIndirectArgsBuffer,
    ) {
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Postprocessor Pre Compute Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.pre_pipeline);
            pass.set_bind_group(0, pre_bind_group, &[]);
            pass.dispatch_workgroups(
                gaussian_count
                    .div_ceil(32)
                    .div_ceil(Postprocessor::WORKGROUP_SIZE),
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
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups_indirect(indirect_args_buffer.buffer(), 0);
        }
    }
}
