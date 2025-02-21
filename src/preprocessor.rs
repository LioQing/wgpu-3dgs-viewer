use crate::{
    CameraBuffer, Error, GaussianCov3dConfig, GaussianPod, GaussianShConfig, GaussiansBuffer,
    GaussiansDepthBuffer, IndirectArgsBuffer, IndirectIndicesBuffer, ModelTransformBuffer,
    QueryBuffer, QueryResultCountBuffer, QueryResultsBuffer, RadixSortIndirectArgsBuffer, Texture,
};

/// Preprocessor to preprocess the Gaussians.
///
/// It computes the depth for [`RadixSorter`](crate::RadixSorter), do frustum culling,
/// and process selection query.
#[derive(Debug)]
pub struct Preprocessor {
    /// The bind group layout.
    #[allow(dead_code)]
    bind_group_layout: wgpu::BindGroupLayout,
    /// The bind group.
    bind_group: wgpu::BindGroup,
    /// The pre compute pipeline.
    pre_pipeline: wgpu::ComputePipeline,
    /// The compute pipeline.
    pipeline: wgpu::ComputePipeline,
    /// The post compute pipeline.
    post_pipeline: wgpu::ComputePipeline,
}

impl Preprocessor {
    /// The workgroup size.
    pub const WORKGROUP_SIZE: u32 = 64;

    /// The bind group layout descriptor.
    pub const BIND_GROUP_LAYOUT_DESCRIPTOR: wgpu::BindGroupLayoutDescriptor<'static> =
        wgpu::BindGroupLayoutDescriptor {
            label: Some("Preprocessor Bind Group Layout"),
            entries: &[
                // Camera uniform buffer
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
                // Model transform uniform buffer
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
                // Gaussian storage buffer
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
                // Indirect args storage buffer
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
                // Radix sort indirect args storage buffer
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
                // Indirect indices storage buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Gaussians depth storage buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
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
                    binding: 7,
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
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Query results storage buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Qeruy texture view
                #[cfg(feature = "query-texture")]
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        };

    /// Create a new preprocessor.
    #[allow(clippy::too_many_arguments)]
    pub fn new<G: GaussianPod>(
        device: &wgpu::Device,
        camera: &CameraBuffer,
        model_transform: &ModelTransformBuffer,
        gaussians: &GaussiansBuffer<G>,
        indirect_args: &IndirectArgsBuffer,
        radix_sort_indirect_args: &RadixSortIndirectArgsBuffer,
        indirect_indices: &IndirectIndicesBuffer,
        gaussians_depth: &GaussiansDepthBuffer,
        query: &QueryBuffer,
        query_result_count: &QueryResultCountBuffer,
        query_results: &QueryResultsBuffer,
        #[cfg(feature = "query-texture")] query_texture: &impl Texture,
    ) -> Result<Self, Error> {
        if (device.limits().max_storage_buffer_binding_size as u64) < gaussians.buffer().size() {
            return Err(Error::ModelSizeExceedsDeviceLimit {
                model_size: gaussians.buffer().size(),
                device_limit: device.limits().max_storage_buffer_binding_size,
            });
        }

        log::debug!("Creating preprocessor bind group layout");
        let bind_group_layout =
            device.create_bind_group_layout(&Self::BIND_GROUP_LAYOUT_DESCRIPTOR);

        log::debug!("Creating preprocessor bind group");
        let bind_group = Self::create_bind_group(
            device,
            &bind_group_layout,
            camera,
            model_transform,
            gaussians,
            indirect_args,
            radix_sort_indirect_args,
            indirect_indices,
            gaussians_depth,
            query,
            query_result_count,
            query_results,
            #[cfg(feature = "query-texture")]
            query_texture,
        );

        log::debug!("Creating preprocessor pipeline layout");
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Preprocessor Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        log::debug!("Creating preprocessor shader module");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Preprocessor Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shader/preprocess.wgsl")
                    .replace(
                        "{{workgroup_size}}",
                        Self::WORKGROUP_SIZE.to_string().as_str(),
                    )
                    .replace("{{gaussian_sh_field}}", G::ShConfig::sh_field())
                    .replace("{{gaussian_cov3d_field}}", G::Cov3dConfig::cov3d_field())
                    .lines()
                    .scan(false, |state, line| {
                        #[cfg(not(feature = "query-texture"))]
                        if line.contains("// Feature query texture begin") {
                            *state = true;
                        } else if line.contains("// Feature query texture begin") {
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

        log::debug!("Creating preprocessor pre pipeline");
        let pre_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Preprocessor Pre Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("pre_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        log::debug!("Creating preprocessor pipeline");
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Preprocessor Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        log::debug!("Creating preprocessor post pipeline");
        let post_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Preprocessor Post Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("post_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        log::info!("Preprocessor created");

        Ok(Self {
            bind_group_layout,
            bind_group,
            pre_pipeline,
            pipeline,
            post_pipeline,
        })
    }

    /// Preprocess the Gaussians.
    pub fn preprocess(&self, encoder: &mut wgpu::CommandEncoder, gaussian_count: u32) {
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Preprocessor Pre Compute Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.pre_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Preprocessor Compute Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(gaussian_count.div_ceil(Self::WORKGROUP_SIZE), 1, 1);
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Preprocessor Post Compute Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.post_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
    }

    /// Update the bind group.
    ///
    /// This is specifically for updating the query texture size.
    ///
    /// This requires the `query-texture` feature.
    #[allow(clippy::too_many_arguments)]
    #[cfg(feature = "query-texture")]
    pub fn update_bind_group<G: GaussianPod>(
        &mut self,
        device: &wgpu::Device,
        camera: &CameraBuffer,
        model_transform: &ModelTransformBuffer,
        gaussians: &GaussiansBuffer<G>,
        indirect_args: &IndirectArgsBuffer,
        radix_sort_indirect_args: &RadixSortIndirectArgsBuffer,
        indirect_indices: &IndirectIndicesBuffer,
        gaussians_depth: &GaussiansDepthBuffer,
        query: &QueryBuffer,
        query_result_count: &QueryResultCountBuffer,
        query_results: &QueryResultsBuffer,
        query_texture: &impl Texture,
    ) {
        self.bind_group = Self::create_bind_group(
            device,
            &self.bind_group_layout,
            camera,
            model_transform,
            gaussians,
            indirect_args,
            radix_sort_indirect_args,
            indirect_indices,
            gaussians_depth,
            query,
            query_result_count,
            query_results,
            query_texture,
        );
    }

    /// Create the bind group.
    #[allow(clippy::too_many_arguments)]
    fn create_bind_group<G: GaussianPod>(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        camera: &CameraBuffer,
        model_transform: &ModelTransformBuffer,
        gaussians: &GaussiansBuffer<G>,
        indirect_args: &IndirectArgsBuffer,
        radix_sort_indirect_args: &RadixSortIndirectArgsBuffer,
        indirect_indices: &IndirectIndicesBuffer,
        gaussians_depth: &GaussiansDepthBuffer,
        query: &QueryBuffer,
        query_result_count: &QueryResultCountBuffer,
        query_results: &QueryResultsBuffer,
        #[cfg(feature = "query-texture")] query_texture: &impl Texture,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Preprocessor Bind Group"),
            layout: bind_group_layout,
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
                // Gaussian storage buffer
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: gaussians.buffer().as_entire_binding(),
                },
                // Indirect args storage buffer
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: indirect_args.buffer().as_entire_binding(),
                },
                // Radix sort indirect args storage buffer
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: radix_sort_indirect_args.buffer().as_entire_binding(),
                },
                // Indirect indices storage buffer
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: indirect_indices.buffer().as_entire_binding(),
                },
                // Gaussians depth storage buffer
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: gaussians_depth.buffer().as_entire_binding(),
                },
                // Query uniform buffer
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: query.buffer().as_entire_binding(),
                },
                // Query result count storage buffer
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: query_result_count.buffer().as_entire_binding(),
                },
                // Query results storage buffer
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: query_results.buffer().as_entire_binding(),
                },
                // Query texture view
                #[cfg(feature = "query-texture")]
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: wgpu::BindingResource::TextureView(query_texture.view()),
                },
            ],
        })
    }
}
