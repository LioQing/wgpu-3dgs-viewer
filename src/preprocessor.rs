use glam::*;

use crate::{
    CameraBuffer, Error, GaussiansDepthBuffer, IndirectArgsBuffer, IndirectIndicesBuffer,
    RadixSortIndirectArgsBuffer,
    core::{BufferWrapper, GaussianPod, GaussiansBuffer, ModelTransformBuffer},
    wesl_utils,
};

/// Preprocessor to preprocess the Gaussians.
///
/// It computes the depth for [`RadixSorter`](crate::RadixSorter) and do frustum culling.
#[derive(Debug)]
pub struct Preprocessor<B = wgpu::BindGroup> {
    /// The workgroup size.
    workgroup_size: u32,
    /// The bind group layout.
    #[allow(dead_code)]
    bind_group_layout: wgpu::BindGroupLayout,
    /// The bind group.
    bind_group: B,
    /// The pre compute pipeline.
    pre_pipeline: wgpu::ComputePipeline,
    /// The compute pipeline.
    pipeline: wgpu::ComputePipeline,
    /// The post compute pipeline.
    post_pipeline: wgpu::ComputePipeline,
}

impl<B> Preprocessor<B> {
    /// Create the bind group.
    #[allow(clippy::too_many_arguments)]
    pub fn create_bind_group<G: GaussianPod>(
        &self,
        device: &wgpu::Device,
        camera: &CameraBuffer,
        model_transform: &ModelTransformBuffer,
        gaussians: &GaussiansBuffer<G>,
        indirect_args: &IndirectArgsBuffer,
        radix_sort_indirect_args: &RadixSortIndirectArgsBuffer,
        indirect_indices: &IndirectIndicesBuffer,
        gaussians_depth: &GaussiansDepthBuffer,
    ) -> wgpu::BindGroup {
        Preprocessor::create_bind_group_static(
            device,
            &self.bind_group_layout,
            camera,
            model_transform,
            gaussians,
            indirect_args,
            radix_sort_indirect_args,
            indirect_indices,
            gaussians_depth,
        )
    }

    /// Get the number of invocations in one workgroup.
    pub fn workgroup_size(&self) -> u32 {
        self.workgroup_size
    }
}

impl Preprocessor {
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
    ) -> Result<Self, Error> {
        if (device.limits().max_storage_buffer_binding_size as u64) < gaussians.buffer().size() {
            return Err(Error::ModelSizeExceedsDeviceLimit {
                model_size: gaussians.buffer().size(),
                device_limit: device.limits().max_storage_buffer_binding_size,
            });
        }

        let this = Preprocessor::new_without_bind_group::<G>(device)?;

        log::debug!("Creating preprocessor bind group");
        let bind_group = this.create_bind_group(
            device,
            camera,
            model_transform,
            gaussians,
            indirect_args,
            radix_sort_indirect_args,
            indirect_indices,
            gaussians_depth,
        );

        Ok(Self {
            workgroup_size: this.workgroup_size,
            bind_group_layout: this.bind_group_layout,
            bind_group,
            pre_pipeline: this.pre_pipeline,
            pipeline: this.pipeline,
            post_pipeline: this.post_pipeline,
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
            pass.dispatch_workgroups(gaussian_count.div_ceil(self.workgroup_size()), 1, 1);
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

    /// Create the bind group statically.
    #[allow(clippy::too_many_arguments)]
    fn create_bind_group_static<G: GaussianPod>(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        camera: &CameraBuffer,
        model_transform: &ModelTransformBuffer,
        gaussians: &GaussiansBuffer<G>,
        indirect_args: &IndirectArgsBuffer,
        radix_sort_indirect_args: &RadixSortIndirectArgsBuffer,
        indirect_indices: &IndirectIndicesBuffer,
        gaussians_depth: &GaussiansDepthBuffer,
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
            ],
        })
    }
}

impl Preprocessor<()> {
    /// Create a new preprocessor without interally managed bind group.
    ///
    /// To create a bind group with layout matched to this preprocessor, use the
    /// [`Preprocessor::create_bind_group`] method.
    pub fn new_without_bind_group<G: GaussianPod>(device: &wgpu::Device) -> Result<Self, Error> {
        let workgroup_size = device
            .limits()
            .max_compute_workgroup_size_x
            .min(device.limits().max_compute_invocations_per_workgroup);

        log::debug!("Creating preprocessor bind group layout");
        let bind_group_layout =
            device.create_bind_group_layout(&Preprocessor::BIND_GROUP_LAYOUT_DESCRIPTOR);

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
                wesl_utils::compiler(G::features())
                    .compile("preprocess")?
                    .to_string()
                    .into(),
            ),
        });

        let compilation_options = wgpu::PipelineCompilationOptions {
            constants: &[("workgroup_size", workgroup_size as f64)],
            ..Default::default()
        };

        log::debug!("Creating preprocessor pre pipeline");
        let pre_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Preprocessor Pre Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("pre_main"),
            compilation_options: compilation_options.clone(),
            cache: None,
        });

        log::debug!("Creating preprocessor pipeline");
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Preprocessor Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: compilation_options.clone(),
            cache: None,
        });

        log::debug!("Creating preprocessor post pipeline");
        let post_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Preprocessor Post Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("post_main"),
            compilation_options,
            cache: None,
        });

        log::info!("Preprocessor created");

        Ok(Self {
            workgroup_size,
            bind_group_layout,
            bind_group: (),
            pre_pipeline,
            pipeline,
            post_pipeline,
        })
    }

    /// Preprocess the Gaussians.
    pub fn preprocess(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        gaussian_count: u32,
    ) {
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Preprocessor Pre Compute Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.pre_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Preprocessor Compute Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(gaussian_count.div_ceil(self.workgroup_size()), 1, 1);
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Preprocessor Post Compute Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.post_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
    }
}
