use crate::{
    CameraBuffer, GaussiansBuffer, GaussiansDepthBuffer, IndirectArgsBuffer, IndirectIndicesBuffer,
    RadixSortIndirectArgsBuffer,
};

/// Preprocessor to preprocess the Gaussians.
///
/// It computes the depth for [`RadixSorter`](crate::RadixSorter) and do frustum culling.
#[derive(Debug)]
pub struct Preprocessor {
    /// The bind group layout.
    #[allow(dead_code)]
    bind_group_layout: wgpu::BindGroupLayout,
    /// The bind group.
    bind_group: wgpu::BindGroup,
    /// The compute pipeline.
    pipeline: wgpu::ComputePipeline,
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
                // Gaussian storage buffer
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
                // Indirect args storage buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
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
                    binding: 3,
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
                    binding: 4,
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
                    binding: 5,
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
    pub fn new(
        device: &wgpu::Device,
        camera: &CameraBuffer,
        gaussians: &GaussiansBuffer,
        indirect_args: &IndirectArgsBuffer,
        radix_sort_indirect_args: &RadixSortIndirectArgsBuffer,
        indirect_indices: &IndirectIndicesBuffer,
        gaussians_depth: &GaussiansDepthBuffer,
    ) -> Self {
        log::debug!("Creating preprocessor bind group layout");
        let bind_group_layout =
            device.create_bind_group_layout(&Self::BIND_GROUP_LAYOUT_DESCRIPTOR);

        log::debug!("Creating preprocessor bind group");
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Preprocessor Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                // Camera uniform buffer
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera.buffer().as_entire_binding(),
                },
                // Gaussian storage buffer
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: gaussians.buffer().as_entire_binding(),
                },
                // Indirect args storage buffer
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: indirect_args.buffer().as_entire_binding(),
                },
                // Radix sort indirect args storage buffer
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: radix_sort_indirect_args.buffer().as_entire_binding(),
                },
                // Indirect indices storage buffer
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: indirect_indices.buffer().as_entire_binding(),
                },
                // Gaussians depth storage buffer
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: gaussians_depth.buffer().as_entire_binding(),
                },
            ],
        });

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
                    .into(),
            ),
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

        log::info!("Preprocessor created");

        Self {
            bind_group_layout,
            bind_group,
            pipeline,
        }
    }

    /// Preprocess the Gaussians.
    pub fn preprocess(&self, encoder: &mut wgpu::CommandEncoder, gaussian_count: u32) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Preprocessor Compute Pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(gaussian_count.div_ceil(Self::WORKGROUP_SIZE), 1, 1);
    }
}
