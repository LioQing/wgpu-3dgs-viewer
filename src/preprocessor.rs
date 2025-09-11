use crate::{
    CameraBuffer, GaussiansDepthBuffer, IndirectArgsBuffer, IndirectIndicesBuffer,
    PreprocessorCreateError, RadixSortIndirectArgsBuffer,
    core::{
        BufferWrapper, ComputeBundle, ComputeBundleBuilder, GaussianPod, GaussiansBuffer,
        ModelTransformBuffer,
    },
    wesl_utils,
};

/// Preprocessor to preprocess the Gaussians.
///
/// It computes the depth for [`RadixSorter`](crate::RadixSorter) and do frustum culling.
#[derive(Debug)]
pub struct Preprocessor<G: GaussianPod, B = wgpu::BindGroup> {
    /// The bind group layout.
    #[allow(dead_code)]
    bind_group_layout: wgpu::BindGroupLayout,
    /// The bind group.
    bind_group: B,
    /// The pre preprocess bundle.
    pre_bundle: ComputeBundle<()>,
    /// The preprocess bundle.
    bundle: ComputeBundle<()>,
    /// The post preprocess bundle.
    post_bundle: ComputeBundle<()>,
    /// The marker for the Gaussian POD type.
    gaussian_pod_marker: std::marker::PhantomData<G>,
}

impl<G: GaussianPod, B> Preprocessor<G, B> {
    /// Create the bind group.
    #[allow(clippy::too_many_arguments)]
    pub fn create_bind_group(
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
        self.bundle.workgroup_size()
    }

    /// Get the bind group layouts.
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }
}

impl<G: GaussianPod> Preprocessor<G> {
    /// The label.
    const LABEL: &str = "Preprocessor";

    /// The main shader module path.
    const MAIN_SHADER: &str = "wgpu_3dgs_viewer::preprocess";

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
    pub fn new(
        device: &wgpu::Device,
        camera: &CameraBuffer,
        model_transform: &ModelTransformBuffer,
        gaussians: &GaussiansBuffer<G>,
        indirect_args: &IndirectArgsBuffer,
        radix_sort_indirect_args: &RadixSortIndirectArgsBuffer,
        indirect_indices: &IndirectIndicesBuffer,
        gaussians_depth: &GaussiansDepthBuffer,
    ) -> Result<Self, PreprocessorCreateError> {
        if (device.limits().max_storage_buffer_binding_size as wgpu::BufferAddress)
            < gaussians.buffer().size()
        {
            return Err(PreprocessorCreateError::ModelSizeExceedsDeviceLimit {
                model_size: gaussians.buffer().size(),
                device_limit: device.limits().max_storage_buffer_binding_size,
            });
        }

        let this = Preprocessor::new_without_bind_group(device)?;

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
            bind_group_layout: this.bind_group_layout,
            bind_group,
            pre_bundle: this.pre_bundle,
            bundle: this.bundle,
            post_bundle: this.post_bundle,
            gaussian_pod_marker: std::marker::PhantomData,
        })
    }

    /// Preprocess the Gaussians.
    pub fn preprocess(&self, encoder: &mut wgpu::CommandEncoder, gaussian_count: u32) {
        self.pre_bundle.dispatch(encoder, 1, [&self.bind_group]);

        self.bundle
            .dispatch(encoder, gaussian_count, [&self.bind_group]);

        self.post_bundle.dispatch(encoder, 1, [&self.bind_group]);
    }

    /// Create the bind group statically.
    #[allow(clippy::too_many_arguments)]
    fn create_bind_group_static(
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

impl<G: GaussianPod> Preprocessor<G, ()> {
    /// Create a new preprocessor without interally managed bind group.
    ///
    /// To create a bind group with layout matched to this preprocessor, use the
    /// [`Preprocessor::create_bind_group`] method.
    pub fn new_without_bind_group(device: &wgpu::Device) -> Result<Self, PreprocessorCreateError> {
        let bind_group_layout =
            device.create_bind_group_layout(&Preprocessor::<G>::BIND_GROUP_LAYOUT_DESCRIPTOR);

        let pre_bundle = ComputeBundleBuilder::new()
            .label(format!("Pre {}", Preprocessor::<G>::LABEL).as_str())
            .bind_group_layout(&Preprocessor::<G>::BIND_GROUP_LAYOUT_DESCRIPTOR)
            .entry_point("pre")
            .main_shader(
                Preprocessor::<G>::MAIN_SHADER
                    .parse()
                    .expect("preprocess module path"),
            )
            .wesl_compile_options(wesl::CompileOptions {
                features: G::wesl_features(),
                ..Default::default()
            })
            .resolver(wesl_utils::resolver())
            .build_without_bind_groups(device)?;

        let bundle = ComputeBundleBuilder::new()
            .label(Preprocessor::<G>::LABEL)
            .bind_group_layout(&Preprocessor::<G>::BIND_GROUP_LAYOUT_DESCRIPTOR)
            .entry_point("main")
            .main_shader(
                Preprocessor::<G>::MAIN_SHADER
                    .parse()
                    .expect("preprocess module path"),
            )
            .wesl_compile_options(wesl::CompileOptions {
                features: G::wesl_features(),
                ..Default::default()
            })
            .resolver(wesl_utils::resolver())
            .build_without_bind_groups(device)?;

        let post_bundle = ComputeBundleBuilder::new()
            .label(format!("Post {}", Preprocessor::<G>::LABEL).as_str())
            .bind_group_layout(&Preprocessor::<G>::BIND_GROUP_LAYOUT_DESCRIPTOR)
            .entry_point("post")
            .main_shader(
                Preprocessor::<G>::MAIN_SHADER
                    .parse()
                    .expect("preprocess module path"),
            )
            .wesl_compile_options(wesl::CompileOptions {
                features: G::wesl_features(),
                ..Default::default()
            })
            .resolver(wesl_utils::resolver())
            .build_without_bind_groups(device)?;

        log::info!("Preprocessor created");

        Ok(Self {
            bind_group_layout,
            bind_group: (),
            pre_bundle,
            bundle,
            post_bundle,
            gaussian_pod_marker: std::marker::PhantomData,
        })
    }

    /// Preprocess the Gaussians.
    pub fn preprocess(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        gaussian_count: u32,
    ) {
        self.pre_bundle.dispatch(encoder, 1, [bind_group]);

        self.bundle.dispatch(encoder, gaussian_count, [bind_group]);

        self.post_bundle.dispatch(encoder, 1, [bind_group]);
    }
}
