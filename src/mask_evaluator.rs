use glam::*;

use crate::{
    GaussianCov3dConfig, GaussianPod, GaussianShConfig, GaussiansBuffer, MaskBuffer, MaskOp,
    MaskOpBuffer, MaskOpShapeBuffer, MaskOpShapePod, ModelTransformBuffer,
};

/// A mask operation tree.
#[derive(Debug)]
pub enum MaskOpTree<'a> {
    Union(Box<MaskOpTree<'a>>, Box<MaskOpTree<'a>>),
    Intersection(Box<MaskOpTree<'a>>, Box<MaskOpTree<'a>>),
    Difference(Box<MaskOpTree<'a>>, Box<MaskOpTree<'a>>),
    Complement(Box<MaskOpTree<'a>>),
    Shape(&'a MaskOpShapePod),
}

impl<'a> MaskOpTree<'a> {
    /// Create a new [`MaskOpTree::Shape`].
    pub fn shape(shape: &'a MaskOpShapePod) -> Self {
        Self::Shape(shape)
    }

    /// Create a new [`MaskOpTree::Union`].
    pub fn union(self, other: Self) -> Self {
        Self::Union(Box::new(self), Box::new(other))
    }

    /// Create a new [`MaskOpTree::Intersection`].
    pub fn intersection(self, other: Self) -> Self {
        Self::Intersection(Box::new(self), Box::new(other))
    }

    /// Create a new [`MaskOpTree::Difference`].
    pub fn difference(self, other: Self) -> Self {
        Self::Difference(Box::new(self), Box::new(other))
    }

    /// Create a new [`MaskOpTree::Complement`].
    pub fn complement(self) -> Self {
        Self::Complement(Box::new(self))
    }

    /// Get a vector over all [`MaskOpTree::Shape`]s in the tree.
    pub fn shapes(&self) -> Vec<&'a MaskOpShapePod> {
        let mut shapes = Vec::new();
        self.shapes_recursive(&mut shapes);
        shapes
    }

    /// Recursively get all [`MaskOpTree::Shape`]s in the tree.
    fn shapes_recursive(&self, shapes: &mut Vec<&'a MaskOpShapePod>) {
        match self {
            MaskOpTree::Union(left, right) => {
                left.shapes_recursive(shapes);
                right.shapes_recursive(shapes);
            }
            MaskOpTree::Intersection(left, right) => {
                left.shapes_recursive(shapes);
                right.shapes_recursive(shapes);
            }
            MaskOpTree::Difference(left, right) => {
                left.shapes_recursive(shapes);
                right.shapes_recursive(shapes);
            }
            MaskOpTree::Complement(inner) => inner.shapes_recursive(shapes),
            MaskOpTree::Shape(shape) => shapes.push(shape),
        }
    }
}

/// A mask evaluator for applying [`MaskOpTree`].
#[derive(Debug)]
pub struct MaskEvaluator {
    /// The workgroup size.
    workgroup_size: UVec3,

    /// The bind group layout.
    bind_group_layout: wgpu::BindGroupLayout,
    /// The pipeline.
    pipeline: wgpu::ComputePipeline,

    /// The shape bind group layout.
    shape_bind_group_layout: wgpu::BindGroupLayout,
    /// The shape pipeline.
    shape_pipeline: wgpu::ComputePipeline,
}

impl MaskEvaluator {
    /// The bind group layout descriptor.
    pub const BIND_GROUP_LAYOUT_DESCRIPTOR: wgpu::BindGroupLayoutDescriptor<'static> =
        wgpu::BindGroupLayoutDescriptor {
            label: Some("Mask Evaluator Bind Group Layout"),
            entries: &[
                // Mask operation buffer
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
                // Source mask buffer
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
                // Destination mask buffer
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
            ],
        };

    /// The bshape ind group layout descriptor.
    pub const SHAPE_BIND_GROUP_LAYOUT_DESCRIPTOR: wgpu::BindGroupLayoutDescriptor<'static> =
        wgpu::BindGroupLayoutDescriptor {
            label: Some("Mask Evaluator Shape Bind Group Layout"),
            entries: &[
                Self::BIND_GROUP_LAYOUT_DESCRIPTOR.entries[0], // Mask operation buffer
                // Shape buffer
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
                Self::BIND_GROUP_LAYOUT_DESCRIPTOR.entries[2], // Destination mask buffer
                // Model transform buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Gaussians buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        };

    /// Create a new mask evaluator.
    pub fn new<G: GaussianPod>(device: &wgpu::Device) -> Self {
        let workgroup_size = uvec3(
            device
                .limits()
                .max_compute_workgroup_size_x
                .min(device.limits().max_compute_invocations_per_workgroup),
            1,
            1,
        );

        log::debug!("Creating mask evaluator bind group layout");
        let bind_group_layout =
            device.create_bind_group_layout(&Self::BIND_GROUP_LAYOUT_DESCRIPTOR);

        log::debug!("Creating mask evaluation shader");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Mask Evaluation Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shader/mask_evaluation.wgsl")
                    .replace(
                        "{{workgroup_size}}",
                        format!(
                            "{}, {}, {}",
                            workgroup_size.x, workgroup_size.y, workgroup_size.z
                        )
                        .as_str(),
                    )
                    .into(),
            ),
        });

        log::debug!("Creating mask evaluator pipeline layout");
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Mask Evaluator Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        log::debug!("Creating mask evaluator pipeline");
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Mask Evaluator Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        log::debug!("Creating mask evaluator shape bind group layout");
        let shape_bind_group_layout =
            device.create_bind_group_layout(&Self::SHAPE_BIND_GROUP_LAYOUT_DESCRIPTOR);

        log::debug!("Creating mask evaluator shape shader");
        let shape_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Mask Evaluation Shape Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shader/mask_evaluation_shape.wgsl")
                    .replace(
                        "{{workgroup_size}}",
                        format!(
                            "{}, {}, {}",
                            workgroup_size.x, workgroup_size.y, workgroup_size.z
                        )
                        .as_str(),
                    )
                    .replace("{{gaussian_sh_field}}", G::ShConfig::sh_field())
                    .replace("{{gaussian_cov3d_field}}", G::Cov3dConfig::cov3d_field())
                    .into(),
            ),
        });

        log::debug!("Creating mask evaluator shape pipeline layout");
        let shape_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Mask Evaluator Shape Pipeline Layout"),
                bind_group_layouts: &[&shape_bind_group_layout],
                push_constant_ranges: &[],
            });

        log::debug!("Creating mask evaluator shape pipeline");
        let shape_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Mask Evaluator Shape Pipeline"),
            layout: Some(&shape_pipeline_layout),
            module: &shape_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            workgroup_size,

            bind_group_layout,
            pipeline,

            shape_bind_group_layout,
            shape_pipeline,
        }
    }

    /// Create the bind group.
    pub fn create_bind_group(
        &self,
        device: &wgpu::Device,
        op: &MaskOpBuffer,
        source: &MaskBuffer,
        dest: &MaskBuffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Mask Evaluator Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: op.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: source.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dest.buffer().as_entire_binding(),
                },
            ],
        })
    }

    /// Create the shape bind group.
    pub fn create_shape_bind_group<G: GaussianPod>(
        &self,
        device: &wgpu::Device,
        op: &MaskOpBuffer,
        shape: &MaskOpShapeBuffer,
        dest: &MaskBuffer,
        model_transform: &ModelTransformBuffer,
        gaussians: &GaussiansBuffer<G>,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Mask Evaluator Shape Bind Group"),
            layout: &self.shape_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: op.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: shape.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dest.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: model_transform.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: gaussians.buffer().as_entire_binding(),
                },
            ],
        })
    }

    /// Evaluate a [`MaskOpTree`] into a mask buffer.
    pub fn evaluate<G: GaussianPod>(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        tree: &MaskOpTree,
        mask: &MaskBuffer,
        model_transform: &ModelTransformBuffer,
        gaussians: &GaussiansBuffer<G>,
    ) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Mask Evaluation Command Encoder"),
        });

        self.evaluate_with_encoder(
            device,
            queue,
            &mut encoder,
            tree,
            mask,
            model_transform,
            gaussians,
        );

        queue.submit(Some(encoder.finish()));
    }

    /// Evaluate a [`MaskOpTree`] into a mask buffer with encoder.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_with_encoder<G: GaussianPod>(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        tree: &MaskOpTree,
        mask: &MaskBuffer,
        model_transform: &ModelTransformBuffer,
        gaussians: &GaussiansBuffer<G>,
    ) {
        let bind_group = match tree {
            MaskOpTree::Union(left, right) => {
                let op = MaskOpBuffer::new(device, MaskOp::Union);
                let source = MaskBuffer::new_with_label(device, "Source", gaussians.len() as u32);

                self.evaluate_with_encoder(
                    device,
                    queue,
                    encoder,
                    left,
                    mask,
                    model_transform,
                    gaussians,
                );
                self.evaluate_with_encoder(
                    device,
                    queue,
                    encoder,
                    right,
                    &source,
                    model_transform,
                    gaussians,
                );

                self.create_bind_group(device, &op, &source, mask)
            }
            MaskOpTree::Intersection(left, right) => {
                let op = MaskOpBuffer::new(device, MaskOp::Intersection);
                let source = MaskBuffer::new_with_label(device, "Source", gaussians.len() as u32);

                self.evaluate_with_encoder(
                    device,
                    queue,
                    encoder,
                    left,
                    mask,
                    model_transform,
                    gaussians,
                );
                self.evaluate_with_encoder(
                    device,
                    queue,
                    encoder,
                    right,
                    &source,
                    model_transform,
                    gaussians,
                );

                self.create_bind_group(device, &op, &source, mask)
            }
            MaskOpTree::Difference(left, right) => {
                let op = MaskOpBuffer::new(device, MaskOp::Difference);
                let source = MaskBuffer::new_with_label(device, "Source", gaussians.len() as u32);

                self.evaluate_with_encoder(
                    device,
                    queue,
                    encoder,
                    left,
                    mask,
                    model_transform,
                    gaussians,
                );
                self.evaluate_with_encoder(
                    device,
                    queue,
                    encoder,
                    right,
                    &source,
                    model_transform,
                    gaussians,
                );

                op.update(queue, MaskOp::Difference);

                self.create_bind_group(device, &op, &source, mask)
            }
            MaskOpTree::Complement(inner) => {
                let op = MaskOpBuffer::new(device, MaskOp::Complement);
                let source = MaskBuffer::new_with_label(device, "Source", gaussians.len() as u32);

                self.evaluate_with_encoder(
                    device,
                    queue,
                    encoder,
                    inner,
                    mask,
                    model_transform,
                    gaussians,
                );

                self.create_bind_group(device, &op, &source, mask)
            }
            MaskOpTree::Shape(shape) => {
                let op = MaskOpBuffer::new(device, MaskOp::Shape);
                let shape = MaskOpShapeBuffer::new(device, shape);

                self.create_shape_bind_group(device, &op, &shape, mask, model_transform, gaussians)
            }
        };

        let gaussian_count = gaussians.len() as u32;
        if let MaskOpTree::Shape(..) = tree {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Mask Shape Evaluation Compute Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.shape_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(gaussian_count.div_ceil(self.workgroup_count()), 1, 1);
        } else {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Mask Evaluation Compute Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(
                gaussian_count.div_ceil(32).div_ceil(self.workgroup_count()),
                1,
                1,
            );
        }
    }

    /// Get the number of invocations in one workgroup.
    fn workgroup_count(&self) -> u32 {
        self.workgroup_size.x * self.workgroup_size.y * self.workgroup_size.z
    }
}
