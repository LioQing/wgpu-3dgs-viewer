use std::{collections::HashMap, hash::Hash};

use crate::*;

/// The buffers for [`Viewer`] related to the world.
#[derive(Debug)]
pub struct MultiModelViewerWorldBuffers {
    pub camera_buffer: CameraBuffer,
    pub gaussian_transform_buffer: GaussianTransformBuffer,
}

impl MultiModelViewerWorldBuffers {
    /// Create a new viewer world buffers.
    pub fn new(device: &wgpu::Device) -> Self {
        log::debug!("Creating camera buffer");
        let camera_buffer = CameraBuffer::new(device);

        log::debug!("Creating gaussian transform buffer");
        let gaussian_transform_buffer = GaussianTransformBuffer::new(device);

        Self {
            camera_buffer,
            gaussian_transform_buffer,
        }
    }

    /// Update the camera.
    pub fn update_camera(
        &mut self,
        queue: &wgpu::Queue,
        camera: &impl CameraTrait,
        texture_size: UVec2,
    ) {
        self.camera_buffer.update(queue, camera, texture_size);
    }

    /// Update the camera with [`CameraPod`].
    pub fn update_camera_with_pod(&mut self, queue: &wgpu::Queue, pod: &CameraPod) {
        self.camera_buffer.update_with_pod(queue, pod);
    }

    /// Update the Gaussian transform.
    pub fn update_gaussian_transform(
        &mut self,
        queue: &wgpu::Queue,
        size: f32,
        display_mode: GaussianDisplayMode,
        sh_deg: GaussianShDegree,
        no_sh0: bool,
        std_dev: f32,
    ) {
        self.gaussian_transform_buffer
            .update(queue, size, display_mode, sh_deg, no_sh0, std_dev);
    }

    /// Update the Gaussian transform with [`GaussianTransformPod`].
    pub fn update_gaussian_transform_with_pod(
        &mut self,
        queue: &wgpu::Queue,
        pod: &GaussianTransformPod,
    ) {
        self.gaussian_transform_buffer.update_with_pod(queue, pod);
    }
}

/// The buffers for [`Viewer`] related to the Guassian model.
#[derive(Debug)]
pub struct MultiModelViewerGaussianBuffers<G: GaussianPod = DefaultGaussianPod> {
    pub model_transform_buffer: ModelTransformBuffer,
    pub gaussians_buffer: GaussiansBuffer<G>,
    pub indirect_args_buffer: IndirectArgsBuffer,
    pub radix_sort_indirect_args_buffer: RadixSortIndirectArgsBuffer,
    pub indirect_indices_buffer: IndirectIndicesBuffer,
    pub gaussians_depth_buffer: GaussiansDepthBuffer,
    #[cfg(feature = "viewer-selection")]
    pub selection_buffer: SelectionBuffer,
    #[cfg(feature = "viewer-selection")]
    pub invert_selection_buffer: selection::PreprocessorInvertSelectionBuffer,
}

impl<G: GaussianPod> MultiModelViewerGaussianBuffers<G> {
    /// Create a new viewer Gaussian buffers.
    pub fn new(device: &wgpu::Device, gaussians: &Gaussians) -> Self {
        Self::new_with(device, GaussiansBuffer::<G>::DEFAULT_USAGES, gaussians)
    }

    /// Create a new viewer Gaussian buffers with all the extra options.
    pub fn new_with(
        device: &wgpu::Device,
        gaussians_buffer_usage: wgpu::BufferUsages,
        gaussians: &Gaussians,
    ) -> Self {
        log::debug!("Creating model transform buffer");
        let model_transform_buffer = ModelTransformBuffer::new(device);

        log::debug!("Creating gaussians buffer");
        let gaussians_buffer =
            GaussiansBuffer::new_with_usage(device, &gaussians.gaussians, gaussians_buffer_usage);

        log::debug!("Creating indirect args buffer");
        let indirect_args_buffer = IndirectArgsBuffer::new(device);

        log::debug!("Creating radix sort indirect args buffer");
        let radix_sort_indirect_args_buffer = RadixSortIndirectArgsBuffer::new(device);

        log::debug!("Creating indirect indices buffer");
        let indirect_indices_buffer =
            IndirectIndicesBuffer::new(device, gaussians.gaussians.len() as u32);

        log::debug!("Creating gaussians depth buffer");
        let gaussians_depth_buffer =
            GaussiansDepthBuffer::new(device, gaussians.gaussians.len() as u32);

        #[cfg(feature = "viewer-selection")]
        let selection_buffer = {
            log::debug!("Creating selection buffer");
            SelectionBuffer::new(device, gaussians.gaussians.len() as u32)
        };

        #[cfg(feature = "viewer-selection")]
        let invert_selection_buffer = {
            log::debug!("Creating invert selection buffer");
            selection::PreprocessorInvertSelectionBuffer::new(device)
        };

        Self {
            model_transform_buffer,
            gaussians_buffer,
            indirect_args_buffer,
            radix_sort_indirect_args_buffer,
            indirect_indices_buffer,
            gaussians_depth_buffer,
            #[cfg(feature = "viewer-selection")]
            selection_buffer,
            #[cfg(feature = "viewer-selection")]
            invert_selection_buffer,
        }
    }

    /// Create a new viewer Gaussian buffers with only the count.
    pub fn new_empty(device: &wgpu::Device, count: usize) -> Self {
        log::debug!("Creating model transform buffer");
        let model_transform_buffer = ModelTransformBuffer::new(device);

        log::debug!("Creating gaussians buffer");
        let gaussians_buffer = GaussiansBuffer::new_empty(device, count);

        log::debug!("Creating indirect args buffer");
        let indirect_args_buffer = IndirectArgsBuffer::new(device);

        log::debug!("Creating radix sort indirect args buffer");
        let radix_sort_indirect_args_buffer = RadixSortIndirectArgsBuffer::new(device);

        log::debug!("Creating indirect indices buffer");
        let indirect_indices_buffer = IndirectIndicesBuffer::new(device, count as u32);

        log::debug!("Creating gaussians depth buffer");
        let gaussians_depth_buffer = GaussiansDepthBuffer::new(device, count as u32);

        #[cfg(feature = "viewer-selection")]
        let selection_buffer = {
            log::debug!("Creating selection buffer");
            SelectionBuffer::new(device, count as u32)
        };

        #[cfg(feature = "viewer-selection")]
        let invert_selection_buffer = {
            log::debug!("Creating invert selection buffer");
            selection::PreprocessorInvertSelectionBuffer::new(device)
        };

        Self {
            model_transform_buffer,
            gaussians_buffer,
            indirect_args_buffer,
            radix_sort_indirect_args_buffer,
            indirect_indices_buffer,
            gaussians_depth_buffer,
            #[cfg(feature = "viewer-selection")]
            selection_buffer,
            #[cfg(feature = "viewer-selection")]
            invert_selection_buffer,
        }
    }

    /// Update the model transform.
    pub fn update_model_transform(
        &mut self,
        queue: &wgpu::Queue,
        pos: Vec3,
        rot: Quat,
        scale: Vec3,
    ) {
        self.model_transform_buffer.update(queue, pos, rot, scale);
    }

    /// Update the model transform with [`ModelTransformPod`].
    pub fn update_model_transform_with_pod(
        &mut self,
        queue: &wgpu::Queue,
        pod: &ModelTransformPod,
    ) {
        self.model_transform_buffer.update_with_pod(queue, pod);
    }
}

/// The bind groups for [`MultiModelViewer`].
#[derive(Debug)]
pub struct MultiModelViewerBindGroups {
    pub preprocessor: wgpu::BindGroup,
    pub radix_sorter: RadixSorterBindGroups,
    pub renderer: wgpu::BindGroup,
}

impl MultiModelViewerBindGroups {
    /// Create a new viewer bind groups.
    pub fn new<G: GaussianPod>(
        device: &wgpu::Device,
        preprocessor: &Preprocessor<G, ()>,
        radix_sorter: &RadixSorter<()>,
        renderer: &Renderer<G, ()>,
        gaussian_buffers: &MultiModelViewerGaussianBuffers<G>,
        world_buffers: &MultiModelViewerWorldBuffers,
    ) -> Self {
        let preprocessor = preprocessor.create_bind_group(
            device,
            &world_buffers.camera_buffer,
            &gaussian_buffers.model_transform_buffer,
            &world_buffers.gaussian_transform_buffer,
            &gaussian_buffers.gaussians_buffer,
            &gaussian_buffers.indirect_args_buffer,
            &gaussian_buffers.radix_sort_indirect_args_buffer,
            &gaussian_buffers.indirect_indices_buffer,
            &gaussian_buffers.gaussians_depth_buffer,
            #[cfg(feature = "viewer-selection")]
            &gaussian_buffers.selection_buffer,
            #[cfg(feature = "viewer-selection")]
            &gaussian_buffers.invert_selection_buffer,
        );
        let radix_sorter = radix_sorter.create_bind_groups(
            device,
            &gaussian_buffers.gaussians_depth_buffer,
            &gaussian_buffers.indirect_indices_buffer,
        );
        let renderer = renderer.create_bind_group(
            device,
            &world_buffers.camera_buffer,
            &gaussian_buffers.model_transform_buffer,
            &world_buffers.gaussian_transform_buffer,
            &gaussian_buffers.gaussians_buffer,
            &gaussian_buffers.indirect_indices_buffer,
        );

        Self {
            preprocessor,
            radix_sorter,
            renderer,
        }
    }
}

/// The model of the [`MultiModelViewer`].
#[derive(Debug)]
pub struct MultiModelViewerModel<G: GaussianPod = DefaultGaussianPod> {
    /// Buffers for the model.
    pub gaussian_buffers: MultiModelViewerGaussianBuffers<G>,

    /// Bind groups for the model.
    pub bind_groups: MultiModelViewerBindGroups,
}

/// The 3D Gaussian splatting viewer for multiple models.
#[derive(Debug)]
pub struct MultiModelViewer<G: GaussianPod = DefaultGaussianPod, K: Hash + std::cmp::Eq = String> {
    pub models: HashMap<K, MultiModelViewerModel<G>>,
    pub world_buffers: MultiModelViewerWorldBuffers,
    pub preprocessor: Preprocessor<G, ()>,
    pub radix_sorter: RadixSorter<()>,
    pub renderer: Renderer<G, ()>,
}

impl<G: GaussianPod, K: Hash + std::cmp::Eq> MultiModelViewer<G, K> {
    /// Create a new viewer.
    pub fn new(
        device: &wgpu::Device,
        texture_format: wgpu::TextureFormat,
    ) -> Result<Self, ViewerCreateError> {
        Self::new_with(device, texture_format, None)
    }

    /// Create a new viewer with all extra options.
    ///
    /// More specifically, you can specify:
    /// - `depth_stencil`: The optional depth stencil state for the renderer.
    pub fn new_with(
        device: &wgpu::Device,
        texture_format: wgpu::TextureFormat,
        depth_stencil: Option<wgpu::DepthStencilState>,
    ) -> Result<Self, ViewerCreateError> {
        let models = HashMap::new();

        log::debug!("Creating world buffers");
        let world_buffers = MultiModelViewerWorldBuffers::new(device);

        log::debug!("Creating preprocessor");
        let preprocessor = Preprocessor::new_without_bind_group(device)?;

        log::debug!("Creating radix sorter");
        let radix_sorter = RadixSorter::new_without_bind_groups(device);

        log::debug!("Creating renderer");
        let renderer = Renderer::new_without_bind_group(device, texture_format, depth_stencil)?;

        log::info!("Viewer created");

        Ok(Self {
            models,
            world_buffers,
            preprocessor,
            radix_sorter,
            renderer,
        })
    }

    /// Insert a new model to the viewer.
    pub fn insert_model(
        &mut self,
        device: &wgpu::Device,
        key: K,
        gaussians: &Gaussians,
    ) -> Option<MultiModelViewerModel<G>> {
        self.insert_model_with(device, key, GaussiansBuffer::<G>::DEFAULT_USAGES, gaussians)
    }

    /// Insert a new model to the viewer with all extra options.
    ///
    /// More specifically, you can specify:
    /// - `gaussians_buffer_usage`: The usage for the gaussians buffer.
    pub fn insert_model_with(
        &mut self,
        device: &wgpu::Device,
        key: K,
        gaussians_buffer_usage: wgpu::BufferUsages,
        gaussians: &Gaussians,
    ) -> Option<MultiModelViewerModel<G>> {
        let gaussian_buffers =
            MultiModelViewerGaussianBuffers::new_with(device, gaussians_buffer_usage, gaussians);
        let bind_groups = MultiModelViewerBindGroups::new(
            device,
            &self.preprocessor,
            &self.radix_sorter,
            &self.renderer,
            &gaussian_buffers,
            &self.world_buffers,
        );
        self.models.insert(
            key,
            MultiModelViewerModel {
                gaussian_buffers,
                bind_groups,
            },
        )
    }

    /// Remove a model from the viewer.
    pub fn remove_model(&mut self, key: &K) -> Option<MultiModelViewerModel<G>> {
        self.models.remove(key)
    }

    /// Update the camera.
    pub fn update_camera(
        &mut self,
        queue: &wgpu::Queue,
        camera: &impl CameraTrait,
        texture_size: UVec2,
    ) {
        self.world_buffers
            .update_camera(queue, camera, texture_size);
    }

    /// Update the camera with [`CameraPod`].
    pub fn update_camera_with_pod(&mut self, queue: &wgpu::Queue, pod: &CameraPod) {
        self.world_buffers.update_camera_with_pod(queue, pod);
    }

    /// Update the model transform.
    pub fn update_model_transform(
        &mut self,
        queue: &wgpu::Queue,
        key: &K,
        pos: Vec3,
        rot: Quat,
        scale: Vec3,
    ) -> Result<(), MultiModelViewerAccessError> {
        self.models
            .get_mut(key)
            .ok_or(MultiModelViewerAccessError::ModelNotFound)?
            .gaussian_buffers
            .update_model_transform(queue, pos, rot, scale);
        Ok(())
    }

    /// Update the model transform with [`ModelTransformPod`].
    pub fn update_model_transform_with_pod(
        &mut self,
        queue: &wgpu::Queue,
        key: &K,
        pod: &ModelTransformPod,
    ) -> Result<(), MultiModelViewerAccessError> {
        self.models
            .get_mut(key)
            .ok_or(MultiModelViewerAccessError::ModelNotFound)?
            .gaussian_buffers
            .update_model_transform_with_pod(queue, pod);
        Ok(())
    }

    /// Update the Gaussian transform.
    pub fn update_gaussian_transform(
        &mut self,
        queue: &wgpu::Queue,
        size: f32,
        display_mode: GaussianDisplayMode,
        sh_deg: GaussianShDegree,
        no_sh0: bool,
        std_dev: f32,
    ) {
        self.world_buffers.update_gaussian_transform(
            queue,
            size,
            display_mode,
            sh_deg,
            no_sh0,
            std_dev,
        );
    }

    /// Update the Gaussian transform with [`GaussianTransformPod`].
    pub fn update_gaussian_transform_with_pod(
        &mut self,
        queue: &wgpu::Queue,
        pod: &GaussianTransformPod,
    ) {
        self.world_buffers
            .update_gaussian_transform_with_pod(queue, pod);
    }

    /// Render the viewer.
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        texture_view: &wgpu::TextureView,
        keys: &[&K],
    ) -> Result<(), MultiModelViewerAccessError> {
        let models = keys
            .iter()
            .map(|key| {
                self.models
                    .get(key)
                    .ok_or(MultiModelViewerAccessError::ModelNotFound)
            })
            .collect::<Result<Vec<_>, _>>()?;

        for model in models.iter() {
            self.preprocessor.preprocess(
                encoder,
                &model.bind_groups.preprocessor,
                model.gaussian_buffers.gaussians_buffer.len() as u32,
            );

            self.radix_sorter.sort(
                encoder,
                &model.bind_groups.radix_sorter,
                &model.gaussian_buffers.radix_sort_indirect_args_buffer,
            );
        }

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Multi Model Viewer Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: texture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            for model in models.iter() {
                self.renderer.render_with_pass(
                    &mut render_pass,
                    &model.bind_groups.renderer,
                    &model.gaussian_buffers.indirect_args_buffer,
                );
            }
        }

        Ok(())
    }
}
