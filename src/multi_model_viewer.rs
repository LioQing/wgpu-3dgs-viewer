use std::{collections::HashMap, hash::Hash};

use crate::*;

/// The buffers for [`Viewer`] related to the world.
#[derive(Debug)]
pub struct MultiModelViewerWorldBuffers {
    pub camera_buffer: CameraBuffer,
    pub gaussian_transform_buffer: GaussianTransformBuffer,
    pub query_buffer: QueryBuffer,
    pub selection_highlight_buffer: SelectionHighlightBuffer,
    pub selection_edit_buffer: SelectionEditBuffer,

    #[cfg(feature = "query-texture")]
    pub query_texture: QueryTexture,
}

impl MultiModelViewerWorldBuffers {
    /// Create a new viewer world buffers.
    pub fn new(
        device: &wgpu::Device,
        #[cfg(feature = "query-texture")] texture_size: UVec2,
    ) -> Self {
        log::debug!("Creating camera buffer");
        let camera_buffer = CameraBuffer::new(device);

        log::debug!("Creating gaussian transform buffer");
        let gaussian_transform_buffer = GaussianTransformBuffer::new(device);

        log::debug!("Creating query buffer");
        let query_buffer = QueryBuffer::new(device);

        log::debug!("Creating selection highlight buffer");
        let selection_highlight_buffer = SelectionHighlightBuffer::new(device);

        log::debug!("Creating selection edit buffer");
        let selection_edit_buffer = SelectionEditBuffer::new(device);

        #[cfg(feature = "query-texture")]
        let query_texture = {
            log::debug!("Creating query texture");
            QueryTexture::new(device, texture_size)
        };

        Self {
            camera_buffer,
            gaussian_transform_buffer,
            query_buffer,
            selection_highlight_buffer,
            selection_edit_buffer,

            #[cfg(feature = "query-texture")]
            query_texture,
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

    /// Update the query.
    ///
    /// There can only be one query at a time.
    pub fn update_query(&mut self, queue: &wgpu::Queue, query: &QueryPod) {
        self.query_buffer.update(queue, query);
    }

    /// Update the Gaussian transform.
    pub fn update_gaussian_transform(
        &mut self,
        queue: &wgpu::Queue,
        size: f32,
        display_mode: GaussianDisplayMode,
        sh_deg: GaussianShDegree,
        no_sh0: bool,
    ) {
        self.gaussian_transform_buffer
            .update(queue, size, display_mode, sh_deg, no_sh0);
    }

    /// Update the Gaussian transform with [`GaussianTransformPod`].
    pub fn update_gaussian_transform_with_pod(
        &mut self,
        queue: &wgpu::Queue,
        pod: &GaussianTransformPod,
    ) {
        self.gaussian_transform_buffer.update_with_pod(queue, pod);
    }

    /// Update the selection highlight.
    pub fn update_selection_highlight(&mut self, queue: &wgpu::Queue, color: Vec4) {
        self.selection_highlight_buffer.update(queue, color);
    }

    /// Update the selection highlight with [`SelectionHighlightPod`].
    pub fn update_selection_highlight_with_pod(
        &mut self,
        queue: &wgpu::Queue,
        pod: &SelectionHighlightPod,
    ) {
        self.selection_highlight_buffer.update_with_pod(queue, pod);
    }

    /// Update the selection edit.
    ///
    /// Set [`GaussianEditFlag::ENABLED`] to apply the edits on the selected Gaussians.
    #[allow(clippy::too_many_arguments)]
    pub fn update_selection_edit(
        &mut self,
        queue: &wgpu::Queue,
        flag: GaussianEditFlag,
        hsv: Vec3,
        contrast: f32,
        exposure: f32,
        gamma: f32,
        alpha: f32,
    ) {
        self.selection_edit_buffer
            .update(queue, flag, hsv, contrast, exposure, gamma, alpha);
    }

    /// Update the selection edit with [`GaussianEditPod`].
    pub fn update_selection_edit_with_pod(&mut self, queue: &wgpu::Queue, pod: &GaussianEditPod) {
        self.selection_edit_buffer.update_with_pod(queue, pod);
    }

    /// Update the query texture size.
    ///
    /// This requires the `query-texture` feature.
    #[cfg(feature = "query-texture")]
    pub fn update_query_texture_size(&mut self, device: &wgpu::Device, size: UVec2) {
        self.query_texture.update_size(device, size);
    }
}

/// The buffers for [`Viewer`] related to the Guassian model.
#[derive(Debug)]
pub struct MultiModelViewerGaussianBuffers<G: GaussianPod = GaussianPodWithShNorm8Cov3dHalfConfigs>
{
    pub model_transform_buffer: ModelTransformBuffer,
    pub gaussians_buffer: GaussiansBuffer<G>,
    pub indirect_args_buffer: IndirectArgsBuffer,
    pub radix_sort_indirect_args_buffer: RadixSortIndirectArgsBuffer,
    pub indirect_indices_buffer: IndirectIndicesBuffer,
    pub gaussians_depth_buffer: GaussiansDepthBuffer,
    pub query_result_count_buffer: QueryResultCountBuffer,
    pub query_results_buffer: QueryResultsBuffer,
    pub postprocess_indirect_args_buffer: PostprocessIndirectArgsBuffer,
    pub selection_buffer: SelectionBuffer,
    pub gaussians_edit_buffer: GaussiansEditBuffer,
}

impl<G: GaussianPod> MultiModelViewerGaussianBuffers<G> {
    /// Create a new viewer Gaussian buffers.
    pub fn new(device: &wgpu::Device, gaussians: &Gaussians) -> Self {
        log::debug!("Creating model transform buffer");
        let model_transform_buffer = ModelTransformBuffer::new(device);

        log::debug!("Creating gaussians buffer");
        let gaussians_buffer = GaussiansBuffer::new(device, &gaussians.gaussians);

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

        log::debug!("Creating query result count buffer");
        let query_result_count_buffer = QueryResultCountBuffer::new(device);

        log::debug!("Creating query results buffer");
        let query_results_buffer =
            QueryResultsBuffer::new(device, gaussians.gaussians.len() as u32);

        log::debug!("Creating postprocess indirect args buffer");
        let postprocess_indirect_args_buffer = PostprocessIndirectArgsBuffer::new(device);

        log::debug!("Creating selection buffer");
        let selection_buffer = SelectionBuffer::new(device, gaussians.gaussians.len() as u32);

        log::debug!("Creating gaussians edit buffer");
        let gaussians_edit_buffer =
            GaussiansEditBuffer::new(device, gaussians.gaussians.len() as u32);

        Self {
            model_transform_buffer,
            gaussians_buffer,
            indirect_args_buffer,
            radix_sort_indirect_args_buffer,
            indirect_indices_buffer,
            gaussians_depth_buffer,
            query_result_count_buffer,
            query_results_buffer,
            postprocess_indirect_args_buffer,
            selection_buffer,
            gaussians_edit_buffer,
        }
    }

    /// Update the model transform.
    pub fn update_model_transform(
        &mut self,
        queue: &wgpu::Queue,
        pos: Vec3,
        quat: Quat,
        scale: Vec3,
    ) {
        self.model_transform_buffer.update(queue, pos, quat, scale);
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
    pub postprocessor: (wgpu::BindGroup, wgpu::BindGroup),
}

impl MultiModelViewerBindGroups {
    /// Create a new viewer bind groups.
    pub fn new<G: GaussianPod>(
        device: &wgpu::Device,
        preprocessor: &Preprocessor<()>,
        radix_sorter: &RadixSorter<()>,
        renderer: &Renderer<()>,
        postprocessor: &Postprocessor<()>,
        gaussian_buffers: &MultiModelViewerGaussianBuffers<G>,
        world_buffers: &MultiModelViewerWorldBuffers,
    ) -> Self {
        let preprocessor = preprocessor.create_bind_group(
            device,
            &world_buffers.camera_buffer,
            &gaussian_buffers.model_transform_buffer,
            &gaussian_buffers.gaussians_buffer,
            &gaussian_buffers.indirect_args_buffer,
            &gaussian_buffers.radix_sort_indirect_args_buffer,
            &gaussian_buffers.indirect_indices_buffer,
            &gaussian_buffers.gaussians_depth_buffer,
            &world_buffers.query_buffer,
            &gaussian_buffers.query_result_count_buffer,
            &gaussian_buffers.query_results_buffer,
            &gaussian_buffers.gaussians_edit_buffer,
            &gaussian_buffers.selection_buffer,
            &world_buffers.selection_edit_buffer,
            #[cfg(feature = "query-texture")]
            &world_buffers.query_texture,
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
            &world_buffers.query_buffer,
            &gaussian_buffers.query_result_count_buffer,
            &gaussian_buffers.query_results_buffer,
            &world_buffers.selection_highlight_buffer,
            &gaussian_buffers.selection_buffer,
            &gaussian_buffers.gaussians_edit_buffer,
        );
        let postprocessor = postprocessor.create_bind_groups(
            device,
            &gaussian_buffers.postprocess_indirect_args_buffer,
            &world_buffers.query_buffer,
            &gaussian_buffers.query_result_count_buffer,
            &gaussian_buffers.query_results_buffer,
            &gaussian_buffers.selection_buffer,
        );

        Self {
            preprocessor,
            radix_sorter,
            renderer,
            postprocessor,
        }
    }
}

/// The model of the [`MultiModelViewer`].
#[derive(Debug)]
pub struct MultiModelViewerModel<G: GaussianPod = GaussianPodWithShNorm8Cov3dHalfConfigs> {
    /// Buffers for the model.
    pub gaussian_buffers: MultiModelViewerGaussianBuffers<G>,

    /// Bind groups for the model.
    pub bind_groups: MultiModelViewerBindGroups,
}

/// The 3D Gaussian splatting viewer for multiple models.
#[derive(Debug)]
pub struct MultiModelViewer<
    G: GaussianPod = GaussianPodWithShNorm8Cov3dHalfConfigs,
    K: Hash + std::cmp::Eq = String,
> {
    pub models: HashMap<K, MultiModelViewerModel<G>>,
    pub world_buffers: MultiModelViewerWorldBuffers,
    pub preprocessor: Preprocessor<()>,
    pub radix_sorter: RadixSorter<()>,
    pub renderer: Renderer<()>,
    pub postprocessor: Postprocessor<()>,
}

impl<G: GaussianPod, K: Hash + std::cmp::Eq> MultiModelViewer<G, K> {
    /// Create a new viewer.
    pub fn new(
        device: &wgpu::Device,
        texture_format: wgpu::TextureFormat,
        #[cfg(feature = "query-texture")] texture_size: UVec2,
    ) -> Self {
        Self::new_with(
            device,
            texture_format,
            None,
            #[cfg(feature = "query-texture")]
            texture_size,
        )
    }

    /// Create a new viewer with all options.
    pub fn new_with(
        device: &wgpu::Device,
        texture_format: wgpu::TextureFormat,
        depth_stencil: Option<wgpu::DepthStencilState>,
        #[cfg(feature = "query-texture")] texture_size: UVec2,
    ) -> Self {
        let models = HashMap::new();

        log::debug!("Creating world buffers");
        let world_buffers = MultiModelViewerWorldBuffers::new(device, texture_size);

        log::debug!("Creating preprocessor");
        let preprocessor = Preprocessor::new_without_bind_group::<G>(device);

        log::debug!("Creating radix sorter");
        let radix_sorter = RadixSorter::new_without_bind_groups(device);

        log::debug!("Creating renderer");
        let renderer = Renderer::new_without_bind_group::<G>(device, texture_format, depth_stencil);

        log::debug!("Creating postprocessor");
        let postprocessor = Postprocessor::new_without_bind_groups(device);

        log::info!("Viewer created");

        Self {
            models,
            world_buffers,
            preprocessor,
            radix_sorter,
            renderer,
            postprocessor,
        }
    }

    /// Insert a new model to the viewer.
    pub fn insert_model(&mut self, device: &wgpu::Device, key: K, gaussians: &Gaussians) {
        let gaussian_buffers = MultiModelViewerGaussianBuffers::new(device, gaussians);
        let bind_groups = MultiModelViewerBindGroups::new(
            device,
            &self.preprocessor,
            &self.radix_sorter,
            &self.renderer,
            &self.postprocessor,
            &gaussian_buffers,
            &self.world_buffers,
        );
        self.models.insert(
            key,
            MultiModelViewerModel {
                gaussian_buffers,
                bind_groups,
            },
        );
    }

    /// Insert models from an iterator.
    pub fn insert_models<'a>(
        &mut self,
        device: &wgpu::Device,
        iter: impl IntoIterator<Item = (K, &'a Gaussians)>,
    ) {
        for (key, gaussians) in iter {
            self.insert_model(device, key, gaussians);
        }
    }

    /// Remove a model from the viewer.
    pub fn remove_model(&mut self, key: &K) {
        self.models.remove(key);
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

    /// Update the query.
    ///
    /// There can only be one query at a time.
    pub fn update_query(&mut self, queue: &wgpu::Queue, query: &QueryPod) {
        self.world_buffers.update_query(queue, query);
    }

    /// Update the model transform.
    pub fn update_model_transform(
        &mut self,
        queue: &wgpu::Queue,
        key: &K,
        pos: Vec3,
        quat: Quat,
        scale: Vec3,
    ) {
        self.models
            .get_mut(key)
            .expect("model not found")
            .gaussian_buffers
            .update_model_transform(queue, pos, quat, scale);
    }

    /// Update the model transform with [`ModelTransformPod`].
    pub fn update_model_transform_with_pod(
        &mut self,
        queue: &wgpu::Queue,
        key: &K,
        pod: &ModelTransformPod,
    ) {
        self.models
            .get_mut(key)
            .expect("model not found")
            .gaussian_buffers
            .update_model_transform_with_pod(queue, pod);
    }

    /// Update the Gaussian transform.
    pub fn update_gaussian_transform(
        &mut self,
        queue: &wgpu::Queue,
        size: f32,
        display_mode: GaussianDisplayMode,
        sh_deg: GaussianShDegree,
        no_sh0: bool,
    ) {
        self.world_buffers.gaussian_transform_buffer.update(
            queue,
            size,
            display_mode,
            sh_deg,
            no_sh0,
        );
    }

    /// Update the Gaussian transform with [`GaussianTransformPod`].
    pub fn update_gaussian_transform_with_pod(
        &mut self,
        queue: &wgpu::Queue,
        pod: &GaussianTransformPod,
    ) {
        self.world_buffers
            .gaussian_transform_buffer
            .update_with_pod(queue, pod);
    }

    /// Update the selection highlight.
    pub fn update_selection_highlight(&mut self, queue: &wgpu::Queue, color: Vec4) {
        self.world_buffers
            .selection_highlight_buffer
            .update(queue, color);
    }

    /// Update the selection highlight with [`SelectionHighlightPod`].
    pub fn update_selection_highlight_with_pod(
        &mut self,
        queue: &wgpu::Queue,
        pod: &SelectionHighlightPod,
    ) {
        self.world_buffers
            .selection_highlight_buffer
            .update_with_pod(queue, pod);
    }

    /// Update the selection edit.
    ///
    /// Set [`GaussianEditFlag::ENABLED`] to apply the edits on the selected Gaussians.
    #[allow(clippy::too_many_arguments)]
    pub fn update_selection_edit(
        &mut self,
        queue: &wgpu::Queue,
        flag: GaussianEditFlag,
        hsv: Vec3,
        contrast: f32,
        exposure: f32,
        gamma: f32,
        alpha: f32,
    ) {
        self.world_buffers
            .selection_edit_buffer
            .update(queue, flag, hsv, contrast, exposure, gamma, alpha);
    }

    /// Update the selection edit with [`GaussianEditPod`].
    pub fn update_selection_edit_with_pod(&mut self, queue: &wgpu::Queue, pod: &GaussianEditPod) {
        self.world_buffers
            .selection_edit_buffer
            .update_with_pod(queue, pod);
    }

    /// Update the query texture size.
    ///
    /// This requires the `query-texture` feature.
    #[cfg(feature = "query-texture")]
    pub fn update_query_texture_size(&mut self, device: &wgpu::Device, size: UVec2) {
        self.world_buffers.update_query_texture_size(device, size);
        for model in self.models.values_mut() {
            model.bind_groups.preprocessor = self.preprocessor.create_bind_group(
                device,
                &self.world_buffers.camera_buffer,
                &model.gaussian_buffers.model_transform_buffer,
                &model.gaussian_buffers.gaussians_buffer,
                &model.gaussian_buffers.indirect_args_buffer,
                &model.gaussian_buffers.radix_sort_indirect_args_buffer,
                &model.gaussian_buffers.indirect_indices_buffer,
                &model.gaussian_buffers.gaussians_depth_buffer,
                &self.world_buffers.query_buffer,
                &model.gaussian_buffers.query_result_count_buffer,
                &model.gaussian_buffers.query_results_buffer,
                &model.gaussian_buffers.gaussians_edit_buffer,
                &model.gaussian_buffers.selection_buffer,
                &self.world_buffers.selection_edit_buffer,
                &self.world_buffers.query_texture,
            );
        }
    }

    /// Render the viewer.
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        texture_view: &wgpu::TextureView,
        keys: &[&K],
    ) -> Result<(), Error> {
        if keys.len() != self.models.len() {
            return Err(Error::ModelCountKeysLenMismatch {
                model_count: self.models.len(),
                keys_len: keys.len(),
            });
        }

        for model in self.models.values() {
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
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            for key in keys.iter() {
                let model = &self.models.get(key).expect("model not found");
                self.renderer.render_with_pass(
                    &mut render_pass,
                    &model.bind_groups.renderer,
                    &model.gaussian_buffers.indirect_args_buffer,
                );
            }
        }

        for model in self.models.values() {
            self.postprocessor.postprocess(
                encoder,
                &model.bind_groups.postprocessor.0,
                &model.bind_groups.postprocessor.1,
                model.gaussian_buffers.gaussians_buffer.len() as u32,
                &model.gaussian_buffers.postprocess_indirect_args_buffer,
            );
        }

        Ok(())
    }

    /// Download the query results from the GPU.
    pub async fn download_query_results(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        key: &K,
    ) -> Result<Vec<QueryResultPod>, Error> {
        query::download(
            device,
            queue,
            &self
                .models
                .get(key)
                .expect("model not found")
                .gaussian_buffers
                .query_result_count_buffer,
            &self
                .models
                .get(key)
                .expect("model not found")
                .gaussian_buffers
                .query_results_buffer,
        )
        .await
    }
}
