//! A [`DepthSorter`] backed by [Lampshade](https://crates.io/crates/lampshade).
//!
//! This module is only available on native targets when the `lampshade-sort` feature is enabled.
//! Lampshade's accelerated counted sorter is used when the device is created with the features and
//! limits advertised by [`lampshade::KeyValueSoaSorter::requirements`], and falls back to
//! [`RadixSorter`] otherwise.

use crate::{
    DepthSortIndirectArgsBuffer, DepthSorter, DepthSorterWithoutBindGroups, GaussiansDepthBuffer,
    IndirectArgsBuffer, IndirectIndicesBuffer, RadixSorter, RadixSorterBindGroups,
    ViewerCreateDepthSorterFactoryContext,
    core::{BufferWrapper, GaussianPod},
};

/// The `u32` word index of `instance_count` within [`wgpu::util::DrawIndirectArgs`].
const INSTANCE_COUNT_WORD: u32 = 1;

/// A [`DepthSorter`] that uses [Lampshade](https://crates.io/crates/lampshade)'s counted key-value
/// SoA sorter.
///
/// The sorter is prepared once with the viewer's buffers and consumes the GPU-written
/// `instance_count` of the draw indirect args buffer without CPU readback. Devices that do not
/// support Lampshade's accelerated backend transparently fall back to [`RadixSorter`].
///
/// To use it, inject it through [`ViewerCreateOptions::depth_sorter_factory`](crate::ViewerCreateOptions::depth_sorter_factory):
///
/// ```rust
/// # use pollster::FutureExt;
/// # async {
/// # use wgpu_3dgs_viewer::core::{self, glam::*};
/// # let instance = wgpu::Instance::new(
/// #     wgpu::InstanceDescriptor::new_without_display_handle_from_env()
/// # );
/// # let adapter = instance
/// #     .request_adapter(&wgpu::RequestAdapterOptions::default())
/// #     .await
/// #     .expect("adapter");
/// # let (device, _queue) = adapter
/// #     .request_device(&wgpu::DeviceDescriptor {
/// #         required_limits: adapter.limits(),
/// #         ..Default::default()
/// #     })
/// #     .await
/// #     .expect("device");
/// # let gaussians = vec![core::Gaussian {
/// #         rot: Quat::IDENTITY,
/// #         pos: Vec3::ZERO,
/// #         color: U8Vec4::ZERO,
/// #         sh: [Vec3::ZERO; 15],
/// #         scale: Vec3::ONE,
/// # }];
/// use wgpu_3dgs_viewer::{
///     LampshadeSorter, DefaultGaussianPod, Viewer, ViewerCreateOptions,
///     core::{BufferWrapper, GaussiansBuffer, IterGaussian},
/// };
///
/// fn create(
///     device: &wgpu::Device,
///     gaussians: &impl IterGaussian,
/// ) -> Viewer<DefaultGaussianPod, LampshadeSorter> {
///     Viewer::new_with_options(
///         device,
///         wgpu::TextureFormat::Rgba8Unorm,
///         gaussians,
///         ViewerCreateOptions {
///             depth_stencil: None,
///             gaussians_buffer_usage:
///                 GaussiansBuffer::<DefaultGaussianPod>::DEFAULT_USAGES,
///             color_write_mask: wgpu::ColorWrites::ALL,
///             cache: None,
///             depth_sorter_factory: |ctx| LampshadeSorter::new(ctx),
///             phantom_data: Default::default(),
///         },
///     )
///     .expect("viewer")
/// }
/// # let viewer = create(&device, &gaussians);
/// # }.block_on();
/// ```
///
/// To use it with [`MultiModelViewer`](crate::MultiModelViewer), inject
/// [`LampshadeSorter::new_without_bind_groups`] instead, which prepares one Lampshade plan per
/// model:
///
/// ```rust
/// # use pollster::FutureExt;
/// # #[cfg(feature = "multi-model")]
/// # async {
/// # let instance = wgpu::Instance::new(
/// #     wgpu::InstanceDescriptor::new_without_display_handle_from_env()
/// # );
/// # let adapter = instance
/// #     .request_adapter(&wgpu::RequestAdapterOptions::default())
/// #     .await
/// #     .expect("adapter");
/// # let (device, _queue) = adapter
/// #     .request_device(&wgpu::DeviceDescriptor {
/// #         required_limits: adapter.limits(),
/// #         ..Default::default()
/// #     })
/// #     .await
/// #     .expect("device");
/// use wgpu_3dgs_viewer::{
///     LampshadeSorter, DefaultGaussianPod, MultiModelViewer, MultiModelViewerCreateOptions,
///     core::{BufferWrapper, GaussiansBuffer},
/// };
///
/// fn create(
///     device: &wgpu::Device,
/// ) -> MultiModelViewer<DefaultGaussianPod, LampshadeSorter<()>> {
///     MultiModelViewer::new_with_options(
///         device,
///         wgpu::TextureFormat::Rgba8Unorm,
///         MultiModelViewerCreateOptions {
///             depth_stencil: None,
///             gaussians_buffer_usage:
///                 GaussiansBuffer::<DefaultGaussianPod>::DEFAULT_USAGES,
///             color_write_mask: wgpu::ColorWrites::ALL,
///             cache: None,
///             depth_sorter_factory: |ctx| LampshadeSorter::new_without_bind_groups(ctx.device),
///             phantom_data: Default::default(),
///         },
///     )
///     .expect("viewer")
/// }
/// # let viewer = create(&device);
/// # }.block_on();
/// ```
#[derive(Debug)]
pub struct LampshadeSorter<B = RadixSorterBindGroups> {
    /// The embedded sorter used as the fallback when Lampshade is unavailable.
    fallback: RadixSorter<B>,
    /// The prepared Lampshade plan, or `None` when the device is not eligible.
    ///
    /// This is only populated by [`LampshadeSorter::new`]. When used without internally managed
    /// bind groups, each model owns its plan inside [`LampshadeSorterBindGroups`].
    plan: Option<LampshadePlan>,
}

/// The bind groups for a single model when [`LampshadeSorter`] is used without internally managed
/// bind groups.
#[derive(Debug)]
pub struct LampshadeSorterBindGroups {
    /// The fallback radix sorter bind groups.
    fallback: RadixSorterBindGroups,
    /// The prepared Lampshade plan, or `None` when the device is not eligible.
    plan: Option<LampshadePlan>,
}

impl LampshadeSorterBindGroups {
    /// Whether the accelerated Lampshade sorter is active for this model.
    ///
    /// When this returns `false`, [`DepthSorterWithoutBindGroups::sort`] uses the embedded
    /// [`RadixSorter`] fallback.
    pub fn is_accelerated(&self) -> bool {
        self.plan.is_some()
    }
}

/// A prepared Lampshade counted sort.
struct LampshadePlan {
    /// The prepared sorter.
    sorter: lampshade::KeyValueSoaSorter,
    /// The keys buffer (Gaussian depths).
    keys: wgpu::Buffer,
    /// The values buffer (indirect indices).
    values: wgpu::Buffer,
    /// The metadata buffer whose `instance_count` word holds the visible count.
    count: wgpu::Buffer,
    /// The maximum number of items that may be sorted.
    capacity: u32,
}

impl std::fmt::Debug for LampshadePlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LampshadePlan")
            .field("capacity", &self.capacity)
            .finish_non_exhaustive()
    }
}

impl LampshadePlan {
    /// Prepare the Lampshade counted sort, returning `None` when it is unavailable.
    fn prepare(
        device: &wgpu::Device,
        gaussians_depth: &GaussiansDepthBuffer,
        indirect_indices: &IndirectIndicesBuffer,
        indirect_args: &IndirectArgsBuffer,
        capacity: u32,
    ) -> Option<Self> {
        let mut sorter =
            lampshade::KeyValueSoaSorter::new_native_for_adapter(device, &device.adapter_info())?;

        match sorter.prepare_counted_from_word(
            gaussians_depth.buffer(),
            indirect_indices.buffer(),
            indirect_args.buffer(),
            INSTANCE_COUNT_WORD,
            capacity,
        ) {
            Ok(()) => {
                log::debug!("Using Lampshade for depth sorting");
                Some(Self {
                    sorter,
                    keys: gaussians_depth.buffer().clone(),
                    values: indirect_indices.buffer().clone(),
                    count: indirect_args.buffer().clone(),
                    capacity,
                })
            }
            Err(error) => {
                log::warn!("Could not prepare Lampshade sorter: {error}");
                None
            }
        }
    }

    /// Record the prepared counted sort.
    fn sort(&self, encoder: &mut wgpu::CommandEncoder) -> Result<(), lampshade::Error> {
        self.sorter.record_reserved_sort_counted_from_word(
            encoder,
            &self.keys,
            &self.values,
            &self.count,
            INSTANCE_COUNT_WORD,
            self.capacity,
        )
    }
}

impl LampshadeSorter {
    /// Create a new Lampshade depth sorter from the viewer's depth sorter factory context.
    ///
    /// The sorter is prepared once here. If the device does not support Lampshade's accelerated
    /// backend, or preparation fails, the sorter falls back to [`RadixSorter`].
    pub fn new<G: GaussianPod>(ctx: ViewerCreateDepthSorterFactoryContext<G>) -> Self {
        let fallback = RadixSorter::new(
            ctx.device,
            ctx.gaussians_depth_buffer,
            ctx.indirect_indices_buffer,
        );

        let capacity = capacity_of(ctx.indirect_indices_buffer);

        let plan = LampshadePlan::prepare(
            ctx.device,
            ctx.gaussians_depth_buffer,
            ctx.indirect_indices_buffer,
            ctx.indirect_args_buffer,
            capacity,
        );

        Self { fallback, plan }
    }

    /// Whether the accelerated Lampshade sorter is active.
    ///
    /// When this returns `false`, [`DepthSorter::sort`] uses the embedded [`RadixSorter`].
    pub fn is_accelerated(&self) -> bool {
        self.plan.is_some()
    }
}

impl LampshadeSorter<()> {
    /// Create a new Lampshade depth sorter without internally managed bind groups.
    ///
    /// The returned sorter implements [`DepthSorterWithoutBindGroups`], which prepares one
    /// Lampshade plan per model in [`DepthSorterWithoutBindGroups::create_bind_groups`]. This is
    /// the variant to use with [`MultiModelViewer`](crate::MultiModelViewer).
    ///
    /// To create bind groups with layout matched to this sorter, use the
    /// [`DepthSorterWithoutBindGroups::create_bind_groups`] method.
    pub fn new_without_bind_groups(device: &wgpu::Device) -> Self {
        Self {
            fallback: RadixSorter::new_without_bind_groups(device),
            plan: None,
        }
    }
}

/// Compute the sort capacity from the indirect indices buffer length.
fn capacity_of(indirect_indices: &IndirectIndicesBuffer) -> u32 {
    (indirect_indices.buffer().size() / std::mem::size_of::<u32>() as u64) as u32
}

impl DepthSorter for LampshadeSorter {
    fn sort(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        indirect_args_buffer: &DepthSortIndirectArgsBuffer,
    ) {
        if let Some(plan) = &self.plan {
            if let Err(error) = plan.sort(encoder) {
                log::warn!("Failed to sort using Lampshade sorter: {error}, using fallback sorter");
                self.fallback.sort(encoder, indirect_args_buffer);
            }
        } else {
            self.fallback.sort(encoder, indirect_args_buffer);
        }
    }
}

impl DepthSorterWithoutBindGroups for LampshadeSorter<()> {
    type BindGroups = LampshadeSorterBindGroups;

    fn create_bind_groups(
        &self,
        device: &wgpu::Device,
        gaussians_depth: &GaussiansDepthBuffer,
        indirect_indices: &IndirectIndicesBuffer,
        indirect_args: &IndirectArgsBuffer,
    ) -> Self::BindGroups {
        let fallback = self
            .fallback
            .create_bind_groups(device, gaussians_depth, indirect_indices);

        let plan = LampshadePlan::prepare(
            device,
            gaussians_depth,
            indirect_indices,
            indirect_args,
            capacity_of(indirect_indices),
        );

        LampshadeSorterBindGroups { fallback, plan }
    }

    fn sort(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_groups: &Self::BindGroups,
        indirect_args_buffer: &DepthSortIndirectArgsBuffer,
    ) {
        if let Some(plan) = &bind_groups.plan {
            if let Err(error) = plan.sort(encoder) {
                log::warn!("Failed to sort using Lampshade sorter: {error}, using fallback sorter");
                self.fallback
                    .sort(encoder, &bind_groups.fallback, indirect_args_buffer);
            }
        } else {
            self.fallback
                .sort(encoder, &bind_groups.fallback, indirect_args_buffer);
        }
    }
}
