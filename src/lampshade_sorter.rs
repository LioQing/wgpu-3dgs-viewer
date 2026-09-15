//! A [`DepthSorter`] backed by [Lampshade](https://crates.io/crates/lampshade).
//!
//! This module is only available on native targets when the `lampshade-sort` feature is enabled.
//! Lampshade's accelerated counted sorter is used when the device is created with the features and
//! limits advertised by [`lampshade::KeyValueSoaSorter::requirements`], and falls back to
//! [`RadixSorter`] otherwise.

use crate::{
    DepthSortIndirectArgsBuffer, DepthSorter, GaussiansDepthBuffer, IndirectArgsBuffer,
    IndirectIndicesBuffer, RadixSorter, ViewerCreateDepthSorterFactoryContext,
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
/// To use it, inject it through [`ViewerCreateOptions::depth_sorter_factory`]:
///
/// ```no_run
/// use wgpu_3dgs_viewer::{
///     LampshadeSorter, BufferWrapper, DefaultGaussianPod, Viewer, ViewerCreateOptions,
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
/// ```
#[derive(Debug)]
pub struct LampshadeSorter {
    /// The embedded sorter used as the fallback when Lampshade is unavailable.
    fallback: RadixSorter,
    /// The prepared Lampshade plan, or `None` when the device is not eligible.
    plan: Option<LampshadePlan>,
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

        let capacity = (ctx.indirect_indices_buffer.buffer().size()
            / std::mem::size_of::<u32>() as u64) as u32;

        let plan = Self::prepare(
            ctx.device,
            ctx.gaussians_depth_buffer,
            ctx.indirect_indices_buffer,
            ctx.indirect_args_buffer,
            capacity,
        );

        Self { fallback, plan }
    }

    /// Prepare the Lampshade counted sort, returning `None` when it is unavailable.
    fn prepare(
        device: &wgpu::Device,
        gaussians_depth: &GaussiansDepthBuffer,
        indirect_indices: &IndirectIndicesBuffer,
        indirect_args: &IndirectArgsBuffer,
        capacity: u32,
    ) -> Option<LampshadePlan> {
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
                Some(LampshadePlan {
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

    /// Whether the accelerated Lampshade sorter is active.
    ///
    /// When this returns `false`, [`DepthSorter::sort`] uses the embedded [`RadixSorter`].
    pub fn is_accelerated(&self) -> bool {
        self.plan.is_some()
    }
}

impl DepthSorter for LampshadeSorter {
    fn sort(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        indirect_args_buffer: &DepthSortIndirectArgsBuffer,
    ) {
        if let Some(plan) = &self.plan {
            plan.sorter
                .record_reserved_sort_counted_from_word(
                    encoder,
                    &plan.keys,
                    &plan.values,
                    &plan.count,
                    INSTANCE_COUNT_WORD,
                    plan.capacity,
                )
                .expect("Viewer's prepared Lampshade buffers remain valid");
        } else {
            self.fallback.sort(encoder, indirect_args_buffer);
        }
    }
}
