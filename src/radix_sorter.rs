use crate::{GaussiansDepthBuffer, IndirectIndicesBuffer, RadixSortIndirectArgsBuffer};

pub type RadixSorterBindGroups = wgpu_sort::InternalSortBuffers;

/// Radix sorter for sorting Gaussians based on their depth (i.e. clipped z value).
#[derive(Debug)]
pub struct RadixSorter<B = RadixSorterBindGroups> {
    /// The sorter.
    ///
    /// Using modified version of the [wgpu_sort](https://crates.io/crates/wgpu_sort) crate.
    sorter: wgpu_sort::GPUSorter,
    /// The internal sort buffers.
    internal_sort_buffers: B,
}

impl<B> RadixSorter<B> {
    /// Create the bind groups.
    pub fn create_bind_groups(
        &self,
        device: &wgpu::Device,
        gaussians_depth: &GaussiansDepthBuffer,
        indirect_indices: &IndirectIndicesBuffer,
    ) -> RadixSorterBindGroups {
        self.sorter.create_internal_sort_buffers(
            device,
            gaussians_depth.buffer(),
            indirect_indices.buffer(),
        )
    }
}

impl RadixSorter {
    /// Create a new radix sorter.
    pub fn new(
        device: &wgpu::Device,
        gaussians_depth: &GaussiansDepthBuffer,
        indirect_indices: &IndirectIndicesBuffer,
    ) -> Self {
        let this = RadixSorter::new_without_bind_groups(device);

        log::debug!("Creating radix sorter internal sort buffers");
        let internal_sort_buffers =
            this.create_bind_groups(device, gaussians_depth, indirect_indices);

        log::info!("Radix sorter created");

        Self {
            sorter: this.sorter,
            internal_sort_buffers,
        }
    }

    /// Sort the Gaussians based on their depth.
    pub fn sort(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        indirect_args_buffer: &RadixSortIndirectArgsBuffer,
    ) {
        self.sorter.sort_indirect(
            encoder,
            &self.internal_sort_buffers,
            indirect_args_buffer.buffer(),
        );
    }
}

impl RadixSorter<()> {
    /// Create a new radix sorter without internally managed bind groups.
    pub fn new_without_bind_groups(device: &wgpu::Device) -> Self {
        log::debug!("Creating radix sorter without bind groups");
        let sorter = wgpu_sort::GPUSorter::new(device, 1);

        log::info!("Radix sorter created");

        Self {
            sorter,
            internal_sort_buffers: (),
        }
    }

    /// Sort the Gaussians based on their depth.
    ///
    /// To create the bind groups with layout matched to this sorter, use the
    /// [`RadixSorter::create_bind_groups`] method.
    pub fn sort(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_groups: &RadixSorterBindGroups,
        indirect_args_buffer: &RadixSortIndirectArgsBuffer,
    ) {
        self.sorter
            .sort_indirect(encoder, bind_groups, indirect_args_buffer.buffer());
    }
}

#[allow(dead_code)]
pub(crate) mod wgpu_sort {
    /*
        From: https://crates.io/crates/wgpu_sort

        BSD 2-Clause License

        Copyright (c) 2024, Simon Niedermayr, Josef Stumpfegger
        All rights reserved.

        Redistribution and use in source and binary forms, with or without
        modification, are permitted provided that the following conditions are met:

        1. Redistributions of source code must retain the above copyright notice, this
        list of conditions and the following disclaimer.

        2. Redistributions in binary form must reproduce the above copyright notice,
        this list of conditions and the following disclaimer in the documentation
        and/or other materials provided with the distribution.

        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
        AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
        IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
        DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
        FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
        DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
        SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
        CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
        OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
        OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    */

    /*
        This file implements a gpu version of radix sort. A good introduction to general purpose radix sort can
        be found here: http://www.codercorner.com/RadixSortRevisited.htm

        The gpu radix sort implemented here is a re-implementation of the Vulkan radix sort found in the fuchsia repos: https://fuchsia.googlesource.com/fuchsia/+/refs/heads/main/src/graphics/lib/compute/radix_sort/
        Currently only the sorting for 32-bit key-value pairs is implemented

        All shaders can be found in radix_sort.wgsl
    */

    use std::{
        mem,
        num::{NonZeroU32, NonZeroU64},
    };

    use bytemuck::bytes_of;
    use wgpu::{ComputePassDescriptor, util::DeviceExt};

    // IMPORTANT: the following constants have to be synced with the numbers in radix_sort.wgsl

    /// workgroup size of histogram shader
    const HISTOGRAM_WG_SIZE: u32 = 256;

    /// one thread operates on 2 prefixes at the same time
    const PREFIX_WG_SIZE: u32 = 1 << 7;

    /// scatter compute shader work group size
    const SCATTER_WG_SIZE: u32 = 1 << 8;

    /// we sort 8 bits per pass
    const RS_RADIX_LOG2: u32 = 8;

    /// 256 entries into the radix table
    const RS_RADIX_SIZE: u32 = 1 << RS_RADIX_LOG2;

    /// number of bytes our keys and values have
    const RS_KEYVAL_SIZE: u32 = 32 / RS_RADIX_LOG2;

    /// TODO describe me
    const RS_HISTOGRAM_BLOCK_ROWS: u32 = 15;

    /// DO NOT CHANGE, shader assume this!!!
    const RS_SCATTER_BLOCK_ROWS: u32 = RS_HISTOGRAM_BLOCK_ROWS;

    /// number of elements scattered by one work group
    const SCATTER_BLOCK_KVS: u32 = HISTOGRAM_WG_SIZE * RS_SCATTER_BLOCK_ROWS;

    /// number of elements scattered by one work group
    pub const HISTO_BLOCK_KVS: u32 = HISTOGRAM_WG_SIZE * RS_HISTOGRAM_BLOCK_ROWS;

    /// bytes per value
    /// currently only 4 byte values are allowed
    const BYTES_PER_PAYLOAD_ELEM: u32 = 4;

    /// number of passed used for sorting
    /// we sort 8 bits per pass so 4 passes are required for a 32 bit value
    const NUM_PASSES: u32 = BYTES_PER_PAYLOAD_ELEM;

    /// Sorting pipeline. It can be used to sort key-value pairs stored in [SortBuffers]
    #[derive(Debug)]
    pub struct GPUSorter {
        zero_p: wgpu::ComputePipeline,
        histogram_p: wgpu::ComputePipeline,
        prefix_p: wgpu::ComputePipeline,
        scatter_even_p: wgpu::ComputePipeline,
        scatter_odd_p: wgpu::ComputePipeline,
    }

    impl GPUSorter {
        pub fn new(device: &wgpu::Device, subgroup_size: u32) -> Self {
            // special variables for scatter shade
            let histogram_sg_size = subgroup_size;
            let rs_sweep_0_size = RS_RADIX_SIZE / histogram_sg_size;
            let rs_sweep_1_size = rs_sweep_0_size / histogram_sg_size;
            let rs_sweep_2_size = rs_sweep_1_size / histogram_sg_size;
            let rs_sweep_size = rs_sweep_0_size + rs_sweep_1_size + rs_sweep_2_size;
            let _rs_smem_phase_1 = RS_RADIX_SIZE + RS_RADIX_SIZE + rs_sweep_size;
            let rs_smem_phase_2 = RS_RADIX_SIZE + RS_SCATTER_BLOCK_ROWS * SCATTER_WG_SIZE;
            // rs_smem_phase_2 will always be larger, so always use phase2
            let rs_mem_dwords = rs_smem_phase_2;
            let rs_mem_sweep_0_offset = 0;
            let rs_mem_sweep_1_offset = rs_mem_sweep_0_offset + rs_sweep_0_size;
            let rs_mem_sweep_2_offset = rs_mem_sweep_1_offset + rs_sweep_1_size;

            let bind_group_layout = Self::bind_group_layout(device);

            let pipeline_layout: wgpu::PipelineLayout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("radix sort pipeline layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

            let raw_shader: &str = include_str!("shader/radix_sort.wgsl");

            // TODO replace with this with pipeline-overridable constants once they are available
            let shader_w_const = format!(
                "const histogram_sg_size: u32 = {:}u;\n\
                const histogram_wg_size: u32 = {:}u;\n\
                const rs_radix_log2: u32 = {:}u;\n\
                const rs_radix_size: u32 = {:}u;\n\
                const rs_keyval_size: u32 = {:}u;\n\
                const rs_histogram_block_rows: u32 = {:}u;\n\
                const rs_scatter_block_rows: u32 = {:}u;\n\
                const rs_mem_dwords: u32 = {:}u;\n\
                const rs_mem_sweep_0_offset: u32 = {:}u;\n\
                const rs_mem_sweep_1_offset: u32 = {:}u;\n\
                const rs_mem_sweep_2_offset: u32 = {:}u;\n{:}",
                histogram_sg_size,
                HISTOGRAM_WG_SIZE,
                RS_RADIX_LOG2,
                RS_RADIX_SIZE,
                RS_KEYVAL_SIZE,
                RS_HISTOGRAM_BLOCK_ROWS,
                RS_SCATTER_BLOCK_ROWS,
                rs_mem_dwords,
                rs_mem_sweep_0_offset,
                rs_mem_sweep_1_offset,
                rs_mem_sweep_2_offset,
                raw_shader
            );
            let shader_code = shader_w_const
                .replace(
                    "{histogram_wg_size}",
                    HISTOGRAM_WG_SIZE.to_string().as_str(),
                )
                .replace("{prefix_wg_size}", PREFIX_WG_SIZE.to_string().as_str())
                .replace("{scatter_wg_size}", SCATTER_WG_SIZE.to_string().as_str());

            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Radix sort shader"),
                source: wgpu::ShaderSource::Wgsl(shader_code.into()),
            });
            let zero_p = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Zero the histograms"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("zero_histograms"),
                compilation_options: Default::default(),
                cache: None,
            });
            let histogram_p = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("calculate_histogram"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("calculate_histogram"),
                compilation_options: Default::default(),
                cache: None,
            });
            let prefix_p = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("prefix_histogram"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("prefix_histogram"),
                compilation_options: Default::default(),
                cache: None,
            });
            let scatter_even_p = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("scatter_even"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("scatter_even"),
                compilation_options: Default::default(),
                cache: None,
            });
            let scatter_odd_p = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("scatter_odd"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("scatter_odd"),
                compilation_options: Default::default(),
                cache: None,
            });

            Self {
                zero_p,
                histogram_p,
                prefix_p,
                scatter_even_p,
                scatter_odd_p,
            }
        }

        fn bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("radix sort bind group layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: Some(
                                NonZeroU64::new(mem::size_of::<SorterState>() as u64).unwrap(),
                            ),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
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
            })
        }

        fn create_keyval_buffers(
            device: &wgpu::Device,
            length: u32,
        ) -> (wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, wgpu::Buffer) {
            // add padding so that our buffer size is a multiple of keys_per_workgroup
            let count_ru_histo = keys_buffer_size(length) * RS_KEYVAL_SIZE;

            // creating the two needed buffers for sorting
            let keys = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("radix sort keys buffer"),
                size: (count_ru_histo * BYTES_PER_PAYLOAD_ELEM) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            // auxiliary buffer for keys
            let keys_aux = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("radix sort keys auxiliary buffer"),
                size: (count_ru_histo * BYTES_PER_PAYLOAD_ELEM) as u64,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });

            let payload_size = length * BYTES_PER_PAYLOAD_ELEM; // make sure that we have at least 1 byte of data;
            let payload = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("radix sort payload buffer"),
                size: payload_size as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            // auxiliary buffer for payload/values
            let payload_aux = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("radix sort payload auxiliary buffer"),
                size: payload_size as u64,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            (keys, keys_aux, payload, payload_aux)
        }

        /// Modified from [`GPUSorter::create_keyval_buffers`].
        fn create_internal_keyval_buffers(
            device: &wgpu::Device,
            length: u32,
        ) -> (wgpu::Buffer, wgpu::Buffer) {
            // add padding so that our buffer size is a multiple of keys_per_workgroup
            let count_ru_histo = keys_buffer_size(length) * RS_KEYVAL_SIZE;

            // auxiliary buffer for keys
            let keys_aux = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("radix sort keys auxiliary buffer"),
                size: (count_ru_histo * BYTES_PER_PAYLOAD_ELEM) as u64,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });

            let payload_size = length * BYTES_PER_PAYLOAD_ELEM; // make sure that we have at least 1 byte of data;

            // auxiliary buffer for payload/values
            let payload_aux = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("radix sort payload auxiliary buffer"),
                size: payload_size as u64,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            (keys_aux, payload_aux)
        }

        // calculates and allocates a buffer that is sufficient for holding all needed information for
        // sorting. This includes the histograms and the temporary scatter buffer
        // @return: tuple containing [internal memory buffer (should be bound at shader binding 1, count_ru_histo (padded size needed for the keyval buffer)]
        fn create_internal_mem_buffer(&self, device: &wgpu::Device, length: u32) -> wgpu::Buffer {
            // currently only a few different key bits are supported, maybe has to be extended

            // The "internal" memory map looks like this:
            //   +---------------------------------+ <-- 0
            //   | histograms[keyval_size]         |
            //   +---------------------------------+ <-- keyval_size                           * histo_size
            //   | partitions[scatter_blocks_ru-1] |
            //   +---------------------------------+ <-- (keyval_size + scatter_blocks_ru - 1) * histo_size
            //   | workgroup_ids[keyval_size]      |
            //   +---------------------------------+ <-- (keyval_size + scatter_blocks_ru - 1) * histo_size + workgroup_ids_size

            let scatter_blocks_ru = scatter_blocks_ru(length);

            let histo_size = RS_RADIX_SIZE * std::mem::size_of::<u32>() as u32;

            let internal_size = (RS_KEYVAL_SIZE + scatter_blocks_ru) * histo_size; // +1 safety

            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Internal radix sort buffer"),
                size: internal_size as u64,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        }

        fn general_info_data(length: u32) -> SorterState {
            SorterState {
                num_keys: length,
                padded_size: keys_buffer_size(length),
                even_pass: 0,
                odd_pass: 0,
            }
        }

        fn record_calculate_histogram(
            &self,
            bind_group: &wgpu::BindGroup,
            length: u32,
            encoder: &mut wgpu::CommandEncoder,
        ) {
            // as we only deal with 32 bit float values always 4 passes are conducted
            let hist_blocks_ru = histo_blocks_ru(length);

            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("zeroing the histogram"),
                    timestamp_writes: None,
                });

                pass.set_pipeline(&self.zero_p);
                pass.set_bind_group(0, bind_group, &[]);
                pass.dispatch_workgroups(hist_blocks_ru, 1, 1);
            }

            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("calculate histogram"),
                    timestamp_writes: None,
                });

                pass.set_pipeline(&self.histogram_p);
                pass.set_bind_group(0, bind_group, &[]);
                pass.dispatch_workgroups(hist_blocks_ru, 1, 1);
            }
        }

        fn record_calculate_histogram_indirect(
            &self,
            bind_group: &wgpu::BindGroup,
            dispatch_buffer: &wgpu::Buffer,
            encoder: &mut wgpu::CommandEncoder,
        ) {
            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("zeroing the histogram"),
                    timestamp_writes: None,
                });

                pass.set_pipeline(&self.zero_p);
                pass.set_bind_group(0, bind_group, &[]);
                pass.dispatch_workgroups_indirect(dispatch_buffer, 0);
            }

            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("calculate histogram"),
                    timestamp_writes: None,
                });

                pass.set_pipeline(&self.histogram_p);
                pass.set_bind_group(0, bind_group, &[]);
                pass.dispatch_workgroups_indirect(dispatch_buffer, 0);
            }
        }

        // There does not exist an indirect histogram dispatch as the number of prefixes is determined by the amount of passes
        fn record_prefix_histogram(
            &self,
            bind_group: &wgpu::BindGroup,
            encoder: &mut wgpu::CommandEncoder,
        ) {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("prefix histogram"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.prefix_p);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(NUM_PASSES, 1, 1);
        }

        fn record_scatter_keys(
            &self,
            bind_group: &wgpu::BindGroup,
            length: u32,
            encoder: &mut wgpu::CommandEncoder,
        ) {
            let scatter_blocks_ru = scatter_blocks_ru(length);

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Scatter keyvals"),
                timestamp_writes: None,
            });

            pass.set_bind_group(0, bind_group, &[]);
            pass.set_pipeline(&self.scatter_even_p);
            pass.dispatch_workgroups(scatter_blocks_ru, 1, 1);

            pass.set_pipeline(&self.scatter_odd_p);
            pass.dispatch_workgroups(scatter_blocks_ru, 1, 1);

            pass.set_pipeline(&self.scatter_even_p);
            pass.dispatch_workgroups(scatter_blocks_ru, 1, 1);

            pass.set_pipeline(&self.scatter_odd_p);
            pass.dispatch_workgroups(scatter_blocks_ru, 1, 1);
        }

        fn record_scatter_keys_indirect(
            &self,
            bind_group: &wgpu::BindGroup,
            dispatch_buffer: &wgpu::Buffer,
            encoder: &mut wgpu::CommandEncoder,
        ) {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("radix sort scatter keyvals"),
                timestamp_writes: None,
            });

            pass.set_bind_group(0, bind_group, &[]);
            pass.set_pipeline(&self.scatter_even_p);
            pass.dispatch_workgroups_indirect(dispatch_buffer, 0);

            pass.set_pipeline(&self.scatter_odd_p);
            pass.dispatch_workgroups_indirect(dispatch_buffer, 0);

            pass.set_pipeline(&self.scatter_even_p);
            pass.dispatch_workgroups_indirect(dispatch_buffer, 0);

            pass.set_pipeline(&self.scatter_odd_p);
            pass.dispatch_workgroups_indirect(dispatch_buffer, 0);
        }

        /// Writes sort commands to command encoder.
        /// If sort_first_n is not none one the first n elements are sorted
        /// otherwise everything is sorted.
        ///
        /// **IMPORTANT**: if less than the whole buffer is sorted the rest of the keys buffer will be be corrupted
        pub fn sort(
            &self,
            encoder: &mut wgpu::CommandEncoder,
            queue: &wgpu::Queue,
            sort_buffers: &impl SortBuffersTrait,
            sort_first_n: Option<u32>,
        ) {
            let bind_group = &sort_buffers.bind_group();
            let num_elements = sort_first_n.unwrap_or(sort_buffers.len());

            // write number of elements to buffer
            queue.write_buffer(sort_buffers.state_buffer(), 0, bytes_of(&num_elements));

            self.record_calculate_histogram(bind_group, num_elements, encoder);
            self.record_prefix_histogram(bind_group, encoder);
            self.record_scatter_keys(bind_group, num_elements, encoder);
        }

        /// Initiates sorting with an indirect call.
        /// The dispatch buffer must contain the struct [wgpu::util::DispatchIndirectArgs].
        ///
        /// number of y and z workgroups must be 1
        ///
        /// x = (N + [HISTO_BLOCK_KVS]- 1 )/[HISTO_BLOCK_KVS],
        /// where N are the first N elements to be sorted
        ///
        /// [SortBuffers::state_buffer] contains the number of keys that will be sorted.
        /// This is set to sort the whole buffer by default.
        ///
        /// **IMPORTANT**: if less than the whole buffer is sorted the rest of the keys buffer will most likely be corrupted.
        pub fn sort_indirect(
            &self,
            encoder: &mut wgpu::CommandEncoder,
            sort_buffers: &impl SortBuffersTrait,
            dispatch_buffer: &wgpu::Buffer,
        ) {
            let bind_group = &sort_buffers.bind_group();

            self.record_calculate_histogram_indirect(bind_group, dispatch_buffer, encoder);
            self.record_prefix_histogram(bind_group, encoder);
            self.record_scatter_keys_indirect(bind_group, dispatch_buffer, encoder);
        }

        /// creates all buffers necessary for sorting
        pub fn create_sort_buffers(
            &self,
            device: &wgpu::Device,
            length: NonZeroU32,
        ) -> SortBuffers {
            let length = length.get();

            let (keys_a, keys_b, payload_a, payload_b) =
                GPUSorter::create_keyval_buffers(device, length);
            let internal_mem_buffer = self.create_internal_mem_buffer(device, length);

            let uniform_infos = Self::general_info_data(length);
            let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("radix sort uniform buffer"),
                contents: bytemuck::bytes_of(&uniform_infos),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("radix sort bind group"),
                layout: &Self::bind_group_layout(device),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: internal_mem_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: keys_a.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: keys_b.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: payload_a.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: payload_b.as_entire_binding(),
                    },
                ],
            });
            // return (uniform_buffer, bind_group);
            SortBuffers {
                keys_a,
                keys_b,
                payload_a,
                payload_b,
                internal_mem_buffer,
                state_buffer: uniform_buffer,
                bind_group,
                length,
            }
        }

        /// creates all buffers necessary for sorting
        ///
        /// Modified from [`GPUSorter::create_sort_buffers`].
        pub fn create_internal_sort_buffers(
            &self,
            device: &wgpu::Device,
            keys: &wgpu::Buffer,
            payload: &wgpu::Buffer,
        ) -> InternalSortBuffers {
            let length = payload.size() as u32 / BYTES_PER_PAYLOAD_ELEM;

            let (keys_b, payload_b) = GPUSorter::create_internal_keyval_buffers(device, length);
            let internal_mem_buffer = self.create_internal_mem_buffer(device, length);

            let uniform_infos = Self::general_info_data(length);
            let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("radix sort uniform buffer"),
                contents: bytemuck::bytes_of(&uniform_infos),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("radix sort bind group"),
                layout: &Self::bind_group_layout(device),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: internal_mem_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: keys.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: keys_b.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: payload.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: payload_b.as_entire_binding(),
                    },
                ],
            });
            // return (uniform_buffer, bind_group);
            InternalSortBuffers {
                keys_b,
                payload_b,
                internal_mem_buffer,
                state_buffer: uniform_buffer,
                bind_group,
                length,
            }
        }
    }

    /// Struct containing information about the state of the sorter.
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
    pub struct SorterState {
        /// number of first n keys that will be sorted
        pub num_keys: u32,
        padded_size: u32,
        even_pass: u32,
        odd_pass: u32,
    }

    /// The trait for [`SortBuffers`] and [`InternalSortBuffers`].
    pub trait SortBuffersTrait {
        /// number of key-value pairs that can be stored in this buffer
        fn len(&self) -> u32;

        /// The keys buffer has padding bytes.
        /// This function returns the number of bytes without padding
        fn keys_valid_size(&self) -> u64 {
            (self.len() * RS_KEYVAL_SIZE) as u64
        }

        /// The bind group used for sorting
        fn bind_group(&self) -> &wgpu::BindGroup;

        /// Buffer containing a [SorterState]
        fn state_buffer(&self) -> &wgpu::Buffer;
    }

    /// Struct containing all buffers necessary for sorting.
    /// The key and value buffers can be read and written.
    #[derive(Debug)]
    pub struct SortBuffers {
        /// keys that are sorted
        keys_a: wgpu::Buffer,
        /// intermediate key buffer for sorting
        #[allow(dead_code)]
        keys_b: wgpu::Buffer,
        /// value/payload buffer that is sorted
        payload_a: wgpu::Buffer,
        /// intermediate value buffer for sorting
        #[allow(dead_code)]
        payload_b: wgpu::Buffer,

        /// buffer used to store intermediate results like histograms and scatter partitions
        #[allow(dead_code)]
        internal_mem_buffer: wgpu::Buffer,

        /// state buffer used for sorting
        state_buffer: wgpu::Buffer,

        /// bind group used for sorting
        bind_group: wgpu::BindGroup,

        // number of key-value pairs
        length: u32,
    }

    impl SortBuffers {
        /// Buffer storing the keys values.
        ///
        /// **WARNING**: this buffer has padding bytes at the end
        ///        use [SortBuffers::keys_valid_size] to get the valid size.
        pub fn keys(&self) -> &wgpu::Buffer {
            &self.keys_a
        }

        /// Buffer containing the values
        pub fn values(&self) -> &wgpu::Buffer {
            &self.payload_a
        }
    }

    impl SortBuffersTrait for SortBuffers {
        fn len(&self) -> u32 {
            self.length
        }

        fn bind_group(&self) -> &wgpu::BindGroup {
            &self.bind_group
        }

        fn state_buffer(&self) -> &wgpu::Buffer {
            &self.state_buffer
        }
    }

    /// Struct containing all buffers necessary for sorting.
    /// The key and value buffers can be read and written.
    ///
    /// This is modified from [`SortBuffers`] to only contains the internal buffers used.
    #[derive(Debug)]
    pub struct InternalSortBuffers {
        /// intermediate key buffer for sorting
        #[allow(dead_code)]
        keys_b: wgpu::Buffer,
        /// intermediate value buffer for sorting
        #[allow(dead_code)]
        payload_b: wgpu::Buffer,

        /// buffer used to store intermediate results like histograms and scatter partitions
        #[allow(dead_code)]
        internal_mem_buffer: wgpu::Buffer,

        /// state buffer used for sorting
        state_buffer: wgpu::Buffer,

        /// bind group used for sorting
        bind_group: wgpu::BindGroup,

        // number of key-value pairs
        length: u32,
    }

    impl SortBuffersTrait for InternalSortBuffers {
        fn len(&self) -> u32 {
            self.length
        }

        fn bind_group(&self) -> &wgpu::BindGroup {
            &self.bind_group
        }

        fn state_buffer(&self) -> &wgpu::Buffer {
            &self.state_buffer
        }
    }

    fn scatter_blocks_ru(n: u32) -> u32 {
        n.div_ceil(SCATTER_BLOCK_KVS)
    }

    /// number of histogram blocks required
    fn histo_blocks_ru(n: u32) -> u32 {
        (scatter_blocks_ru(n) * SCATTER_BLOCK_KVS).div_ceil(HISTO_BLOCK_KVS)
    }

    /// keys buffer must be multiple of HISTO_BLOCK_KVS
    fn keys_buffer_size(n: u32) -> u32 {
        histo_blocks_ru(n) * HISTO_BLOCK_KVS
    }

    /// entire keys buffer size
    pub fn keys_buffer_size_bytes(n: u32) -> u64 {
        keys_buffer_size(n) as u64 * RS_KEYVAL_SIZE as u64 * BYTES_PER_PAYLOAD_ELEM as u64
    }
}
