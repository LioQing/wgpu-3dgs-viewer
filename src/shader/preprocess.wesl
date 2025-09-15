import package::{
    camera::{ Camera, world_to_camera },
    utils::cull,
};
import wgpu_3dgs_core::{
    gaussian::Gaussian,
    model_transform::{ ModelTransform, model_to_world, model_transform_mat },
};

@group(0) @binding(0)
var<uniform> camera: Camera;

@group(0) @binding(1)
var<uniform> model_transform: ModelTransform;

@group(0) @binding(2)
var<storage, read> gaussians: array<Gaussian>;

struct IndirectArgs {
    vertex_count: u32,
    instance_count: atomic<u32>,
    first_vertex: u32,
    first_instance: u32,
}
@group(0) @binding(3)
var<storage, read_write> indirect_args: IndirectArgs;

struct RadixSortDispatchIndirectArgs {
    x: u32,
    y: u32,
    z: u32,
}
@group(0) @binding(4)
var<storage, read_write> radix_sort_indirect_args: RadixSortDispatchIndirectArgs;

@group(0) @binding(5)
var<storage, read_write> indirect_indices: array<u32>;

@group(0) @binding(6)
var<storage, read_write> gaussians_depth: array<f32>;

@if(selection_buffer) @group(0) @binding(7)
var<storage, read> selection: array<u32>;

@if(selection_buffer) @group(0) @binding(8)
var<uniform> invert_selection: u32;

@compute @workgroup_size(1)
fn pre() {
    // Reset instance count
    atomicStore(&indirect_args.instance_count, 0u);
}

override workgroup_size: u32;

@compute @workgroup_size(workgroup_size)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;

    if index >= arrayLength(&gaussians) {
        return;
    }

    // Selection
    @if(selection_buffer) {
        let word_index = index / 32u;
        let bit_index = index % 32u;
        let bit_mask = 1u << bit_index;
        let bit = (selection[word_index] & bit_mask) != 0u;
        let inverted = invert_selection != 0u;
        if inverted == bit {
            return;
        }
    }

    let gaussian = gaussians[index];

    let world_pos = model_to_world(model_transform, gaussian.pos);
    let proj_pos = world_to_camera(camera, world_pos);
    let ndc_pos = proj_pos.xyz / proj_pos.w;

    // Cull
    if cull(ndc_pos, camera) {
        return;
    }

    let culled_index = atomicAdd(&indirect_args.instance_count, 1u);
    indirect_indices[culled_index] = index;

    // Depth
    gaussians_depth[culled_index] = 1.0 - ndc_pos.z;
}

@compute @workgroup_size(1)
fn post() {
    let instance_count = atomicLoad(&indirect_args.instance_count);

    // Set radix sort indirect args
    const histo_block_kvs = 3840u; // wgpu_sort::HISTO_BLOCK_KVS
    radix_sort_indirect_args.x = (instance_count + histo_block_kvs - 1) / histo_block_kvs;
    radix_sort_indirect_args.y = 1u;
    radix_sort_indirect_args.z = 1u;

    // Set the padded depths
    let padded_count = min(
        radix_sort_indirect_args.x * histo_block_kvs,
        arrayLength(&gaussians_depth),
    );
    for (var i = instance_count; i < padded_count; i += 1u) {
        gaussians_depth[i] = 2.0;
    }
}