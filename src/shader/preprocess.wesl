import wgpu_3dgs_core::model_transform::model_transform_mat;
import super::camera::Camera;
import wgpu_3dgs_core::{
    gaussian::Gaussian,
    model_transform::ModelTransform,
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

fn pre() {
    // Reset instance count
    atomicStore(&indirect_args.instance_count, 0u);
}

fn main(index: u32) {
    if index >= arrayLength(&gaussians) {
        return;
    }

    let gaussian = gaussians[index];

    // Cull
    let world_pos = model_transform_mat(model_transform) * vec4<f32>(gaussian.pos, 1.0);
    let proj_pos = camera.proj * camera.view * world_pos;
    let ndc_pos = proj_pos.xyz / proj_pos.w;
    if !(all(ndc_pos >= vec3<f32>(-1.0, -1.0, 0.0)) && all(ndc_pos <= vec3<f32>(1.0))) {
        return;
    }

    let culled_index = atomicAdd(&indirect_args.instance_count, 1u);
    indirect_indices[culled_index] = index;

    // Depth
    gaussians_depth[culled_index] = 1.0 - ndc_pos.z;
}

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