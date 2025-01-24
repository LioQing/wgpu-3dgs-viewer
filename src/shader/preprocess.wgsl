struct Camera {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    size: vec2<f32>,
}
@group(0) @binding(0)
var<uniform> camera: Camera;

struct Gaussian {
    pos: vec3<f32>,
    color: u32,
    cov3d: array<f32, 6>,
}
@group(0) @binding(1)
var<storage, read> gaussians: array<Gaussian>;

struct IndirectArgs {
    vertex_count: u32,
    instance_count: atomic<u32>,
    first_vertex: u32,
    first_instance: u32,
}
@group(0) @binding(2)
var<storage, read_write> indirect_args: IndirectArgs;

struct DispatchIndirectArgs {
    x: u32,
    y: u32,
    z: u32,
}
@group(0) @binding(3)
var<storage, read_write> radix_sort_indirect_args: DispatchIndirectArgs;

@group(0) @binding(4)
var<storage, read_write> indirect_indices: array<u32>;

@group(0) @binding(5)
var<storage, read_write> gaussians_depth: array<f32>;

fn is_on_frustum(pos_proj: vec4<f32>) -> bool {
    let pos_ndc = pos_proj.xyz / pos_proj.w;
    return all(pos_ndc >= vec3<f32>(-1.0, -1.0, 0.0)) && all(pos_ndc <= vec3<f32>(1.0));
}

@compute @workgroup_size({{workgroup_size}})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;

    if index >= arrayLength(&gaussians) {
        return;
    }

    // Reset instance count
    if index == 0 {
        atomicStore(&indirect_args.instance_count, 0u);
    }

    workgroupBarrier();

    // Cull and depth
    let gaussian = gaussians[index];
    let pos_proj = camera.proj * camera.view * vec4<f32>(gaussian.pos, 1.0);

    if !is_on_frustum(pos_proj) {
        return;
    }

    let culled_index = atomicAdd(&indirect_args.instance_count, 1u);
    indirect_indices[culled_index] = index;
    gaussians_depth[culled_index] = pos_proj.z;

    workgroupBarrier();

    // Set radix sort indirect args
    if index == arrayLength(&gaussians) - 1 {
        const histo_block_kvs = 3840u; // Correspond to wgpu_sort::HISTO_BLOCK_KVS
        radix_sort_indirect_args.x =
            (indirect_args.instance_count + histo_block_kvs - 1) / histo_block_kvs;
        radix_sort_indirect_args.y = 1;
        radix_sort_indirect_args.z = 1;
    }
}