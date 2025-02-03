struct Camera {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    size: vec2<f32>,
}
@group(0) @binding(0)
var<uniform> camera: Camera;

struct ModelTransform {
    pos: vec3<f32>,
    quat: vec4<f32>,
    scale: vec3<f32>,
}
@group(0) @binding(1)
var<uniform> model_transform: ModelTransform;

fn model_transform_mat() -> mat4x4<f32> {
    let pos = model_transform.pos.xyz;
    let quat = model_transform.quat;
    let scale = model_transform.scale.xyz;

    let x2 = quat.x + quat.x;
    let y2 = quat.y + quat.y;
    let z2 = quat.z + quat.z;
    let xx = quat.x * x2;
    let xy = quat.x * y2;
    let xz = quat.x * z2;
    let yy = quat.y * y2;
    let yz = quat.y * z2;
    let zz = quat.z * z2;
    let wx = quat.w * x2;
    let wy = quat.w * y2;
    let wz = quat.w * z2;

    let sx = scale.x;
    let sy = scale.y;
    let sz = scale.z;

    return mat4x4<f32>(
        vec4<f32>(
            (1.0 - (yy + zz)) * sx,
            (xy + wz) * sx,
            (xz - wy) * sx,
            0.0,
        ),
        vec4<f32>(
            (xy - wz) * sy,
            (1.0 - (xx + zz)) * sy,
            (yz + wx) * sy,
            0.0,
        ),
        vec4<f32>(
            (xz + wy) * sz,
            (yz - wx) * sz,
            (1.0 - (xx + yy)) * sz,
            0.0,
        ),
        vec4<f32>(pos, 1.0),
    );
}

struct Gaussian {
    pos: vec3<f32>,
    color: u32,
    cov3d: array<f32, 6>,
}
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

struct DispatchIndirectArgs {
    x: u32,
    y: u32,
    z: u32,
}
@group(0) @binding(4)
var<storage, read_write> radix_sort_indirect_args: DispatchIndirectArgs;

@group(0) @binding(5)
var<storage, read_write> indirect_indices: array<u32>;

@group(0) @binding(6)
var<storage, read_write> gaussians_depth: array<f32>;

struct Query {
    content_u32: vec4<u32>,
    content_f32: vec4<f32>,
}
@group(0) @binding(7)
var<uniform> query: Query;

const query_type_none = 0u;
const query_type_hit = 1u;

@group(0) @binding(8)
var<storage, read_write> query_result_count: u32;

fn is_on_frustum(pos_ndc: vec3<f32>) -> bool {
    return all(pos_ndc >= vec3<f32>(-1.0, -1.0, 0.0)) && all(pos_ndc <= vec3<f32>(1.0));
}

@compute @workgroup_size(1)
fn pre_main() {
    // Reset instance count
    atomicStore(&indirect_args.instance_count, 0u);

    // Reset query result count
    if query.content_u32.x != query_type_none {
        query_result_count = 0u;
    }
}

@compute @workgroup_size({{workgroup_size}})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;

    if index >= arrayLength(&gaussians) {
        return;
    }

    let gaussian = gaussians[index];

    // Cull
    let pos_proj = camera.proj * camera.view * model_transform_mat() * vec4<f32>(gaussian.pos, 1.0);
    let pos_ndc = pos_proj.xyz / pos_proj.w;
    if !is_on_frustum(pos_ndc) {
        return;
    }

    let culled_index = atomicAdd(&indirect_args.instance_count, 1u);
    indirect_indices[culled_index] = index;

    // Depth
    gaussians_depth[culled_index] = 1.0 - pos_ndc.z;
}

@compute @workgroup_size(1)
fn post_main() {
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