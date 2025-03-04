struct Camera {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    size: vec2<f32>,
}
@group(0) @binding(0)
var<uniform> camera: Camera;

fn camera_coords(ndc_pos: vec2<f32>) -> vec2<f32> {
    return (ndc_pos * vec2<f32>(1.0, -1.0) + vec2<f32>(1.0)) * camera.size * 0.5;
}

struct ModelTransform {
    pos: vec3<f32>,
    quat: vec4<f32>,
    scale: vec3<f32>,
}
@group(0) @binding(1)
var<uniform> model_transform: ModelTransform;

fn model_transform_mat() -> mat4x4<f32> {
    let pos = model_transform.pos;
    let quat = model_transform.quat;
    let scale = model_transform.scale;

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
    {{gaussian_sh_field}}
    {{gaussian_cov3d_field}}
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

struct Query {
    content_u32: vec4<u32>,
    content_f32: vec4<f32>,
}
@group(0) @binding(7)
var<uniform> query: Query;

const query_type_none = 0u << 24u;
const query_type_hit = 1u << 24u;
const query_type_rect = 2u << 24u;
const query_type_brush = 3u << 24u;
const query_type_texture = 4u << 24u;

fn query_type() -> u32 {
    return query.content_u32.x & 0xFF000000;
}

const query_selection_op_none = 0u << 16u;
const query_selection_op_set = 1u << 16u;
const query_selection_op_remove = 2u << 16u;
const query_selection_op_add = 3u << 16u;

fn query_selection_op() -> u32 {
    return query.content_u32.x & 0x00FF0000;
}

@group(0) @binding(8)
var<storage, read_write> query_result_count: atomic<u32>;

struct QueryResult {
    content_u32: vec4<u32>,
    content_f32: vec4<f32>,
}
@group(0) @binding(9)
var<storage, read_write> query_results: array<QueryResult>;

@compute @workgroup_size(1)
fn pre_main() {
    // Reset instance count
    atomicStore(&indirect_args.instance_count, 0u);

    // Reset query result count
    if query_type() != query_type_none {
        atomicStore(&query_result_count, 0u);
    }
}

fn is_on_frustum(ndc_pos: vec3<f32>) -> bool {
    return all(ndc_pos >= vec3<f32>(-1.0, -1.0, 0.0)) && all(ndc_pos <= vec3<f32>(1.0));
}

fn query_rect(gaussian_index: u32, ndc_pos: vec2<f32>) {
    let top_left = query.content_f32.xy;
    let bottom_right = query.content_f32.zw;
    let coords = camera_coords(ndc_pos);

    if any(coords < top_left) || any(coords > bottom_right) {
        return;
    }

    let index = atomicAdd(&query_result_count, 1u);
    query_results[index] = QueryResult(
        vec4<u32>(gaussian_index, vec3<u32>(0u)),
        vec4<f32>(0.0, 0.0, 0.0, 0.0),
    );
}

fn query_brush(gaussian_index: u32, ndc_pos: vec2<f32>) {
    let radius = f32(query.content_u32.y);
    let start = query.content_f32.xy;
    let end = query.content_f32.zw;
    let coords = camera_coords(ndc_pos);

    let start_to_end = end - start;
    let start_to_coords = coords - start;
    
    let factor = saturate(dot(start_to_coords, start_to_end) / dot(start_to_end, start_to_end));
    let start_to_proj = start_to_end * factor;

    let perp = start_to_coords - start_to_proj;

    if dot(perp, perp) > radius * radius {
        return;
    }

    let index = atomicAdd(&query_result_count, 1u);
    query_results[index] = QueryResult(
        vec4<u32>(gaussian_index, vec3<u32>(0u)),
        vec4<f32>(0.0, 0.0, 0.0, 0.0),
    );
}

struct GaussianEdit {
    flag_hsv: u32,
    contr_expo_gamma_alpha: u32,
}
@group(0) @binding(10)
var<storage, read_write> gaussians_edit: array<GaussianEdit>;

const gaussian_edit_flag_none = 0u;
const gaussian_edit_flag_enabled = 1u << 0u;
const gaussian_edit_flag_hidden = 1u << 1u;
const gaussian_edit_flag_override_color = 1u << 2u;

fn gaussians_edit_flag(index: u32) -> u32 {
    return gaussians_edit[index].flag_hsv & 0x000000FF;
}

fn gaussians_edit_enabled(index: u32) -> bool {
    return (gaussians_edit_flag(index) & gaussian_edit_flag_enabled) != 0;
}

fn gaussians_edit_flag_test(index: u32, test: u32) -> bool {
    let mask = gaussian_edit_flag_enabled | test;
    return (gaussians_edit_flag(index) & mask) == mask;
}

@group(0) @binding(11)
var<storage, read> selection: array<u32>;

fn selection_at(index: u32) -> bool {
    let word_index = index / 32u;
    let bit_index = index % 32u;
    let mask = 1u << bit_index;
    return (selection[word_index] & mask) != 0u;
}

@group(0) @binding(12)
var<uniform> selection_edit: GaussianEdit;

fn selection_edit_flag() -> u32 {
    return selection_edit.flag_hsv & 0x000000FF;
}

fn selection_edit_enabled() -> bool {
    return (selection_edit_flag() & gaussian_edit_flag_enabled) != 0;
}

// Feature query texture begin

@group(0) @binding(13)
var query_texture_view: texture_2d<f32>;

fn query_texture(gaussian_index: u32, ndc_pos: vec2<f32>) {
    let tex_size = vec2<i32>(textureDimensions(query_texture_view));
    let coords = vec2<i32>(camera_coords(ndc_pos));

    if any(coords < vec2<i32>(0)) || any(coords >= tex_size) {
        return;
    }
    
    let texel = textureLoad(query_texture_view, coords, 0);

    if texel.r == 0.0 {
        return;
    }

    let index = atomicAdd(&query_result_count, 1u);
    query_results[index] = QueryResult(
        vec4<u32>(gaussian_index, vec3<u32>(0u)),
        vec4<f32>(0.0, 0.0, 0.0, 0.0),
    );
}

// Feature query texture end

const workgroup_size = vec3<u32>({{workgroup_size}});
const workgroup_count = workgroup_size.x * workgroup_size.y * workgroup_size.z;

@compute @workgroup_size({{workgroup_size}})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let index = wid.x * workgroup_count +
        lid.x +
        lid.y * workgroup_size.x +
        lid.z * workgroup_size.x * workgroup_size.y;

    if index >= arrayLength(&gaussians) {
        return;
    }

    let gaussian = gaussians[index];

    // Edit
    if selection_at(index) && selection_edit_enabled() {
        gaussians_edit[index] = selection_edit;
    }

    // Hidden
    if gaussians_edit_flag_test(index, gaussian_edit_flag_hidden) {
        return;
    }

    // Cull
    let proj_pos = camera.proj * camera.view * model_transform_mat() * vec4<f32>(gaussian.pos, 1.0);
    let ndc_pos = proj_pos.xyz / proj_pos.w;
    if !is_on_frustum(ndc_pos) {
        return;
    }

    let culled_index = atomicAdd(&indirect_args.instance_count, 1u);
    indirect_indices[culled_index] = index;
    
    // Query
    switch query_type() {
        case query_type_rect { query_rect(index, ndc_pos.xy); }
        case query_type_brush { query_brush(index, ndc_pos.xy); }
        // Feature query texture begin
        case query_type_texture { query_texture(index, ndc_pos.xy); }
        // Feature query texture end
        default {}
    }

    // Depth
    gaussians_depth[culled_index] = 1.0 - ndc_pos.z;
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