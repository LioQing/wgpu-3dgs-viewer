struct Query {
    content_u32: vec4<u32>,
    content_f32: vec4<f32>,
}
@group(0) @binding(0)
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

@group(0) @binding(1)
var<storage, read> query_result_count: u32;

struct QueryResult {
    content_u32: vec4<u32>,
    content_f32: vec4<f32>,
}
@group(0) @binding(2)
var<storage, read> query_results: array<QueryResult>;

@group(0) @binding(3)
var<storage, read_write> selection: array<atomic<u32>>;

fn selection_set(index: u32) {
    let word_index = index / 32u;
    let bit_index = index % 32u;
    let mask = 1u << bit_index;
    atomicOr(&selection[word_index], mask);
}

fn selection_clear(index: u32) {
    let word_index = index / 32u;
    let bit_index = index % 32u;
    let mask = 1u << bit_index;
    atomicAnd(&selection[word_index], ~mask);
}

const workgroup_size = vec3<u32>({{workgroup_size}});
const workgroup_count = workgroup_size.x * workgroup_size.y * workgroup_size.z;

// Pre only begin

struct DispatchIndirectArgs {
    x: u32,
    y: u32,
    z: u32,
}
@group(0) @binding(4)
var<storage, read_write> indirect_args: DispatchIndirectArgs;

@compute @workgroup_size({{workgroup_size}})
fn pre_main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let index = wid.x * workgroup_count +
        lid.x +
        lid.y * workgroup_size.x +
        lid.z * workgroup_size.x * workgroup_size.y;

    if index == 0u {
        // Set the dispatch indirect args
        indirect_args.x = (query_result_count + workgroup_size.x - 1u) / workgroup_size.x;
        indirect_args.y = 1u;
        indirect_args.z = 1u;
    }

    // Reset selection if the query selection op is set
    if index < arrayLength(&selection) && query_selection_op() == query_selection_op_set {
        atomicStore(&selection[index], 0u);
    }
}

// Pre only end

@compute @workgroup_size({{workgroup_size}})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let index = wid.x * workgroup_count +
        lid.x +
        lid.y * workgroup_size.x +
        lid.z * workgroup_size.x * workgroup_size.y;

    if index >= query_result_count {
        return;
    }

    let query_result = query_results[index];
    let gaussian_index = query_result.content_u32.x;

    let query_selection_op = query_selection_op();

    if query_selection_op == query_selection_op_none {
        return;
    }

    if (query_selection_op & (1u << 16u)) != 0u { // Set or Add
        selection_set(gaussian_index);
    } else { // Remove
        selection_clear(gaussian_index);
    }
}