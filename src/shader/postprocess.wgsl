struct DispatchIndirectArgs {
    x: u32,
    y: u32,
    z: u32,
}
@group(0) @binding(0)
var<storage, read_write> indirect_args: DispatchIndirectArgs;

struct Query {
    content_u32: vec4<u32>,
    content_f32: vec4<f32>,
}
@group(0) @binding(1)
var<uniform> query: Query;

const query_type_none = 0u << 24u;
const query_type_hit = 1u << 24u;
const query_type_rect = 2u << 24u;
const query_type_brush = 3u << 24u;

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

@group(0) @binding(2)
var<storage, read> query_result_count: u32;

struct QueryResult {
    content_u32: vec4<u32>,
    content_f32: vec4<f32>,
}
@group(0) @binding(3)
var<storage, read> query_results: array<QueryResult>;

@group(0) @binding(4)
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

@compute @workgroup_size({{workgroup_size}})
fn pre_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;

    if index == 0u {
        // Set the dispatch indirect args
        const workgroup_size = {{workgroup_size}}u;
        indirect_args.x = (query_result_count + workgroup_size - 1u) / workgroup_size;
        indirect_args.y = 1u;
        indirect_args.z = 1u;
    }

    // Reset selection if the query selection op is set
    if index < arrayLength(&selection) && query_selection_op() == query_selection_op_set {
        atomicStore(&selection[index], 0u);
    }
}

@compute @workgroup_size({{workgroup_size}})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;

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