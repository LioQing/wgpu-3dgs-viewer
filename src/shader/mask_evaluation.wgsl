@group(0) @binding(0)
var<uniform> op: u32;

const op_union: u32 = 0;
const op_intersection: u32 = 1;
const op_symmetric_difference: u32 = 2;
const op_difference: u32 = 3;
const op_complement: u32 = 4;
const op_shape: u32 = 5;
const op_reset: u32 = 6;

@group(0) @binding(1)
var<storage, read> source: array<u32>;

@group(0) @binding(2)
var<storage, read_write> dest: array<atomic<u32>>;

const workgroup_size = vec3<u32>({{workgroup_size}});
const workgroup_count = workgroup_size.x * workgroup_size.y * workgroup_size.z;

@compute @workgroup_size({{workgroup_size}})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let index = wid.x * workgroup_count +
        lid.x +
        lid.y * workgroup_size.x +
        lid.z * workgroup_size.x * workgroup_size.y;

    if index >= arrayLength(&dest) {
        return;
    }

    let mask = source[index];

    if op == op_union {
        atomicOr(&dest[index], mask);
    } else if op == op_intersection {
        atomicAnd(&dest[index], mask);
    } else if op == op_symmetric_difference {
        atomicXor(&dest[index], mask);
    } else if op == op_difference {
        atomicAnd(&dest[index], ~mask);
    } else if op == op_complement {
        atomicStore(&dest[index], ~atomicLoad(&dest[index]));
    } else if op == op_reset {
        atomicStore(&dest[index], ~0u);
    }
}
