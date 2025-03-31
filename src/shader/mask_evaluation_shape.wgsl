@group(0) @binding(0)
var<uniform> op: u32;

const op_union: u32 = 0;
const op_intersection: u32 = 1;
const op_symmetric_difference: u32 = 2;
const op_difference: u32 = 3;
const op_complement: u32 = 4;
const op_shape: u32 = 5;
const op_reset: u32 = 6;

struct OpShape {
    kind: u32,
    inv_transform: mat4x4<f32>,
}
@group(0) @binding(1)
var<uniform> shape: OpShape;

const shape_kind_box: u32 = 0;
const shape_kind_ellipsoid: u32 = 1;

@group(0) @binding(2)
var<storage, read_write> dest: array<atomic<u32>>;

fn dest_set(index: u32) {
    let word_index = index / 32u;
    let bit_index = index % 32u;
    let mask = 1u << bit_index;
    atomicOr(&dest[word_index], mask);
}

fn dest_clear(index: u32) {
    let word_index = index / 32u;
    let bit_index = index % 32u;
    let mask = 1u << bit_index;
    atomicAnd(&dest[word_index], ~mask);
}

struct ModelTransform {
    pos: vec3<f32>,
    quat: vec4<f32>,
    scale: vec3<f32>,
}
@group(0) @binding(3)
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
@group(0) @binding(4)
var<storage, read> gaussians: array<Gaussian>;

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

    if op != op_shape {
        return;
    }

    let gaussian = gaussians[index];
    let shape_kind = shape.kind;

    let world_pos = model_transform_mat() * vec4<f32>(gaussian.pos, 1.0);
    let proj_pos = shape.inv_transform * world_pos;

    if shape_kind == shape_kind_box {
        if all(abs(proj_pos.xyz) < vec3<f32>(0.5)) {
            dest_set(index);
        } else {
            dest_clear(index);
        }
    } else if shape_kind == shape_kind_ellipsoid {
        if dot(proj_pos.xyz, proj_pos.xyz) < 1.0 {
            dest_set(index);
        } else {
            dest_clear(index);
        }
    }
}
