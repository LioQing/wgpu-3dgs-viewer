// Vertex

const max_radius = 2.0;

struct Camera {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    size: vec2<f32>,
}
@group(0) @binding(0)
var<uniform> camera: Camera;

struct Transform {
    pos: vec3<f32>,
    quat: vec4<f32>,
    scale: vec3<f32>,
}
@group(0) @binding(1)
var<uniform> transform: Transform;

fn transform_mat() -> mat4x4<f32> {
    let pos = transform.pos;
    let quat = transform.quat;
    let scale = transform.scale;

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

fn rotation_mat() -> mat3x3<f32> {
    let quat = transform.quat;

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

    return mat3x3<f32>(
        vec3<f32>(
            1.0 - (yy + zz),
            xy + wz,
            xz - wy,
        ),
        vec3<f32>(
            xy - wz,
            1.0 - (xx + zz),
            yz + wx,
        ),
        vec3<f32>(
            xz + wy,
            yz - wx,
            1.0 - (xx + yy),
        ),
    );
}

struct Gaussian {
    pos: vec3<f32>,
    color: u32,
    cov3d: array<f32, 6>,
}
@group(0) @binding(2)
var<storage, read> gaussians: array<Gaussian>;

@group(0) @binding(3)
var<storage, read> indirect_indices: array<u32>;

fn compute_cov2d(gaussian: Gaussian) -> vec3<f32> {
    let cov3d = gaussian.cov3d;
    let r = rotation_mat();

    let vrk = mat3x3<f32>(
        cov3d[0], cov3d[1], cov3d[2],
        cov3d[1], cov3d[3], cov3d[4],
        cov3d[2], cov3d[4], cov3d[5],
    );

    let focal = vec2<f32>(camera.proj[0][0], camera.proj[1][1]) * camera.size;

    let t = camera.view * transform_mat() * vec4<f32>(gaussian.pos, 1.0);
    let j = transpose(mat3x3<f32>(
        focal.x / t.z, 0.0, -(focal.x * t.x) / (t.z * t.z),
        0.0, focal.y / t.z, -(focal.y * t.y) / (t.z * t.z),
        0.0, 0.0, 0.0,
    ));
    let w = mat3x3<f32>(camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz);

    let cov2d = (j * w * r) * vrk * transpose(j * w * r);

    return vec3<f32>(cov2d[0][0], cov2d[0][1], cov2d[1][1]);
}

fn compute_quad_offset(vert_index: u32) -> vec2<f32> {
    return array<vec2<f32>, 6>(
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0),
    )[vert_index];
}

@vertex
fn vert_main(
    @builtin(vertex_index) vert_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> FragmentInput {
    var out: FragmentInput;

    let gaussian_index = indirect_indices[instance_index];
    let gaussian = gaussians[gaussian_index];
    
    let cov2d = compute_cov2d(gaussian);
    let mid = 0.5 * (cov2d.x + cov2d.z);
    let radius = length(vec2<f32>(0.5 * (cov2d.x - cov2d.z), cov2d.y));
    let lambda_1 = mid + radius;
    let lambda_2 = mid - radius;

    if lambda_2 < 0.0 {
        out.clip_pos = vec4<f32>(0.0, 0.0, 2.0, 1.0);
        return out;
    }

    let diag_vec = normalize(vec2<f32>(cov2d.y, lambda_1 - cov2d.x));
    let diag_vec_ortho = vec2<f32>(diag_vec.y, -diag_vec.x);
    let major_axis = min(max_radius * sqrt(lambda_1), 1024.0) * diag_vec;
    let minor_axis = min(max_radius * sqrt(lambda_2), 1024.0) * diag_vec_ortho;
    let quad_offset = compute_quad_offset(vert_index) * max_radius;
    let pos_proj = camera.proj * camera.view * transform_mat() * vec4<f32>(gaussian.pos, 1.0);
    let pos_ndc = pos_proj.xyz / pos_proj.w;
    let clip_proj = (
        pos_ndc.xy
        + quad_offset.x * major_axis / camera.size
        + quad_offset.y * minor_axis / camera.size
    );

    out.clip_pos = vec4<f32>(clip_proj, pos_ndc.z, 1.0);
    out.quad_offset = quad_offset;
    out.color = unpack4x8unorm(gaussian.color);

    return out;
}

// Fragment

struct FragmentInput {
    @location(0) quad_offset: vec2<f32>,
    @location(1) color: vec4<f32>,

    @builtin(position) clip_pos: vec4<f32>,
}

@fragment
fn frag_main(in: FragmentInput) -> @location(0) vec4<f32> {
    let radius_sqr = dot(in.quad_offset, in.quad_offset);
    if radius_sqr > max_radius * max_radius {
        discard;
    }
    
    let alpha = in.color.a * exp(-radius_sqr);
    return vec4<f32>(in.color.rgb, 1.0) * alpha;
}