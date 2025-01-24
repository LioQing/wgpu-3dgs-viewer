// Vertex

const max_radius = 3.0;

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

@group(0) @binding(2)
var<storage, read> indirect_indices: array<u32>;

fn compute_cov2d(gaussian: Gaussian) -> vec3<f32> {
    let cov3d = gaussian.cov3d;

    let vrk = mat3x3<f32>(
        cov3d[0], cov3d[1], cov3d[2],
        cov3d[1], cov3d[3], cov3d[4],
        cov3d[2], cov3d[4], cov3d[5],
    );

    let t = camera.view * vec4<f32>(gaussian.pos, 1.0);
    let focal = vec2<f32>(
        camera.proj[0][0] * camera.size.x / 2.0,
        camera.proj[1][1] * camera.size.y / 2.0,
    );
    let j = mat3x3<f32>(
        focal.x / t.z, 0.0, -(focal.x * t.x) / (t.z * t.z),
        0.0, focal.y / t.z, -(focal.y * t.y) / (t.z * t.z),
        0.0, 0.0, 0.0,
    );
    let w = transpose(mat3x3<f32>(camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz));

    let cov = transpose(w * j) * vrk * (w * j)
        + mat3x3(
            0.3, 0.0, 0.0,
            0.0, 0.3, 0.0,
            0.0, 0.0, 0.0,
        );

    return vec3<f32>(cov[0][0], cov[0][1], cov[1][1]);
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
    let det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    let mid = 0.5 * (cov2d.x + cov2d.z);
    let lambda_1 = mid + sqrt(max(1e-6, mid * mid - det));
    let lambda_2 = mid - sqrt(max(1e-6, mid * mid - det));

    let diag_vec = normalize(
        vec2<f32>(1, (-cov2d.x + cov2d.y + lambda_1) / (cov2d.y - cov2d.z + lambda_1))
    );
    let diag_vec_ortho = vec2<f32>(diag_vec.y, -diag_vec.x);
    let major_axis = min(max_radius * sqrt(lambda_1), 1024.0) * diag_vec;
    let minor_axis = min(max_radius * sqrt(lambda_2), 1024.0) * diag_vec_ortho;
    let quad_offset = compute_quad_offset(vert_index) * max_radius;
    let pos_proj = camera.proj * camera.view * vec4<f32>(gaussian.pos, 1.0);
    let clip_proj = (
        pos_proj.xy / pos_proj.w
        + quad_offset.x * major_axis / camera.size
        + quad_offset.y * minor_axis / camera.size
    );

    out.clip_pos = vec4<f32>(clip_proj, pos_proj.z / pos_proj.w, 1.0);
    out.quad_offset = quad_offset;
    out.color = gaussian.color;

    return out;
}

// Fragment

struct FragmentInput {
    @location(0) quad_offset: vec2<f32>,
    @location(1) color: u32,

    @builtin(position) clip_pos: vec4<f32>,
}

@fragment
fn frag_main(in: FragmentInput) -> @location(0) vec4<f32> {
    let radius_sqr = dot(in.quad_offset, in.quad_offset);
    if radius_sqr > max_radius * max_radius {
        discard;
    }
    
    let color = unpack4x8unorm(in.color);
    let alpha = color.a * exp(-radius_sqr);
    return vec4<f32>(color.rgb, 1.0) * alpha;
}