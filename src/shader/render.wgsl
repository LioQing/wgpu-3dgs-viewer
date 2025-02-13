// Vertex

const max_radius = 2.0;
const point_size = 0.01;

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

fn camera_aspect_ratio() -> f32 {
    return camera.size.y / camera.size.x;
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

fn model_transform_inv_sr_mat() -> mat3x3<f32> {
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

    return mat3x3<f32>(
        vec3<f32>(
            (1.0 - (yy + zz)) / sx,
            (xy - wz) / sy,
            (xz + wy) / sz,
        ),
        vec3<f32>(
            (xy + wz) / sx,
            (1.0 - (xx + zz)) / sy,
            (yz - wx) / sz,
        ),
        vec3<f32>(
            (xz - wy) / sx,
            (yz + wx) / sy,
            (1.0 - (xx + yy)) / sz,
        ),
    );
}

fn model_scale_rotation_mat() -> mat3x3<f32> {
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

    return mat3x3<f32>(
        vec3<f32>(
            (1.0 - (yy + zz)) * sx,
            (xy + wz) * sx,
            (xz - wy) * sx,
        ),
        vec3<f32>(
            (xy - wz) * sy,
            (1.0 - (xx + zz)) * sy,
            (yz + wx) * sy,
        ),
        vec3<f32>(
            (xz + wy) * sz,
            (yz - wx) * sz,
            (1.0 - (xx + yy)) * sz,
        ),
    );
}

struct GaussianTransform {
    size: f32,
    flags: u32,
}
@group(0) @binding(2)
var<uniform> gaussian_transform: GaussianTransform;

const gaussian_display_mode_splat = 0u;
const gaussian_display_mode_ellipse = 1u;
const gaussian_display_mode_point = 2u;

fn gaussian_transform_display_mode() -> u32 {
    return unpack4xU8(gaussian_transform.flags).x;
}

fn gaussian_transform_sh_deg() -> u32 {
    return unpack4xU8(gaussian_transform.flags).y;
}

fn gaussian_transform_no_sh0() -> bool {
    return unpack4xU8(gaussian_transform.flags).z != 0u;
}

struct Gaussian {
    pos: vec3<f32>,
    color: u32,
    {{gaussian_sh_field}}
    {{gaussian_cov3d_field}}
}
@group(0) @binding(3)
var<storage, read> gaussians: array<Gaussian>;

fn gaussian_sh(gaussian_index: u32, sh_index: u32) -> vec3<f32> {
    return gaussian_unpack_sh(gaussian_index, sh_index);
}

fn gaussian_cov2d(gaussian_index: u32) -> vec3<f32> {
    let gaussian = gaussians[gaussian_index];
    let cov3d = gaussian_unpack_cov3d(gaussian_index);
    let sr = model_scale_rotation_mat();

    let vrk = mat3x3<f32>(
        cov3d[0], cov3d[1], cov3d[2],
        cov3d[1], cov3d[3], cov3d[4],
        cov3d[2], cov3d[4], cov3d[5],
    );

    let focal = vec2<f32>(camera.proj[0][0], camera.proj[1][1]) * camera.size;

    let t = camera.view * model_transform_mat() * vec4<f32>(gaussian.pos, 1.0);
    let j = transpose(mat3x3<f32>(
        focal.x / t.z, 0.0, -(focal.x * t.x) / (t.z * t.z),
        0.0, focal.y / t.z, -(focal.y * t.y) / (t.z * t.z),
        0.0, 0.0, 0.0,
    ));
    let w = mat3x3<f32>(camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz);

    let cov2d = (j * w * sr) * vrk * transpose(j * w * sr);

    let low_pass = vec3<f32>(0.1, 0.0, 0.1);

    return vec3<f32>(cov2d[0][0], cov2d[0][1], cov2d[1][1]) + low_pass;
}

fn gaussian_color(gaussian_index: u32, dir: vec3<f32>, sh_deg: u32, no_sh0: bool) -> vec4<f32> {
    const sh_c1 = 0.4886025;
    const sh_c2 = array<f32, 5>(1.0925484, -1.0925484, 0.3153916, -1.0925484, 0.5462742);
    const sh_c3 = array<f32, 7>(
        -0.5900436, 2.8906114, -0.4570458, 0.3731763, -0.4570458, 1.4453057, -0.5900436
    );

    let i = gaussian_index;
    let x = dir.x;
    let y = dir.y;
    let z = dir.z;

    let color = unpack4x8unorm(gaussians[i].color);
    var result = color.rgb; // 0.5 + SH_C0 * sh[0] already precomputed

    if no_sh0 {
        result = vec3<f32>(0.5);
    }

    if sh_deg >= 1u {
        result += sh_c1 * (
            -gaussian_sh(i, 1u) * y +
            gaussian_sh(i, 2u) * z -
            gaussian_sh(i, 3u) * x
        );

        if sh_deg >= 2u {
            let xx = x * x;
            let yy = y * y;
            let zz = z * z;
            let xy = x * y;
            let yz = y * z;
            let xz = x * z;

            result += 
                sh_c2[0] * xy * gaussian_sh(i, 4u) +
                sh_c2[1] * yz * gaussian_sh(i, 5u) +
                sh_c2[2] * (2.0 * zz - xx - yy) * gaussian_sh(i, 6u) +
                sh_c2[3] * xz * gaussian_sh(i, 7u) +
                sh_c2[4] * (xx - yy) * gaussian_sh(i, 8u);

            if sh_deg >= 3u {
                result += 
                    sh_c3[0] * y * (3.0 * xx - yy) * gaussian_sh(i, 9u) +
                    sh_c3[1] * xy * z * gaussian_sh(i, 10u) +
                    sh_c3[2] * y * (4.0 * zz - xx - yy) * gaussian_sh(i, 11u) +
                    sh_c3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * gaussian_sh(i, 12u) +
                    sh_c3[4] * x * (4.0 * zz - xx - yy) * gaussian_sh(i, 13u) +
                    sh_c3[5] * z * (xx - yy) * gaussian_sh(i, 14u) +
                    sh_c3[6] * x * (xx - 3.0 * yy) * gaussian_sh(i, 15u);
            }
        }
    }

    return vec4<f32>(max(result, vec3<f32>(0.0)), color.a);
}

@group(0) @binding(4)
var<storage, read> indirect_indices: array<u32>;

struct Query {
    content_u32: vec4<u32>,
    content_f32: vec4<f32>,
}
@group(0) @binding(5)
var<uniform> query: Query;

const query_type_none = 0u << 24u;
const query_type_hit = 1u << 24u;
const query_type_rect = 2u << 24u;

fn query_type() -> u32 {
    return query.content_u32.x & 0xFF000000;
}

@group(0) @binding(6)
var<storage, read_write> query_result_count: atomic<u32>;

struct QueryResult {
    content_u32: vec4<u32>,
    content_f32: vec4<f32>,
}
@group(0) @binding(7)
var<storage, read_write> query_results: array<QueryResult>;

struct SelectionHighlight {
    color: vec4<f32>,
}
@group(0) @binding(8)
var<uniform> selection_highlight: SelectionHighlight;

@group(0) @binding(9)
var<storage, read> selection: array<u32>;

fn selection_at(index: u32) -> bool {
    let word_index = index / 32u;
    let bit_index = index % 32u;
    let mask = 1u << bit_index;
    return (selection[word_index] & mask) != 0u;
}

fn quad_offset(vert_index: u32) -> vec2<f32> {
    return array<vec2<f32>, 6>(
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0),
    )[vert_index];
}

fn color(gaussian_index: u32, world_pos: vec3<f32>) -> vec4<f32> {
    let selected = selection_at(gaussian_index);

    if selected && selection_highlight.color.a == 1.0 {
        let color = unpack4x8unorm(gaussians[gaussian_index].color);
        return vec4<f32>(selection_highlight.color.rgb, color.a);
    }

    let world_camera_pos = -(transpose(mat3x3<f32>(
        camera.view[0].xyz,
        camera.view[1].xyz,
        camera.view[2].xyz
    )) * camera.view[3].xyz);
    let world_view_dir = world_camera_pos - world_pos;
    let model_view_dir = model_transform_inv_sr_mat() * world_view_dir;

    let color = gaussian_color(
        gaussian_index,
        -normalize(model_view_dir),
        gaussian_transform_sh_deg(),
        gaussian_transform_no_sh0(),
    );

    if selected && selection_highlight.color.a > 0.0 {
        return vec4<f32>(
            mix(color.rgb, selection_highlight.color.rgb, selection_highlight.color.a),
            color.a,
        );
    }

    return color;
}

@vertex
fn vert_main(
    @builtin(vertex_index) vert_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> FragmentInput {
    var out: FragmentInput;

    let gaussian_index = indirect_indices[instance_index];
    let gaussian = gaussians[gaussian_index];

    let world_pos = model_transform_mat() * vec4<f32>(gaussian.pos, 1.0);
    let view_pos = camera.view * world_pos;
    let proj_pos = camera.proj * view_pos;

    let color = color(gaussian_index, world_pos.xyz);
    let display_mode = gaussian_transform_display_mode();

    if display_mode == gaussian_display_mode_point {
        let quad_offset = quad_offset(vert_index) * point_size * gaussian_transform.size;
        let aspect_ratio = camera_aspect_ratio();
        let clip_pos = proj_pos.xy
            + quad_offset * proj_pos.w * vec2<f32>(aspect_ratio, 1.0) / length(view_pos.xyz);

        out.clip_pos = vec4<f32>(clip_pos, proj_pos.zw);
        out.quad_offset = quad_offset;
        out.color = color;
        out.display_mode = display_mode;
        out.index = gaussian_index;
        out.coords = camera_coords(clip_pos / proj_pos.w);
        out.depth = proj_pos.z / proj_pos.w;
        
        return out;
    }
    
    let cov2d = gaussian_cov2d(gaussian_index);
    let mid = 0.5 * (cov2d.x + cov2d.z);
    let radius = length(vec2<f32>(0.5 * (cov2d.x - cov2d.z), cov2d.y));
    let lambda_1 = mid + radius;
    let lambda_2 = mid - radius;

    if lambda_2 < 0.0 {
        out.clip_pos = vec4<f32>(0.0, 0.0, 2.0, 1.0);
        return out;
    }

    let diag_dir = normalize(vec2<f32>(cov2d.y, lambda_1 - cov2d.x));
    let ortho_diag_dir = vec2<f32>(diag_dir.y, -diag_dir.x);
    let major_len = min(max_radius * sqrt(lambda_1), 1024.0);
    let minor_len = min(max_radius * sqrt(lambda_2), 1024.0);
    let major_axis = major_len * diag_dir * gaussian_transform.size;
    let minor_axis = minor_len * ortho_diag_dir * gaussian_transform.size;

    let quad_offset = quad_offset(vert_index) * max_radius;
    let clip_pos = (
        proj_pos.xy
        + quad_offset.x * proj_pos.w * major_axis / camera.size
        + quad_offset.y * proj_pos.w * minor_axis / camera.size
    );

    out.clip_pos = vec4<f32>(clip_pos, proj_pos.zw);
    out.quad_offset = quad_offset;
    out.color = color;
    out.display_mode = display_mode;
    out.index = gaussian_index;
    out.coords = camera_coords(clip_pos / proj_pos.w);
    out.depth = proj_pos.z / proj_pos.w;

    return out;
}

// Fragment

struct FragmentInput {
    @location(0) quad_offset: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) @interpolate(flat) display_mode: u32,
    @location(3) @interpolate(flat) index: u32,
    @location(4) coords: vec2<f32>,
    @location(5) @interpolate(flat) depth: f32,

    @builtin(position) clip_pos: vec4<f32>,
}

fn splat(in: FragmentInput) -> vec4<f32> {
    let radius_sqr = dot(in.quad_offset, in.quad_offset);
    if radius_sqr > max_radius * max_radius {
        discard;
    }

    let alpha = in.color.a * exp(-radius_sqr);
    return vec4<f32>(in.color.rgb, alpha);
}

fn ellipse(in: FragmentInput) -> vec4<f32> {
    let radius_sqr = dot(in.quad_offset, in.quad_offset);
    if radius_sqr > max_radius * max_radius {
        discard;
    }

    let is_outline = radius_sqr > (max_radius - 0.1) * (max_radius - 0.1);
    let alpha = in.color.a + (1.0 - in.color.a) * f32(is_outline);
    return vec4<f32>(in.color.rgb, alpha);
}

fn point(in: FragmentInput) -> vec4<f32> {
    return vec4<f32>(in.color.rgb, 1.0);
}

fn query_hit(in: FragmentInput, color: vec4<f32>) {
    let coords = query.content_f32.xy;
    let diff = coords - in.coords;

    if dot(diff, diff) >= 1.0 {
        return;
    }

    let index = atomicAdd(&query_result_count, 1u);
    query_results[index] = QueryResult(
        vec4<u32>(in.index, vec3<u32>(0u)),
        vec4<f32>(in.depth, color.a, 0.0, 0.0),
    );
}

@fragment
fn frag_main(in: FragmentInput) -> @location(0) vec4<f32> {
    var color: vec4<f32>;

    if in.display_mode == gaussian_display_mode_splat {
        color = splat(in);
    } else if in.display_mode == gaussian_display_mode_ellipse {
        color = ellipse(in);
    } else if in.display_mode == gaussian_display_mode_point {
        color = point(in);
    }

    if query_type() == query_type_hit {
        query_hit(in, color);
    }

    return color;
}

{{gaussian_sh_unpack}}
{{gaussian_cov3d_unpack}}