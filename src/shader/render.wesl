import package::{
    camera::{ Camera, camera_aspect_ratio },
    utils::{
        cov2d_axes,
        view_color,
    },
};
import wgpu_3dgs_core::{
    gaussian::Gaussian,
    gaussian_transform::{
        GaussianTransform,
        gaussian_display_mode_splat,
        gaussian_display_mode_ellipse,
        gaussian_display_mode_point,
        gaussian_transform_display_mode,
        gaussian_transform_sh_deg,
        gaussian_transform_no_sh0,
    },
    model_transform::{
        ModelTransform,
        model_transform_mat,
        model_transform_inv_sr_mat,
    },
};

// Vertex

const max_radius = 2.0;
const point_size = 0.01;

@group(0) @binding(0)
var<uniform> camera: Camera;

@group(0) @binding(1)
var<uniform> model_transform: ModelTransform;

@group(0) @binding(2)
var<uniform> gaussian_transform: GaussianTransform;

@group(0) @binding(3)
var<storage, read> gaussians: array<Gaussian>;

@group(0) @binding(4)
var<storage, read> indirect_indices: array<u32>;

fn quad_offset(vert_index: u32) -> vec2<f32> {
    switch vert_index {
        case 0u { return vec2<f32>(1.0, -1.0); }
        case 1u { return vec2<f32>(-1.0, -1.0); }
        case 2u { return vec2<f32>(1.0, 1.0); }
        case 3u { return vec2<f32>(-1.0, 1.0); }
        case 4u { return vec2<f32>(1.0, 1.0); }
        case 5u { return vec2<f32>(-1.0, -1.0); }
        default { return vec2<f32>(0.0, 0.0); }
    }
}

fn color(gaussian: Gaussian, world_pos: vec3<f32>) -> vec4<f32> {
    let world_camera_pos = -(transpose(mat3x3<f32>(
        camera.view[0].xyz,
        camera.view[1].xyz,
        camera.view[2].xyz
    )) * camera.view[3].xyz);
    let world_view_dir = world_camera_pos - world_pos;
    let model_view_dir = model_transform_inv_sr_mat(model_transform) * world_view_dir;

    return view_color(
        gaussian,
        -normalize(model_view_dir),
        gaussian_transform_sh_deg(gaussian_transform.flags),
        gaussian_transform_no_sh0(gaussian_transform.flags),
    );
}

@vertex
fn vert_main(
    @builtin(vertex_index) vert_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> FragmentInput {
    // Ensure model_transform is used to prevent dead code elimination
    let _model_transform = model_transform;

    var out: FragmentInput;

    let gaussian_index = indirect_indices[instance_index];
    let gaussian = gaussians[gaussian_index];

    let world_pos = model_transform_mat(model_transform) * vec4<f32>(gaussian.pos, 1.0);
    let view_pos = camera.view * world_pos;
    let proj_pos = camera.proj * view_pos;

    let color = color(gaussian, world_pos.xyz);
    let display_mode = gaussian_transform_display_mode(gaussian_transform.flags);

    if display_mode == gaussian_display_mode_point {
        let quad_offset = quad_offset(vert_index) * point_size * gaussian_transform.size;
        let aspect_ratio = camera_aspect_ratio(camera.size);
        let clip_pos = proj_pos.xy
            + quad_offset * proj_pos.w * vec2<f32>(aspect_ratio, 1.0) / length(view_pos.xyz);

        out.clip_pos = vec4<f32>(clip_pos, proj_pos.zw);
        out.quad_offset = quad_offset;
        out.color = color;
        out.display_mode = display_mode;
        
        return out;
    }
    
    let axes = cov2d_axes(gaussian, model_transform);
    if all(axes == vec4<f32>(0.0, 0.0, 0.0, 0.0)) {
        out.clip_pos = vec4<f32>(0.0, 0.0, 2.0, 1.0);
        return out;
    }

    let major_axis = axes.xy;
    let minor_axis = axes.zw;

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

    return out;
}

// Fragment

struct FragmentInput {
    @location(0) quad_offset: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) @interpolate(flat) display_mode: u32,

    @builtin(position) clip_pos: vec4<f32>,
}

fn splat(in: FragmentInput) -> vec4<f32> {
    let radius_sq = dot(in.quad_offset, in.quad_offset);
    if radius_sq > max_radius * max_radius {
        discard;
    }

    let alpha = in.color.a * exp(-radius_sq);
    return vec4<f32>(in.color.rgb, alpha);
}

fn ellipse(in: FragmentInput) -> vec4<f32> {
    let radius_sq = dot(in.quad_offset, in.quad_offset);
    if radius_sq > max_radius * max_radius {
        discard;
    }

    let is_outline = radius_sq > (max_radius - 0.1) * (max_radius - 0.1);
    let alpha = in.color.a + (1.0 - in.color.a) * f32(is_outline);
    return vec4<f32>(in.color.rgb, alpha);
}

fn point(in: FragmentInput) -> vec4<f32> {
    return vec4<f32>(in.color.rgb, 1.0);
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

    return color;
}
