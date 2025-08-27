import wgpu_3dgs_viewer::{
    camera::{ Camera, camera_texture_to_ndc },
};

// Vertex

@group(0) @binding(0)
var<uniform> camera: Camera;

@group(0) @binding(1)
var<uniform> start: vec2<f32>;

@group(0) @binding(2)
var<uniform> end: vec2<f32>;

@group(0) @binding(3)
var<uniform> radius: f32;

@vertex
fn vert_main(
    @builtin(vertex_index) vert_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> FragmentInput {
    var out: FragmentInput;

    out.index = instance_index;

    if instance_index < 2u {
        let pos = select(
            start,
            end,
            instance_index == 1u,
        );

        var uv = vec2<f32>(0.0);

        switch vert_index {
            case 0u: { uv = vec2<f32>(1.0, -1.0); }
            case 1u: { uv = vec2<f32>(-1.0, -1.0); }
            case 2u: { uv = vec2<f32>(1.0, 1.0); }
            case 3u: { uv = vec2<f32>(-1.0, 1.0); }
            case 4u: { uv = vec2<f32>(1.0, 1.0); }
            case 5u: { uv = vec2<f32>(-1.0, -1.0); }
            default: { uv = vec2<f32>(0.0); }
        }

        let offset = uv * radius;
        let ndc_pos = camera_texture_to_ndc(pos + offset, camera.size);

        out.uv = uv;
        out.ndc_pos = vec4<f32>(ndc_pos, 0.0, 1.0);
    } else {
        let dir = end - start;
        let normal = normalize(vec2<f32>(dir.y, -dir.x));

        let pos = select(
            start,
            end,
            vert_index % 2u == 1u,
        );

        let offset = select(
            normal,
            -normal,
            vert_index < 2u || vert_index == 5u,
        ) * radius;

        let ndc_pos = camera_texture_to_ndc(pos + offset, camera.size);

        out.uv = vec2<f32>(0.0);
        out.ndc_pos = vec4<f32>(ndc_pos, 0.0, 1.0);
    }

    return out;
}

// Fragment

struct FragmentInput {
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) index: u32,

    @builtin(position) ndc_pos: vec4<f32>,
}

@fragment
fn frag_main(in: FragmentInput) -> @location(0) vec4<f32> {
    if in.index < 2u && dot(in.uv, in.uv) > 1.0 {
        discard;
    }

    return vec4<f32>(1.0);
}