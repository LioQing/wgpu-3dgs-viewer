// Vertex

struct Camera {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    size: vec2<f32>,
}
@group(0) @binding(0)
var<uniform> camera: Camera;

fn camera_ndc(coords: vec2<f32>) -> vec2<f32> {
    return (coords / camera.size * 2.0 - 1.0) * vec2<f32>(1.0, -1.0);
}

struct Query {
    content_u32: vec4<u32>,
    content_f32: vec4<f32>,
}
@group(0) @binding(1)
var<uniform> query: Query;

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

@vertex
fn vert_main(
    @builtin(vertex_index) vert_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> FragmentInput {
    var out: FragmentInput;

    out.index = instance_index;

    let radius = f32(query.content_u32.y);
    let start = query.content_f32.xy;
    let end = query.content_f32.zw;

    if instance_index < 2u {
        let pos = select(
            start,
            end,
            instance_index == 1u,
        );

        let uv = quad_offset(vert_index);
        let offset = uv * radius;
        let clip_pos = camera_ndc(pos + offset);

        out.uv = uv;
        out.clip_pos = vec4<f32>(clip_pos, 0.0, 1.0);
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

        let clip_pos = camera_ndc(pos + offset);

        out.uv = vec2<f32>(0.0);
        out.clip_pos = vec4<f32>(clip_pos, 0.0, 1.0);
    }

    return out;
}

// Fragment

struct FragmentInput {
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) index: u32,

    @builtin(position) clip_pos: vec4<f32>,
}

@fragment
fn frag_main(in: FragmentInput) -> @location(0) vec4<f32> {
    if in.index < 2u && dot(in.uv, in.uv) > 1.0 {
        discard;
    }

    return vec4<f32>(1.0);
}