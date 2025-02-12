// Vertex

const outline_width = 1.0;

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

fn query_type() -> u32 {
    return query.content_u32.x & 0xFF000000;
}

struct Camera {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    size: vec2<f32>,
}
@group(0) @binding(1)
var<uniform> camera: Camera;

fn camera_coords(ndc_pos: vec2<f32>) -> vec2<f32> {
    return (ndc_pos * vec2<f32>(1.0, -1.0) + vec2<f32>(1.0)) * camera.size * 0.5;
}

@vertex
fn vert_main(
    @builtin(vertex_index) vert_index: u32,
) -> FragmentInput {
    var out: FragmentInput;

    let query_type = query_type();
    if query_type == query_type_rect {
        let top_left = query.content_f32.xy;
        let bottom_right = query.content_f32.zw;
        
        var pos = vec2<f32>(0.0);
        
        switch vert_index {
            case 0u: { pos = top_left; }
            case 1u: { pos = vec2(bottom_right.x, top_left.y); }
            case 2u: { pos = bottom_right; }
            case 3u: { pos = bottom_right; }
            case 4u: { pos = vec2(top_left.x, bottom_right.y); }
            case 5u: { pos = top_left; }
            default: { pos = vec2(0.0); }
        }

        let clip_pos = vec2<f32>(
            pos.x / camera.size.x * 2.0 - 1.0,
            -(pos.y / camera.size.y * 2.0 - 1.0),
        );

        out.coords = pos;
        out.clip_pos = vec4<f32>(clip_pos, 0.0, 1.0);
    } else {
        let radius = f32(query.content_u32.y);
        let center = query.content_f32.zw;

        var pos = vec2<f32>(0.0);

        switch vert_index {
            case 0u: { pos = center + vec2<f32>(-radius, -radius); }
            case 1u: { pos = center + vec2<f32>(radius, -radius); }
            case 2u: { pos = center + vec2<f32>(radius, radius); }
            case 3u: { pos = center + vec2<f32>(radius, radius); }
            case 4u: { pos = center + vec2<f32>(-radius, radius); }
            case 5u: { pos = center + vec2<f32>(-radius, -radius); }
            default: { pos = vec2(0.0); }
        }

        let clip_pos = vec2<f32>(
            pos.x / camera.size.x * 2.0 - 1.0,
            -(pos.y / camera.size.y * 2.0 - 1.0),
        );

        out.coords = pos;
        out.clip_pos = vec4<f32>(clip_pos, 0.0, 1.0);
    }

    return out;
}

// Fragment

struct FragmentInput {
    @location(0) coords: vec2<f32>,

    @builtin(position) clip_pos: vec4<f32>,
}

@fragment
fn frag_main(in: FragmentInput) -> @location(0) vec4<f32> {
    let query_type = query_type();
    if query_type == query_type_rect {
        let top_left = query.content_f32.xy;
        let bottom_right = query.content_f32.zw;
        
        let dist_from_left = abs(in.coords.x - top_left.x);
        let dist_from_right = abs(in.coords.x - bottom_right.x);
        let dist_from_top = abs(in.coords.y - top_left.y);
        let dist_from_bottom = abs(in.coords.y - bottom_right.y);
        
        if (dist_from_left < outline_width || 
            dist_from_right < outline_width ||
            dist_from_top < outline_width || 
            dist_from_bottom < outline_width) {
            return vec4(1.0);
        }

        discard;
    } else if query_type == query_type_brush {
        let radius = f32(query.content_u32.y);
        let center = query.content_f32.zw;
        
        let diff = in.coords - center;
        let dist_sqr = dot(diff, diff);
        
        if dist_sqr > radius * radius {
            discard;
        }

        if dist_sqr > (radius - outline_width) * (radius - outline_width) {
            return vec4(1.0);
        }

        discard;
    } else {
        let radius = f32(query.content_u32.y);
        let center = query.content_f32.zw;

        if (
            center.x - outline_width / 2.0 <= in.coords.x &&
            in.coords.x < center.x + outline_width / 2.0
        ) || (
            center.y - outline_width / 2.0 <= in.coords.y &&
            in.coords.y < center.y + outline_width / 2.0
        ) {
            return vec4(1.0);
        }
        
        let diff = in.coords - center;
        let dist_sqr = dot(diff, diff);
        
        if dist_sqr > radius * radius {
            discard;
        }

        if dist_sqr > (radius - outline_width) * (radius - outline_width) {
            return vec4(1.0);
        }

        discard;
    }
}