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

@vertex
fn vert_main(
    @builtin(vertex_index) vert_index: u32,
) -> @builtin(position) vec4<f32> {
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

    let clip_pos = camera_ndc(pos);

    return vec4<f32>(clip_pos, 0.0, 1.0);
}

// Fragment

@fragment
fn frag_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0);
}