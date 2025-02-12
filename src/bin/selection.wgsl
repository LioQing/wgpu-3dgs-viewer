// Vertex

const outline_width = 1.0;

@group(0) @binding(0)
var<uniform> selection: vec4<f32>;

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

    let top_left = selection.xy;
    let bottom_right = selection.zw;
    
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
        -(pos.y / camera.size.y * 2.0 - 1.0)
    );

    out.coords = pos;
    out.clip_pos = vec4(clip_pos, 0.0, 1.0);

    return out;
}

// Fragment

struct FragmentInput {
    @location(0) coords: vec2<f32>,

    @builtin(position) clip_pos: vec4<f32>,
}

@fragment
fn frag_main(in: FragmentInput) -> @location(0) vec4<f32> {
    let top_left = selection.xy;
    let bottom_right = selection.zw;
    
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
}