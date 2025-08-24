import wgpu_3dgs_viewer::{
    camera::{ Camera, camera_texture_to_ndc },
};

// Vertex

@group(0) @binding(0)
var<uniform> camera: Camera;

@group(0) @binding(1)
var<uniform> top_left: vec2<f32>;

@group(0) @binding(2)
var<uniform> bottom_right: vec2<f32>;

@vertex
fn vert_main(
    @builtin(vertex_index) vert_index: u32,
) -> @builtin(position) vec4<f32> {
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

    let ndc_pos = camera_texture_to_ndc(pos, camera.size);

    return vec4<f32>(ndc_pos, 0.0, 1.0);
}

// Fragment

@fragment
fn frag_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0);
}