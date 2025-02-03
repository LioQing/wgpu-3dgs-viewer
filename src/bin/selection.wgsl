// Vertex

const marker_size = 20.0;

@group(0) @binding(0)
var<uniform> selected_pos: vec4<f32>;

struct Camera {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    size: vec2<f32>,
}
@group(0) @binding(1)
var<uniform> camera: Camera;

@vertex
fn vert_main(
    @builtin(vertex_index) vert_index: u32,
) -> FragmentInput {
    var out: FragmentInput;

    if selected_pos.w != 1.0 {
        out.clip_pos = vec4<f32>(0.0, 0.0, 2.0, 1.0);
        return out;
    }

    let offset = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 0.6667),
        vec2<f32>(0.7071, -0.3333),
        vec2<f32>(-0.7071, -0.3333),
    )[vert_index];

    out.clip_pos = camera.proj * camera.view * selected_pos;
    out.clip_pos += vec4<f32>(offset / camera.size * marker_size, 0.0, 0.0);

    return out;
}

// Fragment

struct FragmentInput {
    @builtin(position) clip_pos: vec4<f32>,
}

@fragment
fn frag_main(in: FragmentInput) -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}