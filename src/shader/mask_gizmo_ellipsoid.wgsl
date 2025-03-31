// Vertex

struct Camera {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    size: vec2<f32>,
}
@group(0) @binding(0)
var<uniform> camera: Camera;

struct Gizmo {
    color: vec4<f32>,
    transform: mat4x4<f32>,
}
@group(0) @binding(1)
var<storage, read> gizmos: array<Gizmo>;

@vertex
fn vert_main(
    @builtin(vertex_index) vert_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> FragmentInput {
    var out: FragmentInput;

    let axis_index = instance_index % 3u;
    let gizmo_index = instance_index / 3u;

    let gizmo = gizmos[gizmo_index];
    
    const pi = 3.141592653;
    let angle = f32(vert_index) * 2 * pi / 32.0;
    
    let pos = array<vec3<f32>, 3>(
        vec3<f32>(0.0, cos(angle), sin(angle)), // X axis
        vec3<f32>(cos(angle), 0.0, sin(angle)), // Y axis
        vec3<f32>(cos(angle), sin(angle), 0.0), // Z axis
    )[axis_index];
    
    let world_pos = gizmo.transform * vec4<f32>(pos, 1.0);
    
    out.clip_pos = camera.proj * camera.view * world_pos;
    out.color = gizmo.color;

    return out;
}

// Fragment

struct FragmentInput {
    @location(0) color: vec4<f32>,

    @builtin(position) clip_pos: vec4<f32>,
}

@fragment
fn frag_main(in: FragmentInput) -> @location(0) vec4<f32> {
    return in.color;
}
