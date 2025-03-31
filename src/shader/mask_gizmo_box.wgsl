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

    let side_index = instance_index % 4u;
    let gizmo_index = instance_index / 4u;

    let gizmo = gizmos[gizmo_index];

    let vertices = array<vec3<f32>, 8>(
        vec3<f32>(-0.5, -0.5, -0.5), // 0: bottom left front
        vec3<f32>(0.5, -0.5, -0.5),  // 1: bottom right front
        vec3<f32>(0.5, -0.5, 0.5),   // 2: bottom right back
        vec3<f32>(-0.5, -0.5, 0.5),  // 3: bottom left back
        vec3<f32>(-0.5, 0.5, -0.5),  // 4: top left front
        vec3<f32>(0.5, 0.5, -0.5),   // 5: top right front
        vec3<f32>(0.5, 0.5, 0.5),    // 6: top right back
        vec3<f32>(-0.5, 0.5, 0.5),   // 7: top left back
    );
    var side_indices = array<array<u32, 4>, 4>(
        array<u32, 4>(0, 1, 5, 4), // Front face
        array<u32, 4>(1, 2, 6, 5), // Right face
        array<u32, 4>(2, 3, 7, 6), // Back face
        array<u32, 4>(3, 0, 4, 7)  // Left face
    );

    let vertex = vertices[side_indices[side_index][vert_index]];

    let world_pos = gizmo.transform * vec4<f32>(vertex, 1.0);
    let clip_pos = camera.proj * camera.view * world_pos;

    out.clip_pos = clip_pos;
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
