// Vertex

@group(0) @binding(0)
var query_texture_view: texture_2d<f32>;

@group(0) @binding(1)
var query_texture_sampler: sampler;

@group(0) @binding(2)
var<uniform> color: vec4<f32>;

@vertex
fn vert_main(@builtin(vertex_index) vert_index: u32) -> FragmentInput {
    var out: FragmentInput;

    out.uv = vec2<f32>(
        f32((vert_index << 1u) & 2u),
        f32(vert_index & 2u),
    );
    out.clip_pos = vec4<f32>(out.uv * 2.0 - 1.0, 0.0, 1.0);
    out.uv.y = 1.0 - out.uv.y;

    return out;
}

// Fragment

struct FragmentInput {
    @location(0) uv: vec2<f32>,
    
    @builtin(position) clip_pos: vec4<f32>,
}

@fragment
fn frag_main(in: FragmentInput) -> @location(0) vec4<f32> {
    let magnitude = textureSample(query_texture_view, query_texture_sampler, in.uv).r;
    return color * magnitude;
}