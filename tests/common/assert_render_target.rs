use glam::*;
use pollster::FutureExt;
use wgpu_3dgs_core::DownloadableBufferWrapper;

use crate::{common::TestContext, inline_wesl_pkg};

pub const ASSERT_RENDER_TARGET_PACKAGE: wesl::Pkg = inline_wesl_pkg!(
    mod assert_render_target { // Sums up the color components (ceiled to u32) in each local workgroup
        @group(0) @binding(0)
        var texture: texture_2d<f32>;

        @group(0) @binding(1)
        var<storage, read_write> dest: array<atomic<u32>, (1024u / 8u * 1024u / 8u * 4u)>;

        @compute @workgroup_size(8 * 8)
        fn main(
            @builtin(workgroup_id) wid: vec3<u32>,
            @builtin(local_invocation_id) lid: vec3<u32>,
        ) {
            let id = wid.x * 64u + lid.x;
            let tex_x = i32(id % 1024u);
            let tex_y = i32(id / 1024u);
            if tex_y >= 1024 {
                return;
            }
            let color = textureLoad(texture, vec2<i32>(tex_x, tex_y), 0);
            atomicAdd(&dest[id * 4u + 0u], select(0u, 1u, color.r > 0.0));
            atomicAdd(&dest[id * 4u + 1u], select(0u, 1u, color.g > 0.0));
            atomicAdd(&dest[id * 4u + 2u], select(0u, 1u, color.b > 0.0));
            atomicAdd(&dest[id * 4u + 3u], select(0u, 1u, color.a > 0.0));
        }
    }
);

pub const ASSERT_RENDER_TARGET_BIND_GROUP_LAOYOUT_DESCRIPTOR: wgpu::BindGroupLayoutDescriptor<
    'static,
> = wgpu::BindGroupLayoutDescriptor {
    label: Some("Assert Render Target Bind Group Layout"),
    entries: &[
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
    ],
};

pub fn assert_render_target(
    ctx: &TestContext,
    texture_view: &wgpu::TextureView,
    assertion: impl Fn(&[UVec4]),
) {
    // TOOD(https://github.com/LioQing/wgpu-3dgs-core/issues/8): configurable workgroup size.
    let shader = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Assert Render Target Shader"),
            source: wgpu::ShaderSource::Wgsl(
                wesl::compile_sourcemap(
                    &"assert_render_target"
                        .parse()
                        .expect("assert_render_target module path"),
                    &{
                        let mut resolver = wesl::PkgResolver::new();
                        resolver.add_package(&ASSERT_RENDER_TARGET_PACKAGE);
                        resolver
                    },
                    &wesl::NoMangler,
                    &wesl::CompileOptions::default(),
                )
                .expect("compiled assert render target shader")
                .to_string()
                .into(),
            ),
        });

    let dest_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Assert Render Target Destination Buffer"),
        size: 1024 / 8 * 1024 / 8 * std::mem::size_of::<UVec4>() as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let bind_group_layout = ctx
        .device
        .create_bind_group_layout(&ASSERT_RENDER_TARGET_BIND_GROUP_LAOYOUT_DESCRIPTOR);

    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Assert Render Target Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Assert Render Target Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Assert Render Target Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: dest_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Command Encoder"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Assert Render Target Compute Pass"),
            ..Default::default()
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1024 / 8 * 1024 / 8, 1, 1);
    }

    ctx.queue.submit(Some(encoder.finish()));
    ctx.device.poll(wgpu::PollType::Wait).expect("device poll");

    let dest = dest_buffer
        .download(&ctx.device, &ctx.queue)
        .block_on()
        .expect("downloaded dest buffer");

    (assertion)(&dest);
}
