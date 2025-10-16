use pollster::FutureExt;
use wgpu_3dgs_viewer::{
    Camera, CameraPod, Viewer,
    core::{
        DownloadableBufferWrapper, Gaussian, GaussianPodWithShSingleCov3dSingleConfigs, Gaussians,
        glam::*,
    },
};

use crate::{common::TestContext, inline_wesl_pkg};

type G = GaussianPodWithShSingleCov3dSingleConfigs;

const ASSERT_RENDER_TARGET_PACKAGE: wesl::Pkg = inline_wesl_pkg!(
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

const ASSERT_RENDER_TARGET_BIND_GROUP_LAOYOUT_DESCRIPTOR: wgpu::BindGroupLayoutDescriptor<'static> =
    wgpu::BindGroupLayoutDescriptor {
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

struct RenderTargetAsserter {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    dest_buffer: wgpu::Buffer,
    assertion: fn(&[UVec4]),
}

impl RenderTargetAsserter {
    fn new(device: &wgpu::Device, assertion: fn(&[UVec4])) -> Self {
        // TOOD(https://github.com/LioQing/wgpu-3dgs-core/issues/8): configurable workgroup size.
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
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

        let dest_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Assert Render Target Destination Buffer"),
            size: 1024 / 8 * 1024 / 8 * std::mem::size_of::<UVec4>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group_layout =
            device.create_bind_group_layout(&ASSERT_RENDER_TARGET_BIND_GROUP_LAOYOUT_DESCRIPTOR);

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Assert Render Target Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Assert Render Target Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            pipeline,
            bind_group_layout,
            dest_buffer,
            assertion,
        }
    }

    fn assert(&self, device: &wgpu::Device, queue: &wgpu::Queue, texture_view: &wgpu::TextureView) {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Assert Render Target Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.dest_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Command Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Assert Render Target Compute Pass"),
                ..Default::default()
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1024 / 8 * 1024 / 8, 1, 1);
        }

        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::PollType::Wait).expect("device poll");

        let dest = self
            .dest_buffer
            .download(device, queue)
            .block_on()
            .expect("downloaded dest buffer");

        (self.assertion)(&dest);
    }
}

#[test]
fn test_viewer_new_should_create_viewer() {
    let ctx = TestContext::new();
    let gaussians = Gaussians {
        gaussians: vec![Gaussian {
            rot: Quat::IDENTITY,
            pos: Vec3::ZERO + Vec3::Z,
            color: U8Vec4::new(255, 0, 0, 255),
            sh: [Vec3::ZERO; 15],
            scale: Vec3::splat(10.0),
        }],
    };

    let mut viewer =
        Viewer::<G>::new(&ctx.device, wgpu::TextureFormat::Rgba8Unorm, &gaussians).expect("viewer");

    assert_eq!(viewer.gaussians_buffer.len(), 1);

    viewer.update_camera_with_pod(
        &ctx.queue,
        &CameraPod::new(
            // TODO(#8): Fix camera orientation edge case when yaw or pitch is 0.0
            &Camera {
                yaw: 0.1,
                pitch: 0.1,
                ..Camera::new(0.1..1e4, 60f32.to_radians())
            },
            UVec2::new(1024, 1024),
        ),
    );

    let render_target = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Render Target"),
        size: wgpu::Extent3d {
            width: 1024,
            height: 1024,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    let render_target_view = render_target.create_view(&wgpu::TextureViewDescriptor::default());

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Command Encoder"),
        });

    viewer.render(&mut encoder, &render_target_view);

    ctx.queue.submit(Some(encoder.finish()));
    ctx.device.poll(wgpu::PollType::Wait).expect("device poll");

    let asserter = RenderTargetAsserter::new(&ctx.device, |pixels: &[UVec4]| {
        let sum = pixels.iter().sum::<UVec4>();
        assert!(sum.x > 1);
        assert!(sum.y < 1);
        assert!(sum.z < 1);
        assert!(sum.w > 1);
    });

    asserter.assert(&ctx.device, &ctx.queue, &render_target_view);
}
