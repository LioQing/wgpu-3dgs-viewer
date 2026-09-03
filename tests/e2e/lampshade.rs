use glam::{Quat, U8Vec4, UVec2, Vec3};
use wgpu_3dgs_viewer::{
    CameraPod, IndirectArgsBuffer, RadixSortIndirectArgsBuffer, RadixSorter, Viewer,
    core::{BufferWrapper, Gaussian, GaussianPodWithShSingleCov3dSingleConfigs, IterGaussian},
};

use crate::common::{TestContext, given};

type G = GaussianPodWithShSingleCov3dSingleConfigs;
const WIDTH: u32 = 64;

fn scene(count: u32) -> Vec<Gaussian> {
    (0..count)
        .map(|i| Gaussian {
            rot: Quat::IDENTITY,
            pos: Vec3::new(
                (i % 8) as f32 * 0.025 - 0.1,
                ((i / 8) % 8) as f32 * 0.025 - 0.1,
                1.0 + i as f32 * 0.001,
            ),
            color: U8Vec4::new((i * 17) as u8, (i * 31) as u8, 200, 40),
            sh: [Vec3::ZERO; 15],
            scale: Vec3::splat(0.02),
        })
        .collect()
}

fn viewer(ctx: &TestContext, gaussians: &impl IterGaussian) -> Viewer<G> {
    let mut viewer =
        Viewer::new(&ctx.device, wgpu::TextureFormat::Rgba8Unorm, gaussians).expect("viewer");
    viewer.update_camera_with_pod(
        &ctx.queue,
        &CameraPod::new(&given::camera(), UVec2::splat(WIDTH)),
    );
    viewer
}

// The existing helper checks channel occupancy, not RGBA equality. Read the
// actual bytes here to catch depth-order/blending changes.
fn render_pixels(ctx: &TestContext, viewer: &Viewer<G>) -> (Vec<u8>, u32) {
    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Lampshade parity target"),
        size: wgpu::Extent3d {
            width: WIDTH,
            height: WIDTH,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Lampshade parity readback"),
        size: u64::from(WIDTH * WIDTH * 4 + 4),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    viewer.render(&mut encoder, &texture.create_view(&Default::default()));
    // Draw buffers lack COPY_SRC. Read the GPU count via a storage binding
    // without changing production usages just for this test.
    let count = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Visible count copy"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let shader = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Visible count readback"),
            source: wgpu::ShaderSource::Wgsl(
                "@group(0) @binding(0) var<storage, read> args: array<u32>;
             @group(0) @binding(1) var<storage, read_write> count: u32;
             @compute @workgroup_size(1) fn main() { count = args[1]; }"
                    .into(),
            ),
        });
    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Visible count readback"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
    let bindings = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Visible count readback"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: viewer.indirect_args_buffer.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: count.as_entire_binding(),
            },
        ],
    });
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bindings, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&count, 0, &readback, u64::from(WIDTH * WIDTH * 4), 4);
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(WIDTH * 4),
                rows_per_image: Some(WIDTH),
            },
        },
        texture.size(),
    );
    ctx.queue.submit([encoder.finish()]);
    let (sender, receiver) = std::sync::mpsc::channel();
    readback.map_async(wgpu::MapMode::Read, .., move |result| {
        sender.send(result).expect("readback receiver");
    });
    ctx.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device poll");
    receiver
        .recv()
        .expect("map callback")
        .expect("map readback");
    let mapped = readback.get_mapped_range(..).expect("mapped bytes");
    let split = (WIDTH * WIDTH * 4) as usize;
    let visible = u32::from_le_bytes(mapped[split..].try_into().expect("count bytes"));
    (mapped[..split].to_vec(), visible)
}

#[test]
fn test_lampshade_render_matches_embedded_as_visibility_changes() {
    let ctx = TestContext::new_with_lampshade();
    if !lampshade::KeyValueSoaSorter::requirements(&ctx.adapter).accelerated {
        assert!(
            std::env::var_os("LAMPSHADE_REQUIRE_ACCELERATED_TESTS").is_none(),
            "this validation run requires an eligible NVIDIA/Vulkan adapter"
        );
        eprintln!("skipping native Lampshade parity: adapter is not eligible");
        return;
    }

    // Cross the embedded 3840-item dispatch boundary; keys have distinct depths.
    let gaussians = scene(3841);
    let mut accelerated = viewer(&ctx, &gaussians);
    assert!(accelerated.uses_lampshade_sorter());
    let mut embedded = viewer(&ctx, &gaussians);
    embedded.radix_sorter = RadixSorter::new(
        &ctx.device,
        &embedded.gaussians_depth_buffer,
        &embedded.indirect_indices_buffer,
    );
    assert!(!embedded.uses_lampshade_sorter());

    // Reuse each prepared viewer across full, partially culled, zero-visible,
    // and full frames to catch stale counts/workspace on consecutive frames.
    for shift in [0.0, -2.0, -10.0, 0.0] {
        for viewer in [&mut accelerated, &mut embedded] {
            viewer.update_model_transform(&ctx.queue, Vec3::Z * shift, Quat::IDENTITY, Vec3::ONE);
        }
        let (expected, expected_count) = render_pixels(&ctx, &embedded);
        let (actual, visible) = render_pixels(&ctx, &accelerated);
        assert_eq!(visible, expected_count);
        if shift == 0.0 {
            assert_eq!(visible, 3841);
        } else if shift == -10.0 {
            assert_eq!(visible, 0);
        } else {
            assert!(visible > 0 && visible < 3841);
        }
        assert_eq!(actual, expected, "RGBA mismatch at model shift {shift}");
        let colored = actual.chunks_exact(4).any(|pixel| pixel[..3] != [0, 0, 0]);
        assert_eq!(colored, shift != -10.0);
    }

    // Replacing metadata restores the dispatch API; a clone of the original
    // handle still selects the prepared plan.
    let draw = accelerated.indirect_args_buffer.clone();
    accelerated.indirect_args_buffer = IndirectArgsBuffer::new(&ctx.device);
    assert!(!accelerated.uses_lampshade_sorter());
    accelerated.indirect_args_buffer = draw;
    assert!(accelerated.uses_lampshade_sorter());
    let dispatch = accelerated.radix_sort_indirect_args_buffer.clone();
    accelerated.radix_sort_indirect_args_buffer = RadixSortIndirectArgsBuffer::new(&ctx.device);
    assert!(!accelerated.uses_lampshade_sorter());
    accelerated.radix_sort_indirect_args_buffer = dispatch;
    assert!(accelerated.uses_lampshade_sorter());
}

#[test]
fn test_lampshade_falls_back_without_enabled_subgroups() {
    let ctx = TestContext::new();
    assert!(!ctx.device.features().contains(wgpu::Features::SUBGROUP));
    let fallback = viewer(&ctx, &scene(1));
    assert!(!fallback.uses_lampshade_sorter());
    let (pixels, visible) = render_pixels(&ctx, &fallback);
    assert_eq!(visible, 1);
    assert!(pixels.chunks_exact(4).any(|pixel| pixel[..3] != [0, 0, 0]));
}
