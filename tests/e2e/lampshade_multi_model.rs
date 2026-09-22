use glam::{Quat, U8Vec4, UVec2, Vec3};
use wgpu_3dgs_viewer::{
    CameraPod, DefaultDepthSorterWithoutBindGroups, DepthSorterWithoutBindGroups, LampshadeSorter,
    MultiModelViewer, MultiModelViewerCreateOptions,
    core::{BufferWrapper, Gaussian, GaussianPodWithShSingleCov3dSingleConfigs, GaussiansBuffer},
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

fn accelerated_viewer(
    ctx: &TestContext,
    models: &[(&'static str, Vec<Gaussian>)],
) -> MultiModelViewer<G, LampshadeSorter<()>, &'static str> {
    let mut viewer = MultiModelViewer::<G, LampshadeSorter<()>, &str>::new_with_options(
        &ctx.device,
        wgpu::TextureFormat::Rgba8Unorm,
        MultiModelViewerCreateOptions {
            depth_stencil: None,
            gaussians_buffer_usage: GaussiansBuffer::<G>::DEFAULT_USAGES,
            color_write_mask: wgpu::ColorWrites::ALL,
            cache: None,
            depth_sorter_factory: |ctx| LampshadeSorter::new_without_bind_groups(ctx.device),
            phantom_data: Default::default(),
        },
    )
    .expect("viewer");

    for (key, gaussians) in models {
        viewer.insert_model(&ctx.device, *key, gaussians);
    }

    viewer.update_camera_with_pod(
        &ctx.queue,
        &CameraPod::new(&given::camera(), UVec2::splat(WIDTH)),
    );
    viewer
}

fn embedded_viewer(
    ctx: &TestContext,
    models: &[(&'static str, Vec<Gaussian>)],
) -> MultiModelViewer<G, DefaultDepthSorterWithoutBindGroups, &'static str> {
    let mut viewer = MultiModelViewer::<G, DefaultDepthSorterWithoutBindGroups, &str>::new(
        &ctx.device,
        wgpu::TextureFormat::Rgba8Unorm,
    )
    .expect("viewer");

    for (key, gaussians) in models {
        viewer.insert_model(&ctx.device, *key, gaussians);
    }

    viewer.update_camera_with_pod(
        &ctx.queue,
        &CameraPod::new(&given::camera(), UVec2::splat(WIDTH)),
    );
    viewer
}

fn render_pixels<S: DepthSorterWithoutBindGroups>(
    ctx: &TestContext,
    viewer: &MultiModelViewer<G, S, &'static str>,
    keys: &[&&'static str],
) -> Vec<u8> {
    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Lampshade multi-model parity target"),
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
        label: Some("Lampshade multi-model parity readback"),
        size: u64::from(WIDTH * WIDTH * 4),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    viewer
        .render(
            &mut encoder,
            &texture.create_view(&Default::default()),
            keys,
        )
        .expect("render");
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
    readback
        .get_mapped_range(..)
        .expect("mapped bytes")
        .to_vec()
}

#[test]
fn test_lampshade_multi_model_render_matches_embedded_as_visibility_changes() {
    let ctx = TestContext::new_with_lampshade();
    if !lampshade::KeyValueSoaSorter::requirements(&ctx.adapter).accelerated {
        assert!(
            std::env::var_os("LAMPSHADE_REQUIRE_ACCELERATED_TESTS").is_none(),
            "this validation run requires an eligible NVIDIA/Vulkan adapter"
        );
        eprintln!("skipping native Lampshade multi-model parity: adapter is not eligible");
        return;
    }

    // Model "a" crosses the embedded 3840-item dispatch boundary, while model "b" stays small so
    // that each model must use its own count buffer and bindings.
    let models = [("a", scene(3841)), ("b", scene(7))];
    let mut accelerated = accelerated_viewer(&ctx, &models);
    assert!(
        accelerated.models[&"a"]
            .bind_groups
            .depth_sorter
            .is_accelerated()
    );
    assert!(
        accelerated.models[&"b"]
            .bind_groups
            .depth_sorter
            .is_accelerated()
    );
    let mut embedded = embedded_viewer(&ctx, &models);

    // Reuse each prepared viewer across full, partially culled, and fully culled frames to catch
    // stale counts or bindings leaking between models.
    for shift in [0.0, -2.0, -10.0, 0.0] {
        accelerated
            .update_model_transform(&ctx.queue, &"a", Vec3::Z * shift, Quat::IDENTITY, Vec3::ONE)
            .expect("update accelerated");
        embedded
            .update_model_transform(&ctx.queue, &"a", Vec3::Z * shift, Quat::IDENTITY, Vec3::ONE)
            .expect("update embedded");

        let expected = render_pixels(&ctx, &embedded, &[&"a", &"b"]);
        let actual = render_pixels(&ctx, &accelerated, &[&"a", &"b"]);
        assert_eq!(actual, expected, "RGBA mismatch at model shift {shift}");

        let colored = actual
            .as_chunks::<4>()
            .0
            .iter()
            .any(|pixel| pixel[..3] != [0, 0, 0]);
        assert!(colored);
    }
}
