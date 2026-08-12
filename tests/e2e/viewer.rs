use glam::*;
use wgpu_3dgs_core::GaussianMaxStdDev;
use wgpu_3dgs_viewer::{
    CameraPod, Viewer,
    core::{
        Gaussian, GaussianDisplayMode, GaussianPodWithShSingleCov3dSingleConfigs, GaussianShDegree,
        GaussianTransformPod, ModelTransformPod,
    },
};

use crate::common::{TestContext, assert_render_target, given};

type G = GaussianPodWithShSingleCov3dSingleConfigs;

fn render_and_assert(
    ctx: &TestContext,
    viewer: &Viewer<G>,
    render_target: &wgpu::Texture,
    assertion: impl Fn(&[UVec4]),
) {
    let render_target_view = render_target.create_view(&wgpu::TextureViewDescriptor::default());

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Command Encoder"),
        });

    viewer.render(&mut encoder, &render_target_view);

    ctx.queue.submit(Some(encoder.finish()));
    ctx.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device poll");

    assert_render_target(ctx, &render_target_view, assertion);
}

#[test]
fn test_viewer_update_camera_when_with_or_without_pod_should_be_equal() {
    let ctx = TestContext::new();
    let gaussians = vec![Gaussian {
        rot: Quat::IDENTITY,
        pos: Vec3::ZERO + Vec3::Z,
        color: U8Vec4::new(255, 0, 0, 255),
        sh: [Vec3::ZERO; 15],
        scale: Vec3::splat(1.0),
    }];

    let render_target1 = given::render_target_texture(&ctx);
    let render_target2 = given::render_target_texture(&ctx);
    let size = UVec2::new(render_target1.size().width, render_target1.size().height);
    let camera = given::camera();
    let camera_pod = CameraPod::new(&camera, size);

    let mut viewer1 =
        Viewer::<G>::new(&ctx.device, wgpu::TextureFormat::Rgba8Unorm, &gaussians).expect("viewer");
    let mut viewer2 =
        Viewer::<G>::new(&ctx.device, wgpu::TextureFormat::Rgba8Unorm, &gaussians).expect("viewer");

    viewer1.update_camera_with_pod(&ctx.queue, &camera_pod);
    viewer2.update_camera(&ctx.queue, &camera, size);

    render_and_assert(&ctx, &viewer1, &render_target1, |pixels1: &[UVec4]| {
        render_and_assert(&ctx, &viewer2, &render_target2, |pixels2: &[UVec4]| {
            assert_eq!(pixels1, pixels2);
        });
    });
}

#[test]
fn test_viewer_render_should_render_correctly() {
    let ctx = TestContext::new();
    let gaussians = vec![Gaussian {
        rot: Quat::IDENTITY,
        pos: Vec3::ZERO + Vec3::Z,
        color: U8Vec4::new(255, 0, 0, 255),
        sh: [Vec3::ZERO; 15],
        scale: Vec3::splat(1.0),
    }];

    let render_target = given::render_target_texture(&ctx);

    let mut viewer =
        Viewer::<G>::new(&ctx.device, wgpu::TextureFormat::Rgba8Unorm, &gaussians).expect("viewer");

    viewer.update_camera_with_pod(&ctx.queue, &given::camera_pod());

    render_and_assert(&ctx, &viewer, &render_target, |pixels: &[UVec4]| {
        let sum = pixels.iter().sum::<UVec4>();
        assert!(sum.x > 1);
        assert!(sum.y < 1);
        assert!(sum.z < 1);
        assert!(sum.w > 1);
    });
}

#[test]
fn test_adapter_aware_viewer_render_should_render_correctly() {
    let ctx = TestContext::new_with_subgroups();
    let gaussians = vec![Gaussian {
        rot: Quat::IDENTITY,
        pos: Vec3::ZERO + Vec3::Z,
        color: U8Vec4::new(255, 0, 0, 255),
        sh: [Vec3::ZERO; 15],
        scale: Vec3::splat(1.0),
    }];
    let render_target = given::render_target_texture(&ctx);
    let adapter_info = ctx.adapter.get_info();
    let mut viewer = Viewer::<G>::new_for_adapter(
        &ctx.device,
        &adapter_info,
        wgpu::TextureFormat::Rgba8Unorm,
        &gaussians,
    )
    .expect("adapter-aware viewer");
    let _legacy_bind_groups = viewer.radix_sorter.create_bind_groups(
        &ctx.device,
        &viewer.gaussians_depth_buffer,
        &viewer.indirect_indices_buffer,
    );
    viewer.update_camera_with_pod(&ctx.queue, &given::camera_pod());
    render_and_assert(&ctx, &viewer, &render_target, |pixels: &[UVec4]| {
        let sum = pixels.iter().sum::<UVec4>();
        assert!(sum.x > 1);
        assert!(sum.y < 1);
        assert!(sum.z < 1);
        assert!(sum.w > 1);
    });
}

#[test]
fn test_adapter_aware_viewer_renders_without_enabled_subgroups() {
    let ctx = TestContext::new();
    assert!(!ctx.device.features().contains(wgpu::Features::SUBGROUP));
    let gaussians = vec![Gaussian {
        rot: Quat::IDENTITY,
        pos: Vec3::ZERO + Vec3::Z,
        color: U8Vec4::new(255, 0, 0, 255),
        sh: [Vec3::ZERO; 15],
        scale: Vec3::splat(1.0),
    }];
    let adapter_info = ctx.adapter.get_info();
    let mut viewer = Viewer::<G>::new_for_adapter(
        &ctx.device,
        &adapter_info,
        wgpu::TextureFormat::Rgba8Unorm,
        &gaussians,
    )
    .expect("fallback viewer");
    viewer.update_camera_with_pod(&ctx.queue, &given::camera_pod());
    let render_target = given::render_target_texture(&ctx);
    render_and_assert(&ctx, &viewer, &render_target, |pixels: &[UVec4]| {
        assert!(pixels.iter().sum::<UVec4>().x > 1);
    });
}

#[test]
fn test_adapter_aware_viewer_matches_legacy_for_overlapping_multi_gaussian_scene() {
    let ctx = TestContext::new_with_subgroups();
    let gaussians: Vec<_> = (0_u32..64)
        .map(|index| {
            let column = (index % 4) as f32 - 1.5;
            let row = ((index / 4) % 4) as f32 - 1.5;
            let depth_band = (index % 8) as f32;
            Gaussian {
                rot: Quat::IDENTITY,
                pos: Vec3::new(column * 0.08, row * 0.08, 1.0 + depth_band * 0.04),
                color: U8Vec4::new(
                    (31 + index * 17) as u8,
                    (251 - index * 3) as u8,
                    (67 + index * 11) as u8,
                    96,
                ),
                sh: [Vec3::ZERO; 15],
                scale: Vec3::splat(0.18),
            }
        })
        .collect();
    let adapter_info = ctx.adapter.get_info();
    let mut legacy = Viewer::<G>::new(&ctx.device, wgpu::TextureFormat::Rgba8Unorm, &gaussians)
        .expect("legacy viewer");
    let mut adaptive = Viewer::<G>::new_for_adapter(
        &ctx.device,
        &adapter_info,
        wgpu::TextureFormat::Rgba8Unorm,
        &gaussians,
    )
    .expect("adapter-aware viewer");
    #[cfg(feature = "lampshade-sort")]
    if adapter_info.backend == wgpu::Backend::Vulkan
        && adapter_info.vendor == 0x10de
        && adapter_info.device_type == wgpu::DeviceType::DiscreteGpu
        && ctx.device.features().contains(wgpu::Features::SUBGROUP)
    {
        assert!(
            format!("{:?}", adaptive.radix_sorter).contains("lampshade: true"),
            "a compatible NVIDIA/Vulkan device should select Lampshade"
        );
    }
    let camera = given::camera_pod();
    legacy.update_camera_with_pod(&ctx.queue, &camera);
    adaptive.update_camera_with_pod(&ctx.queue, &camera);
    let legacy_target = given::render_target_texture(&ctx);
    let adaptive_target = given::render_target_texture(&ctx);

    render_and_assert(
        &ctx,
        &legacy,
        &legacy_target,
        |legacy_pixels: &[UVec4]| {
            render_and_assert(
                &ctx,
                &adaptive,
                &adaptive_target,
                |adaptive_pixels: &[UVec4]| {
                    assert_eq!(adaptive_pixels, legacy_pixels);
                },
            );
        },
    );
}

fn test_viewer_when_no_sh0_is_set_should_and_render_as_grayscale(
    update_gaussian_transform: impl FnOnce(&mut Viewer<G>, &wgpu::Queue),
) {
    let ctx = TestContext::new();
    let gaussians = vec![Gaussian {
        rot: Quat::IDENTITY,
        pos: Vec3::ZERO + Vec3::Z,
        color: U8Vec4::new(255, 0, 0, 255),
        sh: [Vec3::ZERO; 15],
        scale: Vec3::splat(1.0),
    }];

    let render_target = given::render_target_texture(&ctx);

    let mut viewer =
        Viewer::<G>::new(&ctx.device, wgpu::TextureFormat::Rgba8Unorm, &gaussians).expect("viewer");

    viewer.update_camera_with_pod(&ctx.queue, &given::camera_pod());
    update_gaussian_transform(&mut viewer, &ctx.queue);

    render_and_assert(&ctx, &viewer, &render_target, |pixels: &[UVec4]| {
        let sum = pixels.iter().sum::<UVec4>();
        assert!(sum.x > 1);
        assert!(sum.y > 1);
        assert!(sum.z > 1);
        assert!(sum.w > 1);
    });
}

#[test]
fn test_viewer_update_gaussian_transform_when_no_sh0_is_set_should_render_as_grayscale() {
    test_viewer_when_no_sh0_is_set_should_and_render_as_grayscale(|viewer, queue| {
        viewer.update_gaussian_transform(
            queue,
            1.0,
            GaussianDisplayMode::Splat,
            GaussianShDegree::new(3).expect("sh deg"),
            true,
            GaussianMaxStdDev::new(3.0).expect("max std dev"),
        );
    });
}

#[test]
fn test_viewer_update_gaussian_transform_with_pod_when_no_sh0_is_set_should_render_as_grayscale() {
    test_viewer_when_no_sh0_is_set_should_and_render_as_grayscale(|viewer, queue| {
        viewer.update_gaussian_transform_with_pod(
            queue,
            &GaussianTransformPod::new(
                1.0,
                GaussianDisplayMode::Splat,
                GaussianShDegree::new(3).expect("sh deg"),
                true,
                GaussianMaxStdDev::new(3.0).expect("max std dev"),
            ),
        );
    });
}

fn test_viewer_when_model_pos_is_behind_camera_should_not_render_gaussian(
    update_model_transform: impl FnOnce(&mut Viewer<G>, &wgpu::Queue),
) {
    let ctx = TestContext::new();
    let gaussians = vec![Gaussian {
        rot: Quat::IDENTITY,
        pos: Vec3::ZERO + Vec3::Z,
        color: U8Vec4::new(255, 0, 0, 255),
        sh: [Vec3::ZERO; 15],
        scale: Vec3::splat(1.0),
    }];

    let render_target = given::render_target_texture(&ctx);

    let mut viewer =
        Viewer::<G>::new(&ctx.device, wgpu::TextureFormat::Rgba8Unorm, &gaussians).expect("viewer");

    viewer.update_camera_with_pod(&ctx.queue, &given::camera_pod());
    update_model_transform(&mut viewer, &ctx.queue);

    render_and_assert(&ctx, &viewer, &render_target, |pixels: &[UVec4]| {
        let sum = pixels.iter().sum::<UVec4>();
        assert_eq!(sum.x, 0);
        assert_eq!(sum.y, 0);
        assert_eq!(sum.z, 0);
    });
}

#[test]
fn test_viewer_update_model_transform_when_model_pos_is_behind_camera_should_not_render_gaussian() {
    test_viewer_when_model_pos_is_behind_camera_should_not_render_gaussian(|viewer, queue| {
        viewer.update_model_transform(queue, Vec3::ZERO - Vec3::Z, Quat::IDENTITY, Vec3::ONE);
    });
}

#[test]
fn test_viewer_update_model_transform_with_pod_when_model_pos_is_behind_camera_should_not_render_gaussian()
 {
    test_viewer_when_model_pos_is_behind_camera_should_not_render_gaussian(|viewer, queue| {
        viewer.update_model_transform_with_pod(
            queue,
            &ModelTransformPod::new(Vec3::ZERO - Vec3::Z, Quat::IDENTITY, Vec3::ONE),
        );
    });
}
