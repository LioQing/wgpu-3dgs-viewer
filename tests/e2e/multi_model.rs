use glam::*;
use wgpu_3dgs_core::{BufferWrapper, GaussianMaxStdDev};
use wgpu_3dgs_viewer::{
    CameraPod, DefaultDepthSorterWithoutBindGroups, DefaultGaussianPod, MultiModelViewer,
    MultiModelViewerGaussianBuffers,
    core::{
        Gaussian, GaussianDisplayMode, GaussianShDegree, GaussianTransformPod, Gaussians,
        GaussiansBuffer, ModelTransformPod, ShDegree0,
    },
};

use crate::common::{TestContext, assert_render_target, given};

type G = DefaultGaussianPod;
type S = DefaultDepthSorterWithoutBindGroups;

#[test]
fn test_multi_model_when_storage_degree_mismatches_should_preserve_existing_model() {
    let ctx = TestContext::new();
    let gaussian: Gaussian = Gaussian {
        rot: Quat::IDENTITY,
        pos: Vec3::Z,
        color: Vec4::new(1.0, 0.0, 0.0, 1.0),
        sh: [Vec3::ZERO; 15],
        scale: Vec3::ONE,
    };
    let gaussians = Gaussians::from(vec![gaussian]);
    let mut viewer =
        MultiModelViewer::<G, S, &str>::new(&ctx.device, wgpu::TextureFormat::Rgba8Unorm)
            .expect("viewer");

    assert!(
        viewer
            .insert_model(&ctx.device, "model", &gaussians)
            .expect("model")
            .is_none()
    );
    assert!(
        viewer
            .insert_model(&ctx.device, "model", &gaussians)
            .expect("model")
            .is_some()
    );
    assert_eq!(
        MultiModelViewerGaussianBuffers::<G>::new(&ctx.device, &gaussians)
            .expect("buffers")
            .gaussians_buffer
            .len(),
        1,
    );

    for count in [0, 2] {
        let mismatched = Gaussians::from(vec![gaussian.convert_sh_degree::<ShDegree0>(); count]);
        let error = MultiModelViewerGaussianBuffers::<G>::new(&ctx.device, &mismatched)
            .expect_err("SH degree mismatch");
        assert_eq!(error.actual_degree, 0);
        assert_eq!(error.expected_degree, 3);

        let error = MultiModelViewerGaussianBuffers::<G>::new_with(
            &ctx.device,
            GaussiansBuffer::<G>::DEFAULT_USAGES,
            &mismatched,
        )
        .expect_err("SH degree mismatch");
        assert_eq!(error.actual_degree, 0);
        assert_eq!(error.expected_degree, 3);

        let error = viewer
            .insert_model(&ctx.device, "new", &mismatched)
            .expect_err("SH degree mismatch");
        assert_eq!(error.actual_degree, 0);
        assert_eq!(error.expected_degree, 3);
        assert!(!viewer.models.contains_key("new"));

        let error = viewer
            .insert_model_with(
                &ctx.device,
                "model",
                GaussiansBuffer::<G>::DEFAULT_USAGES,
                &mismatched,
            )
            .expect_err("SH degree mismatch");
        assert_eq!(error.actual_degree, 0);
        assert_eq!(error.expected_degree, 3);
        assert_eq!(viewer.models.len(), 1);
        assert_eq!(
            viewer.models["model"]
                .gaussian_buffers
                .gaussians_buffer
                .len(),
            1
        );
    }
}

fn render_and_assert(
    ctx: &TestContext,
    viewer: &MultiModelViewer<G, S, &str>,
    render_target: &wgpu::Texture,
    keys: &[&&str],
    assertion: impl Fn(&[UVec4]),
) {
    let render_target_view = render_target.create_view(&wgpu::TextureViewDescriptor::default());

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Command Encoder"),
        });

    viewer
        .render(&mut encoder, &render_target_view, keys)
        .expect("render");

    ctx.queue.submit(Some(encoder.finish()));
    ctx.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device poll");

    assert_render_target(ctx, &render_target_view, assertion);
}

#[test]
fn test_multi_model_viewer_gaussian_buffers_new_empty_should_create_buffer_with_correct_size() {
    let ctx = TestContext::new();
    let count = 42;
    let viewer = MultiModelViewerGaussianBuffers::<G>::new_empty(&ctx.device, count);

    assert_eq!(viewer.gaussians_buffer.len(), 42);
    assert_eq!(
        viewer.gaussians_buffer.buffer().size(),
        (count * std::mem::size_of::<G>()) as wgpu::BufferAddress
    );
}

#[test]
fn test_multi_model_viewer_update_camera_when_with_or_without_pod_should_be_equal() {
    let ctx = TestContext::new();
    let red_gaussians: Vec<Gaussian> = vec![Gaussian {
        rot: Quat::IDENTITY,
        pos: Vec3::ZERO + Vec3::Z,
        color: Vec4::new(1.0, 0.0, 0.0, 1.0),
        sh: [Vec3::ZERO; 15],
        scale: Vec3::splat(1.0),
    }];

    let green_gaussians: Vec<Gaussian> = vec![Gaussian {
        rot: Quat::IDENTITY,
        pos: Vec3::new(1.0, 0.0, 1.0),
        color: Vec4::new(0.0, 1.0, 0.0, 1.0),
        sh: [Vec3::ZERO; 15],
        scale: Vec3::splat(1.0),
    }];

    let render_target1 = given::render_target_texture(&ctx);
    let render_target2 = given::render_target_texture(&ctx);
    let size = UVec2::new(render_target1.size().width, render_target1.size().height);
    let camera = given::camera();
    let camera_pod = CameraPod::new(&camera, size);

    let mut viewer1 =
        MultiModelViewer::<G, S, &str>::new(&ctx.device, wgpu::TextureFormat::Rgba8Unorm)
            .expect("viewer");
    let mut viewer2 =
        MultiModelViewer::<G, S, &str>::new(&ctx.device, wgpu::TextureFormat::Rgba8Unorm)
            .expect("viewer");

    viewer1
        .insert_model(&ctx.device, "red", &red_gaussians)
        .expect("model");
    viewer1
        .insert_model(&ctx.device, "green", &green_gaussians)
        .expect("model");
    viewer2
        .insert_model(&ctx.device, "red", &red_gaussians)
        .expect("model");
    viewer2
        .insert_model(&ctx.device, "green", &green_gaussians)
        .expect("model");

    viewer1.update_camera_with_pod(&ctx.queue, &camera_pod);
    viewer2.update_camera(&ctx.queue, &camera, size);

    render_and_assert(
        &ctx,
        &viewer1,
        &render_target1,
        &[&"red", &"green"],
        |pixels1: &[UVec4]| {
            render_and_assert(
                &ctx,
                &viewer2,
                &render_target2,
                &[&"red", &"green"],
                |pixels2: &[UVec4]| {
                    assert_eq!(pixels1, pixels2);
                },
            );
        },
    );
}

#[test]
fn test_multi_model_viewer_render_should_render_correctly() {
    let ctx = TestContext::new();
    let red_gaussians: Vec<Gaussian> = vec![Gaussian {
        rot: Quat::IDENTITY,
        pos: Vec3::ZERO + Vec3::Z,
        color: Vec4::new(1.0, 0.0, 0.0, 1.0),
        sh: [Vec3::ZERO; 15],
        scale: Vec3::splat(1.0),
    }];

    let green_gaussians: Vec<Gaussian> = vec![Gaussian {
        rot: Quat::IDENTITY,
        pos: Vec3::new(1.0, 0.0, 1.0),
        color: Vec4::new(0.0, 1.0, 0.0, 1.0),
        sh: [Vec3::ZERO; 15],
        scale: Vec3::splat(1.0),
    }];

    let render_target = given::render_target_texture(&ctx);

    let mut viewer =
        MultiModelViewer::<G, S, &str>::new(&ctx.device, wgpu::TextureFormat::Rgba8Unorm)
            .expect("viewer");

    viewer
        .insert_model(&ctx.device, "red", &red_gaussians)
        .expect("model");
    viewer
        .insert_model(&ctx.device, "green", &green_gaussians)
        .expect("model");

    viewer.update_camera_with_pod(&ctx.queue, &given::camera_pod());

    render_and_assert(
        &ctx,
        &viewer,
        &render_target,
        &[&"red", &"green"],
        |pixels: &[UVec4]| {
            let sum = pixels.iter().sum::<UVec4>();
            assert!(sum.x > 1);
            assert!(sum.y > 1);
            assert!(sum.z < 1);
            assert!(sum.w > 1);
        },
    );
}

fn test_multi_model_viewer_when_no_sh0_is_set_should_render_as_grayscale(
    update_gaussian_transform: impl FnOnce(&mut MultiModelViewer<G, S, &str>, &wgpu::Queue),
) {
    let ctx = TestContext::new();
    let red_gaussians: Vec<Gaussian> = vec![Gaussian {
        rot: Quat::IDENTITY,
        pos: Vec3::ZERO + Vec3::Z,
        color: Vec4::new(1.0, 0.0, 0.0, 1.0),
        sh: [Vec3::ZERO; 15],
        scale: Vec3::splat(1.0),
    }];

    let green_gaussians: Vec<Gaussian> = vec![Gaussian {
        rot: Quat::IDENTITY,
        pos: Vec3::new(1.0, 0.0, 1.0),
        color: Vec4::new(0.0, 1.0, 0.0, 1.0),
        sh: [Vec3::ZERO; 15],
        scale: Vec3::splat(1.0),
    }];

    let render_target = given::render_target_texture(&ctx);

    let mut viewer =
        MultiModelViewer::<G, S, &str>::new(&ctx.device, wgpu::TextureFormat::Rgba8Unorm)
            .expect("viewer");

    viewer
        .insert_model(&ctx.device, "red", &red_gaussians)
        .expect("model");
    viewer
        .insert_model(&ctx.device, "green", &green_gaussians)
        .expect("model");

    viewer.update_camera_with_pod(&ctx.queue, &given::camera_pod());
    update_gaussian_transform(&mut viewer, &ctx.queue);

    render_and_assert(
        &ctx,
        &viewer,
        &render_target,
        &[&"red", &"green"],
        |pixels: &[UVec4]| {
            let sum = pixels.iter().sum::<UVec4>();
            assert!(sum.x > 1);
            assert!(sum.y > 1);
            assert!(sum.z > 1);
            assert!(sum.w > 1);
        },
    );
}

#[test]
fn test_multi_model_viewer_update_gaussian_transform_when_no_sh0_is_set_should_render_as_grayscale()
{
    test_multi_model_viewer_when_no_sh0_is_set_should_render_as_grayscale(|viewer, queue| {
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
fn test_multi_model_viewer_update_gaussian_transform_with_pod_when_no_sh0_is_set_should_render_as_grayscale()
 {
    test_multi_model_viewer_when_no_sh0_is_set_should_render_as_grayscale(|viewer, queue| {
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

fn test_multi_model_viewer_when_model_pos_is_behind_camera_should_not_render_gaussian(
    update_model_transform: impl FnOnce(&mut MultiModelViewer<G, S, &str>, &wgpu::Queue),
) {
    let ctx = TestContext::new();
    let red_gaussians: Vec<Gaussian> = vec![Gaussian {
        rot: Quat::IDENTITY,
        pos: Vec3::ZERO + Vec3::Z,
        color: Vec4::new(1.0, 0.0, 0.0, 1.0),
        sh: [Vec3::ZERO; 15],
        scale: Vec3::splat(1.0),
    }];

    let green_gaussians: Vec<Gaussian> = vec![Gaussian {
        rot: Quat::IDENTITY,
        pos: Vec3::new(1.0, 0.0, 1.0),
        color: Vec4::new(0.0, 1.0, 0.0, 1.0),
        sh: [Vec3::ZERO; 15],
        scale: Vec3::splat(1.0),
    }];

    let render_target = given::render_target_texture(&ctx);

    let mut viewer =
        MultiModelViewer::<G, S, &str>::new(&ctx.device, wgpu::TextureFormat::Rgba8Unorm)
            .expect("viewer");

    viewer
        .insert_model(&ctx.device, "red", &red_gaussians)
        .expect("model");
    viewer
        .insert_model(&ctx.device, "green", &green_gaussians)
        .expect("model");

    viewer.update_camera_with_pod(&ctx.queue, &given::camera_pod());
    update_model_transform(&mut viewer, &ctx.queue);

    render_and_assert(
        &ctx,
        &viewer,
        &render_target,
        &[&"red", &"green"],
        |pixels: &[UVec4]| {
            let sum = pixels.iter().sum::<UVec4>();
            assert_eq!(sum.x, 0);
            assert_eq!(sum.y, 0);
            assert_eq!(sum.z, 0);
        },
    );
}

#[test]
fn test_multi_model_viewer_update_model_transform_when_model_pos_is_behind_camera_should_not_render_gaussian()
 {
    test_multi_model_viewer_when_model_pos_is_behind_camera_should_not_render_gaussian(
        |viewer, queue| {
            viewer
                .update_model_transform(
                    queue,
                    &"red",
                    Vec3::ZERO - Vec3::Z,
                    Quat::IDENTITY,
                    Vec3::ONE,
                )
                .expect("update red");
            viewer
                .update_model_transform(
                    queue,
                    &"green",
                    Vec3::ZERO - Vec3::Z,
                    Quat::IDENTITY,
                    Vec3::ONE,
                )
                .expect("update green");
        },
    );
}

#[test]
fn test_multi_model_viewer_update_model_transform_with_pod_when_model_pos_is_behind_camera_should_not_render_gaussian()
 {
    test_multi_model_viewer_when_model_pos_is_behind_camera_should_not_render_gaussian(
        |viewer, queue| {
            viewer
                .update_model_transform_with_pod(
                    queue,
                    &"red",
                    &ModelTransformPod::new(Vec3::ZERO - Vec3::Z, Quat::IDENTITY, Vec3::ONE),
                )
                .expect("update red");
            viewer
                .update_model_transform_with_pod(
                    queue,
                    &"green",
                    &ModelTransformPod::new(Vec3::ZERO - Vec3::Z, Quat::IDENTITY, Vec3::ONE),
                )
                .expect("update green");
        },
    );
}

#[test]
fn test_multi_model_viewer_remove_model_should_not_render_removed_model() {
    let ctx = TestContext::new();
    let red_gaussians: Vec<Gaussian> = vec![Gaussian {
        rot: Quat::IDENTITY,
        pos: Vec3::ZERO + Vec3::Z,
        color: Vec4::new(1.0, 0.0, 0.0, 1.0),
        sh: [Vec3::ZERO; 15],
        scale: Vec3::splat(1.0),
    }];

    let green_gaussians: Vec<Gaussian> = vec![Gaussian {
        rot: Quat::IDENTITY,
        pos: Vec3::new(1.0, 0.0, 1.0),
        color: Vec4::new(0.0, 1.0, 0.0, 1.0),
        sh: [Vec3::ZERO; 15],
        scale: Vec3::splat(1.0),
    }];

    let render_target = given::render_target_texture(&ctx);

    let mut viewer =
        MultiModelViewer::<G, S, &str>::new(&ctx.device, wgpu::TextureFormat::Rgba8Unorm)
            .expect("viewer");

    viewer
        .insert_model(&ctx.device, "red", &red_gaussians)
        .expect("model");
    viewer
        .insert_model(&ctx.device, "green", &green_gaussians)
        .expect("model");

    viewer.update_camera_with_pod(&ctx.queue, &given::camera_pod());

    viewer.remove_model(&"green");

    render_and_assert(
        &ctx,
        &viewer,
        &render_target,
        &[&"red"],
        |pixels: &[UVec4]| {
            let sum = pixels.iter().sum::<UVec4>();
            assert!(sum.x > 1);
            assert!(sum.y < 1);
            assert!(sum.z < 1);
            assert!(sum.w > 1);
        },
    );
}
