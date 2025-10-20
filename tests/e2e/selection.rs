use wgpu_3dgs_editor::Modifier;
use wgpu_3dgs_viewer::{
    Viewer,
    core::{
        BufferWrapper, Gaussian, GaussianPodWithShSingleCov3dSingleConfigs, Gaussians,
        GaussiansBuffer, glam::*,
    },
    editor::{BasicSelectionModifier, NonDestructiveModifier, SelectionExpr},
    selection::{ViewportSelector, ViewportSelectorType, create_viewport_bundle},
};

use crate::common::{TestContext, assert_render_target, given};

type G = GaussianPodWithShSingleCov3dSingleConfigs;

fn test_select_modify_render_and_assert(
    render: impl FnOnce(&TestContext, &mut wgpu::CommandEncoder, &mut ViewportSelector),
    assertion: impl FnOnce(&[UVec4]),
) {
    let ctx = TestContext::new();
    let gaussians = Gaussians {
        gaussians: vec![Gaussian {
            rot: Quat::IDENTITY,
            pos: Vec3::ZERO + Vec3::Z,
            color: U8Vec4::new(255, 0, 0, 255),
            sh: [Vec3::ZERO; 15],
            scale: Vec3::splat(1.0),
        }],
    };

    let render_target = given::render_target_texture(&ctx);
    let camera = given::camera();

    let mut viewer = Viewer::<G>::new_with(
        &ctx.device,
        wgpu::TextureFormat::Rgba8Unorm,
        None,
        GaussiansBuffer::<G>::DEFAULT_USAGES | wgpu::BufferUsages::COPY_SRC,
        &gaussians,
    )
    .expect("viewer");

    viewer.update_camera_with_pod(&ctx.queue, &camera);

    let mut selector = ViewportSelector::new(
        &ctx.device,
        &ctx.queue,
        camera.size.as_uvec2(),
        &viewer.camera_buffer,
    )
    .expect("selector");

    let mut selection_modifier = NonDestructiveModifier::new(
        &ctx.device,
        &ctx.queue,
        BasicSelectionModifier::new_with_basic_modifier(
            &ctx.device,
            &viewer.gaussians_buffer,
            &viewer.model_transform_buffer,
            &viewer.gaussian_transform_buffer,
            vec![create_viewport_bundle::<G>(&ctx.device)],
        ),
        &viewer.gaussians_buffer,
    )
    .expect("modifier");

    let selection_bind_group = selection_modifier.modifier.selection.bundles[0]
        .create_bind_group(
            &ctx.device,
            1,
            [
                viewer.camera_buffer.buffer().as_entire_binding(),
                wgpu::BindingResource::TextureView(selector.texture().view()),
            ],
        )
        .expect("bind group");

    selection_modifier.modifier.selection_expr =
        SelectionExpr::Selection(0, vec![selection_bind_group]);
    // TODO(https://github.com/LioQing/wgpu-3dgs-editor/issues/3): Use a better default instantiation method.
    selection_modifier
        .modifier
        .modifier
        .basic_color_modifiers_buffer
        .update_with_override_rgb(&ctx.queue, Vec3::new(0.0, 0.0, 1.0), 1.0, 0.0, 0.0, 1.0);

    let render_target_view = render_target.create_view(&wgpu::TextureViewDescriptor::default());

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Command Encoder"),
        });

    render(&ctx, &mut encoder, &mut selector);

    selection_modifier.apply(
        &ctx.device,
        &mut encoder,
        &viewer.gaussians_buffer,
        &viewer.model_transform_buffer,
        &viewer.gaussian_transform_buffer,
    );
    viewer.render(&mut encoder, &render_target_view);

    ctx.queue.submit(Some(encoder.finish()));
    ctx.device.poll(wgpu::PollType::Wait).expect("device poll");

    assert_render_target(&ctx, &render_target_view, assertion);
}

#[test]
fn test_viewer_when_gaussian_is_in_selected_rectangle_should_be_selected_and_modified() {
    test_select_modify_render_and_assert(
        |ctx: &TestContext, encoder: &mut wgpu::CommandEncoder, selector: &mut ViewportSelector| {
            selector.selector_type = ViewportSelectorType::Rectangle;
            selector.start(&ctx.queue, Vec2::splat(256.0));
            selector.update(&ctx.queue, Vec2::splat(1024.0 - 256.0));
            selector.render(encoder);
        },
        |pixels: &[UVec4]| {
            let sum = pixels.iter().sum::<UVec4>();
            assert!(sum.x < 1);
            assert!(sum.y < 1);
            assert!(sum.z > 1);
            assert!(sum.w > 1);
        },
    );
}

#[test]
fn test_viewer_when_gaussian_is_not_in_selected_rectangle_should_not_be_selected_and_modified() {
    test_select_modify_render_and_assert(
        |ctx: &TestContext, encoder: &mut wgpu::CommandEncoder, selector: &mut ViewportSelector| {
            selector.selector_type = ViewportSelectorType::Rectangle;
            selector.start(&ctx.queue, Vec2::splat(0.0));
            selector.update(&ctx.queue, Vec2::splat(256.0));
            selector.render(encoder);
        },
        |pixels: &[UVec4]| {
            let sum = pixels.iter().sum::<UVec4>();
            assert!(sum.x > 1);
            assert!(sum.y < 1);
            assert!(sum.z < 1);
            assert!(sum.w > 1);
        },
    );
}

#[test]
fn test_viewer_when_gaussian_is_in_selected_brush_should_be_selected_and_modified() {
    test_select_modify_render_and_assert(
        |ctx: &TestContext, encoder: &mut wgpu::CommandEncoder, selector: &mut ViewportSelector| {
            selector.selector_type = ViewportSelectorType::Brush;
            selector.start(&ctx.queue, Vec2::splat(256.0));
            selector.update(&ctx.queue, Vec2::splat(1024.0 - 256.0));
            selector.render(encoder);
        },
        |pixels: &[UVec4]| {
            let sum = pixels.iter().sum::<UVec4>();
            assert!(sum.x < 1);
            assert!(sum.y < 1);
            assert!(sum.z > 1);
            assert!(sum.w > 1);
        },
    );
}

#[test]
fn test_viewer_when_gaussian_is_not_in_selected_brush_should_not_be_selected_and_modified() {
    test_select_modify_render_and_assert(
        |ctx: &TestContext, encoder: &mut wgpu::CommandEncoder, selector: &mut ViewportSelector| {
            selector.selector_type = ViewportSelectorType::Brush;
            selector.start(&ctx.queue, Vec2::splat(0.0));
            selector.update(&ctx.queue, Vec2::splat(256.0));
            selector.render(encoder);
        },
        |pixels: &[UVec4]| {
            let sum = pixels.iter().sum::<UVec4>();
            assert!(sum.x > 1);
            assert!(sum.y < 1);
            assert!(sum.z < 1);
            assert!(sum.w > 1);
        },
    );
}

#[test]
fn test_viewer_when_selection_is_cleared_should_not_be_selected_and_modified() {
    test_select_modify_render_and_assert(
        |ctx: &TestContext, encoder: &mut wgpu::CommandEncoder, selector: &mut ViewportSelector| {
            selector.selector_type = ViewportSelectorType::Rectangle;
            selector.start(&ctx.queue, Vec2::splat(256.0));
            selector.update(&ctx.queue, Vec2::splat(1024.0 - 256.0));
            selector.render(encoder);
            selector.clear(encoder);
        },
        |pixels: &[UVec4]| {
            let sum = pixels.iter().sum::<UVec4>();
            assert!(sum.x > 1);
            assert!(sum.y < 1);
            assert!(sum.z < 1);
            assert!(sum.w > 1);
        },
    );
}
