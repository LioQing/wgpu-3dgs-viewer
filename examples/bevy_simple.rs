//! A minimal Bevy example that renders a 3D Gaussian Splatting model.
//!
//! # Usage
//!
//! ```sh
//! cargo run --example bevy-simple --features bevy -- path/to/model.ply
//! ```
//!
//! # Version Compatibility
//!
//! This example requires that Bevy's internal wgpu version matches the wgpu
//! version used by `wgpu-3dgs-viewer` (28.0). See the `bevy_plugin` module
//! documentation for details on how to align versions.

use std::f32::consts::PI;

use bevy::prelude::*;

use wgpu_3dgs_viewer::bevy_plugin::{GaussianCloud, GaussianSplatSettings, GaussianSplattingPlugin};

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, GaussianSplattingPlugin::default()))
        .add_systems(Startup, setup)
        .add_systems(Update, camera_controller)
        .run();
}

fn setup(mut commands: Commands) {
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| {
            eprintln!("Usage: bevy-simple <path/to/model.ply>");
            std::process::exit(1);
        });

    // 3D camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 0.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // Gaussian splat cloud
    commands.spawn((
        GaussianCloud::from_file(&model_path),
        GaussianSplatSettings::default(),
        Transform::from_rotation(Quat::from_axis_angle(Vec3::Z, PI)),
    ));
}

/// Simple WASD + mouse camera controller.
fn camera_controller(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    mut query: Query<&mut Transform, With<Camera3d>>,
) {
    let Ok(mut transform) = query.single_mut() else {
        return;
    };

    let speed = 2.0 * time.delta_secs();
    let forward = *transform.forward();
    let right = *transform.right();

    if keys.pressed(KeyCode::KeyW) {
        transform.translation += forward * speed;
    }
    if keys.pressed(KeyCode::KeyS) {
        transform.translation -= forward * speed;
    }
    if keys.pressed(KeyCode::KeyD) {
        transform.translation += right * speed;
    }
    if keys.pressed(KeyCode::KeyA) {
        transform.translation -= right * speed;
    }
    if keys.pressed(KeyCode::Space) {
        transform.translation += Vec3::Y * speed;
    }
    if keys.pressed(KeyCode::ShiftLeft) {
        transform.translation -= Vec3::Y * speed;
    }
}
