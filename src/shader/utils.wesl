import super::camera::{ Camera, world_to_camera };
import wgpu_3dgs_core::{
    gaussian::{
        Gaussian,
        gaussian_unpack_cov3d,
        gaussian_unpack_sh,
    },
    model_transform::{
        ModelTransform,
        model_to_world,
        model_transform_mat,
        model_scale_rotation_mat,
    },
};

/// Determine if a Gaussian should be culled given a camera.
fn cull(gaussian: Gaussian, model_transform: ModelTransform, camera: Camera) -> bool {
    let world_pos = model_to_world(model_transform, gaussian.pos);
    let proj_pos = world_to_camera(camera, world_pos);
    let ndc_pos = proj_pos.xyz / proj_pos.w;
    return !(all(ndc_pos >= vec3<f32>(-1.0, -1.0, 0.0)) && all(ndc_pos <= vec3<f32>(1.0)));
}

// Project a 3D covariance matrix into a 2D covariance matrix.
fn cov2d(gaussian: Gaussian, model_transform: ModelTransform) -> vec3<f32> {
    let cov3d = gaussian_unpack_cov3d(gaussian);
    let sr = model_scale_rotation_mat(model_transform);

    let vrk = mat3x3<f32>(
        cov3d[0], cov3d[1], cov3d[2],
        cov3d[1], cov3d[3], cov3d[4],
        cov3d[2], cov3d[4], cov3d[5],
    );

    let focal = vec2<f32>(camera.proj[0][0], camera.proj[1][1]) * camera.size;

    let t = camera.view * model_transform_mat(model_transform) * vec4<f32>(gaussian.pos, 1.0);
    let j = transpose(mat3x3<f32>(
        focal.x / t.z, 0.0, -(focal.x * t.x) / (t.z * t.z),
        0.0, focal.y / t.z, -(focal.y * t.y) / (t.z * t.z),
        0.0, 0.0, 0.0,
    ));
    let w = mat3x3<f32>(camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz);

    let cov2d = (j * w * sr) * vrk * transpose(j * w * sr);

    return vec3<f32>(cov2d[0][0], cov2d[0][1], cov2d[1][1]);
}

// Calculate the diagonal axes of a 2D covariance matrix.
//
// The first two components is the major axis, last two components is the minor axis.
// If the returned vector is (0, 0, 0, 0), it means the Gaussian is not visible.
fn cov2d_axes(gaussian: Gaussian, model_transform: ModelTransform) -> vec4<f32> {
    let cov2d = cov2d(gaussian, model_transform);
    let mid = 0.5 * (cov2d.x + cov2d.z);
    let radius = length(vec2<f32>(0.5 * (cov2d.x - cov2d.z), cov2d.y));
    let lambda_1 = mid + radius;
    let lambda_2 = mid - radius;

    if lambda_2 < 0.0 {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let diag_dir = normalize(vec2<f32>(cov2d.y, lambda_1 - cov2d.x));
    let ortho_diag_dir = vec2<f32>(diag_dir.y, -diag_dir.x);
    let major_len = min(max_radius * sqrt(lambda_1), 1024.0);
    let minor_len = min(max_radius * sqrt(lambda_2), 1024.0);
    let major_axis = major_len * diag_dir * gaussian_transform.size;
    let minor_axis = minor_len * ortho_diag_dir * gaussian_transform.size;

    return vec4<f32>(major_axis, minor_axis);
}

// Calculate the color of a Gaussian based on its SH coefficients and direction.
fn color(gaussian: Gaussian, dir: vec3<f32>, sh_deg: u32, no_sh0: bool) -> vec4<f32> {
    const sh_c1 = 0.4886025;
    const sh_c2 = array<f32, 5>(1.0925484, -1.0925484, 0.3153916, -1.0925484, 0.5462742);
    const sh_c3 = array<f32, 7>(
        -0.5900436, 2.8906114, -0.4570458, 0.3731763, -0.4570458, 1.4453057, -0.5900436
    );

    let x = dir.x;
    let y = dir.y;
    let z = dir.z;

    let color = unpack4x8unorm(gaussian.color);
    var result = color.rgb; // 0.5 + SH_C0 * sh[0] already precomputed

    if no_sh0 {
        result = vec3<f32>(0.5);
    }

    if sh_deg >= 1u {
        result += sh_c1 * (
            -gaussian_unpack_sh(gaussian, 1u) * y +
            gaussian_unpack_sh(gaussian, 2u) * z -
            gaussian_unpack_sh(gaussian, 3u) * x
        );

        if sh_deg >= 2u {
            let xx = x * x;
            let yy = y * y;
            let zz = z * z;
            let xy = x * y;
            let yz = y * z;
            let xz = x * z;

            result += 
                sh_c2[0] * xy * gaussian_unpack_sh(gaussian, 4u) +
                sh_c2[1] * yz * gaussian_unpack_sh(gaussian, 5u) +
                sh_c2[2] * (2.0 * zz - xx - yy) * gaussian_unpack_sh(gaussian, 6u) +
                sh_c2[3] * xz * gaussian_unpack_sh(gaussian, 7u) +
                sh_c2[4] * (xx - yy) * gaussian_unpack_sh(gaussian, 8u);

            if sh_deg >= 3u {
                result += 
                    sh_c3[0] * y * (3.0 * xx - yy) * gaussian_unpack_sh(gaussian, 9u) +
                    sh_c3[1] * xy * z * gaussian_unpack_sh(gaussian, 10u) +
                    sh_c3[2] * y * (4.0 * zz - xx - yy) * gaussian_unpack_sh(gaussian, 11u) +
                    sh_c3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * gaussian_unpack_sh(gaussian, 12u) +
                    sh_c3[4] * x * (4.0 * zz - xx - yy) * gaussian_unpack_sh(gaussian, 13u) +
                    sh_c3[5] * z * (xx - yy) * gaussian_unpack_sh(gaussian, 14u) +
                    sh_c3[6] * x * (xx - 3.0 * yy) * gaussian_unpack_sh(gaussian, 15u);
            }
        }
    }

    return vec4<f32>(max(result, vec3<f32>(0.0)), color.a);
}