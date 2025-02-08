// Spherical harmonics configurations

// sh field - single
sh: array<f32, (3 * 15)>,
// sh field - single

// sh unpack - single
fn gaussian_unpack_sh(gaussian_index: u32, sh_index: u32) -> vec3<f32> {
    return vec3<f32>(
        gaussians[gaussian_index].sh[(sh_index - 1) * 3],
        gaussians[gaussian_index].sh[(sh_index - 1) * 3 + 1],
        gaussians[gaussian_index].sh[(sh_index - 1) * 3 + 2],
    );
}
// sh unpack - single

// sh field - half
sh: array<u32, ((3 * 15 + 1) / 2)>,
// sh field - half

// sh unpack - half
fn gaussian_unpack_sh(gaussian_index: u32, sh_index: u32) -> vec3<f32> {
    let i = (sh_index - 1) * 3;
    let xi = i / 2;
    let yi = (i + 1) / 2;
    let zi = (i + 2) / 2;
    
    if xi == yi {
        return vec3<f32>(
            unpack2x16float(gaussians[gaussian_index].sh[xi]),
            unpack2x16float(gaussians[gaussian_index].sh[zi]).x,
        );
    } else {
        return vec3<f32>(
            unpack2x16float(gaussians[gaussian_index].sh[xi]).y,
            unpack2x16float(gaussians[gaussian_index].sh[yi]),
        );
    }
}
// sh unpack - half

// sh field - min max norm
sh: array<u32, (1 + (3 * 15 + 3) / 4)>,
// sh field - min max norm

// sh unpack - min max norm
fn gaussian_unpack_sh(gaussian_index: u32, sh_index: u32) -> vec3<f32> {
    let minmax = unpack2x16float(gaussians[gaussian_index].sh[0]);

    let i = (sh_index - 1) * 3;
    let xi = i / 4;
    let xj = i % 4;
    let yi = (i + 1) / 4;
    let yj = (i + 1) % 4;
    let zi = (i + 2) / 4;
    let zj = (i + 2) % 4;
    
    let norm = vec3<f32>(
        unpack4x8unorm(gaussians[gaussian_index].sh[1 + xi])[xj],
        unpack4x8unorm(gaussians[gaussian_index].sh[1 + yi])[yj],
        unpack4x8unorm(gaussians[gaussian_index].sh[1 + zi])[zj],
    );

    return minmax.x + norm * (minmax.y - minmax.x);
}
// sh unpack - min max norm

// sh field - none
// sh field - none

// sh unpack - none
fn gaussian_unpack_sh(gaussian_index: u32, sh_index: u32) -> vec3<f32> {
    return vec3<f32>(0.0);
}
// sh unpack - none

// Covariance 3D configurations

// cov3d field - single
cov3d: array<f32, 6>,
// cov3d field - single

// cov3d unpack - single
fn gaussian_unpack_cov3d(gaussian_index: u32) -> array<f32, 6> {
    return gaussians[gaussian_index].cov3d;
}
// cov3d unpack - single

// cov3d field - half
cov3d: array<u32, 3>,
// cov3d field - half

// cov3d unpack - half
fn gaussian_unpack_cov3d(gaussian_index: u32) -> array<f32, 6> {
    let x = unpack2x16float(gaussians[gaussian_index].cov3d[0]);
    let y = unpack2x16float(gaussians[gaussian_index].cov3d[1]);
    let z = unpack2x16float(gaussians[gaussian_index].cov3d[2]);
    return array<f32, 6>(
        x.x,
        x.y,
        y.x,
        y.y,
        z.x,
        z.y,
    );
}
// cov3d unpack - half