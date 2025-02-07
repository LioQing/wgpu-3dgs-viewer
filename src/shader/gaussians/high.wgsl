sh: array<f32, (3 * 15)>,
cov3d: array<f32, 6>,

fn gaussian_unpack_sh(gaussian_index: u32, sh_index: u32) -> vec3<f32> {
    return vec3<f32>(
        gaussians[gaussian_index].sh[(sh_index - 1) * 3],
        gaussians[gaussian_index].sh[(sh_index - 1) * 3 + 1],
        gaussians[gaussian_index].sh[(sh_index - 1) * 3 + 2],
    );
}

fn gaussian_unpack_cov3d(gaussian_index: u32) -> array<f32, 6> {
    return gaussians[gaussian_index].cov3d;
}