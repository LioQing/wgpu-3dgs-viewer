struct GaussianTransform {
    size: f32,
    flags: u32,
}

const gaussian_display_mode_splat = 0u;
const gaussian_display_mode_ellipse = 1u;
const gaussian_display_mode_point = 2u;

fn gaussian_transform_display_mode(flags: u32) -> u32 {
    return unpack4xU8(flags).x;
}

fn gaussian_transform_sh_deg(flags: u32) -> u32 {
    return unpack4xU8(flags).y;
}

fn gaussian_transform_no_sh0(flags: u32) -> bool {
    return unpack4xU8(flags).z != 0u;
}