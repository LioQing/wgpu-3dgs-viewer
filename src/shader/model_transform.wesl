struct ModelTransform {
    pos: vec3<f32>,
    quat: vec4<f32>,
    scale: vec3<f32>,
}

fn model_transform_mat(model_transform: ModelTransform) -> mat4x4<f32> {
    let pos = model_transform.pos;
    let quat = model_transform.quat;
    let scale = model_transform.scale;

    let x2 = quat.x + quat.x;
    let y2 = quat.y + quat.y;
    let z2 = quat.z + quat.z;
    let xx = quat.x * x2;
    let xy = quat.x * y2;
    let xz = quat.x * z2;
    let yy = quat.y * y2;
    let yz = quat.y * z2;
    let zz = quat.z * z2;
    let wx = quat.w * x2;
    let wy = quat.w * y2;
    let wz = quat.w * z2;

    let sx = scale.x;
    let sy = scale.y;
    let sz = scale.z;

    return mat4x4<f32>(
        vec4<f32>(
            (1.0 - (yy + zz)) * sx,
            (xy + wz) * sx,
            (xz - wy) * sx,
            0.0,
        ),
        vec4<f32>(
            (xy - wz) * sy,
            (1.0 - (xx + zz)) * sy,
            (yz + wx) * sy,
            0.0,
        ),
        vec4<f32>(
            (xz + wy) * sz,
            (yz - wx) * sz,
            (1.0 - (xx + yy)) * sz,
            0.0,
        ),
        vec4<f32>(pos, 1.0),
    );
}

fn model_transform_inv_sr_mat(model_transform: ModelTransform) -> mat3x3<f32> {
    let quat = model_transform.quat;
    let scale = model_transform.scale;

    let x2 = quat.x + quat.x;
    let y2 = quat.y + quat.y;
    let z2 = quat.z + quat.z;
    let xx = quat.x * x2;
    let xy = quat.x * y2;
    let xz = quat.x * z2;
    let yy = quat.y * y2;
    let yz = quat.y * z2;
    let zz = quat.z * z2;
    let wx = quat.w * x2;
    let wy = quat.w * y2;
    let wz = quat.w * z2;

    let sx = scale.x;
    let sy = scale.y;
    let sz = scale.z;

    return mat3x3<f32>(
        vec3<f32>(
            (1.0 - (yy + zz)) / sx,
            (xy - wz) / sy,
            (xz + wy) / sz,
        ),
        vec3<f32>(
            (xy + wz) / sx,
            (1.0 - (xx + zz)) / sy,
            (yz - wx) / sz,
        ),
        vec3<f32>(
            (xz - wy) / sx,
            (yz + wx) / sy,
            (1.0 - (xx + yy)) / sz,
        ),
    );
}

fn model_scale_rotation_mat(model_transform: ModelTransform) -> mat3x3<f32> {
    let quat = model_transform.quat;
    let scale = model_transform.scale;

    let x2 = quat.x + quat.x;
    let y2 = quat.y + quat.y;
    let z2 = quat.z + quat.z;
    let xx = quat.x * x2;
    let xy = quat.x * y2;
    let xz = quat.x * z2;
    let yy = quat.y * y2;
    let yz = quat.y * z2;
    let zz = quat.z * z2;
    let wx = quat.w * x2;
    let wy = quat.w * y2;
    let wz = quat.w * z2;

    let sx = scale.x;
    let sy = scale.y;
    let sz = scale.z;

    return mat3x3<f32>(
        vec3<f32>(
            (1.0 - (yy + zz)) * sx,
            (xy + wz) * sx,
            (xz - wy) * sx,
        ),
        vec3<f32>(
            (xy - wz) * sy,
            (1.0 - (xx + zz)) * sy,
            (yz + wx) * sy,
        ),
        vec3<f32>(
            (xz + wy) * sz,
            (yz - wx) * sz,
            (1.0 - (xx + yy)) * sz,
        ),
    );
}