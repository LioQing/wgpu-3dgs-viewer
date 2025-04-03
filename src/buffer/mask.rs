use glam::*;
use wgpu::util::DeviceExt;

use crate::Error;

/// The mask storage buffer for storing masked Gaussians as a bitvec.
#[derive(Debug, Clone)]
pub struct MaskBuffer {
    data: wgpu::Buffer,
    download: wgpu::Buffer,
}

impl MaskBuffer {
    /// Create a new mask buffer.
    pub fn new(device: &wgpu::Device, gaussian_count: u32) -> Self {
        Self::new_with_label(device, "", gaussian_count)
    }

    /// Create a new mask buffer with additional label.
    pub fn new_with_label(device: &wgpu::Device, label: &str, gaussian_count: u32) -> Self {
        let size = gaussian_count.div_ceil(32) * std::mem::size_of::<u32>() as u32;

        let data = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(format!("Mask {label} Buffer").as_str()),
            size: size as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let download = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gaussians Edit Download Buffer"),
            size: size as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self { data, download }
    }

    /// Download the mask edit.
    pub async fn download(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<u32>, Error> {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Mask Download Encoder"),
        });
        self.prepare_download(&mut encoder);
        queue.submit(Some(encoder.finish()));

        self.map_download(device).await
    }

    /// Prepare for downloading the Gaussian edit.
    pub fn prepare_download(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.copy_buffer_to_buffer(self.buffer(), 0, &self.download, 0, self.download.size());
    }

    /// Map the download buffer to read the Gaussian edit.
    pub async fn map_download(&self, device: &wgpu::Device) -> Result<Vec<u32>, Error> {
        let (tx, rx) = oneshot::channel();
        let buffer_slice = self.download.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            if let Err(e) = tx.send(result) {
                log::error!("Error occurred while sending Gaussian edit: {e:?}");
            }
        });
        device.poll(wgpu::Maintain::Wait);
        rx.await??;

        let edits = bytemuck::allocation::pod_collect_to_vec(&buffer_slice.get_mapped_range());
        self.download.unmap();

        Ok(edits)
    }

    /// Get the buffer.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.data
    }

    /// Get the download buffer.
    pub fn download_buffer(&self) -> &wgpu::Buffer {
        &self.download
    }
}

/// The mask shape.
///
/// This is an abstraction over the [`MaskOpShapePod`] and [`MaskGizmoPod`].
#[cfg(feature = "mask-evaluator")]
#[derive(Debug, Clone)]
pub struct MaskShape {
    /// Kind.
    pub kind: MaskShapeKind,
    /// Position.
    pub pos: Vec3,
    /// Rotation.
    pub rotation: Quat,
    /// Scale.
    pub scale: Vec3,
    /// Color.
    #[cfg(feature = "mask-gizmo")]
    pub color: Vec4,
}

#[cfg(feature = "mask-evaluator")]
impl MaskShape {
    /// Create a new mask shape.
    pub fn new(kind: MaskShapeKind) -> Self {
        Self {
            kind,
            pos: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
            #[cfg(feature = "mask-gizmo")]
            color: Vec4::ONE,
        }
    }

    /// Convert to [`MaskOpShapePod`].
    pub fn to_mask_op_shape_pod(&self) -> MaskOpShapePod {
        match self.kind {
            MaskShapeKind::Box => MaskOpShapePod::box_shape(self.pos, self.rotation, self.scale),
            MaskShapeKind::Ellipsoid => {
                MaskOpShapePod::ellipsoid_shape(self.pos, self.rotation, self.scale)
            }
        }
    }

    /// Convert to [`MaskGizmoPod`].
    #[cfg(feature = "mask-gizmo")]
    pub fn to_mask_gizmo_pod(&self) -> MaskGizmoPod {
        MaskGizmoPod::new(self.color, self.pos, self.rotation, self.scale)
    }
}

/// The mask shape uniform buffer for storing mask operation shape.
#[cfg(feature = "mask-evaluator")]
#[derive(Debug, Clone)]
pub struct MaskOpShapeBuffer(wgpu::Buffer);

#[cfg(feature = "mask-evaluator")]
impl MaskOpShapeBuffer {
    /// Create a new mask shape buffer.
    pub fn new(device: &wgpu::Device, mask_shape: &MaskOpShapePod) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Mask Shape Buffer"),
            contents: bytemuck::bytes_of(mask_shape),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Self(buffer)
    }

    /// Update the mask shapes buffer.
    pub fn update(&self, queue: &wgpu::Queue, mask_shapes: &MaskOpShapePod) {
        queue.write_buffer(&self.0, 0, bytemuck::bytes_of(mask_shapes));
    }

    /// Get the buffer.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.0
    }
}

/// The mask shape kinds.
#[cfg(feature = "mask-evaluator")]
#[repr(u16)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MaskShapeKind {
    /// The data is \[x, y, z\].
    Box = 0,

    /// The data is \[rx, ry, rz\].
    Ellipsoid = 1,
}

/// The POD representation of a mask operation shape for evaluation.
#[cfg(feature = "mask-evaluator")]
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MaskOpShapePod {
    /// The mask shape kind.
    pub kind: u32,

    /// The padding.
    _padding: [u32; 3],

    /// The inverse transformation matrix.
    pub inv_transform: Mat4,
}

#[cfg(feature = "mask-evaluator")]
impl MaskOpShapePod {
    /// Create a new mask shape.
    pub const fn new(kind: MaskShapeKind, inv_transform: Mat4) -> Self {
        Self {
            kind: kind as u32,
            _padding: [0; 3],
            inv_transform,
        }
    }

    /// Create a new box mask shape with transform.
    pub fn box_shape_with_transform(transform: Mat4) -> Self {
        Self::new(MaskShapeKind::Box, transform.inverse())
    }

    /// Create a new ellipsoid mask shape.
    pub fn box_shape(pos: Vec3, rotation: Quat, scale: Vec3) -> Self {
        Self::box_shape_with_transform(Mat4::from_scale_rotation_translation(scale, rotation, pos))
    }

    /// Create a new ellipsoid mask shape with transform.
    pub fn ellipsoid_shape_with_transform(transform: Mat4) -> Self {
        Self::new(MaskShapeKind::Ellipsoid, transform.inverse())
    }

    /// Create a new ellipsoid mask shape.
    pub fn ellipsoid_shape(pos: Vec3, rotation: Quat, scale: Vec3) -> Self {
        Self::ellipsoid_shape_with_transform(Mat4::from_scale_rotation_translation(
            scale, rotation, pos,
        ))
    }
}

/// The mask operation uniform buffer for storing mask operations.
#[cfg(feature = "mask-evaluator")]
#[derive(Debug, Clone)]
pub struct MaskOpBuffer(wgpu::Buffer);

#[cfg(feature = "mask-evaluator")]
impl MaskOpBuffer {
    /// Create a new mask operation buffer.
    pub fn new(device: &wgpu::Device, mask_op: MaskOp) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Mask Operation Buffer"),
            contents: bytemuck::bytes_of(&(mask_op as u32)),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Self(buffer)
    }

    /// Update the mask operation buffer.
    pub fn update(&self, queue: &wgpu::Queue, mask_op: MaskOp) {
        queue.write_buffer(&self.0, 0, bytemuck::bytes_of(&(mask_op as u32)));
    }

    /// Get the buffer.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.0
    }
}

/// The mask operation.
#[cfg(feature = "mask-evaluator")]
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskOp {
    /// The union operation.
    Union = 0,

    /// The intersection operation.
    Intersection = 1,

    /// The symmetric difference operation.
    SymmetricDifference = 2,

    /// The difference operation.
    Difference = 3,

    /// The complement operation.
    Complement = 4,

    /// The shape operation.
    Shape = 5,

    /// The reset operation.
    Reset = 6,
}

/// The POD representation of a mask gizmo.
#[cfg(feature = "mask-gizmo")]
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MaskGizmoPod {
    /// The color.
    pub color: Vec4,

    /// The transformation matrix.
    pub transform: Mat4,
}

#[cfg(feature = "mask-gizmo")]
impl MaskGizmoPod {
    /// Create a new mask gizmo.
    pub fn new(color: Vec4, pos: Vec3, rotation: Quat, scale: Vec3) -> Self {
        Self::new_with_transform(
            color,
            Mat4::from_scale_rotation_translation(scale, rotation, pos),
        )
    }

    /// Create a new mask gizmo with transform.
    pub const fn new_with_transform(color: Vec4, transform: Mat4) -> Self {
        Self { color, transform }
    }
}

/// The mask gizmos storage buffer for storing mask gizmos.
#[cfg(feature = "mask-gizmo")]
#[derive(Debug, Clone)]
pub struct MaskGizmosBuffer(wgpu::Buffer);

#[cfg(feature = "mask-gizmo")]
impl MaskGizmosBuffer {
    /// Create a new mask gizmos buffer.
    pub fn new(device: &wgpu::Device, gizmos: &[MaskGizmoPod]) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Mask Gizmos Buffer"),
            contents: bytemuck::cast_slice(gizmos),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        Self(buffer)
    }

    /// Create a new mask gizmos buffer with the specified size.
    pub fn new_empty(device: &wgpu::Device, len: usize) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Mask Gizmos Buffer"),
            size: (len * std::mem::size_of::<MaskGizmoPod>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self(buffer)
    }

    /// Get the buffer.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.0
    }

    /// Get the number of Gaussians.
    pub fn len(&self) -> usize {
        self.0.size() as usize / std::mem::size_of::<MaskGizmoPod>()
    }

    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Update the mask gizmos buffer.
    pub fn update(&self, queue: &wgpu::Queue, gizmos: &[MaskGizmoPod]) {
        if gizmos.len() != self.len() {
            log::error!(
                "Mask gizmo count mismatch, buffer has {}, but {} were provided",
                self.len(),
                gizmos.len()
            );
            return;
        }

        queue.write_buffer(&self.0, 0, bytemuck::cast_slice(gizmos));
    }

    /// Update a range of the buffer.
    pub fn update_range(&self, queue: &wgpu::Queue, start: usize, gizmos: &[MaskGizmoPod]) {
        if start + gizmos.len() > self.len() {
            log::error!(
                "Mask gizmo count mismatch, buffer has {}, but {} were provided starting at {}",
                self.len(),
                gizmos.len(),
                start
            );
            return;
        }

        queue.write_buffer(
            &self.0,
            (start * std::mem::size_of::<MaskGizmoPod>()) as wgpu::BufferAddress,
            bytemuck::cast_slice(gizmos),
        );
    }
}
