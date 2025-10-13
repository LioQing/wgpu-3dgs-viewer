use wgpu::util::DeviceExt;
use wgpu_3dgs_viewer::{
    IndirectArgsBuffer, IndirectIndicesBuffer, RadixSortIndirectArgsBuffer,
    core::{BufferWrapper, DownloadableBufferWrapper},
};

use crate::common::TestContext;

#[test]
fn test_indirect_args_buffer_new_should_return_correct_buffer() {
    let ctx = TestContext::new();
    let buffer = IndirectArgsBuffer::new(&ctx.device);

    assert_eq!(
        buffer.buffer().size(),
        std::mem::size_of::<wgpu::util::DrawIndirectArgs>() as wgpu::BufferAddress
    );
}

#[test]
fn test_indirect_args_buffer_try_from_and_into_wgpu_buffer_should_be_equal() {
    let ctx = TestContext::new();
    let draw_args = wgpu::util::DrawIndirectArgs {
        vertex_count: 6,
        instance_count: 100,
        first_vertex: 0,
        first_instance: 0,
    };
    let wgpu_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Test Indirect Args Buffer"),
            contents: draw_args.as_bytes(),
            usage: IndirectArgsBuffer::DEFAULT_USAGES | wgpu::BufferUsages::COPY_SRC,
        });

    let converted_buffer = IndirectArgsBuffer::try_from(wgpu_buffer.clone()).expect("try_from");
    let wgpu_converted_buffer = wgpu::Buffer::from(converted_buffer.clone());

    let wgpu_downloaded = pollster::block_on(
        wgpu_converted_buffer.download::<wgpu::util::DrawIndirectArgs>(&ctx.device, &ctx.queue),
    )
    .expect("download")[0];
    let converted_downloaded = pollster::block_on(
        converted_buffer.download::<wgpu::util::DrawIndirectArgs>(&ctx.device, &ctx.queue),
    )
    .expect("download")[0];
    let wgpu_converted_downloaded = pollster::block_on(
        wgpu_buffer.download::<wgpu::util::DrawIndirectArgs>(&ctx.device, &ctx.queue),
    )
    .expect("download")[0];

    assert_eq!(
        wgpu_downloaded.vertex_count,
        converted_downloaded.vertex_count
    );
    assert_eq!(
        wgpu_downloaded.instance_count,
        converted_downloaded.instance_count
    );
    assert_eq!(
        wgpu_downloaded.first_vertex,
        converted_downloaded.first_vertex
    );
    assert_eq!(
        wgpu_downloaded.first_instance,
        converted_downloaded.first_instance
    );
    assert_eq!(
        wgpu_downloaded.vertex_count,
        wgpu_converted_downloaded.vertex_count
    );
    assert_eq!(
        wgpu_downloaded.instance_count,
        wgpu_converted_downloaded.instance_count
    );
    assert_eq!(
        wgpu_downloaded.first_vertex,
        wgpu_converted_downloaded.first_vertex
    );
    assert_eq!(
        wgpu_downloaded.first_instance,
        wgpu_converted_downloaded.first_instance
    );
}

#[test]
fn test_radix_sort_indirect_args_buffer_new_should_return_correct_buffer() {
    let ctx = TestContext::new();
    let buffer = RadixSortIndirectArgsBuffer::new(&ctx.device);

    assert_eq!(
        buffer.buffer().size(),
        std::mem::size_of::<wgpu::util::DispatchIndirectArgs>() as wgpu::BufferAddress
    );
}

#[test]
fn test_radix_sort_indirect_args_buffer_try_from_and_into_wgpu_buffer_should_be_equal() {
    let ctx = TestContext::new();
    let dispatch_args = wgpu::util::DispatchIndirectArgs { x: 8, y: 4, z: 2 };
    let wgpu_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Test Radix Sort Indirect Args Buffer"),
            contents: dispatch_args.as_bytes(),
            usage: RadixSortIndirectArgsBuffer::DEFAULT_USAGES | wgpu::BufferUsages::COPY_SRC,
        });

    let converted_buffer =
        RadixSortIndirectArgsBuffer::try_from(wgpu_buffer.clone()).expect("try_from");
    let wgpu_converted_buffer = wgpu::Buffer::from(converted_buffer.clone());

    let wgpu_downloaded = pollster::block_on(
        wgpu_converted_buffer.download::<wgpu::util::DispatchIndirectArgs>(&ctx.device, &ctx.queue),
    )
    .expect("download")[0];
    let converted_downloaded = pollster::block_on(
        converted_buffer.download::<wgpu::util::DispatchIndirectArgs>(&ctx.device, &ctx.queue),
    )
    .expect("download")[0];
    let wgpu_converted_downloaded = pollster::block_on(
        wgpu_buffer.download::<wgpu::util::DispatchIndirectArgs>(&ctx.device, &ctx.queue),
    )
    .expect("download")[0];

    assert_eq!(wgpu_downloaded.x, converted_downloaded.x);
    assert_eq!(wgpu_downloaded.y, converted_downloaded.y);
    assert_eq!(wgpu_downloaded.z, converted_downloaded.z);
    assert_eq!(wgpu_downloaded.x, wgpu_converted_downloaded.x);
    assert_eq!(wgpu_downloaded.y, wgpu_converted_downloaded.y);
    assert_eq!(wgpu_downloaded.z, wgpu_converted_downloaded.z);
}

#[test]
fn test_indirect_indices_buffer_new_should_return_correct_buffer() {
    let ctx = TestContext::new();
    let gaussian_count = 256;
    let buffer = IndirectIndicesBuffer::new(&ctx.device, gaussian_count);

    let expected_size = (gaussian_count * std::mem::size_of::<u32>() as u32) as wgpu::BufferAddress;
    assert_eq!(buffer.buffer().size(), expected_size);
}

#[test]
fn test_indirect_indices_buffer_new_with_different_counts_should_return_correct_buffer() {
    let ctx = TestContext::new();

    for gaussian_count in [1, 64, 128, 512, 1024] {
        let buffer = IndirectIndicesBuffer::new(&ctx.device, gaussian_count);
        let expected_size =
            (gaussian_count * std::mem::size_of::<u32>() as u32) as wgpu::BufferAddress;
        assert_eq!(buffer.buffer().size(), expected_size);
    }
}

#[test]
fn test_indirect_indices_buffer_try_from_and_into_wgpu_buffer_should_be_equal() {
    let ctx = TestContext::new();
    let indices: [u32; 10] = std::array::from_fn(|i| i as u32);
    let wgpu_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Test Indirect Indices Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: IndirectIndicesBuffer::DEFAULT_USAGES | wgpu::BufferUsages::COPY_SRC,
        });

    let converted_buffer = IndirectIndicesBuffer::from(wgpu_buffer.clone());
    let wgpu_converted_buffer = wgpu::Buffer::from(converted_buffer.clone());

    let wgpu_downloaded =
        pollster::block_on(wgpu_converted_buffer.download::<u32>(&ctx.device, &ctx.queue))
            .expect("download");
    let converted_downloaded =
        pollster::block_on(converted_buffer.download::<u32>(&ctx.device, &ctx.queue))
            .expect("download");
    let wgpu_converted_downloaded =
        pollster::block_on(wgpu_buffer.download::<u32>(&ctx.device, &ctx.queue)).expect("download");

    assert_eq!(wgpu_downloaded, converted_downloaded);
    assert_eq!(wgpu_downloaded, wgpu_converted_downloaded);
}
