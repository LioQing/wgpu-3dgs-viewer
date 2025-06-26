use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("{0}")]
    Io(#[from] std::io::Error),
    #[error("vertex not found in PLY")]
    PlyVertexNotFound,
    #[error("vertex property {0} not found in PLY")]
    PlyVertexPropertyNotFound(String),
    #[error("{0}")]
    BufferDownloadOneShotReceive(#[from] oneshot::RecvError),
    #[error("{0}")]
    BufferDownloadAsync(#[from] wgpu::BufferAsyncError),
    #[error(
        "\
        model size ({model_size} bytes) exceeds the device limit ({device_limit} bytes), \
        try smaller model or more aggressive compression\
        "
    )]
    ModelSizeExceedsDeviceLimit { model_size: u64, device_limit: u32 },
    #[error("model count and keys length mismatch: {model_count} != {keys_len}")]
    ModelCountKeysLenMismatch { model_count: usize, keys_len: usize },
    #[error("{0}")]
    DeviceFailedToPoll(#[from] wgpu::PollError),
}
