use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("{0}")]
    Io(#[from] std::io::Error),
    #[error("vertex count not found in PLY header")]
    PlyVertexCountNotFound,
    #[error("{0}")]
    PlyVertexCountParseFailed(#[from] std::num::ParseIntError),
    #[error("not a PLY file")]
    NotPly,
    #[error("PLY header not found")]
    PlyHeaderNotFound,
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
    #[error(
        "model count and render metadata length mismatch: {model_count} != {render_metadata_len}"
    )]
    ModelCountRenderMetadataLenMismatch {
        model_count: usize,
        render_metadata_len: usize,
    },

    #[cfg(feature = "query-texture-tool")]
    #[error("query texture tool already in use")]
    QueryTextureToolAlreadyInUse,

    #[cfg(feature = "query-texture-tool")]
    #[error("query texture tool not in use")]
    QueryTextureToolNotInUse,

    #[cfg(feature = "query-tool")]
    #[error("query tool already in use")]
    QueryToolAlreadyInUse,

    #[cfg(feature = "query-tool")]
    #[error("query tool not in use")]
    QueryToolNotInUse,
}
