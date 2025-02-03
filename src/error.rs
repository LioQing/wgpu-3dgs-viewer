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
}
