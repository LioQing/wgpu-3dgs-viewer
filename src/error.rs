use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("{0}")]
    Io(#[from] std::io::Error),
    #[error("vertex count not found in PLY header")]
    PlyVertexCountNotFound,
    #[error("failed to parse vertex count")]
    PlyVertexCountParseFailed(#[from] std::num::ParseIntError),
    #[error("not a PLY file")]
    NotPly,
    #[error("PLY header not found")]
    PlyHeaderNotFound,
}
