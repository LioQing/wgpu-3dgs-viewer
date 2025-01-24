use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Vertex count not found in PLY header")]
    PlyVertexCountNotFound,
    #[error("Failed to parse vertex count")]
    PlyVertexCountParseFailed(#[from] std::num::ParseIntError),
}