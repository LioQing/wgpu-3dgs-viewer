use thiserror::Error;

use crate::core;

/// The error type for [`Renderer::new`](crate::Renderer::new).
#[derive(Debug, Error)]
pub enum RendererCreateError {
    #[error(
        "\
        model size exceeds the device limit: {model_size} > {device_limit}, \
        try smaller model or more aggressive compression\
        "
    )]
    ModelSizeExceedsDeviceLimit { model_size: u64, device_limit: u32 },
    #[error("{0}")]
    WeslCompile(#[from] wesl::Error),
}

/// The error type for [`Preprocessor::new`](crate::Preprocessor::new).
#[derive(Debug, Error)]
pub enum PreprocessorCreateError {
    #[error(
        "\
        model size exceeds the device limit: {model_size} > {device_limit}, \
        try smaller model or more aggressive compression\
        "
    )]
    ModelSizeExceedsDeviceLimit { model_size: u64, device_limit: u32 },
    #[error("{0}")]
    ComputeBundleBuild(#[from] core::ComputeBundleBuildError),
    #[error("{0}")]
    WeslCompile(#[from] wesl::Error),
}

/// The error type for [`Viewer::new`](crate::Viewer::new).
#[derive(Debug, Error)]
pub enum ViewerCreateError {
    #[error("{0}")]
    RendererCreate(#[from] RendererCreateError),
    #[error("{0}")]
    PreprocessorCreate(#[from] PreprocessorCreateError),
}

/// The error type for accessing model in [`MultiModelViewer`](crate::MultiModelViewer).
#[cfg(feature = "multi-model")]
#[derive(Debug, Error)]
pub enum MultiModelViewerAccessError {
    #[error("model with the given key does not exist")]
    ModelNotFound,
}
