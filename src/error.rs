use thiserror::Error;

use crate::core;

// #[derive(Debug, Error)]
// pub enum Error {
//     #[error("{0}")]
//     Core(Box<core::Error>),
//     #[error(
//         "\
//         model size ({model_size} bytes) exceeds the device limit ({device_limit} bytes), \
//         try smaller model or more aggressive compression\
//         "
//     )]
//     ModelSizeExceedsDeviceLimit { model_size: u64, device_limit: u32 },
//     #[error("model count and keys length mismatch: {model_count} != {keys_len}")]
//     ModelCountKeysLenMismatch { model_count: usize, keys_len: usize },
//     #[error("{0}")]
//     WeslCompile(Box<wesl::Error>),
// }

// impl From<core::Error> for Error {
//     fn from(err: core::Error) -> Self {
//         Error::Core(Box::new(err))
//     }
// }

// impl From<wesl::Error> for Error {
//     fn from(err: wesl::Error) -> Self {
//         Error::WeslCompile(Box::new(err))
//     }
// }

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

/// The error type for [`MultiModelViewer::render`](crate::MultiModelViewer::render).
#[cfg(feature = "multi-model")]
#[derive(Debug, Error)]
pub enum MultiModelViewerRenderError {
    #[error("model and key count mismatch: {model_count} != {key_count}")]
    ModelKeyCountMismatch {
        model_count: usize,
        key_count: usize,
    },
}
