use thiserror::Error;

use crate::core;

#[derive(Debug, Error)]
pub enum Error {
    #[error("{0}")]
    Core(#[from] core::Error),
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
    WeslCompile(Box<wesl::Error>),
}

impl From<wesl::Error> for Error {
    fn from(err: wesl::Error) -> Self {
        Error::WeslCompile(Box::new(err))
    }
}
