//! Streaming support for incremental gaussian loading.
//!
//! Streaming allows loading gaussians progressively into a pre-allocated GPU
//! buffer. Only the loaded portion is dispatched to the preprocessor, so
//! gaussians appear incrementally as they are uploaded.
//!
//! # Usage
//!
//! ```rust,ignore
//! use wgpu_3dgs_viewer::streaming::StreamingConfig;
//!
//! let mut streaming = StreamingConfig::new(total_gaussians);
//! streaming.enabled = true;
//!
//! // Each frame, upload a chunk and advance loaded_count:
//! streaming.loaded_count = uploaded_so_far;
//! ```

/// Configuration for streaming gaussian data.
///
/// When enabled, only `loaded_count` gaussians are dispatched to the
/// preprocessor. Increase `loaded_count` as more data is uploaded to the
/// GPU buffer via `queue.write_buffer()` at the appropriate offset.
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Whether streaming mode is enabled.
    /// When `false`, all gaussians in the buffer are rendered.
    pub enabled: bool,
    /// Total number of gaussians that will eventually be loaded.
    pub total_count: u32,
    /// Number of gaussians currently loaded in the GPU buffer.
    pub loaded_count: u32,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            total_count: 0,
            loaded_count: 0,
        }
    }
}

impl StreamingConfig {
    /// Create a new streaming config for the given total gaussian count.
    pub fn new(total_count: u32) -> Self {
        Self {
            enabled: true,
            total_count,
            loaded_count: 0,
        }
    }

    /// Return the effective gaussian count based on streaming state.
    pub fn effective_count(&self, total: u32) -> u32 {
        if self.enabled {
            total.min(self.loaded_count)
        } else {
            total
        }
    }

    /// Returns the fraction of gaussians loaded (0.0 to 1.0).
    pub fn progress(&self) -> f32 {
        if self.total_count == 0 {
            1.0
        } else {
            self.loaded_count as f32 / self.total_count as f32
        }
    }

    /// Returns `true` if all gaussians have been loaded.
    pub fn is_complete(&self) -> bool {
        self.loaded_count >= self.total_count
    }
}
