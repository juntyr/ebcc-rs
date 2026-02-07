//! Error types for EBCC operations.

use thiserror::Error;

/// Result type for EBCC operations.
pub type EBCCResult<T> = Result<T, EBCCError>;

/// Errors that can occur during EBCC compression/decompression.
#[derive(Error, Debug)]
pub enum EBCCError {
    #[error("Invalid input data: {0}")]
    /// Invalid input data
    InvalidInput(String),

    #[error("Invalid configuration: {0}")]
    /// Invalid configuration
    InvalidConfig(String),

    #[error("Compression failed: {0}")]
    /// Compression failed
    CompressionError(String),

    #[error("Decompression failed: {0}")]
    /// Decompression failed
    DecompressionError(String),
}
