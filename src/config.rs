//! Configuration types for EBCC compression.

use crate::error::{EBCCError, EBCCResult};

/// Residual compression types supported by EBCC.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EBCCResidualType {
    /// No residual compression - base JPEG2000 only
    Jpeg2000Only,
    /// Residual compression with absolute maximum error bound
    AbsoluteError(f32),
    /// Residual compression with relative error bound
    RelativeError(f32),
}

impl EBCCResidualType {
    pub(crate) const fn as_residual(self) -> ebcc_sys::residual_t::Type {
        match self {
            Self::Jpeg2000Only => ebcc_sys::residual_t::NONE,
            Self::AbsoluteError(_) => ebcc_sys::residual_t::MAX_ERROR,
            Self::RelativeError(_) => ebcc_sys::residual_t::RELATIVE_ERROR,
        }
    }

    pub(crate) const fn as_error(self) -> f32 {
        match self {
            Self::Jpeg2000Only => 0.0,
            Self::AbsoluteError(error) | Self::RelativeError(error) => error,
        }
    }
}

/// Configuration for EBCC compression.
#[derive(Debug, Clone, PartialEq)]
pub struct EBCCConfig {
    /// Base compression ratio for JPEG2000 layer
    pub base_cr: f32,

    /// Type of residual compression to apply
    pub residual_compression_type: EBCCResidualType,
}

impl Default for EBCCConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl EBCCConfig {
    /// Create a new EBCC configuration with default values.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            base_cr: 10.0,
            residual_compression_type: EBCCResidualType::Jpeg2000Only,
        }
    }

    /// Create a configuration for JPEG2000-only compression.
    #[must_use]
    pub const fn jpeg2000_only(base_cr: f32) -> Self {
        Self {
            base_cr,
            residual_compression_type: EBCCResidualType::Jpeg2000Only,
        }
    }

    /// Create a configuration for maximum error bounded compression.
    #[must_use]
    pub const fn max_absolute_error_bounded(base_cr: f32, error: f32) -> Self {
        Self {
            base_cr,
            residual_compression_type: EBCCResidualType::AbsoluteError(error),
        }
    }

    /// Create a configuration for relative error bounded compression.
    #[must_use]
    pub const fn relative_error_bounded(base_cr: f32, error: f32) -> Self {
        Self {
            base_cr,
            residual_compression_type: EBCCResidualType::RelativeError(error),
        }
    }

    /// Validate the configuration parameters.
    ///
    /// # Errors
    ///
    /// - [`EBCCError::InvalidConfig`] if `base_cr` is non-positive
    /// - [`EBCCError::InvalidConfig`] if the absolute or relative error bound
    ///   is non-positive
    pub fn validate(&self) -> EBCCResult<()> {
        // Check compression ratio
        if self.base_cr <= 0.0 {
            return Err(EBCCError::InvalidConfig(String::from(
                "Base compression ratio must be positive",
            )));
        }

        // Check residual-specific parameters
        match self.residual_compression_type {
            EBCCResidualType::AbsoluteError(error) | EBCCResidualType::RelativeError(error) => {
                if error <= 0.0 {
                    return Err(EBCCError::InvalidConfig(String::from(
                        "Error bound must be positive",
                    )));
                }
            }
            EBCCResidualType::Jpeg2000Only => {
                // No additional validation needed
            }
        }

        Ok(())
    }
}
