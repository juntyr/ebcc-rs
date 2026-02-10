//! Safe wrapper functions for EBCC compression and decompression.

use std::ptr;
use std::slice;

use ndarray::ArrayView3;
use ndarray::ArrayViewMut3;

use crate::config::EBCCConfig;
use crate::error::{EBCCError, EBCCResult};

/// Encode a 3D data array using EBCC compression.
///
/// # Arguments
///
/// - `data`: 3D input data array
/// - `config`: EBCC configuration
///
/// # Returns
///
/// The compressed data bytes.
///
/// # Errors
///
/// - [`EBCCError::InvalidInput`] if the `data` has any zero-size dimension
/// - [`EBCCError::InvalidInput`] if the size of `data` overflows or would not
///   fit into memory
/// - [`EBCCError::InvalidInput`] if the last two dimensions of `data` are not
///   both at least of size 32
/// - [`EBCCError::InvalidConfig`] if [`config.validate`][`EBCCConfig.validate`]
///   fails
/// - [`EBCCError::InvalidInput`] if the `data` contains any non-finite
///   (infinite or NaN) values
/// - [`EBCCError::CompressionError`] if compression with EBCC fails
///
/// # Examples
///
/// ```rust
/// use ebcc::{ebcc_encode, EBCCConfig};
/// use ndarray::Array;
///
/// # fn main() -> ebcc::EBCCResult<()> {
/// // 2D ERA5-like data: 721x1440
/// let data = Array::from_shape_vec((1, 721, 1440), vec![1.0f32; 721 * 1440]).unwrap();
/// let config = EBCCConfig::max_absolute_error_bounded(30.0, 0.01);
///
/// let compressed = ebcc_encode(data.view(), &config)?;
/// println!(
///     "Compressed {} bytes to {} bytes",
///     data.len() * 4, compressed.len(),
/// );
/// # Ok(())
/// # }
/// ```
pub fn ebcc_encode(data: ArrayView3<f32>, config: &EBCCConfig) -> EBCCResult<Vec<u8>> {
    // Check dimensions
    if data.shape().contains(&0) {
        return Err(EBCCError::InvalidInput(String::from(
            "All dimensions must be > 0",
        )));
    }

    // Check total size doesn't overflow
    let total_elements = data
        .shape()
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| EBCCError::InvalidInput(String::from("Dimension overflow")))?;

    if total_elements > ((isize::MAX as usize) / std::mem::size_of::<f32>()) {
        return Err(EBCCError::InvalidInput(String::from("Data too large")));
    }

    // EBCC requires last two dimensions to be at least 32x32
    if data.dim().1 < 32 || data.dim().2 < 32 {
        return Err(EBCCError::InvalidInput(format!(
            "EBCC requires last two dimensions to be at least 32x32, got {}x{}",
            data.dim().1,
            data.dim().2
        )));
    }

    // Validate configuration
    config.validate()?;

    // Check for NaN or infinity values
    for (i, &value) in data.iter().enumerate() {
        if !value.is_finite() {
            return Err(EBCCError::InvalidInput(format!(
                "Non-finite value {value} at index {i}"
            )));
        }
    }

    // Convert to FFI types
    let mut ffi_config = ebcc_sys::codec_config_t {
        dims: data.dim().into(),
        base_cr: config.base_cr,
        residual_compression_type: config.residual_compression_type.as_residual(),
        residual_cr: 1.0, // Default value for removed field
        error: config.residual_compression_type.as_error(),
    };
    let mut data_copy: Vec<f32> = data.iter().copied().collect(); // C function may modify the input

    // Call the C function
    let mut out_buffer: *mut u8 = ptr::null_mut();
    #[allow(unsafe_code)]
    let compressed_size = unsafe {
        ebcc_sys::ebcc_encode(
            data_copy.as_mut_ptr(),
            &raw mut ffi_config,
            &raw mut out_buffer,
        )
    };

    // Check for errors
    if compressed_size == 0 || out_buffer.is_null() {
        return Err(EBCCError::CompressionError(String::from(
            "ebcc_encode C function returned null or zero size",
        )));
    }

    // Copy the compressed data to a Vec and free the C-allocated memory
    #[allow(unsafe_code)]
    let compressed_data = unsafe {
        let slice = slice::from_raw_parts(out_buffer, compressed_size);
        let vec = slice.to_vec();
        ebcc_sys::free_buffer(out_buffer.cast::<core::ffi::c_void>());
        vec
    };

    Ok(compressed_data)
}

/// Decode into a 3D data array using EBCC decompression.
///
/// # Arguments
///
/// - `compressed_data`: Compressed data bytes produced by [`ebcc_encode`]
/// - `decompressed_data`: 3D output data array
///
/// # Errors
///
/// - [`EBCCError::InvalidInput`] if the `compressed_data` is empty
/// - [`EBCCError::DecompressionError`] if decompression with EBCC fails
/// - [`EBCCError::InvalidInput`] if the decompressed data does not fit into
///   `decompressed_data`
///
/// # Examples
///
/// ```rust
/// use ebcc::{ebcc_encode, ebcc_decode_into, EBCCConfig};
/// use ndarray::Array;
///
/// # fn main() -> ebcc::EBCCResult<()> {
/// let data = Array::from_shape_vec((1, 32, 32), vec![1.0f32; 32 * 32]).unwrap();
/// let config = EBCCConfig::new();
///
/// let compressed = ebcc_encode(data.view(), &config)?;
///
/// let mut decompressed = Array::zeros(data.dim());
/// ebcc_decode_into(&compressed, decompressed.view_mut())?;
/// # Ok(())
/// # }
/// ```
pub fn ebcc_decode_into(
    compressed_data: &[u8],
    mut decompressed_data: ArrayViewMut3<f32>,
) -> EBCCResult<()> {
    if compressed_data.is_empty() {
        return Err(EBCCError::InvalidInput(String::from(
            "Compressed data is empty",
        )));
    }

    let mut compressed_data_copy = Vec::from(compressed_data); // C function may modify the input

    // Call the C function
    let mut out_buffer: *mut f32 = ptr::null_mut();
    #[allow(unsafe_code)]
    let decompressed_size = unsafe {
        ebcc_sys::ebcc_decode(
            compressed_data_copy.as_mut_ptr(),
            compressed_data.len(),
            &raw mut out_buffer,
        )
    };

    // Check for errors
    if decompressed_size == 0 || out_buffer.is_null() {
        return Err(EBCCError::DecompressionError(String::from(
            "ebcc_decode C function returned null or zero size",
        )));
    }

    // Copy the decompressed data to a Vec and free the C-allocated memory
    #[allow(unsafe_code)]
    let decompressed_slice = unsafe { slice::from_raw_parts(out_buffer, decompressed_size) };

    let Ok(decompressed_view) = ArrayView3::from_shape(decompressed_data.dim(), decompressed_slice)
    else {
        return Err(EBCCError::InvalidInput(format!(
            "Decompressed data should be of shape {:?} but decompressed to {} elements",
            decompressed_data.shape(),
            decompressed_slice.len()
        )));
    };

    decompressed_data.assign(&decompressed_view);

    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use ndarray::Array;

    use super::*;

    #[test]
    fn test_encode_decode_roundtrip() -> EBCCResult<()> {
        // Create test data for 32x32 minimum size requirement
        let data = Array::from_shape_vec((1, 32, 32), vec![1.0f32; 32 * 32]).unwrap();

        let config = EBCCConfig::new();

        let compressed = ebcc_encode(data.view(), &config)?;

        let mut decompressed = Array::zeros(data.dim());
        ebcc_decode_into(&compressed, decompressed.view_mut())?;
        // Note: Due to lossy compression, values may not be exactly equal

        Ok(())
    }

    #[test]
    fn test_invalid_config() {
        let data = Array::from_shape_vec((1, 32, 32), vec![1.0f32; 32 * 32]).unwrap();

        let mut config = EBCCConfig::new();
        config.base_cr = -1.0; // Invalid compression ratio

        let result = ebcc_encode(data.view(), &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_nan_input() {
        let mut data = Array::from_shape_vec((1, 32, 32), vec![1.0f32; 32 * 32]).unwrap();
        data[(0, 3, 4)] = f32::NAN; // Insert NaN in the middle

        let config = EBCCConfig::new();

        let result = ebcc_encode(data.view(), &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_compressed_data() {
        let result = ebcc_decode_into(&[], Array::zeros([1, 1, 1]).view_mut());
        assert!(result.is_err());
    }
}
