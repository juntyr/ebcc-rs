//! Safe wrapper functions for EBCC compression and decompression.

use std::{io::Read, ptr, slice};

use ebcc_sys::EBCC_MAX_INTERNAL_IMAGE_DIM;
pub use ebcc_sys::EBCC_NDIMS;
use ndarray::{ArrayView, ArrayViewMut, Dim, Ix};

use crate::config::EBCCConfig;
use crate::error::{EBCCError, EBCCResult};

/// EBCC data dimension.
pub type EbccDim = Dim<[Ix; EBCC_NDIMS]>;

const MIN_IMAGE_HEIGHT: usize = 32;
const MIN_IMAGE_WIDTH: usize = 32;
const CHUNKING_HEADER_MAGIC: &[u8; 4] = b"EBCK";
const CHUNKING_HEADER_VERSION: u32 = 1;
const CHUNKING_HEADER_LEN: usize = 80;

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
/// - [`EBCCError::InvalidInput`] if the last two dimensions of `data` are too
///   small or its EBCC internal image dimensions are outside the supported range
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
/// let config = EBCCConfig::max_absolute_error_bounded(0.01);
///
/// let compressed = ebcc_encode(data.view(), &config)?;
/// println!(
///     "Compressed {} bytes to {} bytes",
///     data.len() * 4, compressed.len(),
/// );
/// # Ok(())
/// # }
/// ```
pub fn ebcc_encode(data: ArrayView<f32, EbccDim>, config: &EBCCConfig) -> EBCCResult<Vec<u8>> {
    validate_data_shape(data)?;
    validate_regular_ebcc_shape(data)?;
    config.validate()?;
    validate_only_finite_data(data)?;

    // Convert to FFI types
    let mut ffi_config = ffi_config(data.dim().into(), config, [0; _]);
    let mut data_copy: Vec<f32> = data.iter().copied().collect(); // C function may modify the input

    // Call the C function
    let mut out_buffer: *mut u8 = ptr::null_mut();
    #[expect(unsafe_code)]
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
    #[expect(unsafe_code)]
    let compressed_data = unsafe {
        let slice = slice::from_raw_parts(out_buffer, compressed_size);
        let vec = slice.to_vec();
        ebcc_sys::free_buffer(out_buffer.cast::<core::ffi::c_void>());
        vec
    };

    Ok(compressed_data)
}

/// Encode a 3D data array using EBCC chunked compression.
///
/// `chunk_dims` controls the chunk shape used by EBCC.
/// Chunks may extend beyond the input dimensions.
/// EBCC pads edge chunks by repeating boundary values.
///
/// # Errors
///
/// - [`EBCCError::InvalidInput`] if the `data` has any zero-size dimension
/// - [`EBCCError::InvalidInput`] if `chunk_dims` contains a zero dimension
/// - [`EBCCError::InvalidInput`] if `chunk_dims` has tile dimensions that are
///   too small or forms EBCC internal image dimensions outside the supported
///   range
/// - [`EBCCError::InvalidConfig`] if [`config.validate`][`EBCCConfig.validate`]
///   fails
/// - [`EBCCError::InvalidInput`] if the `data` contains any non-finite
///   (infinite or NaN) values
/// - [`EBCCError::CompressionError`] if compression with EBCC fails
pub fn ebcc_encode_chunking(
    data: ArrayView<f32, EbccDim>,
    config: &EBCCConfig,
    chunk_dims: [usize; EBCC_NDIMS],
) -> EBCCResult<Vec<u8>> {
    validate_data_shape(data)?;
    validate_chunk_dims(chunk_dims)?;
    config.validate()?;
    validate_only_finite_data(data)?;

    let mut ffi_config = ffi_config(data.dim().into(), config, chunk_dims);
    let mut data_copy: Vec<f32> = data.iter().copied().collect(); // C function may modify the input

    let mut out_buffer: *mut u8 = ptr::null_mut();
    #[expect(unsafe_code)]
    let compressed_size = unsafe {
        ebcc_sys::ebcc_encode_chunking(
            data_copy.as_mut_ptr(),
            &raw mut ffi_config,
            &raw mut out_buffer,
        )
    };

    if compressed_size == 0 || out_buffer.is_null() {
        return Err(EBCCError::CompressionError(String::from(
            "ebcc_encode_chunking C function returned null or zero size",
        )));
    }

    #[expect(unsafe_code)]
    let compressed_data = unsafe {
        let slice = slice::from_raw_parts(out_buffer, compressed_size);
        let vec = slice.to_vec();
        ebcc_sys::free_buffer(out_buffer.cast::<core::ffi::c_void>());
        vec
    };

    Ok(compressed_data)
}

/// Encode a 3D data array using EBCC chunked compression in compatibility mode.
///
/// Passing all-zero `chunk_dims` lets EBCC automatically choose the chunk
/// dimensions. Passing any other chunk shape uses the same validation rules as
/// [`ebcc_encode_chunking`].
///
/// In EBCC compatibility mode, range-relative error bounds are converted to
/// absolute error bounds using the input data range before chunked encoding.
///
/// # Errors
///
/// - [`EBCCError::InvalidInput`] if the `data` has any zero-size dimension
/// - [`EBCCError::InvalidInput`] if `chunk_dims` is partially zero
/// - [`EBCCError::InvalidInput`] if explicit `chunk_dims` has tile dimensions
///   that are too small or forms EBCC internal image dimensions outside the
///   supported range
/// - [`EBCCError::InvalidConfig`] if [`config.validate`][`EBCCConfig.validate`]
///   fails
/// - [`EBCCError::InvalidInput`] if the `data` contains any non-finite
///   (infinite or NaN) values
/// - [`EBCCError::CompressionError`] if compression with EBCC fails
pub fn ebcc_encode_chunking_compat(
    data: ArrayView<f32, EbccDim>,
    config: &EBCCConfig,
    chunk_dims: [usize; EBCC_NDIMS],
) -> EBCCResult<Vec<u8>> {
    validate_data_shape(data)?;
    validate_compat_chunk_dims(chunk_dims)?;
    config.validate()?;
    validate_only_finite_data(data)?;

    let mut ffi_config = ffi_config(data.dim().into(), config, chunk_dims);
    let mut data_copy: Vec<f32> = data.iter().copied().collect(); // C function may modify the input

    let mut out_buffer: *mut u8 = ptr::null_mut();
    #[expect(unsafe_code)]
    let compressed_size = unsafe {
        ebcc_sys::ebcc_encode_chunking_compat(
            data_copy.as_mut_ptr(),
            &raw mut ffi_config,
            &raw mut out_buffer,
        )
    };

    if compressed_size == 0 || out_buffer.is_null() {
        return Err(EBCCError::CompressionError(String::from(
            "ebcc_encode_chunking_compat C function returned null or zero size",
        )));
    }

    #[expect(unsafe_code)]
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
    mut decompressed_data: ArrayViewMut<f32, EbccDim>,
) -> EBCCResult<()> {
    if compressed_data.is_empty() {
        return Err(EBCCError::InvalidInput(String::from(
            "Compressed data is empty",
        )));
    }

    let mut compressed_data_copy = Vec::from(compressed_data); // C function may modify the input

    // Call the C function
    let mut out_buffer: *mut f32 = ptr::null_mut();
    #[expect(unsafe_code)]
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
    #[expect(unsafe_code)]
    let decompressed_slice = unsafe { slice::from_raw_parts(out_buffer, decompressed_size) };

    let Ok(decompressed_view) = ArrayView::from_shape(decompressed_data.dim(), decompressed_slice)
    else {
        #[expect(unsafe_code)]
        unsafe {
            ebcc_sys::free_buffer(out_buffer.cast::<core::ffi::c_void>());
        }

        return Err(EBCCError::InvalidInput(format!(
            "Decompressed data should be of shape {:?} but decompressed to {} elements",
            decompressed_data.shape(),
            decompressed_size
        )));
    };

    decompressed_data.assign(&decompressed_view);

    #[expect(unsafe_code)]
    unsafe {
        ebcc_sys::free_buffer(out_buffer.cast::<core::ffi::c_void>());
    }

    Ok(())
}

/// Decode EBCC chunked compressed data into a 3D data array.
///
/// # Errors
///
/// - [`EBCCError::InvalidInput`] if the `compressed_data` is empty
/// - [`EBCCError::DecompressionError`] if decompression with EBCC fails
/// - [`EBCCError::InvalidInput`] if the decompressed data does not fit into
///   `decompressed_data`
pub fn ebcc_decode_chunking_into(
    compressed_data: &[u8],
    mut decompressed_data: ArrayViewMut<f32, EbccDim>,
) -> EBCCResult<()> {
    if compressed_data.is_empty() {
        return Err(EBCCError::InvalidInput(String::from(
            "Compressed data is empty",
        )));
    }

    if let Some(encoded_dims) = read_dims_from_chunking_header(compressed_data)? {
        let output_dims: [usize; EBCC_NDIMS] = decompressed_data.dim().into();
        if output_dims != encoded_dims {
            return Err(EBCCError::InvalidInput(format!(
                "Chunked EBCC data has shape {encoded_dims:?} but output array has shape {output_dims:?}",
            )));
        }
    }

    let mut compressed_data_copy = Vec::from(compressed_data); // C function may modify the input

    let mut out_buffer: *mut f32 = ptr::null_mut();
    #[expect(unsafe_code)]
    let decompressed_size = unsafe {
        ebcc_sys::ebcc_decode_chunking(
            compressed_data_copy.as_mut_ptr(),
            compressed_data.len(),
            &raw mut out_buffer,
        )
    };

    if decompressed_size == 0 || out_buffer.is_null() {
        return Err(EBCCError::DecompressionError(String::from(
            "ebcc_decode_chunking C function returned null or zero size",
        )));
    }

    #[expect(unsafe_code)]
    let decompressed_slice = unsafe { slice::from_raw_parts(out_buffer, decompressed_size) };

    let Ok(decompressed_view) = ArrayView::from_shape(decompressed_data.dim(), decompressed_slice)
    else {
        #[expect(unsafe_code)]
        unsafe {
            ebcc_sys::free_buffer(out_buffer.cast::<core::ffi::c_void>());
        }

        return Err(EBCCError::InvalidInput(format!(
            "Decompressed data should be of shape {:?} but decompressed to {} elements",
            decompressed_data.shape(),
            decompressed_size
        )));
    };

    decompressed_data.assign(&decompressed_view);

    #[expect(unsafe_code)]
    unsafe {
        ebcc_sys::free_buffer(out_buffer.cast::<core::ffi::c_void>());
    }

    Ok(())
}

fn validate_data_shape(data: ArrayView<f32, EbccDim>) -> EBCCResult<usize> {
    if data.shape().contains(&0) {
        return Err(EBCCError::InvalidInput(String::from(
            "All dimensions must be > 0",
        )));
    }

    let Some(total_elements) = data
        .shape()
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
    else {
        return Err(EBCCError::InvalidInput(String::from("Dimension overflow")));
    };

    if total_elements > ((isize::MAX as usize) / std::mem::size_of::<f32>()) {
        return Err(EBCCError::InvalidInput(String::from("Data too large")));
    }

    Ok(total_elements)
}

fn validate_regular_ebcc_shape(data: ArrayView<f32, EbccDim>) -> EBCCResult<()> {
    // EBCC flattens all dimensions except the last into one internal image height.
    let (depth, height, width) = data.dim();
    let Some(image_height) = depth.checked_mul(height) else {
        return Err(EBCCError::InvalidInput(String::from("Dimension overflow")));
    };

    if height < MIN_IMAGE_HEIGHT
        || !(MIN_IMAGE_HEIGHT..=EBCC_MAX_INTERNAL_IMAGE_DIM).contains(&image_height)
        || !(MIN_IMAGE_WIDTH..=EBCC_MAX_INTERNAL_IMAGE_DIM).contains(&width)
    {
        return Err(EBCCError::InvalidInput(format!(
            "EBCC requires tile dimensions of at least 32 and internal image dimensions at most {EBCC_MAX_INTERNAL_IMAGE_DIM}, got shape {depth}x{height}x{width}",
        )));
    }

    Ok(())
}

fn validate_chunk_dims(chunk_dims: [usize; EBCC_NDIMS]) -> EBCCResult<()> {
    if chunk_dims.contains(&0) {
        return Err(EBCCError::InvalidInput(String::from(
            "All chunk dimensions must be > 0",
        )));
    }

    let [chunk_depth, chunk_height, chunk_width] = chunk_dims;
    let Some(image_height) = chunk_depth.checked_mul(chunk_height) else {
        return Err(EBCCError::InvalidInput(String::from(
            "Chunk dimension overflow",
        )));
    };
    if chunk_height < MIN_IMAGE_HEIGHT
        || !(MIN_IMAGE_HEIGHT..=EBCC_MAX_INTERNAL_IMAGE_DIM).contains(&image_height)
        || !(MIN_IMAGE_WIDTH..=EBCC_MAX_INTERNAL_IMAGE_DIM).contains(&chunk_width)
    {
        return Err(EBCCError::InvalidInput(format!(
            "EBCC requires chunk tile dimensions of at least 32 and internal image dimensions at most {EBCC_MAX_INTERNAL_IMAGE_DIM}, got {chunk_depth}x{chunk_height}x{chunk_width}",
        )));
    }

    let Some(total_elements) = chunk_dims
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
    else {
        return Err(EBCCError::InvalidInput(String::from(
            "Chunk dimension overflow",
        )));
    };

    if total_elements > ((isize::MAX as usize) / std::mem::size_of::<f32>()) {
        return Err(EBCCError::InvalidInput(String::from(
            "Chunk dimensions are too large",
        )));
    }

    Ok(())
}

fn validate_compat_chunk_dims(chunk_dims: [usize; EBCC_NDIMS]) -> EBCCResult<()> {
    if chunk_dims == [0; _] {
        return Ok(());
    }

    validate_chunk_dims(chunk_dims)
}

fn validate_only_finite_data(data: ArrayView<f32, EbccDim>) -> EBCCResult<()> {
    for (i, &value) in data.indexed_iter() {
        if !value.is_finite() {
            return Err(EBCCError::InvalidInput(format!(
                "Non-finite value {value} at index {i:?}"
            )));
        }
    }

    Ok(())
}

const fn ffi_config(
    dims: [usize; EBCC_NDIMS],
    config: &EBCCConfig,
    chunk_dims: [usize; EBCC_NDIMS],
) -> ebcc_sys::codec_config_t {
    ebcc_sys::codec_config_t {
        dims,
        base_cr: config.base_cr,
        residual_compression_type: config.residual_compression_type.as_residual(),
        residual_cr: 1.0, // Default value for removed field
        error: config.residual_compression_type.as_error(),
        chunk_dims,
    }
}

fn read_dims_from_chunking_header(
    compressed_data: &[u8],
) -> EBCCResult<Option<[usize; EBCC_NDIMS]>> {
    if compressed_data.len() < CHUNKING_HEADER_LEN {
        return Ok(None);
    }

    let Some(mut compressed_data) = compressed_data.strip_prefix(CHUNKING_HEADER_MAGIC) else {
        return Ok(None);
    };

    let reader = &mut compressed_data;

    let version = read_u32_le(reader)?;
    if version != CHUNKING_HEADER_VERSION {
        return Err(EBCCError::DecompressionError(format!(
            "Unsupported EBCC chunking header version: {version}",
        )));
    }

    let Ok(ndims) = usize::try_from(read_u32_le(reader)?) else {
        return Err(EBCCError::DecompressionError(String::from(
            "EBCC chunking dimensionality does not fit into usize",
        )));
    };

    if ndims != EBCC_NDIMS {
        return Err(EBCCError::InvalidInput(format!(
            "EBCC chunking streams must be 3D but has {ndims} dimensions"
        )));
    }

    let _padding = read_u32_le(reader)?;

    let mut dims = [0; EBCC_NDIMS];
    for dim in &mut dims {
        let value = read_u64_le(reader)?;
        *dim = usize::try_from(value).map_err(|_| {
            EBCCError::InvalidInput(format!(
                "Chunked EBCC dimension {value} does not fit into usize",
            ))
        })?;
    }

    Ok(Some(dims))
}

fn read_u32_le(reader: &mut &[u8]) -> EBCCResult<u32> {
    let mut array = [0; _];
    let Ok(()) = reader.read_exact(&mut array) else {
        return Err(EBCCError::InvalidInput(String::from(
            "Chunked EBCC header is truncated",
        )));
    };
    Ok(u32::from_le_bytes(array))
}

fn read_u64_le(reader: &mut &[u8]) -> EBCCResult<u64> {
    let mut array = [0; _];
    let Ok(()) = reader.read_exact(&mut array) else {
        return Err(EBCCError::InvalidInput(String::from(
            "Chunked EBCC header is truncated",
        )));
    };
    Ok(u64::from_le_bytes(array))
}

#[cfg(test)]
#[expect(clippy::unwrap_used, clippy::indexing_slicing)]
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

    #[test]
    #[expect(clippy::cast_precision_loss)]
    fn test_encode_decode_chunking_roundtrip() -> EBCCResult<()> {
        let data = Array::from_shape_fn((1, 64, 64), |(_frame, y, x)| (y + x) as f32);
        let config = EBCCConfig::new();

        let compressed = ebcc_encode_chunking(data.view(), &config, [1, 32, 32])?;

        let mut decompressed = Array::zeros(data.dim());
        ebcc_decode_chunking_into(&compressed, decompressed.view_mut())?;

        Ok(())
    }
}
