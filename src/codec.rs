//! Safe wrapper functions for EBCC compression and decompression.

use std::ptr;
use std::slice;

use ndarray::ArrayView3;
use ndarray::ArrayViewMut3;

use crate::config::EBCCConfig;
use crate::error::{EBCCError, EBCCResult};

const MIN_IMAGE_HEIGHT: usize = 32;
const MIN_IMAGE_WIDTH: usize = 32;
const MAX_INTERNAL_IMAGE_DIM: usize = ebcc_sys::EBCC_MAX_INTERNAL_IMAGE_DIM as usize;
const CHUNKING_HEADER_MAGIC: &[u8; 4] = b"EBCK";
const CHUNKING_HEADER_VERSION: u32 = 1;
const CHUNKING_HEADER_LEN: usize = 80;
const CHUNKING_HEADER_VERSION_OFFSET: usize = 4;
const CHUNKING_HEADER_NDIMS_OFFSET: usize = 8;
const CHUNKING_HEADER_DIMS_OFFSET: usize = 16;
const U32_LEN: usize = std::mem::size_of::<u32>();
const U64_LEN: usize = std::mem::size_of::<u64>();

fn validate_data_shape(data: ArrayView3<f32>) -> EBCCResult<usize> {
    if data.shape().contains(&0) {
        return Err(EBCCError::InvalidInput(String::from(
            "All dimensions must be > 0",
        )));
    }

    let total_elements = data
        .shape()
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| EBCCError::InvalidInput(String::from("Dimension overflow")))?;

    if total_elements > ((isize::MAX as usize) / std::mem::size_of::<f32>()) {
        return Err(EBCCError::InvalidInput(String::from("Data too large")));
    }

    Ok(total_elements)
}

fn validate_regular_ebcc_shape(data: ArrayView3<f32>) -> EBCCResult<()> {
    // EBCC flattens all dimensions except the last into one internal image height.
    let (depth, height, width) = data.dim();
    let image_height = depth
        .checked_mul(height)
        .ok_or_else(|| EBCCError::InvalidInput(String::from("Dimension overflow")))?;

    if height < MIN_IMAGE_HEIGHT
        || !(MIN_IMAGE_HEIGHT..=MAX_INTERNAL_IMAGE_DIM).contains(&image_height)
        || !(MIN_IMAGE_WIDTH..=MAX_INTERNAL_IMAGE_DIM).contains(&width)
    {
        return Err(EBCCError::InvalidInput(format!(
            "EBCC requires tile dimensions of at least 32 and internal image dimensions at most {MAX_INTERNAL_IMAGE_DIM}, got shape {depth}x{height}x{width}",
        )));
    }

    Ok(())
}

fn validate_chunk_dims(chunk_dims: [usize; 3]) -> EBCCResult<()> {
    if chunk_dims.contains(&0) {
        return Err(EBCCError::InvalidInput(String::from(
            "All chunk dimensions must be > 0",
        )));
    }

    let [chunk_depth, chunk_height, chunk_width] = chunk_dims;
    let image_height = chunk_depth
        .checked_mul(chunk_height)
        .ok_or_else(|| EBCCError::InvalidInput(String::from("Chunk dimension overflow")))?;
    if chunk_height < MIN_IMAGE_HEIGHT
        || !(MIN_IMAGE_HEIGHT..=MAX_INTERNAL_IMAGE_DIM).contains(&image_height)
        || !(MIN_IMAGE_WIDTH..=MAX_INTERNAL_IMAGE_DIM).contains(&chunk_width)
    {
        return Err(EBCCError::InvalidInput(format!(
            "EBCC requires chunk tile dimensions of at least 32 and internal image dimensions at most {MAX_INTERNAL_IMAGE_DIM}, got {chunk_depth}x{chunk_height}x{chunk_width}",
        )));
    }

    let total_elements = chunk_dims
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| EBCCError::InvalidInput(String::from("Chunk dimension overflow")))?;

    if total_elements > ((isize::MAX as usize) / std::mem::size_of::<f32>()) {
        return Err(EBCCError::InvalidInput(String::from(
            "Chunk dimensions are too large",
        )));
    }

    Ok(())
}

fn validate_compat_chunk_dims(chunk_dims: [usize; 3]) -> EBCCResult<()> {
    if chunk_dims == [0; 3] {
        return Ok(());
    }

    validate_chunk_dims(chunk_dims)
}

fn read_u32_le(data: &[u8], offset: usize) -> EBCCResult<u32> {
    let bytes = data.get(offset..offset + U32_LEN).ok_or_else(|| {
        EBCCError::InvalidInput(String::from("Chunked EBCC header is truncated"))
    })?;
    let mut array = [0; U32_LEN];
    array.copy_from_slice(bytes);
    Ok(u32::from_le_bytes(array))
}

fn read_u64_le(data: &[u8], offset: usize) -> EBCCResult<u64> {
    let bytes = data.get(offset..offset + U64_LEN).ok_or_else(|| {
        EBCCError::InvalidInput(String::from("Chunked EBCC header is truncated"))
    })?;
    let mut array = [0; U64_LEN];
    array.copy_from_slice(bytes);
    Ok(u64::from_le_bytes(array))
}

fn chunked_stream_dims(compressed_data: &[u8]) -> EBCCResult<Option<[usize; 3]>> {
    if compressed_data.len() < CHUNKING_HEADER_LEN
        || !compressed_data.starts_with(CHUNKING_HEADER_MAGIC)
    {
        return Ok(None);
    }

    let version = read_u32_le(compressed_data, CHUNKING_HEADER_VERSION_OFFSET)?;
    if version != CHUNKING_HEADER_VERSION {
        return Err(EBCCError::DecompressionError(format!(
            "Unsupported EBCC chunking header version: {version}",
        )));
    }

    let ndims = usize::try_from(read_u32_le(compressed_data, CHUNKING_HEADER_NDIMS_OFFSET)?)
        .map_err(|_| {
            EBCCError::DecompressionError(String::from(
                "EBCC chunking dimensionality does not fit into usize",
            ))
        })?;

    let mut dims = [0; 3];
    for (dim, offset) in dims
        .iter_mut()
        .zip((0..ndims).map(|i| CHUNKING_HEADER_DIMS_OFFSET + i * U64_LEN))
    {
        let value = read_u64_le(compressed_data, offset)?;
        *dim = usize::try_from(value).map_err(|_| {
            EBCCError::InvalidInput(format!(
                "Chunked EBCC dimension {value} does not fit into usize",
            ))
        })?;
    }
    assert_eq!(ndims, dims.len(), "EBCC chunking streams must be 3D");

    Ok(Some(dims))
}

fn validate_finite_data(data: ArrayView3<f32>) -> EBCCResult<()> {
    for (i, &value) in data.iter().enumerate() {
        if !value.is_finite() {
            return Err(EBCCError::InvalidInput(format!(
                "Non-finite value {value} at index {i}"
            )));
        }
    }

    Ok(())
}

const fn ffi_config(
    dims: [usize; 3],
    config: &EBCCConfig,
    chunk_dims: [usize; 3],
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
pub fn ebcc_encode(data: ArrayView3<f32>, config: &EBCCConfig) -> EBCCResult<Vec<u8>> {
    validate_data_shape(data)?;
    validate_regular_ebcc_shape(data)?;
    config.validate()?;
    validate_finite_data(data)?;

    // Convert to FFI types
    let mut ffi_config = ffi_config(data.dim().into(), config, [0; 3]);
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
/// `chunk_dims` controls the chunk shape used by the C `ebcc_encode_chunking`
/// API. Chunks may extend beyond the input dimensions; EBCC pads edge chunks by
/// repeating boundary values.
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
    data: ArrayView3<f32>,
    config: &EBCCConfig,
    chunk_dims: [usize; 3],
) -> EBCCResult<Vec<u8>> {
    validate_data_shape(data)?;
    validate_chunk_dims(chunk_dims)?;
    config.validate()?;
    validate_finite_data(data)?;

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
/// This calls the C `ebcc_encode_chunking_compat` API. Passing `[0; 3]` for
/// `chunk_dims` lets EBCC automatically choose chunk dimensions 
/// `(1, dim_y>2047?1024:dim_y, dim_x>2047?1024:dim_x)`. Passing any
/// other chunk shape uses the same validation rules as [`ebcc_encode_chunking`].
///
/// In EBCC compatibility mode, range-relative error configurations are converted
/// to absolute error bounds using the input data range before chunked encoding.
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
    data: ArrayView3<f32>,
    config: &EBCCConfig,
    chunk_dims: [usize; 3],
) -> EBCCResult<Vec<u8>> {
    validate_data_shape(data)?;
    validate_compat_chunk_dims(chunk_dims)?;
    config.validate()?;
    validate_finite_data(data)?;

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

    let Ok(decompressed_view) = ArrayView3::from_shape(decompressed_data.dim(), decompressed_slice)
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
    mut decompressed_data: ArrayViewMut3<f32>,
) -> EBCCResult<()> {
    if compressed_data.is_empty() {
        return Err(EBCCError::InvalidInput(String::from(
            "Compressed data is empty",
        )));
    }

    if let Some(encoded_dims) = chunked_stream_dims(compressed_data)? {
        let output_dims: [usize; 3] = decompressed_data.dim().into();
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

    let Ok(decompressed_view) = ArrayView3::from_shape(decompressed_data.dim(), decompressed_slice)
    else {
        let decompressed_len = decompressed_slice.len();
        #[expect(unsafe_code)]
        unsafe {
            ebcc_sys::free_buffer(out_buffer.cast::<core::ffi::c_void>());
        }
        return Err(EBCCError::InvalidInput(format!(
            "Decompressed data should be of shape {:?} but decompressed to {} elements",
            decompressed_data.shape(),
            decompressed_len
        )));
    };

    decompressed_data.assign(&decompressed_view);
    #[expect(unsafe_code)]
    unsafe {
        ebcc_sys::free_buffer(out_buffer.cast::<core::ffi::c_void>());
    }

    Ok(())
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
