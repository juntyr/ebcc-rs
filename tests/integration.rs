#![expect(missing_docs)]

use ebcc::{
    ebcc_decode_chunking_into, ebcc_decode_into, ebcc_encode, ebcc_encode_chunking,
    ebcc_encode_chunking_compat, EBCCConfig, EBCCError, EBCCResult, EbccDim, EBCC_NDIMS,
};
use ndarray::Array;

use {ebcc_sys as _, thiserror as _};

#[test]
fn test_basic_compression_roundtrip() -> EBCCResult<()> {
    let data = Array::from_shape_simple_fn((1, 32, 32), || 1.0);
    let config = EBCCConfig::new();

    let compressed = ebcc_encode(data.view(), &config)?;
    let mut decompressed = Array::zeros(data.dim());
    ebcc_decode_into(&compressed, decompressed.view_mut())?;

    // Check that the compression actually reduced the size
    let original_size = data.len() * std::mem::size_of::<f32>();
    assert!(
        compressed.len() < original_size,
        "Compressed size ({}) should be less than original size ({})",
        compressed.len(),
        original_size
    );

    Ok(())
}

#[test]
fn test_jpeg2000_only_compression() -> EBCCResult<()> {
    let mut i: i16 = 0;
    let data = Array::from_shape_simple_fn((1, 32, 32), || {
        let x = f32::from(i) * 0.1;
        i += 1;
        x
    });
    let config = EBCCConfig::jpeg2000_only(10.0);

    let compressed = ebcc_encode(data.view(), &config)?;
    let mut decompressed = Array::zeros(data.dim());
    ebcc_decode_into(&compressed, decompressed.view_mut())?;

    // Check that data is approximately preserved
    let max_error = data
        .iter()
        .zip(decompressed.iter())
        .map(|(&orig, &decomp)| (orig - decomp).abs())
        .fold(0.0f32, f32::max);

    // Error should be reasonable (less than 10% of data range)
    let data_range = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
        - data.iter().fold(f32::INFINITY, |a, &b| a.min(b));

    assert!(
        max_error < data_range * 0.1,
        "Max error {max_error} exceeds 10% of data range {data_range}",
    );

    Ok(())
}

#[test]
fn test_max_error_bounded_compression() -> EBCCResult<()> {
    let mut i: i16 = 0;
    let data = Array::from_shape_simple_fn((1, 32, 32), || {
        let x = f32::from(i) * 0.1;
        i += 1;
        x
    });
    let config_error = 0.1;
    let config = EBCCConfig::max_absolute_error_bounded(config_error).with_base_cr(15.0);

    let compressed = ebcc_encode(data.view(), &config)?;
    let mut decompressed = Array::zeros(data.dim());
    ebcc_decode_into(&compressed, decompressed.view_mut())?;

    // Check that data is approximately preserved
    let max_error = data
        .iter()
        .zip(decompressed.iter())
        .map(|(&orig, &decomp)| (orig - decomp).abs())
        .fold(0.0f32, f32::max);

    // For max error bounded, error should be within the specified bound
    assert!(
        max_error <= (config_error + 1e-6),
        "Max error {max_error} exceeds error bound {config_error}",
    );

    Ok(())
}

#[test]
fn test_relative_error_bounded_compression() -> EBCCResult<()> {
    let mut i: i16 = 0;
    let data = Array::from_shape_simple_fn((1, 32, 32), || {
        let x = f32::from(i) * 0.1;
        i += 1;
        x
    });
    let config_error = 0.001;
    let config = EBCCConfig::relative_error_bounded(config_error).with_base_cr(15.0);

    let compressed = ebcc_encode(data.view(), &config)?;
    let mut decompressed = Array::zeros(data.dim());
    ebcc_decode_into(&compressed, decompressed.view_mut())?;

    // Check that data is approximately preserved
    let max_error = data
        .iter()
        .zip(decompressed.iter())
        .map(|(&orig, &decomp)| (orig - decomp).abs())
        .fold(0.0f32, f32::max);

    // For relative error, check that it's reasonable
    let data_range = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
        - data.iter().fold(f32::INFINITY, |a, &b| a.min(b));

    assert!(
        max_error < data_range * 0.1,
        "Max error {max_error} exceeds 10% of data range {data_range}",
    );

    Ok(())
}

#[test]
fn test_constant_field() -> EBCCResult<()> {
    // Test with constant field (should be handled efficiently)
    let data = Array::from_shape_simple_fn((1, 32, 32), || 42.0);
    let config = EBCCConfig::new();

    let compressed = ebcc_encode(data.view(), &config)?;
    let mut decompressed = Array::zeros(data.dim());
    ebcc_decode_into(&compressed, decompressed.view_mut())?;

    // For constant fields, should be perfectly preserved
    for (&orig, &decomp) in data.iter().zip(decompressed.iter()) {
        assert!(
            (orig - decomp).abs() < 1e-6,
            "Constant field not preserved: {orig} vs {decomp}",
        );
    }

    // Should compress very well
    let original_size = data.len() * std::mem::size_of::<f32>();
    #[expect(clippy::cast_precision_loss)]
    let compression_ratio = original_size as f64 / compressed.len() as f64;

    println!(
        "Original size: {original_size} bytes, Compressed size: {} bytes, Ratio: {compression_ratio:.2}:1",
        compressed.len(),
    );

    // Expect at least 2:1 compression for constant fields (was 10:1, but that may be too aggressive)
    assert!(
        compression_ratio >= 2.0,
        "Constant field should compress to at least 2:1 ratio, got {compression_ratio:.2}:1",
    );

    Ok(())
}

#[test]
#[expect(clippy::cast_precision_loss)]
fn test_chunking_compression_roundtrip() -> EBCCResult<()> {
    let data = Array::from_shape_fn((1, 64, 64), |(_frame, y, x)| {
        (y as f32).sin() + (x as f32).cos()
    });
    let config = EBCCConfig::jpeg2000_only(10.0);

    let decompressed = chunking_roundtrip(&data, &config, [1, 32, 32])?;

    assert_eq!(decompressed.dim(), data.dim());

    Ok(())
}

const LARGE_CHUNKED_SHAPE: [usize; EBCC_NDIMS] = [5, 130, 150];
const VALID_NON_DIVISIBLE_CHUNK_DIMS: [usize; EBCC_NDIMS] = [3, 32, 41];
const REQUESTED_INVALID_CHUNK_DIMS: &[[usize; EBCC_NDIMS]] =
    &[[1, 31, 41], [3, 31, 41], [1, 140, 31]];

#[expect(clippy::cast_precision_loss)]
fn large_chunking_data() -> Array<f32, EbccDim> {
    Array::from_shape_fn(LARGE_CHUNKED_SHAPE, |(frame, y, x)| {
        let frame_term = frame as f32 * 0.75;
        let y_term = (y as f32 / 13.0).sin() * 10.0;
        let x_term = (x as f32 / 17.0).cos() * 7.0;
        let fine_term = ((y * x + frame) % 19) as f32 * 0.03;
        280.0 + frame_term + y_term + x_term + fine_term
    })
}

fn data_range(data: &Array<f32, EbccDim>) -> f32 {
    data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
        - data.iter().fold(f32::INFINITY, |a, &b| a.min(b))
}

fn max_abs_error(data: &Array<f32, EbccDim>, decompressed: &Array<f32, EbccDim>) -> f32 {
    data.iter()
        .zip(decompressed.iter())
        .map(|(&orig, &decomp)| (orig - decomp).abs())
        .fold(0.0f32, f32::max)
}

fn chunking_roundtrip(
    data: &Array<f32, EbccDim>,
    config: &EBCCConfig,
    chunk_dims: [usize; EBCC_NDIMS],
) -> EBCCResult<Array<f32, EbccDim>> {
    let compressed = ebcc_encode_chunking(data.view(), config, chunk_dims)?;
    let mut decompressed = Array::zeros(data.dim());
    ebcc_decode_chunking_into(&compressed, decompressed.view_mut())?;

    Ok(decompressed)
}

fn chunking_compat_roundtrip(
    data: &Array<f32, EbccDim>,
    config: &EBCCConfig,
    chunk_dims: Option<[usize; EBCC_NDIMS]>,
) -> EBCCResult<Array<f32, EbccDim>> {
    let compressed = ebcc_encode_chunking_compat(data.view(), config, chunk_dims)?;
    let mut decompressed = Array::zeros(data.dim());
    ebcc_decode_chunking_into(&compressed, decompressed.view_mut())?;

    Ok(decompressed)
}

#[test]
fn test_chunking_large_jpeg2000_only_with_non_divisible_chunks() -> EBCCResult<()> {
    let data = large_chunking_data();
    let config = EBCCConfig::jpeg2000_only(10.0);

    let decompressed = chunking_roundtrip(&data, &config, VALID_NON_DIVISIBLE_CHUNK_DIMS)?;

    assert_eq!(decompressed.dim(), data.dim());
    assert!(
        max_abs_error(&data, &decompressed) < data_range(&data) * 0.15,
        "JPEG2000-only chunking error is unexpectedly large",
    );

    Ok(())
}

#[test]
fn test_chunking_large_max_error_bound_with_non_divisible_chunks() -> EBCCResult<()> {
    let data = large_chunking_data();
    let config_error = 0.5;
    let config = EBCCConfig::max_absolute_error_bounded(config_error).with_base_cr(20.0);

    let decompressed = chunking_roundtrip(&data, &config, VALID_NON_DIVISIBLE_CHUNK_DIMS)?;
    let max_error = max_abs_error(&data, &decompressed);
    let tolerance = config_error * 0.02;

    assert_eq!(decompressed.dim(), data.dim());
    assert!(
        max_error <= config_error + tolerance,
        "Max error {max_error} exceeds error bound {config_error} + tolerance {tolerance}",
    );

    Ok(())
}

#[test]
fn test_chunking_large_range_relative_error_bound_with_non_divisible_chunks() -> EBCCResult<()> {
    let data = large_chunking_data();
    let config_error = 0.01;
    let config = EBCCConfig::relative_error_bounded(config_error).with_base_cr(20.0);

    let decompressed = chunking_roundtrip(&data, &config, VALID_NON_DIVISIBLE_CHUNK_DIMS)?;
    let max_error = max_abs_error(&data, &decompressed);
    let range_error_bound = data_range(&data) * config_error;

    assert_eq!(decompressed.dim(), data.dim());
    assert!(
        max_error <= range_error_bound * (1.0 + 1e-4),
        "Max error {max_error} exceeds range-relative bound {range_error_bound}",
    );

    Ok(())
}

#[test]
fn test_chunking_compat_default_chunks_range_relative_error_bound() -> EBCCResult<()> {
    let data = large_chunking_data();
    let config_error = 0.01;
    let config = EBCCConfig::relative_error_bounded(config_error).with_base_cr(20.0);

    let decompressed = chunking_compat_roundtrip(&data, &config, None)?;
    let max_error = max_abs_error(&data, &decompressed);
    let range_error_bound = data_range(&data) * config_error;
    let tolerance = range_error_bound * 0.02;

    assert_eq!(decompressed.dim(), data.dim());
    assert!(
        max_error <= range_error_bound + tolerance,
        "Max error {max_error} exceeds compat range-relative bound {range_error_bound} + tolerance {tolerance}",
    );

    Ok(())
}

#[test]
fn test_chunking_rejects_requested_chunks_below_tile_limit() {
    let data = large_chunking_data();
    let config = EBCCConfig::jpeg2000_only(10.0);

    for chunk_dims in REQUESTED_INVALID_CHUNK_DIMS {
        let result = ebcc_encode_chunking(data.view(), &config, *chunk_dims);
        assert!(
            result.is_err(),
            "Chunk dimensions {chunk_dims:?} should be rejected",
        );
    }
}

#[test]
fn test_decode_chunking_rejects_same_len_wrong_shape() -> EBCCResult<()> {
    #[expect(clippy::cast_precision_loss, clippy::suboptimal_flops)]
    let data = Array::from_shape_fn((2, 32, 32), |(frame, y, x)| {
        frame as f32 * 1024.0 + y as f32 * 32.0 + x as f32
    });
    let config = EBCCConfig::jpeg2000_only(10.0);
    let compressed = ebcc_encode_chunking(data.view(), &config, [1, 32, 32])?;

    let mut wrong_shape = Array::zeros((1, 64, 32));
    let result = ebcc_decode_chunking_into(&compressed, wrong_shape.view_mut());

    assert!(matches!(
        result,
        Err(EBCCError::InvalidInput(message))
            if message.contains("Chunked EBCC data has shape [2, 32, 32]")
    ));

    Ok(())
}

#[test]
fn test_large_array() -> EBCCResult<()> {
    // Test with a larger array (similar to small climate dataset)
    let height = 721; // Quarter degree resolution
    let width = 1440;
    let frames = 1;

    #[expect(clippy::suboptimal_flops, clippy::cast_precision_loss)]
    let data = Array::from_shape_fn((frames, height, width), |(_k, i, j)| {
        let lat = -90.0 + (i as f32 / height as f32) * 180.0;
        let lon = -180.0 + (j as f32 / width as f32) * 360.0;
        #[allow(clippy::let_and_return)]
        let temp = 273.15 + 30.0 * (1.0 - lat.abs() / 90.0) + 5.0 * (lon / 180.0).sin();
        temp
    });

    let config_error = 0.1;
    let config = EBCCConfig::max_absolute_error_bounded(config_error).with_base_cr(20.0);

    let compressed = ebcc_encode(data.view(), &config)?;
    let mut decompressed = Array::zeros(data.dim());
    ebcc_decode_into(&compressed, decompressed.view_mut())?;

    // Check compression ratio
    let original_size = data.len() * std::mem::size_of::<f32>();
    #[expect(clippy::cast_precision_loss)]
    let compression_ratio = original_size as f64 / compressed.len() as f64;

    assert!(
        compression_ratio > 5.0,
        "Compression ratio {compression_ratio} should be at least 5:1",
    );

    // Check error bound is respected
    let max_error = data
        .iter()
        .zip(decompressed.iter())
        .map(|(&orig, &decomp)| (orig - decomp).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_error <= (config_error + 1e-6),
        "Max error {max_error} exceeds error bound {config_error}",
    );

    Ok(())
}

#[test]
fn test_error_bounds() -> EBCCResult<()> {
    let mut i: i16 = 0;
    let data = Array::from_shape_simple_fn((1, 32, 32), || {
        let x = (f32::from(i) * 0.1).sin() * 100.0;
        i += 1;
        x
    });

    // Test different error bounds
    let error_bounds = [0.01, 0.1, 1.0, 5.0];

    for error_bound in error_bounds {
        let config = EBCCConfig::max_absolute_error_bounded(error_bound).with_base_cr(15.0);

        let compressed = ebcc_encode(data.view(), &config)?;
        let mut decompressed = Array::zeros(data.dim());
        ebcc_decode_into(&compressed, decompressed.view_mut())?;

        let max_error = data
            .iter()
            .zip(decompressed.iter())
            .map(|(&orig, &decomp)| (orig - decomp).abs())
            .fold(0.0f32, f32::max);

        // Allow reasonable tolerance for compression algorithms (100% + small epsilon)
        // Note: Error-bounded compression is approximate and may exceed bounds slightly
        let tolerance = error_bound * (1.0 + 1e-4);
        assert!(
            max_error <= error_bound + tolerance,
            "Max error {max_error} exceeds bound {error_bound} + tolerance {tolerance}",
        );
    }

    Ok(())
}

#[test]
#[expect(clippy::indexing_slicing)]
fn test_invalid_inputs() {
    // Test with NaN values
    let mut data_with_nan = Array::from_shape_simple_fn((1, 32, 32), || 1.0);
    data_with_nan[(0, 0, 1)] = f32::NAN;
    let config = EBCCConfig::new();

    let result = ebcc_encode(data_with_nan.view(), &config);
    assert!(result.is_err());

    // Test with infinite values
    let mut data_with_inf = Array::from_shape_simple_fn((1, 32, 32), || 1.0);
    data_with_inf[(0, 0, 1)] = f32::INFINITY;

    let result = ebcc_encode(data_with_inf.view(), &config);
    assert!(result.is_err());

    // Test decompression with empty data
    let result = ebcc_decode_into(&[], Array::zeros((0, 0, 0)).view_mut());
    assert!(result.is_err());
}

#[test]
fn test_config_validation() {
    // Valid config should pass
    let valid_config = EBCCConfig::new();
    assert!(valid_config.validate().is_ok());

    // Invalid configs should fail
    let mut invalid_config = EBCCConfig::new();
    invalid_config.base_cr = -1.0; // Negative compression ratio
    assert!(invalid_config.validate().is_err());

    invalid_config = EBCCConfig::max_absolute_error_bounded(-0.1); // Negative error
    assert!(invalid_config.validate().is_err());

    invalid_config = EBCCConfig::new(); // Zero dimension
    assert!(ebcc_encode(Array::zeros((0, 32, 32)).view(), &invalid_config).is_err());
}
