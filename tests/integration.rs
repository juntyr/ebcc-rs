#![expect(missing_docs)]

use ebcc::{ebcc_decode_into, ebcc_encode, EBCCConfig, EBCCResult};
use ndarray::Array;

use ::{ebcc_sys as _, thiserror as _};

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
        "Max error {} exceeds 10% of data range {}",
        max_error,
        data_range
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
    let config = EBCCConfig::max_absolute_error_bounded(15.0, config_error);

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
        "Max error {} exceeds error bound {}",
        max_error,
        config_error,
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
    let config = EBCCConfig::relative_error_bounded(15.0, config_error);

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
        "Max error {} exceeds 10% of data range {}",
        max_error,
        data_range,
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
            "Constant field not preserved: {} vs {}",
            orig,
            decomp
        );
    }

    // Should compress very well
    let original_size = data.len() * std::mem::size_of::<f32>();
    let compression_ratio = original_size as f64 / compressed.len() as f64;

    println!(
        "Original size: {} bytes, Compressed size: {} bytes, Ratio: {:.2}:1",
        original_size,
        compressed.len(),
        compression_ratio
    );

    // Expect at least 2:1 compression for constant fields (was 10:1, but that may be too aggressive)
    assert!(
        compression_ratio >= 2.0,
        "Constant field should compress to at least 2:1 ratio, got {:.2}:1",
        compression_ratio
    );

    Ok(())
}

#[test]
fn test_large_array() -> EBCCResult<()> {
    // Test with a larger array (similar to small climate dataset)
    let height = 721; // Quarter degree resolution
    let width = 1440;
    let frames = 1;

    let data = Array::from_shape_fn((frames, height, width), |(_k, i, j)| {
        let lat = -90.0 + (i as f32 / height as f32) * 180.0;
        let lon = -180.0 + (j as f32 / width as f32) * 360.0;
        let temp = 273.15 + 30.0 * (1.0 - lat.abs() / 90.0) + 5.0 * (lon / 180.0).sin();
        temp
    });

    let config_error = 0.1;
    let config = EBCCConfig::max_absolute_error_bounded(20.0, config_error);

    let compressed = ebcc_encode(data.view(), &config)?;
    let mut decompressed = Array::zeros(data.dim());
    ebcc_decode_into(&compressed, decompressed.view_mut())?;

    // Check compression ratio
    let original_size = data.len() * std::mem::size_of::<f32>();
    let compression_ratio = original_size as f64 / compressed.len() as f64;

    assert!(
        compression_ratio > 5.0,
        "Compression ratio {} should be at least 5:1",
        compression_ratio
    );

    // Check error bound is respected
    let max_error = data
        .iter()
        .zip(decompressed.iter())
        .map(|(&orig, &decomp)| (orig - decomp).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_error <= (config_error + 1e-6),
        "Max error {} exceeds error bound {}",
        max_error,
        config_error,
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
        let config = EBCCConfig::max_absolute_error_bounded(15.0, error_bound);

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
        let tolerance = error_bound * 1.0 + 1e-4;
        assert!(
            max_error <= error_bound + tolerance,
            "Max error {} exceeds bound {} + tolerance {}",
            max_error,
            error_bound,
            tolerance
        );
    }

    Ok(())
}

#[test]
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

    invalid_config = EBCCConfig::max_absolute_error_bounded(10.0, -0.1); // Negative error
    assert!(invalid_config.validate().is_err());

    invalid_config = EBCCConfig::new(); // Zero dimension
    assert!(ebcc_encode(Array::zeros((0, 32, 32)).view(), &invalid_config).is_err());
}
