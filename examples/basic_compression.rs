//! Basic EBCC compression example.
//!
//! This example demonstrates how to use the EBCC Rust bindings for
//! compressing and decompressing climate data.

use ebcc::{ebcc_decode_into, ebcc_encode, EBCCConfig, EBCCResult};
use ndarray::Array;

#[allow(clippy::cast_precision_loss, clippy::suboptimal_flops)]
fn main() -> EBCCResult<()> {
    println!("EBCC Basic Compression Example");
    println!("=============================");

    // Create some synthetic climate data (ERA5-like grid)
    let height = 721;
    let width = 1440;
    let frames = 1;

    // Generate synthetic temperature data (in Kelvin)
    let data = Array::from_shape_fn((frames, height, width), |(_f, i, j)| {
        // Simple synthetic temperature field with spatial variation
        let lat = -90.0 + ((i as f32) / (height as f32)) * 180.0;
        let lon = -180.0 + ((j as f32) / (width as f32)) * 360.0;

        // Temperature decreases with latitude, with some variation
        #[allow(clippy::let_and_return)]
        let temp = 273.15
            + 30.0 * (1.0 - lat.abs() / 90.0)
            + 5.0 * (lon / 180.0).sin()
            + 2.0 * (lat / 90.0 * 4.0).sin();
        temp
    });

    let total_elements = data.len();

    println!("Generated {total_elements} climate data points");
    println!(
        "Data range: {:.2} to {:.2} K",
        data.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
        data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
    );

    // Test different compression configurations
    let configs = vec![
        ("JPEG2000 only (CR=10)", EBCCConfig::jpeg2000_only(10.0)),
        ("JPEG2000 only (CR=30)", EBCCConfig::jpeg2000_only(30.0)),
        (
            "Max error bound (0.1K)",
            EBCCConfig::max_absolute_error_bounded(20.0, 0.1),
        ),
        (
            "Max error bound (0.01K)",
            EBCCConfig::max_absolute_error_bounded(20.0, 0.01),
        ),
        (
            "Relative error (0.1%)",
            EBCCConfig::relative_error_bounded(20.0, 0.001),
        ),
    ];

    let original_size = total_elements * std::mem::size_of::<f32>();

    for (name, config) in configs {
        println!("\n--- {name} ---");

        // Compress the data
        let start = std::time::Instant::now();
        let compressed = ebcc_encode(data.view(), &config)?;
        let compress_time = start.elapsed();

        // Decompress the data
        let start = std::time::Instant::now();
        let mut decompressed = Array::zeros(data.dim());
        ebcc_decode_into(&compressed, decompressed.view_mut())?;
        let decompress_time = start.elapsed();

        // Calculate compression metrics
        let compression_ratio = (original_size as f64) / (compressed.len() as f64);
        let compressed_size_mb = (compressed.len() as f64) / (1024.0 * 1024.0);
        let original_size_mb = (original_size as f64) / (1024.0 * 1024.0);

        // Calculate error metrics
        let max_error = data
            .iter()
            .zip(decompressed.iter())
            .map(|(&orig, &decomp)| (orig - decomp).abs())
            .fold(0.0f32, f32::max);

        let mse: f64 = data
            .iter()
            .zip(decompressed.iter())
            .map(|(&orig, &decomp)| f64::from(orig - decomp).powi(2))
            .sum::<f64>()
            / (total_elements as f64);
        let rmse = mse.sqrt();

        // Calculate relative error
        let data_range = data
            .iter()
            .fold(f32::INFINITY, |a, &b| a.min(b))
            .max(data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)))
            - data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_relative_error = max_error / data_range * 100.0;

        println!();
        println!("  Original size:     {original_size_mb:.2} MiB");
        println!("  Compressed size:   {compressed_size_mb:.2} MiB");
        println!("  Compression ratio: {compression_ratio:.2}:1");
        println!(
            "  Compression time:  {:.2} ms",
            compress_time.as_secs_f64() * 1000.0
        );
        println!(
            "  Decompression time: {:.2} ms",
            decompress_time.as_secs_f64() * 1000.0
        );
        println!("  Max error:         {max_error:.4} K");
        println!("  RMSE:              {rmse:.4} K");
        println!("  Max relative error: {max_relative_error:.4}%");
    }

    println!("\nCompression example completed successfully!");

    Ok(())
}
