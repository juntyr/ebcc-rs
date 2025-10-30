//! [![CI Status]][workflow] [![MSRV]][repo] [![Latest Version]][crates.io]
//! [![Rust Doc Crate]][docs.rs] [![Rust Doc Main]][docs]
//!
//! [CI Status]: https://img.shields.io/github/actions/workflow/status/juntyr/sperr-rs/ci.yml?branch=main
//! [workflow]: https://github.com/juntyr/sperr-rs/actions/workflows/ci.yml?query=branch%3Amain
//!
//! [MSRV]: https://img.shields.io/badge/MSRV-1.82.0-blue
//! [repo]: https://github.com/juntyr/sperr-rs
//!
//! [Latest Version]: https://img.shields.io/crates/v/sperr
//! [crates.io]: https://crates.io/crates/sperr
//!
//! [Rust Doc Crate]: https://img.shields.io/docsrs/sperr
//! [docs.rs]: https://docs.rs/sperr/
//!
//! [Rust Doc Main]: https://img.shields.io/badge/docs-main-blue
//! [docs]: https://juntyr.github.io/sperr-rs/sperr
//!
//! High-level bindigs to the [SPERR] compressor.
//!
//! [SPERR]: https://github.com/NCAR/SPERR

use std::ffi::c_int;

use ndarray::{ArrayView2, ArrayView3, ArrayViewMut2, ArrayViewMut3};

#[derive(Copy, Clone, PartialEq, Debug)]
/// SPERR compression mode / quality control
pub enum CompressionMode<'a> {
    /// Fixed bit-per-pixel rate
    BitsPerPixel {
        /// positive bits-per-pixel
        bpp: f64,
    },
    /// Fixed peak signal-to-noise ratio
    PeakSignalToNoiseRatio {
        /// positive peak signal-to-noise ratio
        psnr: f64,
    },
    /// Fixed point-wise (absolute) error
    PointwiseError {
        /// positive point-wise (absolute) error
        pwe: f64,
    },
    /// Fixed quantisation step
    QuantisationStep {
        /// positive quantisation step
        q: f64,
    },
    /// Quantity of Interest
    QuantityOfInterest {
        /// positive point-wise (absolute) error
        tol: f64,
        /// quantity of interest expression
        qoi: &'a str,
        /// high precision
        high_prec: bool,
    },
}

#[derive(Debug, thiserror::Error)]
/// Errors that can occur during compression and decompression with SPERR
pub enum Error {
    /// one or more parameters is invalid
    #[error("one or more parameters is invalid")]
    InvalidParameter,
    /// compressed data is missing the header
    #[error("compressed data is missing the header")]
    DecompressMissingHeader,
    /// cannot decompress to an array with a different shape
    #[error("cannot decompress to an array with a different shape")]
    DecompressShapeMismatch,
    /// other error
    #[error("other error")]
    Other,
}

impl<'a> CompressionMode<'a> {
    const fn as_mode(self) -> c_int {
        match self {
            Self::BitsPerPixel { .. } => 1,
            Self::PeakSignalToNoiseRatio { .. } => 2,
            Self::PointwiseError { .. } => 3,
            Self::QuantisationStep { .. } => -4,
            Self::QuantityOfInterest { .. } => 5,
        }
    }

    const fn as_quality(self) -> f64 {
        match self {
            Self::BitsPerPixel { bpp: quality }
            | Self::PeakSignalToNoiseRatio { psnr: quality }
            | Self::PointwiseError { pwe: quality }
            | Self::QuantisationStep { q: quality }
            | Self::QuantityOfInterest { tol: quality, .. } => quality,
        }
    }

    const fn as_high_prec(self) -> bool {
        match self {
            Self::BitsPerPixel { .. }
            | Self::PeakSignalToNoiseRatio { .. }
            | Self::PointwiseError { .. }
            | Self::QuantisationStep { .. } => false,
            Self::QuantityOfInterest { high_prec, .. } => high_prec,
        }
    }

    const fn as_qoi(self) -> &'a str {
        match self {
            Self::BitsPerPixel { .. }
            | Self::PeakSignalToNoiseRatio { .. }
            | Self::PointwiseError { .. }
            | Self::QuantisationStep { .. } => "",
            Self::QuantityOfInterest { qoi, .. } => qoi,
        }
    }
}

/// Compress a 2d `src` slice of data with the compression `mode`.
///
/// # Errors
///
/// Errors with
/// - [`Error::InvalidParameter`] if the compression `mode` is invalid
/// - [`Error::Other`] if another error occurs inside SPERR
#[allow(clippy::missing_panics_doc)]
pub fn compress_2d<T: Element>(
    src: ArrayView2<T>,
    mode: CompressionMode,
) -> Result<Vec<u8>, Error> {
    let src = src.as_standard_layout();

    let mut dst = std::ptr::null_mut();
    let mut dst_len = 0;

    assert!(!matches!(mode, CompressionMode::QuantityOfInterest { .. }));

    #[allow(unsafe_code)] // Safety: FFI
    let res = unsafe {
        sperr_sys::sperr_comp_2d(
            src.as_ptr().cast(),
            T::IS_FLOAT.into(),
            src.dim().1,
            src.dim().0,
            mode.as_mode(),
            mode.as_quality(),
            true.into(),
            std::ptr::addr_of_mut!(dst),
            std::ptr::addr_of_mut!(dst_len),
        )
    };

    match res {
        0 => (), // ok
        #[allow(clippy::unreachable)]
        1 => unreachable!("sperr_comp_2d: dst is not pointing to a NULL pointer"),
        2 => return Err(Error::InvalidParameter),
        -1 => return Err(Error::Other),
        #[allow(clippy::panic)]
        _ => panic!("sperr_comp_2d: unknown error kind {res}"),
    }

    #[allow(unsafe_code)] // Safety: dst is initialized by sperr_comp_2d
    let compressed =
        Vec::from(unsafe { std::slice::from_raw_parts(dst.cast_const().cast::<u8>(), dst_len) });

    #[allow(unsafe_code)] // Safety: FFI, dst is allocated by sperr_comp_2d
    unsafe {
        sperr_sys::free_dst(dst);
    }

    Ok(compressed)
}

/// Decompress a 2d SPERR-compressed `compressed` buffer into the `decompressed`
/// array.
///
/// # Errors
///
/// Errors with
/// - [`Error::DecompressMissingHeader`] if the `compressed` buffer does not
///   start with the 10 byte SPERR header
/// - [`Error::DecompressShapeMismatch`] if the `decompressed` array is of a
///   different shape than the header indicates
/// - [`Error::Other`] if another error occurs inside SPERR
#[allow(clippy::missing_panics_doc)]
pub fn decompress_into_2d<T: Element>(
    compressed: &[u8],
    mut decompressed: ArrayViewMut2<T>,
) -> Result<(), Error> {
    let Some((header, compressed)) = compressed.split_at_checked(10) else {
        return Err(Error::DecompressMissingHeader);
    };

    let mut dim_x = 0;
    let mut dim_y = 0;
    let mut dim_z = 0;
    let mut is_float = 0;

    #[allow(unsafe_code)] // Safety: FFI
    unsafe {
        sperr_sys::sperr_parse_header(
            header.as_ptr().cast(),
            std::ptr::addr_of_mut!(dim_x),
            std::ptr::addr_of_mut!(dim_y),
            std::ptr::addr_of_mut!(dim_z),
            std::ptr::addr_of_mut!(is_float),
        );
    }

    if (dim_z, dim_y, dim_x) != (1, decompressed.dim().0, decompressed.dim().1) {
        return Err(Error::DecompressShapeMismatch);
    }

    let mut dst = std::ptr::null_mut();

    #[allow(unsafe_code)] // Safety: FFI
    let res = unsafe {
        sperr_sys::sperr_decomp_2d(
            compressed.as_ptr().cast(),
            compressed.len(),
            T::IS_FLOAT.into(),
            decompressed.dim().1,
            decompressed.dim().0,
            std::ptr::addr_of_mut!(dst),
        )
    };

    match res {
        0 => (), // ok
        #[allow(clippy::unreachable)]
        1 => unreachable!("sperr_decomp_2d: dst is not pointing to a NULL pointer"),
        -1 => return Err(Error::Other),
        #[allow(clippy::panic)]
        _ => panic!("sperr_decomp_2d: unknown error kind {res}"),
    }

    #[allow(unsafe_code)] // Safety: dst is initialized by sperr_decomp_2d
    let dec =
        unsafe { ArrayView2::from_shape_ptr(decompressed.dim(), dst.cast_const().cast::<T>()) };
    decompressed.assign(&dec);

    #[allow(unsafe_code)] // Safety: FFI, dst is allocated by sperr_decomp_2d
    unsafe {
        sperr_sys::free_dst(dst);
    }

    Ok(())
}

/// Compress a 3d `src` volume of data with the compression `mode` using the
/// preferred `chunks`.
///
/// # Errors
///
/// Errors with
/// - [`Error::InvalidParameter`] if the compression `mode` is invalid
/// - [`Error::Other`] if another error occurs inside SPERR
#[allow(clippy::missing_panics_doc)]
pub fn compress_3d<T: Element>(
    src: ArrayView3<T>,
    mode: CompressionMode,
    chunks: (usize, usize, usize),
) -> Result<Vec<u8>, Error> {
    let src = src.as_standard_layout();

    let mut dst = std::ptr::null_mut();
    let mut dst_len = 0;

    let mut qoi = Vec::from(mode.as_qoi().as_bytes());
    qoi.push(b'\0');

    #[allow(unsafe_code)] // Safety: FFI
    let res = unsafe {
        sperr_sys::sperr_comp_3d(
            src.as_ptr().cast(),
            T::IS_FLOAT.into(),
            src.dim().2,
            src.dim().1,
            src.dim().0,
            chunks.2,
            chunks.1,
            chunks.0,
            mode.as_mode(),
            mode.as_quality(),
            0,
            std::ptr::addr_of_mut!(dst),
            std::ptr::addr_of_mut!(dst_len),
            qoi.as_ptr().cast(),
            mode.as_high_prec(),
        )
    };

    match res {
        0 => (), // ok
        #[allow(clippy::unreachable)]
        1 => unreachable!("sperr_comp_3d: dst is not pointing to a NULL pointer"),
        2 => return Err(Error::InvalidParameter),
        -1 => return Err(Error::Other),
        #[allow(clippy::panic)]
        _ => panic!("sperr_comp_3d: unknown error kind {res}"),
    }

    #[allow(unsafe_code)] // Safety: dst is initialized by sperr_comp_3d
    let compressed =
        Vec::from(unsafe { std::slice::from_raw_parts(dst.cast_const().cast::<u8>(), dst_len) });

    #[allow(unsafe_code)] // Safety: FFI, dst is allocated by sperr_comp_3d
    unsafe {
        sperr_sys::free_dst(dst);
    }

    Ok(compressed)
}

/// Decompress a 3d SPERR-compressed `compressed` buffer into the `decompressed`
/// array.
///
/// # Errors
///
/// Errors with
/// - [`Error::DecompressShapeMismatch`] if the `decompressed` array is of a
///   different shape than the SPERR header indicates
/// - [`Error::Other`] if another error occurs inside SPERR
#[allow(clippy::missing_panics_doc)]
pub fn decompress_into_3d<T: Element>(
    compressed: &[u8],
    mut decompressed: ArrayViewMut3<T>,
) -> Result<(), Error> {
    let mut dim_x = 0;
    let mut dim_y = 0;
    let mut dim_z = 0;
    let mut is_float = 0;

    #[allow(unsafe_code)] // Safety: FFI
    unsafe {
        sperr_sys::sperr_parse_header(
            compressed.as_ptr().cast(),
            std::ptr::addr_of_mut!(dim_x),
            std::ptr::addr_of_mut!(dim_y),
            std::ptr::addr_of_mut!(dim_z),
            std::ptr::addr_of_mut!(is_float),
        );
    }

    if (dim_z, dim_y, dim_x)
        != (
            decompressed.dim().0,
            decompressed.dim().1,
            decompressed.dim().2,
        )
    {
        return Err(Error::DecompressShapeMismatch);
    }

    let mut dst = std::ptr::null_mut();

    #[allow(unsafe_code)] // Safety: FFI
    let res = unsafe {
        sperr_sys::sperr_decomp_3d(
            compressed.as_ptr().cast(),
            compressed.len(),
            T::IS_FLOAT.into(),
            0,
            std::ptr::addr_of_mut!(dim_x),
            std::ptr::addr_of_mut!(dim_y),
            std::ptr::addr_of_mut!(dim_z),
            std::ptr::addr_of_mut!(dst),
        )
    };

    match res {
        0 => (), // ok
        #[allow(clippy::unreachable)]
        1 => unreachable!("sperr_decomp_3d: dst is not pointing to a NULL pointer"),
        -1 => return Err(Error::Other),
        #[allow(clippy::panic)]
        _ => panic!("sperr_decomp_3d: unknown error kind {res}"),
    }

    #[allow(unsafe_code)] // Safety: dst is initialized by sperr_decomp_3d
    let dec =
        unsafe { ArrayView3::from_shape_ptr(decompressed.dim(), dst.cast_const().cast::<T>()) };
    decompressed.assign(&dec);

    #[allow(unsafe_code)] // Safety: FFI, dst is allocated by sperr_decomp_3d
    unsafe {
        sperr_sys::free_dst(dst);
    }

    Ok(())
}

/// Marker trait for element types that can be compressed with SPERR
pub trait Element: sealed::Element {}

impl Element for f32 {}
impl sealed::Element for f32 {
    const IS_FLOAT: bool = true;
}

impl Element for f64 {}
impl sealed::Element for f64 {
    const IS_FLOAT: bool = false;
}

mod sealed {
    pub trait Element: Copy {
        const IS_FLOAT: bool;
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use ndarray::{linspace, logspace, Array1, Array3};

    use super::*;

    fn compress_decompress(mode: CompressionMode) {
        let data = linspace(1.0, 10.0, 128 * 128 * 128).collect::<Array1<f64>>()
            + logspace(2.0, 0.0, 5.0, 128 * 128 * 128)
                .rev()
                .collect::<Array1<f64>>();
        let data: Array3<f64> = data
            .into_shape_clone((128, 128, 128))
            .expect("create test data array");

        let compressed =
            compress_3d(data.view(), mode, (64, 64, 64)).expect("compression should not fail");

        let mut decompressed = Array3::<f64>::zeros(data.dim());
        decompress_into_3d(compressed.as_slice(), decompressed.view_mut())
            .expect("decompression should not fail");

        let data: Array3<f64> = Array3::zeros((64, 64, 1));

        let compressed =
            compress_3d(data.view(), mode, (256, 256, 256)).expect("compression should not fail");

        let mut decompressed = Array3::<f64>::zeros(data.dim());
        decompress_into_3d(compressed.as_slice(), decompressed.view_mut())
            .expect("decompression should not fail");
    }

    #[test]
    fn compress_decompress_bpp() {
        compress_decompress(CompressionMode::BitsPerPixel { bpp: 2.0 });
    }

    #[test]
    fn compress_decompress_psnr() {
        compress_decompress(CompressionMode::PeakSignalToNoiseRatio { psnr: 30.0 });
    }

    #[test]
    fn compress_decompress_pwe() {
        compress_decompress(CompressionMode::PointwiseError { pwe: 0.1 });
    }

    #[test]
    fn compress_decompress_q() {
        compress_decompress(CompressionMode::QuantisationStep { q: 3.0 });
    }

    #[test]
    fn compress_decompress_qoi() {
        compress_decompress(CompressionMode::QuantityOfInterest {
            tol: 0.1,
            qoi: "x^2",
            high_prec: false,
        });

        compress_decompress(CompressionMode::QuantityOfInterest {
            tol: 0.1,
            qoi: "log(x,10)",
            high_prec: false,
        });
    }
}
