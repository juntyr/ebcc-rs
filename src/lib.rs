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
//! High-level bindigs to the [QPET-SPERR] compressor.
//!
//! [QPET-SPERR]: https://github.com/JLiu-1/QPET-Artifact/tree/sperr_qpet_revision

use std::num::NonZeroUsize;

use ndarray::{ArrayView3, ArrayViewMut3};

#[derive(Copy, Clone, PartialEq, Debug)]
#[non_exhaustive]
/// QPET-SPERR compression mode / quality control
pub enum CompressionMode<'a> {
    /// Symbolic Quantity of Interest
    SymbolicQuantityOfInterest {
        /// quantity of interest expression
        qoi: &'a str,
        /// 3D block size (z,y,x) over which the quantity of interest errors
        /// are averaged, 1x1x1 for pointwise
        qoi_block_size: (NonZeroUsize, NonZeroUsize, NonZeroUsize),
        /// positive (pointwise) absolute error bound over the quantity of
        /// interest
        qoi_pwe: f64,
        /// optional positive pointwise absolute error bound over the data
        data_pwe: Option<f64>,
        /// positive quantity of interest k parameter (3.0 is a good default)
        qoi_k: f64,
        /// high precision mode for SPERR, useful for small error bounds
        high_prec: bool,
    },
}

#[derive(Debug, thiserror::Error)]
/// Errors that can occur during compression and decompression with QPET-SPERR
pub enum Error {
    /// one or more parameters is invalid
    #[error("one or more parameters is invalid")]
    InvalidParameter,
    /// cannot decompress to an array with a different shape
    #[error("cannot decompress to an array with a different shape")]
    DecompressShapeMismatch,
    /// other error
    #[error("other error")]
    Other,
}

/// Compress a 3d `src` volume of data with the compression `mode` using the
/// preferred `chunks`.
///
/// The compressed output can be decompressed with QPET-SPERR or SPERR.
///
/// # Errors
///
/// Errors with
/// - [`Error::InvalidParameter`] if the compression `mode` is invalid
/// - [`Error::Other`] if another error occurs inside QPET-SPERR
#[allow(clippy::missing_panics_doc)]
pub fn compress_3d<T: Element>(
    src: ArrayView3<T>,
    mode: CompressionMode,
    chunks: (usize, usize, usize),
) -> Result<Vec<u8>, Error> {
    let src = src.as_standard_layout();

    let mut dst = std::ptr::null_mut();
    let mut dst_len = 0;

    let CompressionMode::SymbolicQuantityOfInterest {
        qoi,
        qoi_block_size,
        qoi_pwe,
        data_pwe,
        qoi_k,
        high_prec,
    } = mode;

    let mut qoi = Vec::from(qoi.as_bytes());
    qoi.push(b'\0');
    let qoi = qoi;

    #[allow(unsafe_code)] // Safety: FFI
    let res = unsafe {
        qpet_sperr_sys::qpet_sperr_comp_3d(
            src.as_ptr().cast(),
            T::IS_FLOAT.into(),
            src.dim().2,
            src.dim().1,
            src.dim().0,
            chunks.2,
            chunks.1,
            chunks.0,
            data_pwe.unwrap_or(f64::MAX),
            0,
            std::ptr::addr_of_mut!(dst),
            std::ptr::addr_of_mut!(dst_len),
            qoi.as_ptr().cast(),
            qoi_pwe,
            qoi_block_size.2.get(),
            qoi_block_size.1.get(),
            qoi_block_size.0.get(),
            qoi_k,
            high_prec,
        )
    };

    match res {
        0 => (), // ok
        #[allow(clippy::unreachable)]
        1 => unreachable!("qpet_sperr_comp_3d: dst is not pointing to a NULL pointer"),
        2 => return Err(Error::InvalidParameter),
        -1 => return Err(Error::Other),
        #[allow(clippy::panic)]
        _ => panic!("qpet_sperr_comp_3d: unknown error kind {res}"),
    }

    #[allow(unsafe_code)] // Safety: dst is initialized by qpet_sperr_comp_3d
    let compressed =
        Vec::from(unsafe { std::slice::from_raw_parts(dst.cast_const().cast::<u8>(), dst_len) });

    #[allow(unsafe_code)] // Safety: FFI, dst is allocated by qpet_sperr_comp_3d
    unsafe {
        qpet_sperr_sys::free_dst(dst);
    }

    Ok(compressed)
}

/// Decompress a 3d (QPET-)SPERR-compressed `compressed` buffer into the
/// `decompressed` array.
///
/// # Errors
///
/// Errors with
/// - [`Error::DecompressShapeMismatch`] if the `decompressed` array is of a
///   different shape than the SPERR header indicates
/// - [`Error::Other`] if another error occurs inside QPET-SPERR
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
        qpet_sperr_sys::sperr_parse_header(
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
        qpet_sperr_sys::sperr_decomp_3d(
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
        qpet_sperr_sys::free_dst(dst);
    }

    Ok(())
}

/// Marker trait for element types that can be compressed with QPET-SPERR
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

    const ONE: NonZeroUsize = NonZeroUsize::MIN;
    const THREE: NonZeroUsize = ONE.saturating_add(2);

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
    fn compress_decompress_square() {
        compress_decompress(CompressionMode::SymbolicQuantityOfInterest {
            qoi: "x^2",
            qoi_block_size: (ONE, ONE, ONE),
            qoi_pwe: 0.1,
            data_pwe: None,
            qoi_k: 3.0,
            high_prec: false,
        });

        // compress_decompress(CompressionMode::SymbolicQuantityOfInterest {
        //     qoi: "x^2",
        //     qoi_block_size: (THREE, THREE, THREE),
        //     qoi_pwe: 0.1,
        //     data_pwe: None,
        //     qoi_k: 3.0,
        //     high_prec: false,
        // });
    }

    #[test]
    fn compress_decompress_log10() {
        compress_decompress(CompressionMode::SymbolicQuantityOfInterest {
            qoi: "log(x,10)",
            qoi_block_size: (ONE, ONE, ONE),
            qoi_pwe: 0.1,
            data_pwe: None,
            qoi_k: 3.0,
            high_prec: true,
        });

        // compress_decompress(CompressionMode::SymbolicQuantityOfInterest {
        //     qoi: "log(x,10)",
        //     qoi_block_size: (THREE, THREE, THREE),
        //     qoi_pwe: 0.1,
        //     data_pwe: None,
        //     qoi_k: 3.0,
        //     high_prec: true,
        // });
    }
}
