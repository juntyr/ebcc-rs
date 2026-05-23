//! [![CI Status]][workflow] [![MSRV]][repo] [![Latest Version]][crates.io]
//! [![Rust Doc Crate]][docs.rs] [![Rust Doc Main]][docs]
//!
//! [CI Status]: https://img.shields.io/github/actions/workflow/status/juntyr/ebcc-rs/ci.yml?branch=main
//! [workflow]: https://github.com/juntyr/ebcc-rs/actions/workflows/ci.yml?query=branch%3Amain
//!
//! [MSRV]: https://img.shields.io/badge/MSRV-1.82.0-blue
//! [repo]: https://github.com/juntyr/ebcc-rs
//!
//! [Latest Version]: https://img.shields.io/crates/v/ebcc-sys
//! [crates.io]: https://crates.io/crates/ebcc-sys
//!
//! [Rust Doc Crate]: https://img.shields.io/docsrs/ebcc-sys
//! [docs.rs]: https://docs.rs/ebcc-sys/
//!
//! [Rust Doc Main]: https://img.shields.io/badge/docs-main-blue
//! [docs]: https://juntyr.github.io/ebcc-rs/ebcc_sys
//!
//! Low-level bindigs to the [EBCC] compressor.
//!
//! [EBCC]: https://github.com/spcl/EBCC

#![allow(missing_docs)] // bindgen

use std::ffi::c_uint;

#[allow(unsafe_code)] // sys-crate
#[allow(clippy::indexing_slicing)] // bindgen tests
mod bindings {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

pub use bindings::{
    codec_config_t, ebcc_decode, ebcc_decode_chunking, ebcc_encode, ebcc_encode_chunking,
    ebcc_encode_chunking_compat, free_buffer, residual_t,
};

#[expect(clippy::manual_assert)]
pub const EBCC_NDIMS: usize = const {
    let _: c_uint = bindings::NDIMS;

    if size_of::<c_uint>() > size_of::<usize>() {
        panic!("NDIMS might not fit into usize");
    }

    bindings::NDIMS as usize
};

#[expect(clippy::manual_assert)]
pub const EBCC_MAX_INTERNAL_IMAGE_DIM: usize = const {
    let _: c_uint = bindings::EBCC_MAX_INTERNAL_IMAGE_DIM;

    if size_of::<c_uint>() > size_of::<usize>() {
        panic!("EBCC_MAX_INTERNAL_IMAGE_DIM might not fit into usize");
    }

    bindings::EBCC_MAX_INTERNAL_IMAGE_DIM as usize
};
