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

pub const EBCC_CHUNKING_HEADER_MAGIC: &[u8] = const {
    let magic: &[u8] = bindings::EBCC_CHUNKING_HEADER_MAGIC;

    let [magic @ .., b'\0'] = magic else {
        panic!("EBCC_CHUNKING_HEADER_MAGIC is not a null-terminated cstr");
    };

    magic
};

#[expect(clippy::manual_assert)]
pub const EBCC_CHUNKING_HEADER_VERSION: u32 = const {
    let _: c_uint = bindings::EBCC_CHUNKING_HEADER_VERSION;

    if size_of::<c_uint>() > size_of::<u32>() {
        panic!("EBCC_CHUNKING_HEADER_VERSION might not fit into u32");
    }

    #[allow(clippy::unnecessary_cast)]
    {
        bindings::EBCC_CHUNKING_HEADER_VERSION as u32
    }
};

#[expect(clippy::manual_assert)]
pub const EBCC_MAX_INTERNAL_IMAGE_DIM: usize = const {
    let _: c_uint = bindings::EBCC_MAX_INTERNAL_IMAGE_DIM;

    if size_of::<c_uint>() > size_of::<usize>() {
        panic!("EBCC_MAX_INTERNAL_IMAGE_DIM might not fit into usize");
    }

    bindings::EBCC_MAX_INTERNAL_IMAGE_DIM as usize
};

#[expect(clippy::manual_assert)]
pub const EBCC_MIN_INTERNAL_IMAGE_DIM: usize = const {
    let _: c_uint = bindings::EBCC_MIN_INTERNAL_IMAGE_DIM;

    if size_of::<c_uint>() > size_of::<usize>() {
        panic!("EBCC_MIN_INTERNAL_IMAGE_DIM might not fit into usize");
    }

    bindings::EBCC_MIN_INTERNAL_IMAGE_DIM as usize
};

#[expect(clippy::manual_assert)]
pub const EBCC_NDIMS: usize = const {
    let _: c_uint = bindings::NDIMS;

    if size_of::<c_uint>() > size_of::<usize>() {
        panic!("NDIMS might not fit into usize");
    }

    bindings::NDIMS as usize
};
