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
#![allow(unsafe_code)] // sys-crate
#![allow(clippy::indexing_slicing)] // bindgen tests

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
