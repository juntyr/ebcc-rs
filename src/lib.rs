//! [![CI Status]][workflow] [![MSRV]][repo] [![Latest Version]][crates.io]
//! [![Rust Doc Crate]][docs.rs] [![Rust Doc Main]][docs]
//!
//! [CI Status]: https://img.shields.io/github/actions/workflow/status/juntyr/ebcc-rs/ci.yml?branch=main
//! [workflow]: https://github.com/juntyr/ebcc-rs/actions/workflows/ci.yml?query=branch%3Amain
//!
//! [MSRV]: https://img.shields.io/badge/MSRV-1.82.0-blue
//! [repo]: https://github.com/juntyr/ebcc-rs
//!
//! [Latest Version]: https://img.shields.io/crates/v/ebcc
//! [crates.io]: https://crates.io/crates/ebcc
//!
//! [Rust Doc Crate]: https://img.shields.io/docsrs/ebcc
//! [docs.rs]: https://docs.rs/ebcc/
//!
//! [Rust Doc Main]: https://img.shields.io/badge/docs-main-blue
//! [docs]: https://juntyr.github.io/ebcc-rs/ebcc
//!
//! High-level bindigs to the [EBCC] compressor.
//!
//! [EBCC]: https://github.com/spcl/EBCC

mod codec;
mod config;
mod error;

pub use codec::{ebcc_decode_into, ebcc_encode};
pub use config::{EBCCConfig, ResidualType};
pub use error::{EBCCError, EBCCResult};
