#![expect(missing_docs)]
#![expect(clippy::expect_used)]

// Adapted from sz3-sys's build script by Robin Ole Heinemann, licensed under GPL-3.0-only.

use std::{
    env,
    path::{Path, PathBuf},
};

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=lib.cpp");
    println!("cargo::rerun-if-changed=wrapper.hpp");
    println!("cargo::rerun-if-changed=QPET-Artifact");

    let out_dir = env::var("OUT_DIR")
        .map(PathBuf::from)
        .expect("missing OUT_DIR");

    let gmp_root = Path::new("/opt/homebrew/opt/gmp");

    let zstd_root = env::var("DEP_ZSTD_ROOT")
        .map(PathBuf::from)
        .expect("missing zstd dependency");

    // use cmake to build QPET-SPERR
    let mut config = cmake::Config::new("QPET-Artifact");
    // < symengine config
    config.define("BUILD_SHARED_LIBS", "OFF");
    config.define("BUILD_BENCHMARKS", "OFF");
    config.define("BUILD_TESTS", "OFF");
    // > symengine config
    // < QPET-SPERR config
    config.define("BUILD_SHARED_LIBS", "OFF");
    config.define("BUILD_UNIT_TESTS", "OFF");
    config.define("BUILD_CLI_UTILITIES", "OFF");
    config.define(
        "USE_OMP",
        if cfg!(feature = "openmp") {
            "ON"
        } else {
            "OFF"
        },
    );
    // > QPET-SPERR config
    config.cflag(format!("-I{}", gmp_root.join("include").display()));
    config.cxxflag(format!("-I{}", gmp_root.join("include").display()));
    config.cflag(format!(
        "-I{}",
        out_dir.join("build").join("symengine").display()
    ));
    config.cxxflag(format!(
        "-I{}",
        out_dir.join("build").join("symengine").display()
    ));
    config.cflag(format!("-I{}", zstd_root.join("include").display()));
    config.cxxflag(format!("-I{}", zstd_root.join("include").display()));
    let qpet_sperr_out = config.build();

    println!(
        "cargo::rustc-link-search=native={}",
        qpet_sperr_out.display()
    );
    println!(
        "cargo::rustc-link-search=native={}",
        qpet_sperr_out.join("lib").display()
    );
    println!(
        "cargo::rustc-link-search=native={}",
        qpet_sperr_out.join("lib64").display()
    );
    println!("cargo::rustc-link-lib=static=QPET-SPERR");
    println!("cargo::rustc-link-lib=static=symengine");
    // TODO: println!("cargo::rustc-link-lib=static=teuchos"); // only in debug mode
    println!("cargo::rustc-link-lib=static=zstd");
    // TODO: build gmp ourselves
    println!(
        "cargo::rustc-link-search=native={}",
        gmp_root.join("lib").display()
    );
    // TODO: once we build gmp ourselves, always link it statically
    println!("cargo::rustc-link-lib=gmp");

    let cargo_callbacks = bindgen::CargoCallbacks::new();
    let bindings = bindgen::Builder::default()
        .clang_arg("-x")
        .clang_arg("c++")
        .clang_arg("-std=c++17")
        .clang_arg(format!("-I{}", qpet_sperr_out.join("include").display()))
        .clang_arg(format!("-I{}", zstd_root.join("include").display()))
        .header("wrapper.hpp")
        .parse_callbacks(Box::new(cargo_callbacks))
        .allowlist_function("sperr_parse_header")
        .allowlist_function("qpet_sperr_comp_3d")
        .allowlist_function("sperr_decomp_3d")
        .allowlist_function("free_dst")
        // MSRV 1.82
        .rust_target(match bindgen::RustTarget::stable(82, 0) {
            Ok(target) => target,
            #[expect(clippy::panic)]
            Err(err) => panic!("{err}"),
        })
        .generate()
        .expect("Unable to generate bindings");

    let out_path =
        PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR should be set in a build script"));
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");

    let mut build = cc::Build::new();

    build
        .cpp(true)
        .std("c++17")
        .include(qpet_sperr_out.join("include"))
        .include(zstd_root.join("include"))
        .include(Path::new("QPET-Artifact").join("src"))
        .file("lib.cpp")
        .warnings(false);

    build.compile("myQPETSPERR");
}
