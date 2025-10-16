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
    println!("cargo::rerun-if-changed=SPERR");

    // use cmake to build SPERR
    let mut config = cmake::Config::new("SPERR");
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
    config.cflag("-DEXPERIMENTING");
    config.cxxflag("-DEXPERIMENTING");
    let sperr_out = config.build();

    println!("cargo::rustc-link-search=native={}", sperr_out.display());
    println!(
        "cargo::rustc-link-search=native={}",
        sperr_out.join("lib").display()
    );
    println!(
        "cargo::rustc-link-search=native={}",
        sperr_out.join("lib64").display()
    );
    println!("cargo::rustc-link-lib=static=SPERR");

    let cargo_callbacks = bindgen::CargoCallbacks::new();
    let bindings = bindgen::Builder::default()
        .clang_arg("-x")
        .clang_arg("c++")
        .clang_arg("-std=c++17")
        .clang_arg(format!("-I{}", sperr_out.join("include").display()))
        .header("wrapper.hpp")
        .parse_callbacks(Box::new(cargo_callbacks))
        .allowlist_function("sperr_comp_2d")
        .allowlist_function("sperr_decomp_2d")
        .allowlist_function("sperr_parse_header")
        .allowlist_function("sperr_comp_3d")
        .allowlist_function("sperr_decomp_3d")
        .allowlist_function("sperr_trunc_3d")
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
        .include(sperr_out.join("include"))
        .include(Path::new("SPERR").join("src"))
        .file("lib.cpp")
        .warnings(false);

    build.compile("mySPERR");
}
