#![expect(missing_docs)]
#![expect(clippy::expect_used)]

// Adapted from sz3-sys's build script by Robin Ole Heinemann, licensed under GPL-3.0-only.

use std::{
    env,
    path::{Path, PathBuf},
};

#[allow(clippy::too_many_lines)]
fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=lib.cpp");
    println!("cargo::rerun-if-changed=wrapper.hpp");
    println!("cargo::rerun-if-changed=SPERR");

    let out_dir = env::var("OUT_DIR")
        .map(PathBuf::from)
        .expect("missing OUT_DIR");

    let gmp_root = env::var("DEP_GMP_OUT_DIR")
        .map(PathBuf::from)
        .expect("missing gmp dependency");

    let zstd_root = env::var("DEP_ZSTD_ROOT")
        .map(PathBuf::from)
        .expect("missing zstd dependency");

    // use cmake to build QPET-SPERR
    let mut config = cmake::Config::new("SPERR");
    if let Ok(ar) = env::var("AR") {
        config.define("CMAKE_AR", ar);
    }
    if let Ok(ld) = env::var("LD") {
        config.define("CMAKE_LINKER", ld);
    }
    if let Ok(nm) = env::var("NM") {
        config.define("CMAKE_NM", nm);
    }
    if let Ok(objdump) = env::var("OBJDUMP") {
        config.define("CMAKE_OBJDUMP", objdump);
    }
    if let Ok(ranlib) = env::var("RANLIB") {
        config.define("CMAKE_RANLIB", ranlib);
    }
    if let Ok(strip) = env::var("STRIP") {
        config.define("CMAKE_STRIP", strip);
    }
    // < symengine config
    config.define("BUILD_SHARED_LIBS", "OFF");
    config.define("BUILD_BENCHMARKS", "OFF");
    config.define("BUILD_TESTS", "OFF");
    config.define("GMP_LIBRARY", gmp_root.join("lib"));
    config.define("GMP_INCLUDE_DIR", gmp_root.join("include"));
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
    // < include paths
    let gmp_include = gmp_root.join("include");
    let gmp_include_display = gmp_include.display();
    config.cflag(format!("-I{gmp_include_display}"));
    config.cxxflag(format!("-I{gmp_include_display}"));
    let symengine_include = out_dir.join("build").join("symengine");
    let symengine_include_display = symengine_include.display();
    config.cflag(format!("-I{symengine_include_display}"));
    config.cxxflag(format!("-I{symengine_include_display}"));
    let zstd_include = zstd_root.join("include");
    let zstd_include_display = zstd_include.display();
    config.cflag(format!("-I{zstd_include_display}"));
    config.cxxflag(format!("-I{zstd_include_display}"));
    if config.get_profile() == "Debug" {
        let teuchos_include = out_dir
            .join("build")
            .join("symengine")
            .join("symengine")
            .join("utilities")
            .join("teuchos");
        let teuchos_include_display = teuchos_include.display();
        config.cflag(format!("-I{teuchos_include_display}"));
        config.cxxflag(format!("-I{teuchos_include_display}"));
    }
    // > include paths
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
    if config.get_profile() == "Debug" {
        println!("cargo::rustc-link-lib=static=teuchos");
    }
    println!("cargo::rustc-link-lib=static=zstd");
    println!(
        "cargo::rustc-link-search=native={}",
        gmp_root.join("lib").display()
    );
    println!("cargo::rustc-link-lib=static=gmp");

    let qpet_sperr_include = qpet_sperr_out.join("include");
    let qpet_sperr_include_display = qpet_sperr_include.display();

    let cargo_callbacks = bindgen::CargoCallbacks::new();
    let bindings = bindgen::Builder::default()
        .clang_arg("-x")
        .clang_arg("c++")
        .clang_arg("-std=c++17")
        .clang_arg(format!("-I{qpet_sperr_include_display}"))
        .clang_arg(format!("-I{zstd_include_display}"))
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
        .include(qpet_sperr_include)
        .include(zstd_include)
        .include(Path::new("SPERR").join("src"))
        .file("lib.cpp")
        .warnings(false);

    build.compile("myQPETSPERR");
}
