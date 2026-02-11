#![expect(missing_docs)]
#![expect(clippy::expect_used)]

use std::env;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=EBCC");

    let out_dir = env::var("OUT_DIR")
        .map(PathBuf::from)
        .expect("missing OUT_DIR");

    let target = env::var("TARGET").expect("missing TARGET");
    let no_threads = target == "wasm32-unknown-unknown" || target.starts_with("wasm32-wasi");

    let ebcc_src = Path::new("EBCC").join("src");

    // Build the static library using CMake from src/ directory
    let mut config = cmake::Config::new(&ebcc_src);
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
    // < openjp2 config
    if no_threads {
        config.define("OPJ_USE_THREAD", "OFF");
    }
    // > openjp2 config
    // < zstd config
    if no_threads {
        config.define("ZSTD_MULTITHREAD_SUPPORT", "OFF");
    }
    // > zstd config
    let ebcc_out = config.build();

    // Tell cargo to look for libraries in the CMake build directory
    println!(
        "cargo::rustc-link-search=native={}",
        ebcc_out.join("lib").display()
    );
    println!(
        "cargo::rustc-link-search=native={}",
        ebcc_out.join("lib64").display()
    );

    // Link against the static EBCC library and its dependencies
    println!("cargo::rustc-link-lib=static=ebcc");
    println!("cargo::rustc-link-lib=static=openjp2");
    println!("cargo::rustc-link-lib=static=zstd");

    let bindings = bindgen::Builder::default()
        .header(format!("{}", ebcc_src.join("ebcc_codec.h").display()))
        .clang_arg(format!("-I{}", ebcc_src.display()))
        // Tell bindgen to generate bindings for these types and functions
        .allowlist_type("codec_config_t")
        .allowlist_type("residual_t")
        .allowlist_function("ebcc_encode")
        .allowlist_function("ebcc_decode")
        .allowlist_function("free_buffer")
        // Use constified enum module for better enum handling
        .constified_enum_module("residual_t")
        // Generate comments from C headers
        .generate_comments(true)
        // Use core instead of std for no_std compatibility
        .use_core()
        // Generate layout tests
        .layout_tests(true)
        // Don't generate recursively for system headers
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // MSRV 1.82
        .rust_target(match bindgen::RustTarget::stable(82, 0) {
            Ok(target) => target,
            #[expect(clippy::panic)]
            Err(err) => panic!("{err}"),
        })
        .generate()
        .expect("Unable to generate bindings");

    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
