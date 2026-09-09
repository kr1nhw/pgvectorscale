//! Build script for the vectorscale extension.
//!
//! On Linux, `cargo pgrx test` links a standalone unit-test executable that
//! references PostgreSQL backend symbols from `#[pg_test]` bodies that
//! `--gc-sections` cannot discard (Rust does not emit per-function sections).
//! Those symbols only exist inside the postgres executable, so test builds
//! link weak stub definitions (see `test_stubs.c`).
//!
//! The stubs must NOT reach the extension shared library: inside postgres
//! they would bind locally and shadow the real backend symbols.  Scoping is
//! done in `lib.rs` with a `#[cfg(all(test, target_os = "linux"))] #[link]`
//! extern block, which applies only when the crate is compiled as a test;
//! this build script compiles the stub archive manually (not via
//! `cc::Build::compile`, whose automatic `rustc-link-lib` directive would
//! leak the archive into every artifact) and only publishes its search path.

use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=test_stubs.c");

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "linux" {
        return;
    }

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR not set"));
    let obj = out_dir.join("test_stubs.o");
    let lib = out_dir.join("libtest_stubs.a");

    let mut cc = cc::Build::new();
    cc.file("test_stubs.c").warnings(false);
    let compiler = cc.get_compiler();
    let mut compile = compiler.to_command();
    compile.arg("-c").arg("test_stubs.c").arg("-o").arg(&obj);
    let status = compile
        .status()
        .expect("failed to spawn compiler for test_stubs.c");
    assert!(status.success(), "failed to compile test_stubs.c");

    let mut ar = cc::Build::new().get_archiver();
    ar.arg("crs").arg(&lib).arg(&obj);
    let status = ar.status().expect("failed to spawn archiver for test_stubs");
    assert!(status.success(), "failed to archive test_stubs.o");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
}
