/*
 * Weak stub definitions for PostgreSQL backend symbols.
 *
 * `cargo pgrx test` builds a standalone unit-test executable that links the
 * whole extension rlib.  Rust does not emit per-function sections, so
 * `--gc-sections` cannot discard a `#[pg_test]` body that shares a codegen
 * unit with its (kept) client-side `#[test]` wrapper.  Those bodies reference
 * backend symbols (SPI_*, palloc, ereport internals) that only exist inside
 * the postgres executable; on Linux that is a hard link error, and even with
 * `--unresolved-symbols=ignore-all` the binary fails at load time because
 * `-z now` resolves the GOT eagerly.
 *
 * This file is compiled to an object and linked ONLY into test binaries
 * (build.rs emits `cargo:rustc-link-arg-tests=<obj>` when cfg(test) is set
 * and the target OS is Linux).  Every definition is weak, so a real
 * definition (there is none in the standalone test binary) would win.  The
 * referenced code paths are never executed client-side; if one ever is, the
 * stub aborts loudly instead of silently corrupting state.
 *
 * It is NOT linked into the extension shared library (the library target is
 * not a "test" target), where these symbols must resolve to the real backend.
 */
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

static void test_stub_called(const char *name)
{
    fprintf(stderr,
            "hnswsq test stub invoked for %s: a client-side test executed "
            "PostgreSQL backend code, which is unsupported\n",
            name);
    abort();
}

/* --- data symbols --- */
__attribute__((weak)) void *CurrentMemoryContext = NULL;
__attribute__((weak)) void *TopMemoryContext = NULL;
__attribute__((weak)) void *ErrorContext = NULL;
__attribute__((weak)) void *PG_exception_stack = NULL;
__attribute__((weak)) void *error_context_stack = NULL;
__attribute__((weak)) char *errcontext_msg = NULL;
__attribute__((weak)) uint64_t SPI_processed = 0;
__attribute__((weak)) void *SPI_tuptable = NULL;
__attribute__((weak)) int SPI_result = 0;

/* --- memory management --- */
__attribute__((weak)) void *palloc(size_t size) { (void) size; test_stub_called("palloc"); return NULL; }
__attribute__((weak)) void *palloc0(size_t size) { (void) size; test_stub_called("palloc0"); return NULL; }
__attribute__((weak)) void *repalloc(void *p, size_t size) { (void) p; (void) size; test_stub_called("repalloc"); return NULL; }
__attribute__((weak)) void pfree(void *p) { (void) p; test_stub_called("pfree"); }

/* --- ereport machinery (PG17 signatures) --- */
__attribute__((weak)) int errstart(int elevel, const char *domain) { (void) elevel; (void) domain; test_stub_called("errstart"); return 0; }
__attribute__((weak)) int errcode(int sqlerrcode) { (void) sqlerrcode; test_stub_called("errcode"); return 0; }
__attribute__((weak)) int errmsg(const char *fmt, ...) { (void) fmt; test_stub_called("errmsg"); return 0; }
__attribute__((weak)) int errdetail(const char *fmt, ...) { (void) fmt; test_stub_called("errdetail"); return 0; }
__attribute__((weak)) int errhint(const char *fmt, ...) { (void) fmt; test_stub_called("errhint"); return 0; }
__attribute__((weak)) int errposition(int cursorpos) { (void) cursorpos; test_stub_called("errposition"); return 0; }
__attribute__((weak)) void errfinish(const char *filename, int lineno, const char *funcname)
{
    (void) filename; (void) lineno; (void) funcname;
    test_stub_called("errfinish");
}
__attribute__((weak)) void *CopyErrorData(void) { test_stub_called("CopyErrorData"); return NULL; }
__attribute__((weak)) void FreeErrorData(void *edata) { (void) edata; test_stub_called("FreeErrorData"); }

/* --- SPI --- */
__attribute__((weak)) int SPI_connect(void) { test_stub_called("SPI_connect"); return 1; }
__attribute__((weak)) int SPI_finish(void) { test_stub_called("SPI_finish"); return 1; }
__attribute__((weak)) int SPI_execute(const char *src, int read_only, long tcount)
{
    (void) src; (void) read_only; (void) tcount;
    test_stub_called("SPI_execute");
    return -1;
}
__attribute__((weak)) unsigned int SPI_gettypeid(void *tupdesc, int colnum)
{
    (void) tupdesc; (void) colnum;
    test_stub_called("SPI_gettypeid");
    return 0;
}
