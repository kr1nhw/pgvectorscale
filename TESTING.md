# Testing Guide for pgvectorscale

pgvectorscale has two main types of tests:

1. **Rust Tests** - Using PGRX's `#[pg_test]` framework (can be in any source file)
2. **Python Tests** - Using pytest for multi-process concurrency testing

## Rust Tests

```bash
# Run all Rust tests
cd pgvectorscale && RUST_TEST_THREADS=1 cargo pgrx test pg17

# Run specific test
cd pgvectorscale && RUST_TEST_THREADS=1 cargo pgrx test pg17 test_name
```

`RUST_TEST_THREADS=1` is required: the pgrx framework initializes from the
libtest main thread only (its test binary panics when first invoked from a
worker thread), and the hnswsq vacuum-lifecycle scaffolds need their deleted
rows to be globally dead — a parallel test's open transaction would keep them
alive and make VACUUM skip them.

### Linux: PostgreSQL symbol stubs for the test binary

On Linux, `cargo pgrx test` links a standalone unit-test executable that
references PostgreSQL backend symbols (`SPI_*`, `palloc`, ereport internals)
from `#[pg_test]` bodies.  Rust does not emit per-function sections, so
`--gc-sections` cannot discard a test body that shares a codegen unit with
its (kept) client-side `#[test]` wrapper, and those symbols only exist inside
the postgres executable.

`pgvectorscale/test_stubs.c` (compiled by `build.rs`) provides weak stub
definitions of those symbols.  A `#[cfg(all(test, target_os = "linux"))]
#[link(name = "test_stubs", kind = "static")]` block in `src/lib.rs` links the
stub archive into test binaries only — the extension shared library never
sees the stubs, so inside postgres the real backend symbols are used.  If a
new pgrx/backend symbol is referenced from a test build, the link fails with
an "undefined symbol" list; add the missing symbol to `test_stubs.c` (weak
definitions are inert unless executed, and the stub aborts loudly if a
client-side test ever actually calls backend code).

Run the test suite as a non-root user (`initdb` refuses root):
```bash
useradd -m pgtest
cd pgvectorscale && PGRX_HOME=/root/.pgrx RUST_TEST_THREADS=1 \
    cargo pgrx test pg17 --runas pgtest
```

## Python Tests

```bash
# Setup (creates .venv virtual environment)
make test-python-setup

# Run all Python tests
make test-python

# Run specific categories
pytest tests/ -m concurrency -v    # Multi-process concurrency tests
pytest tests/ -m integration -v    # Basic integration tests

# For PGRX development (custom port)
DB_PORT=28817 ./scripts/run-python-tests.sh
```

### Test Markers

- `@pytest.mark.concurrency` - Multi-process concurrency tests
- `@pytest.mark.integration` - Basic integration tests

### Prerequisites

For PGRX development:
```bash
cd pgvectorscale && cargo pgrx start pg17
cargo pgrx install --features pg17
```