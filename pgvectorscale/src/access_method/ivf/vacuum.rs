//! IVF index vacuum implementation.

use pgrx::*;

/// Bulk delete tuples from the IVF index.
#[pg_guard]
pub unsafe extern "C-unwind" fn ambulkdelete(
    _info: *mut pg_sys::IndexVacuumInfo,
    _stats: *mut pg_sys::IndexBulkDeleteResult,
    _callback: pg_sys::IndexBulkDeleteCallback,
    _callback_state: *mut std::os::raw::c_void,
) -> *mut pg_sys::IndexBulkDeleteResult {
    // TODO: Implement bulk delete
    _stats
}

/// Cleanup after vacuum.
#[pg_guard]
pub unsafe extern "C-unwind" fn amvacuumcleanup(
    _info: *mut pg_sys::IndexVacuumInfo,
    _stats: *mut pg_sys::IndexBulkDeleteResult,
) -> *mut pg_sys::IndexBulkDeleteResult {
    // TODO: Implement vacuum cleanup
    _stats
}
