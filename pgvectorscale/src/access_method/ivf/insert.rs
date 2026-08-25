//! IVF index insert implementation.

use pgrx::*;

/// Insert a tuple into the IVF index.
#[pg_guard]
pub unsafe extern "C-unwind" fn aminsert(
    _index: pg_sys::Relation,
    _values: *mut pg_sys::Datum,
    _isnull: *mut bool,
    _heap_tid: pg_sys::ItemPointer,
    _heap: pg_sys::Relation,
    _check_unique: pg_sys::IndexUniqueCheck::Type,
    _index_unchanged: bool,
    _index_info: *mut pg_sys::IndexInfo,
) -> bool {
    // TODO: Implement insert logic
    false
}
