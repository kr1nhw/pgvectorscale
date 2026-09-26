//! Relptr for the hnswsq port — pgvector's `HnswPtr` reimplemented.
//!
//! pgvector's `HnswPtr` is a union of an absolute pointer and a PostgreSQL
//! `relptr` (an offset from a base address).  `relptr_store`/`relptr_access`
//! are `static inline` in `utils/relptr.h` and therefore absent from the pgrx
//! bindings, so they are reimplemented here as trivial offset arithmetic —
//! exactly the semantics of `HnswPtrAccess`/`HnswPtrStore` in the reference
//! (`base == NULL ? hp.ptr : relptr_access(base, hp.relptr)`).
//!
//! Every pointer that can cross shared memory (parallel builds) goes through
//! this type; backend-private graphs use the same code path with `base == NULL`
//! (absolute pointers), which is also how pgvector keeps one implementation
//! for both.  All the arithmetic lives in this one module: this is the single
//! place where a port rots, so every accessor has a unit test below.

use pgrx::pg_sys;

/// A pointer that is either absolute (`base == NULL`) or an offset from
/// `base` (a relptr).  Matches pgvector's `HnswPtrDeclare` unions: the two
/// variants share storage, and the interpretation is chosen by the `base`
/// argument of each accessor, never by the value itself.
#[repr(C)]
#[derive(Clone, Copy)]
pub union HnswPtr {
    /// Absolute pointer (backend-private graphs).
    pub ptr: *mut u8,
    /// Byte offset from the shared area base (parallel builds).
    pub relptr: usize,
}

/// `HnswPtrAccess`: resolve `hp` against `base` (null base = absolute).
///
/// # Safety
/// The caller must pair `base` with the convention `hp` was stored with: an
/// offset stored relative to one base is meaningless against another.
#[inline]
pub unsafe fn access<T>(base: *mut u8, hp: HnswPtr) -> *mut T {
    if base.is_null() {
        hp.ptr.cast()
    } else {
        base.add(hp.relptr).cast()
    }
}

/// `HnswPtrStore`: write `value` into `hp` under the given `base` convention.
/// A null value stores relptr offset 0 (PostgreSQL's `relptr_store` checks
/// for NULL before computing the offset).
///
/// # Safety
/// As [`access`]; additionally `value` must point into the same allocation as
/// `base` when `base` is non-null (the offset is computed with `offset_from`).
#[inline]
pub unsafe fn store<T>(base: *mut u8, hp: &mut HnswPtr, value: *mut T) {
    if base.is_null() {
        hp.ptr = value.cast();
    } else if value.is_null() {
        hp.relptr = 0;
    } else {
        hp.relptr = (value as *mut u8).offset_from(base) as usize;
    }
}

/// `HnswPtrIsNull`.
///
/// # Safety
/// See [`access`].
#[inline]
pub unsafe fn is_null(base: *mut u8, hp: HnswPtr) -> bool {
    if base.is_null() {
        hp.ptr.is_null()
    } else {
        hp.relptr == 0
    }
}

/// `HnswPtrEqual`.
///
/// # Safety
/// See [`access`].
#[inline]
pub unsafe fn equal(base: *mut u8, hp1: HnswPtr, hp2: HnswPtr) -> bool {
    if base.is_null() {
        hp1.ptr == hp2.ptr
    } else {
        hp1.relptr == hp2.relptr
    }
}

/// `HnswPtrPointer` — the absolute form, for the tie-breaker ordering of
/// candidate lists in backend-private graphs.
#[inline]
pub fn pointer(hp: HnswPtr) -> *mut u8 {
    // SAFETY: reading the union field is always sound.
    unsafe { hp.ptr }
}

/// `HnswPtrOffset` — the relptr form, for the tie-breaker ordering of
/// candidate lists in shared-memory graphs.
#[inline]
pub fn offset(hp: HnswPtr) -> usize {
    // SAFETY: reading the union field is always sound.
    unsafe { hp.relptr }
}

/// pgvector's `relptr_offset` for PG < 14.5 would subtract one; the port
/// targets PG >= 14.5 offset semantics (offset 0 = null, values are exact
/// byte offsets), so no adjustment is needed.  This constant documents that.
pub const _NO_RELPTR_ADJUSTMENT: () = ();

#[cfg(test)]
mod tests {
    use super::*;

    /// Store/access round-trips under both conventions, including null and
    /// the tie-breaker accessors.
    #[test]
    fn test_hnsw_ptr_roundtrip() {
        let mut region = [0u8; 256];
        let base = region.as_mut_ptr();

        let mut hp = HnswPtr {
            ptr: std::ptr::null_mut(),
        };
        unsafe {
            assert!(is_null(std::ptr::null_mut(), hp));
            assert!(is_null(base, hp));

            // Absolute store/access
            let slot_a = base.add(16).cast::<u32>();
            store(std::ptr::null_mut(), &mut hp, slot_a);
            assert!(!is_null(std::ptr::null_mut(), hp));
            assert_eq!(access::<u32>(std::ptr::null_mut(), hp), slot_a);
            assert_eq!(pointer(hp), slot_a as *mut u8);

            // Relative store/access against the same base
            let slot_b = base.add(64).cast::<u32>();
            store(base, &mut hp, slot_b);
            assert!(!is_null(base, hp));
            assert_eq!(access::<u32>(base, hp), slot_b);
            assert_eq!(offset(hp), 64);

            // Equal under each convention
            let mut hp2 = HnswPtr {
                ptr: std::ptr::null_mut(),
            };
            store(base, &mut hp2, slot_b);
            assert!(equal(base, hp, hp2));
            // Note: comparing under the OTHER convention reads the union's
            // aliased field and is meaningless by design (pgvector's union
            // has the same property).

            // Store null relatively
            store(base, &mut hp2, std::ptr::null_mut::<u32>());
            assert!(is_null(base, hp2));
        }
    }

    /// A relptr stored against one base must not silently alias another;
    /// offsets are exact byte distances.
    #[test]
    fn test_relptr_exact_offsets() {
        let mut region = [0u8; 128];
        let base = region.as_mut_ptr();
        let mut hp = HnswPtr {
            ptr: std::ptr::null_mut(),
        };
        unsafe {
            store(base, &mut hp, base.add(17).cast::<u8>());
            assert_eq!(offset(hp), 17);
            assert_eq!(access::<u8>(base, hp), base.add(17));
        }
    }
}
