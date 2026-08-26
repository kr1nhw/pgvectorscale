//! IVF index options parsing and GUC definitions.

use memoffset::*;
use pgrx::{pg_sys::AsPgCStr, prelude::*, set_varsize_4b, void_ptr, PgRelation};
use std::{ffi::CStr, fmt::Debug};

use crate::access_method::storage::StorageType;

/// Default number of inverted lists (centroids) for IVF index.
const DEFAULT_LISTS: i32 = 100;

/// Default storage type string for IVF index.
const IVF_DEFAULT_STORAGE_TYPE_STR: &str = "plain";

// DO NOT derive Clone for this struct. The storage layout string comes at the end and wouldn't be copied properly.
#[derive(Debug, PartialEq)]
#[repr(C)]
pub struct TSVIvfOptions {
    /* varlena header (do not touch directly!) */
    #[allow(dead_code)]
    vl_len_: i32,

    pub storage_layout_offset: i32,
    pub lists: i32,
}

impl TSVIvfOptions {
    /// Extract options from a relation, using defaults if none are set.
    pub fn from_relation(relation: &PgRelation) -> PgBox<TSVIvfOptions> {
        if relation.rd_index.is_null() {
            panic!("'{}' is not an IVF index", relation.name())
        } else if relation.rd_options.is_null() {
            // use defaults
            let mut ops = unsafe { PgBox::<TSVIvfOptions>::alloc0() };
            ops.storage_layout_offset = 0;
            ops.lists = DEFAULT_LISTS;
            unsafe {
                set_varsize_4b(
                    ops.as_ptr().cast(),
                    std::mem::size_of::<TSVIvfOptions>() as i32,
                );
            }
            ops.into_pg_boxed()
        } else {
            unsafe { PgBox::from_pg(relation.rd_options as *mut TSVIvfOptions) }
        }
    }

    /// Get the storage type from the options.
    pub fn get_storage_type(&self) -> StorageType {
        let s = self.get_str(self.storage_layout_offset, || {
            IVF_DEFAULT_STORAGE_TYPE_STR.to_owned()
        });

        StorageType::from_str(s.as_str())
    }

    /// Get the number of inverted lists.
    pub fn get_lists(&self) -> i32 {
        if self.lists < 1 || self.lists > 32768 {
            panic!("lists must be between 1 and 32768");
        }
        self.lists
    }

    /// Helper to extract a string option from the options struct.
    fn get_str<F: FnOnce() -> String>(&self, offset: i32, default: F) -> String {
        if offset == 0 {
            default()
        } else {
            let opts = self as *const _ as void_ptr as usize;
            let value =
                unsafe { CStr::from_ptr((opts + offset as usize) as *const std::os::raw::c_char) };

            value.to_str().unwrap().to_owned()
        }
    }
}

// GUC variables for IVF index
pub static IVF_PROBES: pgrx::GucSetting<i32> = pgrx::GucSetting::<i32>::new(1);
pub static IVF_ITERATIVE_SCAN: pgrx::GucSetting<i32> = pgrx::GucSetting::<i32>::new(0);
pub static IVF_MAX_PROBES: pgrx::GucSetting<i32> = pgrx::GucSetting::<i32>::new(32768);
pub static IVF_TOP_K: pgrx::GucSetting<i32> = pgrx::GucSetting::<i32>::new(1000);

static mut RELOPT_KIND_IVF: pg_sys::relopt_kind::Type = 0;

/// Initialize GUC variables and reloptions for IVF index.
pub unsafe fn init() {
    // Register GUC variables
    pgrx::GucRegistry::define_int_guc(
        unsafe { std::ffi::CStr::from_ptr("ivf.probes".as_pg_cstr()) },
        unsafe {
            std::ffi::CStr::from_ptr(
                "The number of probes to use when searching an IVF index".as_pg_cstr(),
            )
        },
        unsafe {
            std::ffi::CStr::from_ptr(
                "Higher value increases recall at the cost of speed.".as_pg_cstr(),
            )
        },
        &IVF_PROBES,
        1,
        32768,
        pgrx::GucContext::Userset,
        pgrx::GucFlags::default(),
    );

    pgrx::GucRegistry::define_int_guc(
        unsafe { std::ffi::CStr::from_ptr("ivf.iterative_scan".as_pg_cstr()) },
        unsafe {
            std::ffi::CStr::from_ptr(
                "Whether to use iterative scan for IVF index (0 = disabled, 1 = enabled)"
                    .as_pg_cstr(),
            )
        },
        unsafe {
            std::ffi::CStr::from_ptr(
                "When enabled, uses iterative scan to improve recall.".as_pg_cstr(),
            )
        },
        &IVF_ITERATIVE_SCAN,
        0,
        1,
        pgrx::GucContext::Userset,
        pgrx::GucFlags::default(),
    );

    pgrx::GucRegistry::define_int_guc(
        unsafe { std::ffi::CStr::from_ptr("ivf.max_probes".as_pg_cstr()) },
        unsafe {
            std::ffi::CStr::from_ptr("The maximum number of probes for iterative scan".as_pg_cstr())
        },
        unsafe {
            std::ffi::CStr::from_ptr(
                "Maximum number of probes to use during iterative scan.".as_pg_cstr(),
            )
        },
        &IVF_MAX_PROBES,
        1,
        32768,
        pgrx::GucContext::Userset,
        pgrx::GucFlags::default(),
    );

    pgrx::GucRegistry::define_int_guc(
        unsafe { std::ffi::CStr::from_ptr("ivf.top_k".as_pg_cstr()) },
        unsafe {
            std::ffi::CStr::from_ptr(
                "The number of top candidates kept per IVF search".as_pg_cstr(),
            )
        },
        unsafe {
            std::ffi::CStr::from_ptr(
                "Bounds the search to the top K candidates by estimate before exact recheck; \
                 must be at least the query's LIMIT. Higher values preserve recall."
                    .as_pg_cstr(),
            )
        },
        &IVF_TOP_K,
        1,
        1_000_000,
        pgrx::GucContext::Userset,
        pgrx::GucFlags::default(),
    );

    // Register reloptions for IVF index
    RELOPT_KIND_IVF = pg_sys::add_reloption_kind();

    pg_sys::add_string_reloption(
        RELOPT_KIND_IVF,
        "storage_layout".as_pg_cstr(),
        "Storage layout: either plain or memory_optimized".as_pg_cstr(),
        IVF_DEFAULT_STORAGE_TYPE_STR.as_pg_cstr(),
        Some(validate_storage_layout),
        pg_sys::AccessExclusiveLock as pg_sys::LOCKMODE,
    );

    pg_sys::add_int_reloption(
        RELOPT_KIND_IVF,
        "lists".as_pg_cstr(),
        "The number of inverted lists (centroids) for the IVF index".as_pg_cstr(),
        DEFAULT_LISTS,
        1,
        32768,
        pg_sys::AccessExclusiveLock as pg_sys::LOCKMODE,
    );
}

#[pg_guard]
extern "C-unwind" fn validate_storage_layout(value: *const std::os::raw::c_char) {
    if value.is_null() {
        // use a default value
        return;
    }

    let value = unsafe { CStr::from_ptr(value) }
        .to_str()
        .expect("failed to parse storage_layout value");
    _ = StorageType::from_str(value);
}

/// Parse index options for IVF.
#[allow(clippy::unneeded_field_pattern)] // b/c of offset_of!()
#[pg_guard]
pub unsafe extern "C-unwind" fn amoptions(
    reloptions: pg_sys::Datum,
    validate: bool,
) -> *mut pg_sys::bytea {
    warning!("IVF amoptions: entering");
    
    fn make_relopt_parse_elt(
        optname: &str,
        opttype: pg_sys::relopt_type::Type,
        offset: i32,
    ) -> pg_sys::relopt_parse_elt {
        #[cfg(not(feature = "pg18"))]
        {
            pg_sys::relopt_parse_elt {
                optname: optname.as_pg_cstr(),
                opttype: opttype,
                offset,
            }
        }
        #[cfg(feature = "pg18")]
        {
            pg_sys::relopt_parse_elt {
                optname: optname.as_pg_cstr(),
                opttype: opttype,
                offset,
                isset_offset: 0,
            }
        }
    }

    let tab: [pg_sys::relopt_parse_elt; 2] = [
        make_relopt_parse_elt(
            "storage_layout",
            pg_sys::relopt_type::RELOPT_TYPE_STRING,
            offset_of!(TSVIvfOptions, storage_layout_offset) as i32,
        ),
        make_relopt_parse_elt(
            "lists",
            pg_sys::relopt_type::RELOPT_TYPE_INT,
            offset_of!(TSVIvfOptions, lists) as i32,
        ),
    ];

    /* Parse the user-given reloptions */
    let rdopts = pg_sys::build_reloptions(
        reloptions,
        validate,
        RELOPT_KIND_IVF,
        std::mem::size_of::<TSVIvfOptions>(),
        tab.as_ptr(),
        tab.len() as i32,
    );

    rdopts as *mut pg_sys::bytea
}

#[cfg(any(test, feature = "pg_test"))]
#[pgrx::pg_schema]
mod tests {
    use super::*;
    use crate::access_method::storage::StorageType;
    use pgrx::*;

    #[pg_test]
    unsafe fn test_ivf_options_defaults() -> spi::Result<()> {
        Spi::run(
            "CREATE TABLE test(encoding vector(3));
        CREATE INDEX idxtest
                  ON test
               USING ivf(encoding);",
        )?;

        let index_oid =
            Spi::get_one::<pg_sys::Oid>("SELECT 'idxtest'::regclass::oid")?.expect("oid was null");
        let indexrel = PgRelation::from_pg(pg_sys::RelationIdGetRelation(index_oid));
        let options = TSVIvfOptions::from_relation(&indexrel);
        assert_eq!(options.get_lists(), DEFAULT_LISTS);
        assert_eq!(options.get_storage_type(), StorageType::Plain);
        Ok(())
    }

    #[pg_test]
    unsafe fn test_ivf_options_custom() -> spi::Result<()> {
        Spi::run(
            "CREATE TABLE test(encoding vector(3));
        CREATE INDEX idxtest
                  ON test
               USING ivf(encoding)
               WITH (lists=50, storage_layout=plain);",
        )?;

        let index_oid =
            Spi::get_one::<pg_sys::Oid>("SELECT 'idxtest'::regclass::oid")?.expect("oid was null");
        let indexrel = PgRelation::from_pg(pg_sys::RelationIdGetRelation(index_oid));
        let options = TSVIvfOptions::from_relation(&indexrel);
        assert_eq!(options.get_lists(), 50);
        assert_eq!(options.get_storage_type(), StorageType::Plain);
        Ok(())
    }
}
