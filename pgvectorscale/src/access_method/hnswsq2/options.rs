//! hnswsq2 index options parsing and GUC definitions.
//!
//! Same reloption surface as the retired engine (`storage_layout`, `m`,
//! `ef_construction`, `sample_size`) so the SQL-facing configuration of an
//! index is unchanged.  The engine-specific GUCs of the old framework
//! (`build_engine`, `backfill`, `parallel_stage`, `build_backlink_mode`,
//! `build_stats`) are retired with it; the GUCs below are the pgvector set
//! (`ef_search`, `iterative_scan`, `max_scan_tuples`, `scan_mem_multiplier`)
//! plus `build_seed`, which the determinism gates pin.

use memoffset::*;
use pgrx::pg_sys::AsPgCStr;
use pgrx::{pg_sys, prelude::*, set_varsize_4b, void_ptr, PgRelation};
use std::ffi::CStr;
use std::fmt::Debug;

use crate::access_method::hnswsq::quantize::HnswPrecision;

/// Default max neighbors per node per upper layer (layer 0 gets 2×).
const DEFAULT_M: i32 = 16;

/// Default search width during build/insert.
const DEFAULT_EF_CONSTRUCTION: i32 = 64;

/// Default SQ8 calibration sample size: 0 means "auto" (30000 at build).
const DEFAULT_SAMPLE_SIZE_OPTION: i32 = 0;

/// Default storage layout (precision).  The IEEE layouts (`ieeefp16`,
/// `ieeefp8`) are training-free; `f8` (SQ8) is calibrated at build.
const HNSW_DEFAULT_STORAGE_TYPE_STR: &str = "plain";

/// Build-time default for the SQ8 calibration reservoir sample.
pub const DEFAULT_SAMPLE_SIZE: usize = 30000;

/// Default search width at query time (pgvector `hnsw_ef_search`).
pub static HNSW2_EF_SEARCH: pgrx::GucSetting<i32> = pgrx::GucSetting::<i32>::new(40);

/// Iterative scan mode (pgvector `hnsw_iterative_scan`).
#[derive(
    pgrx::PostgresGucEnum, Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default,
)]
pub enum IterativeScanMode {
    Off,
    #[default]
    Relaxed,
    Strict,
}

impl IterativeScanMode {
    pub fn as_i32(self) -> i32 {
        match self {
            IterativeScanMode::Off => 0,
            IterativeScanMode::Relaxed => 1,
            IterativeScanMode::Strict => 2,
        }
    }
}

pub static HNSW2_ITERATIVE_SCAN: pgrx::GucSetting<IterativeScanMode> =
    pgrx::GucSetting::<IterativeScanMode>::new(IterativeScanMode::Relaxed);
pub const ITERATIVE_SCAN_OFF: i32 = 0;
pub const ITERATIVE_SCAN_RELAXED: i32 = 1;
pub const ITERATIVE_SCAN_STRICT: i32 = 2;

/// Max tuples before an iterative scan stops (pgvector `hnsw_max_scan_tuples`,
/// -1 = unlimited).
pub static HNSW2_MAX_SCAN_TUPLES: pgrx::GucSetting<i32> = pgrx::GucSetting::<i32>::new(-1);

/// work_mem multiplier for the scan (pgvector `hnsw_scan_mem_multiplier`,
/// fixed at the pgvector default of 1.0 — pgrx has no f32 GucSetting
/// constructor, and the multiplier is a rarely-tuned knob).
pub const HNSW2_SCAN_MEM_MULTIPLIER: f64 = 1.0;

/// Build RNG: entropy in production, or the pinned `hnswsq2.build_seed` value
/// (tests set it so builds — and recall assertions — are deterministic).
pub static HNSW2_BUILD_SEED: pgrx::GucSetting<i32> = pgrx::GucSetting::<i32>::new(-1);

// DO NOT derive Clone for this struct. The storage layout string comes at the
// end and wouldn't be copied properly.
#[derive(Debug, PartialEq)]
#[repr(C)]
pub struct Hnsw2Options {
    /* varlena header (do not touch directly!) */
    #[allow(dead_code)]
    vl_len_: i32,

    pub storage_layout_offset: i32,
    pub m: i32,
    pub ef_construction: i32,
    pub sample_size: i32,
}

impl Hnsw2Options {
    /// Extract options from a relation, using defaults if none are set.
    pub fn from_relation(relation: &PgRelation) -> PgBox<Hnsw2Options> {
        if relation.rd_index.is_null() {
            panic!("'{}' is not an hnswsq2 index", relation.name())
        } else if relation.rd_options.is_null() {
            // use defaults
            let mut ops = unsafe { PgBox::<Hnsw2Options>::alloc0() };
            ops.storage_layout_offset = 0;
            ops.m = DEFAULT_M;
            ops.ef_construction = DEFAULT_EF_CONSTRUCTION;
            ops.sample_size = DEFAULT_SAMPLE_SIZE_OPTION;
            unsafe {
                set_varsize_4b(ops.as_ptr().cast(), std::mem::size_of::<Hnsw2Options>() as i32);
            }
            ops.into_pg_boxed()
        } else {
            unsafe { PgBox::from_pg(relation.rd_options as *mut Hnsw2Options) }
        }
    }

    /// Get the node vector precision from the options.
    pub fn get_precision(&self) -> HnswPrecision {
        let s = self.get_str(self.storage_layout_offset, || {
            HNSW_DEFAULT_STORAGE_TYPE_STR.to_owned()
        });
        HnswPrecision::parse(s.as_str())
    }

    /// Max neighbors per node per upper layer (layer 0 gets `2*m`).
    pub fn get_m(&self) -> u16 {
        if self.m < 4 || self.m > 100 {
            panic!("m must be between 4 and 100");
        }
        self.m as u16
    }

    /// Search width during build/insert.
    pub fn get_ef_construction(&self) -> u32 {
        if self.ef_construction < 4 || self.ef_construction > 1000 {
            panic!("ef_construction must be between 4 and 1000");
        }
        self.ef_construction as u32
    }

    /// SQ8 calibration sample size (0 = auto).
    pub fn get_sample_size(&self) -> usize {
        if self.sample_size < 0 || self.sample_size > 1_000_000 {
            panic!("sample_size must be between 0 and 1000000");
        }
        if self.sample_size == 0 {
            DEFAULT_SAMPLE_SIZE
        } else {
            self.sample_size as usize
        }
    }

    fn get_str<F: FnOnce() -> String>(
        &self,
        offset: i32,
        _default: F,
    ) -> String {
        // storage layout string is written directly into the options by
        // PostgreSQL (the offset points into rd_options bytes)
        let p = (self as *const Hnsw2Options as *const u8).wrapping_add(offset as usize);
        if offset == 0 {
            return _default();
        }
        unsafe {
            let c = CStr::from_ptr(p.cast());
            c.to_str().expect("invalid storage_layout").to_owned()
        }
    }
}

static mut RELOPT_KIND_HNSW2: pg_sys::relopt_kind::Type = 0;

/// Initialize GUC variables and reloptions for the hnswsq2 access method.
pub unsafe fn init() {
    pgrx::GucRegistry::define_int_guc(
        c"hnswsq2.ef_search",
        c"The search width (ef) used when querying an hnswsq2 index",
        c"Higher values increase recall at the cost of speed; must be at least the query's LIMIT.",
        &HNSW2_EF_SEARCH,
        1,
        1000,
        pgrx::GucContext::Userset,
        pgrx::GucFlags::default(),
    );

    pgrx::GucRegistry::define_enum_guc(
        c"hnswsq2.iterative_scan",
        c"Continues index scans to find more tuples (off, relaxed, strict)",
        c"Filtered queries need this; relaxed stops early when the index runs out of candidates, strict also enforces exact non-decreasing distance order.",
        &HNSW2_ITERATIVE_SCAN,
        pgrx::GucContext::Userset,
        pgrx::GucFlags::default(),
    );

    pgrx::GucRegistry::define_int_guc(
        c"hnswsq2.max_scan_tuples",
        c"Max tuples an hnswsq2 iterative scan examines (-1 = unlimited)",
        c"Bounds the work a filtered query can do; only applies when iterative_scan is on.",
        &HNSW2_MAX_SCAN_TUPLES,
        -1,
        i32::MAX,
        pgrx::GucContext::Userset,
        pgrx::GucFlags::default(),
    );

    pgrx::GucRegistry::define_int_guc(
        c"hnswsq2.build_seed",
        c"RNG seed for hnswsq2 index builds (-1 = entropy)",
        c"Pins the level-assignment RNG so repeated builds of the same data are identical; -1 (default) seeds from entropy as in production.",
        &HNSW2_BUILD_SEED,
        -1,
        i32::MAX,
        pgrx::GucContext::Userset,
        pgrx::GucFlags::default(),
    );

    RELOPT_KIND_HNSW2 = pg_sys::add_reloption_kind();

    pg_sys::add_string_reloption(
        RELOPT_KIND_HNSW2,
        "storage_layout".as_pg_cstr(),
        "Node vector precision: plain, ieeefp16 (f16), ieeefp8, or f8 (sq8)"
            .as_pg_cstr(),
        HNSW_DEFAULT_STORAGE_TYPE_STR.as_pg_cstr(),
        Some(validate_storage_layout),
        pg_sys::AccessExclusiveLock as pg_sys::LOCKMODE,
    );

    pg_sys::add_int_reloption(
        RELOPT_KIND_HNSW2,
        "m".as_pg_cstr(),
        "Maximum number of neighbors per node per upper layer (layer 0 uses 2*m)"
            .as_pg_cstr(),
        DEFAULT_M,
        4,
        100,
        pg_sys::AccessExclusiveLock as pg_sys::LOCKMODE,
    );

    pg_sys::add_int_reloption(
        RELOPT_KIND_HNSW2,
        "ef_construction".as_pg_cstr(),
        "The search list size used during build and insert".as_pg_cstr(),
        DEFAULT_EF_CONSTRUCTION,
        4,
        1000,
        pg_sys::AccessExclusiveLock as pg_sys::LOCKMODE,
    );

    pg_sys::add_int_reloption(
        RELOPT_KIND_HNSW2,
        "sample_size".as_pg_cstr(),
        "Vectors reservoir-sampled for SQ8 (f8) calibration (0 = auto, 30000)"
            .as_pg_cstr(),
        DEFAULT_SAMPLE_SIZE_OPTION,
        0,
        1_000_000,
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
    _ = HnswPrecision::parse(value);
}

/// Parse index options for hnswsq2.
#[allow(clippy::unneeded_field_pattern)] // b/c of offset_of!()
#[pg_guard]
pub unsafe extern "C-unwind" fn amoptions(
    reloptions: pg_sys::Datum,
    validate: bool,
) -> *mut pg_sys::bytea {
    fn make_relopt_parse_elt(
        optname: &str,
        opttype: pg_sys::relopt_type::Type,
        offset: i32,
    ) -> pg_sys::relopt_parse_elt {
        #[cfg(not(feature = "pg18"))]
        {
            pg_sys::relopt_parse_elt {
                optname: optname.as_pg_cstr(),
                opttype,
                offset,
            }
        }
        #[cfg(feature = "pg18")]
        {
            pg_sys::relopt_parse_elt {
                optname: optname.as_pg_cstr(),
                opttype,
                offset,
                isset_offset: 0,
            }
        }
    }

    let tab: [pg_sys::relopt_parse_elt; 4] = [
        make_relopt_parse_elt(
            "storage_layout",
            pg_sys::relopt_type::RELOPT_TYPE_STRING,
            offset_of!(Hnsw2Options, storage_layout_offset) as i32,
        ),
        make_relopt_parse_elt(
            "m",
            pg_sys::relopt_type::RELOPT_TYPE_INT,
            offset_of!(Hnsw2Options, m) as i32,
        ),
        make_relopt_parse_elt(
            "ef_construction",
            pg_sys::relopt_type::RELOPT_TYPE_INT,
            offset_of!(Hnsw2Options, ef_construction) as i32,
        ),
        make_relopt_parse_elt(
            "sample_size",
            pg_sys::relopt_type::RELOPT_TYPE_INT,
            offset_of!(Hnsw2Options, sample_size) as i32,
        ),
    ];

    unsafe {
        pg_sys::build_reloptions(
            reloptions,
            validate,
            RELOPT_KIND_HNSW2,
            std::mem::size_of::<Hnsw2Options>(),
            tab.as_ptr(),
            tab.len() as i32,
        ) as *mut pg_sys::bytea
    }
}
