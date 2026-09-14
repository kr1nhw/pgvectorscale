//! hnswsq index options parsing and GUC definitions.

use memoffset::*;
use pgrx::{pg_sys::AsPgCStr, prelude::*, set_varsize_4b, void_ptr, PgRelation};
use std::{ffi::CStr, fmt::Debug};

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

// DO NOT derive Clone for this struct. The storage layout string comes at the
// end and wouldn't be copied properly.
#[derive(Debug, PartialEq)]
#[repr(C)]
pub struct TSVHnswOptions {
    /* varlena header (do not touch directly!) */
    #[allow(dead_code)]
    vl_len_: i32,

    pub storage_layout_offset: i32,
    pub m: i32,
    pub ef_construction: i32,
    pub sample_size: i32,
}

impl TSVHnswOptions {
    /// Extract options from a relation, using defaults if none are set.
    pub fn from_relation(relation: &PgRelation) -> PgBox<TSVHnswOptions> {
        if relation.rd_index.is_null() {
            panic!("'{}' is not an hnswsq index", relation.name())
        } else if relation.rd_options.is_null() {
            // use defaults
            let mut ops = unsafe { PgBox::<TSVHnswOptions>::alloc0() };
            ops.storage_layout_offset = 0;
            ops.m = DEFAULT_M;
            ops.ef_construction = DEFAULT_EF_CONSTRUCTION;
            ops.sample_size = DEFAULT_SAMPLE_SIZE_OPTION;
            unsafe {
                set_varsize_4b(
                    ops.as_ptr().cast(),
                    std::mem::size_of::<TSVHnswOptions>() as i32,
                );
            }
            ops.into_pg_boxed()
        } else {
            unsafe { PgBox::from_pg(relation.rd_options as *mut TSVHnswOptions) }
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

    /// SQ8 calibration sample size; `None` = auto (build default).
    pub fn get_sample_size(&self) -> Option<usize> {
        if self.sample_size < 0 {
            panic!("sample_size must be >= 0 (0 = auto)");
        }
        if self.sample_size == 0 {
            None
        } else {
            Some(self.sample_size as usize)
        }
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

/// `hnswsq.ef_search`: layer-0 search width at query time (the main
/// recall/latency dial).  Must be >= the query LIMIT for full recall.
pub static HNSWSQ_EF_SEARCH: pgrx::GucSetting<i32> = pgrx::GucSetting::<i32>::new(40);

/// `hnswsq.build_stats`: emit per-phase build timing counters (search,
/// neighbor selection, backlink pair distances, backlink selection) as a
/// WARNING when a build finishes.  Off by default; used by the benchmark
/// harness to see where build time goes.
pub static HNSWSQ_BUILD_STATS: pgrx::GucSetting<bool> = pgrx::GucSetting::<bool>::new(false);

/// `hnswsq.build_seed`: RNG seed for the build (level assignment and the SQ8
/// calibration sample).  -1 keeps the production behaviour (entropy); tests pin
/// it so index builds — and therefore recall assertions — are deterministic.
pub static HNSWSQ_BUILD_SEED: pgrx::GucSetting<i32> = pgrx::GucSetting::<i32>::new(-1);

/// `hnswsq.build_backlink_mode`: how backlink edges are admitted during an
/// in-memory build.  1 (default) = the exact incremental re-prune, whose output
/// is bit-identical to re-running the full neighbor-selection heuristic over
/// the merged candidate set; 0 = the Lance-style ranked list (append, prune on
/// overflow, skip edges that cannot beat the target's current worst neighbour).
///
/// Measured on clustered 16-dim data (1 k nodes, same build seed): ranked is
/// *slower* (backlink_select 508 ms vs 294 ms) and slightly lower recall
/// (0.995 vs 1.000), so the exact mode stays the default; ranked is kept for
/// experimentation on other data shapes and m0/ef_construction settings.  Both modes are WAL/format-neutral: this is in-memory build
/// bookkeeping only.
pub static HNSWSQ_BACKLINK_MODE: pgrx::GucSetting<i32> = pgrx::GucSetting::<i32>::new(1);

/// `hnswsq.build_engine`: which in-memory build engine `CREATE INDEX` uses.
///
/// 0 (default) = the legacy `MemGraph` engine (exact incremental backlink
/// re-prune, `list_dists`/`list_masks`); 1 = the new flat engine (flat slabs, ids
/// only, pgvector-style append/shrink backlinks).  Temporary: it exists so both
/// engines can be A/B'd in one binary while the new one is brought up, and it
/// disappears with the legacy engine (see
/// `.design/hnswsq_parallel_build_todos.md`).
pub static HNSWSQ_BUILD_ENGINE: pgrx::GucSetting<i32> = pgrx::GucSetting::<i32>::new(0);

/// `hnswsq.build_backfill`: fill a node's own neighbour list to capacity with the
/// closest *pruned* candidates (the legacy engine's behaviour) instead of keeping
/// only the occlusion heuristic's output.
///
/// 0 (default) = heuristic only, which is the decided flat-engine policy; 1 = with
/// backfill.  Measured at 100k dim-128: backfill costs +51% build time (apply 2.2x,
/// because every backlink then lands on a saturated target and pays the full
/// re-measure plus occlusion walk) and buys +9.5 recall points at ef 40 on that hard
/// dataset, plus 93 fewer nodes without an incoming edge.  Temporary knob so the 1M
/// BIGANN operating point can decide it without code churn; it goes away with the
/// engine consolidation.
pub static HNSWSQ_BUILD_BACKFILL: pgrx::GucSetting<i32> = pgrx::GucSetting::<i32>::new(0);

/// `hnswsq.build_workers`: worker threads used for parallel backlink pruning
/// during an in-memory build (0 = auto).  Purely in-process parallelism over
/// the in-memory graph; the transactional insert path is unaffected.
pub static HNSWSQ_BUILD_WORKERS: pgrx::GucSetting<i32> = pgrx::GucSetting::<i32>::new(0);

/// `hnswsq.parallel_stage`: **debug only** -- stop a parallel worker after this many steps, so
/// a crash in the worker path can be bisected without recompiling.  `0` is a real build.
///
/// The stages are, in order: 9 before the worker does anything at all, 1 after opening the
/// relations, 2 after `BuildIndexInfo`, 3 after `table_beginscan_parallel`, 4 after the worker's
/// `BuildState`, then -- inside the row callback -- 5 before reading the tuple, 6 after
/// extracting the vector, 7 after deciding the level.  Anything else means "run to completion".  Stage 9 exists to separate "my
/// worker code is wrong" from "PostgreSQL's worker startup is unhappy with how I drove it".  It exists because the worker path is new and a segfault there tells you
/// nothing about which call caused it.
pub static HNSWSQ_PARALLEL_STAGE: pgrx::GucSetting<i32> = pgrx::GucSetting::<i32>::new(0);

static mut RELOPT_KIND_HNSW: pg_sys::relopt_kind::Type = 0;

/// Initialize GUC variables and reloptions for the hnswsq access method.
pub unsafe fn init() {
    pgrx::GucRegistry::define_int_guc(
        unsafe { std::ffi::CStr::from_ptr("hnswsq.ef_search".as_pg_cstr()) },
        unsafe {
            std::ffi::CStr::from_ptr(
                "The search width (ef) used when querying an hnswsq index".as_pg_cstr(),
            )
        },
        unsafe {
            std::ffi::CStr::from_ptr(
                "Higher values increase recall at the cost of speed; must be at \
                 least the query's LIMIT."
                    .as_pg_cstr(),
            )
        },
        &HNSWSQ_EF_SEARCH,
        1,
        1000,
        pgrx::GucContext::Userset,
        pgrx::GucFlags::default(),
    );

    pgrx::GucRegistry::define_bool_guc(
        unsafe { std::ffi::CStr::from_ptr("hnswsq.build_stats".as_pg_cstr()) },
        unsafe {
            std::ffi::CStr::from_ptr(
                "Report per-phase hnswsq build timing counters when a build finishes".as_pg_cstr(),
            )
        },
        unsafe {
            std::ffi::CStr::from_ptr(
                "Diagnostic only: adds a few Instant::now() calls per inserted node.".as_pg_cstr(),
            )
        },
        &HNSWSQ_BUILD_STATS,
        pgrx::GucContext::Userset,
        pgrx::GucFlags::default(),
    );

    pgrx::GucRegistry::define_int_guc(
        unsafe { std::ffi::CStr::from_ptr("hnswsq.build_backlink_mode".as_pg_cstr()) },
        unsafe {
            std::ffi::CStr::from_ptr(
                "Backlink admission during hnswsq builds (0 = ranked/cutoff, 1 = exact)".as_pg_cstr(),
            )
        },
        unsafe {
            std::ffi::CStr::from_ptr(
                "0 selects the Lance-style ranked list (append, prune on overflow, skip edges \
                 worse than the target's current worst neighbour); 1 selects the exact \
                 incremental re-prune.  Build-only; index contents and transactional \
                 behaviour are otherwise unchanged."
                    .as_pg_cstr(),
            )
        },
        &HNSWSQ_BACKLINK_MODE,
        0,
        1,
        pgrx::GucContext::Userset,
        pgrx::GucFlags::default(),
    );

    pgrx::GucRegistry::define_int_guc(
        unsafe { std::ffi::CStr::from_ptr("hnswsq.build_engine".as_pg_cstr()) },
        unsafe {
            std::ffi::CStr::from_ptr(
                "In-memory build engine (0 = legacy MemGraph, 1 = flat engine)".as_pg_cstr(),
            )
        },
        unsafe {
            std::ffi::CStr::from_ptr(
                "Development switch for A/B testing the two in-memory build engines; \
                 the two can differ slightly in graph quality because their backlink \
                 policies differ."
                    .as_pg_cstr(),
            )
        },
        &HNSWSQ_BUILD_ENGINE,
        0,
        1,
        pgrx::GucContext::Userset,
        pgrx::GucFlags::default(),
    );

    pgrx::GucRegistry::define_int_guc(
        unsafe { std::ffi::CStr::from_ptr("hnswsq.build_backfill".as_pg_cstr()) },
        unsafe {
            std::ffi::CStr::from_ptr(
                "Fill own neighbour lists with the closest pruned candidates (0 = heuristic only)".as_pg_cstr(),
            )
        },
        unsafe {
            std::ffi::CStr::from_ptr(
                "Development knob: backfill trades build time for recall; measured at 100k dim-128 it costs ~51% build time and gains ~9.5 recall points at ef 40."
                    .as_pg_cstr(),
            )
        },
        &HNSWSQ_BUILD_BACKFILL,
        0,
        1,
        pgrx::GucContext::Userset,
        pgrx::GucFlags::default(),
    );

    pgrx::GucRegistry::define_int_guc(
        unsafe { std::ffi::CStr::from_ptr("hnswsq.parallel_stage".as_pg_cstr()) },
        unsafe {
            std::ffi::CStr::from_ptr(
                "Debug only: stop a parallel worker after N steps (0 = run the build)."
                    .as_pg_cstr(),
            )
        },
        unsafe {
            std::ffi::CStr::from_ptr(
                "1 relations, 2 index info, 3 scan, 4 worker state; 0 or >4 runs the build."
                    .as_pg_cstr(),
            )
        },
        &HNSWSQ_PARALLEL_STAGE,
        0,
        9,
        pgrx::GucContext::Userset,
        pgrx::GucFlags::default(),
    );

    pgrx::GucRegistry::define_int_guc(
        unsafe { std::ffi::CStr::from_ptr("hnswsq.build_seed".as_pg_cstr()) },
        unsafe {
            std::ffi::CStr::from_ptr(
                "RNG seed for hnswsq index builds (-1 = entropy)".as_pg_cstr(),
            )
        },
        unsafe {
            std::ffi::CStr::from_ptr(
                "Pins the level-assignment RNG so repeated builds of the same data are                  identical; -1 (default) seeds from entropy as in production."
                    .as_pg_cstr(),
            )
        },
        &HNSWSQ_BUILD_SEED,
        -1,
        i32::MAX,
        pgrx::GucContext::Userset,
        pgrx::GucFlags::default(),
    );

    pgrx::GucRegistry::define_int_guc(
        unsafe { std::ffi::CStr::from_ptr("hnswsq.build_workers".as_pg_cstr()) },
        unsafe {
            std::ffi::CStr::from_ptr(
                "Worker threads for in-memory build backlink pruning (0 = auto)".as_pg_cstr(),
            )
        },
        unsafe {
            std::ffi::CStr::from_ptr(
                "Parallelizes only the in-memory build; index contents and transactional \
                 behaviour are unchanged. 0 selects min(cores, 4); 1 disables parallelism."
                    .as_pg_cstr(),
            )
        },
        &HNSWSQ_BUILD_WORKERS,
        0,
        64,
        pgrx::GucContext::Userset,
        pgrx::GucFlags::default(),
    );

    RELOPT_KIND_HNSW = pg_sys::add_reloption_kind();

    pg_sys::add_string_reloption(
        RELOPT_KIND_HNSW,
        "storage_layout".as_pg_cstr(),
        "Node vector precision: plain, ieeefp16 (f16), ieeefp8, or f8 (sq8)"
            .as_pg_cstr(),
        HNSW_DEFAULT_STORAGE_TYPE_STR.as_pg_cstr(),
        Some(validate_storage_layout),
        pg_sys::AccessExclusiveLock as pg_sys::LOCKMODE,
    );

    pg_sys::add_int_reloption(
        RELOPT_KIND_HNSW,
        "m".as_pg_cstr(),
        "Maximum number of neighbors per node per upper layer (layer 0 uses 2*m)"
            .as_pg_cstr(),
        DEFAULT_M,
        4,
        100,
        pg_sys::AccessExclusiveLock as pg_sys::LOCKMODE,
    );

    pg_sys::add_int_reloption(
        RELOPT_KIND_HNSW,
        "ef_construction".as_pg_cstr(),
        "The search list size used during build and insert".as_pg_cstr(),
        DEFAULT_EF_CONSTRUCTION,
        4,
        1000,
        pg_sys::AccessExclusiveLock as pg_sys::LOCKMODE,
    );

    pg_sys::add_int_reloption(
        RELOPT_KIND_HNSW,
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

/// Parse index options for hnswsq.
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

    let tab: [pg_sys::relopt_parse_elt; 4] = [
        make_relopt_parse_elt(
            "storage_layout",
            pg_sys::relopt_type::RELOPT_TYPE_STRING,
            offset_of!(TSVHnswOptions, storage_layout_offset) as i32,
        ),
        make_relopt_parse_elt(
            "m",
            pg_sys::relopt_type::RELOPT_TYPE_INT,
            offset_of!(TSVHnswOptions, m) as i32,
        ),
        make_relopt_parse_elt(
            "ef_construction",
            pg_sys::relopt_type::RELOPT_TYPE_INT,
            offset_of!(TSVHnswOptions, ef_construction) as i32,
        ),
        make_relopt_parse_elt(
            "sample_size",
            pg_sys::relopt_type::RELOPT_TYPE_INT,
            offset_of!(TSVHnswOptions, sample_size) as i32,
        ),
    ];

    /* Parse the user-given reloptions */
    let rdopts = pg_sys::build_reloptions(
        reloptions,
        validate,
        RELOPT_KIND_HNSW,
        std::mem::size_of::<TSVHnswOptions>(),
        tab.as_ptr(),
        tab.len() as i32,
    );

    rdopts as *mut pg_sys::bytea
}

#[cfg(any(test, feature = "pg_test"))]
#[pgrx::pg_schema]
mod tests {
    use super::*;
    use pgrx::*;

    #[pg_test]
    unsafe fn test_hnswsq_options_defaults() -> spi::Result<()> {
        crate::access_method::hnswsq::lock_suite_for_test();
        Spi::run(
            "CREATE TABLE test(encoding vector(3));
        CREATE INDEX idxtest
                  ON test
               USING hnswsq(encoding);",
        )?;

        let index_oid =
            Spi::get_one::<pg_sys::Oid>("SELECT 'idxtest'::regclass::oid")?.expect("oid was null");
        let indexrel = PgRelation::from_pg(pg_sys::RelationIdGetRelation(index_oid));
        let options = TSVHnswOptions::from_relation(&indexrel);
        assert_eq!(options.get_m(), DEFAULT_M as u16);
        assert_eq!(options.get_ef_construction(), DEFAULT_EF_CONSTRUCTION as u32);
        assert_eq!(options.get_precision(), HnswPrecision::Plain);
        assert_eq!(options.get_sample_size(), None);
        Ok(())
    }

    #[pg_test]
    unsafe fn test_hnswsq_options_custom() -> spi::Result<()> {
        crate::access_method::hnswsq::lock_suite_for_test();
        Spi::run(
            "CREATE TABLE test(encoding vector(3));
        CREATE INDEX idxtest
                  ON test
               USING hnswsq(encoding)
               WITH (storage_layout = ieeefp8, m = 24, ef_construction = 128, sample_size = 500);",
        )?;

        let index_oid =
            Spi::get_one::<pg_sys::Oid>("SELECT 'idxtest'::regclass::oid")?.expect("oid was null");
        let indexrel = PgRelation::from_pg(pg_sys::RelationIdGetRelation(index_oid));
        let options = TSVHnswOptions::from_relation(&indexrel);
        assert_eq!(options.get_m(), 24);
        assert_eq!(options.get_ef_construction(), 128);
        assert_eq!(options.get_precision(), HnswPrecision::IeeeFp8);
        assert_eq!(options.get_sample_size(), Some(500));
        Ok(())
    }

    #[pg_test]
    unsafe fn test_hnswsq_options_aliases() -> spi::Result<()> {
        crate::access_method::hnswsq::lock_suite_for_test();
        Spi::run(
            "CREATE TABLE test(encoding vector(3));
        CREATE INDEX idx16 ON test USING hnswsq(encoding) WITH (storage_layout = f16);
        CREATE INDEX idxsq8 ON test USING hnswsq(encoding) WITH (storage_layout = sq8);",
        )?;

        let oid16 =
            Spi::get_one::<pg_sys::Oid>("SELECT 'idx16'::regclass::oid")?.expect("oid was null");
        let rel16 = PgRelation::from_pg(pg_sys::RelationIdGetRelation(oid16));
        assert_eq!(
            TSVHnswOptions::from_relation(&rel16).get_precision(),
            HnswPrecision::IeeeFp16
        );

        let oid8 =
            Spi::get_one::<pg_sys::Oid>("SELECT 'idxsq8'::regclass::oid")?.expect("oid was null");
        let rel8 = PgRelation::from_pg(pg_sys::RelationIdGetRelation(oid8));
        assert_eq!(
            TSVHnswOptions::from_relation(&rel8).get_precision(),
            HnswPrecision::Sq8
        );
        Ok(())
    }
}
