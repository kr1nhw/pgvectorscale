//! `agentvec` index options.
//!
//! All AgentVec tuning knobs are per-index reloptions (`WITH (...)`) so a
//! single database can hold indexes with different leveling policies.  The
//! defaults are the starting hypotheses from the design's maintenance-policy
//! section, not fixed product requirements:
//!
//! ```text
//! HOT  <= 50K vectors        (small, mutable, high write rate)
//! WARM ~100K - 5M vectors    (recently consolidated, moderate mutation)
//! COLD > 5M vectors          (large, stable, aggressively compressed)
//! ```
//!
//! Options that no Phase-1 code path consumes yet are still defined (and
//! documented as reserved) so the on-disk format and the SQL surface do not
//! have to change again when the phase that uses them lands.

use memoffset::*;
use pgrx::{pg_sys::AsPgCStr, prelude::*, PgRelation};
use std::ffi::CStr;
use std::fmt::Debug;

use crate::access_method::hnswsq::quantize::HnswPrecision;

/// Vectors a HOT segment absorbs before it is sealed.  The foreground INSERT
/// that observes the threshold only performs the metadata changes that make
/// future inserts use a new HOT segment; the sealed one is consolidated
/// asynchronously.
const DEFAULT_HOT_SEGMENT_MAX_ROWS: i32 = 50_000;

/// Default number of concurrently appendable HOT segments.
///
/// 1 preserves the original single-HOT-segment behaviour, which is also what
/// the measured insert-throughput experiments support on current hardware: K
/// independent hnswsq regions (emulated with K independent indexes) produced
/// no aggregate throughput change at 8, 16 or 32 concurrent inserters, because
/// the foreground insert cost is per-row CPU and buffer-manager work rather
/// than contention on one graph's pages.  K > 1 is therefore an explicit
/// opt-in for workloads that need independent seal schedules (so one
/// consolidation does not stall writers) or a bound on the per-graph size.
const DEFAULT_HOT_SEGMENT_COUNT: i32 = 1;

/// Sentinel for "adapt the HOT segment count to the observed write load",
/// used as the `hot_segment_count` reloption value.
pub const HOT_SEGMENT_COUNT_AUTO: usize = 0;

/// Hard ceiling on the HOT segment pool, so a misconfiguration cannot make
/// every scan fan out over an unbounded number of graphs.
pub const MAX_HOT_SEGMENT_COUNT: usize = 16;

/// Target row count of a WARM segment (Phase 6 leveling target).
const DEFAULT_WARM_SEGMENT_TARGET_ROWS: i32 = 1_000_000;

/// Target row count of a COLD segment (Phase 6 leveling target).
const DEFAULT_COLD_SEGMENT_TARGET_ROWS: i32 = 10_000_000;

/// RaBitQ bits per dimension for WARM/COLD segments (Phase 3).
const DEFAULT_RABITQ_BITS: i32 = 1;

/// IVF lists (centroids) per WARM/COLD segment (Phase 3).
const DEFAULT_IVF_LISTS: i32 = 100;

/// IVF lists probed per query (Phase 3).
const DEFAULT_IVF_PROBES: i32 = 10;

/// Maximum segments the deterministic router activates per query (active: phase 7).
const DEFAULT_ROUTER_TOP_M: i32 = 8;

/// Maximum segments activated per logical group (Phase 7).
const DEFAULT_ROUTER_GROUP_TOP_M: i32 = 2;

/// Rows per maintenance batch (Phase 4/5).
const DEFAULT_MIGRATION_BATCH_ROWS: i32 = 5_000;

/// Minimum milliseconds between maintenance passes (Phase 4).
const DEFAULT_MAINTENANCE_INTERVAL: i32 = 60_000;

/// Upper bound on bytes a single maintenance transaction may write (Phase 4).
const DEFAULT_MAINTENANCE_MAX_BYTES: i32 = 64 * 1024 * 1024;

/// Candidates rescored exactly during the rerank stage (Phase 8).
const DEFAULT_RERANK_K: i32 = 100;

/// Candidates a single segment scan keeps before the merge.
///
/// `0` means "unbounded": the exact `FLAT` executor returns every live entry
/// in distance order, while approximate executors (HNSW HOT, IVF segments)
/// use their own ef (`hnswsq.ef_search`).  A positive value bounds each
/// segment's candidate heap.
const DEFAULT_SEARCH_CANDIDATES: i32 = 0;

/// Storage layout of a HOT HNSW segment (the embedded hnswsq region's
/// `storage_layout`: plain / ieeefp16 / ieeefp8 / f8 / sq8 / sq16).
///
/// `plain` is the default because the HOT segment is a bounded write cache:
/// it holds at most `hot_segment_max_rows` entries, so the larger codes of an
/// uncompressed layout stay cheap in absolute terms (a 20 000-row dim-128
/// index measures 16 MB with `plain` vs 7.4 MB with `f8`, i.e. ~40 MB vs
/// ~18 MB at the 50 000-row cap), while the quantized layouts pay for that
/// saving on every foreground insert.  Measured on 32 vCPU / PG 17.11, dim
/// 128, m=16, ef_construction=64, 8 concurrent inserters: `plain` 4 982 rows/s
/// vs `f8` 4 263 (+17 %), and 14 171 vs 12 152 at 32 inserters; insert CPU
/// falls from 1 231 to 937 us/row because the SQ8 codec is ~43 % of the insert
/// profile.  Recall is unaffected (1.0000@10 against exact ground truth on
/// both layouts in the 30k-row check), and the training-free layout also makes
/// consolidation cheaper: it decodes inline instead of fetching every exact
/// vector from the heap.
const DEFAULT_HOT_STORAGE_LAYOUT_STR: &str = "plain";

/// HNSW `m` of a HOT segment (neighbors per layer-0 node).
const DEFAULT_HOT_M: i32 = 16;

/// HNSW `ef_construction` of a HOT segment.
const DEFAULT_HOT_EF_CONSTRUCTION: i32 = 64;

/// The parsed reloptions of an `agentvec` index.
///
/// Do NOT derive `Clone`: the struct maps the on-disk `rd_options` byte string
/// and any future trailing string option would not be copied by a derived
/// `Clone`.
#[derive(Debug, PartialEq)]
#[repr(C)]
pub struct TSVAgentVecOptions {
    /* varlena header (do not touch directly!) */
    #[allow(dead_code)]
    vl_len_: i32,

    pub hot_storage_layout_offset: i32,
    pub hot_m: i32,
    pub hot_ef_construction: i32,

    pub hot_segment_max_rows: i32,
    pub hot_segment_count: i32,
    pub warm_segment_target_rows: i32,
    pub cold_segment_target_rows: i32,
    pub rabitq_bits: i32,
    pub ivf_lists: i32,
    pub ivf_probes: i32,
    pub router_top_m: i32,
    pub router_group_top_m: i32,
    pub migration_batch_rows: i32,
    pub maintenance_interval: i32,
    pub maintenance_max_bytes: i32,
    pub rerank_k: i32,
    pub search_candidates: i32,
}

impl TSVAgentVecOptions {
    /// Extract options from a relation, using defaults when none are set.
    pub fn from_relation(relation: &PgRelation) -> PgBox<TSVAgentVecOptions> {
        if relation.rd_index.is_null() {
            panic!("'{}' is not an agentvec index", relation.name())
        } else if relation.rd_options.is_null() {
            let mut ops = unsafe { PgBox::<TSVAgentVecOptions>::alloc0() };
            ops.hot_storage_layout_offset = 0;
            ops.hot_m = DEFAULT_HOT_M;
            ops.hot_ef_construction = DEFAULT_HOT_EF_CONSTRUCTION;
            ops.hot_segment_max_rows = DEFAULT_HOT_SEGMENT_MAX_ROWS;
            ops.hot_segment_count = DEFAULT_HOT_SEGMENT_COUNT;
            ops.warm_segment_target_rows = DEFAULT_WARM_SEGMENT_TARGET_ROWS;
            ops.cold_segment_target_rows = DEFAULT_COLD_SEGMENT_TARGET_ROWS;
            ops.rabitq_bits = DEFAULT_RABITQ_BITS;
            ops.ivf_lists = DEFAULT_IVF_LISTS;
            ops.ivf_probes = DEFAULT_IVF_PROBES;
            ops.router_top_m = DEFAULT_ROUTER_TOP_M;
            ops.router_group_top_m = DEFAULT_ROUTER_GROUP_TOP_M;
            ops.migration_batch_rows = DEFAULT_MIGRATION_BATCH_ROWS;
            ops.maintenance_interval = DEFAULT_MAINTENANCE_INTERVAL;
            ops.maintenance_max_bytes = DEFAULT_MAINTENANCE_MAX_BYTES;
            ops.rerank_k = DEFAULT_RERANK_K;
            ops.search_candidates = DEFAULT_SEARCH_CANDIDATES;
            unsafe {
                pgrx::set_varsize_4b(
                    ops.as_ptr().cast(),
                    std::mem::size_of::<TSVAgentVecOptions>() as i32,
                );
            }
            ops.into_pg_boxed()
        } else {
            unsafe { PgBox::from_pg(relation.rd_options as *mut TSVAgentVecOptions) }
        }
    }

    /// Rows a HOT segment absorbs before being sealed.
    pub fn get_hot_segment_max_rows(&self) -> u64 {
        if self.hot_segment_max_rows < 1 {
            panic!("hot_segment_max_rows must be >= 1");
        }
        self.hot_segment_max_rows as u64
    }

    /// HOT segments foreground inserts may append to concurrently.
    ///
    /// `1` (the default) keeps the original single-appendable-segment
    /// behaviour.  A larger value keeps that many HOT segments published and
    /// appendable at once, so the writer side is not restricted to one
    /// hnswsq region: a segment being sealed hands its writers to a sibling
    /// instead of stalling them, and each region carries its own metapage,
    /// insert page, header page and entry point.
    ///
    /// `0` selects the adaptive mode (see [`HOT_SEGMENT_COUNT_AUTO`]).
    pub fn get_hot_segment_count(&self) -> usize {
        if self.hot_segment_count < 0 {
            panic!("hot_segment_count must be >= 0");
        }
        self.hot_segment_count as usize
    }

    /// Storage layout of a HOT HNSW segment.
    pub fn get_hot_precision(&self) -> HnswPrecision {
        let s = self.get_str(self.hot_storage_layout_offset, || {
            DEFAULT_HOT_STORAGE_LAYOUT_STR.to_owned()
        });
        HnswPrecision::parse(&s)
    }

    /// HNSW `m` of a HOT segment.
    pub fn get_hot_m(&self) -> usize {
        if self.hot_m < 2 || self.hot_m > 100 {
            panic!("hot_m must be between 2 and 100");
        }
        self.hot_m as usize
    }

    /// HNSW `ef_construction` of a HOT segment.
    pub fn get_hot_ef_construction(&self) -> usize {
        if self.hot_ef_construction < 4 || self.hot_ef_construction > 1000 {
            panic!("hot_ef_construction must be between 4 and 1000");
        }
        self.hot_ef_construction as usize
    }

    /// Helper to extract a string option from the options struct.
    fn get_str<F: FnOnce() -> String>(&self, offset: i32, default: F) -> String {
        if offset == 0 {
            default()
        } else {
            let opts = self as *const _ as usize;
            let value =
                unsafe { CStr::from_ptr((opts + offset as usize) as *const std::os::raw::c_char) };
            value.to_str().unwrap().to_owned()
        }
    }

    /// Target rows per WARM segment.
    pub fn get_warm_segment_target_rows(&self) -> u64 {
        self.warm_segment_target_rows.max(1) as u64
    }

    /// Target rows per COLD segment.
    pub fn get_cold_segment_target_rows(&self) -> u64 {
        self.cold_segment_target_rows.max(1) as u64
    }

    /// RaBitQ bits per dimension for WARM/COLD segments.
    pub fn get_rabitq_bits(&self) -> u8 {
        if self.rabitq_bits < 1 || self.rabitq_bits > 8 {
            panic!("rabitq_bits must be between 1 and 8");
        }
        self.rabitq_bits as u8
    }

    /// IVF lists per WARM/COLD segment.
    pub fn get_ivf_lists(&self) -> u16 {
        if self.ivf_lists < 1 || self.ivf_lists > 32768 {
            panic!("ivf_lists must be between 1 and 32768");
        }
        self.ivf_lists as u16
    }

    /// Lists probed per query.
    pub fn get_ivf_probes(&self) -> u16 {
        if self.ivf_probes < 1 || self.ivf_probes > 32768 {
            panic!("ivf_probes must be between 1 and 32768");
        }
        self.ivf_probes as u16
    }

    /// Maximum segments the router activates.
    pub fn get_router_top_m(&self) -> u32 {
        if self.router_top_m < 1 {
            panic!("router_top_m must be >= 1");
        }
        self.router_top_m as u32
    }

    /// Maximum segments activated per group.
    pub fn get_router_group_top_m(&self) -> u32 {
        if self.router_group_top_m < 1 {
            panic!("router_group_top_m must be >= 1");
        }
        self.router_group_top_m as u32
    }

    /// Rows per maintenance batch.
    pub fn get_migration_batch_rows(&self) -> u64 {
        if self.migration_batch_rows < 1 {
            panic!("migration_batch_rows must be >= 1");
        }
        self.migration_batch_rows as u64
    }

    /// Minimum milliseconds between maintenance passes.
    pub fn get_maintenance_interval(&self) -> i32 {
        if self.maintenance_interval < 0 {
            panic!("maintenance_interval must be >= 0");
        }
        self.maintenance_interval
    }

    /// Upper bound on bytes written per maintenance transaction.
    pub fn get_maintenance_max_bytes(&self) -> usize {
        if self.maintenance_max_bytes < 0 {
            panic!("maintenance_max_bytes must be >= 0");
        }
        self.maintenance_max_bytes as usize
    }

    /// Candidates rescored exactly during the rerank stage.
    pub fn get_rerank_k(&self) -> usize {
        if self.rerank_k < 1 {
            panic!("rerank_k must be >= 1");
        }
        self.rerank_k as usize
    }

    /// Per-segment candidate bound; `None` means exhaustive.
    pub fn get_search_candidates(&self) -> Option<usize> {
        if self.search_candidates < 0 {
            panic!("search_candidates must be >= 0 (0 = exhaustive)");
        }
        if self.search_candidates == 0 {
            None
        } else {
            Some(self.search_candidates as usize)
        }
    }
}

static mut RELOPT_KIND_AGENTVEC: pg_sys::relopt_kind::Type = 0;

/// Whether `ambuild` launches the per-database maintenance background worker
/// (tests turn it off; `agentvec_run_maintenance()` stays available either way).
pub static MAINTENANCE_WORKER_ENABLED: pgrx::GucSetting<bool> = pgrx::GucSetting::<bool>::new(true);

/// Register the `agentvec` reloptions.  Called from `_PG_init`.
pub unsafe fn init() {
    RELOPT_KIND_AGENTVEC = pg_sys::add_reloption_kind();

    pg_sys::add_string_reloption(
        RELOPT_KIND_AGENTVEC,
        "hot_storage_layout".as_pg_cstr(),
        "Storage layout of a HOT HNSW segment: plain, ieeefp16, ieeefp8, f8, sq8, sq16"
            .as_pg_cstr(),
        DEFAULT_HOT_STORAGE_LAYOUT_STR.as_pg_cstr(),
        Some(validate_hot_storage_layout),
        pg_sys::AccessExclusiveLock as pg_sys::LOCKMODE,
    );
    add_int_reloption(
        "hot_m",
        "HNSW m of a HOT segment (neighbors per layer-0 node)",
        DEFAULT_HOT_M,
        2,
        100,
    );
    add_int_reloption(
        "hot_ef_construction",
        "HNSW ef_construction of a HOT segment",
        DEFAULT_HOT_EF_CONSTRUCTION,
        4,
        1000,
    );
    add_int_reloption(
        "hot_segment_max_rows",
        "Rows a HOT segment absorbs before it is sealed",
        DEFAULT_HOT_SEGMENT_MAX_ROWS,
        1,
        2_000_000_000,
    );
    add_int_reloption(
        "hot_segment_count",
        "HOT segments that accept inserts concurrently (1 = the original \
         single appendable segment; 0 = adapt to the observed write load)",
        DEFAULT_HOT_SEGMENT_COUNT,
        0,
        MAX_HOT_SEGMENT_COUNT as i32,
    );
    add_int_reloption(
        "warm_segment_target_rows",
        "Target row count of a WARM segment (reserved: phase 6)",
        DEFAULT_WARM_SEGMENT_TARGET_ROWS,
        1,
        2_000_000_000,
    );
    add_int_reloption(
        "cold_segment_target_rows",
        "Target row count of a COLD segment (reserved: phase 6)",
        DEFAULT_COLD_SEGMENT_TARGET_ROWS,
        1,
        2_000_000_000,
    );
    add_int_reloption(
        "rabitq_bits",
        "RaBitQ bits per dimension for WARM/COLD segments (reserved: phase 3)",
        DEFAULT_RABITQ_BITS,
        1,
        8,
    );
    add_int_reloption(
        "ivf_lists",
        "IVF lists (centroids) per WARM/COLD segment (reserved: phase 3)",
        DEFAULT_IVF_LISTS,
        1,
        32768,
    );
    add_int_reloption(
        "ivf_probes",
        "IVF lists probed per query (reserved: phase 3)",
        DEFAULT_IVF_PROBES,
        1,
        32768,
    );
    add_int_reloption(
        "router_top_m",
        "Maximum segments the router activates per query (phase 7)",
        DEFAULT_ROUTER_TOP_M,
        1,
        32768,
    );
    add_int_reloption(
        "router_group_top_m",
        "Maximum segments the router activates per logical group (reserved: phase 7)",
        DEFAULT_ROUTER_GROUP_TOP_M,
        1,
        32768,
    );
    add_int_reloption(
        "migration_batch_rows",
        "Rows per maintenance batch (reserved: phase 4)",
        DEFAULT_MIGRATION_BATCH_ROWS,
        1,
        2_000_000_000,
    );
    add_int_reloption(
        "maintenance_interval",
        "Minimum milliseconds between maintenance passes (reserved: phase 4)",
        DEFAULT_MAINTENANCE_INTERVAL,
        0,
        2_000_000_000,
    );
    add_int_reloption(
        "maintenance_max_bytes",
        "Bytes a single maintenance transaction may write (reserved: phase 4)",
        DEFAULT_MAINTENANCE_MAX_BYTES,
        0,
        2_000_000_000,
    );
    add_int_reloption(
        "rerank_k",
        "Candidates rescored exactly during the rerank stage (reserved: phase 8)",
        DEFAULT_RERANK_K,
        1,
        2_000_000_000,
    );
    add_int_reloption(
        "search_candidates",
        "Candidates kept per segment scan (0 = exhaustive/exact)",
        DEFAULT_SEARCH_CANDIDATES,
        0,
        2_000_000_000,
    );

    pgrx::GucRegistry::define_bool_guc(
        c"agentvec.maintenance_worker",
        c"Run the agentvec maintenance background worker",
        c"When off, maintenance runs only through agentvec_run_maintenance().",
        &MAINTENANCE_WORKER_ENABLED,
        pgrx::GucContext::Userset,
        pgrx::GucFlags::default(),
    );
}

/// Register one int reloption of the `agentvec` kind.
unsafe fn add_int_reloption(name: &str, desc: &str, default: i32, min: i32, max: i32) {
    pg_sys::add_int_reloption(
        RELOPT_KIND_AGENTVEC,
        name.as_pg_cstr(),
        desc.as_pg_cstr(),
        default,
        min,
        max,
        pg_sys::AccessExclusiveLock as pg_sys::LOCKMODE,
    );
}

/// Reject unknown `hot_storage_layout` values at CREATE INDEX time.
#[pg_guard]
extern "C-unwind" fn validate_hot_storage_layout(value: *const std::os::raw::c_char) {
    if value.is_null() {
        return;
    }
    let value = unsafe { CStr::from_ptr(value) }
        .to_str()
        .expect("failed to parse hot_storage_layout value");
    _ = HnswPrecision::parse(value);
}

/// Parse the `WITH (...)` options of an `agentvec` index.
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

    let tab: [pg_sys::relopt_parse_elt; 17] = [
        make_relopt_parse_elt(
            "hot_storage_layout",
            pg_sys::relopt_type::RELOPT_TYPE_STRING,
            offset_of!(TSVAgentVecOptions, hot_storage_layout_offset) as i32,
        ),
        make_relopt_parse_elt(
            "hot_m",
            pg_sys::relopt_type::RELOPT_TYPE_INT,
            offset_of!(TSVAgentVecOptions, hot_m) as i32,
        ),
        make_relopt_parse_elt(
            "hot_ef_construction",
            pg_sys::relopt_type::RELOPT_TYPE_INT,
            offset_of!(TSVAgentVecOptions, hot_ef_construction) as i32,
        ),
        make_relopt_parse_elt(
            "hot_segment_max_rows",
            pg_sys::relopt_type::RELOPT_TYPE_INT,
            offset_of!(TSVAgentVecOptions, hot_segment_max_rows) as i32,
        ),
        make_relopt_parse_elt(
            "hot_segment_count",
            pg_sys::relopt_type::RELOPT_TYPE_INT,
            offset_of!(TSVAgentVecOptions, hot_segment_count) as i32,
        ),
        make_relopt_parse_elt(
            "warm_segment_target_rows",
            pg_sys::relopt_type::RELOPT_TYPE_INT,
            offset_of!(TSVAgentVecOptions, warm_segment_target_rows) as i32,
        ),
        make_relopt_parse_elt(
            "cold_segment_target_rows",
            pg_sys::relopt_type::RELOPT_TYPE_INT,
            offset_of!(TSVAgentVecOptions, cold_segment_target_rows) as i32,
        ),
        make_relopt_parse_elt(
            "rabitq_bits",
            pg_sys::relopt_type::RELOPT_TYPE_INT,
            offset_of!(TSVAgentVecOptions, rabitq_bits) as i32,
        ),
        make_relopt_parse_elt(
            "ivf_lists",
            pg_sys::relopt_type::RELOPT_TYPE_INT,
            offset_of!(TSVAgentVecOptions, ivf_lists) as i32,
        ),
        make_relopt_parse_elt(
            "ivf_probes",
            pg_sys::relopt_type::RELOPT_TYPE_INT,
            offset_of!(TSVAgentVecOptions, ivf_probes) as i32,
        ),
        make_relopt_parse_elt(
            "router_top_m",
            pg_sys::relopt_type::RELOPT_TYPE_INT,
            offset_of!(TSVAgentVecOptions, router_top_m) as i32,
        ),
        make_relopt_parse_elt(
            "router_group_top_m",
            pg_sys::relopt_type::RELOPT_TYPE_INT,
            offset_of!(TSVAgentVecOptions, router_group_top_m) as i32,
        ),
        make_relopt_parse_elt(
            "migration_batch_rows",
            pg_sys::relopt_type::RELOPT_TYPE_INT,
            offset_of!(TSVAgentVecOptions, migration_batch_rows) as i32,
        ),
        make_relopt_parse_elt(
            "maintenance_interval",
            pg_sys::relopt_type::RELOPT_TYPE_INT,
            offset_of!(TSVAgentVecOptions, maintenance_interval) as i32,
        ),
        make_relopt_parse_elt(
            "maintenance_max_bytes",
            pg_sys::relopt_type::RELOPT_TYPE_INT,
            offset_of!(TSVAgentVecOptions, maintenance_max_bytes) as i32,
        ),
        make_relopt_parse_elt(
            "rerank_k",
            pg_sys::relopt_type::RELOPT_TYPE_INT,
            offset_of!(TSVAgentVecOptions, rerank_k) as i32,
        ),
        make_relopt_parse_elt(
            "search_candidates",
            pg_sys::relopt_type::RELOPT_TYPE_INT,
            offset_of!(TSVAgentVecOptions, search_candidates) as i32,
        ),
    ];

    let rdopts = pg_sys::build_reloptions(
        reloptions,
        validate,
        RELOPT_KIND_AGENTVEC,
        std::mem::size_of::<TSVAgentVecOptions>(),
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
    unsafe fn test_agentvec_options_defaults() -> spi::Result<()> {
        Spi::run(
            "CREATE TABLE test_options_defaults(encoding vector(3));
             CREATE INDEX idxtest_options_defaults
                   ON test_options_defaults
                USING agentvec(encoding);",
        )?;

        let index_oid =
            Spi::get_one::<pg_sys::Oid>("SELECT 'idxtest_options_defaults'::regclass::oid")?
                .expect("oid was null");
        let indexrel = PgRelation::from_pg(pg_sys::RelationIdGetRelation(index_oid));
        let options = TSVAgentVecOptions::from_relation(&indexrel);

        assert_eq!(options.get_hot_segment_max_rows(), 50_000);
        assert_eq!(options.get_rabitq_bits(), 1);
        assert_eq!(options.get_search_candidates(), None);
        assert_eq!(options.get_rerank_k(), 100);
        Ok(())
    }

    #[pg_test]
    unsafe fn test_agentvec_options_custom() -> spi::Result<()> {
        Spi::run(
            "CREATE TABLE test_options_custom(encoding vector(3));
             CREATE INDEX idxtest_options_custom
                   ON test_options_custom
                USING agentvec(encoding)
                WITH (hot_segment_max_rows=7, search_candidates=25, rabitq_bits=4);",
        )?;

        let index_oid =
            Spi::get_one::<pg_sys::Oid>("SELECT 'idxtest_options_custom'::regclass::oid")?
                .expect("oid was null");
        let indexrel = PgRelation::from_pg(pg_sys::RelationIdGetRelation(index_oid));
        let options = TSVAgentVecOptions::from_relation(&indexrel);

        assert_eq!(options.get_hot_segment_max_rows(), 7);
        assert_eq!(options.get_search_candidates(), Some(25));
        assert_eq!(options.get_rabitq_bits(), 4);
        Ok(())
    }
}
