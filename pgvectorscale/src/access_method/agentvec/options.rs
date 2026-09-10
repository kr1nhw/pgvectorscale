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
use std::fmt::Debug;

/// Vectors a HOT segment absorbs before it is sealed.  The foreground INSERT
/// that observes the threshold only performs the metadata changes that make
/// future inserts use a new HOT segment; the sealed one is consolidated
/// asynchronously.
const DEFAULT_HOT_SEGMENT_MAX_ROWS: i32 = 50_000;

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

/// Maximum segments the deterministic router activates per query (Phase 7).
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
/// `0` means "exhaustive": the exact `FLAT` executor returns every live entry
/// in distance order, so query semantics never depend on a candidate bound.
/// A positive value bounds each segment's candidate heap (the eventual ANN
/// behaviour, where the bound is what keeps search sublinear).
const DEFAULT_SEARCH_CANDIDATES: i32 = 0;

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

    pub hot_segment_max_rows: i32,
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
            ops.hot_segment_max_rows = DEFAULT_HOT_SEGMENT_MAX_ROWS;
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

/// Register the `agentvec` reloptions.  Called from `_PG_init`.
pub unsafe fn init() {
    RELOPT_KIND_AGENTVEC = pg_sys::add_reloption_kind();

    add_int_reloption(
        "hot_segment_max_rows",
        "Rows a HOT segment absorbs before it is sealed",
        DEFAULT_HOT_SEGMENT_MAX_ROWS,
        1,
        2_000_000_000,
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
        "Maximum segments the router activates per query (reserved: phase 7)",
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

    let tab: [pg_sys::relopt_parse_elt; 13] = [
        make_relopt_parse_elt(
            "hot_segment_max_rows",
            pg_sys::relopt_type::RELOPT_TYPE_INT,
            offset_of!(TSVAgentVecOptions, hot_segment_max_rows) as i32,
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

        let index_oid = Spi::get_one::<pg_sys::Oid>(
            "SELECT 'idxtest_options_defaults'::regclass::oid",
        )?
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
