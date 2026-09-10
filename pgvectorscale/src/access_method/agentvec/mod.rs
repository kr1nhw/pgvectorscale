//! `agentvec`: a multi-level (HOT / WARM / COLD) segmented vector index.
//!
//! This access method implements the storage and lifecycle skeleton of the
//! AgentVec design:
//!
//! ```text
//! block 0 : AgentVecMetaPage        single-item page, atomic read-modify-write
//! block 1 : SegmentDirectory        chained item, copy-on-write republished
//! dynamic : segment header pages    single-page atomic publication point
//! dynamic : segment payload pages   append-only entry chains (one per run)
//! ```
//!
//! The invariant the whole design rests on is that **logical identity is
//! stable while the physical representation is versioned**:
//!
//! * a segment is identified by a stable `segment_id` in the directory;
//! * readers only ever walk *published* state — the segment header's frozen
//!   runs plus its (unsealed) active chain — and the directory item they
//!   captured is immutable, so no reader needs to consult live mutable state
//!   while it executes;
//! * writers publish by atomically rewriting the single page that owns the
//!   state they changed (segment header, or the meta page when a segment is
//!   created/sealed).
//!
//! Phase 1 (this module) provides the lifecycle skeleton and one owned
//! `FLAT` segment algorithm: exact, exhaustive search over the stored
//! vectors.  `FLAT` is both the correctness baseline every later ANN
//! algorithm is measured against and the executor that makes freshly
//! committed writes immediately searchable.

pub mod build;
pub mod directory;
pub mod flat;
pub mod insert;
pub mod meta_page;
pub mod options;
pub mod scan;
#[cfg(any(test, feature = "pg_test"))]
pub mod tests;
pub mod vacuum;

use pgrx::iter::TableIterator;
use pgrx::*;

use crate::access_method::distance::{
    distance_type_cosine, distance_type_inner_product, distance_type_l2,
};

/// Support function number 1 is the distance-type function, matching
/// pgvector's opclass layout (and `diskann`'s `DISKANN_DISTANCE_TYPE_PROC`).
pub const AGENTVEC_DISTANCE_TYPE_PROC: u16 = 1;

/// The main access method handler. Returns an `IndexAmRoutine` describing the
/// capabilities and callbacks of the `agentvec` index.
#[pg_extern(sql = "
    CREATE OR REPLACE FUNCTION agentvec_amhandler(internal) RETURNS index_am_handler PARALLEL SAFE IMMUTABLE STRICT COST 0.0001 LANGUAGE c AS '@MODULE_PATHNAME@', '@FUNCTION_NAME@';

    DO $$
    DECLARE
        c int;
    BEGIN
        SELECT count(*)
        INTO c
        FROM pg_catalog.pg_am a
        WHERE a.amname = 'agentvec';

        IF c = 0 THEN
            CREATE ACCESS METHOD agentvec TYPE INDEX HANDLER agentvec_amhandler;
        END IF;
    END;
    $$;
")]
fn agentvec_amhandler(_fcinfo: pg_sys::FunctionCallInfo) -> PgBox<pg_sys::IndexAmRoutine> {
    let mut amroutine =
        unsafe { PgBox::<pg_sys::IndexAmRoutine>::alloc_node(pg_sys::NodeTag::T_IndexAmRoutine) };

    // Like pgvector's ivfflat, agentvec has no search strategies: it only
    // answers `ORDER BY <distance operator>` queries.
    amroutine.amstrategies = 0;
    // Support function 1 (distance type) is required; the remaining slots are
    // reserved so later phases can add support functions without changing the
    // declared count.
    amroutine.amsupport = 5;

    amroutine.amcanorder = false;
    amroutine.amcanorderbyop = true;
    amroutine.amcanbackward = false;
    amroutine.amcanunique = false;
    amroutine.amcanmulticol = false;
    amroutine.amoptionalkey = true;
    amroutine.amsearcharray = false;
    amroutine.amsearchnulls = false;
    amroutine.amstorage = false;
    amroutine.amclusterable = false;
    amroutine.ampredlocks = false;
    // Parallel scan/build are Phase 9: the AM callbacks for them
    // (amestimateparallelscan / aminitparallelscan / amparallelrescan) are not
    // implemented yet, so the AM must not advertise support.
    amroutine.amcanparallel = false;
    amroutine.amcanbuildparallel = false;
    amroutine.amcaninclude = false;
    // WARM/COLD builds (Phase 3) train centroids from a bounded sample, which
    // is what maintenance_work_mem bounds.
    amroutine.amusemaintenanceworkmem = true;
    amroutine.amoptsprocnum = 0;
    amroutine.amkeytype = pg_sys::InvalidOid;

    amroutine.amvalidate = Some(agentvec_validate);
    amroutine.ambuild = Some(build::ambuild);
    amroutine.ambuildempty = Some(build::ambuildempty);
    amroutine.aminsert = Some(insert::aminsert);
    amroutine.ambulkdelete = Some(vacuum::ambulkdelete);
    amroutine.amvacuumcleanup = Some(vacuum::amvacuumcleanup);
    amroutine.amcostestimate = Some(agentvec_amcostestimate);
    amroutine.amoptions = Some(options::amoptions);
    amroutine.ambeginscan = Some(scan::ambeginscan);
    amroutine.amrescan = Some(scan::amrescan);
    amroutine.amgettuple = Some(scan::amgettuple);
    amroutine.amgetbitmap = None;
    amroutine.amendscan = Some(scan::amendscan);

    amroutine.into_pg_boxed()
}

/// Register the operator classes for the `agentvec` access method.  Idempotent
/// so the same SQL works for install and upgrade.
extension_sql!(
    r#"
DO $$
DECLARE
    have_cos_ops int;
    have_l2_ops int;
    have_ip_ops int;
BEGIN
    SELECT count(*)
    INTO have_cos_ops
    FROM pg_catalog.pg_opclass c
    WHERE c.opcname = 'vector_cosine_ops'
    AND c.opcmethod = (SELECT oid FROM pg_catalog.pg_am am WHERE am.amname = 'agentvec')
    AND c.opcnamespace = (SELECT oid FROM pg_catalog.pg_namespace where nspname='@extschema@');

    SELECT count(*)
    INTO have_l2_ops
    FROM pg_catalog.pg_opclass c
    WHERE c.opcname = 'vector_l2_ops'
    AND c.opcmethod = (SELECT oid FROM pg_catalog.pg_am am WHERE am.amname = 'agentvec')
    AND c.opcnamespace = (SELECT oid FROM pg_catalog.pg_namespace where nspname='@extschema@');

    SELECT count(*)
    INTO have_ip_ops
    FROM pg_catalog.pg_opclass c
    WHERE c.opcname = 'vector_ip_ops'
    AND c.opcmethod = (SELECT oid FROM pg_catalog.pg_am am WHERE am.amname = 'agentvec')
    AND c.opcnamespace = (SELECT oid FROM pg_catalog.pg_namespace where nspname='@extschema@');

    IF have_cos_ops = 0 THEN
        CREATE OPERATOR CLASS vector_cosine_ops
        FOR TYPE vector USING agentvec AS
            OPERATOR 1 <=> (vector, vector) FOR ORDER BY float_ops,
            FUNCTION 1 distance_type_cosine();
    END IF;

    IF have_l2_ops = 0 THEN
        CREATE OPERATOR CLASS vector_l2_ops
        DEFAULT FOR TYPE vector USING agentvec AS
            OPERATOR 1 <-> (vector, vector) FOR ORDER BY float_ops,
            FUNCTION 1 distance_type_l2();
    END IF;

    IF have_ip_ops = 0 THEN
        CREATE OPERATOR CLASS vector_ip_ops
        FOR TYPE vector USING agentvec AS
            OPERATOR 1 <#> (vector, vector) FOR ORDER BY float_ops,
            FUNCTION 1 distance_type_inner_product();
    END IF;
END;
$$;
"#,
    name = "agentvec_operator_classes",
    requires = [
        agentvec_amhandler,
        distance_type_cosine,
        distance_type_l2,
        distance_type_inner_product
    ]
);

/// Validate the operator class.  Always true for now: the opclass contract is
/// just "support function 1 returns the distance type".
#[pg_guard]
pub extern "C-unwind" fn agentvec_validate(_opclassoid: pg_sys::Oid) -> bool {
    true
}

/// Inspect the segment directory of an `agentvec` index from SQL.
///
/// The directory is the index's own view of its physical segments, so this is
/// the read-only way to see what the lifecycle is doing: which segments exist,
/// at which level, in which state, and how full they are.  Counts come from the
/// segment headers (the authoritative source updated under the header lock);
/// the directory's own copies are publication-time snapshots.
#[pg_extern]
fn agentvec_index_info(
    index: PgRelation,
) -> TableIterator<
    'static,
    (
        name!(segment_id, i64),
        name!(level, String),
        name!(state, String),
        name!(algorithm, String),
        name!(ownership, String),
        name!(is_hot, bool),
        name!(generation, i64),
        name!(epoch, i64),
        name!(num_entries, i64),
        name!(live_entries, i64),
        name!(dead_entries, i64),
        name!(sealed_runs, i32),
        name!(has_active_chain, bool),
        name!(header_block, i64),
        name!(dimension, i32),
        name!(format_version, i32),
    ),
> {
    let meta = meta_page::AgentVecMetaPage::fetch(&index);
    let directory = meta.load_directory(&index);
    let hot_segment_id = meta.get_hot_segment_id();

    let rows: Vec<_> = directory
        .segments
        .iter()
        .map(|segment| {
            let header = directory::AgentVecSegmentHeader::load(&index, segment.header);
            (
                segment.segment_id as i64,
                segment.level().as_str().to_string(),
                segment.state().as_str().to_string(),
                segment.algorithm().as_str().to_string(),
                segment.ownership().as_str().to_string(),
                segment.segment_id == hot_segment_id,
                segment.generation as i64,
                segment.epoch as i64,
                header.num_entries as i64,
                header.live_entries() as i64,
                header.dead_entries as i64,
                header.sealed.len() as i32,
                header.active.is_some(),
                segment.header.block_number as i64,
                segment.dimension as i32,
                segment.format_version as i32,
            )
        })
        .collect();

    TableIterator::new(rows)
}

/// Cost estimate for the `agentvec` access method.
///
/// Two things matter here:
///
/// 1. Without ORDER BY keys the index cannot answer the query at all — it can
///    only produce a bounded candidate stream, so `count(*)` or a plain
///    index-only scan would silently undercount.  Refuse to estimate (an
///    infinite cost) so the planner never chooses it for those.
/// 2. Otherwise defer to `genericcostestimate` with the row estimate of the
///    index.  Phase 1's `FLAT` segments are scanned exhaustively, so the
///    number of index tuples examined is the whole index; later phases lower
///    this through the segment router.
#[pg_guard(immutable, parallel_safe)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C-unwind" fn agentvec_amcostestimate(
    root: *mut pg_sys::PlannerInfo,
    path: *mut pg_sys::IndexPath,
    loop_count: f64,
    index_startup_cost: *mut pg_sys::Cost,
    index_total_cost: *mut pg_sys::Cost,
    index_selectivity: *mut pg_sys::Selectivity,
    index_correlation: *mut f64,
    index_pages: *mut f64,
) {
    // NOTE: no logging in here — the planner calls this for many candidate
    // paths per query, so a WARNING per call floods the server log.
    *index_startup_cost = 0.0;
    *index_total_cost = 0.0;
    *index_selectivity = 1.0;
    *index_correlation = 0.0;
    *index_pages = 1.0;

    if path.is_null() || root.is_null() {
        return;
    }
    let path_ref = match path.as_ref() {
        Some(p) => p,
        None => return,
    };
    if path_ref.indexinfo.is_null() {
        return;
    }

    // This AM only answers ORDER BY (vector distance) searches.  Without
    // order-by keys it produces no ordered candidate stream at all, so a plan
    // that used it would silently undercount (`count(*)` would return zero
    // rows) — refuse it as expensively as possible.
    if path_ref.indexorderbys.is_null() || pg_sys::list_length(path_ref.indexorderbys) == 0 {
        *index_startup_cost = f64::INFINITY;
        *index_total_cost = f64::INFINITY;
        *index_selectivity = 0.0;
        *index_correlation = 0.0;
        *index_pages = 0.0;
        #[cfg(feature = "pg18")]
        {
            // PostgreSQL 18 replaced the `disable_cost` convention with a
            // per-path `disabled_nodes` count that is compared BEFORE cost
            // (`compare_path_costs`), so an infinite-cost path still beats a
            // path whose scan type the user disabled: with
            // `SET enable_seqscan = off` the seq scan carries one disabled
            // node and this path carried none, so `count(*)` was planned
            // through the index and returned 0.  Marking this path with two
            // disabled nodes restores the "never choose me" behaviour.  This
            // is the same fix pgvector's HNSW applies (see the pgsql-hackers
            // "On disable_cost" thread).
            (*path).path.disabled_nodes = 2;
        }
        return;
    }

    let indexinfo_ref = match path_ref.indexinfo.as_ref() {
        Some(info) => info,
        None => return,
    };

    let total_index_tuples = if indexinfo_ref.tuples > 0.0 {
        indexinfo_ref.tuples
    } else {
        1000.0
    };

    let mut generic_costs = pg_sys::GenericCosts {
        numIndexTuples: total_index_tuples,
        ..Default::default()
    };
    pg_sys::genericcostestimate(root, path, loop_count, &mut generic_costs);

    *index_startup_cost = generic_costs.indexStartupCost;
    *index_total_cost = generic_costs.indexTotalCost;
    *index_selectivity = generic_costs.indexSelectivity;
    *index_correlation = generic_costs.indexCorrelation;
    *index_pages = generic_costs.numIndexPages;
}
