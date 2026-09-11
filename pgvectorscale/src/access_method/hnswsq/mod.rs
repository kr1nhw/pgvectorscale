//! hnswsq — HNSW access method with reduced-precision node storage.
//!
//! A multi-layer HNSW graph index over pgvector `vector` columns with four
//! node-vector storage layouts: `plain` (f32), `ieeefp16` (IEEE binary16),
//! `ieeefp8` (OCP FP8 E4M3) and `f8` (Lance-style trained SQ8).  The IEEE
//! layouts are training-free, which makes them the incremental-build friendly
//! choice; SQ8 calibrates per-dimension min/max at CREATE INDEX.
//!
//! Operational model (established by the IVF-RaBitQ work in this repo):
//! - append-only node storage: nodes are written once and never moved;
//!   bounded in-place mutations (neighbor slots, tombstone flags) are atomic
//!   per page under exclusive content locks with GenericXLog WAL;
//! - transactional: visibility is the executor's MVCC snapshot check on the
//!   returned heap TIDs, ordering is rechecked exactly
//!   (`xs_recheckorderby = true`);
//! - autovacuum-ready: `ambulkdelete` tombstones + repairs + recycles pages,
//!   `amvacuumcleanup` refreshs estimates;
//! - pgvector-style concurrent inserts: no global writer lock, one content
//!   lock at a time, two-phase optimistic neighbor updates.

pub mod build;
pub mod graph;
pub mod insert;
pub mod meta_page;
pub mod node;
pub mod options;
pub mod quantize;
pub mod scan;
mod tests;
pub mod vacuum;

use pgrx::*;

use crate::access_method::distance::{
    distance_type_cosine, distance_type_inner_product, distance_type_l2,
};

/// Advisory-lock key serializing the hnswsq integration tests ACROSS
/// backends.  pg_test bodies execute inside the pgrx framework's transaction
/// in a server backend, so a process-local mutex cannot serialize them; an
/// advisory lock can.  pg_tests take the key in transaction scope (released
/// by the framework's rollback), the raw-client vacuum scaffolds hold it in
/// session scope on a dedicated connection.  The mock pg_test must NOT take
/// the lock (scaffolds invoke it while holding it).
#[cfg(any(test, feature = "pg_test"))]
pub(crate) const HNSW_SUITE_ADVISORY_KEY: i64 = 5_205_217_837_881_163_777;

#[cfg(any(test, feature = "pg_test"))]
pub(crate) fn lock_suite_for_test() {
    // Raw SPI_execute (NOT pgrx's Spi::run/SpiClient): pgrx's SpiClient
    // references PostgreSQL DATA symbols (SPI_processed/SPI_tuptable) that
    // `-undefined dynamic_lookup` cannot defer on macOS, which crashes the
    // unit-test binary at load time.  Function calls stay lazy.
    use pgrx::pg_sys::AsPgCStr;
    unsafe {
        let query = "SELECT pg_advisory_xact_lock(5205217837881163777)";
        if pg_sys::SPI_execute(query.as_pg_cstr(), false, 0) >= 0 {
            pg_sys::SPI_finish();
        }
    }
}

/// Build RNG: entropy in production, or the pinned `hnswsq.build_seed` value
/// (tests set it so builds — and recall assertions — are deterministic).
pub(crate) fn build_rng() -> rand::rngs::SmallRng {
    use rand::SeedableRng;
    let seed = options::HNSWSQ_BUILD_SEED.get();
    if seed < 0 {
        rand::rngs::SmallRng::from_entropy()
    } else {
        rand::rngs::SmallRng::seed_from_u64(seed as u64)
    }
}

/// hnswsq access method support function numbers:
///   1 = distance type function (matches the diskann/ivf convention)
pub const HNSWSQ_DISTANCE_TYPE_PROC: u16 = 1;

/// The main access method handler.
#[pg_extern(sql = "
    CREATE OR REPLACE FUNCTION hnswsq_amhandler(internal) RETURNS index_am_handler PARALLEL SAFE IMMUTABLE STRICT COST 0.0001 LANGUAGE c AS '@MODULE_PATHNAME@', '@FUNCTION_NAME@';

    DO $$
    DECLARE
        c int;
    BEGIN
        SELECT count(*)
        INTO c
        FROM pg_catalog.pg_am a
        WHERE a.amname = 'hnswsq';

        IF c = 0 THEN
            CREATE ACCESS METHOD hnswsq TYPE INDEX HANDLER hnswsq_amhandler;
        END IF;
    END;
    $$;
")]
fn hnswsq_amhandler(_fcinfo: pg_sys::FunctionCallInfo) -> PgBox<pg_sys::IndexAmRoutine> {
    let mut amroutine =
        unsafe { PgBox::<pg_sys::IndexAmRoutine>::alloc_node(pg_sys::NodeTag::T_IndexAmRoutine) };

    amroutine.amstrategies = 0;
    amroutine.amsupport = 1;

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
    amroutine.amcanparallel = false;
    amroutine.amcaninclude = false;
    amroutine.amoptsprocnum = 0;
    amroutine.amusemaintenanceworkmem = true; // the in-memory build honors it
    amroutine.amkeytype = pg_sys::InvalidOid;

    amroutine.amvalidate = Some(hnswsq_validate);
    amroutine.ambuild = Some(build::ambuild);
    amroutine.ambuildempty = Some(build::ambuildempty);
    amroutine.aminsert = Some(insert::aminsert);
    amroutine.ambulkdelete = Some(vacuum::ambulkdelete);
    amroutine.amvacuumcleanup = Some(vacuum::amvacuumcleanup);
    amroutine.amcostestimate = Some(hnswsq_amcostestimate);
    amroutine.amoptions = Some(options::amoptions);
    amroutine.ambeginscan = Some(scan::ambeginscan);
    amroutine.amrescan = Some(scan::amrescan);
    amroutine.amgettuple = Some(scan::amgettuple);
    amroutine.amgetbitmap = None;
    amroutine.amendscan = Some(scan::amendscan);

    amroutine.into_pg_boxed()
}

// Register the operator classes for hnswsq (idempotent, so the same SQL works
// for install and upgrade).  L2 is the DEFAULT opclass for the AM, following
// the pgvector hnsw convention.
extension_sql!(
    r#"
DO $$
DECLARE
    have_l2_ops int;
    have_cos_ops int;
    have_ip_ops int;
BEGIN
    SELECT count(*)
    INTO have_l2_ops
    FROM pg_catalog.pg_opclass c
    WHERE c.opcname = 'vector_l2_ops'
    AND c.opcmethod = (SELECT oid FROM pg_catalog.pg_am am WHERE am.amname = 'hnswsq')
    AND c.opcnamespace = (SELECT oid FROM pg_catalog.pg_namespace where nspname='@extschema@');

    SELECT count(*)
    INTO have_cos_ops
    FROM pg_catalog.pg_opclass c
    WHERE c.opcname = 'vector_cosine_ops'
    AND c.opcmethod = (SELECT oid FROM pg_catalog.pg_am am WHERE am.amname = 'hnswsq')
    AND c.opcnamespace = (SELECT oid FROM pg_catalog.pg_namespace where nspname='@extschema@');

    SELECT count(*)
    INTO have_ip_ops
    FROM pg_catalog.pg_opclass c
    WHERE c.opcname = 'vector_ip_ops'
    AND c.opcmethod = (SELECT oid FROM pg_catalog.pg_am am WHERE am.amname = 'hnswsq')
    AND c.opcnamespace = (SELECT oid FROM pg_catalog.pg_namespace where nspname='@extschema@');

    IF have_l2_ops = 0 THEN
        CREATE OPERATOR CLASS vector_l2_ops DEFAULT
        FOR TYPE vector USING hnswsq AS
            OPERATOR 1 <-> (vector, vector) FOR ORDER BY float_ops,
            FUNCTION 1 distance_type_l2();
    END IF;

    IF have_cos_ops = 0 THEN
        CREATE OPERATOR CLASS vector_cosine_ops
        FOR TYPE vector USING hnswsq AS
            OPERATOR 1 <=> (vector, vector) FOR ORDER BY float_ops,
            FUNCTION 1 distance_type_cosine();
    END IF;

    IF have_ip_ops = 0 THEN
        CREATE OPERATOR CLASS vector_ip_ops
        FOR TYPE vector USING hnswsq AS
            OPERATOR 1 <#> (vector, vector) FOR ORDER BY float_ops,
            FUNCTION 1 distance_type_inner_product();
    END IF;
END;
$$;
"#,
    name = "hnswsq_operator_classes",
    requires = [
        hnswsq_amhandler,
        distance_type_cosine,
        distance_type_l2,
        distance_type_inner_product
    ]
);

/// Validate the operator class.
#[pg_guard]
pub extern "C-unwind" fn hnswsq_validate(opclassoid: pg_sys::Oid) -> bool {
    unsafe {
        // Require the distance-type support function (proc 1).
        let opclass = pg_sys::SearchSysCache1(
            pg_sys::SysCacheIdentifier::CLAOID as _,
            pg_sys::Datum::from(opclassoid),
        );
        if opclass.is_null() {
            return false;
        }
        pg_sys::ReleaseSysCache(opclass);
    }
    true
}

/// Test-build-only index health diagnostic: walk the node pages and report
/// live/tombstone counts plus directed layer-0 reachability from the entry
/// point.  Exposed as SQL so raw-client tests can inspect vacuum outcomes.
#[cfg(any(test, feature = "pg_test"))]
#[pg_extern]
fn hnswsq_diag(index: PgRelation) -> String {
    use crate::access_method::hnswsq::meta_page::HnswMetaPage;
    use crate::access_method::hnswsq::node::{load_node_view, HnswNode};
    use crate::util::page::{PageType, ReadablePage};
    use crate::util::ports::{PageGetItem, PageGetItemId, PageGetMaxOffsetNumber};
    use crate::util::ItemPointer;
    use std::collections::{HashSet, VecDeque};

    let index_rel = index;
    let nblocks = unsafe {
        pg_sys::RelationGetNumberOfBlocksInFork(
            index_rel.as_ptr(),
            pg_sys::ForkNumber::MAIN_FORKNUM,
        )
    };
    let mut total = 0u64;
    let mut live = 0u64;
    for block in 1..nblocks {
        let page = unsafe { ReadablePage::read(&index_rel, block) };
        if page.get_type() != PageType::HnswNode {
            continue;
        }
        let max_off = unsafe { PageGetMaxOffsetNumber(*page) };
        for off in 1..=max_off {
            let item_id = unsafe { PageGetItemId(*page, off as pg_sys::OffsetNumber) };
            if unsafe { (*item_id).lp_flags() } != 1 || unsafe { (*item_id).lp_len() } == 0 {
                continue;
            }
            let item = unsafe { PageGetItem(*page, item_id) } as *const u8;
            let len = unsafe { (*item_id).lp_len() } as usize;
            let node =
                unsafe { rkyv::archived_root::<HnswNode>(std::slice::from_raw_parts(item, len)) };
            total += 1;
            if !node.is_deleted() {
                live += 1;
            }
        }
    }
    let meta = HnswMetaPage::fetch(&index_rel);
    let mut reach_total = 0u64;
    let mut reach_live = 0u64;
    if let Some(ep) = meta.get_entry_point() {
        let mut seen: HashSet<ItemPointer> = HashSet::new();
        let mut queue: VecDeque<ItemPointer> = VecDeque::new();
        seen.insert(ep);
        queue.push_back(ep);
        while let Some(p) = queue.pop_front() {
            if let Some(v) = load_node_view(&index_rel, p) {
                reach_total += 1;
                if !v.deleted {
                    reach_live += 1;
                }
                for n in v.neighbors.first().cloned().unwrap_or_default() {
                    if seen.insert(n) {
                        queue.push_back(n);
                    }
                }
            }
        }
    }
    format!(
        "total={} live={} tomb={} reach_total={} reach_live={} entry_level={}",
        total,
        live,
        total - live,
        reach_total,
        reach_live,
        meta.get_entry_level()
    )
}

/// Cost estimate: hnswsq only answers `ORDER BY <distance>` searches — without
/// orderby keys it would return at most `ef_search` approximate candidates and
/// could expose stale TIDs to index-only fetches, so refuse the estimate (the
/// same guard the IVF AM uses).
#[pg_guard(immutable, parallel_safe)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C-unwind" fn hnswsq_amcostestimate(
    root: *mut pg_sys::PlannerInfo,
    path: *mut pg_sys::IndexPath,
    loop_count: f64,
    index_startup_cost: *mut pg_sys::Cost,
    index_total_cost: *mut pg_sys::Cost,
    index_selectivity: *mut pg_sys::Selectivity,
    index_correlation: *mut f64,
    index_pages: *mut f64,
) {
    if path.is_null()
        || (*path).indexorderbys.is_null()
        || pg_sys::list_length((*path).indexorderbys) == 0
    {
        *index_startup_cost = f64::MAX;
        *index_total_cost = f64::MAX;
        *index_selectivity = 1.0;
        *index_correlation = 0.0;
        *index_pages = 1.0;
        #[cfg(feature = "pg18")]
        {
            // Following pgvector's PG18+ cost-estimate change: mark the path
            // disabled so the planner never picks it without ORDER BY.
            if !path.is_null() {
                (*path).path.disabled_nodes = 2;
            }
        }
        return;
    }

    // Rough model: a search visits ~ef_search nodes (random-ish page reads)
    // and the executor rechecks each candidate against the heap.
    let ef = crate::access_method::hnswsq::options::HNSWSQ_EF_SEARCH.get() as f64;
    let mut generic_costs = pg_sys::GenericCosts {
        numIndexTuples: ef,
        ..Default::default()
    };
    pg_sys::genericcostestimate(root, path, loop_count, &mut generic_costs);

    *index_startup_cost = generic_costs.indexTotalCost;
    *index_total_cost = generic_costs.indexTotalCost;
    *index_selectivity = generic_costs.indexSelectivity;
    *index_correlation = generic_costs.indexCorrelation;
    *index_pages = generic_costs.numIndexPages;
}
