//! hnswsq2 — the pgvector-core port (see the module docs of each file).
//!
//! This module builds the new engine under its own access method name
//! (`hnswsq2`) so both engines coexist in one cluster for the A/B parity
//! gates.  At retirement (gate 5) the module is renamed `hnswsq` in place,
//! the AM/opclass SQL below is renamed accordingly, and the old engine is
//! deleted.

pub mod build;
pub mod insert;
pub mod options;
pub mod ptr;
pub mod scan;
pub mod types;
pub mod utils;
pub mod vacuum;
#[cfg(any(test, feature = "pg_test"))]
mod tests;

use pgrx::*;

use crate::access_method::distance::{
    distance_type_cosine, distance_type_inner_product, distance_type_l2,
};

/// hnswsq2 access method support function numbers:
///   1 = distance type function (matches the diskann/ivf/hnswsq convention)
pub const HNSW2_DISTANCE_TYPE_PROC: u16 = 1;

/// The main access method handler.
#[pg_extern(sql = "
    CREATE OR REPLACE FUNCTION hnswsq2_amhandler(internal) RETURNS index_am_handler PARALLEL SAFE IMMUTABLE STRICT COST 0.0001 LANGUAGE c AS '@MODULE_PATHNAME@', '@FUNCTION_NAME@';

    DO $$
    DECLARE
        c int;
    BEGIN
        SELECT count(*)
        INTO c
        FROM pg_catalog.pg_am a
        WHERE a.amname = 'hnswsq2';

        IF c = 0 THEN
            CREATE ACCESS METHOD hnswsq2 TYPE INDEX HANDLER hnswsq2_amhandler;
        END IF;
    END;
    $$;
")]
fn hnswsq2_amhandler(_fcinfo: pg_sys::FunctionCallInfo) -> PgBox<pg_sys::IndexAmRoutine> {
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

    amroutine.amvalidate = Some(hnswsq2_validate);
    amroutine.ambuild = Some(build::ambuild);
    amroutine.ambuildempty = Some(build::ambuildempty);
    amroutine.aminsert = Some(insert::aminsert);
    amroutine.ambulkdelete = Some(vacuum::ambulkdelete);
    amroutine.amvacuumcleanup = Some(vacuum::amvacuumcleanup);
    amroutine.amcostestimate = Some(hnswsq2_amcostestimate);
    amroutine.amoptions = Some(options::amoptions);
    amroutine.ambeginscan = Some(scan::ambeginscan);
    amroutine.amrescan = Some(scan::amrescan);
    amroutine.amgettuple = Some(scan::amgettuple);
    amroutine.amgetbitmap = None;
    amroutine.amendscan = Some(scan::amendscan);

    amroutine.into_pg_boxed()
}

// Register the operator classes for hnswsq2 (idempotent, so the same SQL works
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
    AND c.opcmethod = (SELECT oid FROM pg_catalog.pg_am am WHERE am.amname = 'hnswsq2')
    AND c.opcnamespace = (SELECT oid FROM pg_catalog.pg_namespace where nspname='@extschema@');

    SELECT count(*)
    INTO have_cos_ops
    FROM pg_catalog.pg_opclass c
    WHERE c.opcname = 'vector_cosine_ops'
    AND c.opcmethod = (SELECT oid FROM pg_catalog.pg_am am WHERE am.amname = 'hnswsq2')
    AND c.opcnamespace = (SELECT oid FROM pg_catalog.pg_namespace where nspname='@extschema@');

    SELECT count(*)
    INTO have_ip_ops
    FROM pg_catalog.pg_opclass c
    WHERE c.opcname = 'vector_ip_ops'
    AND c.opcmethod = (SELECT oid FROM pg_catalog.pg_am am WHERE am.amname = 'hnswsq2')
    AND c.opcnamespace = (SELECT oid FROM pg_catalog.pg_namespace where nspname='@extschema@');

    IF have_l2_ops = 0 THEN
        CREATE OPERATOR CLASS vector_l2_ops DEFAULT
        FOR TYPE vector USING hnswsq2 AS
            OPERATOR 1 <-> (vector, vector) FOR ORDER BY float_ops,
            FUNCTION 1 distance_type_l2();
    END IF;

    IF have_cos_ops = 0 THEN
        CREATE OPERATOR CLASS vector_cosine_ops
        FOR TYPE vector USING hnswsq2 AS
            OPERATOR 1 <=> (vector, vector) FOR ORDER BY float_ops,
            FUNCTION 1 distance_type_cosine();
    END IF;

    IF have_ip_ops = 0 THEN
        CREATE OPERATOR CLASS vector_ip_ops
        FOR TYPE vector USING hnswsq2 AS
            OPERATOR 1 <#> (vector, vector) FOR ORDER BY float_ops,
            FUNCTION 1 distance_type_inner_product();
    END IF;
END;
$$;
"#,
    name = "hnswsq2_operator_classes",
    requires = [
        hnswsq2_amhandler,
        distance_type_cosine,
        distance_type_l2,
        distance_type_inner_product
    ]
);

/// Validate the operator class.
#[pg_guard]
pub extern "C-unwind" fn hnswsq2_validate(opclassoid: pg_sys::Oid) -> bool {
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

/// Test-build-only index health diagnostic: walk the graph pages and report
/// element/neighbor tuple counts plus directed layer-0 reachability from the
/// entry point.  Exposed as SQL so raw-client tests can inspect vacuum
/// outcomes.
#[cfg(any(test, feature = "pg_test"))]
#[pg_extern]
fn hnswsq2_diag(index: PgRelation) -> String {
    use crate::access_method::hnswsq2::types::*;
    use crate::access_method::hnswsq2::utils::*;
    use crate::util::ports::{PageGetItem, PageGetItemId, PageGetMaxOffsetNumber};
    use std::collections::{HashSet, VecDeque};

    let index_rel = index.as_ptr();
    let mut total = 0u64;
    let mut live = 0u64;
    let mut deleted = 0u64;
    let mut invalid_refs = 0u64;
    let mut head = pg_sys::InvalidBlockNumber;
    let mut entry = None;
    unsafe {
        let buf = pg_sys::ReadBuffer(index_rel, METAPAGE_BLKNO);
        pg_sys::LockBuffer(buf, pg_sys::BUFFER_LOCK_SHARE as i32);
        let page = pg_sys::BufferGetPage(buf);
        let metap = page_get_meta(page);
        head = (*metap).graph_head;
        if (*metap).entry_blkno != pg_sys::InvalidBlockNumber {
            entry = Some(crate::util::ItemPointer::new(
                (*metap).entry_blkno,
                (*metap).entry_offno,
            ));
        }
        pg_sys::UnlockReleaseBuffer(buf);
    }

    let mut blkno = head;
    unsafe {
        while blkno != pg_sys::InvalidBlockNumber {
            let buf = pg_sys::ReadBuffer(index_rel, blkno);
            pg_sys::LockBuffer(buf, pg_sys::BUFFER_LOCK_SHARE as i32);
            let page = pg_sys::BufferGetPage(buf);
            let maxoff = PageGetMaxOffsetNumber(page);
            for off in 1..=maxoff as pg_sys::OffsetNumber {
                let item_id = PageGetItemId(page, off);
                if (*item_id).lp_len() == 0 {
                    continue;
                }
                let item = PageGetItem(page, item_id);
                let tup = item.cast::<ElementTupleData>();
                if (*tup).type_ == ELEMENT_TUPLE_TYPE {
                    total += 1;
                    if (*tup).deleted != 0 {
                        deleted += 1;
                    } else if pgrx::itemptr::item_pointer_get_block_number_no_check(
                        (*tup).heaptid,
                    ) != pg_sys::InvalidBlockNumber
                    {
                        live += 1;
                    } else {
                        invalid_refs += 1;
                    }
                }
            }
            blkno = (*page_opaque(page)).nextblkno;
            pg_sys::UnlockReleaseBuffer(buf);
        }
    }

    // Layer-0 reachability from the entry point (BFS over neighbor tuples).
    let mut reach = 0u64;
    let mut bad_tids = 0u64;
    let mut nlists = 0u64;
    let mut max_list = 0u64;
    if let Some(ep) = entry {
        let mut seen: HashSet<crate::util::ItemPointer> = HashSet::new();
        let mut queue: VecDeque<crate::util::ItemPointer> = VecDeque::new();
        seen.insert(ep);
        queue.push_back(ep);
        unsafe {
            while let Some(p) = queue.pop_front() {
                let mut elem = init_element_from_block(p.block_number, p.offset);
                let support = init_support(index_rel);
                let mut dist = 0.0f32;
                let ok = load_element_impl(
                    p.block_number,
                    p.offset,
                    Some(&mut dist),
                    None,
                    index_rel,
                    &support,
                    true,
                    None,
                    Some(&mut *elem),
                    None,
                );
                if ok.is_none() {
                    bad_tids += 1;
                    continue;
                }
                reach += 1;
                let mut tids = vec![pg_sys::ItemPointerData::default(); get_layer_m(
                    get_m(index_rel),
                    0,
                )];
                let m = get_m(index_rel);
                if load_neighbor_tids(
                    &mut *elem,
                    &mut tids,
                    index_rel,
                    m,
                    get_layer_m(m, 0),
                    0,
                ) {
                    nlists += 1;
                    let mut len = 0u64;
                    for t in &tids {
                        if pgrx::itemptr::item_pointer_get_block_number_no_check(*t)
                            == pg_sys::InvalidBlockNumber
                        {
                            break;
                        }
                        len += 1;
                        let np =
                            crate::util::ItemPointer::new(ip_block(t), ip_offset(t));
                        if seen.insert(np) {
                            queue.push_back(np);
                        }
                    }
                    max_list = max_list.max(len);
                }
            }
        }
    }
    format!(
        "total={} live={} deleted={} invalid={} head={} entry={:?} reach={} bad_tids={} nlists={} max_list={}",
        total,
        live,
        deleted,
        invalid_refs,
        head,
        entry,
        reach,
        bad_tids,
        nlists,
        max_list
    )
}

/// Test-build-only raw page dump: every element tuple's offset/heaptid/level
/// and the first few neighbor TIDs of its layer-0 list.
#[cfg(any(test, feature = "pg_test"))]
#[pg_extern]
fn hnswsq2_dump(index: PgRelation) -> String {
    use crate::access_method::hnswsq2::types::*;
    use crate::access_method::hnswsq2::utils::*;
    use crate::util::ports::{PageGetItem, PageGetItemId, PageGetMaxOffsetNumber};

    let index_rel = index.as_ptr();
    let mut out = String::new();
    unsafe {
        let mut head = pg_sys::InvalidBlockNumber;
        {
            let buf = pg_sys::ReadBuffer(index_rel, METAPAGE_BLKNO);
            pg_sys::LockBuffer(buf, pg_sys::BUFFER_LOCK_SHARE as i32);
            let page = pg_sys::BufferGetPage(buf);
            let metap = page_get_meta(page);
            head = (*metap).graph_head;
            out.push_str(&format!(
                "meta: precision={} m={} ef={} dims={}\n",
                (*metap).precision,
                (*metap).m,
                (*metap).ef_construction,
                (*metap).dimensions
            ));
            pg_sys::UnlockReleaseBuffer(buf);
        }
        let mut blkno = head;
        let support = init_support(index_rel);
        let m = get_m(index_rel);
        while blkno != pg_sys::InvalidBlockNumber {
            let buf = pg_sys::ReadBuffer(index_rel, blkno);
            pg_sys::LockBuffer(buf, pg_sys::BUFFER_LOCK_SHARE as i32);
            let page = pg_sys::BufferGetPage(buf);
            let maxoff = PageGetMaxOffsetNumber(page);
            for off in 1..=maxoff as pg_sys::OffsetNumber {
                let item_id = PageGetItemId(page, off);
                if (*item_id).lp_len() == 0 {
                    continue;
                }
                let item = PageGetItem(page, item_id);
                let tup = item.cast::<ElementTupleData>();
                if (*tup).type_ == ELEMENT_TUPLE_TYPE {
                    let hb = ip_block(&(*tup).heaptid);
                    let ho = ip_offset(&(*tup).heaptid);
                    // Read the neighbor tuple's layer-0 section.
                    let mut elem = init_element_from_block(blkno, off);
                    load_element_from_tuple(&mut *elem, tup, true, true, support.codec.vector_bytes());
                    let mut tids = vec![pg_sys::ItemPointerData::default(); get_layer_m(m, 0)];
                    let mut nids = String::from("-");
                    if load_neighbor_tids(
                        &mut *elem,
                        &mut tids,
                        index_rel,
                        m,
                        get_layer_m(m, 0),
                        0,
                    ) {
                        let parts: Vec<String> = tids
                            .iter()
                            .take_while(|t| ip_block(t) != pg_sys::InvalidBlockNumber)
                            .map(|t| format!("{}/{}", ip_block(t), ip_offset(t)))
                            .collect();
                        nids = parts.join(",");
                    }
                    out.push_str(&format!(
                        "blk={} off={} heap={}/{} lvl={} np={}/{} n0=[{}]\n",
                        blkno,
                        off,
                        hb,
                        ho,
                        (*tup).level,
                        (*elem).neighbor_page,
                        (*elem).neighbor_offno,
                        nids
                    ));
                    // Raw first bytes of the element tuple's neighbor tuple
                    // (debugging aid): header + first few tids as hex.
                    if (*tup).level == 0 {
                        let mut nt = init_element_from_block(blkno, off);
                        load_element_from_tuple(&mut *nt, tup, false, false, support.codec.vector_bytes());
                        let nbuf = pg_sys::ReadBuffer(index_rel, (*nt).neighbor_page);
                        pg_sys::LockBuffer(nbuf, pg_sys::BUFFER_LOCK_SHARE as i32);
                        let npage = pg_sys::BufferGetPage(nbuf);
                        let nitem = PageGetItem(npage, PageGetItemId(npage, (*nt).neighbor_offno));
                        let raw = std::slice::from_raw_parts(nitem, 4 + 6 * 8);
                        out.push_str(&format!("  raw={:02x?}\n", &raw[..40]));
                        pg_sys::UnlockReleaseBuffer(nbuf);
                    }
                }
            }
            blkno = (*page_opaque(page)).nextblkno;
            pg_sys::UnlockReleaseBuffer(buf);
        }
    }
    out
}

/// Cost estimate: hnswsq2 only answers `ORDER BY <distance>` searches —
/// without orderby keys it would return at most `ef_search` approximate
/// candidates, so refuse the estimate (the same guard the old AM uses).
#[pg_guard(immutable, parallel_safe)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C-unwind" fn hnswsq2_amcostestimate(
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
            if !path.is_null() {
                (*path).path.disabled_nodes = 2;
            }
        }
        return;
    }

    // Rough model: a search visits ~ef_search nodes (random-ish page reads)
    // and the executor rechecks each candidate against the heap.
    let ef = options::HNSW2_EF_SEARCH.get() as f64;
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
