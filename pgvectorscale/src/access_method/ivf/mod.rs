//! IVF (Inverted File) access method implementation.
//!
//! This module registers the `ivf` access method with PostgreSQL, providing
//! an inverted-file index for approximate nearest neighbor search.

pub mod build;
pub mod centroid;
pub mod centroid_page;
pub mod entry;
pub mod insert;
pub mod list_directory;
pub mod meta_page;
pub mod options;
pub mod scan;
pub mod segment;
pub mod simd;
pub mod vacuum;

use pgrx::*;

use crate::access_method::distance::{
    distance_type_cosine, distance_type_inner_product, distance_type_l2,
};

/// IVF access method support function numbers.
/// Matches pgvector's IVF support function layout:
///   1 = distance function
///   2 = K-means distance (optional)
///   3 = ... (reserved)
///   4 = ... (reserved)
///   5 = ... (reserved)
pub const IVF_DISTANCE_PROC: u16 = 1;

/// The main access method handler. Returns an `IndexAmRoutine` that tells
/// PostgreSQL about the capabilities and callbacks of the IVF index.
#[pg_extern(sql = "
    CREATE OR REPLACE FUNCTION ivf_amhandler(internal) RETURNS index_am_handler PARALLEL SAFE IMMUTABLE STRICT COST 0.0001 LANGUAGE c AS '@MODULE_PATHNAME@', '@FUNCTION_NAME@';

    DO $$
    DECLARE
        c int;
    BEGIN
        SELECT count(*)
        INTO c
        FROM pg_catalog.pg_am a
        WHERE a.amname = 'ivf';

        IF c = 0 THEN
            CREATE ACCESS METHOD ivf TYPE INDEX HANDLER ivf_amhandler;
        END IF;
    END;
    $$;
")]
fn ivf_amhandler(_fcinfo: pg_sys::FunctionCallInfo) -> PgBox<pg_sys::IndexAmRoutine> {
    warning!("IVF handler: entering ivf_amhandler");
    
    let mut amroutine =
        unsafe { PgBox::<pg_sys::IndexAmRoutine>::alloc_node(pg_sys::NodeTag::T_IndexAmRoutine) };

    // IVF has no search strategies (like pgvector's ivfflat)
    amroutine.amstrategies = 0;
    // 5 support functions matching pgvector IVF
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
    amroutine.amcanparallel = false;
    amroutine.amcaninclude = false;
    amroutine.amoptsprocnum = 0;
    amroutine.amusemaintenanceworkmem = true; // IVF uses maintenance_work_mem during build
    amroutine.amkeytype = pg_sys::InvalidOid;

    // Wire up callback functions
    amroutine.amvalidate = Some(ivf_validate);
    amroutine.ambuild = Some(build::ambuild);
    amroutine.ambuildempty = Some(build::ambuildempty);
    amroutine.aminsert = Some(insert::aminsert);
    amroutine.ambulkdelete = Some(vacuum::ambulkdelete);
    amroutine.amvacuumcleanup = Some(vacuum::amvacuumcleanup);
    amroutine.amcostestimate = Some(ivf_amcostestimate);
    amroutine.amoptions = Some(options::amoptions);
    amroutine.ambeginscan = Some(scan::ambeginscan);
    amroutine.amrescan = Some(scan::amrescan);
    amroutine.amgettuple = Some(scan::amgettuple);
    amroutine.amgetbitmap = None;
    amroutine.amendscan = Some(scan::amendscan);

    amroutine.into_pg_boxed()
}

/// Register the operator classes for the IVF access method (idempotent, so the
/// same SQL works for install and upgrade).
extension_sql!(
    r#"
DO $$
DECLARE
    have_cos_ops int;
    have_l2_ops int;
    have_ip_ops int;
BEGIN
    -- Has cosine operator class been installed for IVF?
    SELECT count(*)
    INTO have_cos_ops
    FROM pg_catalog.pg_opclass c
    WHERE c.opcname = 'vector_cosine_ops'
    AND c.opcmethod = (SELECT oid FROM pg_catalog.pg_am am WHERE am.amname = 'ivf')
    AND c.opcnamespace = (SELECT oid FROM pg_catalog.pg_namespace where nspname='@extschema@');

    -- Has L2 operator class been installed for IVF?
    SELECT count(*)
    INTO have_l2_ops
    FROM pg_catalog.pg_opclass c
    WHERE c.opcname = 'vector_l2_ops'
    AND c.opcmethod = (SELECT oid FROM pg_catalog.pg_am am WHERE am.amname = 'ivf')
    AND c.opcnamespace = (SELECT oid FROM pg_catalog.pg_namespace where nspname='@extschema@');

    -- Has inner product operator class been installed for IVF?
    SELECT count(*)
    INTO have_ip_ops
    FROM pg_catalog.pg_opclass c
    WHERE c.opcname = 'vector_ip_ops'
    AND c.opcmethod = (SELECT oid FROM pg_catalog.pg_am am WHERE am.amname = 'ivf')
    AND c.opcnamespace = (SELECT oid FROM pg_catalog.pg_namespace where nspname='@extschema@');

    IF have_cos_ops = 0 THEN
        CREATE OPERATOR CLASS vector_cosine_ops
        FOR TYPE vector USING ivf AS
            OPERATOR 1 <=> (vector, vector) FOR ORDER BY float_ops,
            FUNCTION 1 distance_type_cosine();
    END IF;

    IF have_l2_ops = 0 THEN
        CREATE OPERATOR CLASS vector_l2_ops
        FOR TYPE vector USING ivf AS
            OPERATOR 1 <-> (vector, vector) FOR ORDER BY float_ops,
            FUNCTION 1 distance_type_l2();
    END IF;

    IF have_ip_ops = 0 THEN
        CREATE OPERATOR CLASS vector_ip_ops
        FOR TYPE vector USING ivf AS
            OPERATOR 1 <#> (vector, vector) FOR ORDER BY float_ops,
            FUNCTION 1 distance_type_inner_product();
    END IF;
END;
$$;
"#,
    name = "ivf_operator_classes",
    requires = [
        ivf_amhandler,
        distance_type_cosine,
        distance_type_l2,
        distance_type_inner_product
    ]
);

/// Validate the operator class. Always returns true for now.
#[pg_guard]
pub extern "C-unwind" fn ivf_validate(_opclassoid: pg_sys::Oid) -> bool {
    true
}

/// Cost estimate placeholder for the IVF access method.
#[pg_guard(immutable, parallel_safe)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C-unwind" fn ivf_amcostestimate(
    root: *mut pg_sys::PlannerInfo,
    path: *mut pg_sys::IndexPath,
    loop_count: f64,
    index_startup_cost: *mut pg_sys::Cost,
    index_total_cost: *mut pg_sys::Cost,
    index_selectivity: *mut pg_sys::Selectivity,
    index_correlation: *mut f64,
    index_pages: *mut f64,
) {
    warning!("IVF amcostestimate: entering");
    
    // Initialize with safe defaults
    *index_startup_cost = 0.0;
    *index_total_cost = 0.0;
    *index_selectivity = 1.0;
    *index_correlation = 0.0;
    *index_pages = 1.0;

    // Check for null pointers
    if path.is_null() || root.is_null() {
        warning!("IVF amcostestimate: null pointers detected");
        return;
    }

    let path_ref = match path.as_ref() {
        Some(p) => p,
        None => {
            warning!("IVF amcostestimate: path.as_ref() failed");
            return;
        }
    };

    // Check if indexinfo is null
    if path_ref.indexinfo.is_null() {
        warning!("IVF amcostestimate: path.indexinfo is null");
        return;
    }

    // The ivf index only answers ORDER BY (vector distance) searches: without
    // orderby keys it would return at most ivf.top_k approximate candidates,
    // which would undercount (e.g. count(*) or plain index-only scans) and may
    // expose stale TIDs to index-only heap fetches.  Refuse to estimate so the
    // planner never chooses it for non-ORDER-BY queries.
    if path_ref.indexorderbys.is_null() || pg_sys::list_length(path_ref.indexorderbys) == 0 {
        warning!("IVF amcostestimate: no orderby keys, refusing estimate");
        *index_startup_cost = f64::MAX;
        *index_total_cost = f64::MAX;
        *index_selectivity = 1.0;
        *index_correlation = 0.0;
        *index_pages = 1.0;
        return;
    }

    let indexinfo_ref = match path_ref.indexinfo.as_ref() {
        Some(info) => info,
        None => {
            warning!("IVF amcostestimate: indexinfo.as_ref() failed");
            return;
        }
    };

    warning!("IVF amcostestimate: got indexinfo, tuples={}", indexinfo_ref.tuples);

    // Estimate cost based on index size
    let total_index_tuples = if indexinfo_ref.tuples > 0.0 {
        indexinfo_ref.tuples
    } else {
        1000.0 // Default estimate
    };

    // Simple cost model: assume we scan 1% of the index
    let num_index_tuples = total_index_tuples * 0.01;

    let mut generic_costs = pg_sys::GenericCosts {
        numIndexTuples: num_index_tuples,
        ..Default::default()
    };

    // Only call genericcostestimate if we have valid parameters
    if !root.is_null() && !path.is_null() {
        warning!("IVF amcostestimate: calling genericcostestimate");
        pg_sys::genericcostestimate(root, path, loop_count, &mut generic_costs);

        *index_startup_cost = generic_costs.indexTotalCost;
        *index_total_cost = generic_costs.indexTotalCost;
        *index_selectivity = generic_costs.indexSelectivity;
        *index_correlation = generic_costs.indexCorrelation;
        *index_pages = generic_costs.numIndexPages;
        warning!("IVF amcostestimate: completed");
    }
}
