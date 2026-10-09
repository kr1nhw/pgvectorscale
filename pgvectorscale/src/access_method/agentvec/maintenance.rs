//! Phase 4: the AgentVec maintenance worker.
//!
//! One dynamic background worker per database that contains `agentvec`
//! indexes.  It is launched on demand from `ambuild` (the guard checks
//! `pg_stat_activity` so repeated DDL does not stack workers), connects to
//! its database via `bgw_extra`, and loops:
//!
//! ```text
//! pass: for each agentvec index, convert sealed HOT segments (bounded by
//!       migration_batch_rows, one index at a time, committed per index)
//!   -> sleep for the indexes' minimum maintenance_interval
//! ```
//!
//! The worker never exits on its own (an empty pass just sleeps again) and
//! is registered with `bgw_restart_time = -1` (no postmaster restart): a
//! crash is recovered by the next `ambuild` launch.  `SIGTERM` stops it
//! cleanly.  Under compute environments without dynamic background workers
//! (Neon), the launch fails silently and maintenance runs through the
//! synchronous `agentvec_run_maintenance()` SQL function instead.

use pgrx::bgworkers::{
    BackgroundWorker, BackgroundWorkerBuilder, BgWorkerStartTime, SignalWakeFlags,
};
use pgrx::*;
use std::ffi::CStr;
use std::time::Duration;

/// `pg_stat_activity.backend_type` of the worker.
pub const WORKER_TYPE: &str = "agentvec maintenance worker";

/// The exported entry point the postmaster loads from the extension library.
#[pg_guard]
#[unsafe(no_mangle)]
pub extern "C-unwind" fn agentvec_maintenance_worker_main(_arg: pg_sys::Datum) {
    BackgroundWorker::attach_signal_handlers(SignalWakeFlags::SIGHUP | SignalWakeFlags::SIGTERM);
    // Connect to the database named in bgw_extra explicitly (the
    // BGWORKER_BACKEND_DATABASE_CONNECTION auto-connect did not reliably
    // happen for this dynamically registered worker on PG18).
    BackgroundWorker::connect_worker_to_spi(Some(BackgroundWorker::get_extra()), None);
    let first_interval = BackgroundWorker::transaction(|| pass_interval_ms());

    // Sleep one full interval before the first pass: maintenance is
    // background work and must never race a still-running bulk load.  It
    // also keeps the worker out of the pg_test windows (the suite runs with
    // agentvec.maintenance_worker = off anyway).
    BackgroundWorker::wait_latch(Some(Duration::from_millis(first_interval)));

    // NB: pgrx 0.16's BackgroundWorker::worker_continue() is a no-op
    // constant, so the exit condition is the signal flags themselves.
    loop {
        // SPI work runs inside a worker transaction (the canonical pgrx
        // pattern): a started transaction + pushed snapshot around the pass.
        let converted = BackgroundWorker::transaction(|| maintenance_pass());
        if converted > 0 {
            pgrx::log!(
                "agentvec maintenance worker converted {converted} rows in database {}",
                BackgroundWorker::get_extra()
            );
        }

        let interval_ms = BackgroundWorker::transaction(|| pass_interval_ms());
        BackgroundWorker::wait_latch(Some(Duration::from_millis(interval_ms)));
        if BackgroundWorker::sigterm_received()
            || BackgroundWorker::sigint_received()
            || unsafe { pg_sys::ShutdownRequestPending != 0 }
        {
            break;
        }
    }
    pgrx::log!("agentvec maintenance worker exiting");
}

/// Launch the maintenance worker for the current database unless one is
/// already running.  Failures (e.g. no dynamic workers, like Neon compute)
/// are logged and ignored: the synchronous SQL entry point remains
/// available.
pub fn launch_worker_for_current_database() {
    if !super::options::MAINTENANCE_WORKER_ENABLED.get() {
        return;
    }
    // Cheap liveness guard through pg_stat_activity (the worker's backend
    // type is its bgw_type).
    let running = Spi::get_one::<bool>(
        "SELECT EXISTS (SELECT 1 FROM pg_stat_activity WHERE backend_type = 'agentvec maintenance worker' AND datname = current_database())",
    );
    if let Ok(Some(true)) = running {
        return;
    }

    let dbname = unsafe { CStr::from_ptr(pg_sys::get_database_name(pg_sys::MyDatabaseId)) }
        .to_str()
        .unwrap_or_default()
        .to_string();
    if dbname.is_empty() {
        return;
    }

    let result = BackgroundWorkerBuilder::new("agentvec maintenance worker")
        .set_type(WORKER_TYPE)
        .set_function("agentvec_maintenance_worker_main")
        .set_library(concat!(
            env!("CARGO_PKG_NAME"),
            "-",
            env!("CARGO_PKG_VERSION")
        ))
        .set_extra(&dbname)
        .enable_spi_access()
        .set_start_time(BgWorkerStartTime::ConsistentState)
        .set_restart_time(None) // no postmaster restart loop; ambuild relaunches
        .load_dynamic();
    if result.is_err() {
        pgrx::log!(
            "agentvec: could not start the maintenance worker (dynamic background workers unavailable?): maintenance stays manual via agentvec_run_maintenance()"
        );
    }
}

/// One maintenance pass: every agentvec index of this database gets
/// `agentvec_maybe_maintain` (the per-index schedule gate).  Each index runs in
/// its own subtransaction, so a failure is rolled back for that index only and
/// the pass carries on.  Returns the number of rows converted.
pub(super) fn maintenance_pass() -> i64 {
    let indexes = list_indexes();
    let mut converted_total: i64 = 0;
    for oid in indexes {
        converted_total += maintain_one_index(oid).unwrap_or(0);
    }
    converted_total
}

/// The worker's per-index step, isolated in a subtransaction.
///
/// A failing step (`agentvec_maybe_maintain` raises) must not take anything
/// else down with it.  `Spi::get_one` reports only SPI-level failures as `Err`;
/// a failing SQL statement raises a Postgres ERROR, which aborts the enclosing
/// transaction — every later statement in the pass would then fail with
/// "current transaction is aborted" — and, with no outer error handler in a
/// background worker, would turn into a FATAL that terminates the worker
/// process (which is registered without a restart).  Catching the error and
/// rolling the subtransaction back keeps the pass and the worker alive; the
/// failure is logged for this index only.  The index's maintenance schedule is
/// not advanced, so the next pass retries it.
///
/// Returns `None` when the index's step failed.
pub(super) fn maintain_one_index(oid: pg_sys::Oid) -> Option<i64> {
    unsafe {
        let saved_ctx = pg_sys::CurrentMemoryContext;
        let saved_owner = pg_sys::CurrentResourceOwner;
        pg_sys::BeginInternalSubTransaction(std::ptr::null());
        // Stay in the caller's memory context: the subtransaction's own context
        // is destroyed with it (the PL/pgSQL exception-block pattern).
        pg_sys::CurrentMemoryContext = saved_ctx;

        // `Ok(Ok(n))` is a successful step, `Ok(Err(spi))` an SPI-level failure
        // (the transaction stays usable), `Err(())` the caught Postgres error
        // the handler above has already logged and rolled back.
        let outcome: Result<Result<Option<i64>, spi::Error>, ()> = PgTryBuilder::new(|| {
            Ok(Spi::get_one::<i64>(&format!(
                "SELECT agentvec_maybe_maintain({oid}::regclass)"
            )))
        })
        .catch_others(|caught| {
            // The error left the subtransaction aborted, and may have left the
            // current memory context inside it: switch out before rolling back.
            pg_sys::CurrentMemoryContext = saved_ctx;
            pg_sys::RollbackAndReleaseCurrentSubTransaction();
            pg_sys::CurrentMemoryContext = saved_ctx;
            pg_sys::CurrentResourceOwner = saved_owner;
            pgrx::log!(
                "agentvec maintenance: index {oid} failed: {}",
                caught_message(&caught)
            );
            Err(())
        })
        .execute();

        let converted = match outcome {
            Ok(Ok(n)) => n.unwrap_or(0),
            Ok(Err(err)) => {
                // SPI-level failure: the transaction is still usable, so the
                // rest of the pass can proceed.
                pgrx::log!("agentvec maintenance: index {oid} failed: {err}");
                0
            }
            // Already logged and rolled back by the handler.
            Err(()) => return None,
        };

        pg_sys::ReleaseCurrentSubTransaction();
        pg_sys::CurrentMemoryContext = saved_ctx;
        pg_sys::CurrentResourceOwner = saved_owner;
        Some(converted)
    }
}

/// The message of a caught error.  pgrx copies the error data into Rust memory
/// when it catches, so reading it after the rollback is safe.
fn caught_message(caught: &pg_sys::panic::CaughtError) -> String {
    match caught {
        pg_sys::panic::CaughtError::PostgresError(report)
        | pg_sys::panic::CaughtError::ErrorReport(report) => report.message().to_string(),
        pg_sys::panic::CaughtError::RustPanic { ereport, .. } => ereport.message().to_string(),
    }
}

/// The oids of every agentvec index of this database.
fn list_indexes() -> Vec<pg_sys::Oid> {
    Spi::connect(|client| {
        let mut indexes = Vec::new();
        let sql = "SELECT c.oid FROM pg_class c JOIN pg_am a ON a.oid = c.relam \
                   WHERE a.amname = 'agentvec' AND c.relkind = 'i'";
        let table = client.select(sql, None, &[])?;
        for row in table {
            indexes.push(row.get::<pg_sys::Oid>(1)?.expect("oid"));
        }
        Ok::<_, spi::Error>(indexes)
    })
    .unwrap_or_default()
}

/// The minimum `maintenance_interval` across the database's agentvec
/// indexes (reloption default applied): the worker sleeps this long between
/// passes.
fn pass_interval_ms() -> u64 {
    Spi::connect(|client| {
        let sql = "SELECT COALESCE(MIN((SELECT o.option_value::int FROM pg_options_to_table(c.reloptions) o \
                                        WHERE o.option_name = 'maintenance_interval')), 60000)::bigint \
                   FROM pg_class c JOIN pg_am a ON a.oid = c.relam \
                   WHERE a.amname = 'agentvec' AND c.relkind = 'i'";
        client
            .select(sql, None, &[])
            .ok()
            .and_then(|mut t| t.next())
            .and_then(|row| row.get::<i64>(1).ok().flatten())
            .map(|ms| ms.max(50) as u64)
            .unwrap_or(60_000)
    })
}

/// Run one bounded maintenance batch on an index regardless of schedule —
/// the manual counterpart of the worker (tests, ops, Neon fallback).
#[pg_extern]
fn agentvec_run_maintenance(index: PgRelation, budget_rows: default!(i64, "-1")) -> i64 {
    let budget = if budget_rows < 0 {
        super::options::TSVAgentVecOptions::from_relation(&index).get_migration_batch_rows() as i64
    } else {
        budget_rows
    };
    let converted = unsafe { maintain_inner(&index, budget) };
    unsafe {
        super::meta_page::AgentVecMetaPage::update(&index, |meta| meta.mark_maintained());
    }
    converted
}

/// The worker's per-index step: run a bounded batch only when the index's
/// `maintenance_interval` has elapsed since the last pass (fresh indexes are
/// initialized as just-maintained by `ambuild`, so they are not converted
/// until their interval passes).  Returns the number of rows converted.
#[pg_extern]
fn agentvec_maybe_maintain(index: PgRelation) -> i64 {
    let meta = super::meta_page::AgentVecMetaPage::fetch(&index);
    let options = super::options::TSVAgentVecOptions::from_relation(&index);
    let due = {
        let interval_ms = options.get_maintenance_interval() as i64;
        interval_ms <= 0
            || (unsafe { pg_sys::GetCurrentTimestamp() } - meta.get_last_maintenance_at())
                >= interval_ms * 1000
    };
    if !due {
        return 0;
    }
    let budget = options.get_migration_batch_rows() as i64;
    let converted = unsafe { maintain_inner(&index, budget) };
    unsafe {
        super::meta_page::AgentVecMetaPage::update(&index, |meta| meta.mark_maintained());
    }
    converted
}

/// Convert sealed segments until nothing is sealed or the row budget is spent.
unsafe fn maintain_inner(index: &PgRelation, budget: i64) -> i64 {
    let mut converted: i64 = 0;
    loop {
        let n = super::consolidate::consolidate_inner(index);
        if n <= 0 {
            break;
        }
        converted += n;
        if converted >= budget {
            break;
        }
    }
    converted
}
