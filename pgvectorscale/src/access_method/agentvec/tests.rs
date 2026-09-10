//! Tests for the `agentvec` access method.
//!
//! Two kinds of coverage:
//!
//! * **Behavioural** — create/insert/update/delete/vacuum/reindex through SQL,
//!   always with `enable_seqscan = off` so the plan really exercises the index
//!   (results are compared against explicitly known exact distances).
//! * **Structural** — the on-disk model is inspected directly (meta page,
//!   directory, segment headers) so the tests fail if sealing or the directory
//!   stop being what the design says they are.

#[cfg(any(test, feature = "pg_test"))]
#[pgrx::pg_schema]
mod tests {
    use pgrx::*;

    use crate::access_method::agentvec::directory::{
        AgentVecDirectory, AgentVecSegmentHeader, SegmentState,
    };
    use crate::access_method::agentvec::meta_page::AgentVecMetaPage;

    /// Open an index relation by name.
    fn index_relation(name: &str) -> PgRelation {
        let oid = Spi::get_one::<pg_sys::Oid>(&format!("SELECT '{name}'::regclass::oid"))
            .unwrap()
            .expect("index oid was null");
        unsafe { PgRelation::from_pg(pg_sys::RelationIdGetRelation(oid)) }
    }

    /// The ids returned by an approximate-nearest-neighbour query, in order.
    fn search_ids(table: &str, column: &str, op: &str, query: &str, limit: usize) -> Vec<i32> {
        Spi::connect(|client| {
            let sql = format!(
                "SELECT id FROM {table} ORDER BY {column} {op} '{query}' LIMIT {limit}"
            );
            let table = client.select(&sql, None, &[])?;
            let mut ids = Vec::new();
            for row in table {
                ids.push(row.get::<i32>(1)?.expect("id was null"));
            }
            Ok::<Vec<i32>, spi::Error>(ids)
        })
        .unwrap()
    }

    /// Force index usage for the rest of the test transaction.
    fn force_index_scan() {
        Spi::run("SET enable_seqscan = off").unwrap();
    }

    #[pg_test]
    fn test_agentvec_l2_exact_search() -> spi::Result<()> {
        Spi::run(
            "CREATE TABLE t_av_l2(id int, v vector(3));
             INSERT INTO t_av_l2 VALUES
                (1, '[1,0,0]'), (2, '[2,0,0]'), (3, '[3,0,0]'), (4, '[4,0,0]');
             CREATE INDEX idx_av_l2 ON t_av_l2 USING agentvec (v vector_l2_ops);",
        )?;
        force_index_scan();

        assert_eq!(
            search_ids("t_av_l2", "v", "<->", "[0,0,0]", 4),
            vec![1, 2, 3, 4],
            "L2 order must be exact"
        );
        assert_eq!(
            search_ids("t_av_l2", "v", "<->", "[100,0,0]", 2),
            vec![4, 3],
            "closest first"
        );
        Ok(())
    }

    /// A row inserted after the index exists is searchable immediately, with no
    /// background step (design §32.1).
    #[pg_test]
    fn test_agentvec_insert_is_immediately_searchable() -> spi::Result<()> {
        Spi::run(
            "CREATE TABLE t_av_ins(id int, v vector(3));
             CREATE INDEX idx_av_ins ON t_av_ins USING agentvec (v vector_l2_ops);",
        )?;
        force_index_scan();

        // Empty index first: the plan must still work (no rows).
        assert!(search_ids("t_av_ins", "v", "<->", "[0,0,0]", 5).is_empty());

        Spi::run(
            "INSERT INTO t_av_ins VALUES (1, '[5,0,0]');
             INSERT INTO t_av_ins VALUES (2, '[1,0,0]');
             INSERT INTO t_av_ins VALUES (3, '[3,0,0]');",
        )?;
        assert_eq!(
            search_ids("t_av_ins", "v", "<->", "[0,0,0]", 3),
            vec![2, 3, 1],
            "inserted rows must be immediately visible to the index"
        );
        Ok(())
    }

    /// UPDATE inserts the new version and leaves the old one to heap visibility,
    /// so the query sees only the new vector.
    #[pg_test]
    fn test_agentvec_update_reflects_new_vector() -> spi::Result<()> {
        Spi::run(
            "CREATE TABLE t_av_upd(id int, v vector(3));
             INSERT INTO t_av_upd VALUES (1, '[9,0,0]'), (2, '[8,0,0]');
             CREATE INDEX idx_av_upd ON t_av_upd USING agentvec (v vector_l2_ops);",
        )?;
        force_index_scan();
        assert_eq!(search_ids("t_av_upd", "v", "<->", "[0,0,0]", 2), vec![2, 1]);

        Spi::run("UPDATE t_av_upd SET v = '[0.5,0,0]' WHERE id = 1")?;
        force_index_scan();
        assert_eq!(
            search_ids("t_av_upd", "v", "<->", "[0,0,0]", 2),
            vec![1, 2],
            "the updated vector must be the one the index returns"
        );
        Ok(())
    }

    /// A tombstoned entry is skipped by scans and stops consuming candidate
    /// slots.
    ///
    /// The tombstone is applied directly (`flat::mark_dead`, the same call
    /// `ambulkdelete` makes) rather than through VACUUM: whether a VACUUM
    /// removes a deleted entry depends on PostgreSQL's vacuum cutoff, and in a
    /// test suite where other backends hold snapshots the tuple stays
    /// "recently dead", which would make the assertion depend on scheduling
    /// rather than on this AM's behaviour.
    #[pg_test]
    fn test_agentvec_tombstoned_entries_are_skipped() -> spi::Result<()> {
        Spi::run(
            "CREATE TABLE t_av_tomb(id int, v vector(2));
             INSERT INTO t_av_tomb VALUES (1, '[1,0]'), (2, '[2,0]'), (3, '[3,0]');
             CREATE INDEX idx_av_tomb ON t_av_tomb USING agentvec (v vector_l2_ops)
                WITH (search_candidates = 2);",
        )?;
        force_index_scan();
        assert_eq!(
            search_ids("t_av_tomb", "v", "<->", "[0,0]", 2),
            vec![1, 2],
            "the bound keeps the two nearest entries"
        );

        // Tombstone the nearest entry of the HOT segment's chain.
        let index = index_relation("idx_av_tomb");
        let meta = AgentVecMetaPage::fetch(&index);
        let directory = meta.load_directory(&index);
        let segment = directory
            .get(meta.get_hot_segment_id())
            .expect("the index has a HOT segment");
        let header = AgentVecSegmentHeader::load(&index, segment.header);
        let chain = header.chain_starts()[0];
        let mut victim: Option<(pg_sys::BlockNumber, pg_sys::OffsetNumber)> = None;
        unsafe {
            crate::access_method::agentvec::flat::for_each_entry(
                &index,
                chain,
                |block, offset, bytes| {
                    if victim.is_none()
                        && crate::access_method::agentvec::flat::decode_state(bytes)
                            == crate::access_method::agentvec::flat::STATE_LIVE
                    {
                        victim = Some((block, offset));
                    }
                },
            );
            let (block, offset) = victim.expect("a live entry to tombstone");
            crate::access_method::agentvec::flat::mark_dead(&index, block, offset);
        }

        assert_eq!(
            search_ids("t_av_tomb", "v", "<->", "[0,0]", 2),
            vec![2, 3],
            "a tombstoned entry must not consume a candidate slot"
        );
        // The segment's dead counter is maintained by `ambulkdelete`, not by
        // the byte-level tombstone itself; that bookkeeping is asserted by
        // `test_agentvec_bulkdelete_tombstones_and_counts`.
        Ok(())
    }

    /// `ambulkdelete` tombstones exactly the entries the callback reports dead
    /// and keeps the segment's dead count in step with them.
    ///
    /// The AM callback is invoked directly with a synthetic callback (VACUUM
    /// is what normally drives it) so the assertion does not depend on
    /// PostgreSQL's vacuum cutoff, which in a suite with concurrent backends
    /// can leave a deleted tuple "recently dead" and therefore untouchable.
    #[pg_test]
    fn test_agentvec_bulkdelete_tombstones_and_counts() -> spi::Result<()> {
        Spi::run(
            "CREATE TABLE t_av_bd(id int, v vector(2));
             INSERT INTO t_av_bd VALUES (1, '[1,0]'), (2, '[2,0]'), (3, '[3,0]');
             CREATE INDEX idx_av_bd ON t_av_bd USING agentvec (v vector_l2_ops);",
        )?;
        force_index_scan();
        assert_eq!(search_ids("t_av_bd", "v", "<->", "[0,0]", 3), vec![1, 2, 3]);

        // The heap TID of the row that we will tell the AM is dead.
        let ctid = Spi::get_one::<String>("SELECT ctid::text FROM t_av_bd WHERE id = 1")?
            .expect("ctid");
        let (block, offset) = parse_ctid(&ctid);

        struct KillOne {
            block: u32,
            offset: u16,
        }
        unsafe extern "C-unwind" fn kill_one(
            tid: *mut pg_sys::ItemPointerData,
            state: *mut std::os::raw::c_void,
        ) -> bool {
            let target = &*(state as *const KillOne);
            pgrx::itemptr::item_pointer_get_block_number(tid) == target.block
                && pgrx::itemptr::item_pointer_get_offset_number(tid) == target.offset
        }

        let index = index_relation("idx_av_bd");
        let mut kill = KillOne { block, offset };
        let mut info = pg_sys::IndexVacuumInfo::default();
        info.index = index.as_ptr();
        let results = unsafe {
            crate::access_method::agentvec::vacuum::ambulkdelete(
                &mut info,
                std::ptr::null_mut(),
                Some(kill_one),
                &mut kill as *mut KillOne as *mut std::os::raw::c_void,
            )
        };

        unsafe {
            assert_eq!((*results).tuples_removed, 1.0, "one dead entry reported");
            assert_eq!((*results).num_index_tuples, 2.0, "two live entries remain");
        }

        // The tombstone is visible in the scan and in the segment's counters.
        force_index_scan();
        assert_eq!(
            search_ids("t_av_bd", "v", "<->", "[0,0]", 3),
            vec![2, 3],
            "the tombstoned entry must not be returned"
        );
        let (dead, live, total) = Spi::get_one::<String>(
            "SELECT sum(dead_entries) || '/' || sum(live_entries) || '/' || sum(num_entries)
               FROM agentvec_index_info('idx_av_bd')",
        )?
        .map(|row| {
            let mut parts = row.split('/');
            (
                parts.next().unwrap().parse::<i64>().unwrap(),
                parts.next().unwrap().parse::<i64>().unwrap(),
                parts.next().unwrap().parse::<i64>().unwrap(),
            )
        })
        .expect("counters");
        assert_eq!((dead, live, total), (1, 2, 3));
        Ok(())
    }

    /// Parse PostgreSQL's `(block,offset)` ctid text.
    fn parse_ctid(ctid: &str) -> (u32, u16) {
        let inner = ctid.trim_start_matches('(').trim_end_matches(')');
        let mut parts = inner.split(',');
        let block = parts.next().expect("ctid block").parse().expect("block");
        let offset = parts.next().expect("ctid offset").parse().expect("offset");
        (block, offset)
    }

    /// VACUUM runs against the index and leaves its answers correct.
    ///
    /// VACUUM cannot run inside the SPI test transaction, so this test brings
    /// up the test instance and drives everything through a second connection
    /// (the pattern `access_method::vacuum` uses for the diskann AM).  The
    /// assertions are deliberately independent of PostgreSQL's vacuum cutoff;
    /// the tombstone semantics themselves are covered by
    /// `test_agentvec_tombstoned_entries_are_skipped`.
    #[cfg(test)]
    #[test]
    fn test_agentvec_vacuum_keeps_results_correct() {
        // This test owns its own connection and DDL, so it must not overlap
        // with itself (pg_test's rollback protection does not apply to it).
        static VAC_MUTEX: once_cell::sync::Lazy<std::sync::Mutex<()>> =
            once_cell::sync::Lazy::new(std::sync::Mutex::default);
        let _lock = VAC_MUTEX
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        pgrx_tests::run_test(
            "agentvec_vacuum_mock_fn",
            None,
            crate::pg_test::postgresql_conf_options(),
        )
        .unwrap();

        let (mut client, _) = pgrx_tests::client().unwrap();
        client
            .batch_execute(
                "CREATE TABLE t_av_vac(id int, v vector(2));
                 INSERT INTO t_av_vac VALUES (1, '[1,0]'), (2, '[2,0]'), (3, '[3,0]');
                 CREATE INDEX idx_av_vac ON t_av_vac USING agentvec (v vector_l2_ops);",
            )
            .unwrap();
        client.execute("SET enable_seqscan = off", &[]).unwrap();

        let query = "SELECT id FROM t_av_vac ORDER BY v <-> '[0,0]' LIMIT 3";
        let ids: Vec<i32> = client
            .query(query, &[])
            .unwrap()
            .iter()
            .map(|row| row.get::<_, i32>(0))
            .collect();
        assert_eq!(ids, vec![1, 2, 3]);

        client
            .execute("DELETE FROM t_av_vac WHERE id = 1", &[])
            .unwrap();
        let ids: Vec<i32> = client
            .query(query, &[])
            .unwrap()
            .iter()
            .map(|row| row.get::<_, i32>(0))
            .collect();
        assert_eq!(ids, vec![2, 3], "a deleted row must not be returned");

        // VACUUM cannot run inside a transaction block, so it needs its own
        // statement (batch_execute would wrap it with the DELETE).
        client.execute("VACUUM t_av_vac", &[]).unwrap();

        let ids: Vec<i32> = client
            .query(query, &[])
            .unwrap()
            .iter()
            .map(|row| row.get::<_, i32>(0))
            .collect();
        assert_eq!(ids, vec![2, 3], "results are unchanged by vacuum");

        client.batch_execute("DROP TABLE t_av_vac").unwrap();
    }

    /// Only a mock: it brings up the test instance that the connection-based
    /// test above drives.
    #[pg_test]
    fn agentvec_vacuum_mock_fn() -> spi::Result<()> {
        Ok(())
    }

    /// Reaching the HOT threshold seals the segment and opens a new one; the
    /// sealed segment stays searchable, so results span all segments in order.
    #[pg_test]
    fn test_agentvec_hot_seal_and_multi_segment_search() -> spi::Result<()> {
        Spi::run(
            "CREATE TABLE t_av_seal(id int, v vector(2));
             CREATE INDEX idx_av_seal ON t_av_seal USING agentvec (v vector_l2_ops)
                WITH (hot_segment_max_rows = 2);
             INSERT INTO t_av_seal
                SELECT g, ('[' || g || ',0]')::vector FROM generate_series(1, 7) AS g;",
        )?;
        force_index_scan();

        // 7 rows at 2 rows per HOT segment: the last insert observes the
        // threshold, so segments are created at rows 3, 5 and 7.
        let index = index_relation("idx_av_seal");
        let meta = AgentVecMetaPage::fetch(&index);
        let directory: AgentVecDirectory = meta.load_directory(&index);
        assert_eq!(
            directory.segments.len(),
            4,
            "expected four HOT segments, got {:?}",
            directory
                .segments
                .iter()
                .map(|s| (s.segment_id, s.state, s.vector_count))
                .collect::<Vec<_>>()
        );
        for (i, segment) in directory.segments.iter().enumerate() {
            let header = AgentVecSegmentHeader::load(&index, segment.header);
            let expected_state = if segment.segment_id == meta.get_hot_segment_id() {
                SegmentState::Published
            } else {
                SegmentState::QueuedForMigration
            };
            assert_eq!(segment.state(), expected_state, "segment {i} state");
            if segment.segment_id != meta.get_hot_segment_id() {
                assert_eq!(
                    header.sealed.len(),
                    1,
                    "a sealed segment must have exactly one frozen run"
                );
            }
        }

        // The same structure is visible from SQL, which is how a user (or a
        // later phase's maintenance worker) inspects the directory.
        let hot_segments = Spi::get_one::<i64>(
            "SELECT count(*) FROM agentvec_index_info('idx_av_seal') WHERE level = 'hot'",
        )?
        .expect("count");
        assert_eq!(hot_segments, 4);
        let queued: i64 = Spi::get_one::<i64>(
            "SELECT count(*) FROM agentvec_index_info('idx_av_seal') WHERE state = 'queued_for_migration'",
        )?
        .expect("count");
        assert_eq!(queued, 3, "every sealed segment awaits maintenance");
        let indexed: i64 = Spi::get_one::<i64>(
            "SELECT sum(num_entries)::bigint FROM agentvec_index_info('idx_av_seal')",
        )?
        .expect("sum");
        assert_eq!(indexed, 7, "every inserted row is accounted for");

        // Distance from [0,0] is |g|, so exact order is 1..7 across segments.
        assert_eq!(
            search_ids("t_av_seal", "v", "<->", "[0,0]", 7),
            vec![1, 2, 3, 4, 5, 6, 7],
            "search must merge every segment in distance order"
        );
        Ok(())
    }

    /// Cosine and inner-product operator classes, including the cosine
    /// normalization the stored representation relies on.
    #[pg_test]
    fn test_agentvec_cosine_and_inner_product_ops() -> spi::Result<()> {
        Spi::run(
            "CREATE TABLE t_av_cos(id int, v vector(2));
             INSERT INTO t_av_cos VALUES
                (1, '[1,0]'), (2, '[1,1]'), (3, '[0,1]'), (4, '[-1,0]');
             CREATE INDEX idx_av_cos ON t_av_cos USING agentvec (v vector_cosine_ops);",
        )?;
        force_index_scan();
        assert_eq!(
            search_ids("t_av_cos", "v", "<=>", "[1,0]", 4),
            vec![1, 2, 3, 4],
            "cosine order must ignore magnitude"
        );

        Spi::run(
            "CREATE TABLE t_av_ip(id int, v vector(2));
             INSERT INTO t_av_ip VALUES (1, '[1,0]'), (2, '[2,0]'), (3, '[3,0]');
             CREATE INDEX idx_av_ip ON t_av_ip USING agentvec (v vector_ip_ops);",
        )?;
        force_index_scan();
        assert_eq!(
            search_ids("t_av_ip", "v", "<#>", "[1,0]", 3),
            vec![3, 2, 1],
            "inner product orders by descending dot product"
        );
        Ok(())
    }

    /// `search_candidates` bounds the candidate stream (0 = exhaustive).
    #[pg_test]
    fn test_agentvec_search_candidates_bound() -> spi::Result<()> {
        Spi::run(
            "CREATE TABLE t_av_bound(id int, v vector(2));
             INSERT INTO t_av_bound
                SELECT g, ('[' || g || ',0]')::vector FROM generate_series(1, 10) AS g;
             CREATE INDEX idx_av_bound ON t_av_bound USING agentvec (v vector_l2_ops)
                WITH (search_candidates = 3);",
        )?;
        force_index_scan();

        let ids = search_ids("t_av_bound", "v", "<->", "[0,0]", 10);
        assert_eq!(
            ids,
            vec![1, 2, 3],
            "a bounded scan returns at most `search_candidates` rows"
        );
        Ok(())
    }

    /// REINDEX rebuilds from the heap and produces the same answers.
    #[pg_test]
    fn test_agentvec_reindex() -> spi::Result<()> {
        Spi::run(
            "CREATE TABLE t_av_reindex(id int, v vector(2));
             INSERT INTO t_av_reindex
                SELECT g, ('[' || g || ',0]')::vector FROM generate_series(1, 5) AS g;
             CREATE INDEX idx_av_reindex ON t_av_reindex USING agentvec (v vector_l2_ops)
                WITH (hot_segment_max_rows = 2);",
        )?;
        force_index_scan();
        let before = search_ids("t_av_reindex", "v", "<->", "[0,0]", 5);
        assert_eq!(before, vec![1, 2, 3, 4, 5]);

        Spi::run("REINDEX INDEX idx_av_reindex")?;
        force_index_scan();
        assert_eq!(
            search_ids("t_av_reindex", "v", "<->", "[0,0]", 5),
            before,
            "REINDEX must reproduce the same index contents"
        );
        Ok(())
    }

    /// The index answers ORDER BY queries only: a plan that would need an
    /// unordered answer must not pick it.
    #[pg_test]
    fn test_agentvec_planner_uses_index_only_for_order_by() -> spi::Result<()> {
        Spi::run(
            "CREATE TABLE t_av_plan(id int, v vector(2));
             INSERT INTO t_av_plan VALUES (1, '[1,0]'), (2, '[2,0]');
             CREATE INDEX idx_av_plan ON t_av_plan USING agentvec (v vector_l2_ops);",
        )?;
        force_index_scan();

        let ordered = plan_for("SELECT id FROM t_av_plan ORDER BY v <-> '[0,0]' LIMIT 1")?;
        assert!(
            ordered.contains("Index Scan") && ordered.contains("idx_av_plan"),
            "ORDER BY query must use the agentvec index, got:\n{ordered}"
        );

        // A count(*) cannot be answered by this AM at all (it has no ordered
        // candidate stream), so it must be planned without the index and must
        // return the true count -- including under `enable_seqscan = off`,
        // where the planner has to fall back to a disabled seq scan rather
        // than to an infinite-cost index path.
        let unordered = plan_for("SELECT count(*) FROM t_av_plan")?;
        assert!(
            !unordered.contains("idx_av_plan"),
            "a count(*) plan must not use the agentvec index, got:\n{unordered}"
        );
        let count = Spi::get_one::<i64>("SELECT count(*) FROM t_av_plan")?.expect("count");
        assert_eq!(count, 2, "count(*) must return the true row count");
        Ok(())
    }

    /// `EXPLAIN (COSTS OFF)` as a single string.
    fn plan_for(query: &str) -> spi::Result<String> {
        Spi::connect(|client| {
            let table = client.select(&format!("EXPLAIN (COSTS OFF) {query}"), None, &[])?;
            let mut plan = String::new();
            for row in table {
                if let Some(line) = row.get::<String>(1)? {
                    plan.push_str(&line);
                    plan.push('\n');
                }
            }
            Ok::<String, spi::Error>(plan)
        })
    }

    /// The on-disk state a fresh index starts from: meta at block 0 with a
    /// directory holding exactly one published, empty HOT segment.
    #[pg_test]
    fn test_agentvec_empty_index_layout() -> spi::Result<()> {
        Spi::run(
            "CREATE TABLE t_av_empty(id int, v vector(4));
             CREATE INDEX idx_av_empty ON t_av_empty USING agentvec (v vector_l2_ops);",
        )?;

        let index = index_relation("idx_av_empty");
        let meta = AgentVecMetaPage::fetch(&index);
        assert_eq!(meta.get_num_dimensions(), 4);
        assert_eq!(meta.get_num_tuples(), 0);

        let directory = meta.load_directory(&index);
        assert_eq!(directory.segments.len(), 1);
        let segment = &directory.segments[0];
        assert_eq!(segment.segment_id, meta.get_hot_segment_id());
        assert_eq!(segment.state(), SegmentState::Published);
        assert_eq!(segment.algorithm().as_str(), "flat");
        assert_eq!(segment.ownership().as_str(), "owned");

        let header = AgentVecSegmentHeader::load(&index, segment.header);
        assert_eq!(header.num_entries, 0);
        assert!(header.sealed.is_empty());
        assert!(header.active.is_none());
        Ok(())
    }
}
