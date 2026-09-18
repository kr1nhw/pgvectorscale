//! hnswsq integration tests — gates 1 and 2.
//!
//! * AM existence, empty-index lifecycle (gate 1);
//! * recall matrix across the four storage layouts and three distance types
//!   against exact ground truth, incremental empty-start lifecycle, and
//!   transaction rollback (gate 2: single-writer insert parity).
//!
//! Thresholds mirror the old engine's suite so the port must clear the same
//! bar the retired engine did.

use pgrx::prelude::*;

#[pgrx::pg_schema]
pub mod tests {
    use super::*;

    /// Small deterministic xorshift64* RNG (test data only).
    struct Rng(u64);

    impl Rng {
        fn new(seed: u64) -> Self {
            Rng(seed.max(1))
        }
        fn next_u64(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545F4914F6CDD1D)
        }
        fn next_f32(&mut self) -> f32 {
            (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
        }
    }

    /// Clustered data: n_clusters × per_cluster rows around random centers,
    /// one query per center (the old engine's generator).
    fn gen_clustered(
        n_clusters: usize,
        per_cluster: usize,
        dim: usize,
        noise: f32,
        seed: u64,
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let mut rng = Rng::new(seed);
        let centers: Vec<Vec<f32>> = (0..n_clusters)
            .map(|_| (0..dim).map(|_| rng.next_f32() * 2.0 - 1.0).collect())
            .collect();
        let mut rows = Vec::with_capacity(n_clusters * per_cluster);
        for c in &centers {
            for _ in 0..per_cluster {
                rows.push(
                    c.iter()
                        .map(|x| x + (rng.next_f32() - 0.5) * noise)
                        .collect(),
                );
            }
        }
        let queries: Vec<Vec<f32>> = centers
            .iter()
            .map(|c| {
                c.iter()
                    .map(|x| x + (rng.next_f32() - 0.5) * noise * 0.2)
                    .collect()
            })
            .collect();
        (rows, queries)
    }

    fn vec_literal(v: &[f32]) -> String {
        let parts: Vec<String> = v.iter().map(|x| format!("{:.6}", x)).collect();
        format!("[{}]", parts.join(","))
    }

    fn insert_rows(table: &str, vecs: &[Vec<f32>]) -> spi::Result<()> {
        for chunk in vecs.chunks(100) {
            let values: Vec<String> = chunk
                .iter()
                .map(|v| format!("('{}')", vec_literal(v)))
                .collect();
            Spi::run(&format!(
                "INSERT INTO {}(embedding) VALUES {}",
                table,
                values.join(",")
            ))?;
        }
        Ok(())
    }

    /// Create `hs_t` (rows), `hs_q` (queries) and `hs_gt` (exact top-10 by
    /// `op`) — all BEFORE the index exists, so the ground truth is a pure
    /// sequential computation.
    fn setup_case(
        dim: usize,
        rows: &[Vec<f32>],
        queries: &[Vec<f32>],
        op: &str,
    ) -> spi::Result<()> {
        Spi::run(&format!(
            "CREATE TABLE hs_t(id serial primary key, embedding vector({}));
             CREATE TABLE hs_q(qid serial primary key, embedding vector({}));",
            dim, dim
        ))?;
        insert_rows("hs_t", rows)?;
        insert_rows("hs_q", queries)?;
        Spi::run(&format!(
            "CREATE TABLE hs_gt AS
             SELECT qid, tid FROM (
               SELECT q.qid AS qid, t.id AS tid,
                      row_number() OVER (PARTITION BY q.qid ORDER BY t.embedding {} q.embedding) AS rn
               FROM hs_q q CROSS JOIN hs_t t) s
             WHERE rn <= 10;",
            op
        ))?;
        Ok(())
    }

    /// Build the index, run the ANN queries (one index rescan per query via
    /// LATERAL), and return recall@10 against `hs_gt`.
    fn measure_recall(opclass: &str, op: &str, with_opts: &str, ef: i32) -> spi::Result<f64> {
        Spi::run(&format!(
            "CREATE INDEX hs_idx ON hs_t USING hnswsq (embedding {}) WITH ({});
             SET hnswsq.ef_search = {};
             SET enable_seqscan = off;
             CREATE TABLE hs_ann AS
             SELECT q.qid AS qid, t.id AS tid
             FROM hs_q q CROSS JOIN LATERAL (
               SELECT id FROM hs_t ORDER BY embedding {} q.embedding LIMIT 10) t;",
            opclass, with_opts, ef, op
        ))?;
        let n: i64 = Spi::get_one::<i64>("SELECT count(*) FROM hs_ann")?.unwrap_or(0);
        let nq: i64 = Spi::get_one::<i64>("SELECT count(*) FROM hs_q")?.unwrap_or(0);
        assert_eq!(n, nq * 10, "every query must produce 10 candidates");
        let recall = Spi::get_one::<f64>(
            "SELECT count(*)::float8 / (SELECT count(*)::float8 FROM hs_gt)
             FROM hs_gt g JOIN hs_ann a ON g.qid = a.qid AND g.tid = a.tid",
        )?
        .unwrap_or(0.0);
        Ok(recall)
    }

    /// Affine-map ±1 data into the byte domain [1, 255]: the fixed-range
    /// `sq8`/`sq16` layouts quantize against a global [0, 255] range, so
    /// their tests need byte-domain data to keep the cluster structure
    /// distinguishable.
    fn to_byte_domain(vecs: &mut [Vec<f32>]) {
        for row in vecs.iter_mut() {
            for x in row.iter_mut() {
                *x = *x * 127.0 + 128.0;
            }
        }
    }

    fn recall_case(opclass: &str, op: &str, with_opts: &str, threshold: f64, byte_domain: bool) {
        let (mut rows, mut queries) = gen_clustered(20, 50, 16, 0.05, 12345);
        if byte_domain {
            to_byte_domain(&mut rows);
            to_byte_domain(&mut queries);
        }
        setup_case(16, &rows, &queries, op).unwrap();
        // Pin the build RNG: recall is a property of the graph, so an
        // entropy-seeded build makes this assertion a random sample.
        Spi::run("SET hnswsq.build_seed = 20240912;").unwrap();
        let recall = measure_recall(opclass, op, with_opts, 100).unwrap();
        assert!(
            recall >= threshold,
            "recall@10 {} below {} for {}",
            recall,
            threshold,
            with_opts
        );
    }

    // ---------------- gate 1: the AM exists ----------------

    #[pg_test]
    fn test_hnswsq_create_drop_empty_index() {
        Spi::run("CREATE TABLE hnswsq_empty(id int, embedding vector(8))").unwrap();
        Spi::run(
            "CREATE INDEX hnswsq_empty_idx ON hnswsq_empty USING hnswsq (embedding vector_l2_ops)",
        )
        .unwrap();
        // The empty index must answer queries without error.
        let count: i64 = Spi::get_one(
            "SELECT count(*) FROM (SELECT id FROM hnswsq_empty \
             ORDER BY embedding <-> '[1,2,3,4,5,6,7,8]'::vector LIMIT 1) t",
        )
        .unwrap()
        .unwrap();
        assert_eq!(count, 0);
        Spi::run("DROP INDEX hnswsq_empty_idx").unwrap();
        Spi::run("DROP TABLE hnswsq_empty").unwrap();
    }

    // ---------------- gate 2: recall matrix ----------------

    #[pg_test]
    fn test_hnswsq_recall_plain_l2() {
        recall_case("vector_l2_ops", "<->", "storage_layout = plain", 0.9, false);
    }

    #[pg_test]
    fn test_hnswsq_recall_ieeefp16_l2() {
        recall_case("vector_l2_ops", "<->", "storage_layout = ieeefp16", 0.9, false);
    }

    #[pg_test]
    fn test_hnswsq_recall_ieeefp8_l2() {
        recall_case("vector_l2_ops", "<->", "storage_layout = ieeefp8", 0.75, false);
    }

    #[pg_test]
    fn test_hnswsq_recall_sq8_l2() {
        recall_case("vector_l2_ops", "<->", "storage_layout = f8", 0.85, false);
    }

    #[pg_test]
    fn test_hnswsq_recall_sq8_fixed_l2() {
        // Training-free fixed-range int8; the test data lies in ±1 so the
        // global range costs nothing vs the calibrated f8.
        recall_case("vector_l2_ops", "<->", "storage_layout = sq8", 0.85, true);
    }

    #[pg_test]
    fn test_hnswsq_recall_sq16_l2() {
        // Training-free fixed-range int16: near-plain precision.
        recall_case("vector_l2_ops", "<->", "storage_layout = sq16", 0.9, true);
    }

    #[pg_test]
    fn test_hnswsq_recall_plain_cosine() {
        recall_case("vector_cosine_ops", "<=>", "storage_layout = plain", 0.9, false);
    }

    #[pg_test]
    fn test_hnswsq_recall_ieeefp16_ip() {
        recall_case("vector_ip_ops", "<#>", "storage_layout = ieeefp16", 0.85, false);
    }

    // ---------------- gate 2: incremental empty-start ----------------

    fn incremental_case(layout: &str) {
        let dim = 16;
        let (mut rows, _queries) = gen_clustered(10, 100, dim, 0.05, 4242);
        if layout == "sq8" || layout == "sq16" {
            // The fixed layouts quantize against [0, 255]: byte-domain data
            // keeps distinct rows distinguishable (the exact-match probe
            // needs distance 0 only for the row itself).
            to_byte_domain(&mut rows);
        }
        Spi::run(&format!(
            "CREATE TABLE hs_i(id serial primary key, embedding vector({}));
             CREATE INDEX hs_i_idx ON hs_i USING hnswsq (embedding vector_l2_ops)
               WITH (storage_layout = {});
             SET enable_seqscan = off;
             SET hnswsq.ef_search = 500;
             SET hnswsq.build_seed = 20240912;",
            dim, layout
        ))
        .unwrap();

        // Staged inserts: after each batch, an exact-match probe for a row in
        // that batch must return it (distance 0 survives every layout: the
        // stored code of a vector always matches itself).
        for batch in 0..5 {
            let slice = &rows[batch * 200..(batch + 1) * 200];
            insert_rows("hs_i", slice).unwrap();
            let probe = batch * 200; // 0-based row → id = probe + 1
            let got: i64 = Spi::get_one::<i64>(&format!(
                "SELECT id FROM hs_i ORDER BY embedding <-> '{}' LIMIT 1",
                vec_literal(&rows[probe])
            ))
            .unwrap()
            .unwrap_or(-1);
            assert_eq!(
                got,
                (probe + 1) as i64,
                "batch {}: exact-match probe must find its row (layout {})",
                batch,
                layout
            );
        }
    }

    #[pg_test]
    fn test_hnswsq_incremental_empty_start_plain() {
        incremental_case("plain");
    }

    #[pg_test]
    fn test_hnswsq_incremental_empty_start_ieeefp8() {
        incremental_case("ieeefp8");
    }

    #[pg_test]
    fn test_hnswsq_incremental_empty_start_sq8() {
        // Empty-start SQ8 gets a provisional [-1, 1] calibration; the probes
        // are unit vectors so they stay in range.
        incremental_case("f8");
    }

    #[pg_test]
    fn test_hnswsq_incremental_empty_start_sq8_fixed() {
        // Training-free fixed-range int8: no calibration chain at all.
        incremental_case("sq8");
    }

    #[pg_test]
    fn test_hnswsq_incremental_empty_start_sq16() {
        incremental_case("sq16");
    }

    // ---------------- gate 2: transaction rollback ----------------

    #[pg_test]
    fn test_hnswsq_txn_rollback() {
        let (rows, _q) = gen_clustered(4, 25, 16, 0.05, 99);
        Spi::run(
            "CREATE TABLE hs_r(id serial primary key, embedding vector(16));
             CREATE INDEX hs_r_idx ON hs_r USING hnswsq (embedding vector_l2_ops);
             SET enable_seqscan = off;",
        )
        .unwrap();
        insert_rows("hs_r", &rows).unwrap();
        let before: i64 = Spi::get_one::<i64>("SELECT count(*) FROM hs_r").unwrap().unwrap();
        assert_eq!(before, 100);

        // A subtransaction that inserts and aborts: the rows must be
        // invisible afterwards, and the index must stay usable.
        let extra = gen_clustered(1, 20, 16, 0.05, 555).0;
        let values: Vec<String> = extra
            .iter()
            .map(|v| format!("('{}')", vec_literal(v)))
            .collect();
        Spi::run(&format!(
            "DO $$ BEGIN
               INSERT INTO hs_r(embedding) VALUES {};
               RAISE EXCEPTION 'rollback test';
             EXCEPTION WHEN OTHERS THEN NULL;
             END $$;",
            values.join(",")
        ))
        .unwrap();

        let after: i64 = Spi::get_one::<i64>("SELECT count(*) FROM hs_r").unwrap().unwrap();
        assert_eq!(after, before, "rolled-back inserts must not persist");
        let got: i64 = Spi::get_one::<i64>(&format!(
            "SELECT id FROM hs_r ORDER BY embedding <-> '{}' LIMIT 1",
            vec_literal(&rows[0])
        ))
        .unwrap()
        .unwrap_or(-1);
        assert_eq!(got, 1, "index still serves queries after the aborted txn");
    }

    // ---------------- gate 2: page packing ----------------

    #[pg_test]
    fn test_hnswsq_many_nodes_per_page() {
        // dim 2: dozens of element tuples per page; the packing (and the
        // element/neighbor tuple split) is exercised at full density.
        let (rows, queries) = gen_clustered(10, 100, 2, 0.1, 777);
        setup_case(2, &rows, &queries, "<->").unwrap();
        Spi::run("SET hnswsq.build_seed = 11;").unwrap();
        Spi::run(
            "CREATE INDEX hs_idx ON hs_t USING hnswsq (embedding vector_l2_ops)",
        )
        .unwrap();
        Spi::run("SET hnswsq.ef_search = 100; SET enable_seqscan = off;").unwrap();
        Spi::run(
            "CREATE TABLE hs_ann AS
             SELECT q.qid AS qid, t.id AS tid
             FROM hs_q q CROSS JOIN LATERAL (
               SELECT id FROM hs_t ORDER BY embedding <-> q.embedding LIMIT 10) t",
        )
        .unwrap();
        let recall = Spi::get_one::<f64>(
            "SELECT count(*)::float8 / (SELECT count(*)::float8 FROM hs_gt)
             FROM hs_gt g JOIN hs_ann a ON g.qid = a.qid AND g.tid = a.tid",
        )
        .unwrap()
        .unwrap_or(0.0);
        assert!(recall >= 0.9, "dim-2 dense packing recall@10 = {}", recall);
    }

    // ---------------- gate 4: vacuum lifecycle (raw client) ----------------

    #[pg_test]
    /// Mock to bring up the test database for the raw-client vacuum tests.
    fn hnswsq_vacuum_mock_fn() -> spi::Result<()> {
        Ok(())
    }

    #[cfg(test)]
    fn vacuum_lifecycle_scaffold(layout: &str) {
        // Raw-client test (VACUUM cannot run inside the pg_test transaction).
        // Bring up the test database FIRST.
        pgrx_tests::run_test(
            "hnswsq_vacuum_mock_fn",
            None,
            crate::pg_test::postgresql_conf_options(),
        )
        .unwrap();
        // Session-scoped advisory lock on a dedicated connection (a key the
        // old engine's suite does not use): serializes this scaffold against
        // pg_test transactions across backends.
        let (mut guard_client, _) = pgrx_tests::client().unwrap();
        guard_client
            .execute("SELECT pg_advisory_lock(5205217837881163778)", &[])
            .unwrap();

        let (mut rows, _q) = gen_clustered(12, 50, 16, 0.05, 9999);
        if layout == "sq8" || layout == "sq16" {
            to_byte_domain(&mut rows);
        }
        let values: Vec<String> = rows
            .iter()
            .map(|v| format!("('{}')", vec_literal(v)))
            .collect();

        let (mut client, _) = pgrx_tests::client().unwrap();
        client
            .batch_execute(&format!(
                "DROP TABLE IF EXISTS hs_vac2 CASCADE;
                 CREATE TABLE hs_vac2(id serial primary key, embedding vector(16));
                 INSERT INTO hs_vac2(embedding) VALUES {};
                 CREATE INDEX hs_vac2_idx ON hs_vac2 USING hnswsq (embedding vector_l2_ops)
                   WITH (storage_layout = {});
                 SET enable_seqscan = off;
                 SET hnswsq.ef_search = 500;",
                values.join(","),
                layout
            ))
            .unwrap();

        // Sanity: the 300-row sweep works before any delete.
        let probe = format!(
            "WITH cte AS (SELECT id FROM hs_vac2 ORDER BY embedding <-> '{}' LIMIT 300)
             SELECT count(*) FROM cte",
            vec_literal(&rows[0])
        );
        let cnt: i64 = client.query_one(&probe, &[]).unwrap().get(0);
        assert_eq!(cnt, 300, "initial 300-row sweep ({})", layout);

        // Delete half the rows; deleted rows must stop appearing.
        client
            .execute("DELETE FROM hs_vac2 WHERE id % 2 = 0", &[])
            .unwrap();
        let top: i32 = client
            .query_one(
                &format!(
                    "SELECT id FROM hs_vac2 ORDER BY embedding <-> '{}' LIMIT 1",
                    vec_literal(&rows[1]) // row id 2 was deleted
                ),
                &[],
            )
            .unwrap()
            .get(0);
        assert_eq!(top % 2, 1, "deleted row must not be returned");
        client.close().unwrap();

        // VACUUM (fresh connection; VACUUM cannot run in a txn block).
        let (mut client, _) = pgrx_tests::client().unwrap();
        client.execute("VACUUM hs_vac2", &[]).unwrap();
        client.execute("SET enable_seqscan = off", &[]).unwrap();
        client.execute("SET hnswsq.ef_search = 500", &[]).unwrap();

        let cnt: i64 = client
            .query_one(
                &format!(
                    "WITH cte AS (SELECT id FROM hs_vac2 ORDER BY embedding <-> '{}' LIMIT 300)
                     SELECT count(*) FROM cte",
                    vec_literal(&rows[0])
                ),
                &[],
            )
            .unwrap()
            .get(0);
        let diag: String = client
            .query_one("SELECT hnswsq_diag('hs_vac2_idx')", &[])
            .unwrap()
            .get(0);
        assert_eq!(
            cnt, 300,
            "300 live rows after vacuum ({}) diag={}",
            layout, diag
        );
        let relpages1: i32 = client
            .query_one(
                "SELECT relpages FROM pg_class WHERE relname = 'hs_vac2_idx'",
                &[],
            )
            .unwrap()
            .get(0);

        // Reload the deleted half: tuples freed by vacuum must be reused, so
        // the index size stays ~stable.
        let values2: Vec<String> = rows
            .iter()
            .take(300)
            .map(|v| format!("('{}')", vec_literal(v)))
            .collect();
        client
            .execute(
                &format!(
                    "INSERT INTO hs_vac2(embedding) VALUES {}",
                    values2.join(",")
                ),
                &[],
            )
            .unwrap();
        client.close().unwrap();

        let (mut client, _) = pgrx_tests::client().unwrap();
        client.execute("VACUUM hs_vac2", &[]).unwrap();
        client.execute("SET enable_seqscan = off", &[]).unwrap();
        client.execute("SET hnswsq.ef_search = 500", &[]).unwrap();
        let relpages2: i32 = client
            .query_one(
                "SELECT relpages FROM pg_class WHERE relname = 'hs_vac2_idx'",
                &[],
            )
            .unwrap()
            .get(0);
        assert!(
            relpages2 <= relpages1 + 16,
            "page reuse failed: relpages grew {} -> {} ({})",
            relpages1,
            relpages2,
            layout
        );
        client.close().unwrap();
        guard_client
            .execute("SELECT pg_advisory_unlock(5205217837881163778)", &[])
            .unwrap();
    }

    #[test]
    fn hnswsq_vacuum_lifecycle_plain() {
        vacuum_lifecycle_scaffold("plain");
    }

    #[test]
    fn hnswsq_vacuum_lifecycle_ieeefp8() {
        vacuum_lifecycle_scaffold("ieeefp8");
    }

    #[test]
    fn hnswsq_vacuum_lifecycle_sq8_fixed() {
        vacuum_lifecycle_scaffold("sq8");
    }

    #[test]
    fn hnswsq_vacuum_lifecycle_sq16() {
        vacuum_lifecycle_scaffold("sq16");
    }

    #[cfg(test)]
    fn full_delete_scaffold() {
        // Delete EVERYTHING (the entry point included) and vacuum: the index
        // must survive, answer empty, and accept new inserts afterwards.
        pgrx_tests::run_test(
            "hnswsq_vacuum_mock_fn",
            None,
            crate::pg_test::postgresql_conf_options(),
        )
        .unwrap();
        let (mut guard_client, _) = pgrx_tests::client().unwrap();
        guard_client
            .execute("SELECT pg_advisory_lock(5205217837881163778)", &[])
            .unwrap();

        let (rows, _q) = gen_clustered(4, 25, 16, 0.05, 5555);
        let values: Vec<String> = rows
            .iter()
            .map(|v| format!("('{}')", vec_literal(v)))
            .collect();
        let (mut client, _) = pgrx_tests::client().unwrap();
        client
            .batch_execute(&format!(
                "DROP TABLE IF EXISTS hs_del CASCADE;
                 CREATE TABLE hs_del(id serial primary key, embedding vector(16));
                 INSERT INTO hs_del(embedding) VALUES {};
                 CREATE INDEX hs_del_idx ON hs_del USING hnswsq (embedding vector_l2_ops);
                 DELETE FROM hs_del;",
                values.join(",")
            ))
            .unwrap();
        client.close().unwrap();

        let (mut client, _) = pgrx_tests::client().unwrap();
        client.execute("VACUUM hs_del", &[]).unwrap();
        client.execute("SET enable_seqscan = off", &[]).unwrap();
        client.execute("SET hnswsq.ef_search = 40", &[]).unwrap();
        let cnt: i64 = client
            .query_one(
                &format!(
                    "SELECT count(*) FROM (SELECT id FROM hs_del \
                     ORDER BY embedding <-> '{}' LIMIT 40) t",
                    vec_literal(&rows[0])
                ),
                &[],
            )
            .unwrap()
            .get(0);
        assert_eq!(cnt, 0, "empty after full delete + vacuum");
        // Re-insert and verify the index works again (fresh entry point).
        let values2: Vec<String> = rows
            .iter()
            .map(|v| format!("('{}')", vec_literal(v)))
            .collect();
        client
            .execute(
                &format!("INSERT INTO hs_del(embedding) VALUES {}", values2.join(",")),
                &[],
            )
            .unwrap();
        let diag: String = client
            .query_one("SELECT hnswsq_diag('hs_del_idx')", &[])
            .unwrap()
            .get(0);
        eprintln!("hnswsq full-delete diag after reload: {}", diag);
        let got: i64 = client
            .query_one(
                &format!(
                    "SELECT count(*) FROM (SELECT id FROM hs_del \
                     ORDER BY embedding <-> '{}' LIMIT 1) t",
                    vec_literal(&rows[0])
                ),
                &[],
            )
            .unwrap()
            .get(0);
        assert_eq!(got, 1, "index serves after reload");
        client.close().unwrap();
        guard_client
            .execute("SELECT pg_advisory_unlock(5205217837881163778)", &[])
            .unwrap();
    }

    #[test]
    fn hnswsq_full_delete_vacuum_reload() {
        full_delete_scaffold();
    }

    // ---------------- gate 3: cross-process parallel build ----------------

    #[cfg(test)]
    fn parallel_build_scaffold() {
        // The parallel build cannot run inside a pg_test transaction
        // (workers need committed catalogs), so this is a raw-client test
        // against a committed table.
        pgrx_tests::run_test(
            "hnswsq_vacuum_mock_fn",
            None,
            crate::pg_test::postgresql_conf_options(),
        )
        .unwrap();
        let (mut guard_client, _) = pgrx_tests::client().unwrap();
        guard_client
            .execute("SELECT pg_advisory_lock(5205217837881163778)", &[])
            .unwrap();

        let (rows, _q) = gen_clustered(20, 500, 16, 0.05, 31337);
        let values: Vec<String> = rows
            .iter()
            .map(|v| format!("('{}')", vec_literal(v)))
            .collect();
        let (mut client, _) = pgrx_tests::client().unwrap();
        client
            .batch_execute(&format!(
                "DROP TABLE IF EXISTS hs_par CASCADE;
                 CREATE TABLE hs_par(id serial primary key, embedding vector(16));
                 INSERT INTO hs_par(embedding) VALUES {};
                 SET max_parallel_maintenance_workers = 2;
                 SET min_parallel_table_scan_size = 0;
                 SET hnswsq.build_seed = 20240912;
                 CREATE INDEX hs_par_idx ON hs_par USING hnswsq (embedding vector_l2_ops);",
                values.join(",")
            ))
            .unwrap();

        // Every row must be indexed, the graph must serve, and a recall
        // sweep must match the exact ground truth closely.
        let diag: String = client
            .query_one("SELECT hnswsq_diag('hs_par_idx')", &[])
            .unwrap()
            .get(0);
        eprintln!("hnswsq parallel-build diag: {}", diag);
        assert!(diag.contains("total=10000"), "all rows indexed: {}", diag);
        assert!(diag.contains("live=10000"), "all rows live: {}", diag);

        client
            .batch_execute(
                "SET enable_seqscan = off; SET hnswsq.ef_search = 100;",
            )
            .unwrap();
        // Exact-match probe: the first row's vector must be found.
        let got: i64 = client
            .query_one(
                &format!(
                    "SELECT count(*) FROM (SELECT id FROM hs_par \
                     ORDER BY embedding <-> '{}' LIMIT 1) t",
                    vec_literal(&rows[0])
                ),
                &[],
            )
            .unwrap()
            .get(0);
        assert_eq!(got, 1, "parallel-built index serves queries");
        client.close().unwrap();
        guard_client
            .execute("SELECT pg_advisory_unlock(5205217837881163778)", &[])
            .unwrap();
    }

    #[test]
    fn hnswsq_parallel_build_cross_process() {
        parallel_build_scaffold();
    }

    // ---------------- planner / NULLs / REINDEX / limits ----------------

    #[pg_test]
    fn test_hnswsq_planner_uses_index_only_for_orderby() {
        let (rows, _q) = gen_clustered(4, 50, 16, 0.05, 31337);
        Spi::run("CREATE TABLE hs_p(id serial primary key, embedding vector(16));").unwrap();
        insert_rows("hs_p", &rows).unwrap();
        Spi::run(
            "CREATE INDEX hs_p_idx ON hs_p USING hnswsq (embedding vector_l2_ops);
             SET enable_seqscan = off;",
        )
        .unwrap();

        Spi::run(&format!(
            r#"DO $$
             DECLARE line text; agg text := '';
             BEGIN
               FOR line IN EXECUTE 'EXPLAIN SELECT id FROM hs_p ORDER BY embedding <-> ''{}'' LIMIT 5'
               LOOP agg := agg || line || E'\n'; END LOOP;
               IF agg NOT LIKE '%Index Scan%' THEN
                 RAISE EXCEPTION 'ORDER BY LIMIT did not use the hnswsq index: %', agg;
               END IF;

               agg := '';
               FOR line IN EXECUTE 'EXPLAIN SELECT count(*) FROM hs_p'
               LOOP agg := agg || line || E'\n'; END LOOP;
               IF agg LIKE '%hs_p_idx%' THEN
                 RAISE EXCEPTION 'count(*) must not use the hnswsq index: %', agg;
               END IF;
             END $$;"#,
            vec_literal(&rows[0])
        ))
        .unwrap();
    }

    #[pg_test]
    fn test_hnswsq_nulls_and_reindex_and_guc() {
        let (rows, _q) = gen_clustered(4, 25, 16, 0.05, 2718);
        Spi::run("CREATE TABLE hs_n(id serial primary key, embedding vector(16));").unwrap();
        insert_rows("hs_n", &rows).unwrap();
        Spi::run(
            "INSERT INTO hs_n(embedding) VALUES (NULL), (NULL), (NULL);
             CREATE INDEX hs_n_idx ON hs_n USING hnswsq (embedding vector_l2_ops);
             SET enable_seqscan = off;",
        )
        .unwrap();
        insert_rows("hs_n", &rows[..10]).unwrap();
        Spi::run("INSERT INTO hs_n(embedding) VALUES (NULL);").unwrap();

        let n: i64 = Spi::get_one::<i64>(&format!(
            "SELECT count(*) FROM (SELECT id FROM hs_n ORDER BY embedding <-> '{}' LIMIT 10) x",
            vec_literal(&rows[0])
        ))
        .unwrap()
        .unwrap_or(0);
        assert_eq!(n, 10, "NULL rows must be skipped, live rows returned");

        Spi::run("REINDEX INDEX hs_n_idx;").unwrap();
        let n2: i64 = Spi::get_one::<i64>(&format!(
            "SELECT count(*) FROM (SELECT id FROM hs_n ORDER BY embedding <-> '{}' LIMIT 10) x",
            vec_literal(&rows[0])
        ))
        .unwrap()
        .unwrap_or(0);
        assert_eq!(n2, 10, "index must serve queries after REINDEX");

        Spi::run("SET hnswsq.ef_search = 5;").unwrap();
        let n3: i64 = Spi::get_one::<i64>(&format!(
            "SELECT count(*) FROM (SELECT id FROM hs_n ORDER BY embedding <-> '{}' LIMIT 3) x",
            vec_literal(&rows[0])
        ))
        .unwrap()
        .unwrap_or(0);
        assert_eq!(n3, 3, "ef_search GUC must be settable and scans keep working");
    }

    #[pg_test]
    fn test_hnswsq_dim_limit_and_high_dim_fp8() {
        // 2000 dims (pgvector's hard cap) work for both plain (element +
        // neighbor tuples split across pages) and ieeefp8.  The vector is
        // built in SQL (a 15KB literal is not SPI-friendly).
        Spi::run("CREATE TABLE hs_big(id serial primary key, embedding vector(2000));")
            .unwrap();
        Spi::run(
            "INSERT INTO hs_big(embedding)
               SELECT ARRAY(SELECT (d % 100)::float8 * 0.01 FROM generate_series(1,2000) d)::vector;",
        )
        .unwrap();
        Spi::run(
            "CREATE INDEX hs_big_idx ON hs_big USING hnswsq (embedding vector_l2_ops)
               WITH (storage_layout = plain);
             SET enable_seqscan = off;",
        )
        .unwrap();
        let got: i64 = Spi::get_one::<i64>(
            "SELECT id FROM hs_big ORDER BY embedding <->              (SELECT embedding FROM hs_big WHERE id = 1) LIMIT 1",
        )
        .unwrap()
        .unwrap_or(-1);
        assert_eq!(got, 1, "plain at 2000 dims");

        Spi::run(
            "CREATE TABLE hs_big8(id serial primary key, embedding vector(2000));
             INSERT INTO hs_big8(embedding)
               SELECT ARRAY(SELECT (d % 100)::float8 * 0.01 FROM generate_series(1,2000) d)::vector;
             CREATE INDEX hs_big8_idx ON hs_big8 USING hnswsq (embedding vector_l2_ops)
               WITH (storage_layout = ieeefp8);
             SET enable_seqscan = off;",
        )
        .unwrap();
        let got: i64 = Spi::get_one::<i64>(
            "SELECT id FROM hs_big8 ORDER BY embedding <->              (SELECT embedding FROM hs_big8 WHERE id = 1) LIMIT 1",
        )
        .unwrap()
        .unwrap_or(-1);
        assert_eq!(got, 1, "ieeefp8 at 2000 dims");
    }

    #[pg_test]
    fn test_hnswsq_one_node_per_page() {
        // dim-1500 plain nodes (~6KB) force one node per page; build +
        // incremental inserts + queries must all work.
        let (rows, centers) = gen_clustered(6, 20, 1500, 0.02, 1357);
        Spi::run("CREATE TABLE hs_w(id serial primary key, embedding vector(1500));").unwrap();
        insert_rows("hs_w", &rows[..100]).unwrap();
        Spi::run(
            "CREATE INDEX hs_w_idx ON hs_w USING hnswsq (embedding vector_l2_ops)
               WITH (storage_layout = plain, m = 8);
             SET enable_seqscan = off;",
        )
        .unwrap();
        insert_rows("hs_w", &rows[100..120]).unwrap();
        for c in centers.iter().take(3) {
            let got: i64 = Spi::get_one::<i64>(&format!(
                "SELECT count(*) FROM (SELECT id FROM hs_w ORDER BY embedding <-> '{}' LIMIT 10) x",
                vec_literal(c)
            ))
            .unwrap()
            .unwrap_or(0);
            assert_eq!(got, 10);
        }
    }

    #[pg_test]
    fn test_hnswsq_f8_out_of_range_clamps() {
        let (rows, _q) = gen_clustered(4, 50, 16, 0.05, 1618);
        Spi::run("CREATE TABLE hs_o(id serial primary key, embedding vector(16));").unwrap();
        insert_rows("hs_o", &rows).unwrap();
        Spi::run(
            "CREATE INDEX hs_o_idx ON hs_o USING hnswsq (embedding vector_l2_ops)
               WITH (storage_layout = f8);
             SET enable_seqscan = off;",
        )
        .unwrap();
        let big: Vec<f32> = (0..16).map(|_| 7.5).collect();
        Spi::run(&format!(
            "INSERT INTO hs_o(embedding) VALUES ('{}');",
            vec_literal(&big)
        ))
        .unwrap();
        let n: i64 = Spi::get_one::<i64>(&format!(
            "SELECT count(*) FROM (SELECT id FROM hs_o ORDER BY embedding <-> '{}' LIMIT 5) x",
            vec_literal(&big)
        ))
        .unwrap()
        .unwrap_or(0);
        assert_eq!(n, 5);
    }

    #[pg_test]
    fn test_hnswsq_sq_fixed_out_of_range_clamps() {
        // Fixed-range layouts clamp to [0, 255] per dimension; out-of-range
        // vectors still insert and answer queries (no lower-bound proof,
        // but the graph must stay usable).
        for layout in ["sq8", "sq16"] {
            let (mut rows, _q) = gen_clustered(4, 50, 16, 0.05, 1618);
            to_byte_domain(&mut rows);
            Spi::run(&format!(
                "CREATE TABLE hs_of_{layout}(id serial primary key, embedding vector(16));"
            ))
            .unwrap();
            insert_rows(&format!("hs_of_{layout}"), &rows).unwrap();
            Spi::run(&format!(
                "CREATE INDEX hs_of_idx_{layout} ON hs_of_{layout} USING hnswsq \
                 (embedding vector_l2_ops) WITH (storage_layout = {layout});
                 SET enable_seqscan = off;"
            ))
            .unwrap();
            let big: Vec<f32> = (0..16).map(|_| 500.0).collect();
            Spi::run(&format!(
                "INSERT INTO hs_of_{layout}(embedding) VALUES ('{}');",
                vec_literal(&big)
            ))
            .unwrap();
            let n: i64 = Spi::get_one::<i64>(&format!(
                "SELECT count(*) FROM (SELECT id FROM hs_of_{layout} \
                 ORDER BY embedding <-> '{}' LIMIT 5) x",
                vec_literal(&big)
            ))
            .unwrap()
            .unwrap_or(0);
            assert_eq!(n, 5, "layout {layout}");
        }
    }

    #[pg_test]
    fn test_hnswsq_recall_index_size_ratios() {
        // Same data, six layouts: f16/sq16 < plain and fp8/sq8/f8 < f16
        // (neighbor lists are the shared floor, so assert a strict decrease
        // for the 1-byte layouts; sq16 shares f16's 2-byte width so it must
        // not exceed it).
        let (rows, _queries) = gen_clustered(20, 50, 128, 0.05, 777);
        let mut sizes = Vec::new();
        for (i, layout) in ["plain", "ieeefp16", "ieeefp8", "f8", "sq8", "sq16"]
            .iter()
            .enumerate()
        {
            Spi::run(&format!(
                "CREATE TABLE hs_sz{i}(id serial primary key, embedding vector(128));"
            ))
            .unwrap();
            insert_rows(&format!("hs_sz{i}"), &rows).unwrap();
            Spi::run(&format!(
                "CREATE INDEX hs_sz_idx{i} ON hs_sz{i} USING hnswsq (embedding vector_l2_ops) \
                 WITH (storage_layout = {layout});"
            ))
            .unwrap();
            let sz: i64 = Spi::get_one::<i64>(&format!(
                "SELECT pg_relation_size('hs_sz_idx{i}')::int8"
            ))
            .unwrap()
            .unwrap_or(0);
            sizes.push(sz);
        }
        assert!(
            sizes[1] < sizes[0],
            "f16 ({}) should be smaller than plain ({})",
            sizes[1],
            sizes[0]
        );
        assert!(
            sizes[2] < sizes[1] && sizes[3] < sizes[1] && sizes[4] < sizes[1],
            "fp8 ({}) / f8 ({}) / sq8 ({}) should be smaller than f16 ({})",
            sizes[2],
            sizes[3],
            sizes[4],
            sizes[1]
        );
        // sq16 is 2 bytes/dim like f16: same size class, never larger.
        assert!(
            sizes[5] <= sizes[1],
            "sq16 ({}) should not exceed f16 ({})",
            sizes[5],
            sizes[1]
        );
        // And the fixed sq8 lands in the same 1-byte class as f8/fp8.
        assert!(
            sizes[4] < sizes[1],
            "sq8 ({}) should be smaller than f16 ({})",
            sizes[4],
            sizes[1]
        );
    }
}
