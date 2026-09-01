p = "/data1/pgvectorscale-rabitq/pgvectorscale/Cargo.toml"
s = open(p).read()
old = "build_parallel = []\n"
new = ('build_parallel = []\n'
       '# Build against Neon\x27s PostgreSQL fork, which has a modified smgr ABI\n'
       '# (`smgropen` takes an extra `relpersistence` argument) and a different\n'
       '# `FMGR_ABI_EXTRA` (requires `pgrx/unsafe-postgres`).\n'
       'neon = []\n')
assert s.count(old) == 1, s.count(old)
open(p, "w").write(s.replace(old, new, 1))

p2 = "/data1/pgvectorscale-rabitq/pgvectorscale/src/access_method/ivf/entry.rs"
s2 = open(p2).read()
old2 = """            let reln = (*rel).rd_smgr;
            let reln = if reln.is_null() {
                pg_sys::smgropen((*rel).rd_locator, (*rel).rd_backend)
            } else {
                reln
            };"""
new2 = """            let reln = (*rel).rd_smgr;
            let reln = if reln.is_null() {
                // Neon\x27s fork extends `smgropen` with a `relpersistence`
                // argument; vanilla PostgreSQL does not.
                #[cfg(feature = "neon")]
                {
                    pg_sys::smgropen(
                        (*rel).rd_locator,
                        (*rel).rd_backend,
                        (*(*rel).rd_rel).relpersistence,
                    )
                }
                #[cfg(not(feature = "neon"))]
                {
                    pg_sys::smgropen((*rel).rd_locator, (*rel).rd_backend)
                }
            } else {
                reln
            };"""
assert s2.count(old2) == 1, s2.count(old2)
open(p2, "w").write(s2.replace(old2, new2, 1))
print("patched ok")
