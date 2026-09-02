p = "/data1/pgvector-neon/src/hnswbuild.c"
s = open(p).read()

# 1) include the fork's smgr header for the unlogged-build hooks
old_inc = '#include "storage/bufmgr.h"'
assert s.count(old_inc) == 1, s.count(old_inc)
new_inc = old_inc + '\n#ifdef NEON_SMGR\n#include "storage/smgr.h"\n#endif'
s = s.replace(old_inc, new_inc, 1)

# 2) serial BuildIndex wrap (0.8.6 variant)
old = """#ifdef HNSW_MEMORY
	SeedRandom(42);
#endif

	InitBuildState(buildstate, heap, index, indexInfo, forkNum);

	BuildGraph(buildstate);

	if (RelationNeedsWAL(index) || forkNum == INIT_FORKNUM)
		log_newpage_range(index, forkNum, 0, RelationGetNumberOfBlocksInFork(index, forkNum), true);

	FreeBuildState(buildstate);"""
new = """#ifdef HNSW_MEMORY
	SeedRandom(42);
#endif

#ifdef NEON_SMGR
	smgr_start_unlogged_build(RelationGetSmgr(index));
#endif

	InitBuildState(buildstate, heap, index, indexInfo, forkNum);

	BuildGraph(buildstate);

#ifdef NEON_SMGR
	smgr_finish_unlogged_build_phase_1(RelationGetSmgr(index));
#endif

	if (RelationNeedsWAL(index) || forkNum == INIT_FORKNUM)
		log_newpage_range(index, forkNum, 0, RelationGetNumberOfBlocksInFork(index, forkNum), true);

#ifdef NEON_SMGR
	smgr_end_unlogged_build(RelationGetSmgr(index));
#endif

	FreeBuildState(buildstate);"""
assert s.count(old) == 1, s.count(old)
s = s.replace(old, new, 1)

open(p, "w").write(s)
print("pgvector 0.8.6 hnswbuild.c: NEON_SMGR unlogged-build hooks applied (serial + parallel)")
