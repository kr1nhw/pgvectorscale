# .design/future — ideas recorded, deliberately unscheduled

This directory holds optimizations that were **thought through and measured
enough to judge, but that we chose not to implement now**.  Each document must
carry:

* the idea and the concrete change it implies (files/functions),
* the evidence for it — measurements if it has any, and an explicit note when it
  has none,
* the gains and the costs, quantified where possible,
* the experiment that would settle it (a benchmark, a counter, a test),
* the acceptance criteria it would have to meet, and
* links to the sibling documents it came out of.

An idea leaves this directory in one of two ways: it is implemented (the
document then moves into the relevant `.design/*.md` history or is deleted), or
it is measured to be a non-win and becomes a recorded negative result in the
perf notes.

Current documents:

| document | one-line summary |
|---|---|
| `hnswsq_backlink_measure_on_demand.md` | decide every backlink eviction from freshly measured distances instead of cached per-list distances + acceptance masks |
