# recall@10 vs latency — ivfrq & hnsw on vanilla PG17 vs Neon (x86)

| config | engine | param | value | recall@10 | p50 (ms) | p99 (ms) |
|---|---|---|---|---|---|---|
| hnsw-neon | hnsw | ef_search | 10 | 67.68 | 130.377 | 224.084 |
| hnsw-neon | hnsw | ef_search | 20 | 79.08 | 175.222 | 388.616 |
| hnsw-neon | hnsw | ef_search | 40 | 88.50 | 292.850 | 504.434 |
| hnsw-neon | hnsw | ef_search | 80 | 95.10 | 514.641 | 776.807 |
| hnsw-neon | hnsw | ef_search | 160 | 97.80 | 904.897 | 1220.966 |
| hnsw-neon | hnsw | ef_search | 320 | 99.00 | 1592.769 | 2220.374 |
| hnsw-neon | hnsw | ef_search | 640 | 99.50 | 2842.271 | 4254.562 |
| hnsw-vanilla | hnsw | ef_search | 10 | 71.30 | 1.611 | 4.165 |
| hnsw-vanilla | hnsw | ef_search | 20 | 80.80 | 2.118 | 5.574 |
| hnsw-vanilla | hnsw | ef_search | 40 | 88.40 | 2.948 | 6.617 |
| hnsw-vanilla | hnsw | ef_search | 80 | 94.30 | 4.478 | 9.417 |
| hnsw-vanilla | hnsw | ef_search | 160 | 97.90 | 6.315 | 14.926 |
| hnsw-vanilla | hnsw | ef_search | 320 | 99.00 | 8.988 | 27.778 |
| hnsw-vanilla | hnsw | ef_search | 640 | 99.70 | 12.679 | 43.839 |
| ivfrq-neon | ivf | probes | 1 | 47.23 | 389.687 | 422.798 |
| ivfrq-neon | ivf | probes | 2 | 61.50 | 397.716 | 427.371 |
| ivfrq-neon | ivf | probes | 4 | 76.80 | 403.749 | 424.640 |
| ivfrq-neon | ivf | probes | 8 | 88.50 | 418.409 | 457.222 |
| ivfrq-neon | ivf | probes | 16 | 94.90 | 464.119 | 536.424 |
| ivfrq-neon | ivf | probes | 32 | 98.10 | 520.722 | 589.665 |
| ivfrq-neon | ivf | probes | 64 | 99.40 | 626.435 | 743.478 |
| ivfrq-neon | ivf | probes | 128 | 99.60 | 803.403 | 962.328 |
| ivfrq-neon | ivf | probes | 256 | 99.40 | 1142.316 | 1334.262 |
| ivfrq-vanilla | ivf | probes | 1 | 47.10 | 1.636 | 4.675 |
| ivfrq-vanilla | ivf | probes | 2 | 61.31 | 2.267 | 6.669 |
| ivfrq-vanilla | ivf | probes | 4 | 76.57 | 2.649 | 7.150 |
| ivfrq-vanilla | ivf | probes | 8 | 88.00 | 3.071 | 7.781 |
| ivfrq-vanilla | ivf | probes | 16 | 94.40 | 3.600 | 8.339 |
| ivfrq-vanilla | ivf | probes | 32 | 97.80 | 4.525 | 10.507 |
| ivfrq-vanilla | ivf | probes | 64 | 99.30 | 6.421 | 12.066 |
| ivfrq-vanilla | ivf | probes | 128 | 99.80 | 9.402 | 15.319 |
| ivfrq-vanilla | ivf | probes | 256 | 99.80 | 15.157 | 22.382 |
