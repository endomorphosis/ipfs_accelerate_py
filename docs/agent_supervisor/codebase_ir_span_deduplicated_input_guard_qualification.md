# Codebase IR deduplicated whole-call input guard review

Two fresh-process CUDA cohorts per width compare complete baseline calls, the held whole-input guard and a separate candidate that deduplicates method-binding records by exact owner identity and attribute name. The retained 768D head has one historical optimizer step; the 4096D head and inputs remain synthetic. Candidates default to CUDA when available with optimized=False opt-out. Existing selected profiles and all 32 production acceptance rows remain unchanged.

## Complete inference observations

| Dimension | Cohort | Rows | Baseline ms | Whole-input ms | Deduplicated ms | Baseline/whole-input | Baseline/deduplicated | Whole-input/deduplicated | Paired whole-input/deduplicated range |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 768 | A | 1 | 25.158 | 39.835 | 39.916 | 0.632 | 0.630 | 0.998 | 0.976 to 1.306 |
| 768 | A | 16 | 87.094 | 95.350 | 94.291 | 0.913 | 0.924 | 1.011 | 0.972 to 1.140 |
| 768 | A | 32 | 157.334 | 156.584 | 156.557 | 1.005 | 1.005 | 1.000 | 0.930 to 1.378 |
| 768 | B | 1 | 25.748 | 40.497 | 40.237 | 0.636 | 0.640 | 1.006 | 0.973 to 1.048 |
| 768 | B | 16 | 93.321 | 99.342 | 99.591 | 0.939 | 0.937 | 0.997 | 0.971 to 1.092 |
| 768 | B | 32 | 154.270 | 152.288 | 153.637 | 1.013 | 1.004 | 0.991 | 0.948 to 1.345 |
| 4096 | A | 1 | 21.497 | 35.058 | 34.973 | 0.613 | 0.615 | 1.002 | 0.981 to 1.031 |
| 4096 | A | 16 | 94.614 | 97.633 | 98.911 | 0.969 | 0.957 | 0.987 | 0.890 to 1.019 |
| 4096 | A | 32 | 176.183 | 169.606 | 171.332 | 1.039 | 1.028 | 0.990 | 0.947 to 1.117 |
| 4096 | B | 1 | 21.528 | 34.537 | 34.928 | 0.623 | 0.616 | 0.989 | 0.907 to 1.110 |
| 4096 | B | 16 | 96.840 | 100.445 | 99.830 | 0.964 | 0.970 | 1.006 | 0.973 to 1.176 |
| 4096 | B | 32 | 177.328 | 168.355 | 168.751 | 1.053 | 1.051 | 0.998 | 0.945 to 1.030 |

Ratios above one mean the denominator route had lower median latency for that fixed case. Each process repeats all six route permutations twice per row count, so each route occupies each of the three positions four times. Timed public calls include completion synchronization and have no Python observer. Every return also has a separate guarded snapshot of all four output tensors. Each cohort retains 126 complete decisions and 126 four-logit panels, including 108 timed CUDA calls; the four runs total 504 complete returns and 432 timed calls. All three comparison families and both balanced six-trial blocks are retained. These fixed-fixture observations confer no general performance or production admission.

## Guard preservation and provenance

The deduplicated candidate retains the held source-bound private decode copies and whole-call input guard. Its only additional change removes six repeated binding records for identical owner objects and attribute names; distinct receivers, aliases and wrapped checks remain. Every binding check first checks the independently captured immutable inventory roots and counts, then checks each distinct live inherited attribute. Independent function-local input anchors still prevent root or nested snapshot replacement from blessing changed inputs. All checkpoint, receipt, policy, result, per-row cache, tensor, reference-byte and source checks remain inherited, together with every boundary and per-row cancellation poll. It does not substitute authenticated lease polling or cache a successful currentness check.

Canonical decisions and all four output tensors are checked against CPU references, alongside exact checkpoint/model state, RNG, sources, ambient policy and resource cleanup. Constructor, CPU reference, warmup, refusal-control and separate numeric scopes record six native head restorations and zero Adam construction, training fits, optimizer steps or train(True). Each candidate route has seven guarded caller/snapshot corruption refusals and a final actual owned-child cancellation refusal, totaling sixteen controls per cohort. Timed CUDA is unprofiled, so these counters are not whole-call attestations.

No fresh encoder execution, native Leanstral output provenance, trained 4096D qualification, due renewal or TTL expiry qualification is established. Earlier 8D/384D formula evidence and the CG8 whole-input comparison remain pinned by predecessor closures; CG7 span candidates are retained transitively through CG8. Their 396 CG8 control cases are not counted as new CG9 cases.

542 current positive portable control cases passed with zero failures, errors or skips. Four independent ordinary audits reread all 293 files and 37 current source/fixture/configuration pins per successful cohort, rederive input and checkpoint bindings, complete decisions from facets and checked numeric panels, margins, three-way balanced orders, medians and paired ratios. The local CIDv1 closure verifies ordinary retained bytes without executing retained Python, publishing to IPFS or granting proof authority. Failed predecessors are retained and excluded from positive counts.

Next work follows measured costs and requires complete mutation controls for any checkpoint or source comparison change. Larger resumable signed scans and per-lineage federation remain open. Supervisor planning must distinguish proposed formalizations from checker-verified evidence tied to the exact repository snapshot.

Retained evidence: [768D cohort A](../../../../artifacts/codebase_ir_terminal_bench/span-binding-dedup-qualification-20261004-01/result.json), [768D cohort B](../../../../artifacts/codebase_ir_terminal_bench/span-binding-dedup-qualification-20261004-02/result.json), [4096D cohort A](../../../../artifacts/codebase_ir_terminal_bench/span-binding-dedup-qualification-20261004-03/result.json), [4096D cohort B](../../../../artifacts/codebase_ir_terminal_bench/span-binding-dedup-qualification-20261004-04/result.json), [Ordinary audit 768D A](../../../../artifacts/codebase_ir_terminal_bench/span-binding-dedup-audit-20261004-01/audit.json), [Ordinary audit 768D B](../../../../artifacts/codebase_ir_terminal_bench/span-binding-dedup-audit-20261004-02/audit.json), [Ordinary audit 4096D A](../../../../artifacts/codebase_ir_terminal_bench/span-binding-dedup-audit-20261004-03/audit.json), [Ordinary audit 4096D B](../../../../artifacts/codebase_ir_terminal_bench/span-binding-dedup-audit-20261004-04/audit.json), [Retained 8D/384D closure](../../../../artifacts/codebase_ir_terminal_bench/formula-lease-cadence-review-20261004-01/closure.json), [Prior CG8 whole-input comparison and its CG7 predecessor](../../../../artifacts/codebase_ir_terminal_bench/span-whole-input-guard-review-20261004-01/closure.json).
