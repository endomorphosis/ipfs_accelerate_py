# Codebase IR whole-call input guard review

Two fresh-process CUDA cohorts per width compare complete baseline calls with a candidate that replaces only the outer request input guard. The retained 768D head has one historical optimizer step; the 4096D head and inputs remain synthetic. The candidate defaults to CUDA when available with optimized=False opt-out. Existing selected profiles and all 32 production acceptance rows remain unchanged.

## Complete inference observations

| Dimension | Cohort | Rows | Baseline ms | Whole-input guard ms | Baseline/candidate | Paired ratio range |
| --- | --- | --- | --- | --- | --- | --- |
| 768 | A | 1 | 23.151 | 37.101 | 0.624 | 0.595 to 0.648 |
| 768 | A | 16 | 85.091 | 91.280 | 0.932 | 0.640 to 0.989 |
| 768 | A | 32 | 151.648 | 148.746 | 1.020 | 0.731 to 1.376 |
| 768 | B | 1 | 22.287 | 36.029 | 0.619 | 0.608 to 0.635 |
| 768 | B | 16 | 84.830 | 91.264 | 0.929 | 0.577 to 0.957 |
| 768 | B | 32 | 153.261 | 151.209 | 1.014 | 0.726 to 1.331 |
| 4096 | A | 1 | 19.717 | 31.361 | 0.629 | 0.559 to 0.667 |
| 4096 | A | 16 | 96.829 | 100.097 | 0.967 | 0.962 to 0.984 |
| 4096 | A | 32 | 169.621 | 165.220 | 1.027 | 0.992 to 1.069 |
| 4096 | B | 1 | 23.944 | 38.597 | 0.620 | 0.607 to 0.672 |
| 4096 | B | 16 | 99.815 | 104.502 | 0.955 | 0.891 to 1.043 |
| 4096 | B | 32 | 178.897 | 175.180 | 1.021 | 0.995 to 1.063 |

Ratios above one mean the candidate had lower median latency for that fixed case. Each process runs both route orders six times per row count, so each route occupies each position six times. Timed public calls include completion synchronization and have no Python observer. Every return also has a separate guarded snapshot of all four output tensors. Each cohort retains 84 complete decisions and 84 four-logit panels, including 72 timed CUDA calls; the four runs total 336 complete returns and 288 timed calls. These fixed-fixture observations confer no general performance or production admission.

## Guard preservation and provenance

The candidate uses source-bound private copies of the held decode methods. Its narrow input substitution compares current finite builtin input trees with immutable snapshots and retains canonical JSON fallback. Independent function-local anchors prevent guard or snapshot replacement from blessing changed inputs. All checkpoint, receipt, policy, result, per-row cache, tensor, reference-byte and source checks remain inherited, together with every boundary and per-row cancellation poll. It does not substitute authenticated lease polling or cache a successful currentness check.

Canonical decisions and all four output tensors are checked against CPU references, alongside exact checkpoint/model state, RNG, sources, ambient policy and resource cleanup. Constructor, CPU reference, warmup, refusal-control and separate numeric scopes record four native head restorations and zero Adam construction, training fits, optimizer steps or train(True). Timed CUDA is unprofiled, so these counters are not whole-call attestations.

No fresh encoder execution, native Leanstral output provenance, trained 4096D qualification, due renewal or TTL expiry qualification is established. Earlier 8D/384D formula gains and the slower separate CG7 span candidates remain retained in their predecessor closures and were not counted as new runs.

396 current positive portable control cases passed with zero failures, errors or skips. Four independent ordinary audits reread all 208 files per successful cohort, rederive input and checkpoint bindings, complete decisions from facets and checked numeric panels, margins, balanced orders, medians and paired ratios, and recheck current source bytes. The local CIDv1 closure verifies ordinary retained bytes without executing retained Python, publishing to IPFS or granting proof authority. Failed predecessors are retained and excluded from positive counts.

Next work follows measured costs and requires complete mutation controls for any checkpoint or source comparison change. Larger resumable signed scans and per-lineage federation remain open. Supervisor planning must distinguish proposed formalizations from checker-verified evidence tied to the exact repository snapshot.

Retained evidence: [768D cohort A](../../../../artifacts/codebase_ir_terminal_bench/span-whole-input-guard-qualification-20261004-04/result.json), [768D cohort B](../../../../artifacts/codebase_ir_terminal_bench/span-whole-input-guard-qualification-20261004-05/result.json), [4096D cohort A](../../../../artifacts/codebase_ir_terminal_bench/span-whole-input-guard-qualification-20261004-06/result.json), [4096D cohort B](../../../../artifacts/codebase_ir_terminal_bench/span-whole-input-guard-qualification-20261004-07/result.json), [Ordinary audit 768D A](../../../../artifacts/codebase_ir_terminal_bench/span-whole-input-guard-audit-20261004-04/audit.json), [Ordinary audit 768D B](../../../../artifacts/codebase_ir_terminal_bench/span-whole-input-guard-audit-20261004-05/audit.json), [Ordinary audit 4096D A](../../../../artifacts/codebase_ir_terminal_bench/span-whole-input-guard-audit-20261004-06/audit.json), [Ordinary audit 4096D B](../../../../artifacts/codebase_ir_terminal_bench/span-whole-input-guard-audit-20261004-07/audit.json), [Retained 8D/384D closure](../../../../artifacts/codebase_ir_terminal_bench/formula-lease-cadence-review-20261004-01/closure.json), [Prior span candidates](../../../../artifacts/codebase_ir_terminal_bench/span-inference-candidates-review-20261004-02/closure.json).
