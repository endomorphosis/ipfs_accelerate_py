# Codebase IR span inference candidate review

Two fresh-process CUDA cohorts per width compare complete baseline, cached-input and authenticated-lease calls for the retained 768D head and the synthetic 4096D head. These separate candidates use CUDA when available and retain optimized=False opt-out. Existing selected profiles and all 32 production acceptance rows remain unchanged.

## Complete inference observations

| Dimension | Cohort | Rows | Baseline ms | Cached input ms | Authenticated lease ms | Baseline/cache | Baseline/lease |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 768 | A | 1 | 23.717 | 25.607 | 32.789 | 0.926 | 0.723 |
| 768 | A | 16 | 86.714 | 98.097 | 123.572 | 0.884 | 0.702 |
| 768 | A | 32 | 153.179 | 173.454 | 220.959 | 0.883 | 0.693 |
| 768 | B | 1 | 24.828 | 27.619 | 34.597 | 0.899 | 0.718 |
| 768 | B | 16 | 90.722 | 101.534 | 132.819 | 0.894 | 0.683 |
| 768 | B | 32 | 153.724 | 173.981 | 225.521 | 0.884 | 0.682 |
| 4096 | A | 1 | 23.625 | 24.511 | 30.894 | 0.964 | 0.765 |
| 4096 | A | 16 | 100.130 | 105.380 | 138.250 | 0.950 | 0.724 |
| 4096 | A | 32 | 180.623 | 186.818 | 255.338 | 0.967 | 0.707 |
| 4096 | B | 1 | 22.171 | 24.092 | 30.942 | 0.920 | 0.717 |
| 4096 | B | 16 | 97.141 | 101.077 | 131.289 | 0.961 | 0.740 |
| 4096 | B | 32 | 171.685 | 176.592 | 231.210 | 0.972 | 0.743 |

Ratios above one mean the candidate had lower median latency for that fixed case. Each process uses all six route orders twice per row count, so every route occupies each position four times. Public-call timing includes completion synchronization and no observer; a separate guarded snapshot checks all four output tensors for every return. Each cohort retains 126 complete decisions and 126 separate four-logit panels, including 108 timed CUDA calls. The four runs total 504 complete returns and 432 timed calls. Results apply to these retained fixtures; they confer no general performance or production admission.

## Lease and input scope

The span baseline already performs read-only lease cancellation checks without inline renewal. The authenticated-lease candidate adds exact child-key and live ancestry checks at every existing boundary and per-row poll, plus renewal only when due. It cannot claim the durable-write savings observed in the separate 8D/384D formula experiment. The cached-input route replaces only the unused per-row cached-output JSON binding and digest with guarded immutable input comparisons; the outer whole-call input and result guards remain intact. The two changes are deliberately measured as separate candidates here.

Complete canonical decisions, all four output tensors, source and checkpoint bindings, model state, RNG, ambient settings and lease closure are checked. Seven native refusal controls per cohort exercise caller input aliasing, lease keys, helper replacement, paired helper/snapshot replacement and actual owned-child cancellation. Each run observes six head restorations. Constructor, CPU-reference, warmup, refusal-control and separate numeric scopes record zero Adam construction, training fits, optimizer steps or train(True); timed CUDA calls are unprofiled, so those counters are not whole-call attestations. Untimed synchronous helper observations describe only those checked scopes; they do not attest timed or background I/O. Short CUDA runs do not qualify renewal when due or actual TTL expiry.

## Retained lineage and validation

The 768D checkpoint has one historical optimizer step and historical embedding receipts. The 4096D fixture has zero steps and synthetic unreceipted inputs. No fresh encoder execution, native Leanstral output provenance or trained 4096D qualification is established. The retained 8D/384D formula observations remain in their predecessor closure and were not counted as new runs.

366 current positive portable test cases passed with zero failures, errors or skips. Independent ordinary audits reread all 292 files per successful cohort, recompute decision and numeric coverage, balanced orders, medians and paired ratios, and recheck current source bytes. The closure manifest uses local CIDv1 identities for ordinary byte integrity; it neither executes retained Python nor publishes to IPFS or grants proof authority. Failed predecessors are retained and excluded from positive counts.

Next work is measured guard-cost reduction and a separately qualified combined candidate, followed by larger resumable signed successor scans and per-lineage federation. Supervisor planning must distinguish proposed formalizations from checker-verified evidence tied to the exact repository snapshot.

Retained evidence: [768D cohort A](../../../../artifacts/codebase_ir_terminal_bench/span-inference-candidates-qualification-20261004-01/result.json), [768D cohort B](../../../../artifacts/codebase_ir_terminal_bench/span-inference-candidates-qualification-20261004-03/result.json), [4096D cohort A](../../../../artifacts/codebase_ir_terminal_bench/span-inference-candidates-qualification-20261004-04/result.json), [4096D cohort B](../../../../artifacts/codebase_ir_terminal_bench/span-inference-candidates-qualification-20261004-05/result.json), [Ordinary audit 768D A](../../../../artifacts/codebase_ir_terminal_bench/span-inference-candidates-audit-20261004-01/audit.json), [Ordinary audit 768D B](../../../../artifacts/codebase_ir_terminal_bench/span-inference-candidates-audit-20261004-02/audit.json), [Ordinary audit 4096D A](../../../../artifacts/codebase_ir_terminal_bench/span-inference-candidates-audit-20261004-03/audit.json), [Ordinary audit 4096D B](../../../../artifacts/codebase_ir_terminal_bench/span-inference-candidates-audit-20261004-04/audit.json), [Retained 8D/384D closure](../../../../artifacts/codebase_ir_terminal_bench/formula-lease-cadence-review-20261004-01/closure.json).
