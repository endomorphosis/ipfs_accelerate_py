# Four-head inference guard cost profiling

The diagnostic increment covers 8D, 384D, 768D and 4096D. It identifies input snapshot/structural work and durable renewals as candidates for investigation. It establishes no speedup: every reported time includes instrumentation overhead, and the previous uninstrumented mixed result remains in force. CUDA when available, CPU opt-out and existing selected routes remain unchanged.

## Actual run and integrity scope

[Formula run](../../../../artifacts/codebase_ir_terminal_bench/formula-guard-cost-profile-20261004-03/result.json): 42 public reports and 42 complete projections, 18 Python/native profiles, 12 native Adam restorations and zero new training. [768D run](../../../../artifacts/codebase_ir_terminal_bench/dimension-768-guard-cost-profile-20261004-03/result.json): 18 public reports, 12 complete four-logit snapshots, 6 Python profiles and 2 native traces. [4096D run](../../../../artifacts/codebase_ir_terminal_bench/dimension-4096-guard-cost-profile-20261004-04/result.json): 27 public reports, 18 four-logit snapshots, 9 Python profiles and 3 native traces. All retained numeric comparisons and canonical decisions passed.

The formula checkpoints retain 30 training steps. The 768 checkpoint retains one child step and an independently byte-bound three-step parent;30 was its historical time budget. The 4096 checkpoint is synthetic and untrained. No encoder or native Leanstral endpoint is invoked in this increment, and native ownership/origin and trained 4096 qualification remain open.

Python phase profiles cover each CUDA route at1,16 and 32 rows. Dimensional native traces cover only one row per route;16/32 native traces are explicitly absent. Formula native traces cover all three counts. Native buffers are bounded by fixed workloads with postcollection checks, not allocator enforcement. Source-selected exclusive spans reconcile to complete public spans; unselected work remains in its nearest selected parent. Inclusive costs must not be added, and Python/native clocks are not aligned.

## Selected costs at 32 rows

Values below are milliseconds from one instrumented observation. Guard constructor cost includes only work inside the public call; 768 has 34 guard constructions for input/cached-output/provenance work. These are not cold model or Adam constructor timings.

| Head | Retained diagnostic route | Public wall | Guard constructor exclusive | Guard.matches exclusive | fsync exclusive |
| --- | --- | --- | --- | --- | --- |
| 8 | bitwise_v2_cuda | 157.43 | 2.46 | 23.20 | 29.58 |
| 384 | bitwise_v2_cuda | 302.16 | 68.54 | 60.50 | 31.24 |
| 768 | bitwise_v1_cuda | 1659.66 | 470.21 | 235.65 | 0.00 |
| 4096 | bitwise_v2_cuda | 3927.19 | 1459.09 | 70.40 | 0.00 |

The [complete selected-cost summary](../../../../artifacts/codebase_ir_terminal_bench/guard-cost-review-inputs-20261004-01/summary.json) retains all 33 Python profiles and 23 native trace joins. It derives summaries from audited producer aggregates; the ordinary readers independently reconstruct raw span/aggregate consistency. Python profiling observes per-atom identity callbacks, so structural/input costs can be strongly amplified. The numbers do not predict uninstrumented savings.

Each formula guarded32-row call records four renewals, four durable replacements and eight fsyncs. Fsync wall time is 25–31ms across v1/v2 versus roughly 0.4–0.5ms threadCPU. Flock totals under 0.1ms give no observed contention. Original public profiles exclude caller lease management outside the call, so zero observed scheduler spans do not imply zero whole-job scheduler work. Native32-row CUDA event durations total under 1ms for the formula routes; these are event sums, not complete-call latency.

## Closed evidence and next experiment

[Formula audit](../../../../artifacts/codebase_ir_terminal_bench/formula-guard-cost-audit-20261004-02/audit.json), [768D audit](../../../../artifacts/codebase_ir_terminal_bench/dimension-768-guard-cost-audit-20261004-01/audit.json) and [4096D audit](../../../../artifacts/codebase_ir_terminal_bench/dimension-4096-guard-cost-audit-20261004-01/audit.json) passed exact result/file joins, current source pins, numeric/public reports, raw Python/native trace consistency and cleanup. These ordinary readers import no tensor library or model and do not execute retained Python. They establish ordinary-byte consistency, not execution or proof attestation.

The [final controls](../../../../artifacts/codebase_ir_terminal_bench/guard-cost-final-controls-20261004-03/receipt.json) passed 408 receipt cases: 210 test items plus 198 subtests, with no failures/errors/skips. Earlier stages remain retained and are excluded from this count. [Preserved refusals](../../../../artifacts/codebase_ir_terminal_bench/guard-cost-review-inputs-20261004-01/preserved-refusals.json) records admission pressure, source-code identity collision, checkpoint-step metadata, CPU-profiler PID interpretation and native/callback diagnostic caps. Successors preserve the refused evidence. Callback limits are 1M for formula, 4M for 768 and 8M for the specifically pinned 4096 v6 diagnostic; other limits and the shared admission configuration are unchanged.

Each successful job observed root/child releases and returned owned Torch allocation to its baseline. The [final resource observation](../../../../artifacts/codebase_ir_terminal_bench/guard-cost-resource-closure-20261004-01/receipt.json) reports zero allocations, active leases and waiters. No foreign process actions or resource-gate changes occurred.

The [follow-up plan](../../../ipfs_datasets/docs/autoencoders/ir_family_guard_cost_profiling_followup_plan.json) selects a separate immutable input guard/atomic-array experiment. It may omit canonical JSON/digest only for input snapshots whose digest is unused; checkpoint and receipt identity digests remain. Preserve exact finite atoms, signed zero, float64-level mutations, fallback JSON semantics, source/alias identities and all four fresh custody/cancellation boundaries. Require focused mutation controls and balanced uninstrumented complete-call qualification before selecting a faster route.

The [machine review](../architecture/repository_proof_index_and_codebase_ir.guard_cost_review.json) joins the three audits, summary, controls and configuration. The ordinary closure records local CIDv1 DAG-JSON manifest identifiers for explicitly selected archives; it performs no IPFS publication. All 32 production acceptance rows remain unchanged and open. Full resumable successor scans, signed worker dispatch, native Leanstral ownership, one-model-per-lineage federation and later gradient synchronization remain in the carried plan.
