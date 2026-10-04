# Input snapshot candidate qualification

The candidate does not show a consistent CUDA speedup. The 8D singleton median was 14.3%, 44.4% slower than v2 in cohorts A and B; the 384D 32-row median saved 4.8%, 1.3%. Every case except cohort B's 8D singleton has paired timings on both sides of 1, so the candidate remains experimental and the selected inference routes are unchanged.

The new input guard avoids canonical JSON and digest creation for ordinary temporary mutation snapshots. Checkpoint identities and receipt digests retain their existing format.

The current CPU control stages pass 793 cases with no failures, errors or skips. Two independently audited fresh-process CUDA cohorts each retain 240 public returns, 240 complete projections, 216 timed calls and 12 cold Adam restorations. Both checkpoints retain their prior 30 training steps; this stage performs no new training. Each count runs all six route orders twice.

The table compares complete guarded public calls plus completion synchronization. Ratios above 1 favor the new v3 input guard. Each pair has 12 samples. Block ratios use each consecutive balanced six-order block; paired ranges expose within-run variation.

The original caller-admitted route is faster in both cohorts. The bitwise routes exercise additional custody checks, which contribute to complete-call time.

| Cohort | Head | Rows | Original ms | v2 ms | v3 ms | v2/v3 | Original/v3 | Paired range | Six-order block ratios |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 8 | 1 | 6.559 | 46.790 | 53.476 | 0.8750 | 0.1227 | 0.6753–1.6451 | 0.9134, 0.9708 |
| A | 8 | 16 | 36.457 | 72.760 | 73.968 | 0.9837 | 0.4929 | 0.8618–1.0747 | 0.9766, 0.9855 |
| A | 8 | 32 | 9.231 | 47.522 | 46.943 | 1.0123 | 0.1966 | 0.7885–1.4474 | 1.0420, 1.0001 |
| A | 384 | 1 | 8.032 | 49.769 | 52.724 | 0.9440 | 0.1523 | 0.7554–1.2533 | 1.0254, 0.8375 |
| A | 384 | 16 | 39.760 | 87.680 | 87.019 | 1.0076 | 0.4569 | 0.9389–1.1976 | 1.0316, 0.9822 |
| A | 384 | 32 | 13.541 | 57.952 | 55.169 | 1.0505 | 0.2455 | 0.9825–1.4096 | 1.1347, 1.0420 |
| B | 8 | 1 | 16.457 | 47.083 | 67.980 | 0.6926 | 0.2421 | 0.5817–0.9797 | 0.6697, 0.8101 |
| B | 8 | 16 | 36.656 | 74.742 | 75.571 | 0.9890 | 0.4851 | 0.8519–1.0475 | 0.9804, 1.0023 |
| B | 8 | 32 | 8.957 | 55.253 | 57.540 | 0.9603 | 0.1557 | 0.6234–1.3504 | 1.0653, 0.8889 |
| B | 384 | 1 | 9.748 | 61.397 | 53.254 | 1.1529 | 0.1830 | 0.6515–1.5995 | 1.1172, 1.0628 |
| B | 384 | 16 | 42.016 | 92.592 | 90.060 | 1.0281 | 0.4665 | 0.9378–1.2053 | 1.0322, 1.0542 |
| B | 384 | 32 | 14.320 | 66.793 | 65.957 | 1.0127 | 0.2171 | 0.8270–1.0542 | 1.0057, 1.0044 |

These are descriptive local timings. They do not establish a general speedup or permit automatic default promotion. CUDA timed intervals have no Python profiler; source-bound training and restore observations cover construction, CPU references, warmups and controls.

The 8D/384D candidate changes only per-call rows, projection IDs and latents snapshots. All four fresh custody, currentness and cancellation boundaries, numerical kernels, checkpoint ownership, and cleanup remain inherited. Per cohort, native controls retain 28 earlier CUDA refusals and add six actual post-forward snapshot custody refusals across both heads; 20 primitive input refusals have a separate ordinary scope. Each cohort validates complete projections within 5e-5 of the original CPU reference and compares every other public field, allowing only the known finite decision-margin numeric difference and implementation metadata. The ordinary reader independently reconstructs CPU projection algebra; it does not replay the recurrent grammar.

The 768D and 4096D adapters change the request-local cached-row input binding. CPU tensor controls preserve cached output clones, sign checks, input tensors, single use and independent local snapshot custody. Their CUDA latency and complete native decode qualification remain pending. The retained 768D fixture is a one-step checkpoint; the 4096D fixture remains synthetic and untrained, and supplies no native Leanstral output evidence.

The memory-pressure admission refusal and the earlier refusal-message test failure are retained. The test expected a later method refusal, while the implementation correctly refused at the alias check; its successor accepts either refusal. No shared admission configuration was changed. The final observation has zero allocated resources, live leases and waiters. All 32 production acceptance rows remain open. Existing CUDA selection and explicit CPU opt-out remain; qualified optimizations are still intended to default on with an opt-out.

The next formula candidate should separate fresh authenticated lease checks from due TTL renewal. The existing cancellation reader checks ancestry without writing, but it does not authenticate the lease key; using it alone would lose a current authority check. A new source-bound candidate needs that authentication at every boundary, race and expiry controls, and the existing durable write sequence for real renewal. This is a proposed next step, with no scheduler change or measured gain in this stage.

Evidence: [cohort A](../../../../artifacts/codebase_ir_terminal_bench/formula-input-guard-candidate-qualification-20261004-01/result.json), [audit A](../../../../artifacts/codebase_ir_terminal_bench/formula-input-guard-candidate-audit-20261004-01/audit.json), [cohort B](../../../../artifacts/codebase_ir_terminal_bench/formula-input-guard-candidate-qualification-20261004-02/result.json), [audit B](../../../../artifacts/codebase_ir_terminal_bench/formula-input-guard-candidate-audit-20261004-02/audit.json), [CPU stage 1](../../../../artifacts/codebase_ir_terminal_bench/input-guard-candidate-controls-20261004-01/receipt.json), [CPU stage 2](../../../../artifacts/codebase_ir_terminal_bench/input-guard-reader-span-controls-20261004-02/receipt.json), [CPU stage 3](../../../../artifacts/codebase_ir_terminal_bench/input-guard-assembly-v2-controls-20261004-01/receipt.json), [resource closure](../../../../artifacts/codebase_ir_terminal_bench/input-guard-candidate-resource-closure-20261004-01/receipt.json), [input guard](../../../ipfs_datasets/ipfs_datasets_py/optimizers/logic_theorem_optimizer/input_content_guard.py), [formula candidate](../../../ipfs_datasets/ipfs_datasets_py/optimizers/logic_theorem_optimizer/modal_latent_formula_bitwise_device_inference_v3.py), [span candidate](../../../ipfs_datasets/ipfs_datasets_py/optimizers/logic_theorem_optimizer/legal_span_cached_input_device_inference.py), [preceding cost profile](codebase_multi_head_guard_cost_profiling.md).
