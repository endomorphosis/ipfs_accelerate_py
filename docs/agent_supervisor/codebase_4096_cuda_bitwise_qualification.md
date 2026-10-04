# 4096D CUDA execution and bitwise verification

The untrained 4096D architecture fixture now passes actual CPU/CUDA replay, including the independent checkpoint-reference byte anchor. A separate bitwise inference session adds a measured verification candidate while preserving the original session and checkpoint producers. This continues the [768D/4096D qualification](codebase_768_4096_head_followup_qualification.md). Native Leanstral embeddings, positive-step head training, model quality and repository scan admission remain separate gates.

## Original 4096D session

The [qualified v2 run](../../../../artifacts/codebase_ir_terminal_bench/native-4096-synthetic-head-device-v2-qualification-20261004-04/result.json) completed in 5.24 seconds with zero fits, optimizer constructions, optimizer steps or encoder calls. Checkpoint and Adam bytes were unchanged. It used 32 short authored sources, an untrained GRU with hidden size 8, and a real `[4, 4096]` adapter whose output layer is initialized to zero. This establishes the numerical architecture and inference protocol, without establishing learned latent conditioning or a trained Leanstral model.

Each timing is the median of three warm complete public inference calls. It includes checkpoint/source checks, full tensor comparisons, reference-byte copies, input validation and CPU canonical decisions.

| Rows | CPU singleton opt-out / batched CUDA |
| --- | --- |
| 1 | 0.59x |
| 16 | 1.20x |
| 32 | 1.51x |

The GPU is slower for one row in this fixture. Canonical decisions, source tokens, facet selections, input hashes and admission flags matched exactly. All four saved logit tensors were compared against actual singleton CPU forwards for each batch size; the largest error was 1.20e-7, within the declared absolute 5e-5 bound. Floating diagnostic scores are retained and evaluated numerically instead of requiring byte-identical float serialization.

Peak Torch allocation was 51,148,800 bytes and process RSS was 1,760,972,800 bytes. Strict GRU precision restored the admitted ambient policy. Owned children and the root lease closed; post-workspace Torch allocation and active lease count returned to zero. These are reported execution measurements and local custody checks, without kernel-resource or execution-origin attestation.

The three preceding failed attempts remain retained. Two refused the harness's incorrect freshness requirement: the unchanged scheduler's CUDA hardware telemetry had already initialized the fresh worker's context. The third reached CUDA but compared whole result rows, including floating diagnostics. The qualified harness accepts the scheduler bootstrap only with zero existing tensor allocations before head imports, then compares canonical decisions independently of numerical tolerances. Its source is preserved for replay.

## Bitwise verification and session

The new [bitwise guard](../../../ipfs_datasets/ipfs_datasets_py/optimizers/logic_theorem_optimizer/owned_tensor_bitwise_guard.py) reinterprets contiguous float32 storage with `view(int32)` and checks every bit plus reference finiteness. Equal finite bit patterns imply finite current values and preserve signed-zero detection. Every CUDA chunk contributes to one final host decision. CPU and explicit opt-out retain the existing numeric, sign-bit and finiteness comparisons. Metadata, independent-storage and conservative temporary-memory admission precede tensor views and allocations.

On two authored CUDA tensors shaped `[128, 4096]` and `[128, 128]`, the candidate measured 1.64x faster than the existing combined guard and 2.08x faster than the scalar reference lane. This is a standalone primitive measurement. All three lanes refused changed values, signed zero, the smallest positive float32 subnormal, NaN and infinity in both state and reference. The full v2 session also refused model-only subnormal corruption, paired model/reference corruption and anchor replacement. Matching paired finite mutations require the session's independent immutable checkpoint anchor; the standalone helper does not provide that custody.

The separate [BitwiseDeviceLeanstral4096SpanSession](../../../ipfs_datasets/ipfs_datasets_py/optimizers/logic_theorem_optimizer/legal_span_4096_bitwise_device_inference.py) replaces only the tensor comparator. Its guard body retains every original custody clause and check ordering. It binds the inherited implementation, class, methods and checkpoint schema, retains all five full checks in a batched call, and exposes the profile `native-4096-source-span-batched-device-bitwise-float32-cpu-decisions/v1`. CUDA/batching defaults on; `optimized=False` keeps the inherited CPU opt-out profile and numerical behavior. Synthetic inference still requires explicit `synthetic_unreceipted=True`; caller vectors or JSON cannot open native Leanstral admission.

The new controls passed: [87 guard cases](../../../../artifacts/codebase_ir_terminal_bench/native-4096-bitwise-guard-controls-20261004-01/receipt.json), [27 session cases](../../../../artifacts/codebase_ir_terminal_bench/native-4096-bitwise-session-controls-20261004-01/receipt.json) and [30 owner-diagnostic cases](../../../../artifacts/codebase_ir_terminal_bench/native-4096-owner-diagnostic-controls-20261004-01/receipt.json), with zero failures, errors or skips. They complement the previous 199 unchanged head/guard controls. No fits or optimizer steps occur in these controls.

The [separate full-session benchmark](../../../ipfs_datasets/benchmarks/qualify_synthetic_4096_bitwise_session_device.py) also [passed actual four-lane qualification](../../../../artifacts/codebase_ir_terminal_bench/native-4096-bitwise-session-device-qualification-20261004-01/result.json) in 9.23 seconds. Both CUDA sessions were admitted together and timed with four alternating-order samples per batch, with each lane running first twice. CPU timings used three samples. All decisions, logit tolerances and corruption refusals passed; the largest logit error remained 1.20e-7. The standalone guard still measured 1.67x faster, but that gain did not improve complete inference:

| Rows | Original CUDA / bitwise v1 CUDA | Original CPU / bitwise v1 CPU |
| --- | --- | --- |
| 1 | 0.95x | 0.73x |
| 16 | 0.99x | 0.81x |
| 32 | 0.98x | 0.78x |

Ratios below one mean the bitwise session was slower. Review identified duplicated original implementation checks and discarded receipt construction in v1's hot path. This result does not support selecting v1 as a performance improvement.

The separate [v2 session](../../../ipfs_datasets/ipfs_datasets_py/optimizers/logic_theorem_optimizer/legal_span_4096_bitwise_device_inference_v2.py) removes that duplication. Each inherited public `_check` still verifies the original transitive sources; each `_pure_check` verifies current new-module bytes and class/callable bindings, and the actual comparator verifies its own source before tensor views. Descriptions reuse freshly verified results. The independent implementation entry point still verifies both sets of sources. No verification success is cached, and all five tensor/reference checks and immutable anchor transfers remain. Cancellation callbacks retain their original surrounding checks. The new session uses the distinct profile `native-4096-source-span-batched-device-bitwise-float32-cpu-decisions/v2`, with optimization defaulting on and `optimized=False` selecting the inherited CPU path.

Its [27 controls](../../../../artifacts/codebase_ir_terminal_bench/native-4096-bitwise-session-v2-controls-20261004-01/receipt.json) passed, including actual verifier call counts, source/dependency replacement, AST custody equivalence, CPU opt-out and paired mutations. The [four-lane v2 run](../../../../artifacts/codebase_ir_terminal_bench/native-4096-bitwise-session-v2-device-qualification-20261004-01/result.json) passed in 9.51 seconds using the same fixture and four alternating-order CUDA observations per batch:

| Rows | Original CUDA / bitwise v2 CUDA | Original CPU / bitwise v2 CPU |
| --- | --- | --- |
| 1 | 1.18x | 0.87x |
| 16 | 1.04x | 0.88x |
| 32 | 1.02x | 0.90x |

The CUDA medians improved in this bounded run, while CPU calls remained slower. Four samples without confidence intervals do not establish repeatable gains near one. The original CPU/CUDA route remains the selected route; v2 is a separate candidate API. The numeric fixture remains untrained and does not qualify native Leanstral outputs, model quality or full repository workloads. The saved canonical return is the final timed result; the separate logit replay is checked numerically. The harness does not retain every timed return. Its `complete_gpu_session_speedups_qualified` field means consistent timing evidence; the separate `bitwise_full_session_speedup_demonstrated` field records whether any observed median ratio exceeds one.

The [closed reader](../../../ipfs_datasets/benchmarks/audit_synthetic_4096_head_device.py) passed [73 authored controls](../../../../artifacts/codebase_ir_terminal_bench/native-4096-closed-audit-controls-20261004-03/receipt.json). It independently checks externally pinned ordinary result bytes, retained/current producer joins, complete float32 checkpoint/reference anchors, input hashes, canonical projections, all four logit shapes/errors, raw timing medians and ratios, mutation refusals, and reported lease/allocation cleanup. Matching CPU/private projections cannot grant authority through row or nested formal-output flags. Comparator receipt schema, mode and implementation must match the declared lane, and malformed containers return structured refusals. It imports no Torch or retained producer code and grants no execution or proof authority.

The [owned closed audits](../../../../artifacts/codebase_ir_terminal_bench/native-4096-device-closed-audits-20261004-03/receipt.json) accepted all three successful archives with their current sources, preserved v1's false speedup result, and retained all three failed attempts as unqualified. A [prior audit attempt](../../../../artifacts/codebase_ir_terminal_bench/native-4096-device-closed-audits-20261004-02/receipt.json) timed out before admission because observed memory pressure exceeded the unchanged 2% limit. The retry admitted normally after pressure subsided. The final suites contain 244 new passing controls: 87 guard, 27 v1 session, 27 v2 session, 30 owner diagnostic and 73 closed reader cases. The earlier 43- and 44-case reader receipts remain historical; they are not added to that total. Together with the preceding 199 unchanged head/guard controls, 443 distinct cases have passing retained receipts. All own leases closed, and the final ordinary resource snapshot reported zero active leases.

## Leanstral owner prerequisite observation

The [new read-only diagnostic](../../../ipfs_datasets/ipfs_datasets_py/logic/formalization/autoencoder/source_embeddings_4096_owner_diagnostic.py) completed a [bounded native observation](../../../../artifacts/codebase_ir_terminal_bench/leanstral4096-owner-prerequisite-diagnostic-20261004-01/receipt.json). Before/after systemd, process birth, executable, command digest, source/build/library and model-stat observations matched. Secrets are excluded. The model was statted; its weight bytes were not read or verified.

At observation time, the existing launch lacked embedding mode and explicit last pooling. Local llama.cpp headers and the library expose native output-width, pooling, tokenization and pooled-output APIs, but the current Python routes cannot borrow the foreign service's model/context pointers. GET metadata and exported symbols do not establish an operation owner or native numerical outputs. Actual output width and device remain unverified, and all production/proof authority flags remain false. The service and its configuration were untouched.

The next owner milestone requires complete, untruncated native4096 outputs, an admitted model-content verification boundary, native entry/closing callbacks, pooling/token/context/device observations, single-client admission and owned cancellation/cleanup. Only then can receipt-bound positive-step training and immutable trained-head evaluation precede family-specific scans, signed successor dispatch and same-lineage federation. All 32 production acceptance rows remain unchanged and open.

## Follow-up across all widths

The [additive follow-up plan](../../../ipfs_datasets/docs/autoencoders/ir_family_dimension_cuda_guard_followup_plan.json) retains all four widths and all four IR families. The earlier 8D/384D qualifications and 768D numerical replay remain separate evidence. A legal single-source span fixture cannot qualify CodebaseIR, IntentIR or SecurityIR task semantics.

For 384D/768D rollout, use new versioned session profiles around the existing formula, span and embedding implementations. Independently anchor checkpoint references before applying paired-mutation authority; the bitwise helper alone cannot provide it. Admit only contiguous float32 state to that comparator and retain exact checks for non-float32 buffers. Compare full public calls on trained heads and varied source lengths under the same configured resource owner before selecting any new default. Reuse existing CPU/CUDA opt-out behavior, keep family/task/encoder/checkpoint identities explicit, and preserve all previous qualified producer bytes.

Subsequent milestones remain complete repository dispositions and resumable successors for admitted family/task profiles, signed worker dispatch joined to current source/model/checker identities, and one federated model per compatible lineage. No cross-width averaging is admitted. Future gradient hooks remain scoped to `ipfs_accelerate_py` and `mcp-plus-plus`.

The [machine review](../architecture/repository_proof_index_and_codebase_ir.cuda_guard_review.json) pins this report, plan, current and retained sources, successful measurements, failed attempts, and bounded directory manifests. CIDv1 values address local DAG-JSON manifest bytes; no IPFS publication or additional execution/proof authority is implied.
