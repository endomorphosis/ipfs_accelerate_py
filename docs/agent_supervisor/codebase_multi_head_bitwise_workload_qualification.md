# Bitwise verification across inference heads and workloads

This increment follows the [4096D CUDA and verification qualification](codebase_4096_cuda_bitwise_qualification.md). It preserves the previously qualified producers and adds separate inference profiles. Width, IR family, task codec and encoder/checkpoint lineage remain distinct admission identities.

## Broader 4096D measurements

The [new workload producer](../../../ipfs_datasets/benchmarks/qualify_synthetic_4096_guard_workloads.py) completed its [actual CUDA run](../../../../artifacts/codebase_ir_terminal_bench/native-4096-guard-workloads-qualification-20261004-01/result.json) in 47.13 seconds, including 46.08 seconds after root admission. It uses the same untrained checkpoint as the preceding qualification: hidden size 8, an actual `[4, 4096]` adapter, zero output-adapter initialization, no learned latent conditioning and no native Leanstral encoder.

Nine authored workloads cover 5, 13 and 25 source tokens with 8, 16 and 32 rows. Token widths are equal within each workload; this does not qualify ragged batches, arbitrary repository files or the 256-token model maximum. Each workload retains three CPU observations per original/candidate lane and twelve paired CUDA observations per lane. Order alternates, with each CUDA lane first six times. All 270 timed public returns and their canonical projections are retained. Serialization occurs outside the measured public call.

| Source tokens | Rows | Original CUDA / bitwise v2 CUDA | Conservative observed gain |
| --- | --- | --- | --- |
| 5 | 8 | 1.060x | Yes |
| 5 | 16 | 1.036x | No |
| 5 | 32 | 1.033x | No |
| 13 | 8 | 1.063x | Yes |
| 13 | 16 | 1.048x | No |
| 13 | 32 | 1.017x | No |
| 25 | 8 | 1.030x | No |
| 25 | 16 | 1.034x | No |
| 25 | 32 | 1.038x | No |

These are ratios of lane medians for complete inference calls, including source/checkpoint verification, tensor comparisons, immutable reference-byte transfers, input validation and CPU canonical decisions. The separate paired-ratio analysis uses all twelve adjacent observations. Its conservative rule requires every paired ratio above 1.02, paired median above 1.05, both first-order group medians above 1.02 and a descriptive bootstrap lower bound above 1.02. Only two workloads meet that rule; the whole-suite flag is false.

The bootstrap resamples complete pairs with a fixed seed and retains its raw ratios, method and interval. Serial correlation, stationarity and independence are not established, so nominal intervals do not establish future repeatability. Large timing outliers remain in the evidence. The selected existing route is preserved; these observations support a candidate comparison rather than universal default promotion.

Every retained timed result matched the singleton CPU reference's complete canonical projection and declared device profile. All four logits were replayed once per lane/workload within checked private forward scopes and compared at an absolute 5e-5 tolerance. This does not claim that every timed call's logits were separately retained. Peak Torch allocation was 51,971,584 bytes. Checkpoint and Adam bytes, listed producer bytes and ambient precision settings were unchanged. There were zero fits, optimizer constructions, optimizer steps or encoder calls. Owned child/root leases closed, and workspace cleanup restored Torch allocation and active leases to zero.

The [independent ordinary reader](../../../ipfs_datasets/benchmarks/audit_synthetic_4096_guard_workloads.py) passed [38 controls](../../../../artifacts/codebase_ir_terminal_bench/native-4096-workload-reader-controls-20261004-01/receipt.json) and the [actual archive audit](../../../../artifacts/codebase_ir_terminal_bench/native-4096-workload-closed-audit-20261004-01/audit.json). It checks all 397 closed archive paths, all 270 public returns and canonical joins, complete four-logit snapshots, checkpoint/immutable-anchor bytes, declared source profiles, cleanup and independently recomputed statistics. The largest independently recomputed logit error is 1.78814e-7. Current producer sources match the retained copies. Missing, aliased, changed, nonfinite or false-authority archives refuse. Ordinary-byte consistency does not attest CUDA execution or grant proof authority.

## Separate 768D profile

The [new 768D session](../../../ipfs_datasets/ipfs_datasets_py/optimizers/logic_theorem_optimizer/legal_span_device_bitwise_inference.py) preserves the existing batched numerical and canonical kernels. Its constructor establishes immutable checkpoint reference bytes from the validated CPU restoration before GPU upload. Current mutable references are checked against those bytes at every original full-value boundary, then the bitwise comparator checks model/reference equality. Matching finite or signed-zero mutations of both mutable copies therefore refuse.

The new profile is `native-768-source-span-batched-device-bitwise-checkpoint-anchor-float32/v1`. Optimization defaults on and `optimized=False` retains CPU singleton execution. The original strict CUDA GRU profile is unchanged. Additional policy/lease identity and conservative anchor-memory checks are declared in the new profile. Unsupported non-float32, noncontiguous or unresolved lazy tensor metadata refuse rather than silently entering the comparator.

The [42 CPU controls](../../../../artifacts/codebase_ir_terminal_bench/native-768-bitwise-controls-20261004-02/receipt.json) passed, including source/method replacement, paired mutations before and after forward, anchor replacement, CPU opt-out, memory bounds and original guard ordering. Their reused fixture has explicitly authored positive step/moment fields; it supplies no actual fitting or encoder evidence. The [first attempt](../../../../artifacts/codebase_ir_terminal_bench/native-768-bitwise-controls-20261004-01/receipt.json) timed out before admission under the unchanged pressure gate and ran no tests.

## Separate 8D and 384D formula profile

The [new formula adapter](../../../ipfs_datasets/ipfs_datasets_py/optimizers/logic_theorem_optimizer/modal_latent_formula_bitwise_device_inference.py) supports the existing `legacy_hub_v1` 8D and `current_legal_v2` 384D checkpoint lineages. Its profile is `modal-latent-formula-bitwise-owned-device-float32/v1`. CUDA is selected when available by default; `optimized=False` selects CPU. Existing producers remain byte-pinned and separate from this candidate.

The adapter restores and validates the CPU checkpoint once, establishes immutable float32 reference bytes before reference cloning and GPU upload, and preserves inherited scalar/batched formula decoding. It adds owned admission, cancellation, source/method/storage/hook checks and entry/exit input guards. Its CPU path also checks physical model bytes, so a flushed subnormal mutation cannot pass through numerical equality. CUDA comparisons use integer float32 bits. Each construction retains one existing Adam restoration, with no optimizer steps or fitting; optimizer-free inference restoration is a future startup optimization.

All [142 current CPU controls](../../../../artifacts/codebase_ir_terminal_bench/native-formula-bitwise-controls-20261004-05/receipt.json) passed. They include packed-storage and singleton-stride regressions; the portable checkpoints are authored zero-fit fixtures. The earlier [138-case CPU pass](../../../../artifacts/codebase_ir_terminal_bench/native-formula-bitwise-controls-20261004-04/receipt.json) is historical and is not counted again. A [native CUDA attempt](../../../../artifacts/codebase_ir_terminal_bench/bitwise-trained-head-device-qualification-20261004-02/result.json) exposed a CUDA GRU packing mismatch: the original anchor layout included CPU physical storage offsets, which differ after CUDA packs recurrent parameters. The fix uses logical name/shape/element coverage for the immutable byte anchor, captures separate model/reference physical layouts after upload, and independently compares the uploaded model's complete bytes with the CPU checkpoint anchor before accepting construction. Every subsequent boundary checks those physical layouts as well as complete byte/value custody.

Two [admission](../../../../artifacts/codebase_ir_terminal_bench/native-formula-bitwise-controls-20261004-02/receipt.json) [timeouts](../../../../artifacts/codebase_ir_terminal_bench/native-formula-bitwise-controls-20261004-03/receipt.json) ran no tests under the unchanged memory-pressure gate. An earlier [caller preflight refusal](../../../../artifacts/codebase_ir_terminal_bench/native-formula-bitwise-controls-20261004-01/preflight-refusal.json) records a corrected nonexistent filename in the source list.

## Retained trained-head CUDA replay

The [trained-head producer](../../../ipfs_datasets/benchmarks/qualify_bitwise_trained_head_devices.py) completed its [CUDA run](../../../../artifacts/codebase_ir_terminal_bench/bitwise-trained-head-device-qualification-20261004-04/result.json) in 40.65 seconds, including 39.42 seconds after root admission. It reused two 30-step formula checkpoints and a retained one-step 768D child checkpoint with historical receipt-bound GTE vectors. These are small diagnostic fixtures. There were no new embeddings, fits or optimizer steps; eight formula session constructions each restored existing Adam state. The 768D restoration remains optimizer-free.

The archive retains 240 public reports: 216 timed paired returns across three heads and row counts 1, 16 and 32, eighteen CPU reference/opt-out returns and six CUDA warmups. Every formula return retains its complete projected vector and compares it with CPU at 5e-5 absolute tolerance. Every timed public result matches the complete canonical CPU projection. All four 768D logits have separate checked snapshots once per lane/count; this does not mean that every timed call retained logits. Each CUDA lane runs first six times among twelve pairs. Both CUDA sessions are admitted together, and no CPU model test overlaps timing.

| Head | Rows | Original CUDA median, ms | Candidate CUDA median, ms | Original / candidate |
| --- | --- | --- | --- | --- |
| 8D formula | 1 | 6.27 | 45.24 | 0.139x |
| 8D formula | 16 | 36.69 | 76.82 | 0.478x |
| 8D formula | 32 | 8.88 | 49.00 | 0.181x |
| 384D formula | 1 | 7.55 | 49.95 | 0.151x |
| 384D formula | 16 | 40.05 | 85.72 | 0.467x |
| 384D formula | 32 | 15.35 | 63.02 | 0.244x |
| 768D span | 1 | 21.93 | 23.37 | 0.939x |
| 768D span | 16 | 87.81 | 90.45 | 0.971x |
| 768D span | 32 | 160.13 | 162.93 | 0.983x |

The new formula sessions are substantially slower on these fixtures; the 768D sessions are slightly slower. This is a negative performance result, despite successful integrity and numerical qualification. The new adapters add custody checks that the original formula/768D profiles do not provide, so this comparison does not establish identical corruption behavior. It supplies no basis for replacing the selected route. Timings include existing source-guard diagnostic output and complete public-call verification; they do not isolate a tensor kernel.

Native controls reject paired finite mutations, admitted zero-to-signed-zero changes, anchor replacement, subnormal bits, input mutation after observed forward, post-forward paired mutations and cancellation. Formula subnormal controls replace an existing nonzero value; only the 768D native control demonstrates zero-to-smallest-subnormal refusal. Peak allocation was 51,226,624 bytes. All twelve child releases and the root release were observed, framework workspace cleanup returned Torch allocation to zero, and the shared scheduler reported zero active leases. Source, checkpoint, Adam and precision-policy bytes/settings remained unchanged.

The [first trained replay](../../../../artifacts/codebase_ir_terminal_bench/bitwise-trained-head-device-qualification-20261004-01/result.json) timed out at child admission; the [third](../../../../artifacts/codebase_ir_terminal_bench/bitwise-trained-head-device-qualification-20261004-03/result.json) timed out before root admission. Both remain unqualified. The second attempt exposed the corrected GRU layout bug and retains its seven partial public reports. Historical failed-attempt constructor counts cover completed restorations only; they are not counts of every attempted constructor.

## Next measured optimization

Static inspection found substantial avoidable receipt work in the new formula profile. A warm call has four fresh verification boundaries, with about 24 direct source-file SHA reads per boundary, excluding input guards. It invokes native source verification eight times and creates twenty implementation/currentness deep copies per call; three boundary receipts are discarded. The original route has two boundaries and about eighteen direct file reads per call. These are source-level counts, not measured cost attribution. The native source pin checks five listed files and resolves the compiler/decompiler/parser checkout; it is not a whole-repository incremental Merkle verification.

The next separate formula profile should preserve all four currentness boundaries, full checkpoint/reference comparisons and explicit lease renewal, while reusing observations only within the same fresh boundary and formatting receipts only when they are returned. A first candidate can reduce duplicate native verification from eight to four passes without caching prior successful checks. It also needs a live binding and refusal control for the inherited `CheckpointContentGuard.matches` method, which the current formula profile does not bind separately. That gap remains a candidate-promotion prerequisite. Optimizer-free restoration should be measured separately as a cold-start change; it does not explain warm-call ratios.

The next comparison should retain the current producers, compare original/v1/v2 on the same trained fixtures, and keep raw balanced complete-call samples, exact decisions and complete numerical outputs. The broader 4096D reader's retained-entry enumeration also needs an allocation bound before collecting directory names; its present qualification covers the fixed 397-file archive. The new closure assembler checks bounded directory entries before collecting them. Native Leanstral ownership, receipt-bound positive-step 4096D training, family task semantics, full scans, signed dispatch and one-model-per-lineage federation retain their existing gates. All 32 production acceptance rows remain open.

## Final evidence closure

The [trained-head reader](../../../ipfs_datasets/benchmarks/review_bitwise_trained_head_devices.py) passed [78 controls](../../../../artifacts/codebase_ir_terminal_bench/bitwise-trained-head-reader-controls-20261004-03/receipt.json) and the [actual archive audit](../../../../artifacts/codebase_ir_terminal_bench/bitwise-trained-head-closed-audit-20261004-02/audit.json). It checks all 697 files and 240 unique public reports, all 216 paired returns, 160 formula projections and twelve complete span-logit snapshots. It independently joins inputs, source profiles, float32 checkpoint anchors, lineage bindings, canonical filenames, child ownership, numeric precision and cleanup. All four span logit arrays independently reproduce their public decisions. It rejects coordinated report tampering, repeated panel references, wrong device scopes, nested authority claims and malformed/nonfinite JSON. The audit executes no model, encoder, tensor library or retained Python.

The [closure builder's 64 controls](../../../../artifacts/codebase_ir_terminal_bench/guard-workload-builder-controls-20261004-01/receipt.json) passed. Together with 142 formula, 42 span, 38 broader-reader and 78 trained-reader controls, this increment has 364 passing cases. The prior 443-control qualification is inherited historical evidence and was not rerun. Failed and superseded attempts remain retained and are excluded from the current total.

The [additive plan](../../../ipfs_datasets/docs/autoencoders/ir_family_dimension_guard_workload_followup_plan.json) and [machine review](../architecture/repository_proof_index_and_codebase_ir.guard_workload_review.json) record these scopes and the bounded local CIDv1 DAG-JSON manifests. Those CIDs identify manifest bytes; they do not attest native execution, prove formal correctness or publish data to IPFS. The selected optimized defaults and CPU opt-out remain unchanged, and every production acceptance row remains open.
