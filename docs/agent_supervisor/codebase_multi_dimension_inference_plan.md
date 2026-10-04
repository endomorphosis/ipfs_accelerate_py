# Repository inference across 8D, 384D, 768D and Leanstral 4096D

This expands the repository proof-index work to all four requested input widths. It covers the encoder and the learned decoder head separately. A width in an inventory is a proposed input representation, not evidence that a trained head, CUDA path or proof exists. Each family, task, input profile and decoder architecture has its own lineage and qualification gate.

The completed [successor milestone](codebase_successor_expansion_qualification.md) remains a closed historical report. Its full CPU8D successor scan, signed native worker dispatch, Legal8D/384D formula execution and GTE-small CUDA measurements do not grant qualification to a new width or CodebaseIR decoder.

## Concrete scope

| Width | Source encoder | Head work | Current qualification scope |
| --- | --- | --- | --- |
| 8D | Registered bounded structural features; CPU producer remains pinned | Existing native Legal heads and CodebaseIR inference lineage | Full bounded CPU successor scan; native Legal formula CUDA fixture |
| 384D | Pinned GTE-small, CUDA selected by the new private device session when available | Existing Legal formula/device heads; general Codebase384D remains a separate gate | Native numerical GTE/Legal CUDA evidence; published Legal384D private head replay |
| 768D | Pinned complete GTE multilingual model, native 768 values, CLS/L2; new persistent device session | New private device session for the native dimensional source-span checkpoint architecture | CPU controls first, then actual pinned-model encoder and fresh bounded head CPU/CUDA comparison; no selected production head asserted |
| 4096D | Leanstral reports input hidden width 4096; numerical output requires verified embedding-mode owner integration | A separate 4096-input adapter/head and resumable optimizer lineage after real outputs are available | Read-only current-service preflight and protocol controls; current embedding capability is unqualified |

The 768D input width differs from its decoder's hidden width, adapter rank and output vocabulary. The existing warm-start initialization is not a trained production head. Likewise, the current Leanstral model's reported 4096 hidden width does not establish an embedding endpoint, pooling policy, native output width, numerical embedding or learned head. Structural hashes and token limits of 4096 are unrelated to this model width.

## Inference implementation and admission

Optimization remains opt-out: new device sessions use CUDA when available; `optimized=False` selects CPU. New device/dtype/pooling profiles preserve the identities of historical CPU producers. Old checkpoints retain their original producer pins and cannot silently inherit a different representation.

For 768D, load every stored encoder/classifier tensor once from the complete pinned model, retain owned tensors and tokenizer, and batch source rows through the resident encoder. Verify the full seven-file asset manifest on admission. Keep ordinary-file descriptors and check file identity, source implementation, tokenizer state and private tensor values at entry and exit. The tensor comparison must detect writes through `.data`, including writes that leave Torch's version counter unchanged. These local custody checks are sequential observations, not an atomic filesystem snapshot or a distributed execution attestation.

For each call, reject overlength input instead of truncating; record actual token IDs, padded batch shape, forward device/dtype, profile and finite native output width. Bound rows, total tokens, padded tokens and attention cells independently. Bucket compatible complete spans by token length before batching once correspondence and ordering tests pass. Record cold admission separately from warm throughput; report small-batch regressions as well as gains. Cancellation and wall deadlines surround native operations and cannot interrupt an already running kernel.

The head owns a private copy of its authenticated parameters. Its CPU and CUDA paths must preserve inherited architecture and canonical decisions within a declared numerical tolerance, without fitting during inference. Preserve checkpoint and Adam state, source profile, PID/thread ownership and nonreentrance. Acquire the shared resource lease before GPU allocation; charge RAM and GPU allocations against the same physical pool on unified-memory hosts. A child session releases only its own lease. Record actual device memory and measured process memory rather than infer memory use from model-file size.

The native768 GRU needs strict CUDA float32 precision. Its scoped CuDNN policy disables TF32, preserves the other ambient flags, synchronizes before restoration and verifies that the ambient policy was restored. Torch exposes these flags process-wide, so use an owned inference worker process without unrelated concurrent CuDNN operations in the same process. The batched head computes the complete valid-source batch, copies four output tensors to CPU once and uses the inherited per-source canonical decoder. It rejects an over-budget padded batch before tensor allocation; it does not silently truncate or change devices.

A head checkpoint binds its input `context_contract.representation_id` to the encoder's `profile_id` and its `context_contract.producer_sha256` to `profile_sha256`. Setting `optimized=False` on the head runs the CPU reference for those same admitted inputs. Changing the encoder's device or limits changes its profile and requires an explicitly compatible checkpoint. Pass the embedding receipts and separately admitted receipt pins into the head. Generate full receipt pins at the trusted producer boundary with the existing `legal_span_formula.checkpoint_digest(receipt)` helper, then retain and deliver those pins through the checkpoint admission channel. That helper uses ASCII canonical JSON; the encoder's profile and embedding digest fields use UTF-8 canonical JSON. The distinction matters for non-ASCII receipt IDs. Recomputing a pin from consumer-supplied data establishes only that data's self-consistency.

```python
# Inputs and pins have already crossed the admitted producer boundary.
decoded = head.decode_formal_logic(
    texts, vectors, embedding_receipts=receipts,
    expected_receipt_sha256s=trusted_receipt_pins,
)
```

Leanstral integration requires an owner-verified current process/model/backend/launch binding, embedding mode, native output dimension, tokenizer, pooling, normalization, physical batch limit, context limit and actual device. The installed read-only metadata omits several of these fields. The new adapter must refuse numeric production until that owner integration exists; caller JSON and plausible 4096-number arrays cannot grant authority. Client concurrency stays bounded independently of the server's advertised slots. Existing generation service configuration is preserved.

The 4096D head milestone follows that encoder gate. Add a versioned checkpoint architecture and admission path for a genuine 4096-input adapter, with its own Adam state, training index and family/task codec. The current device source-span implementation accepts 768D only; preserve that checkpoint schema and producer identity while adding the new lineage. Bind each training row to the verified Leanstral output receipt and source/context bytes. Evaluate source-only, true-latent and ablated inputs, then replay the same immutable checkpoint on CPU and CUDA without fitting. Controls must reject 768D or 384D inputs, stale encoder receipts, incompatible optimizer state and cross-family checkpoints. Measure canonical output parity, memory and throughput separately before enabling repository scans or federated aggregation for that head. Padding a smaller vector or projecting it to 4096 does not satisfy the native Leanstral input gate.

## Repository scans and proof-index correspondence

1. Capture one repository source head and complete membership/disposition catalog. Each dimensional scan references that same admitted source and semantic head. Include inferred, deferred, opaque, parse-failed, unsupported and unindexed members so faster inference cannot hide missing files.
2. Extract complete source/context spans with exact byte ranges, dependencies and tokenizer measurements. Seal profile-specific budgets before training. A compact vector is a learned representation; it does not contain an independently established formal proof.
3. Cache embeddings by source/context digest, complete encoder/tokenizer/profile identity, dtype and normalization. Cache head candidates by embedding identity plus checkpoint/task/codec/output budget. Share raw source capture across widths; keep numerical cache namespaces separate.
4. Convert candidates into typed IR and proof obligations. Independently validate applicable obligations with the selected symbolic checker and retain evidence keyed to the source head, obligation, checker version and premises. DuckDB/DuckLake tables distinguish candidate logic, checked evidence and proof eligibility.
5. On repository change, rescan changed members and dependency closure, resume bounded pages from the exact admitted predecessor, and retain dispositions for the full successor. Reuse only artifacts whose source/context/model/checker identities still match. No persistent freshness token bypasses signed admission or the worker's currentness checks.
6. Join IntentIR requests to the current typed CodebaseIR evidence and explicit missing obligations. The supervisor planner may select tests, edits, training or prover work to close those obligations; it cannot treat a model output or a retrieval match as a proof.

Extend the existing full-scan, signed successor and worker controls independently for each new producer/task. Test source races, replacement, stale joins, changed checkpoint/Adam, duplicate dispatch, cancellation and restart boundaries. Preserve the all-32 production table and its open gates until their own evidence is complete.

## Training, storage and federation

Train CodebaseIR on the fly as a candidate child of an admitted same-family/task/profile parent. Freeze the repository training index, source/semantic roots, tokenizer/profile, task codec and evaluation split. Keep shorter-span replay and compare source-only, true-latent and ablation controls before promotion. Fresh bounded setup fits used for numerical CUDA qualification are reported separately and do not select a production model.

Use tensor-native immutable checkpoint payloads with a small canonical metadata manifest for future training artifacts. Keep weights and Adam moments out of repeated JSON conversion. Use Parquet for immutable worker updates and scan/index rows, with one serialized owner per DuckDB/DuckLake catalog. Quack acceleration can be evaluated behind that storage contract; a database file is not a concurrent worker synchronization protocol. Existing JSON checkpoint formats remain readable and pinned until a versioned equivalent loader and integrity boundary are qualified.

Bind immutable payloads to exact bytes, shape/dtype/tensor names and CIDv1 manifests. Verify complete bytes at first admission, then use owned file/tensor custody for repeated local operations. Incremental transfer verification, content addressing and current runtime integrity are separate checks. A changed shard invalidates its tensor and manifest identity; an unchanged hash alone does not prove a current process used it.

Federation combines compatible updates into one model per lineage. Each round binds base checkpoint, optimizer policy, dimensions, family/task/profile, training index, worker identity, sample weighting and finite update tensors. Do not average 8D, 384D, 768D and 4096D parameters, or two different heads with the same input width. Admit signed updates, reject stale/duplicate rounds and publish one immutable aggregate after evaluation. Add future gradient synchronization through ipfs_accelerate_py and MCP++ using the same compatibility/admission contract, explicit step IDs, cancellation and bounded transport. Scheduling hooks do not assert working distributed gradient execution.

## Ordered milestones

| Milestone | Deliverable and exit gate |
| --- | --- |
| M1: Native 768D sessions | Additive encoder/head APIs; meaningful CPU custody and device controls; unchanged historical producer pins |
| M2: Actual 768D execution | Real pinned model CPU/CUDA, native 768 outputs and token parity; bounded fresh-head canonical parity; warm/cold timings; checkpoint/Adam unchanged during inference; resources closed |
| M3a: Leanstral owner capability | Read-only native preflight now; trusted owner witness/backend support next; genuine untruncated 4096 outputs required before a head fit or throughput claim |
| M3b: Native 4096D head | Separate versioned architecture, complete receipt-bound training index and Adam lineage; source-only/latent/ablation evaluation; immutable CPU/CUDA replay, numeric and canonical parity, bounded memory and throughput before scan admission |
| M4: Four-width repository cache | Profile-specific schema/cache adapters, full dispositions and source correspondence; resumable successor and signed worker qualification per admitted task |
| M5: Training and federation | Tensor payload/checkpoint equivalence, immutable update shards, same-lineage aggregation, independent promotion/replay; later explicit gradient transport qualification |

The [16-cell expansion inventory](../../../ipfs_datasets/docs/autoencoders/ir_family_dimension_inference_expansion_plan.json) tracks all four widths across CodebaseIR, SecurityIR, LegalIR and IntentIR as planning scope. It creates no runtime database, remote repository, trained model or production qualification.

The [current qualification report](codebase_multi_dimension_inference_qualification.md) records M1's implementation and controls, M2's encoder measurements and pending head CUDA benchmark, and M3a's discovery controls. M3b through M5 remain planned exit gates; the metadata preflight does not establish a trained 4096D head.
