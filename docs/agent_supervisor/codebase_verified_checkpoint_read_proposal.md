# Verified native checkpoint read proposal

This is a proposed optimization, with no implementation, benchmark or qualification credit. It applies to registered native candidate checkpoints used by source-bound CodebaseIR feature runs. It supplies no Legal 384D package integration, 768D asset-owner integration or 4096D GGUF integration. Native stage outcomes belong in separate qualification reports and retained receipts.

## Observed receiving cost

A distinct source-bound candidate currently takes three complete artifact reads and hashes before lineage validation:

1. [AutoencoderRegistry.verify_artifact](../../../ipfs_datasets/ipfs_datasets_py/duckdb_control/autoencoder_registry.py#L258) calls `_hash_file` at line 85 and verifies SHA256 and byte count.
2. [autoencoder_runtime_registry._read_candidate](../../../ipfs_datasets/ipfs_datasets_py/optimizers/logic_theorem_optimizer/autoencoder_runtime_registry.py#L792) calls that verifier, then opens the file again at line 800, hashes its bytes and parses the closed candidate envelope.
3. [codebase_source_training._read_candidate](../../../ipfs_datasets/ipfs_datasets_py/logic/software_contracts/codebase_source_training.py#L210) opens the file again at line 215, hashes it and requires exact equality with canonical `_wire(saved)` bytes.

The source-bound training limits are 16 MiB per candidate and 32 MiB for ancestry, defined at lines 105–119 of the training module. The runtime's general native candidate interface is 32 MiB; the registry's default artifact ceiling is 512 MiB. The proposed caller must retain its stricter source-bound limit before opening a file.

[InventoryLineageContext.read_candidate](../../../ipfs_datasets/ipfs_datasets_py/logic/software_contracts/codebase_inventory_lineage.py#L100) already reuses a distinct verified candidate within one admitted operation and checks its retained canonical bytes for mutation. This proposal removes repeated transport work on a cache miss. It does not introduce a persistent cache or remove full lineage replay. The separate [closing model fence](../../../ipfs_datasets/ipfs_datasets_py/logic/software_contracts/codebase_inventory_scan.py#L199) independently rereads registry rows and hashes every ancestral artifact; that fence remains mandatory.

## Proposed owner API and receiving flow

Add `AutoencoderRegistry.read_verified_artifact(artifact, *, max_bytes, checkpoint=None) -> bytes`. Its result is immutable ordinary data, not an admission token, execution capability or reusable freshness receipt. The native caller remains responsible for resource admission and its existing deadline.

- Validate the exact SHA256/positive-byte descriptor and exact positive integer `max_bytes`. Charge the declared bytes against the caller and configured registry limits before opening or allocating. Keep existing ancestry and operation retention limits; count the temporary raw and canonical buffers.
- Require the same live, open registry process and its durable artifact-root binding. Capture owner generation for the operation and reject a changed owner or generation at receiving and close. Derive the path from the verified digest; accept no caller path, URL or alternate root.
- Hold directory descriptors for the canonical artifact root and digest prefix. Open the digest leaf relative to those descriptors with `O_NOFOLLOW`, `O_NONBLOCK` and `O_CLOEXEC`. Reject symlink directory aliases, nonregular files, a changed root/prefix or an unexpected descriptor size.
- Read the one held file descriptor in bounded chunks, maintaining SHA256 and the actual byte count. Check cancellation and deadline before opening, between chunks and after the read. Read at most the expected count plus one byte so growth fails without unbounded allocation.
- Compare the complete digest and count with the descriptor. Recheck held-file device, inode, size, modification and change times and the current leaf identity before returning; reject truncation, growth, replacement or mutation. Recheck the live registry owner. Metadata checks complement the complete hash and never replace it.
- Parse those exact returned bytes into the closed `{contract, feature_space, state, report}` candidate. For the source-bound profile, reject duplicate keys, malformed JSON, nonfinite numerical values, unknown fields and noncanonical bytes; require `raw == _wire(saved)` under the existing checkpoint encoding. Check cancellation/deadline again after parsing and canonical comparison.

Use a private receiving function that returns both the verified raw bytes and parsed candidate to the source-bound caller. Preserve the existing public candidate-reader return shape. The source-bound wrapper consumes the same raw bytes for its canonical comparison instead of opening the artifact a third time. Discard raw bytes when no longer needed; retain only the already bounded operation-local snapshot used for mutation checks.

Native numerical checkpoint floats remain valid under their existing finite-value, shape and optimizer validators. The signed planning contract's float-free JSON rules must not be substituted for this numerical checkpoint codec. A raw artifact SHA256 can be encoded as a CIDv1 `raw` identity only after actual bytes and exact canonical equality verify; a stored digest or a filesystem stat is insufficient evidence by itself.

## Checks that remain independent

Keep registry version identity, variant manifest, completed producer-run receipt, parent variant and numerical parent-state linkage. Preserve full contract, target-adapter, feature-space, projection, optimizer-policy, parameter and Adam validation. Keep prefit-free receiving, source-bound targets, source generation, active native SQL/CAS evidence, historical source closure, complete task population and current source observations at their original boundaries.

The closing fence rereads the actual registry rows and actual artifact bytes after the operation; it cannot use the entry byte buffer or an entry-success flag. Existing cancellation, overall deadlines, memory admission, owner cleanup and final context-exit failure propagation remain unchanged. No proof-cache obligation is discharged by weights, latents or a checkpoint hash.

Default the optimization on with an exact boolean opt-out at the trusted receiving owner/caller. The reference route retains its current repeated reads and full validations. Existing scan `optimized=False` must select that reference route explicitly. A shared registry alone cannot silently choose a different legal or source-embedding profile.

## Implementation and acceptance gates

Implement the registry read helper, the private native candidate byte/parser seam and the source-bound wrapper as a separate milestone after the selected qualification sources are unfrozen. Add meaningful native controls before running a fresh selected generation; changes to pinned semantic modules invalidate older generation compatibility.

- Instrument genuine native reads: one distinct candidate version takes three entry reads in the reference route and one in the optimized route, plus the unchanged independent closing read. Report actual bytes and distinct versions; count within-operation cache hits separately.
- Require optimized/reference equality for registry rows, full parsed candidate, contract, parameter and Adam state, parent ancestry, source bindings and inference results. No fitting, training, model promotion or proof authority may occur while receiving.
- Refuse wrong digest, wrong or boolean length, missing bytes, oversized declarations, truncated/growing files, symlink leaf/prefix/root, nonregular files, replaced inodes and in-place mutation during the held read. Include mutation after entry that only the independent closing read can catch.
- Refuse malformed, duplicate-key, unknown-field, noncanonical and nonfinite checkpoint data. Preserve rejection of altered variant manifests, completed-run metadata, parent state, parameters, Adam, model rows, current source/CAS/SQL data and stale catalog heads.
- Refuse a closed or fork-inherited registry, changed generation/root, cancellation before any read, cancellation during chunks or parse, and deadline expiry at entry or closing. Verify native transactions and the actual owned lease are released on every refusal; no successful result escapes a failed final close.
- Run bounded native candidate/lineage controls and existing registry, source-training, scan/default/opt-out/resume and receiving regressions. Then run a fresh full-population qualification with current implementation pins. Claim a speedup only from measured stages and retained evidence, not from the static read-count target.

## Separate four-head work

The [four-head integration note](codebase_larger_head_native_owner_integration_note.md) and [open Repository Proof Index work](../architecture/repository_proof_index_and_codebase_ir.todo.md) retain the broader integration and proof gates. The [full300 plan](codebase_full300_preview_and_compact_worker_plan.md) remains the source-population and worker-contract plan.

| Scope | Separate next step and qualification limit |
| --- | --- |
| CodebaseIR feature run, default latent 8 | Apply the proposed registered-candidate read path and preserve its structural float64 CPU inference and 1–64 latent-width contract. An 8-wide latent is not a semantic-text source embedding. |
| Legal 8D / published 384D and GTE-small | Existing legal inference sessions and the GTE-small CPU/CUDA model cache have separate checks. [Checkpoint Hub](../../../ipfs_datasets/ipfs_datasets_py/logic/formalization/autoencoder/checkpoint_hub.py#L129) verifies package bytes before a distinct loader rereads them; any shared-byte package loader needs a separate codec/custody proposal and real CPU/CUDA regression qualification. |
| 768D source embeddings | [embed_rows](../../../ipfs_datasets/ipfs_datasets_py/logic/formalization/autoencoder/source_embeddings_768.py#L199) loads a model per call and observes assets before load, after load and after inference. A bounded same-owner model session could reuse loading while retaining all asset/current-byte and token checks. The current reference is CPU float32 eager attention; CUDA and a trained/admitted CodebaseIR lineage require separate profiles and qualification. |
| 4096D Leanstral source embeddings | [run_full_forward](../../../ipfs_datasets/ipfs_datasets_py/logic/formalization/autoencoder/source_embeddings_4096_full_owner.py#L602) hashes the held GGUF model FD on every native operation. Amortize that hash and loading over genuinely authored rows in one bounded run; the [verified adapter](../../../ipfs_datasets/ipfs_datasets_py/logic/formalization/autoencoder/source_embeddings_4096_verified.py#L291) retains completed same-owner outputs and allows up to 128 rows. It is not a current-source or model-freshness capability. A persistent worker, incremental integrity profile or CUDA owner needs a separate native protocol, custody and cleanup qualification. |

Source representation width, structural feature-column count, learned latent width and trained task/formula head are distinct identities. Faster checkpoint reading does not qualify a 384D, 768D or 4096D CodebaseIR head, convert a lineage, increase a cap or establish formal correctness. Only checker-accepted evidence bound to the exact source generation and policy can discharge proof obligations.
