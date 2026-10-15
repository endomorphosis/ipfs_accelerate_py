# Explicit artifact mirror custody qualification

**All 35 selected checks passed; the candidate remains unapplied.** Thirty synthetic controls and five retained-artifact probes qualify a read-only mirror under the current UID. This extends the [tracked store and expected pin milestone](codebase_tracked_store_and_pin_inventory_qualification_v1.md) by preserving original artifact references in a separately approved mirror. It supplies local custody evidence, without installing or starting a worker.

The [owned receipt](../../../../artifacts/codebase_ir_terminal_bench/compact-artifact-mirror-owned-checks-20261011-02/receipt.json) is 66,323 bytes, SHA256 `c6629a5ecd78c949af55228f8304f681256b8e1e317feed2640f2035c7cb0baa`. [Child observations](../../../../artifacts/codebase_ir_terminal_bench/compact-artifact-mirror-owned-checks-20261011-02/checks.json) are 41,723 bytes, SHA256 `9a200d87a7b965c45009689562695a6d3c0e1b54800c00f57d9ae629cb186191`. Owned elapsed time was 15.726 seconds, including a 15,608 ms bounded child. These are custody-control durations, not inference benchmarks.

## Separate mirror identity and explicit inventory

The [mirror implementation](../../../../artifacts/codebase_ir_terminal_bench/compact-artifact-worker-mirror-candidate-20261011-02/mirror.py) gives its configuration a separate CID. Each original reference retains its source store ID, role, codec, CID, SHA256 and byte count. Copying an owner store to another root would change that store's identity; the mirror preserves the original source identities.

The caller supplies one complete canonical source pin@2, independent source-pin and source-store CIDs, and the expected owner/plan/task/lineage retention binding. The closed inventory includes that pin, its complete ordered references and false authority flags. Publication, reader opening, complete verification and final close decode the complete selected closure and replay its original summary and any recorded context through the explicitly supplied codec. The codec retains the existing eight roles; no signed companion or V8 child role is added.

The [candidate README](../../../../artifacts/codebase_ir_terminal_bench/compact-artifact-worker-mirror-candidate-20261011-02/README.md) records the pre-run design. The owning harness pins the unchanged previous store, journal, reference contract, context contract and fixtures. Injected Python helpers remain owner-controlled code; their memory and global state are outside execution attestation.

## Publication and reader custody

Initial support requires the current UID, private 0700 directories and immutable 0444 single-link ordinary files. It does not qualify another UID, a container mount, an ACL or a remote transport. Publication creates a new root only, writes the complete inventory last, fsyncs files and directories, and independently reopens the result with the original held directory identities.

References and material bytes are detached before helper callbacks. Complete aggregate charges precede body decoding and publication: at most 32 unique members and 32 MiB, with the original per-role body ceilings. Source pins are bounded to 256 KiB and inventories to 384 KiB. Caller-bound configuration and inventory identities select the resolver; no ambient registry, source-store callback, caller URL or network fallback exists.

The reader holds directory descriptors, walks canonical ancestors with `O_NOFOLLOW`, and compares held/named file identities before and after reads. Its exact root and blob populations reject missing and extra objects. Process, thread, configuration, codec identity and reentrancy guards apply throughout. `read(reference)` returns exact bytes only for an inventory-listed original reference. Final close replays complete custody, deactivates the reader and attempts every owned descriptor release even when verification refuses.

Interrupted publication may retain incomplete files, which opening refuses. There is no overwrite, repair, garbage collection or atomic transaction across the source, mirror and caller's independently trusted inventory anchor. Publication reports `external_inventory_anchor_persisted=False`; the caller must establish that persistence before any future worker installation. Arbitrary same-user mutation and hardware crash attestation are outside the custody claim.

## Controls and retained full scans

The [sealed execution plan](../../../../artifacts/codebase_ir_terminal_bench/compact-artifact-worker-mirror-candidate-20261011-02/plan.json) pins 39 ordinary inputs. The [30 synthetic controls](../../../../artifacts/codebase_ir_terminal_bench/compact-artifact-worker-mirror-candidate-20261011-02/controls.py) cover original five-role replay, compatible complete metadata/context replay, preserved source identities, exclusion of unpinned metadata, independent identity/binding forks, missing materials, corruption, links, modes, extra files, root replacement, thread/closed-handle refusal and callback mutation after snapshots.

A distinct aggregate control supplies a coherently rehashed source-pin declaration with three unique 16-MiB RAW references. It allocates no bodies and requires the exact 32-MiB aggregate refusal before any codec helper lookup. Final-close controls verify inactive state, an empty owned-descriptor inventory and actual `EBADF` for every saved descriptor. Compatible full metadata positives remain synthetic and their signature bytes remain unverified.

The [retained probes](../../../../artifacts/codebase_ir_terminal_bench/compact-artifact-worker-mirror-candidate-20261011-02/retained_probe.py) published and reopened both original five-role full-scan closures. Together their source store contains 25 unique inner payloads totaling 2,362,315 bytes, with exactly two tracked source pins.

| Retained mirror | Members | Payload bytes | Actual result |
| --- | ---: | ---: | --- |
| Default | 13 | 1,494,217 | Complete publication, exact reads and reopen |
| Reference | 13 | 1,494,240 | Complete publication, exact reads and reopen |

Each caller separately fsynced an inventory-anchor file and its parent directory, reloaded its exact three fields, and used those reloaded mirror/inventory identities to open the reader. Each anchor was 238 bytes. The source pin and reference bytes remain unchanged; the mirror and source store have different custody IDs.

Three further probes required the actual refusals `reference is outside the selected mirror inventory`, `complete exact mirror population required`, and `complete mirror population changed`. A valid source object from the other scan did not trigger fallback. Deleting a mirror member refused opening while the source object remained retained. Deleting a member under a live reader refused final close and closed all saved owned descriptors. Restoration of exact fixture bytes served only the distinct negative checks; it is not a production repair API. All temporary source, journal, mirror and anchor workspaces were removed.

Retained exports wrap canonical bodies in `{artifact_cid, value}`. The probe selects the inner value, encodes RAW numerical roles with their ASCII canonical form and roots with UTF-8 DAG-JSON, and checks the reconstructed CID against the export before publication. The [first unqualified receipt](../../../../artifacts/codebase_ir_terminal_bench/compact-artifact-mirror-owned-checks-20261011-01/receipt.json) preserves an export-reader framing failure after its 30 synthetic passes and successful cleanup. Only the corrected receipt above supplies the complete 35-check credit.

The historical parent and current scan still have different declared source/model lineages. These real positives concern retained original-role custody. They do not establish a compatible complete successor context, parent planning replay, new signatures or current source/model acceptance.

## Owned cleanup and finite integrity closure

The owned job used unchanged existing MAIN with two CPU slots, 2048 MiB and one child, a 180-second deadline, 120-second lease TTL and 30-second automatic heartbeat. Source and output bytes closed under the live reservation. The child was reaped, readers/stdin joined, the workspace removed, the owning lease released and MAIN left idle.

Six observed audit categories were zero: project-target imports, model-file reads, database reads, network calls, subprocess calls and unselected repository pathname reads. Descriptor-relative provenance is outside that audit inference; framed output publication follows the audited phase. Inference/training zeros follow from selected source and absence of backend calls. The child loaded no captured native project implementation; the parent used the captured scheduler/process boundary for reservation and cleanup.

The [final finite integrity closure](../../../../artifacts/codebase_ir_terminal_bench/worker-mirror-final-close-20261011-01.json) adds two bounded sequential ordinary-file observations of retained evidence and this milestone's selected inputs and report links. Removed temporary mirrors and anchors supply historical receipt observations only and are excluded from current input pins. The closure grants no native/signature/model replay or continuous currentness.

## Next signed and native integration gates

The live worker route remains manifest@7, declaration@1, pending@3 and planning/public@5. The proposed manifest@8 profile needs its own complete signed schema and decoder before any runtime switch. Next add the genuine signed-companion role and complete parent/companion joins, then actual signature checks and full child/parent/companion planning replay.

Preserve the distinct canonical encodings: existing [signature authority](../../ipfs_accelerate_py/agent_supervisor/control/profile_authority.py) signs ASCII-escaped canonical payload JSON, while [content identity](../../ipfs_accelerate_py/agent_supervisor/proof/formal_verification_contracts.py) uses canonical UTF-8 JSON. A future signature qualification must include a non-ASCII payload fixture. Recorded CID/SHA agreement does not authenticate a signature.

Approve actual worker-readable custody, add explicit shared native receiving with fresh source/model/checker/root-policy observations after application callbacks, and preserve one-use close and scope exit. Complete staged Intent installation/rollback, reservation, worker construction, START, authored checks, completion and STOP before activation.

No live package source changed. Trained semantic 8D, 384D, 768D, 4096D Leanstral and CUDA qualification remain separate from structural 53-feature/latent-8 custody evidence. Formal proof still requires checker-accepted evidence bound to the exact source and policy; latents and similarity scores supply planning inputs. Existing proof-index production exits remain open.
