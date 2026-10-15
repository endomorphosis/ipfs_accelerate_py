# Tracked artifact store and expected pin inventory qualification

**All 51 selected checks passed; the candidate remains unapplied.** The owned job ran 44 synthetic controls and seven probes of retained scan artifacts. This advances the [recorded successor context milestone](codebase_compact_successor_input_context_qualification_v1.md) with a private metadata codec, context-aware durable pins and a separate expected-pin journal. The journal lets recovery detect missing pins even when the remaining store files are individually valid.

The [owned receipt](../../../../artifacts/codebase_ir_terminal_bench/compact-artifact-tracked-owned-checks-20261007-01/receipt.json) is 62,276 bytes, SHA256 `7f9892f94fb95723f9a1fd2f281d56ef9f84595997e21f27acc4e9fc1cb9f9bf`. [Child observations](../../../../artifacts/codebase_ir_terminal_bench/compact-artifact-tracked-owned-checks-20261007-01/checks.json) are 35,863 bytes, SHA256 `f60c0ac5a23c3311daae022e3cf1b0087295f0ea404b822047feaf56a574cda5`. Owned elapsed time was 13.914 seconds, including a 13,806 ms bounded child. These durations describe custody and refusal controls; they establish no inference speedup.

## Private metadata and context aware pins

The [codec](../../../../artifacts/codebase_ir_terminal_bench/compact-artifact-tracked-store-candidate-20261007-01/codec.py) preserves the five existing roles and adds exactly three: legacy DAG-JSON completion@1, successor selection@1 and source delta@1. Existing five-role closure summary bytes remain unchanged. No signed companion or V8 child-admission role is registered.

Additional metadata requires the complete caller-bound recorded context, its CID and independent public owner/profile values. The new [store](../../../../artifacts/codebase_ir_terminal_bench/compact-artifact-tracked-store-candidate-20261007-01/store.py) uses configuration and pin schema version 2, persists that closed context header and replays it during metadata-pin recovery. It refuses missing or partial context and coherently rehashed context drift. The pin carries a compact summary; the full graph and task projection remain in the supplied context checker. Pure recorded joins grant no signature, currentness, planning replay or execution authority.

The copied reference and context contracts retain their original bytes. The [sealed README](../../../../artifacts/codebase_ir_terminal_bench/compact-artifact-tracked-store-candidate-20261007-01/README.md) records the pre-run design and its source-review status; the actual passing evidence is the receipt above. Captured helper functions still depend on the supplied Python modules' globals, whose custody belongs to the owning harness.

## Independent expected pin journal

The [journal](../../../../artifacts/codebase_ir_terminal_bench/compact-artifact-tracked-store-candidate-20261007-01/ledger.py) records an ordered, immutable chain of expected pin CIDs under a separate owned root. Each canonical DAG-JSON entry binds the sequence, previous-entry CID, pin CID, journal identity and store identity. A caller-supplied anchor binds the expected sequence and entry CID. Recovery verifies the complete bounded chain and requires every expected durable store pin.

`TrackedStore` first publishes and durably verifies the store pin, then appends and durably verifies its journal entry, and returns the new anchor. The caller must persist that anchor independently before a future Intent installation. The API always reports `external_anchor_persisted=False`; only the caller can establish that separate persistence. Bare journal append does not verify a store pin's existence; the integrated tracked store performs that join.

The two roots do not form an atomic commit. A failure after pin publication can leave a retained untracked pin. An entry ahead of the caller's saved anchor is an undeclared tail and refuses recovery. There is no automatic tail adoption, repair, unpinning or garbage collection. Rolling back both the journal and its independently trusted anchor is outside the claim, as are hardware crash attestation and protection from arbitrary same-user mutation.

Journal creation accepts a genesis anchor only for an empty root. Exact types, canonical disjoint roots, current owner, store identity, private-directory custody, held descriptor identities and process/thread ownership are checked. Each append writes one new entry but replays the bounded chain, capped at 64 unique entries; this is not constant-time verification.

## Quotas and duplicate publication

The blob namespace accounts for all retained names, including unpinned blobs and temporary files: at most 64 MiB of logical bytes and 512 names. New publication precharges the two-name, two-payload link/unlink window. An existing CID is read and fsynced without allocating another temporary copy. Quota observations use held file metadata before blob-body reads. These are application bounds, not a kernel disk quota.

Closures remain limited to 32 unique members and 32 MiB; parent, root and RAW body ceilings remain 8, 4 and 16 MiB. Durable pins are limited to 256 KiB, with at most 64 final pins and 64 MiB of repeated recovery-closure reads. The 44 [synthetic controls](../../../../artifacts/codebase_ir_terminal_bench/compact-artifact-tracked-store-candidate-20261007-01/controls.py) exercise quota and journal bounds, duplicate publication, actual sparse logical-byte quota refusal, preflight before callbacks, detached configuration/anchor inputs, substituted handles, context recovery, deleted/forked journal entries and a failed entry fsync. Compatible complete metadata pin/recovery positives use synthetic fixtures with unverified signature bytes.

## Seven retained artifact probes

The [sealed plan](../../../../artifacts/codebase_ir_terminal_bench/compact-artifact-tracked-store-candidate-20261007-01/plan.json) pins 43 ordinary inputs. The real probes published, read and reopened 30 unique retained inner payloads totaling 3,120,642 bytes. Five extra legacy/root/selection/delta metadata objects remained unpinned and still counted toward the blob quota.

Exactly two durable pins and two journal entries were recovered using a separately saved and reloaded sequence-2 anchor. The original five-role default and reference closures each contain 13 members, totaling 1,494,217 and 1,494,240 bytes respectively. Both were fully retained and verified. The caller fsynced the anchor file and its parent directory outside both store roots. The temporary anchor and store workspace were removed during cleanup; their receipt observations are historical evidence, not current file pins.

| Probe | Actual result |
| --- | --- |
| Publish and read retained metadata | Exact role bytes and CID identities preserved across 30 payloads |
| Default full metadata pin | Exact `lineage_binding` refusal; no pin publication or journal advancement |
| Reference full metadata pin | Same exact refusal and no advancement |
| Independently anchored reopen | Both complete 13-member pins recovered; all 30 blobs counted |
| Delete an expected store pin | `expected durable pin is missing`; external anchor preserved |
| Delete an anchored journal entry | `missing_anchored_entry`; store pins retained |
| Reopen with a stale anchor | `undeclared_tail`; no automatic adoption |

The full default/reference metadata closures each supplied 17 members, with 2,243,409 and 2,243,433 bytes and a 7,484-byte context. The historical parent and current scan declare different source heads and model variants, so both required the exact reason `immutable parent/current selection lineage differs`. The real artifacts therefore establish a correct lineage refusal, not a compatible complete metadata-pin positive. Fixture-only restoration of exact deleted bytes enabled subsequent negative restart checks; it does not implement production repair.

## Owned run and finite integrity closure

The job used unchanged existing MAIN with two CPU slots, 2048 MiB and one child, a 180-second deadline, 120-second lease TTL and 30-second automatic heartbeat. All selected source/output bytes closed under the live reservation. The child was reaped, readers/stdin joined, its workspace removed, the owning reservation released and MAIN left idle. Independent read-only review found no concrete gaps in the exact outcomes, input/output identities, artifact framing, anchor joins or lifecycle accounting.

Six observed work-audit categories were zero: project-target imports, model-file reads, database reads, network calls, subprocess calls and unselected repository pathname reads. Descriptor-relative provenance is outside that audit inference. The audited phase ends before separately framed output publication. Inference/training zeros follow from the selected source and absence of backend calls. The child loaded no captured native project implementation; the parent used the captured scheduler/process boundary for reservation and cleanup.

The [final finite integrity closure](../../../../artifacts/codebase_ir_terminal_bench/tracked-store-final-close-20261007-01.json) records two bounded sequential observations of retained evidence and this milestone's ordinary inputs, including the report's local links. It excludes the removed temporary anchor path and adds no native/model/target/signature replay or continuous currentness.

## Remaining integration gates

Next qualify persistent worker mirror custody and the complete signed companion and V8 child role decoders. Then establish actual V8 signatures and full child/parent/companion planning replay, an explicit native shared receiving branch with fresh source/model/checker/root-policy observations, and one-use close and scope exit. Staged Intent installation and rollback, reservation, worker construction, START, authored checks, completion and STOP remain required before enabling the new profile.

No live package code changed in this milestone. Signed worker admission/dispatch, formal proof and production activation remain pending. Structural 53-feature/latent-8 evidence is separate from trained semantic 8D, 384D, 768D, 4096D Leanstral and CUDA qualification. Existing repository-proof-index production exits remain open.
