# Compact successor artifact contract design

This is a proposed design, with no implementation or qualification credit. It extends the [full 300 source plan](codebase_full300_preview_and_compact_worker_plan.md) by moving complete signed worker inputs into bounded immutable artifacts while retaining the 262,144-byte Intent task-body limit. Native stage outcomes belong in separate qualification reports and their retained receipts.

## Proposed contract versions

Use manifest version 8, preview context version 2, pending contract version 4, planning receipt/reference version 6 and public instruction version 6 as one explicit profile. Existing manifest version 7, context version 1, pending version 3 and receipt/public version 5 retain their exact formats and validation.

The version 8 manifest preserves every original parent declaration except its permitted schema/context changes. Its context references the complete immutable version 6 parent admission and signed companion. Pending version 4 references one complete child admission artifact containing the signed child manifest, graph and signed planning receipt. This avoids independent graph copies that could disagree. Exact task specifications, dependencies, acceptance checks, outputs, owner identity and pending identities remain explicit in pending data and must match full replay.

An artifact reference has exactly `schema`, `artifact_schema`, `codec`, `cid`, `sha256`, `bytes` and `store_id`. `artifact_schema` selects a closed role decoder. Parent admissions have the composite shape `{manifest, graph, receipt}`; their decoder must check the manifest version 6 and receipt version 4 signatures and schemas, rather than require a nonexistent top-level schema.

Use [supervisor canonical JSON and content identity](../../ipfs_accelerate_py/agent_supervisor/proof/formal_verification_contracts.py): canonical UTF-8 bytes without a trailing newline, a CIDv1 DAG-JSON/sha2-256 identity, and exact-byte SHA256 and length. Accept bounded plain builtins under the existing float-free signed-wire rules. Reject duplicate keys, unknown fields, noncanonical bytes, boolean byte counts and alternate codecs.

Proposed initial bounds are 2 KiB per reference, 8 MiB per artifact, 32 MiB per unique closure, 32 unique members and reference-graph depth 8. These are review inputs to check against measured complete-population artifacts, not qualified limits. Decoded objects retain their existing bounded wire validators. Charge aggregate bytes before reads. Resolve references only at registered schema positions; reject cycles, conflicting aliases and role/schema mismatches.

## Trusted storage and offline custody

The owner-storage model is [local receipt loading](../../ipfs_accelerate_py/agent_supervisor/runtime/local_planning_admission.py): canonical private lifecycle directories, owner UID, single-link regular files, bounded bytes, `O_NOFOLLOW`, canonical reload and signature checks. Add a separate owner-controlled artifact store with held directory descriptors, checked file identities before and after reading, atomic publication, enforced file/directory fsync and exact verification of an existing object before deduplication. Published objects are immutable through the store API.

The existing [datasets ImmutableCAS](../../../ipfs_datasets/ipfs_datasets_py/logic/software_contracts/cache.py) offers canonical CID verification and atomic publication. Its generic read API does not supply this worker-specific owner custody or durable pin policy. It must not become a trusted worker resolver merely because its bytes have a CID.

Bind `store_id` to the owner profile and trusted configuration. Derive locations from verified identities; accept no caller-supplied path, URL, ambient store registry or network fallback. An offline worker receives the complete closure in a separate approved, owner-owned, immutable worker-readable mirror. Private signing credentials and private owner directories remain inaccessible. Mirror selection and UID/root checks are explicit; signatures and all decoded identities are checked again locally.

Saved artifacts permit historical replay. They do not carry a native owner capability or reusable freshness token. Missing offline members reject before reservation or START. The [public instruction route](../../ipfs_accelerate_py/agent_supervisor/runtime/router_public_instruction.py) must keep its exact repository/worktree and current source checks while selecting the new profile explicitly.

## Complete replay and the four bindings

Keep these fields equal in the manifest declaration, planning receipt/reference, pending contract and worker context:

- `parent_admission_cid`
- `signed_bridge_cid`
- `bridge_cid`
- `codebase_preview_successor_context_cid`

The first three retain the original parent and companion identities. The fourth hashes the new reference declaration without including itself. The signed companion continues to reference the immutable parent, avoiding a child/companion identity cycle.

Resolve the complete bounded closure and verify signatures, exact parent equality, graph identity, complete source/task population and every task requirement at materialization, public replay, reservation, construction, START and completion. Reuse the closed parent/bridge checks in the [pure context validator](../../ipfs_accelerate_py/agent_supervisor/runtime/codebase_preview_successor_worker_context.py), but validate the actual version 8 signature and schema separately. Rewriting version 8 into an unsigned synthetic version 7 object is not a valid replay route.

Executing boundaries also require fresh original materials and current source/model, structural, policy, capability, provider, configuration, task-source and checker observations. Preserve the [current worker route](../../ipfs_accelerate_py/agent_supervisor/runtime/codebase_preview_successor_worker_admission.py): application reads and callbacks finish before final native close; detached closure, pin and task bytes survive callbacks; staged writes and closing/context-exit failures roll back the whole Intent population. Generic materialization continues to reject preview worker profiles.

Reject mixed versions, downgrade attempts and inline/reference hybrids. Initial support is fresh installation only. Existing pending tasks remain immutable; migration must be explicit and must not silently reinterpret or update old contracts.

## Durable pins and initial retention policy

Publish and durably pin the entire closure before beginning an Intent transaction. Bind pins to the owner, plan, tasks and lineage. Prepare pin custody outside the writer lock alongside the detached staged population. A failed installation may retain conservative orphan pins; it must never remove objects needed by committed tasks.

The initial store has no garbage collection. A later explicit lifetime milestone must account for installed/pending tasks, live workers and their native cleanup, restart cursors, successor/rollback lineages and audit retention. Crash recovery verifies durable pin records without treating them as execution authority. Offline copies need the same retained dependency lifetime.

## Companion only preparation before worker integration

The [existing native helper](../../benchmarks/agent_supervisor/container_coding/preview_successor_native_profile.py) prepares a preview, signs its companion, then signs and admits a version 7 child. A separately named proposed companion-only preparation API would stop after [author_current_preview_successor_bridge](../../ipfs_accelerate_py/agent_supervisor/runtime/codebase_preview_successor_admission.py). It would preserve genuine typed inputs, the immutable full parent/graph, original capture limits and fresh root/checker observers. Independent receiving would call the existing `verify_current_preview_successor_bridge` as its own bounded stage.

Static control-flow counting finds seven full successor-admission operations after preview in the worker helper: two for companion authoring, two for child manifest authoring and three for child admission verification. Stopping after companion authoring removes five operations. This is a scope/cost hypothesis, not a measured speedup. Companion authoring and independent receiving each retain complete two-pass native verification, existing deadlines and final native closure.

That profile reports parent, signed companion and companion-body identities, complete request/material/preview/graph/population joins and false worker integration. It creates no child context CID or version 7 admission, installs no tasks and grants no START, completion or proof credit. Existing worker preparation behavior remains separate.

## Disjoint rollout and acceptance gates

1. Add proposed `codebase_preview_successor_artifact_contract.py` and `codebase_preview_successor_artifact_store.py` with closed reference, publication, custody and pin controls. Do not change legacy consumers in this stage.
2. Add proposed `codebase_compact_preview_successor_worker_context.py` with full pure replay and pending reconstruction. Test explicit version pairing and rejection before adding runtime paths.
3. Add proposed `codebase_compact_preview_successor_worker_admission.py` with current native receiving and the existing bounded staged Intent transaction. Make only explicit schema switches in local planning, public instructions and execution consumers; preserve legacy validation and STOP/cancellation custody.
4. Qualify a fresh complete 300-source worker with real installation, dispatch, START, authored checks, completion and cleanup. Publish a new native source generation after edits and compare all optimized/opt-out pages and fresh-process continuations. Preview-only evidence cannot supply this credit.

Meaningful controls must cover:

- Missing, truncated and same-size changed artifacts; inode/path replacement, foreign owner/store, symlinks, hardlinks and interrupted publication.
- Canonical wire, role decoders, cycles, closure limits, signature forks, four-CID mismatches and removed/reordered tasks, dependencies, validations, acceptance or outputs.
- Old/new cross-version rejection, unchanged legacy replay, offline missing members, pin crash recovery and conservative rollback retention.
- Genuine late source/policy/checker drift, final native close and context-exit rollback, complete Intent population, restart receiving and STOP/cancellation drain.

Retain separate control and native receipts, source pins, actual resource observations and a standalone finite byte-closure reader. That reader verifies integrity and consistency without importing retained producers or granting execution authority.

The [four-head integration note](codebase_larger_head_native_owner_integration_note.md) retains separate 8D, 384D, 768D and 4096D scopes. Compact storage does not qualify a new source width, latent lineage, task codec, CPU/CUDA profile or trained Leanstral head. Proof-cache obligations discharge only through checker-accepted formal evidence bound to the exact source generation and policy; latents and similarity scores remain planning inputs.
