# Cold-owner cleanup custody: bounded design audit

Read-only audit, 2026-10-07. Source: `1f818293ff7561f1c6de49e89ac87806f7812d19`.
No production edits, Docker operations, provider requests, or account registry mutations.
No applicable AGENTS.md was found in this checkout or its relevant ancestors.

## Recommendation

Implement a **future-only, independently admitted cleanup owner checkpoint** before admitting cold-owner success. A new file under the cleanup directory, an empty lifecycle journal, or a self-hashed manifest cannot establish that no prior effect existed.

Keep current native Docker gates and process-only refusal guards. The ordinary Harbor FullSupervisorAgent already has a task-container deployment path; this native nested-container recovery design is not a prerequisite for exercising that separate admitted benchmark route.

The smallest sound first increment is durable registration plus refusal on missing/replaced/restored custody. Positive cold completion additionally requires a bounded inventory of every announced binding, joined to current producer completion and provider CAS evidence. Do not implement positive recovery by reopening a directory and constructing a fresh warm observer alone.

## What existing stores do and do not prove

| Existing mechanism | Reuse | Required limitation |
|---|---|---|
| `control/profile_authority.py` account root registry and external monotonic lifecycle anchor | Existing private directory/file readers, retained metadata validation, create-once account root identity, exact registration, independently pinned root DID, atomic persistence, and target locking patterns. | New cleanup-domain record/API; do not overload the exact closed profile-anchor schema or change an existing legal/profile identity. The anchor must be outside repository, state/run roots, task workspace, and their backup/rollback domain. No root signing key is passed to providers/watchdogs. Current `_registry_root()` creates its directory even when a later root-key load uses `create=False`; cold observation therefore requires a strict non-creating root opener rather than calling that helper unchanged. Existing local profile capabilities explicitly deny destructive cleanup; the new inventory domain grants observation only unless separately admitted. |
| `task_sources/plan_revision_store.py` CAS and guarded continuations | Immutable bounded inventory snapshots, exact predecessor chain, current-pointer CAS, and durable records. | Its constructor creates missing directories/index and normally performs recovery. It is unsuitable as a read-only cold loader. Use an explicit open-existing authority view or existing secure readers after validating the independently expected store identity; missing required pointer/CAS is UNKNOWN. |
| `control/plan_execution_store.py` secure readers and process-birth/execution-lease chains | Existing exact schema/hash/path validation, task/slice/owner fence joins, and bounded generation patterns. | Current birth schema lacks cleanup registration and inventory head. A missing birth continuation currently returns `None`; never infer a new owner from that in the future contract. Version new launch/admission/birth evidence explicitly. |
| `control/provider_attempt_store.py` DurableProviderAttemptCAS | Existing `create_if_missing=False`, expected directory identity, exact observation, terminal cleanup authority and monotonic completion progress. Store each scoped binding's locator in the inventory. | It covers one known logical attempt; no API proves the complete set of attempts. Directory enumeration or missing reservation is not an owner inventory. |
| `control/lifecycle_orchestrator.py` LifecycleSagaStore | Existing target serialization and saga durability for recording cleanup UNKNOWN/receipt references. | `history()` returns empty for a missing file; reader is not an independent, immutable, rollback-resistant expected owner head. Do not reinterpret its absence as an empty inventory. |
| Managed PID projection and `.terminal.json` active/terminal receipts | Existing exact inode and birth projection checks; add references in a future receipt version. | Current active binding is explicitly non-authoritative; missing/archived artifacts and bare-PID death cannot certify Docker cleanup. |

The independent expected checkpoint must come from current admission outside the mutable task state. It cannot be supplied only by the repository under test, restored local journal, or a newly discovered registry file. Define the rollback domain precisely: losing or rolling back **all** independent authority is UNKNOWN, not proof of a fresh machine. A valid old signature or hash alone does not establish freshness.

## Minimal future contract

Use a distinct typed owner receipt, for example `managed-cleanup-owner@1`, with:

- admitted external registry/root identity and exact target registration key;
- canonical repository, state/run/cleanup paths, profile/run/configuration identities, and fencing epoch;
- exact cleanup directory device/inode/mode/uid and inventory-store directory identity;
- unique birth generation and nonce, immutable parent/managed process birth once captured;
- admitted source/launch/profile references, current owner generation, and prior closed-owner receipt;
- bounded inventory head CID, sequence, previous head CID, and `prepared`, `active`, `stopping`, or `closed` state.

The immutable inventory snapshot contains at most the existing 128 binding limit. Each entry records its reserved stem/path, provider/name, independent provider-CAS locator when scoped, announced binding identity or pending-publication identity, current immutable record/fence identity, and eventual completion/terminal-progress IDs. Retain closed entries until a separately admitted generation rollover; never delete an entry to mean completion. CAS snapshots may retain full validated binding payloads for diagnosis, but a copy does not replace the original retained inode's authority.

Prefer existing plan CAS for snapshot storage when an admitted plan store already exists. For non-plan-bound owners, registration in the external lifecycle authority must remain mandatory; do not manufacture a permissive default store. A shared narrow inventory-store adapter can reuse current durable file/CAS primitives without importing the retained donor's large recovery closure.

The external current-head pointer is updated by the admitted owner under the exact target fence. A trusted owner-side writer may acknowledge provider announcements; providers/watchdogs must not acquire the account root key. A new private control capability or reviewed existing owner gateway is needed for that acknowledgement. This requirement is part of the versioned launch contract, not an ambient environment opt-in.

## Exact current hooks

| Hook | Required future ordering |
|---|---|
| `runtime/multi_supervisor_runner.py::start_track`, immediately after `CleanupDirectoryAnchor.open_before_launch` and before `subprocess.Popen` | Load required external registration without bootstrap-on-missing. Validate prior owner is closed. Commit/fsync a fresh prepared owner and empty inventory head binding the directory inode; round-trip it before birth. Current code near lines 7315–7318 is the insertion seam. |
| Same function, gated birth capture and `_persist_plan_bound_process_birth` before `os.write(...PLAN_BOUND_LAUNCH_GATE_SUCCESS)` | Bind the actual managed PID/start ticks/boot identity and the cleanup owner receipt/head to a future typed birth/lease record. Commit before releasing the gate. Ordinary ungated births need an equivalent pre-release gate before supporting this positive contract. |
| `_activate_run_generation_binding` | Future active receipt references the independently admitted cleanup owner generation and every selected track registration. Existing bare-PID stale-active acceptance must not replace an unresolved cleanup-bearing predecessor. |
| `runtime/grok_cli_runner.py::_DockerContainerLease.create` before native watchdog launch | Announce/reserve exact intended binding stem/provider/name and owned resource scope in the external inventory and obtain a durable owner acknowledgement. No factory enablement until this is connected. |
| `_docker_cleanup_watchdog_main` prepared binding publication (current lines 8701–8741), before the readiness payload | Join actual watchdog birth and prepared binding bytes/inode to the announced entry, persist the inventory update, then acknowledge readiness. A crash between announcement and file publication remains a pending entry, never an empty set. |
| `publish_command_bound_cleanup_authority` and `_DockerContainerLease._publish_cleanup_binding` | Update the same entry monotonically before command-bound release; bind create command/environment identities. Preserve the exact private channel and binding lock order. |
| `create_inert_container`, before create journal transitions to `create_armed` and before private socket `D` | Require the durable inventory acknowledgement covering that command-bound entry. This is the last no-effect boundary. |
| `_publish_docker_termination_fence_binding`, before releasing provider start | Persist the exact running CID/image/kernel fence reference in the inventory; provider start remains gated until both local record and independent inventory commit. |
| `_finalize_verified_cleanup_completion` after terminal CAS completion and authority/resource retirement | Owner may append a closed entry only after the existing observer verifies exact completed evidence and fresh Docker/kernel absence. Do not let a watchdog's success code close the inventory. |
| `_terminate_managed_process`, `stop_tracks` without Popen, strict plan-bound fence observer, LifecycleOrchestrator `_prove_absent`/STOP/RESTART/repair | Resolve required external owner generation first. Reopen only the exact persisted namespace identity, replay the complete expected inventory, and retain UNKNOWN on any missing entry. All paths consume the same typed receipt; none treats local absence as bootstrap. |

Lock ordering must be designed before mutation: owner inventory/target fence first; producer per-binding locks must not call back into an owner that is waiting while holding that same binding lock. Prefer prepare/ack transitions across immutable records with no lock held during RPC, followed by exact revalidation. Interrupted cross-store publication is retained as a pending inventory entry; it is not rolled back by removing evidence.

## Cold observation sequence

1. Verify current admission, independently expected registration/root/store identity, exact owner generation and current inventory head. Open existing only. Missing/replaced/rolled-back authority refuses recovery.
2. Revalidate full inventory chain within its bound and the original lifecycle/task/attempt ownership. Determine old owner/process births with conservative `/proc` observation. Cold observation alone grants no signal/removal authority.
3. Reopen cleanup directory only if its current inode equals the persisted expected one. Build the observer from the expected inventory, not from whichever entries happen to exist. Missing directory/entry/completion is UNKNOWN even if all PIDs and Docker names are absent.
4. For completed entries, reuse current exact retained authority, resource retirement, terminal CAS, Docker CID/name, and kernel-scope checks. Old or foreign boots/fences do not become current signal authority.
5. Under the same admitted fence, record a typed closed-owner receipt binding the final inventory head and fresh absence observation. Revalidate after final process absence as in the current warm STOP. Only then may a successor generation be admitted.
6. Mutating dead-watchdog replay, reconstructed missing authority, and unknown-create recovery remain separate explicitly admitted repair work. Do not infer them from this observation contract.

## Required qualification before positive recovery

- Owner killed before/after registration, Popen, birth commit, and gate release.
- Announced binding before file publication; prepared publication before readiness; command registration before create; fence commit before provider start.
- New owner with namespace removed, same-path replacement, missing one binding, deleted entire inventory, stale valid snapshot, stale pointer, or rolled-back local state.
- Positive completed records using exact original namespace/inodes and terminal CAS; mismatched task/attempt/profile/fence/store/boot identities refuse.
- Two recovery owners racing the current generation; only one admitted transition, no provider replay or extra removal dispatch.
- Every cold/process-only caller rejects missing required owner checkpoint, including successor birth, no-Popen STOP, lifecycle repair, and cached terminal replay.
- Actual signed admission test and crash-point filesystem persistence tests; no test may create a missing authority namespace while observing it.

## Retained donor conclusion

`fleet-supervisor-maintenance-20260910` at `f7194e65775f000c8d8924e700a54b4067d83298` has parent-held cleanup directory anchors and extensive exact record/Docker cleanup helpers. Its anchor remains a retained descriptor; the inspected owner code does not supply this independent persisted namespace inventory or future admission checkpoint. It is an algorithm reference, not a drop-in cold-owner recovery implementation.
