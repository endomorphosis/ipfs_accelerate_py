# One-attempt native task admission for exact source reads

The owner-private [task admission guard](../../ipfs_accelerate_py/agent_supervisor/runtime/current_source_task_admission.py)
joins an existing process-serialized task claim to the original Quack owner's
current task row and active owner-attested daemon grant. It captures the actual
task alias, revision and full canonical body CID, then can bind the
[native source reader](current_source_lookup.md) to that snapshot. It constructs
no additional database owner, coordinator, CAS, server, worktree or model.

This is one admission component, not automatic Terminal Bench activation. It
does not verify allocation ownership, compile a worker context, authenticate a
remote coding launch or register a model process. Those handoffs remain separate
work. A daemon or broader supervisor process cannot substitute for the actual
coding child required by the existing source-worker direct-`Popen` rule.

The separate [native worktree allocation guard](current_source_worktree_allocation.md)
can additionally join this exact handle to the sealed Portal projection and
managed-worktree lifecycle. Its optional allocation check does not change this
task-only snapshot's `allocation_ownership_verified=False` observation or
complete the remote context/process handoff.

## Host capture from existing native authority

Before capture, the caller must provision existing authority files and parents
as owner-controlled paths: canonical nonsymlink regular files owned by the
owner UID, with one link and no group/other write bits, inside owner-controlled
parents. Ordinary Quack startup does not guarantee those permissions. This
guard checks and retains them; it does not silently chmod or repair the paths.

```python
from ipfs_accelerate_py.agent_supervisor.runtime.current_source_task_admission import (
    capture_native_source_task_admission,
)

# These are already admitted, live host objects and native observations.
admission = capture_native_source_task_admission(
    server=original_ready_server,
    attempt_authority=existing_process_serialized_coordinator,
    task_cid=actual_native_task_cid,
    expected_claim_identity={
        "attempt_id": actual_attempt_id,
        "claim_id": actual_claim_id,
        "attempt_number": actual_attempt_number,
        "lease_id": actual_lease_id,
        "owner_session_id": actual_owner_session_id,
        "fencing_token": actual_fencing_token,
        "fence_epoch": actual_fence_epoch,
    },
    peer_identity=(actual_daemon_pid, actual_owner_uid, actual_process_start_ticks),
    client_id=actual_daemon_client_id,
)
observation = admission.require_current()
```

The factory requires the exact `QuackStateServer` and
`ProcessSerializedDatabaseCoordinator` instances. The supplied coordinator is
an existing authority adapter rather than a new lease database. Its physical
file, configuration and lifecycle are retained; the handle remains private to
the original server process. Snapshot dictionaries are detached observations,
not serialized admission credentials, and cannot construct a handle.

The actual already-open daemon adapter can have
`require_existing_authority=False`; the guard preserves that supplied object
and configuration. Its normal native opening path can repair bookkeeping
metadata before the claim callback. A metadata-only prior corruption may
therefore be repaired while a genuine live claim still passes. This is not
strict read-before-repair schema-integrity verification. That stronger boundary
requires a native coordinator contract extension or a separately qualified
existing-only prerequisite, rather than constructing another adapter here.

The native claim API checks the exact task/attempt/claim/lease/owner/fence tuple,
current running attempt, live lease, latest fence and absence of logical
completion before and after its callback. Native fencing retains its normal
metadata, lease/event and expiry-handling writes; this is not a zero-write
observer. The guard acquires coordinator custody before the original server
lock. It then reads the actual native row and repeats
the read before returning. The original server, gateway, connection, lock,
identity and current store generation/fence/UUID must still match.

The task must be in progress with its native `database_attempt_admitted`
receipt, claimed execution phase and first execution revision. Its receipt's
owner-derived process attestation must reproduce the current active grant and
the actual live daemon PID/UID/start tuple. Expired, revoked or foreign grants,
dead/reused peers, changed claim identity, completed claims, stale fences and
changed native task body/revision refuse. The client ID must identify the
expected implementation daemon; a name or caller-provided hash does not grant
authority.

The native receipt and active grant establish owner-attested process custody.
They do not independently prove an `AdmittedBenchmarkRuntime` bootstrap receipt,
sealed allocation or compiled context. A production transport must additionally
join those actual runtime objects before making the source helper available.

## Strict task snapshot and source binding

The guard reads the full `body_json` as a bounded canonical JSON object, with
duplicate keys and noncanonical encoding refused. It retains the actual
canonical body bytes and derives `expected_task_body_cid` from the entire body.
It does not strip metadata, normalize an old row, replace the body with its
planning contract, or infer current revision from a semantic index.

The first successful observation pins the task alias, revision, full body and
claim identity. Later `require_current()` calls must reproduce that exact
snapshot. Same-owned live lease/grant renewal can remain valid when the pinned
task body and claim tuple stay unchanged. A new task revision, different phase,
body change or attempt requires explicit new admission, even if the source file
bytes happen to match.

```python
binding = admission.bind_current_source_lookup(
    repository=independently_admitted_allocated_root,
    cas=existing_source_cas,
    expected_inventory_cid=admitted_source_inventory_cid,
    request=admitted_source_inventory_request,
    limits=admitted_source_limits,
    context_cid=independently_admitted_context_cid,
    allowed_paths=tuple(sorted(admitted_public_raw_path_hex)),
    max_response_bytes=16384,
    max_slice_bytes=4096,
)
```

The convenience binder supplies actual task CID/revision/full-body CID from
the captured native row. The host still supplies the existing source CAS,
retained inventory, committed repository request, context, file scope and bounds.
It does not discover an owner/store locator or derive a new private database.
The source reader retains the exact supplied objects and independently verifies
its current committed tree/population and source bytes.

The native gateway remains bind-once. After successful publication, the guard
retains that exact reader, CAS, root and canonical source binding. A later
failure can leave the gateway bound; it does not silently undo, replace or
rebind the published reader. The caller retains its existing owner cleanup
duty. There is no atomic transaction spanning claim custody, CAS/source replay
and gateway publication.

## Optional broker guard and process boundary

```python
from ipfs_accelerate_py.agent_supervisor.runtime.source_lookup_worker_broker import (
    start_source_lookup_worker_broker,
)

session = start_source_lookup_worker_broker(
    server=original_ready_server,
    canonical_repository=independently_admitted_canonical_root,
    workspace=independently_admitted_allocated_root,
    boundary=deployed_boundary,
    prompt=exact_original_worker_prompt,
    timeout_seconds=90,
    task_admission=admission,
)
```

The optional guard is an exact private object reference, joined to the original
server and its already bound source reader. The broker refences its native
claim and source join around reads while preserving separate native success
byte accounting. A validated backend success can remain in that ledger even
when a later claim or currentness check refuses to publish the source result.
Handshake/error/partial transport costs and provider usage remain unknown;
failure is not evidence of zero cost or no prior effect.

A reader bound through this handle carries its private admission marker, so a
broker cannot omit or substitute the guard when selecting that reader. Direct
privileged native SDK reads retain their generic body/source-currentness checks;
they do not automatically refence the coordinator through this optional broker
guard. The snapshot's `task_content_read_only` flag describes the task content,
not a promise of zero bookkeeping writes.

The [source-worker channel](source_lookup_worker.md) still requires the actual
trusted launcher to register its direct coding child after `Popen`, then validates
kernel peer identity and exact descendants. This guard cannot broaden that
registration to a daemon, run root or caller-nominated PID. The normal Terminal
daemon currently allocates the coding worktree and launches the owner wrapper
in a different process from the original Quack owner. An authenticated
allocation/claim/compiled-prompt/birth handoff is still needed at that boundary;
the new handle does not implement it.

## Identity and refresh limits

These identities retain separate meanings:

| Identity | Meaning |
| --- | --- |
| Native task CID, alias, revision and full body CID | Current canonical task row and its exact admitted attempt |
| Source inventory `task_id` | Historical inventory label; not native claim custody |
| Signed planning contract CID | Contract field identity; not the complete current task body |
| Semantic scope/root/repository CIDs | Their actual scoped semantic artifacts and producer namespace |
| Canonical Git commit/tree OIDs and population CID | Native committed repository acquisition |
| Context CID | Independently host-admitted source context; not inferred from an alias or bundle hash |
| Private store ID/path and owner identity | Original storage custody; no model-selectable owner locator |

The [semantic source-view channel](worker_source_views.md) can publish refreshed
working-source views through its own separately admitted, unverified snapshot
semantics. That capability is preserved. The native source channel's first
profile continues to require clean committed canonical and allocated roots.
Edits require fresh native task/source evidence and a separately admitted
binding lifecycle; a matching semantic view or unchanged historical inventory
does not restore native claim or current-source authority.

Checks are sequential. A protected claim callback is not an atomic filesystem
writer lease, source semantics proof, omission permission or whole-task
completion. The observation explicitly leaves `allocation_ownership_verified`
false and grants no proof, execution or completion authority. Required source
facts stay inline until the actual production handoff and token/reward
experiments qualify a different policy.

The [host qualification](evidence/current-source-task-admission-20261012/README.md)
passed all 82 distinct cases, including 17 new actual-claim controls and the
existing reader, worker-channel and typed-owner regressions. Collection and
execution used normal pytest plugins and separate AST seals; all 29 recorded
source bindings stayed unchanged. The original permission-related fixture
failure is retained alongside the corrected run. The tests use actual native
materialization, typed admission, grants and process-serialized claims; Quack
listener lifecycle and worker deployment metadata are authored seams. They do
not exercise this guard through a separate coding-worker UID or qualify a live
model, official task reward or total token improvement. Existing LegalIR,
SecurityIR, CodebaseIR and IntentIR inventories, embeddings and separate
schema/version/decoder-task/8D/384D/768D checkpoints remain reused unchanged.
