# Native worktree allocation admission for exact source reads

The owner-private [worktree allocation guard](../../ipfs_accelerate_py/agent_supervisor/runtime/current_source_worktree_allocation.py)
joins an existing [native task admission](current_source_task_admission.md) to
the actual database-to-Portal execution projection and managed Git worktree.
The host independently admits the canonical repository, allocated workspace,
worktree root, attempt root, merge target and board namespace. The guard checks
those inputs against existing native records instead of treating a filesystem
path, task alias or source hash as allocation ownership.

This adds an allocation prerequisite to the opt-in
[source-worker channel](source_lookup_worker.md). It does not automatically
activate ordinary Terminal Bench tasks, compile model context, replay runtime
bootstrap receipts or authenticate the remote coding-child launch. Existing
LegalIR, SecurityIR, CodebaseIR and IntentIR inventories, embeddings and separate
schema/version/decoder-task/8D/384D/768D checkpoints remain reused unchanged.

## Existing authority and host policy

Capture starts with the exact live task-admission object held by the original
Quack owner. Its current native task, claim, daemon peer and active grant still
have to pass. A detached snapshot, copied path, reconstructed daemon name or
caller-provided body CID cannot replace that object.

The host supplies its admitted repository and allocation policy. This guard
does not discover a private store locator, open another database owner,
construct a competing coordinator or allocate a new worktree. It reads the
existing sealed `DatabasePortalExecutionBridge` projection and the existing
`WorktreeLifecycleStore` records. Their native identity and filesystem controls
remain prerequisites, rather than authority created by a successful source
read.

The first profile requires a canonical checkout with an actual `.git`
directory, plus a distinct registered linked workspace. Bare repositories and a
canonical repository represented by another linked worktree are unsupported.
The caller preprovisions canonical nonsymlink, owner-controlled authority
directories and single-link regular control files, including the registered
Git control directory and files. Group/world-writable authority paths refuse.
The guard checks these permissions and retained physical identities; it does
not chmod or repair them. Read facades reuse the task admission's original
connection with schema installation disabled, and lifecycle reads use an
explicit existing store directory. They create no connection or storage owner.

```python
from ipfs_accelerate_py.agent_supervisor.runtime.current_source_worktree_allocation import (
    capture_native_source_worktree_allocation,
)

allocation = capture_native_source_worktree_allocation(
    task_admission=existing_native_task_admission,
    canonical_repository=host_admitted_canonical_checkout,
    workspace=host_admitted_allocated_worktree,
    worktree_root=host_admitted_worktree_root,
    attempt_root=host_admitted_database_portal_attempt_root,
    merge_target_branch=host_admitted_merge_target,
    board_namespace=host_admitted_board_namespace,
)
observation = allocation.require_current()
```

## Native projection and lifecycle joins

The sealed attempt projection must name the same native task alias, task CID,
attempt, claim and execution ordinal as the retained task admission. Its
repository, attempt location and namespace must reproduce the host's admitted
scope. The task projection bytes, binding and directory controls are checked
through the existing bridge validation rules.

The allocated workspace must be the actual managed linked worktree in the
admitted Git repository, under the admitted worktree root. Its Git worktree
marker, branch and shared repository identity are checked with the native
lifecycle record and task/attempt index. The record must join the expected
lane, state directory, merge target and live daemon process birth. An unrelated
clean checkout containing identical bytes does not establish that join.

The identities have distinct meanings:

| Identity | Meaning |
| --- | --- |
| Native task CID, alias and full body CID | Actual current database task and exact task-admission snapshot |
| Portal canonical task CID | Canonical identity derived by the native Portal projection; it need not equal the native database task CID |
| Database attempt ordinal and claim tuple | Native running attempt, owner session, claim lease and database fence |
| Lifecycle task/attempt identity | Managed worktree ownership identity, including the native database attempt encoding |
| Lifecycle lease, fence and record ID | Separate worktree ownership record; not an alias for the database claim lease or fencing token |
| Git commit, tree, common directory and worktree marker | Committed source identity and actual linked-worktree registration |
| Context CID | Independently admitted source context; allocation ownership alone does not derive it |

The guard must preserve these namespace distinctions. A planning-contract CID,
historical source-inventory task label or matching source tree cannot replace
the actual native task and lifecycle joins.

Capture pins the complete projection, binding, Git marker and registered Git
control bytes together with their physical file identities. This includes
projection status bytes that the bridge itself can update; this first profile
requires fresh admission after such a change. Replacing one of these controls
with identical bytes still refuses. The lifecycle record's
ownership fields, lease ID and record ID remain fixed. Its fence, updated time
and expiry may advance monotonically for the same owned live lease; rollback,
ownership changes, inactive state and expiry refuse. Renewal replaces the
native lifecycle and index files individually, so their physical file identities are
checked within each observation rather than pinned forever. A concurrent
renewal can make an observation incoherent and require a retry. This is not an
atomic filesystem lease or permission to ignore a failed check.

The detached observation reports `allocation_ownership_verified=True` for this
specific native join. `bootstrap_verified`, `compiled_context_verified`,
`coding_process_verified` and proof/execution/completion authority remain false.
It cannot be serialized into a usable allocation handle. The task admission's
own observation still reports allocation verification false; its contract is
unchanged.

## Source binding and broker use

The allocation guard binds source through its retained task-admission object,
using the admitted allocated workspace. The caller still supplies the existing
CAS, retained source inventory, committed repository request, context, public
file scope and response bounds. Native exact-source checks retain their clean
committed allocated source profile. The worker broker independently adds the
canonical checkout's complete source-byte check before exposing its proxy grant.

```python
binding = allocation.bind_current_source_lookup(
    cas=existing_source_cas,
    expected_inventory_cid=host_admitted_source_inventory_cid,
    request=host_admitted_committed_source_request,
    limits=host_admitted_source_limits,
    context_cid=host_admitted_source_context_cid,
    allowed_paths=tuple(sorted(host_admitted_public_raw_path_hex)),
    max_response_bytes=16384,
    max_slice_bytes=4096,
)

from ipfs_accelerate_py.agent_supervisor.runtime.source_lookup_worker_broker import (
    start_source_lookup_worker_broker,
)

session = start_source_lookup_worker_broker(
    server=original_ready_server,
    canonical_repository=host_admitted_canonical_checkout,
    workspace=host_admitted_allocated_worktree,
    boundary=deployed_boundary,
    prompt=exact_original_worker_prompt,
    task_admission=existing_native_task_admission,
    worktree_admission=allocation,
)
# The existing trusted launcher still registers its actual direct Popen child
# with session.note_spawned(child) and closes this session in its finally path.
```

A source reader published through this guard carries its private allocation
marker. The worker broker requires the same exact allocation handle as
`worktree_admission`, together with the retained native `task_admission`.
Omitting the handle or substituting a serialized observation cannot bypass an
allocation-marked reader. Unmarked readers retain their existing optional-guard
behavior.

The existing gateway remains bind-once. If a late check fails after source
publication, the published service and marker remain consumed. Failure does
not silently unbind, replace or rebind that service. The original owner retains
its cleanup responsibility.

The broker repeats task and allocation checks around a source read. Normal
native coordinator bookkeeping retains its existing effects; task-content
observation is not a promise of zero metadata writes. A validated native source
success is charged to its separate backend byte ledger before a later guard
can refuse publication. Refusal therefore does not imply zero cost or absence
of earlier effects. Final source-byte observations follow the closing custody
checks, without a filesystem writer lease or cross-store atomic transaction.

Direct privileged native SDK reads retain their generic task-body/source checks.
This optional broker guard does not silently add coordinator and allocation
fencing to every privileged native operation.

## Bind a pinned historical nomination

The allocation guard can also consume the exact
[historical source-scope nomination](repository_source_scope.md), using its
retained task admission and allocated workspace:

```python
binding = allocation.bind_nominated_current_source_lookup(
    scope=existing_reviewed_source_nomination,
    expected_scope_cid=independently_pinned_nomination_cid,
    cas=existing_source_cas,
    expected_inventory_cid=host_admitted_source_inventory_cid,
    request=host_admitted_committed_source_request,
    limits=host_admitted_source_limits,
    context_cid=host_admitted_source_context_cid,
    max_response_bytes=16384,
    max_slice_bytes=4096,
)
observation = allocation.require_nominated_source_scope()
```

The caller supplies no alternate workspace or path list. The task binder
cold-replays the complete retained inventory, including omitted source, against
the independent inventory, request, limit and scope pins. It derives the exact
selected paths from that replay. Both allocation bind methods share one
whole-operation guard; reentrant or competing binds cannot adopt another
reader. The original task and allocation handles, CAS, reader and binding bytes
remain retained live objects. An observation dictionary cannot replace them.

The observer returns `supervisor-native-nominated-source-worktree-allocation@1`.
It preserves the task nomination's scope, frontier, inventory, source binding,
context and current task identities, adding the actual `allocation_cid` and
`allocation_ownership_verified=True`. The separate task-only observer continues
to report allocation verification false. `bootstrap_verified`,
`compiled_context_verified`, `coding_process_verified` and proof, execution,
completion and omission authority remain false. The historical nomination's
own flags are unchanged.

Before returning, the observer repeats native task and allocation fencing,
checks current projection/lifecycle custody, joins the original reader and
nomination, and checks the complete allocated working-source population after
native callbacks. Lifecycle loss, stale source, changed reader or scope, and
substituted retained objects refuse. These are sequential checks; they do not
exclude other filesystem writers. Publication failures retain an already bound
reader and its original allocation marker. They do not undo the gateway's
bind-once state or release the owner's cleanup responsibility.

Canonical Git registration and control ownership are allocation checks;
canonical working-source bytes are independently checked by the broker. A valid
allocation nomination can therefore still be refused when that broker joins
the canonical and allocated originals. Neither an allocation observation nor a
historical source hash bypasses the broker's separate admission.

The nomination permits at most 64 selected files and 1,000,000 total original
bytes, counted per selected path, including instruction, smoke, profile or other
required support.
The coding broker independently permits at most 32 selected paths, with its
existing descriptor and response bounds. A 33-path nomination can be valid for
the allocation binder while refusing broker construction before a proxy grant.
It is never truncated into a smaller worker scope. Unselected frontier entries
cannot be read through the bound reader; expansion requires a separately admitted
source lifecycle.

The [nominated allocation evidence](evidence/nominated-source-allocation-20261014/README.md)
records this composition separately from the earlier allocation qualification
below. It does not imply an authenticated coding-child launch, automatic signed
Terminal scope adoption, reconstruction accuracy or token savings.

## Remaining production handoff and qualification

Allocation ownership is separate from an `AdmittedBenchmarkRuntime` bootstrap
receipt, a compiled worker context and the actual remote coding process birth.
The normal Terminal daemon allocates and launches coding in a different process
from the original Quack owner. Its authenticated allocation/context/actual-child
handoff still needs qualification. Registering the broader daemon as the source
worker root would widen access to unrelated descendants and is not an accepted
substitute for the existing direct coding-child `Popen` rule.

The retained-CAS/source-inventory supplier also remains an explicit production
prerequisite. Ordinary single-lane daemon execution does not automatically
provide the process-serialized coordinator required by task admission. These
integration gaps remain visible rather than being replaced by a new owner,
synthetic allocation receipt or implicit path lookup.

The semantic [working-source view reader](worker_source_views.md) and
[candidate context reader](worker_candidate_context.md) retain their separate
unverified candidate refresh semantics. They do not grant native allocation,
claim, source-rebinding or formal-proof authority. Accepted publication, mapped
edits, native source restart and working-overlay admission still require their
own qualified lifecycles.

All **118 distinct host tests passed**, including 31 new allocation cases, five
native lifecycle controls and 82 retained task/source/typed-owner cases, with
zero failures, errors or skips. All 37 direct source/test bindings stayed fixed;
normal pytest plugins and fresh collection/execution AST seals remained enabled.
The [qualification evidence](evidence/current-source-worktree-admission-20261012/README.md)
retains exact commands, hashes, logs, JUnit and independent review. This does not
claim a separate-worker-UID exercise of the new guard, a live model call,
official task reward, formal proof or total token savings. Required source facts remain inline until the actual
production handoff and matched complete-session measurements qualify another
policy.
