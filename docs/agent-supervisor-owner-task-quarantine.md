# Native retained-task quarantine

The daemon can preserve two exactly diagnosed unresolved attempts while admitting
independent tasks after a separate complete native reconciliation audit:

* a supersession reconciliation saga at its commit barrier whose original native
  failed phase records `terminalized_for_retry`;
* a running Portal attempt with a durable entered binding, an unknown started
  provider callback, no returned provider/effect receipt, and an absent nested
  state file.

Missing state alone is never admission evidence. The native execution owner
captures all retained task execution rows, coordinator task/attempt/lease rows,
canonical task revision, native diagnosis evidence, and workspace custody.
The legacy finalizer fixture is the exact original producer from
`5abf57b9f6a3c998784401b9f191b1e955adc218`; its test reproduces the terminal conflict
without inserting a synthetic failed phase or receipt.

## Authority and custody

The current authenticated Quack owner appends the closed
`owner_task_quarantine@1` mutation: one immutable domain event and one metadata
history commitment in the same transaction. Every ordinary owner task mutation,
including replay and a different lane's request, validates the entire bounded
quarantine population before its existing CAS/validation/queue logic. The event
head binds the owner generation/schema revision and canonical task revision.
Missing, malformed, altered, truncated, or over-bound history blocks admission.

The exclusive local execution owner must then acknowledge the same native bytes
and install a coordinator commit fence. Retained task, attempt, callback, phase,
saga, lease, resource and completion rows keep their existing values. Expiry does
not release their leases, and retained resources remain occupied. Local process
birth, live Quack owner identity, exact snapshot and workspace custody are checked
again before independent claims or provider/effect dispatch. Read-only checks may
run while a previously admitted independent provider is active; installation
requires an idle execution owner.

The exact missing-state attempt may have no durable workspace association. The
protocol does not pick a historical workspace. Instead, it freezes the entire
explicitly configured old workspace root, including all existing pool rows and
matching lifecycle rows. Existing shared task/resource claims for the exact board
checkout join that snapshot. Their native stale-owner paths cannot remove them,
and the existing path-overlap guard continues treating them as occupied.
Independent work uses a recorded fresh sibling root. Fresh task/resource claims
are allowed only through the normal claim/CAS/DAG/resource guards; the recorded
old claim files remain unchanged.

A repository shared custody lock composes the pool, lifecycle, serialized claim
updates, Portal implementation operation, and cleanup paths. Freeze installation
holds it exclusively. Global Git worktree pruning, reflog expiration, and object GC are deferred
while custody exists, including maintenance from nested submodule repositories. A fresh root never reuses any ambiguous existing pool entry. The archive
of an old board is diagnostic evidence and is not an input to this protocol.

## Admission and limits

A first blocked Quack reconciliation pass may install exactly proved fences. It
still reports blocked/continuation-required. A later complete native audit must
admit the remaining population before any independent claim or callback.
Public running-attempt observation remains complete. Retained attempts are never
reported reconciled, quiesced, restart-safe, completion-authorized, or mergeable.
Revocation removes independent admission and still preserves retained custody;
this protocol deliberately has no automatic custody-release or retry grant.

Bounds: 4096 central events / 1 MiB of event bodies; 8192 rows / 4 MiB for each
retained execution or coordination census; 8192 observed workspace/claim files /
16 MiB, including bytes scanned in other lifecycle/claim rows. Workspace freeze
records are individually bounded at 256 KiB. Exceeding a bound is a visible block.
Fresh root creation, malformed/changed local acknowledgements, owner-generation
changes, and incomplete subsequent audits require revalidation or remain blocked.

This is a native runtime protocol. All owners and filesystem mutation processes
sharing an affected repository must use the supporting runtime before deployment;
an old binary does not acquire the new custody lock. A coordinated stopped-root
upgrade and review of current native evidence are required for the first live
adoption. This patch contains no live-board repair, archive import, task success
conversion, publication, or merge action.
