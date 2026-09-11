# Candidate lifecycle deletion journal

This native component records the actual removal of one exact terminal lifecycle
record and task index. It is intended for future protected candidate handoffs.
It is not yet wired into the Portal producer or Bridge recovery path, and does
not itself settle callbacks, release task claims, or accept candidates.

## Native APIs

`WorktreeLifecycleStore.delete_candidate_observed(expected,
handoff_receipt_id=...)` admits the exact terminal record and index before any
move. `resume_candidate_observed_delete` requires its already prepared journal;
it cannot create an operation from an absent record. `observe_candidate_deletion`
is noncreating and read-only, including its shared advisory locks. It returns
only a frozen committed observation or no receipt; malformed or foreign evidence
raises `CandidateDeletionUnverified`.

The inputs bind the exact task, attempt, lease, fence, owner birth, repository,
store, and immutable handoff receipt ID. The component does not authorize a caller
to invoke these APIs; the future integration must retain native attempt and
callback admission before calling the mutating methods.

## Persistence and crash recovery

The existing owned native store can remain mode `0775`. A complete owned `0700`
child directory is published without replacement, containing an immutable scope
record bound to the repository, store, and child directory identities. Existing
unsafe, foreign, or unbound children are rejected. The observer never provisions
this directory. Existing directory permissions are not changed.

The operation holds the existing task-index then workspace advisory guards.
Original files must be exact owned regular inodes with matching bounded payloads;
both file descriptors remain held. The prepared receipt records their original
identities and byte digests and is fsynced before any move. The kernel's existing
`renameat2(RENAME_NOREPLACE)` primitive moves each original inode from the pinned
canonical directory into the pinned private journal. Both directories are fsynced.
A committed immutable receipt is published only after both moves are observed.
The publisher retains and returns its exact inode identity, which is reobserved
before success. A mutating retry of an existing committed receipt reasserts the
receipt file and both directory durability dependencies; the observer stays
strictly read-only.

Rename legitimately changes inode ctime. Before commit, the retained inode must
match the original device, inode, owner, mode, size, mtime, link count, and exact
bytes. The committed receipt captures full post-move metadata including ctime;
later observations require that exact state.

A crash after intent can resume the exact original moves. After the first move,
its retained original inode and the exact remaining original authorize only that
remaining move. After both moves, their retained inodes can reconstruct the same
committed receipt. A retry reasserts the durable intent dependencies before any
remaining move. Canonical absence alone never supplies proof. Missing retained
inodes, changed bindings, foreign canonical replacements, or publication
collisions deny recovery without overwriting the foreign state.

The legacy `compare_and_delete` and the strict observation API introduced with the
candidate closure path keep their existing behavior. Existing absent rows and
older unknown callbacks are not migrated or retrospectively certified.

## Qualification and remaining work

Tests use real native lifecycle records and independent child processes that exit
after private-directory publication, prepared publication, each inode move, and
committed publication. Other cases exercise fsync failures, replay, foreign
inodes and bindings, directory replacement, no-replace collisions, and a strictly
noncreating observer. Existing lifecycle and callback tests remain green.

Native actors must honor the existing advisory locks. The component detects
namespace drift and never intentionally replaces a collision; it does not claim
atomic compare-and-rename against an arbitrary writer bypassing those locks.
The Linux no-replace primitive and a same-filesystem journal are required; there
is no weaker fallback.

Retained original inodes and immutable receipts have no automatic garbage
collection. Interrupted unpublished private temporaries are retained and never
adopted. A separate retention policy will be needed for sustained use.

The next integration must bind true pre-delete candidate preservation and actual
worktree removal evidence to this native receipt. Bridge recovery must use a
distinct verified recovery branch, not manufacture historical finish events.
Pooled callback custody remains a separate extension. This source-only component
has made no live owner, source, task, callback, or accepted-root changes.
