# Protected candidate rejection closure

A rejected candidate can be retryable while its outer database provider callback
still has an unknown outcome. Diagnostic codes, a failed attempt, and a stopped
process do not settle that callback. This path recognizes a separate, fully
observed terminal rejection before allowing ordinary retry disposition.

## Initial supported path

The producer supports one non-pooled ephemeral workspace, one local Portal
attempt, and a protected Codex fallback that returned zero before proposal
validation rejected its candidate (`proposal_gate_failed`). There must be retry
budget remaining. The candidate must be committed to its exact rescue branch;
there must be no queue publication, merge, or task completion.

The verifier joins all of these observations:

1. The original signed route and invocation, exact provider outcome, and current
   native provider-attempt CAS at `terminal` with `completion_committed` cleanup.
   The cleanup chain must contain actual fenced Docker absence, not plain PGID
   absence, a timeout, or a mutable runner flag.
2. The exact attempt projection and hash-verified Portal event chain: start,
   proposal rejection, lifecycle terminal observation, lifecycle deletion
   observation, preserved candidate, and finish.
3. The actual lifecycle transition from the captured prior record to its exact
   terminal successor, followed by observed deletion of that terminal record and
   its matching task index under the existing native locks.
4. The current rescue ref/commit, unchanged task contract, repository binding,
   database attempt/claim/lease/session/fences, and original callback fingerprint.

`observe_agent_implementation_terminal_cleanup` is a read-only historical API.
It opens only an existing exact private CAS store and repeats the same terminal
observation after signature verification. It permits the exact signed workspace
to have been disposed, while rejecting existing symlink components and requiring
the canonical repository to exist. Its frozen JSON observation contains historical
bindings for comparison, not route/decision/capability objects. All four ordinary
public effect APIs retain their strict path checks and signatures; the observation
does not authorize another launch.

The outer daemon accepts closure only through its actual bound Portal bridge.
It compares and swaps the exact original unknown callback into the distinct
`database-provider-callback-candidate-rejected@1` receipt. Neither that receipt nor
the Portal closure sets completion authority. The daemon records its ordinary
FAILED/retry evidence and canonical retry receipt, then releases only its exact,
still-live accepted claim. Existing deferral, validation-retry, and consumed-attempt
schemas are unchanged.

## Recovery boundaries

A replay revalidates the same provider CAS and Portal/lifecycle evidence. Crashes
after finish, callback CAS, FAILED phase, retry CAS, or claim release must not run
the provider again. The ordinary terminal-retry reconciler can finish an interrupted
closed-callback disposition. A task/contract/claim/fingerprint mismatch denies the
update; it does not repair or rebind the old evidence.

An expired claim is not directly released by this path. In the disposable typed
qualification, after real lease expiry the existing coordinator expiration path
marked the old claim expired and admitted the natural successor with a higher
fence. The callback receipt remained unchanged and the provider ran once.

The lifecycle store's legacy `compare_and_delete` remains idempotent. The new
`compare_and_delete_observed` requires present, exact owned regular records,
successful removal of their held inodes, and directory fsync. A missing row,
foreign record/index, lost unlink, or failed persistence cannot produce the
post-delete event.

There is still an intentional unknown-outcome window: deletion can succeed and
the process can die before publishing the post-delete event. The current lifecycle
store has no immutable deletion tombstone from which to reconstruct that event.
An absent row is therefore insufficient; the callback stays unknown. Recovery of
that window would require a separately reviewed durable deletion journal, not an
inference or retroactive receipt. `CandidateClosureObservationUnknown` also keeps
Portal task, implementation and resource claims, the selected dispatch intent,
and unfinished state out of generic exception/finally cleanup.

## Qualification and limits

Tests use actual disposable signed native route/CAS storage, typed owner grants,
reservation/admission, task control, Git preservation, lifecycle transitions, and
Portal event verification. Docker removal/absence observations and the accepted
source capsule are explicit test doubles. The TCP Quack transport is a test double;
the authenticated typed gateway is real. These tests are not a live protected
container or current-root source admission certificate.

The public Portal caller test supplies five disposable typed analytical receipts
to the real residual gate and uses a sealed fixture packet bound to the actual Git
tree and task. The existing invocation and lifecycle checks remain active; the
process runner is a test double. This is fixture receipt authority, not live native
admission. It reaches the real missing-cleanup producer guard and finalizers, and
verifies that ordinary exceptions finish while the distinct unknown handoff
retains custody.

Pooled workspace/native callback settlement is a separate required extension.
Plain provider routes, retained historical unknown callbacks without the complete
new evidence, final-budget attempts, and incomplete candidate/queue disposition
remain outside this path. No live board, owner, callback, source pin, accepted
root, or existing receipt is changed by staging this implementation.
