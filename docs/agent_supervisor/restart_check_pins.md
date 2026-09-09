# Automatic restart checks

`DatabaseTaskSource.restart_check_pin(anchor_heads=..., anchor_cursor=...)`
derives a current task-head check from retained intent events. Call it from the
native restart preflight after resolving the configured database owner. A board
which progressed after its initial launch no longer needs a manually edited
event-cursor constant for each restart.

The original anchor is immutable. Each call reads the event suffix, current
heads, revision rows and completion receipts in one database snapshot, validates
their content and ordering, and returns the current `event_cursor` and `pin_cid`.
No mutable JSON cache is trusted. Reopening the task source derives the same pin
again; checking the pin does not modify task status, retry budgets or receipts.
For Quack, the closed read runs through `quack_query` on the server because an
attached Quack database cannot perform multiple streaming scans in one query.
Transport errors fail closed and do not open a local database as a fallback.

The native caller must bind the returned cursor and task heads to its current
owner, plan, source and launch checks, and recheck for concurrent progress before
using it. The result is an observation, not launch, source-change, completion or
effect-settlement authority. A read replica remains a replica. Existing claim,
effect, accepted-source and final-closeout checks remain required. A source pin
can advance only through that board's native accepted-source transition.

`refresh_restart_check(check)` automatically repeats a native check up to three
times when it raises `RestartCheckPinChanged` because progress raced its reads.
Call this before any token retirement or worker launch. It never retries an
integrity, source or owner failure, and sustained contention remains a bounded
typed failure instead of an unbounded loop.

The verifier accepts retained task status changes, validation events and evidence
events. It rejects missing or duplicate history, changed task identities,
unexplained revisions, malformed completion receipts, and plan/definition or
unknown event kinds. These require native requalification instead of silently
moving a pin. Reads are bounded to 512 tasks and 4096 events after the anchor;
an exceeded bound is a typed failure requiring a newly qualified native anchor.

SAWM's campaign adapter uses this API automatically when its current event cursor
is later than the sealed M70 anchor. The adapter retains the original 342 anchor
and exposes the derived pin in the native preflight report. Other board-specific
restart facades must use the same API where they compare frozen task-head pins;
this API does not override their source or owner-generation policies.
