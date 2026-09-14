# Legacy verification recovery with a quarantined predecessor

An older Portal could reconcile a quarantined merge request, start another
implementation in the same database attempt, then stop on a protected-path
verification lock timeout. The original single-lifecycle retry deliberately
rejects that history. `legacy_quarantined_predecessor` supplies a separate,
bounded reconciliation profile for this sequence.

`inspect_legacy_retry_profile` selects either the unchanged single-lifecycle
profile or the verified predecessor profile. It checks the immutable database
attempt projection, task revision and plan, full event chain, both provider
finishes, exact candidate identity, failed dirty-checkout merge outcomes, and
quarantine reconciliation before the later uncommitted timeout. Extra provider
lifecycles, queued candidates, applied or uncertain merges, and later callbacks
require separate recovery.

`hold_legacy_verification_retry_observation` composes that inspection with the
appropriate native queue guard. For a predecessor, the queue must contain
exactly the recorded task candidate, remain quarantined with no live claim,
and retain matching failure history. The candidate branch and Git tree must
still resolve. It rechecks the projection, event bytes, and queue file identity
before yielding `(evidence, queue_evidence)` while retaining the writer guard.
The existing queue is read in place, without installing a store or resetting
queue rows, claim generations, attempts, event history, or retained workspaces.

An operator must retain native process and source maintenance custody across
the observation and its separately authorized Quack task compare-and-swap.
It must recheck current task and source authority and keep the old candidate
and workspace. These APIs return observation records with `retry_authorized`
and `completion_authority` false; hashes or saved JSON records are not grants
or proof of prior-process closure. A new attempt requires fresh Portal
validation and consumes its own bounded attempt. No retained candidate is
accepted as task completion by this profile.

The profile does not provision a legacy queue's Quack owner or recover two
pending candidates. Native queue migration, supervisor source adoption,
independent callback recovery, and final publication remain separate steps.
