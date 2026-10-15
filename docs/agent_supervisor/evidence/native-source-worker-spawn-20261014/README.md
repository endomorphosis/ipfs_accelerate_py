# Native source broker launch gates, 2026-10-14

This increment carries the original source broker's native task, allocation and
optional historical nomination through before-spawn checking and actual child
registration. It builds on published accelerator
`83ad1ca1a43ae53d3642efe8799d25a9df8cebf3` in an owned worktree. The
[parallel survey](parallel-work-survey.json) preserves the separate integration
of committed capsule-budget and per-turn usage work; foreign trees are unchanged.

## Behavior

The [launch guide](../../source_lookup_worker.md#check-before-launch-and-before-registration)
uses the existing broker and trusted launcher's literal `Popen`. A broker with
native task or allocation handles requires a successful `require_before_spawn()`
before registration. It checks the original endpoints, native task/claim,
allocation and nomination joins, then complete canonical and allocated source.
Checks after CAS/coordinator callbacks preserve the final currentness boundary.
Preparation does not authorize an executable, provider command or arbitrary
child. The trusted launcher retains that policy and its credential discipline.

`note_spawned()` retains the actual live `Popen` child, direct-parent/kernel
birth and allocated working directory. Closing custody and source checks pass
before reads are enabled. Reentry, forged process handles, stale preparation,
substitution and duplicate registration refuse. Legacy sessions without native
handles keep their existing registration route.

Before-spawn refusal must prevent a child launch. Refusal after birth closes the
session, revokes the source proxy and leaves registration disabled. The actual
process remains available for the caller's termination/reaping responsibility.
The broker does not kill the child or undo its earlier effects. Sequential
checks do not exclude filesystem writers. Existing per-read checks, exact source
grants and the independent 32-path broker ceiling remain in force alongside the
nomination's 64-file/1,000,000-total-byte ceiling, including selected supports.

## Qualification

The [runner](qualification_runner.py), [twelve-suite roster](targets.json) and
[listed dependency inventory](source-inventory.json) bind a new complete run.
There is no separate collection or addition of previous results to its count.
The listed scope contains 218 bindings: 167 accelerator and 51 datasets files
and supporting records. Actual datasets execution is
`ebdf6539a4afaf1f10c20744cb14222ef9db4d2b`; kit stays at
`8ac00bb1974b1378e12091d546d8425af34049e0`. The list is not a complete import
closure or toolchain attestation. Runtime databases and state remain private.
The sixth merged run passed 276 distinct tests across twelve suites, including
53 new cases, in 1018.995 seconds, with zero failures, errors or skips. Collection
and JUnit identities agree and all 218 listed bindings stayed unchanged. The
[first attempt](qualification-01/commands.json) was deliberately interrupted
after static review found two middle CAS loads under the owner lock. It collected
272 cases and retained 238 JUnit cases with no failures, errors or skips; it is
not a complete passing run. Its raw evidence and source snapshots remain intact.
The [interruption](qualification-interruption.json) and
[correction record](lock-order-correction.json) keep the exact source hashes and
original input metadata. The corrected middle loads run outside SERVER so a CAS
callback can enter COOR; surrounding owner/source closing checks stay in place.
The [second complete attempt](qualification-02/commands.json) retained 275 passes
and one legacy-fixture failure across 276 distinct cases. The
[fixture correction](fixture-correction.json) uses the reused legacy reader's
own payload and expected source bytes; runtime production source stays unchanged.
The [third complete attempt](qualification-03/commands.json) also retained 275
passes and one failure: its legacy assertion incorrectly expected the entire
file from one bounded response. The
[bounded-page correction](bounded-page-fixture-correction.json) requests and
verifies an exact prefix with continuation metadata. The
[focused fourth run](qualification-04/commands.json) passed that one corrected
case in 25.642 seconds, using a [separate runner](focused_fixture_runner.py)
without provider doubles. The fifth complete run passed all 276 distinct cases
in 1096.213 seconds, with zero failures, errors or skips and all 218 bindings
unchanged. No earlier or focused counts are added to it. Runtime production source stays unchanged
across both legacy-fixture corrections.

Four new controls target canonical and allocated middle loads during preparation
and registration. They assert SERVER is released before entering real task and
allocation coordinator fences, avoiding a deadlock even in the negative control.

The [premerge execution](qualification-05/commands.json) remains independently
retained. The [final merged execution record](qualification-06/commands.json), raw
[stdout](qualification-06/tests.stdout.log), [stderr](qualification-06/tests.stderr.log),
[JUnit](qualification-06/junit.xml) and changed-source snapshots retain the
original run. The [documentation runner](documentation_runner.py) executes the
four exact local workflow blocks and checks the maintained guides, plan and
this record. Its [receipt](documentation-checks.json) reports local documentation
checks separately from hosted CI; it runs no additional tests.

The native fixture retains real Git, CAS, claim/coordinator/grant, sealed Portal
projection and managed worktree/lifecycle records. New probes launch real inert
same-UID Python children in the allocated directory, observe kernel birth and
working-directory facts, and terminate/reap every child in `finally` paths.
The children do not import or execute target source. Owner-side source retrieval
and actual same-owner socket refusal are distinct from a worker-UID qualification.
The existing router suite exercises authored provider replies and UID/descriptor
adapters. Quack listener/deployment metadata remain authored seams.

No live model/provider, training, solver, Docker or official task verifier is
invoked. Inert process birth is not coding-model dispatch or compiled-context
adoption. No proof, execution, completion or omission authority is supplied by
source nomination or allocation observations. All earlier evidence and separate
IR family/schema/version/decoder-task and 8D/384D/768D checkpoints, embeddings,
model-manager entries and DuckDB/DuckLake assets remain unchanged.

## Remaining production integration

The [updated plan](../../reversible_codebase_views_plan.md#native-broker-launch-boundary-2026-10-14)
keeps two deployment prerequisites separate: authenticate the cross-process
native daemon/wrapper claim and child ancestry, and retain an explicit permission
delegation alongside the original allocation. Ordinary Terminal candidate
chmods change pinned directory modes and Git-marker witnesses. Neither copied
observations nor a relaxed generic directory guard supplies that delegation.
Join signed profile `@4` and compiled instruction `@5` independently, qualify
the deployed worker, then freeze matched complete-session usage experiments.
This host boundary establishes no reconstruction or token-saving result.

The [permission and process handoff proposal](permission-handoff-design.json)
specifies four subsequent increments and their acceptance, refusal and cleanup
checks. Read-only provisioning precedes allocation capture; writable delegation
retains the original sealed witnesses; wrapper/child provenance and signed
context are separate joins. This is a proposed design, with no permission change
or deployed worker qualification claimed.

## Incoming acceptance declarations

Upstream `03ac037734` adds acceptance-checker declarations tied to immutable
native graph revisions. The [incoming review](incoming-acceptance-binding-review.json)
records the three changed paths, all outside this run's 218 bindings. Existing
native source, allocation, broker and router wiring is unchanged. The new API
declares checks; it does not execute a checker or create source/proof authority.
Its ten statically counted cases are not counted as executed here. Publication
preserves this separate upstream work alongside the qualified launch gates.

## Incoming parallel integration

Published `e4b8c4bba2` incorporates capsule-budget, per-turn usage, repository
session/context and acceptance-declaration progress. The
[integration review](incoming-integration-review.json) records 417 incoming paths:
none overlap the seven owned source/test/guide paths, but 28 of the 218 listed
dependencies change. Those changes are preserved and the merged state is qualified
in a fresh complete sixth run. That merged run passes all 276 cases; the fifth
premerge result remains separate and is not added to its count.
Incoming tests outside the twelve-suite roster are not counted as executed here.
Existing worker-side repository refresh/edit generations do not themselves
authorize a successor native Quack task, allocation or source grant.

The [indexed-proof context followup](indexed-proof-context-followup.json) identifies
a subsequent bounded adapter using existing DuckDB/capsule and proof-cache assets.
It requires the complete admitted cache key, current generation and independently
checked source-identity correspondence, preserving mandatory evidence and residuals.
Exact source-backed restoration, learned IR reconstruction, scoped proof status
and complete-session token usage remain separate measurements. This is static
planning evidence, without new implementation, execution or savings claims.
