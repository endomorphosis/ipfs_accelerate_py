# Signed worker context handoff qualification, 2026-10-13

The [coding-router guide](../../terminal_worker_signed_scope.md) describes the
public-signature and compiled-selection join for signed Terminal profiles. It
preserves complete public inputs, original effects and separate native claims.

## Execution records

The final [commands](qualification-03/commands.json),
[stdout](qualification-03/tests.stdout.log), [stderr](qualification-03/tests.stderr.log)
and [JUnit identities](qualification-03/junit.xml) retain the corrected full
15-suite invocation. The three new suites contain 96 cases: 21 public handoff,
10 integrated router/source-view and 65 archive compatibility controls.
The final run collected and executed **511 distinct passing cases**, with zero
failures, errors or skips, in 560.067 seconds of runner-observed elapsed time.
Collection, executed JUnit identities and raw pytest summary agree. All 227
listed sources stayed unchanged during final execution; datasets files matched
immutable Git objects. The work began on 2026-10-13 and final qualification
completed on 2026-10-14, as recorded in the UTC command receipts.

The [initial full run](qualification-01/commands.json) executed 511 distinct
cases: 480 passed and 31 had fixture setup errors. All errors encountered the
existing public-instruction storage guard. The new shared fixtures inherited
umask `0002`, making their runtime directories `0775`; public instruction storage
requires no group/world write. The [fixture correction](fixture-correction.json)
sets the two new fixtures' repository/runtime directories to `0755`, removing
group write. No production guard, runtime source or qualification-runner byte
changed between attempts. Initial raw logs, JUnit and nine changed-source
snapshots remain retained. The [second full run](qualification-02/commands.json)
then passed 501 cases with ten integrated-router failures: the same-process
fixture inherited its live owner's environment credentials. The existing router
correctly refused that authority before provider initialization. The
[second fixture correction](fixture-correction-02.json) scrubs the real deployed
worker credential names only around the authored worker invocation and restores
the host owner environment afterward. Product guards remain unchanged. All second
raw outputs and snapshots are retained too. There are three full qualification
invocations and no separate collection. The [runner](qualification_runner.py) records each attempt's
actual source versions, environment, arguments, timing and raw output hashes.
Earlier successful results are not used as this increment's execution result.
Two read-only diagnostic replays used one retained authored fixture: the
[public loader replay](diagnostic-01/commands.json) and
[authored router replay](diagnostic-02/commands.json). Both preserve their own
source bindings and raw outputs. They are not pytest qualifications, real model
calls, or native child execution; the latter uses an authored provider and UID
seam outside the host owner's environment. Their source versions remain separate
from the final corrected fixture.

Execution uses accelerator base
`f100a21ecb2edc7c827989bf0144478f3911d4d2`, plus five frozen production changes,
one updated compatibility test and three new test files. This base preserves
published native Codex formal-JSON planning transport alongside the earlier
signed profile implementation. The actual immutable datasets execution revision
is `59561dd2979e49fd26aa3fcfed6a5edbca1f718f`, preserving the new LegalIR
actor-ranking publication. Kit remains
`8ac00bb1974b1378e12091d546d8425af34049e0`. The 227 listed source bindings
comprise 176 accelerator and 51 datasets files, with immutable datasets Git-object
checks. This is an explicit dependency list, not a complete import closure or
full toolchain attestation. Previous phases retain their original execution pins.

## What is exercised

Owner preparation selects public instruction artifact `@5` for signed Terminal
profile `@4`. The worker verifies public signature, complete source hashes in
canonical and linked allocated checkouts, actual compiled semantic chunks,
retained native producer artifacts, exact selected/program/required populations,
raw canonical profile and selected byte bounds. A deterministic bounded inclusion
summary retains profile/declaration/manifest/frontier/scope/root/payload identities.

A real 70-original fixture keeps 73 inputs signed while selecting five semantic
sources and two vector inputs. It traverses initial indexing, task admission,
exact context reuse, a real native Quack owner capture, Portal/Todo capsule
compilation and the actual router function. The source views are real; their UID,
container-descriptor routing and provider reply are explicitly authored seams.
Selected profile/program originals can be discovered and read through that view
object; an omitted path refuses. This does not restrict the provider's other file
tools across its allocated checkout. The test launches no model coding child or
source socket broker.

Both legacy and instruction-reference metadata views retain exact profile bytes
and reproduce the complete observed authored provider input historically. Eight
controls mutate selected, omitted, support or profile originals in canonical or
allocated checkout during provider initialization. Generation refuses with the
specific currentness phase and false closing observation; authored generation
calls remain zero. The successful scoped route records
`terminal_semantic_scope_dispatch_rechecked=True` only after its closing join.

Other controls cover independently valid alternate semantic producers, missing
semantic evidence, re-pinned declaration substitution/downgrade, altered signed
manifest, selected/omitted currentness, private-owner APIs forbidden at the
public worker boundary and mutation during the final complete-source replay.
Public historical replay remains usable after current source drift and allocated
worktree removal, using baseline Git originals and retained producer artifacts.
It verifies selected original lengths there; the earlier capsule-only audit's
separate `all_selected_lengths_independently_replayed=False` observation retains
its own narrower meaning.

Fresh archive capability `@4` pins 46 files, adding 14 profile/selection/consumer
owners and explicit feature versions while retaining child custody `@1`. Missing,
stale, duplicate and copied feature/source inventories refuse before Harbor.
Extracted package members match reviewed bytes; the standalone profile producer
runs in an isolated subprocess. Extension placeholders are inert, and the archive
fixture is not a full executable runtime or installed worker qualification.

## Authority, assets and experiment limits

The public signature and original-source replay do not establish a live native
task claim, managed allocation or actual child process. Their nested inclusion
flags remain false, as do proof, execution, completion and omission authority.
Historical reconstruction does not replay dispatch. Current checks are sequential
observations without an atomic writer lease, rollback or unknown-cost guarantee.
The complete source replay keeps its independent four-million-byte bound; selected
context keeps 64 total files and one million original per-path bytes.

Git, signatures, DuckDB/Intent, vector/semantic producers, linked worktrees,
compiled capsules, native warm capture, read-only source views and historical
model-input reconstruction are real. Planner graphs, original programs, provider
replies and descriptor/UID seams are authored. Native usage counters in the provider
stub are authored fixture values and establish no real token measurement. This
qualification dispatches no external provider/model, training, solver, Docker,
target-code execution or official benchmark verifier. Private runtime databases
are not published. Existing decoder checkpoints, embeddings, IR/dimension/schema/
task inventories and model-manager entries are preserved.

The [parallel survey](parallel-work-survey.json) retains the existing local
compact-edit/usage-growth/declared-contract planning stack, absent from the queried
main and matching remote branch at observation, and the separate repository
compiler/planning policy collision. It does not establish absence under every
GitHub ref. Their results are not relabelled as this
qualification, and their worktrees are untouched. Native source broker activation
still needs authenticated cross-process claim/coordinator derivation, nominated
allocation attachment, a reviewed permissions handoff and its independent 32-path
transport bound. Rebuild and pin a complete deployment archive before live worker
qualification. The [comprehensive plan](../../reversible_codebase_views_plan.md#signed-worker-selection-handoff-2026-10-13)
keeps native adoption and complete-session matched token/task experiments next.
No learned reconstruction gain, proof, official success, provider token saving,
latency benefit or billing reduction is established by this increment.

## Documentation and review custody

The [documentation runner](documentation_runner.py) checks retained raw results,
source bindings and fixture-only correction, then executes the four exact pinned
offline documentation workflow blocks. It checks links/anchors in three changed
guides and this README. The [receipt](documentation-checks.json) and
[log](documentation-workflow.log) retain outcomes. The checker does not run pytest,
and its own binding lies outside the 227 qualification sources. Prior evidence
remains unchanged. The root-authored [review observations](review-observations.json)
record collaboration feedback and source hashes; no reviewer-authored execution
receipt is claimed. Later publication and hosted-CI observations are retained in
the parent publication record without relabelling these tested bytes.
