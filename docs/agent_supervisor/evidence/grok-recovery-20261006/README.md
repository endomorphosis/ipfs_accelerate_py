# Grok structured planning and supervisor recovery — 2026-10-06

**Trial10 passed the original `tune-mjcf` verifier but failed native shutdown.
Trial11 timed out, received reward 0.0, and completed native process cleanup.**
Both full-arm runs consumed real CPU GTE-small retrieval and fresh Source384
checkpoint inference, independently admitted two goals and one task, and ran
Grok coding through `llm_router`. A task pass followed by clean native shutdown
has not yet been demonstrated in a live run.

| Observation | Trial10 | Trial11 |
| --- | --- | --- |
| Original verifier | **1.0**, one selected task | **0.0**, same selected task |
| Learned index | **4 symbols / 11 capsules**, 17.40 s | **4 symbols / 11 capsules**, 18.19 s |
| Fresh Source384 inference | **9.74 s** | **10.60 s** |
| Coding | Returned in **534.12 s** | Timed out after **600.25 s** |
| Planning native tokens | **21,719** | **22,806** |
| Coding native tokens | **1,191,013** | **Unknown** |
| Combined native tokens | **1,212,732** | **Unknown** |
| Native lifecycle | START succeeded; STOP success absent; close failed | START / STOP / close succeeded; zero tracked processes |
| Callback / task | Task completed, revision 4 | Callback outcome unknown; task in progress, revision 3 |

The [trial10 performance record](performance-observation-10.json) includes
**1,037,440 cached input tokens** within its combined count; do not add them
again. Trial11's [performance record](performance-observation-11.json) preserves
missing coding usage as unknown. Neither run establishes verified billing,
matched baseline savings or whole-suite performance. The
[trial11 timeout supplement](timeout-settlement-observation-11.json) keeps
successful process cleanup separate from callback settlement: a complete native
exit receipt was unavailable, and the retained evidence does not identify the
refused custody check. No retry or task-completion authority follows from STOP.

Trial10 used clean source `b966623d3de7457b4139e7d9235a5976f275dd46`;
trial11 used the later shutdown correction at
`0e2d9a8c62eabdfbd42fcad505c370365474514a`, with datasets
`5171a632c6b9f0ecb2939d29d2ad74992cbfeb11` and kit
`a9b98beac1ef14278b4adf4f2289cd509530fb86`. The correction passed **120 targeted
tests** before trial11. Publication then merged concurrent Codex schema changes:
**213 separate, overlapping tests** passed at
`0a8a1b3e8448a6604899bedcbc5175dc08bf6825`; that merged source was not live
benchmarked. The [cumulative qualification summary](qualification-summary-origin-codex-merge-final.json)
retains **58 separate runs**, including earlier failures, without summing counts.

Symbolic Doctor still reported a residual without a repair/proof candidate.
Actual index and checkpoint consumption do not establish whole-task formal
verification or parallel-agent advantage. Three broader collection/API contract
gaps remain unresolved. Trial09 failed before the driver during Python runtime
download, so its reward remains **unknown**; earlier zero rewards and all
historical uncertainty remain preserved below.

## Changes

The canonical planner's JSON schema now reaches Grok's native `--json-schema`
option through `llm_router`, with a narrow provider-specific projection described
below. The adapter validates the result against the original canonical schema; the
runner validates it again before the existing planner checks graph references,
scope, evidence and admission. A valid JSON object alone grants no execution or
completion authority. Prose is not repaired into an accepted proposal. Schema
validation dependencies are pinned and checked by the container preflight.

For the supported single-task direct planning profile, preparation now supplies
a typed contract derived from the signed task. Grok's native schema intersects
the canonical grammar with that exact contract, including the required scope,
outputs, validation commands and acceptance criteria. Returned responses are
not rewritten. The original canonical validator, strict graph parser and
independent signed admission still run; the generation constraint grants no
authority. A rejected task contract can now retain bounded field names, counts
and changed record-member names without exporting model text or command values.
Other planning profiles are outside this new contract's qualified scope.

The qualified source also includes the upstream checkpoint-context integration
from `e768`; it was merged before the final contract qualification.

The canonical planner now also accepts the explicit public data labels
`application/xml`, `application/x-ndjson` and `text/calendar`. This closes a
catalog/schema mismatch while preserving exact signed media-type agreement;
unknown labels and allowed-but-unsigned changes still fail validation/admission.
Catalog-shape fixtures exercise preparation, the router process adapter, strict
parsing and signed admission using authored input bytes.

An explicit `source384-5cpu-20gib-planner180@1` profile increases the container
limit to 20 GiB. The preceding 16 GiB profile remains available. Both retain
5 CPUs and the same time/tool budgets; the generic 6 GiB replay estimate,
4 GiB child estimate and 20% headroom rule are unchanged. Runs under these two
profiles are different resource experiments, not a matched performance comparison.

A separate `source384-5cpu-20gib-coding600@1` profile raises the coding provider
cap from 300 to 600 seconds and the coding watchdog cap to 660 seconds. Both are
clamped to remaining work time minus a 25-second reserve. The outer 900-second
budget, 840-second work deadline, 60-second cleanup allocation, 180-second
planning cap, 20 GiB limit and 5 CPUs stay unchanged. The driver records the
selected coding cap and actual clamped timeout separately. These are different
coding budgets even where the common outer comparison controls match. The
source used by the sixth trial did **not** include the subsequent
callback-settlement correction; that trial failed during START before coding.

The supervisor now distinguishes its installed source from an unrelated coding
target. Target repository commits no longer cause false supervisor updates.
Real supervisor source changes, including deletion, remain detectable. Reloads
preserve original interpreter flags and recognized launch arguments, including
the configured wrapper. A separate native exec reproduction showed why this
matters: dropping `-P` left the same process alive but invisible to the native
root observer; preserving argv kept the root visible.

The longer recovery test uncovered a second defect. Watchdog maintenance wrote
coordination and execution DuckDB files into the task checkout. Admission then
correctly rejected unexpected source files. Maintenance now assigns paths in
its existing run-state directory, and single-lane Quack initialization honors
the explicit coordination path. Source checks, process identity, lease and
completion authority remain enforced.

The container report also records runtime closure separately from STOP and
tracked process counts. An empty tracked tree cannot hide refused custody of a
still-live launched child.

A subsequent cleanup repair at `d1223249f986502e3c86e4644ec2222c3e3313c8`
persists the independently witnessed launch root before inspecting descendants.
If START then fails, the existing authorized mutation-recovery path can terminate
that exact owned tree and independently check process absence before STOP.
It keeps the original failed START receipt, records the lifecycle as `FAILED`
and the control mutation as `REPAIRED`, and grants no task-completion or relaunch
authority. Original permit, source/profile identity, lease/fence and transaction
revision checks still apply. Its separate clean-source qualification and the
later combined qualification with callback-settlement and captured-record
corrections both passed. Trial07 started successfully and therefore did not
exercise this interrupted-START repair during the live run.

## Qualification

The lifecycle frozen-source run at `e048ee22e` passed **294 tests** in
**527.22 seconds** of recorded wall time (pytest reported 525.12 seconds), including
actual coding timeouts before and after the 300-second watchdog grace,
independent source revalidation, fresh daemon birth, STOP and exited launched
children. The [qualification summary](qualification-summary-final.json) retains
earlier failures and source bindings separately; overlapping passes are not
summed. After the provider-schema correction, **267 focused tests** passed in
**46.35 seconds** of recorded wall time (pytest reported 44.31 seconds) at
`038056d8277a36ca8690d51ce9fa8e4890dfecb7`. These runs overlap and are not
a combined distinct-test count. The lifecycle implementation did not
change between those commits; the real timeout tests were not repeated.

The final combined task-contract qualification at `987e0ad1f1c22ef70531449a6cbf49c786afe152`
passed **484 tests**, with zero failures, errors or skips, in **594.00 seconds**
of recorded wall time (pytest reported 591.82 seconds). Source pins remained
clean and unchanged. The [task-contract qualification summary](qualification-summary-task-contract.json)
retains earlier development failures and separate source bindings. This count
overlaps the earlier runs. The lifecycle implementation remains unchanged from
its 294-test qualification; the long native timeout tests were not rerun here.

After the public media-type correction and the merged compaction controls,
**748 tests** passed at `47281b5b94feed197b27c8ea4fd0504225337a54` with zero
failures, errors or skips in **311.36 seconds** of recorded wall time (pytest
reported 308.72 seconds). Clean source pins remained unchanged. The
[public-media qualification summary](qualification-summary-public-media.json)
binds this overlapping run separately; it is not a whole-repository pass.

The [20 GiB profile qualification](qualification-summary-memory20.json) passed
**265 tests** at `6e3171c913b76b569b85600179199f2b57ec5b52`, with zero failures,
errors or skips in **130.50 seconds** of recorded wall time. Source pins were
clean and unchanged. This is another overlapping targeted qualification.

The [coding600 qualification](qualification-summary-coding600.json) passed
**285 tests** at `0a8309047c4e2d8f780c470a02a8453cfe01ceec`, with zero failures,
errors or skips in **98.35 seconds** of recorded wall time. Clean source pins
remained unchanged. Its cumulative export preserves the earlier 18 rows and
adds two development runs and this final run; overlapping counts are not summed.
The [artifact review](coding600-artifact-independent-review-01.json) independently
checks the new recipe, observer and closed exporter. Synthetic checks cover
25 observer cases, 49 exporter cases and 22 static recipe assertions; these
checks do not establish a live task result.

The qualified implementation at `47281b5b` was pushed to `origin/main`. GitHub's documentation,
qualification and serving-contract jobs did not start: their annotations report
an account lock from a billing issue. The local documentation gate passed three
checks. These local results do not establish a successful GitHub CI run.
The [publication/CI observation](publication-ci-observation-01.json) retains the
affected check IDs and source binding.

An additional legacy CLI test could not collect because its source-only gateway
imports a missing helper. The test, importer and provider are byte-identical in
the base and final revisions. That disabled gateway has no callers in the
current admitted benchmark path. The [gap review](preexisting-gateway-import-gap-review.json)
records this preexisting limitation; this is not a whole-repository suite pass.
Two initial maintenance fixture checks omitted the fixture's native read
credential; the corrected authenticated check passed before the final run.

The [native exec observation](reload-witness-observation.json),
[lifecycle review](independent-review.json), and
[sidecar review](sidecar-initialization-review.json) bind their exact tested
revisions or file hashes. The exec experiment predates the sidecar edit; the
tested reload method is unchanged. The original historical Grok bootstrap
rejection was not retained. These reproductions establish matching defects,
not retrospective certainty about that historical rejection.

The [signed cleanup development qualification](qualification-summary-start-cleanup-development.json)
passed **6 integration cases** with no failures, errors or skips in **34.35 seconds**
of recorded wall time (pytest reported 32.56 seconds). It exercised real native
processes and the admitted typed owner without model calls: failed START,
authorized cleanup, STOP/close, authority and profile-drift refusals, stale
revisions, and lost control responses before or after committed CAS. Original
failed START evidence was preserved, with no relaunch or task mutation. The
[development binding review](native-start-cleanup-development-review-01.json)
links unchanged tested file hashes to the later clean commit `d1223249f`; the
recorded development checkout itself was dirty. The subsequent
[clean START-recovery qualification](qualification-summary-start-cleanup-final.json)
passed **100 tests** with zero failures, errors or skips at `d1223249f`, in
**143.23 seconds** of recorded wall time (pytest reported 141.34 seconds).
Source pins remained clean and unchanged. This overlapping run includes the
six integration cases; counts are not summed. It qualifies the separate START
recovery source; the later combined run is reported separately below. It does
not establish trial07 task completion.

The separate callback-settlement qualification at `e4e0ded3296261cbaf5d3dd0304c57604fdf6c85`
did not qualify the combined change. Its first broader run failed collection;
the [collection-gap review](native-settlement-preexisting-collection-gap-review-01.json)
binds the preexisting missing imports. The second run passed **440 tests** and
failed **17**, with no errors or skips, in **550.60 seconds** of recorded wall time.
The [matched existing-boundary comparison](native-settlement-existing-boundaries-comparison-01.json)
reproduced those same 17 failing test identities on the preceding `0a830904`
source. They are not all obsolete expectations: six expose substantive captured
record ownership/replacement gaps addressed by the captured-record correction
below and included in the later combined qualification. Other failures concern
older generic-exception behavior and absent callback APIs; separate native
settlement tests do not establish equivalent coverage. These failures remain
visible and prevent a whole-repository pass claim.

The captured-record correction at `d862fe847532a83e3dfb57d237ce581c3f44dc09`
requires exact persisted ownership and a matching task index when activating,
moving a pooled workspace, settling, terminalizing or deleting a captured
lifecycle record. A path lookup cannot replace that capture; reused lease/fence
values do not authorize a different record. Missing state does not prove this
caller finalized it, and failed comparisons preserve the captured state.
The [captured-custody development summary](qualification-summary-captured-custody-development.json)
retains four separate runs: 56 passes, a collection error with no executed tests,
87 passes and finally **159 passes** with no failures, errors or skips in
**182.56 seconds** of recorded wall time (pytest reported 180.38 seconds).
The final development source stayed unchanged but was dirty. Its tests include
actual authored provider processes, signed native admission, successful runtime
cleanup, pool transitions and replacement-owner denials, without model calls.
The [committed source binding](captured-lifecycle-committed-source-binding-01.json)
confirms the four tested files exactly match `d862fe847`; it does not relabel the
development run as a clean committed-source qualification.

The [captured-custody review](captured-lifecycle-custody-development-review-01.json)
explicitly supersedes the sufficiency of the earlier
[settlement review's no-blocker conclusion](native-provider-failure-independent-review-01.json):
the broader tests subsequently found a substantive ownership gap. Both reviews
remain available. The extra [pool collection-gap review](lifecycle-pool-collection-gap-review-01.json)
records an unchanged legacy test importing the absent `portal_task_identity`
helper on both prior sources. No removed API was restored, and the current
native pool tests do not claim equivalent coverage of that uncollected suite.

A subsequent [collection-contract audit](preexisting-collection-contract-audit-01.json)
confirms three unresolved contracts; they must not all be dismissed as obsolete
fixtures. The pool test needs a migration to verified projection-local identity.
The protected-path suite expects a same-attempt recovery API for which current
cross-attempt clearance/retirement/adoption is not equivalent. Candidate-rejection
cleanup has missing runtime observer/custody joins as well as a missing test
helper. No removed authority was restored, and these suites were not relabelled
passing or requalified by the static audit.

The [cumulative qualification history](qualification-summary-combined-recovery-final.json)
retains **35 recorded run rows**, including the original failed collections,
the 440-pass/17-failure run, its matched 2-pass/17-failure baseline, the dirty
captured-custody development and the clean 100-test START-recovery qualification.
This is a count of runs, not distinct passing tests. Earlier informal settlement
development remains in the [initial development report](native-provider-failure-development-01.json)
and [pool/signed supplement](native-provider-failure-development-02.json). Repeated cases are never
summed. The combined source qualification and trial07's failed task result are
recorded separately below.

The subsequent [cleanup-observation development run](qualification-summary-start-cleanup-observation-development.json)
passed **53 tests** in **61.10 seconds** of recorded wall time: 35 focused
observation cases, six signed cleanup integrations and 12 driver dispatch cases.
Source remained unchanged during this dirty development run. The
[committed byte binding](start-cleanup-observation-committed-source-binding-01.json)
confirms the three tested files match both `912d0af3a2594b26f0b71731c4e85dea8f3a08f0`
and the combined source `020e6adccc9ad56b1df0ee2b5984ce0adcb39d89`. The
development run remains distinct from the clean combined qualification.

The report observes the original failed-START process proof and control repair
receipt separately. If a lost post-CAS response leaves the control convenience
receipt missing, the observation remains partial. Process absence means the
recorded marker-bound tree, not a continuously current global absence claim.
These observations grant no completion, retry or execution authority, and
`runtime.close()` remains independent of diagnostic success. The
[independent observation review](start-cleanup-observation-independent-review-01.json)
and [71-case exporter safety review](fresh-live-inspector-safety-review-08.json)
record these boundaries. The exporter preserves the original trial01–06
summaries byte-for-byte. A separate [captured-owner review](independent-captured-lifecycle-review-02.json)
checked the final activation and pooled-rebind lock ordering and capture checks.

The final combined native-recovery qualification passed **282 tests** across
12 targeted suites at clean source `020e6adccc9ad56b1df0ee2b5984ce0adcb39d89`,
with zero failures, errors or skips, in **327.73 seconds** of recorded wall time.
Source remained unchanged. This includes captured ownership, native failed-child
settlement, signed admission, successful runtime cleanup, interrupted START
repair and bounded repair observations. Provider boundaries were authored;
these tests made no model calls. The cumulative history binds this run separately
and preserves the earlier failures. It is a targeted qualification, not a
whole-repository pass or a successful live task result.

After trial07, the router's bounded preflight diagnostic development passed
**40 tests**, then **21 updated focused tests**, with unchanged source during
each dirty development run. The [router review](router-preflight-diagnostics-independent-review-01.json)
binds the implementation at `b79159ca24e189c1747eb2544913367ba15f36fa`.
A separate [native bridge review](native-bridge-diagnostics-independent-review-01.json)
binds `403e8045bd8755bc188d8d91d80241e935667c2a`: closed child error metadata
can be observed from a bounded tail of the exact native log. Such child-reported
metadata cannot establish dispatch, process custody or settlement authority.
Neither review reconstructs the unretained cause of trial07's bridge failure.
The subsequent clean combined diagnostic qualification passed as recorded below;
the controlled follow-up trial08 closed as recorded below.

The [diagnostic qualification history](qualification-summary-terminal-diagnostics-final.json)
preserves all preceding rows and now contains **44 recorded run rows**, with no
sum of overlapping test counts. Native diagnostic producer development passed
48 tests and then 52 tests. The first collector run passed 68 tests and failed
one actual signed integration: production disables the optional JSON event
projection the collector had expected. That failed run remains recorded.

A dedicated bounded sidecar now retains the closed diagnostic in the explicit
native state directory, bound to the exact task and attempt hashes. The collector
uses the created runtime's nested state path and requires matching task revision,
status and failure-receipt attempt identity. Native database records remain
authoritative; the sidecar grants no authority. Sidecar development passed
31 tests. Corrected collector development passed **71 tests**, including a
real signed START, authored preflight failure, blocked task, STOP and runtime
closure with optional event export absent. These runs made no LLM calls.

The [committed implementation/export review](terminal-native-diagnostic-committed-export-review-01.json)
binds the reviewed bytes to `585cf2adccdbaff8a838a5f5438ca80ea75c8b0e`.
The [100-case exporter review](fresh-live-inspector-safety-review-11.json)
checks closed diagnostic fields against validators whose files match the
selected archive revision, and preserves the original trial01–07 summaries.
The clean combined diagnostic gate then passed **127 tests** across seven
selected suites with zero failures, errors or skips, in **168.74 seconds** of
recorded wall time, at unchanged clean source
`585cf2adccdbaff8a838a5f5438ca80ea75c8b0e`. The
[final qualification review](terminal-diagnostics-final-qualification-review-01.json)
independently checks the counts, hashes and preserved prior 43 rows. Provider
boundaries were authored; the tests made no live model calls. These overlapping
runs do not establish a whole-repository pass or reconstruct trial07's cause.
The controlled lexical trial08 subsequently closed with reward 0 as recorded below.

## Live structured transport

The [readiness probe](grok-schema-readiness.json) passed in **4.690 seconds**.
It made one explicit `grok_cli` call requesting `grok-4.7` with high reasoning,
using pinned CLI 1.0.46 and no provider fallback. A nonce supplied only inside
the JSON prompt was echoed exactly in schema-valid output. Native logs recorded
zero advertised tools and zero tool invocations; cleanup completed. This tests
real prompt delivery and structured output in an isolated readiness image,
separate from the full task image.

The native final envelope reported 3,111 tokens; completeness and billing remain
unknown. Requested model identity is recorded, without independent provider-side
model attestation. The [structured transport review](grok-structured-review.json)
keeps these limits separate from plan admission and task correctness. Development
counts and pending-at-review-time notes in the transport reviews are historical;
the completed source-bound runs above are the current test qualification.

A later [canonical-schema probe](canonical-schema-diagnosis-01.json) reproduced
an HTTP 400 rejection of the original schema's top-level `$id` as a
`uri-reference`. This is observed in the synthetic probe; the exact first task
error was not retained. The router now recognizes the complete canonical
planner grammar and omits only that top-level annotation in the native wire
schema. All assertions, local references and original local validation remain
unchanged. Other schemas and modified lookalikes are not rewritten. Receipts
bind both canonical and native schema hashes. Native receipts can also retain
the closed `schema_rejected` / HTTP 400 / `schema_id_invalid` / `/$id` diagnosis
without reflecting provider messages; timeout retains precedence.

The [corrected canonical probe](grok-schema-projection-qualified.json) passed in
**7.413 seconds** with one call and one model round: the exact authored proposal
passed the original validator and strict graph parser, with zero tools and clean
shutdown. Its 6,903 observed native tokens remain incomplete billing evidence.
The [projection review](grok-structured-projection-review.json) binds this
correction separately from the earlier small readiness probe.

The [signed-task schema probe](signed-task-schema-probe-01.json) at `987e0ad1f`
passed in **26.346 seconds**. It exercised the real prepared direct-task
contract and native `const`/`allOf`/`anyOf` intersection against an authored
fixture, without benchmark inputs. The returned graph exactly matched that
fixture, passed the original canonical validator and strict parser, and passed
independent local signed admission. Native logs recorded zero advertised tools
and zero calls; cleanup completed and all three source pins remained unchanged.
Its **14,856 native tokens** have unknown completeness and unverified billing.
This establishes provider compatibility, not benchmark task success.

The subsequent [public tune-profile preflight](public-tune-profile-preflight-01.json)
at `47281b5b` passed in **2.192 seconds** with no live provider call. It uses
the public catalog's XML declarations with authored minimal inputs, exercises
the real process adapter with a synthetic executable, and verifies canonical
and native schemas, the strict two-goal/one-task parser and independent signed
admission. The response and signed declarations are unchanged; three execution
requirements remain pending. It does not establish task completion.

The [first archive review](archive-review-01.json), bound to `e048ee22e`,
independently rehashed the whole 1,410,167,325-byte archive and checked all
27,350 regular members against its manifest, including the actual Grok binary.
The [corrected archive review](archive-review-02.json), bound to `038056d827`,
separately verified its 1,410,178,734-byte archive and all 27,350 regular members.
Each archive matches its own clean frozen source checkouts; archive qualification
alone is not a task result. Preparation retains 900 seconds total, 840 seconds
work, 60 seconds cleanup, 180 seconds planning and 300 seconds coding.

The [third archive review](archive-review-03.json), bound to `987e0ad1f`,
verified the complete 1,410,196,657-byte archive and all 27,352 regular members.
All seven task-profile files are byte-identical. Planning requests zero native
tools; the later coding phase permits its separate six-tool profile.
The [fourth archive review](archive-review-04.json) at `47281b5b` verified the
complete 1,410,204,145-byte archive and all 27,353 regular members.
The [fifth archive review](archive-review-05.json) at `6e3171c9` verified the
complete 1,410,204,467-byte archive and all 27,353 regular members before the
separate 20 GiB experiment.
The [live resource observation](resource-observation-05.json) verifies that the
fifth container actually has 21,474,836,480 bytes in both Docker's limit and
cgroup `memory.max`, unlimited `memory.high`, and a CPU quota of 5. The
[independent observer review](explicit20gib-independent-review-01.json) and
[13 synthetic checks](explicit20gib-observer-safety-review-01.json) cover tighter
or malformed `memory.high` rejection; they do not substitute for that live limit
observation.

The [sixth archive review](archive-review-06.json) at `0a830904` verified the
complete 1,410,205,082-byte archive and all 27,353 regular members. The sixth
[profile-bound resource observation](resource-observation-06.json) binds the
prepared coding600 profile to its exact container and confirms the same actual
20 GiB limit, unlimited `memory.high` and 5-CPU quota. Resource observations do
not attest the actual provider deadline. The driver's separate cap/clamped
fields record the configured allowance; a coding invocation receipt is also
needed to establish provider execution under it, and the sixth trial has none.

The [seventh archive review](archive-review-07.json) at `020e6adc` independently
verified the complete 1,410,182,521-byte archive and all 27,353 regular members.
Its [preparation review](preparation-review-07.json) preserves unchanged task
profile/input bindings and the selected Source384 checkpoint. The seventh
[resource observation](resource-observation-07.json) confirms the prepared
coding600 profile's actual 20 GiB limit, unlimited `memory.high` and 5-CPU quota.

## Fresh task result

The first fresh full indexed `tune-mjcf` trial received **official reward 0**.
The [trial summary](live-trial-summary-01.json) binds its original verifier
result. Total invocation time was 171.85 seconds, including 121.31 seconds of
agent setup and 24.05 seconds of agent execution.

Actual indexing produced **4 symbols and 11 capsules** in 14.31 seconds. Fresh
Source384 inference took 9.74 seconds (`neural_inference_replayed=false`) and consumed checkpoint
`2ca38dfcc05536315fc3e2c0647b710b930ef4066b474061a7b4e5bfb9a258c5`.
Its program population was one Python file; three harness-support files remained
separate. Source384 advice retained nomination-only status. The initial index
reported `learned_embeddings=false`; checkpoint consumption does not imply that
every index used learned embeddings. See the [indexed-path observation](indexed-path-observation-01.json).

Planning returned a native execution-error envelope after 2.33 seconds, before
local schema validation, plan admission or START. No coding, Doctor execution or
admitted task world state was observed. Usage is unknown. Worker cleanup exited
zero and the owned container was removed. The exact native error message did not
survive the existing Harbor download boundary; [closed failure metadata](planning-error-observation-01.json)
cannot identify its cause. The subsequent synthetic probe established the
schema compatibility defect and verified its correction.

The [second fresh task trial](live-trial-summary-02.json), using the corrected
schema projection at `038056d827`, also received **official reward 0**. It ran
for 260.90 seconds including 131.34 seconds of setup and 101.52 seconds of agent
execution. Grok returned normally after 77.16 seconds; its output passed the
original canonical schema and the strict planner accepted **2 goals, 1 task,
and 7 evidence records**, with no fallback. Native logs observed zero advertised
tools and no tool calls in the bounded [tool observation](native-tools-grok-tune-mjcf-02.json).

The local admission gate then raised `LocalPlanningError`: the proposal changed
signed scope, acceptance, dependency or command. No admitted plan, START, coding,
or Doctor execution followed. The [admission observation](planning-admission-observation-02.json)
identifies the precise source branch; its combined exception did not retain the
particular differing field. The private proposal was not retained on the host,
so this run cannot be replayed to identify that field. Cleanup exited zero and
the owned container was removed.

The native final envelope reported **20,404 planning tokens**: 11,284 input,
1,536 cached input and 7,584 output, including 5,997 reasoning tokens. Completeness
and billing remain unverified; these are not successful-task token costs.
Actual indexing again produced 4 symbols and 11 capsules, with fresh pinned
Source384 inference. The [second indexed-path observation](indexed-path-observation-02.json)
retains timings, reference digests and the nomination-only authority limits.
No task success, savings or baseline advantage is claimed.

The [third fresh task trial](live-trial-summary-03.json) at `987e0ad1f` received
**official reward 0** after a preparation failure, before any provider invocation.
Its total invocation time was **153.06 seconds**, including 122.07 seconds of
setup and 2.33 seconds of agent execution. The retained error is a `ValueError`
from `terminal_planner_contract.validate_task_contract`: the public profile's
`application/xml` output is absent from the canonical planner's media-type enum.
The [closed diagnosis](preparation-failure-observation-03.json) reproduces the
single mismatch using the exact historical grammar and public declaration shape.
The earlier authored probe covered Python and text outputs, so it did not expose
this format gap. Host benchmark preparation builds the bundle/configuration;
the failing task-contract preparation runs later inside the container.

This attempt did not hydrate indexes, run Source384 inference, admit a plan,
START a supervisor, or dispatch coding/Doctor work. There are zero recorded
provider invocations; token usage remains unknown. The bounded
[third tool observation](native-tools-grok-tune-mjcf-03.json) found no native
logs, which does not establish a zero-tool session. Cleanup returned zero, the
owned container was removed, and the source pins were unchanged through the run.
Its failed result remains separate from the earlier planning-admission failure.

The [fourth fresh task trial](live-trial-summary-04.json), at `47281b5b`, passed
the corrected preparation and independently admitted **2 goals and 1 task**.
Native **START and STOP both succeeded**; runtime closure was explicitly
attempted and succeeded, with zero remaining tracked processes and cleanup
return code zero. The owned container was removed and source pins remained
unchanged. The task still received **official reward 0**: the 840-second work
budget expired with the task at `in_progress`, revision 3, before a coding
provider call. Total invocation time was **991.23 seconds**, including 120.56
seconds of setup and 844.15 seconds of agent execution.

The [fourth indexed-path observation](indexed-path-observation-04.json) records
actual hydration of 4 symbols and 11 capsules in **13.52 seconds**, fresh
Source384 inference in **9.02 seconds**, and reuse of the same index in the
admitted context. That context selected **3 worker capsules / 29,875 semantic
bytes**, with no new embedding calls. Source384 remained nomination-only;
the index still reported `learned_embeddings=false`.

Doctor ran with zero provider calls and returned one residual successor, no
work proposal and no repair candidate. Its selected generic keyword-renaming
workflow did not cover the XML output or task-data semantics. The four retained
reasons are incomplete source inventory, unsupported output effect/language,
no supported keyword mismatch, and unavailable task-data contract. Lean and Z3
were present, but no local proof stage or proof receipt was reported. The
assessment does not establish whole-task behavior, parallel decomposition or
formal proof of the task.

The [public-data symbolic gap review](public-data-symbolic-gap-review-01.json)
recommends a future finite NDJSON field-projection contract and independent
input/output correspondence checker before admitting a new Doctor operator.
Existing structural parsers and prover availability do not establish data-task
semantics. XML numerical behavior and calendar recurrence need separate reviewed
contracts; this proposed integration is not implemented or benchmark-qualified
by these runs.

The later coding gate repeatedly deferred on `proof_memory_headroom`. Six
[retained resource observations](live-dispatch-wait-04-04.json) reported
9,204–9,382 MiB available against a 9,421 MiB requirement: 6,144 MiB plus
3,277 MiB headroom, a shortfall of 39–217 MiB. The callbacks/effects had not
started and no provider attempt was consumed. Fresh native heartbeats and
completed daemon passes distinguish this from a stopped supervisor; an earlier
`ready:1` database observation was stale. No live admission rule or memory
limit was changed during the run.

The only recorded model call was planning: **18,060 native tokens** comprised
13,329 input, 1,536 cached input and 3,195 output, including 1,464 reasoning
tokens. Completeness and billing are unverified, and aggregate trial usage
remains unknown. The [fourth tool observation](native-tools-grok-tune-mjcf-04.json)
records one planning session with zero advertised tools and no tool events;
it does not claim observation of a coding session. No successful-task token
score or advantage over a matched baseline is established.


The [fifth fresh task trial](live-trial-summary-05.json), at `6e3171c9`, used the
explicit 20 GiB profile and reached coding through `llm_router`. Its plan again
independently admitted **2 goals and 1 task**. Native START and STOP succeeded;
runtime closure was attempted and succeeded, with zero remaining tracked
processes and cleanup return code zero. The owned container was removed and all
three source pins stayed clean and unchanged. The task received **official
reward 0**. Total invocation time was **995.25 seconds**, including 121.97 seconds
of setup and 845.64 seconds of agent execution.

The [fifth indexed-path observation](indexed-path-observation-05.json) records
4 symbols and 11 capsules hydrated in **13.38 seconds**, fresh pinned Source384
inference in **8.86 seconds**, and reuse of that index in the admitted context.
The selected context again contained 3 worker capsules / 29,875 semantic bytes.
Doctor retained the same four residual reasons and no repair/proof candidate;
checkpoint inference, semantic coverage and proof-authority limitations remain
as described for the fourth trial. This run performed no training.

The 20 GiB run crossed the earlier pre-coding resource gate: a coding provider
call actually started, and the retained live observations contain no resource
retry. That call raised `TimeoutExpired` after **300.228 seconds** against its
300-second allowance, with no final native envelope or coding token totals.
The [retained timeout and settlement observation](live-admission-05-09.json)
shows the native bridge exception followed by repeated
`provider_callback_outcome_unknown` deferrals. The
[historical source-bound diagnosis](coding-settlement-observation-05.json)
matches the retained exception hash to `portal_provider_failed` and the exact
`started_outcome_unknown` callback branch in the tested source. This left the
task `in_progress` until the original work deadline. Crossing the memory gate
is established by actual coding dispatch; the separate coding timeout and
unresolved settlement remained failures in this run. The longer wait did not
establish task completion or permit an automatic retry.

The [fifth tool observation](native-tools-grok-tune-mjcf-05.json) separates the
zero-tool planning exposure from the six-tool coding exposure. The additional
[closed completion observation](closed-tool-completions-grok-tune-mjcf-05.json)
records **22 completed tool outcomes**, including 12 native
`run_terminal_command` outcomes: 11 succeeded and one failed. Their durations
were 8–2,699 milliseconds. Completion records do not establish which operation,
if any, remained active at the provider deadline; command categories and
arguments were not inspected.

Planning reported **21,624 native tokens**: 13,646 input, 1,152 cached input and
6,826 output, including 4,905 reasoning tokens. Coding and aggregate usage remain
unknown, and billing is unverified. No successful-task token score or matched
baseline advantage follows from this larger-memory experiment. The
[closure observation](trial-closure-observation-05.json) binds the final result,
cleanup observation and source pins without exporting private task/model data.


The [sixth fresh task trial](live-trial-summary-06.json) at `0a830904` also
received **official reward 0**, ending after **277.22 seconds** rather than
using its larger coding allowance. Setup took 123.34 seconds and agent execution
126.34 seconds. Planning independently admitted 2 goals and 1 task. The driver
recorded a selected coding cap and computed timeout of 600 seconds, but there is
no coding provider invocation receipt; this attempt does not establish execution
under that allowance or test the subsequent callback-settlement correction.

Native START returned a `ProcessIdentityMismatch`: a process did not match the
selected run/profile markers. The transaction entered `repair_required` without
recording an applied effect. STOP then conflicted because that lifecycle transition was
still active. The final runtime custody guard refused closure while live launched
children remained. The [source-bound lifecycle diagnosis](lifecycle-conflict-observation-06.json)
identifies these exact source branches. The particular differing marker, a PID
reuse explanation, and an exec race are not established by the retained metadata.

The [sixth closure observation](trial-closure-observation-06.json) preserves
**START conflict, STOP conflict, runtime close failed and 2 tracked processes**
at that boundary. Worker cleanup later returned zero and the container was
removed; that teardown does not convert the refused native closure into success.
P/D/K remained clean and unchanged through the run.

The [sixth indexed-path observation](indexed-path-observation-06.json) records
actual hydration of 4 symbols and 11 capsules in **13.65 seconds**, fresh
Source384 inference in **9.19 seconds**, and same-index admitted-context reuse.
Doctor again returned one residual successor with no repair/proof candidate and
the same four retained coverage reasons. The formal-proof and completion limits
remain unchanged. The [sixth tool observation](native-tools-grok-tune-mjcf-06.json)
recorded one planning exposure with zero advertised tools and no tool outcomes;
no coding exposure was observed in its bounded windows.

Planning reported **22,963 native tokens**: 13,553 input, 1,280 cached input and
8,130 output, including 6,278 reasoning tokens. These are failed-run planning
observations with unverified billing/completeness. They are not successful-task
costs or evidence that the larger coding budget improves performance.

The [seventh fresh task trial](live-trial-summary-07.json), at combined source
`020e6adc`, received **official reward 0** after **278.92 seconds** total.
Agent setup took 126.14 seconds and execution 125.24 seconds. Planning again
independently admitted **2 goals and 1 task**. Native **START and STOP succeeded**,
runtime closure succeeded, zero tracked processes remained and worker cleanup
returned zero. The owned container was removed, with clean P/D/K unchanged
through closure; the [seventh closure observation](trial-closure-observation-07.json)
retains those independent checks. Since START succeeded, the new repair
observation correctly reports `no_failed_start`; no cleanup repair is claimed.

The task settled to **blocked, revision 4** on its first attempt. A native
`database_task_claim_failure` receipt records `terminal_portal_bridge_error`,
zero effect claims and no automatic retry authority. This is a failed task
settlement, not a completion receipt. Unlike trial05's repeated unknown-outcome
deferrals, the driver observed terminal state and stopped promptly. The
[closed settlement observation](terminal-settlement-observation-07.json) preserves
this distinction. The exact bridge failure reason did not survive the retained
report: its native exception/traceback lists are empty. One recorded native
attempt is not proof of an actual Grok coding dispatch; there is no coding
router receipt, and coding dispatch and usage remain unknown. The retained
receipt also does not establish whether the new waited-child settlement
capability was exercised. No retry was run.

The [seventh indexed-path observation](indexed-path-observation-07.json) records
4 symbols and 11 capsules in **14.23 seconds**, fresh Source384 inference in
**9.75 seconds**, and reuse of the same index in admitted context. That context
selected 3 capsules / 29,875 semantic bytes in **3.60 seconds**. Doctor again
returned a residual with the same four coverage reasons and no repair/proof
candidate. A full task semantics contract, broader operators and a proof of
the task's numerical/XML behavior remain gaps.

The [retrieval-mode observation](retrieval-mode-observation-07.json) clarifies
`learned_embeddings=false`: this run selected **lexical TF-IDF vectors over
qualified symbol names**, persisted through DuckDB and the DuckLake metadata
projection. The separate neural retrieval snapshot selector was absent.
These are populated vectors, not hash vectors. Source384 separately used its
pinned GTE-small embedding snapshot and security384 checkpoint for fresh
inference over one declared Python program file; this does not select neural
embeddings for the retrieval index. The explanation binds the completed
construction path to archived source; the publication observer did not reopen
the private databases. Source384 remains nomination-only, with no proof or
completion authority and no training in this run.

Planning reported **21,316 native tokens**: 13,296 input, 1,536 cached input and
6,484 output, including 4,869 reasoning tokens. The bounded
[seventh tool observation](native-tools-grok-tune-mjcf-07.json) retained one
planning exposure with zero advertised tools and no tool outcomes; it observed
no six-tool coding exposure. Coding and aggregate usage remain unknown and
billing is unverified. This unsuccessful run establishes neither a successful
task token score nor a performance advantage over a matched baseline.

## Separate local neural retrieval qualification

The separate retrieval fix at `2bb615d917e41c65af2fbca802cb30ae4e3e8b08`
removes the hardcoded MiniLM revision from the learned retrieval configuration.
Preparation now selects the exact revision from the deployment manifest, and
setup, run and execute reject an active pin mismatch before deployment or
dispatch. No-index and legacy lexical selections keep neural retrieval disabled.
The [retrieval review](retrieval-revision-and-local-gte-review-01.json) binds
**52 passing development tests** to the committed four source files; the dirty
development result is preserved without relabelling it as a clean-source run.

An authored two-symbol fixture then used the existing local GTE-small snapshot
`17e1f347d17fe144873b1201da91788898c639cd` for **three real CPU embedding calls**.
The [local probe](local-gte-retrieval-probe-02.json) produced 384-dimensional
vectors, passed the provider canary, persisted and reopened the DuckDB index
unchanged, replayed both native AST rows and projected metadata into DuckLake.
It took **3.43 seconds** inside qualification, or **4.43 seconds** for the
[bounded process](local-gte-retrieval-process-02.json). It performed no training,
model downloads or LLM calls. Embedding inputs were qualified symbol names;
this is neither whole-function semantic retrieval nor formal correctness proof.

The [first process attempt](local-gte-retrieval-process-01.json) failed before
model loading because the added Harbor package path selected tokenizers 0.23.2
with Transformers 4.52.1. The second attempt used the existing compatible user
packages, including tokenizers 0.21.4, without changing installed dependencies.
The successful process peaked at **1,090,304 KiB RSS** and loaded **133,440,000
bytes** of FP32 parameters. These tiny-fixture measurements are not hard memory
bounds, incremental container overhead or benchmark throughput claims. Generic
memory reservations remain unchanged.

This retrieval change was absent from the sources used by trial07 and
trial08, which retained lexical retrieval. Its CPU probe does not
establish CUDA behavior, successful benchmark completion, token savings or an
advantage over the baseline. Adding the selected snapshot to a later bundle also
adds a separate retrieval copy of its nine assets, **67,691,071 uncompressed
bytes**; its existing Source384 copy remains distinct and compressed growth has
not been measured.

## Eighth controlled trial and retained native failure

The [eighth live summary](live-trial-summary-08.json) records official reward
**0** at unchanged source `585cf2adccdbaff8a838a5f5438ca80ea75c8b0e`, with the
same lexical retrieval, Grok profile, 20 GiB limit and configured 600-second
coding allowance as trial07. Planning independently admitted two goals and one
task. START, STOP and runtime closure succeeded; the
[closure receipt](trial-closure-observation-08.json) records zero remaining
processes, successful worker cleanup and removal of the owned container.
The task finished blocked at revision 4.

This time the [native settlement observation](terminal-settlement-observation-08.json)
retained `portal_provider_failed`, callback state `failed_outcome_settled`,
return code 1, and native observations of leader reaping, absent process group,
absent adopted children and finalized lifecycle. The exported diagnostic remains
observation-only. The child router diagnostic and coding invocation receipt
were missing; neither missing item proves whether a model call occurred.
Coding and aggregate token usage remain unknown.

Source inspection subsequently found a concrete launcher contract mismatch:
the deployed worker refused timeouts above 300 seconds before importing the
router, while this profile supplied 600 seconds. This explains why that rejection
cannot emit the router's structured error envelope. The [isolated Docker comparison](offline-worker-budget-comparison-01.json)
then reproduced that blocker through the actual owner → sudo → worker path
with networking disconnected and no real credentials or model calls. Across
six cases, the old worker rejected 600 before router entry; 300 reached the
router's authored invalid-model check. The corrected worker with an old router
preserved 300 and refused 600. With the corrected router, 600 reached its
argument validation and 601 remained rejected. These checks used a rebuilt
image from the same public task Dockerfile and settings, not the exact trial08
image. The [independent review](offline-worker-budget-independent-review-01.json)
verifies the old/new worker and router hashes. This source-bound reproduction
supports the configuration diagnosis; it does not retroactively recover trial08
stderr or change that trial's missing child diagnostic.

The [offline attempt history](offline-worker-budget-attempt-history-01.json)
also preserves two failures. The first could not start an incompatible registry
image. The second failed a case assertion without retaining enough evidence to
establish the cause; a mixed-version explanation is not claimed as proven.
Its initial Harbor stop left a stopped container behind, which the final audit
explicitly removed with its network. The final audit verified all three owned
probe containers and networks absent. The successful baseline's initial
`cleanup_pending` receipt is historical; the later comparison records removal.

The worker now shares the router's absolute 600-second ceiling. An old archived
router with no declared constant uses the legacy 300-second worker ceiling;
import errors are not swallowed. Existing profiles still select 300 seconds
and their original watchdog budgets. This does not add per-profile enforcement
to the worker itself or widen outer deadlines.

The [worker archive capability fix](worker-archive-capability-committed-review-01.json)
at `d39afb6df3cedbb15b46cb4a2777c340bd499376` rejects extended coding budgets
unless the archive declares the exact worker template, shared ceiling and
matching installer/router source hashes. Preparation, execution, setup and run
check compatibility before planning or deployment; legacy 300-second profiles
remain usable and no silent budget downgrade is introduced. Tests cover the
GTE revision and 600-second capability in the same preparation and driver path.
The [expanded history](qualification-summary-worker-capability-final.json)
contains **50 separate recorded runs**. It preserves the initial 64/86-test
runs, the 45-test legacy fallback run, an 88-test pre-fallback run mistakenly
started before a refused cherry-pick was applied, and the corrected **95-test**
combined development run. Each source binding remains distinct; counts are
not summed. The final development run took **53.00 seconds**, with unchanged
dirty source whose bytes match the subsequent commit. The subsequent
[clean combined gate](combined-worker-capability-final-review-01.json) passed
**197 tests** across eight selected suites, with zero failures, errors or skips,
in **84.80 seconds** of recorded wall time at unchanged clean source
`b966623d3de7457b4139e7d9235a5976f275dd46`. It combines worker limits, legacy
compatibility, archive guards, retrieval selection and relevant driver controls.
The subsequent trial09 failed during setup, as recorded below. These targeted
tests do not establish benchmark success or whole-repository correctness.

The [eighth indexed-path observation](indexed-path-observation-08.json) records
4 symbols and 11 capsules in **19.82 seconds**, fresh Source384 inference in
**15.35 seconds**, and admitted reuse of three capsules / 29,875 semantic bytes.
The [retrieval observation](retrieval-mode-observation-08.json) again separates
lexical TF-IDF retrieval from actual Source384 GTE/checkpoint inference. Doctor
returned a residual without a repair/proof candidate. Planning reported
**21,343 native tokens**: 13,762 input, 1,152 cached input and 6,429 output,
including 4,515 reasoning tokens. Billing and usage completeness are unverified.
No successful task score, token savings or matched baseline advantage is claimed.

## Ninth attempted learned-retrieval run

The [ninth archive review](archive-review-09.json) independently verified
27,366 members and the complete 1,472,485,784-byte archive against clean pinned
source. The [preparation review](preparation-review-09.json) and
[independent configuration binding](learned-preparation-independent-review-09.json)
confirm the exact GTE-small revision, nine local retrieval assets totaling
67,691,071 bytes, unchanged task inputs, pinned Source384 checkpoint and reviewed
600-second worker capability. Configuration and 12 runtime source hashes match
the selected commit. This preparation selected CPU neural retrieval; it does
not establish that embedding inference ran.

The [ninth live summary](live-trial-summary-09.json) reports **unknown reward**:
no original verifier result was produced. The attempt stopped during agent
setup after **73.15 seconds**, with **91.83 seconds** total recorded invocation
time. The [setup diagnosis](setup-failure-observation-09.json) joins the exact
source-owned deployment exception to the retained installation log: uv 0.9.24
installed, then the pinned CPython 3.12.12 aarch64 asset download failed with
HTTP 500 after three retries. The failing network intermediary is not
established. A zero outer recipe exit does not make the task successful.

The driver did not start. The [index observation](indexed-path-observation-09.json)
therefore remains unavailable; no actual retrieval hydration, Source384
inference, plan admission, Doctor run or coding result is established. Token
usage remains unknown. The [resource observation](resource-observation-09.json)
records the selected 20 GiB/5-CPU profile separately from actual Docker/cgroup
limits, which were not observed before the container disappeared.

The [closure observation](trial-closure-observation-09.json) verifies clean,
unchanged source pins and the exact owned container absent after cleanup.
Native START, STOP, runtime closure, remaining process counts and worker cleanup
return code remain unobserved. The [independent publication review](trial09-independent-publication-review-01.json)
checks the retained source/log bindings and these unknown fields without
exporting raw logs or private task data. This failed setup attempt does not
qualify neural retrieval in the benchmark, establish task completion, or supply
a successful-task token score. It also changed retrieval from trial08, so no
matched performance advantage is claimed. Any later run must retain this failed
attempt separately.


## Tenth full-arm task result

The [tenth live summary](live-trial-summary-10.json) records original verifier
reward **1.0**, with a native task state of `completed`, revision 4. Setup took
**121.02 seconds**, agent execution **669.90 seconds**, and verification
**19.33 seconds**. Total invocation time was **830.21 seconds**. The exact
[reused-archive preparation](preparation-review-10.json) retained trial09's
reviewed archive, model/checkpoint, provider and resource settings. A public
[runtime asset availability check](runtime-python-availability-09-01.json)
returned HTTP 200 through the host network before this fresh attempt; that HEAD
request downloaded no asset bytes and did not attest container-network equivalence.
Trial09's HTTP 500 failure remains separately recorded.

The [learned runtime observation](learned-vector-runtime-observation-10.json)
and [independent live result check](learned-runtime-independent-review-10.json)
confirm **three actual CPU embedding calls over eight texts**, 384-dimensional
vectors, passed canary checks, four indexed symbols, four replayed native AST
fact rows and a successful DuckLake metadata projection. The embedding input is
**qualified symbol names only**. GTE-small revision
`17e1f347d17fe144873b1201da91788898c639cd` used local safetensors, no remote
code and a 512-token sequence limit. The observers performed no inference or
database operations. This is nomination-only retrieval, without semantic or
completion authority; it is not whole-function formalization.

The [retrieval distinction](retrieval-mode-observation-10.json) binds actual
neural retrieval separately from Source384 checkpoint inference. The
[indexed-path observation](indexed-path-observation-10.json) confirms four
symbols and 11 capsules hydrated in **17.40 seconds**. Fresh Source384 inference
consumed checkpoint `2ca38dfcc05536315fc3e2c0647b710b930ef4066b474061a7b4e5bfb9a258c5`
in **9.74 seconds**, with no training or neural replay. Admitted context reused
the same index in **3.71 seconds**, selecting three worker capsules / 29,875
semantic bytes with no new embedding calls. Source384 covered one declared
Python program file; task-data semantics and the XML numerical behavior remain
outside the qualified symbolic operator. Doctor returned a residual and no
repair/proof candidate. No parallel execution or whole-program proof is claimed.

Planning independently admitted two goals and one task. The native coding
session returned normally after **534.12 seconds**, under the recorded
600-second timeout. Its final envelope reported **1,191,013 tokens**: 123,145
uncached input, 1,035,520 cached input and 32,348 output, including 20,345 reasoning
tokens. Planning reported **21,719 tokens**: 12,946 input, 1,920 cached input and
6,853 output, including 5,011 reasoning tokens. The combined **1,212,732** count
already includes cached input; reasoning is included within output. Harbor's
input count of 1,173,531 includes cached input and must not be added to the
separate cache count. Both native sessions have final receipts, but complete
usage coverage and dollar billing are still unverified.

The [tenth tool observation](native-tools-grok-tune-mjcf-10.json) recorded the
separate zero-tool planning and six-tool coding exposures. Its last successful
bounded snapshot contained 23 tool outcomes: 21 successes and two failures.
These are observed completed outcomes, not a guarantee of exhaustive tool
coverage. The [resource observation](resource-observation-10.json) verifies
actual 20 GiB Docker/cgroup memory limits, unlimited `memory.high` and five CPUs.
The [preparation/observer review](trial10-preparation-observer-independent-review-01.json)
keeps exact archive/configuration/resource bindings separate from task correctness.

Native START succeeded, but no successful STOP receipt was retained and
`runtime.close()` raised `RuntimeError`. The driver reported `task_completed=false`
despite the native completed task and passing external verifier; worker cleanup
returned zero. The [closure observation](trial-closure-observation-10.json)
confirms the exact owned container was subsequently absent and all source pins
remained clean and unchanged. Container teardown does not establish native custody
closure. The [source-bound diagnosis](closure-failure-diagnosis-10.json) identifies
the guard refusing custody release while live launched children remained. The
original STOP exception was not retained; the close failure could mask it in
the existing `finally` structure, so that underlying cause remains unknown.

The [automatic native monitor](automatic-native-monitor-review-10.json)
observed daemon replacement and a supervisor parent change near shutdown, but
not their cause or parent exit status. It recorded no OOM kills within its
polling coverage. All three observer children exited zero, which does not make
native shutdown successful. The [independent result review](trial10-independent-publication-review-01.json)
rechecks the original verifier reward, source-bound close error, token arithmetic,
actual learned evidence and unresolved contracts. The subsequent lifecycle correction and its separate qualification are described
below; this historical trial is not evidence for that later cleanup fix.

Trial10 changes retrieval from lexical trial08, and earlier trials also differ
in source fixes and resource/time allowances. The passing task therefore does
not isolate the value of learned retrieval, symbolic reasoning or a larger
budget. A matched baseline/ablation and broader task coverage remain necessary
to measure efficiency advantages.


## Separate shutdown correction and qualification

After trial10, a provider-free signed/native completion fixture reproduced a
specific STOP defect at `b966623d3`: reading the original START transaction
re-ran its mutable source-context validation after publication changed that
source. The clean baseline failed with `StaleTreeError`. This establishes the
reproduced mechanism, not the unretained original STOP exception from trial10.
The fixture used an explicitly authored source-hash validator; it made no
checkpoint-inference or model-call claim. The
[development review](stop-custody-development-review-01.json) preserves the
failure and its independently authorized cleanup.

The correction at `2fd5bd6728bd27e0523f460d43cf31c738170e49` reads the exact
locally issued START transaction under a freshly issued STOP request. Target,
bounds, authorization, live lease and immutable coordinates are checked under
the existing service/store locks, followed by an exact transaction-identity
join. This read grants no repair authority. An interrupted START still requires
its original permit, currentness checks and recovery revision; normal STOP and
the live-child custody guard are unchanged. The
[independent implementation review](shutdown-independent-implementation-review-01.json)
checks foreign-request/transaction rejection, revoked profile, missing grant,
refused lease, expired replay budget and uncertain repair-state controls.

A separate diagnostic change at `66977152816dc4007bc720f107fa7a8cc1555268`
retains bounded STOP and runtime-close failure slots. It identifies whether the
failure occurred in STOP invocation, response serialization, process observation,
context refresh or runtime closure. Only reviewed type/reason names and exact
installed source-frame names/line numbers are retained; messages and source
bodies are excluded. Original exception propagation and cleanup remain unchanged,
including cancellation and diagnostic-collection failures. The
[120-case exporter review](fresh-live-inspector-safety-review-13.json) binds the
production validator to the exact selected source and preserves completed
trial01–10 summaries byte-for-byte.

The [shutdown development history](qualification-summary-shutdown-development.json)
contains **56 separate run rows**, preserving the prior 50. The new rows are
the failing one-case baseline; **13 passed / 1 failed**, then **7 passed** for
the custody change; and **39 passed**, **79 passed / 1 failed**, then **80 passed**
for shutdown diagnostics. The two development failures were test-authoring
errors and remain recorded. Each run's source stayed unchanged, but development
runs remain marked dirty. The [custody commit binding](stop-custody-commit-binding-01.json)
and independent review bind final tested bytes to their commits without
relabelling those runs as clean-source qualifications. Counts overlap and are
not summed. Both changes are combined at `0e2d9a8c62eabdfbd42fcad505c370365474514a`.

The [clean combined shutdown gate](shutdown-final-qualification-review-01.json)
passed **120 tests** across seven selected suites, with zero failures, errors or
skips, in **304.55 seconds** of recorded wall time (pytest reported 301.85 seconds).
Source pins remained clean and unchanged. It includes completed-context native
STOP, authorization/transaction controls, interrupted-START repair, admitted
runtime behavior and driver diagnostics. This is separate from the 120 synthetic
artifact-export checks above. The [cumulative final summary](qualification-summary-shutdown-final.json)
retains **57 separate runs**, preserving the preceding 56 exactly; overlapping
test counts are not summed. These tests made no paid provider calls and do not
establish a whole-repository pass. The later trial11 used this source and a
newly reviewed archive; its separate outcome follows. Neither qualification
nor the later run retroactively changes trial10's failed shutdown.

## Trial11: provider timeout with clean process shutdown

The [preparation review](trial11-preparation-independent-review-01.json) binds
14 critical source files, the new archive, exact task inputs and the same
Grok 4.7 / CLI 1.0.46 profile. It retains the 20 GiB / 5 CPU limits, 180-second
planning cap and 600-second coding cap under the 900-second driver allowance,
840-second work deadline and 60-second cleanup allocation. Harbor setup and
verification have separate timings. The [resource observation](resource-observation-11.json)
verified the actual container limits, including unlimited `memory.high`.

The [learned runtime observation](learned-vector-runtime-observation-11.json)
records pinned GTE-small revision `17e1f347d17fe144873b1201da91788898c639cd`
on CPU, 384 dimensions, three local model calls over eight texts, a passing
canary and DuckLake metadata projection. Embedding inputs were qualified symbol
names only. The [indexed path](indexed-path-observation-11.json) built four
symbols and eleven capsules in **18.19 seconds**, including fresh Source384
checkpoint inference in **10.60 seconds**. The admitted context reused three
capsules and 29,875 semantic bytes in **3.83 seconds**, with no new embedding
calls. The [retrieval distinction](retrieval-mode-observation-11.json) preserves
separate retrieval and checkpoint evidence; no training, full-program semantic
coverage or proof authority is claimed.

Planning independently admitted two goals and one task in **79.42 seconds**
and reported **22,806 native tokens**. Coding timed out after **600.25 seconds**;
its final usage envelope was absent. The [original verifier result](live-trial-summary-11.json)
was **0.0**, while the native task remained in progress at revision three. The
last successful bounded tool snapshot contained 32 successful and one failed
tool outcome. These are observed outcomes, not task-completion evidence or a
complete billing ledger. The driver ultimately exhausted its overall work
budget; total recorded invocation time was **1,001.49 seconds**, including
external setup and verification stages.

The [timeout and settlement observation](timeout-settlement-observation-11.json)
records `started_outcome_unknown` and no native-exit receipt. Retained diagnostics
do not distinguish which custody, lifecycle or state join withheld the receipt.
The callback remains unsettled; its claim is not released for an automatic retry.
The [automatic native monitor](automatic-native-monitor-review-11.json) observed
no OOM, memory-max or memory-high events within its polling coverage; that does
not diagnose the provider timeout. A restart counter change without a newly
observed daemon birth is not treated as proof of a daemon replacement.

The [closure observation](trial-closure-observation-11.json) independently
records successful START, STOP and `runtime.close()`, zero remaining tracked
processes, worker cleanup return code zero and the exact container absent.
All three observer children exited zero, and no bounded shutdown failure was
reported. Its historical `full_lifecycle_qualified` field is restricted by the
timeout supplement to START/STOP, tracked-tree absence and runtime closure;
it does **not** qualify task success or callback settlement. The task did not
complete, so post-STOP context refresh was ineligible and live STOP after a
successful task was not exercised. The [independent publication review](trial11-independent-publication-review-01.json)
rechecks verifier reward, hashes, source-bound timeout, usage uncertainty,
learned evidence and this narrower cleanup conclusion.

## Concurrent merge qualification

After trial11 closed, concurrent native Codex structured-planning support and
contextual decoder documentation were merged into clean source
`0a8a1b3e8448a6604899bedcbc5175dc08bf6825`. The
[independent merged-source review](independent-origin-codex-merge-review-01.json)
and recorded gate passed **213 tests** across nine router, schema and worker
capability suites, with zero failures, errors or skips, in **89.16 seconds**.
Source pins stayed unchanged. Grok task-contract admission, closed diagnostic
boundaries, worker600 support and the legacy300 fallback remain covered.

The [final cumulative qualification summary](qualification-summary-origin-codex-merge-final.json)
adds this as row 58 and preserves the previous 57 exactly. The 213-test merge
gate overlaps prior coverage and is separate from the 120-test native shutdown
gate. It made no provider calls and did not rerun trial11; historical trial11
remains bound to `0e2d9a8c6`. No whole-repository pass, matched efficiency
advantage, or task pass followed by clean live shutdown is claimed.

## Evidence boundaries

Published exports contain counts, closed diagnostics, hashes and source
bindings. Raw model responses, hidden verifier or solution bodies, credentials
and runtime databases remain outside this export. The included qualification
and live-result producers require retained local artifacts. Their synthetic
checks validate metadata handling, not benchmark correctness or formal proof.
