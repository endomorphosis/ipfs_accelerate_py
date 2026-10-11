# Durable provider cleanup: current status and remaining integration

Audit date: 2026-10-11. STOP integration starts from supervisor revision
`07eed982c7150406480eefdf0c57cdd2f772b671`.

## Source assertion requirements, 2026-10-11

Datasets `7286ee17f` now exposes numerical source assertions as pending
requirements linked to their exact ProgramIR commands, predicate expressions,
source spans and versioned assignment definitions. Tuple component paths and
right-hand-side dependencies survive rebinding. The records retain predicate
polarity, argument order, keyword names and unknown effects. Unsupported control
flow or statements stop extraction; a deferred assertion failure message permits
predicate extraction while the function remains partially translated.

These records describe a local assertion if execution reaches it with assertions
enabled. They establish neither execution nor truth, and do not become global or
callee postconditions. Library identities, NumPy comparison defaults and runtime
truthiness remain unresolved. The shared ProgramIR and first-order projections
retain these requirements, and Lean emits definitions with explicit interpretation
parameters. This change performs no model training and does not expand learned
Source384 coverage.

Supervisor `a5d7a6322` persists the records through its world and metadata lakes,
replays their source bindings, and supplies them before goal decomposition. The
first integration test exposed the planner's six-level nesting limit. The
consumer now flattens operand lists and refers to full assignment records by
identity, preserving the existing limit. An exact request-payload check verifies
the assertion summary survives planning and admitted-context reuse.

On the public `largest-eigenval` sources, all four assertions and both compound
contracts fit in a **7,804-byte** planner summary. Both native lakes reopened and
source replay passed. The full index contains **45 pending obligations, zero
discharged**; the bounded summary retains eight obligation rows and accounts for
37 omitted rows. The assertions preserve the actual not-allclose-to-zero,
not-NaN, not-Inf and allclose-residual checks. They do not invent a dominance test,
vector-finiteness check, tolerance or norm threshold.

**202 datasets tests and 20 supervisor context tests passed**, including native
Lean positive/negative tests, source-replay tampering, omission accounting and
exact planner delivery. Both generated public-source Lean modules also compiled.
These checks qualify formalization and integration, not Python correctness or
benchmark success. The separate streaming benchmark uses the previously frozen
archive and therefore does not contain this assertion extension.

The [datasets qualification](evidence/numerical-assertions-20261011/datasets-qualification.json),
[consumer test history](evidence/numerical-assertions-20261011/consumer-tests.json)
and [public-source context qualification](evidence/numerical-assertions-20261011/public-context.json)
retain source hashes, test scope, exact summary contents and pending-proof limits.

## Coding progress and numerical value contracts, 2026-10-11

Supervisor `bfda2ab7e079` enables bounded `streaming-json` observations for Grok
coding through the shared `llm_router`; planning keeps its existing JSON/schema
transport. The collector independently records final events, direct process exit,
and stdout/stderr EOF. It distinguishes a running-process timeout from an exited
process whose descendant still holds a pipe. An `end_turn` event cannot override
the deadline. Malformed, truncated, ambiguous, cancelled or incomplete responses
cannot be returned as successful text. The returned text is the last segment
delimited by native per-response usage events; final accounting uses only the
terminal envelope, never a sum of intermediate usage records.

Progress contains fixed event/tool/status buckets, byte counts, monotonic timing
offsets and closed outcome fields. It exports no model text, tool arguments,
arbitrary labels or stderr contents. Event origin remains unauthenticated.
Callbacks are throttled to five seconds plus a final observation; cancellation
still propagates through owned-child cleanup. The runner binds advisory records
to its invocation and preserves the final snapshot in the benchmark receipt and
comparison. Progress grants no execution, proof, completion, retry or settlement
authority. Archive compatibility now also binds the shared router and capture
producers. An additional router regression exposed and fixed explicit command
templates being replaced by an installed native CLI during discovery.

Datasets `fecc4ecbf2b6` adds a conservative ProgramIR contract connecting the
tuple returned by `numpy.linalg.eig`, the index from `argmax(abs(values))`, and
the returned value and matching vector column. Unsupported control flow,
rebindings, mutation or ambiguous value relationships cause abstention. Seven
pending requirements cover runtime interpretation, dimensions, value/column
correspondence, magnitude selection, nonzero output, finiteness and approximate
residuals. No floating-point tolerance or NumPy refinement is assumed.

The supervisor preserves bounded compound contracts and their source roles in
the planner summary, with complete omission counts, through the existing native
world/metadata DuckLake stores and deterministic replay. Both public numerical
modules passed actual Lean 4.34.1 `lake build`: two matched contracts, four sampled
functions (two structurally complete and two partial), **41 pending source
obligations and zero discharged**. Lean checks definitions and explicit
conditional mathematics; it does not prove the Python/NumPy implementation.
This extension performs no model inference or training and does not change the
previously unsupported Source384 candidates.

A subsequent provider-free full-supervisor preflight qualified this same archive
in the actual task environment: **218.21 seconds** inside the driver, 230.64
seconds including the recorder. Source384 actually loaded its pinned checkpoint
once and decoded four candidates, all still unsupported by its scalar source
contract. Numerical hydration retained one world record and three catalog/link
pairs across the native lakes; the admitted summary exactly matched the initial
summary, including both compound contracts and the count of 41 pending obligations.
Native START and STOP succeeded, heartbeat reads advanced from 24 to 31, no native
or worker processes remained, and the exact container was absent twice. This
preflight made zero provider calls and grants no live planner/worker-delivery,
task-completion, protected nested-Docker or cold-owner recovery qualification.

A fresh archive (`10f8e35960883b9dd0fdf3d96161960818ab67bf9cb0bb21c5e0c950059bd3c1`)
passed the authored Docker router diagnostic: one Grok 4.7 call, four turns,
**15.50 seconds** in the provider invocation, and five independent output checks
passed. The full diagnostic including setup took 332.31 seconds. Its 88 stream
frames included 61 text events, four per-response usage events, three tool calls
and one terminal event. No frames were malformed, oversized or unknown. One tool
name fell into the unknown bucket; the effective tool allowlist remains
unverified. The final event arrived at 11.59 seconds and process exit at 15.33
seconds; both pipes reached EOF. This directly confirms that a terminal model
event and process exit are distinct observations, without diagnosing the older
30-minute failures. Five initial/periodic/final observations reached the callback.

The terminal envelope reported 23,008 tokens (12,734 input, 9,856 cached input,
418 output). Usage completeness was not reported and billing was not verified;
these are observed counters, not a complete cost or token-savings result. The
protected checker remained unchanged; the exact owned container was absent in
two subsequent observations. All source, archive, helper and run-spec bindings
were unchanged. This diagnostic uses the router and isolated worker boundary,
not the full supervisor planning lifecycle or the official Terminal Bench
verifier. The previous benchmark scores remain unchanged.

The [closed qualification](evidence/coding-progress-20261011/qualification.json)
joins the frozen source/archive identities, live diagnostic, cleanup and local
checks. The [test census](evidence/coding-progress-20261011/tests.json) contains
747 unique passing JUnit identities after targeted reruns resolved all four
initial failures. It distinguishes repeated executions and the source-hash
evidence available for each run; this is not one monolithic final-HEAD test run.
The parser's separate 63 passing cases are excluded from that deduplicated count
because only its terminal summary and after-test hashes were retained. The
[numerical qualification](evidence/coding-progress-20261011/numerical.json) and
[independent review](evidence/coding-progress-20261011/numerical-independent-review.json)
retain the two actual public-source Lake results and their pending-proof scope.
The [independent live diagnostic audit](evidence/coding-progress-20261011/tiny-stream-independent-review.json)
rechecks the closed stream, source/archive bindings, five checks and cleanup.
The [independent full-supervisor preflight audit](evidence/coding-progress-20261011/full-preflight-independent-review.json)
verifies checkpoint inference, context persistence, native lifecycle and cleanup,
while retaining the unsupported candidates and pending obligations.

## Long-budget Docker pilot and planner delivery, 2026-10-11

A fresh archive at supervisor `f9dcf5f80121`, datasets `db432c5a93a2`, and kit
`8ac00bb1974b` qualified in the actual task container before model calls. The
provider-free check took 473.10 seconds inside the driver. It exercised checkpoint
inference, numerical grammar, both DuckLake stores, and native START/STOP; the
owner heartbeat advanced and cleanup left no native or worker processes.
The archive SHA-256 is
`1d71fa6129b09cbac4ecee6daba357af589c2d626cc4892358a6c96ca461049a`.

The preflight now accepts the numerical profile and replays its source, producer,
artifact, and catalog bindings immediately before START. The live observer binds
the exact numerical summary to the initial native planner request submitted to
`llm_router`, its hash, and the returned invocation receipt. This verifies the
request boundary; it does not assert byte equality with a provider-transformed
prompt or delivery to the coding worker. No-index checks absence in its actual
planner request. Focused preflight tests passed 49/49; observer qualification
passed 111 unique tests. Separate pair, preflight-observer, and result-reader
helper suites passed 26, 22, and 18 tests respectively; no combined unique count
is claimed across these suites.

Both arms selected `source384-5cpu-20gib-formalization@1` and
`python-numerical-structure@1`, using Grok 4.7 through `llm_router`. Full enabled
lexical retrieval, Source384 inference, and numerical structure; no-index disabled
them. This was one fresh attempt per arm on `largest-eigenval`.

| Supervisor arm | Official reward | Agent execution (s) | Agent setup (s) | Total tokens |
| --- | ---: | ---: | ---: | --- |
| Full: lexical indexes + Source384 + numerical structure | 0 | 1,964.95 | 227.83 | Unknown |
| No indexes | 0 | 1,898.31 | 167.67 | Unknown |

Both coding calls reached their 1,800-second limit with `TimeoutExpired`.
Neither Harbor run raised an outer exception. START and STOP succeeded, no native
processes remained, worker cleanup returned 0, and each exact owned container
was observed absent twice. Original benchmark inputs and frozen source checkouts
were unchanged. Actual container observations confirmed 5 CPUs and 20 GiB;
other declared controls were compared as configuration. The full planner request
contained the exact numerical summary, while no-index verified its absence and
had no indexed context or Doctor dispatch. Full initial context took 14.55 seconds;
the observed failures occurred during coding, after preparation.

Planning usage is only a partial observation: full reported 24,738 tokens and
no-index reported 18,055. Full coding timed out without a usage envelope.
No-index retained a native envelope reporting 52 turns and 4,173,308 tokens:
562,655 input, 3,499,264 cached input, and 111,389 output. Its process nevertheless
timed out, `task_complete_observed` was false, and usage completeness was absent.
These observations establish model activity but not why the process failed to
finish. Aggregate tokens and costs remain unknown; these are not token-savings
results.

The full numerical context contained four functions, two structurally complete
and two partial, with 27 pending obligations and zero discharged. Hydration
retained one world record and three catalog/link pairs. Source384 loaded its
pinned checkpoint, but all four learned candidates remained unsupported by its
scalar source contract. Retrieval remained lexical. No native Codex baseline was
rerun, and this pilot does not establish an advantage for the complete indexed
system. Unrelated task containers were observed during full setup and early
execution; none were observed at no-index setup. Host contention was not controlled.

The closed full artifacts do not retain candidate output hashes or a staged patch
after timeout. Unchanged original benchmark inputs are a separate observation
and cannot establish whether Grok edited its private worktree. Its JSON output
mode also supplies no intermediate tool or turn counters. The next diagnostic
must retain bounded output observations and provider progress before drawing a
conclusion about why coding used the full allowance.

The [closed pilot qualification](evidence/long-budget-pilot-20261011/qualification.json)
records source and archive identities, both results, numerical request joins,
preflight coverage, cleanup, and limitations. An
[independent paired audit](evidence/long-budget-pilot-20261011/independent-pair-audit.json)
rechecked the exact receipts and retained coding outcome metadata.

A subsequent observation fix at supervisor `16d687bd6828` now records bounded
hash/size/existence facts for admitted output paths in the owner-bound candidate
and canonical checkout after settlement and cleanup, before container removal.
Native task/attempt/baseline bindings select the allocation. Unsafe, missing or
unstable evidence remains unavailable; diagnostic import and timeout failures
cannot suppress the primary result. Existing archive gates bind its producers,
and the comparison exports a bounded projection. **256 unique tests passed**,
including final observer and comparison reruns; the
[separate qualification](evidence/long-budget-pilot-20261011/output-observer-qualification.json)
records source and test-artifact hashes and the qualification runs. This fix was not used in the pair
above and cannot recover its deleted candidate worktrees. The observations do not
establish writer identity, original allocation inode, publication, or correctness.

A subsequent [authored router diagnostic](evidence/long-budget-pilot-20261011/tiny-router-diagnostic.json)
used the frozen archive, Grok 4.7 through `llm_router`, the same coding tool policy,
and the isolated UID 1001 boundary, with a four-turn/90-second cap. Grok edited
the tiny function and returned normally in **16.47 seconds**. A separate protected
checker verified the accepted AST and passed five cases; it did not observe
whether the model ran its own requested check. The entire diagnostic took
153.94 seconds, including setup and successful container cleanup. This was one
provider call and did not exercise full supervisor planning or the benchmark.
The initial diagnostic attempt made zero provider calls: its host environment
lacked `multiformats`. Reusing the qualified campaign's explicit dependency paths
fixed that setup issue; 26 helper tests passed before the fresh successful run.
The tiny success establishes basic tool-based editing on this route, while the
benchmark's long rollout and pending numerical semantics remain unresolved.

## Reserved preparation budget and numerical structure, 2026-10-11

The explicit `source384-5cpu-20gib-formalization@1` resource profile raises the
previous coding allowance and gives preparation a separate cumulative budget.
Select it for both comparison arms; older profiles retain their recorded limits.
Preparation includes initial indexing/formalization and post-plan context/Doctor
work. Planning pauses that allowance; it cannot refill it or spend the coding
window. The previous 300-second coding cap already began after preparation;
the separate outer 900-second deadline also charged preparation.

| Phase | Allowance |
| --- | ---: |
| Preparation and formalization, cumulative | 3,600 s |
| Planning, including a 600 s provider call | 660 s |
| Native construction/START | 180 s |
| Coding and settlement, including a 1,800 s provider call | 1,860 s |
| Cleanup | 120 s |
| Outer driver / exec / Harbor | 6,420 / 6,450 / 6,480 s |

The native owner starts after preparation and has a separately signed 2,160 s
lifetime. The named profile reaches the Source384 worker and coding worker;
historical default ceilings and bounded post-STOP refresh remain unchanged.
This is a longer finite experiment, not an unlimited process lifetime.
The implementation window starts after START and includes the worker's final
context/prompt checks. Its 60 s headroom is shared by those checks and provider
settlement; arbitrarily slow worker checks are not granted additional time.

The separate opt-in `--numerical-formalization-profile` value
`python-numerical-structure@1` connects declared public Python functions to the
datasets library's numerical frontend and shared logic modules. Tuple assignment,
qualified calls, indexing/slicing, and tuple returns produce native ProgramIR.
The existing syntax bridge, many-sorted first-order kernel and SMT compiler emit
source-bound pending obligations; Lean contains definitions and conditional
finite-index lemmas. TLA+/transition, temporal, separation, concurrency,
refinement, DCEC, TDFOL and frame semantics remain explicitly unsupported without
an appropriate source-grounded interpretation. NumPy call behavior is opaque.

The supervisor persists those reports, pending-obligation summaries and world
records through its native DuckDB/DuckLake owners, then reopens and verifies both
lakes. It checks exact declared source bytes, producer hashes, report replay and
artifact inventories before supplying a bounded summary to the initial planner.
The admitted context retains the same receipt. This supplies the planner before
goal generation; direct delivery of this new numerical report to the coding
worker is not claimed. No-index records the selection as disabled. Explicit
selection requires matching archived producers and matched comparison controls.

An inference-only check of public `largest-eigenval/environment/src/eigen.py`
and `eval.py` produced four function records: the target and `ref_solution` are
structurally complete, while `test_eigen_pair` and `test_speedup` retain explicit
unsupported diagnostics. There are **27 pending obligations, zero discharged**.
The native lake replay found one world record and three matching catalog/link
pairs. No benchmark verifier inputs, model calls, training or source execution
were used for that check. Separate authored Lean qualification passed actual
`lake build`; removing the nonempty premise from an index lemma correctly failed.

The numerical path is a deterministic grammar extension. It does not improve the
Source384 checkpoint's learned coverage or prove NumPy semantics. This initial
qualification did not measure task success or token usage; the separate
long-budget pilot above records subsequent matched trials.
The [closed qualification receipt](evidence/numerical-budget-20261011/qualification.json)
records the tested source revisions, test counts, public-source coverage, native
Lake input/tool hashes, and the distinction between compilation and source proofs.
Final integrated supervisor checks passed **114/114**, with all 33,825 tracked
Python files across the three selected checkouts unchanged during the run.
Separate budget regressions passed **235/235**. Datasets qualification passed
**286 unique tests**: 285 in its development environment and the single optional
Z3 check using the already-installed host backend. Both actual public-source
Lean artifacts compiled with Lake 4.34.1. A real timer regression also reproduced
and fixed suppressed phase expiry, while preserving legacy timeout behavior.

## Source, lifecycle, and Grok catalog qualification, 2026-10-11

Three targeted fixes passed local validation. The datasets source adapter now
accepts Unicode in comments while preserving executable-source
restrictions; **194 tests passed** at datasets `5d5769f0b9`. Six authored supported
source examples also compiled with native Lean/Lake. Compilation establishes the
bounded generated definitions' syntax, not Python equivalence, security
correctness, or proof authority. Supervisor `0d886f8bc` validates status against
the current PID and process birth even when optional startup grace is disabled;
**91 health tests passed**, including seven previously reproduced failures.
Those health tests used datasets `5c69724316`; the subsequent Docker preflight
used datasets `5d5769f0b9`.

A fresh bundle was required after an exact worker-capability check correctly
rejected the older archive with newer host runner bytes. The replacement archive
`8ef6baa3441b0f4ea9a3a58f1ad50d112a10cc9139be587fcc513acfcd24a115`
contains supervisor `0d886f8bc`, datasets `5d5769f0b9`, and kit `8ac00bb197`.
Using the existing 20 GiB profile, the provider-free Harbor task-container
preflight qualified again: **531.17 seconds** including recorder overhead,
519.13 seconds inside the driver, and **29.12 seconds** for native START/STOP.
The isolated UID 1001 worker received matching task, retrieval, semantic and
world references in its 60,012-byte prompt. Two healthy observations advanced
the same owner's sequence from 26 to 33. STOP left no residual native or worker
processes, and the exact owned container was observed absent twice. An independent
review checked 86 evidence joins. This remains an authored public task graph,
with no model-generated planning or official verifier in the preflight.

Actual Source384 inference loaded the pinned checkpoint once, but **all four
sampled functions remain unsupported**: three fail the two-parameter source
shape requirement and one still fails the executable-source encoding guard.
There were no solver calls, verified formalizations, source edits, training, or
model promotion. The Doctor abstained, and Intent advice remained
`fail_open_no_checkpoint`. The six authored Lake examples do not change this
coverage result for the benchmark's NumPy implementation.

At this earlier qualification, a numerical structural profile was proposed for
the public target and `ref_solution`. That profile and its pending obligations
are now implemented separately above. The existing scalar source binder and
native Lean path remain unchanged pending reviewed numerical semantics. This
earlier qualification establishes no numerical proofs.

The next full trial stopped during setup because its native `grok models`
output did not match the earlier parser's bare-header grammar. That trial has
**no reward and no verifier result**, rather than a zero score; it made no text
generation calls. Its no-index partner remains unexecuted. Isolated metadata
probes then confirmed the pinned CLI's exact public login preamble, declared
default model, and four-model catalog. Supervisor `980b506fa` accepts that
specific layout only with ordered fields and a matching starred default row,
retains the bare-header grammar, and rejects contradictory or unknown output.
Expanded authentication refusals still override a positive-looking catalog.
**111 readiness, setup, and deployment tests passed** with no skips.

The revised parser also returned `ready` against an actual isolated credential
copy: native exit 0, 150 stdout bytes, no stderr, four catalog models including
Grok 4.7, and no text generation calls. Its disposable container was removed and
observed absent twice. Raw native output and credential contents were not retained
in the published evidence. This catalog observation grants no authentication or
subsequent-call authority. The parser is supplied by the newer host as a reviewed
standalone script; the runtime archive remains at `0d886f8bc`, and its exact worker
capability still matches the host at `980b506fa`.

Fresh campaign03 full and no-index preparations passed strict control comparison
under the newer host and unchanged archive. Both trials completed with official
reward **0**, no Harbor exception, and a coding `TimeoutExpired` at the 300-second
cap. This is one attempt per arm on `largest-eigenval`, using Grok 4.7 through
`llm_router`:

| Supervisor arm | Official reward | Agent execution (s) | Agent setup (s) | Total tokens |
| --- | ---: | ---: | ---: | --- |
| Full: lexical indexes + Source384 | 0 | 466.50 | 278.66 | Unknown |
| No indexes | 0 | 422.32 | 345.70 | Unknown |

Planning took 89.27 seconds for full and 104.42 seconds for no-index; outer
recorders took 772.83 and 793.39 seconds respectively. Both retained unchanged
original task inputs and frozen host sources. Their supervisors reported
`task_completed=false`, authoritative START and STOP succeeded, no native
processes remained, and worker cleanup returned 0. Each exact owned container
was observed absent twice. The final control comparison matches declared
configuration; separate live container observations confirm 5 CPUs and 20 GiB
for each arm. Those observations do not establish enforcement of every declared
control. No-index used the same archive with Source384 disabled for the ablation
and no indexed context, Doctor advice, or semantic translation.

The live context audit verified four indexed symbols, eight capsules, the four
unsupported Source384 candidates supplied before planning, and delivery of
retrieval, world, semantic, and Doctor residual context to the coding invocation.
Identifier translation covered 19 mappings and 50 occurrences. Its recorded
prompt sizes are byte counts for the router task prompt, excluding provider
harness/system context; they do not establish token savings or task correctness.
The measured profile used **lexical retrieval**: the archive has no learned
retrieval requirements or retrieval-model revision. Source384 separately consumed
its pinned GTE embeddings and neural checkpoint. This trial therefore does not
qualify a neural-retrieval or all-features configuration.

A future learned-retrieval trial needs a new archive with `--model-snapshot` set
to the pinned GTE snapshot and the Source384 configuration, followed by fresh
preflight and matched preparations. The current learned retrieval implementation
embeds symbol names on CPU; body-aware retrieval requires separate work. The live
trial also exposed a telemetry conversion issue: final progress summaries showed
`unrecognized` for native lifecycle enums while the authoritative START and STOP
receipts reported `succeeded`. A fourth fix at supervisor `212c23ec9`, created
after the trial sources were frozen, normalizes only the exact native lifecycle
enum for future progress reporting. **83 tests passed**, including the two
reproduced projection failures. Retained trial observations and the measured
host at `980b506fa` remain unchanged; this telemetry repair grants no lifecycle
authority and does not establish a cause or fix for the coding timeout.

Each arm has two receipted provider calls, but coding usage is unavailable, so
**aggregate tokens and cost remain unknown**. Planning alone reported 22,183
native tokens for full (13,277 input, 1,536 cached input, 7,370 output) and 18,464
for no-index (8,682 input, 1,920 cached input, 7,862 output). These partial counts
must not be used as trial totals or to claim token savings.

This single matched pair establishes no performance advantage. Native Codex was
not rerun, source proofs remain unestablished, and the full suite is incomplete.
A next bounded pair can use the existing
`source384-5cpu-20gib-coding600@1` profile with fresh matched preparations to test
the longer coding budget; no such run is reported here. Learned retrieval still
requires the separate new archive and preflight described above. The
[bounded October 11 evidence](evidence/supervisor-recovery-20261011/qualification.json)
separates live trial outcomes from the completed provider-free preflight and
catalog checks. The protected nested-Docker and cold-owner recovery gaps below
remain unchanged.

Publication preserves the concurrent census work merged from `origin/main`.
At combined supervisor `1d832b2d9`, datasets `9e5df0d19`, and kit `8ac00bb197`,
**332 regression tests passed with no failures or skips** across native
progress, Grok readiness/deployment, and census/dispatch interactions.
Tracked Python source hashes remained unchanged. This post-merge validation
does not replace the frozen source identities of the measured trials above.

## Task-container qualification and live pilot, 2026-10-07

The separate Harbor task-container route passed a real provider-free preflight on
`largest-eigenval`. The admitted supervisor started, delivered code-retrieval,
semantic and world-model context to the isolated UID 1001 worker, observed two
advancing healthy heartbeats, and stopped with no residual native or worker
processes. The owned container was independently observed absent twice. The
native phase took 28.07 seconds. This used the existing
`source384-5cpu-20gib-planner180@1` profile and archive
`fdd8511b4b7619592fd2cb9ba720adc022139bd88a344a760250e0b69b9d04a6`,
bound to supervisor `89d1c4d5a`, datasets `5c69724316`, and kit `8ac00bb197`.
The public task graph was authored for qualification; no model-generated plan or
benchmark success is established by that preflight.

Two earlier preflights remain recorded as failures. A controlled scheduler
reproduction exposed an oversized waiting root imposing a shared cooldown on
already admitted child work. The fix preserves the root refusal, actual pressure
checks and child capacity limits while allowing sufficiently reserved work to
progress. The next container run retained its actual admission diagnostic:
with no active root reservation, 9,400 MiB available could not cover the
6,144 MiB request plus 3,277 MiB required headroom. Selecting the existing 20 GiB
profile resolved the subsequent qualification; no safety threshold was lowered.
The first container's exact failed gate was not retained and remains unknown.

Source384 loaded the pinned checkpoint once in the qualified preflight and ran
inference over four functions. All four candidates remained unsupported: three
required two distinct parameters, and one failed the source-encoding guard.
These observations establish checkpoint consumption, not correct NumPy/loop
formalization, verified source semantics, or a symbolic repair.

The live full indexed Grok trial then reached `llm_router` planning, but the native
provider refused authentication. The original verifier recorded reward **0**;
token and cost totals remain **unknown**. The matching no-index arm was prepared
with identical comparison controls but was not executed after that refusal.
There is no new matched performance result, token-savings claim, native Codex
comparison, or complete-suite score. At the October 7 cutoff, the local pinned CLI
also reported unauthenticated and required a refreshed sign-in. The October 11
isolated catalog observations above supersede that readiness observation, without
establishing authentication for subsequent model calls.

Subsequent host harness changes sort only Harbor's two set-valued retry filters
before sealing new configurations, preserve historical control replay, and check
the isolated Grok credential copy before indexing or planning. The bounded
`models` probe emits only a closed status receipt, rejects exit-zero negative
authentication and unknown output formats, and can run against retained archives
through its reviewed standalone script. It may refresh or invalidate the
container's copied credential. At the October 7 cutoff, its positive catalog
parser had only fixture coverage. That coverage limitation is superseded by the
October 11 actual isolated catalog observation above; neither observation grants
authentication authority for a later model call.
Provider-free preflight defaults remain unchanged.

Final merged-source harness qualification passed **254 tests with no skips**;
the scheduler change passed **188 tests** after its datasets merge. Frozen runtime
qualification and later host changes are identified separately in the
[bounded recovery evidence](evidence/supervisor-recovery-20261007/qualification.json).
Raw runtime databases, signing keys, credential contents and native provider text
are excluded from that publication. The protected nested-Docker and cold-owner
limits below remain in effect.

## Working scope

The native provider-attempt CAS can record terminal cleanup intent and completion.
The restored runner protocol supplies the matching private bindings, socket
watchdog, exact container/process fences, resource retirement, and scoped cleanup
replay. These are prerequisites, not an enabled native launch profile.

`observe_agent_implementation_terminal_cleanup` independently verifies an existing
terminal CAS and its original signed authority. It can observe the exact disposed
workspace, but cannot create a state store, return new launch authority, settle a
task, or release a claim. The ordinary public effect APIs still require their
strict live workspace checks. The Portal diagnostic audit attaches this cleanup
observation only after its existing route/outcome checks. A companion capacity
receipt prevents successful candidate-cleanup projection.

Ordinary unsigned Grok/Codex container routes retain their separate compatibility
implementation. Their watchdog dispatch mode is excluded from implementation
worker counts. Duplicate identical `.git` mounts in standalone workspaces are
removed before command validation. Defaults remain Grok 4.7 and GPT-6.1-sol,
with route selection owned by `llm_router`.

The multi-supervisor process owner now retains a private cleanup directory before
launch and reuses its observer across STOP retries. It protects exactly bound
native watchdog and removal-issuer process births, fences ordinary descendants,
and requires retained completion authority, matching signed terminal CAS where
applicable, resource retirement, and independent Docker/kernel absence before
reporting cleanup complete. Missing or changed previously observed authority
keeps STOP unresolved. Observation does not issue container removal.

The lifecycle orchestrator has a separate process-only implementation. It now
refuses START, STOP, RESTART, and interrupted-START repair when cleanup custody is
present or unavailable, before signals or success publication. A nonterminal
refusal is persisted in its saga journal and survives record deletion and owner
recreation. This guard does not authorize cleanup. Plan-bound observations and
manager paths without a retained Popen also refuse visible cleanup custody.

## Disabled paths and benchmark blockers

These blockers concern the protected nested-Docker provider route. The existing
Harbor `FullSupervisorAgent` instead deploys the admitted runtime and isolated
worker inside the original task container, and routes model calls through
`llm_router`. That benchmark path can be qualified separately without enabling
the native lease factory or substituting the missing protected image.

Native lease creation and native internal watchdog/removal CLI entrypoints are
disabled before effects until all owners and signed launch contracts agree. There
is no environment override. Existing private protocol functions are exercised by
offline fixtures; their presence does not establish live lifecycle qualification.

1. **Owner recovery and admission:** cleanup observation is implemented for the
   same retained Popen owner. Cold-owner recovery does not yet prove historical
   custody after records disappear, and the lifecycle orchestrator has only a
   refusal guard, not an admitted cleanup owner. Plan-bound and no-Popen guards
   are conservative current-state observations, not durable recovery proofs.
   Existing historical namespaces also block successor launch until explicit
   recovery/retention handling is admitted. Unsigned compatibility watchdogs
   retain their previous behavior and are not upgraded to the native protocol.
   The [cold-owner recovery plan](cold_owner_cleanup_recovery_plan.md) maps the
   required independently admitted checkpoint, complete inventory, and exact
   pre-dispatch persistence hooks. A new local sidecar cannot supply that
   authority when the whole namespace or its history has disappeared.
2. **Signed launch agreement:** the current Codex builder and authority validator
   use a direct `/usr/bin/env` suffix. The native adapter expects a different
   gated launch, and its own positive command grammar is incompatible with that
   adapter. The caller also has no admitted `WorkerNetworkProfile`. Protected
   routes must not fall back to compatibility leases. Created-container recovery
   must not feed a raw prompt to a native start gate before recording a fence.
3. **Outer callback:** current `database-portal-callback-intent@1` lacks the
   immutable contract/tree/fingerprint inputs required by the retained candidate
   recovery protocol. The Bridge and daemon do not implement its complete joins.
   A terminal provider CAS is insufficient to settle this callback.
4. **Live fixture image:** the approved image
   `sha256:74c4a6ff67f397f8a10b058851d218896b2f1ee0f2cddf47741219b734de93a6`
   was absent locally at this audit. No substitute image was treated as approved.

## Required migration order

1. Extend the retained process-owner observation into admitted lifecycle and
   cold-owner recovery, including durable namespace history, successor admission,
   and retention. Preserve the new refusal latches and exact watchdog custody.
   Qualify runner/watchdog death, PID reuse, record replacement, delayed removal,
   and interrupted persistence across owner restart. Then qualify a reviewed
   disposable `/bin/sleep` container probe using the approved local image, with
   no provider calls. The historical probe needs migration before execution.
2. Version the signed native launch contract explicitly. Preserve the old command
   receipt's meaning; bind the exact start gate, image, environment, mounts and
   network policy in the new version. Acquire stdin custody, record the running
   fence, then release input. Test fresh launch, adoption and all terminal races.
   Enable factory/CLI dispatch only after the owner and route agree.
3. Introduce a future-only callback intent before dispatch, binding the exact
   task/attempt/claim/lease/session/fences, callback key, Bridge, contract digest,
   repository tree and immutable fingerprint. Do not synthesize these for old
   unknown rows.
4. Connect native observed lifecycle deletion and its prepared journal to the
   Portal preservation/finalization hooks. Reuse the journal's retained-inode,
   exact-index and directory persistence checks. Recovery may resume a prepared
   operation; it may not infer deletion from missing rows. Preserve current
   Doctor/shutdown wrappers and exact-record guards.
5. Add Bridge verification of the hash-linked event prefix, immutable attempt
   binding, exact rescue commit and current cleanup CAS. Only an admitted outer
   daemon may resume mutation under heartbeat. CAS the exact original callback
   into a distinct rejected-candidate receipt; record ordinary FAILED/retry
   disposition before releasing the exact still-live claim. Expired claims stay
   with the existing expiry/successor mechanism.
6. Qualify interruption/replay at every boundary through a public supervisor
   tick. Require one provider invocation, no queue publication/completion for a
   rejected candidate, and unchanged custody for old/incomplete receipts. Only
   then rerun Terminal Bench and publish new correctness/token measurements.

## Evidence boundaries

Current offline tests use real signed current-model routes, Git workspaces,
private CAS files and terminal resource retirement. Docker absence/dispatch are
explicit doubles. They validate the cleanup protocol and observation joins, not
live Docker STOP, a successful protected provider launch, callback settlement,
or a new Terminal Bench score. Historical helper tests using retired model
profiles or missing fixture exports require migration and are not counted as
passing current qualification.

STOP integration tests additionally use actual private binding/completion files,
the current cleanup producer, persisted lifecycle saga journals, and real local
process shutdown/startup repair. They cover repeated STOP, deleted/replaced
records, detached removal issuers, exact signed CAS joins, and unchanged file
contents/inodes during read-only observation. Eighteen tests in the older
`test_agent_supervisor_multi_supervisor_shutdown.py` suite fail identically on
the starting revision because they reference missing or retired contracts;
they are recorded separately and are not included in passing totals.
The older generation-status module also cannot collect on that revision because
`_SupervisorStatusGenerationBinding` is absent. The current health suite's stale
two-attempt status-reader assertion was updated to the four-attempt contract
introduced in September; production retry behavior was not changed.

The frozen STOP qualification reports **222 passing tests, no skips**: 70 cleanup
and disabled-dispatch checks, 76 lifecycle/startup-repair checks, and 76 current
runner health checks. Commands, source hashes, baseline failure classifications,
and the independent review are retained in
[`evidence/durable-stop-20261007/`](evidence/durable-stop-20261007/README.md).

The detailed migration audit and donor provenance are retained with this change's
qualification artifacts. No live task board, accepted source pin, existing claim,
foreign service or container is rewritten by these repairs.
