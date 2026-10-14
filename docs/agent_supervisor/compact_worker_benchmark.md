# Compact worker benchmark: source views and metadata

This experiment uses `largest-eigenval` from Terminal Bench 2 with the full
indexed supervisor, the `llm_router` Grok 4.7 route and the same runtime archive.
It isolates source-view selection first, then metadata representation. It does
not rerun the native Codex or no-index baselines.

All arms use `mapped-span@2`, model planning, legacy coding replies, Source384,
the `python-numerical-control-structure@3` profile, one trial, no harness retries,
five CPUs and 20 GiB. The source pair differs only in source-view profile and
output directory. The third treatment keeps the compact source profile and
changes only metadata view and output directory. Configuration, source,
checkpoint, runtime and task hashes are retained before execution.
The runtime used commit `3f08cc43cbd96c9f8a5114091f65566a5962dbbc`.
The [retained evidence](evidence/compact-worker-live-20261014/README.md) binds
the source pins, releases, launch results and reporting scripts. All three
trials exited normally, with no harness retries; their containers were removed.

## Results

| Full-supervisor treatment | Official reward | Reported native tokens | Cached tokens within that count | Wall time including setup |
| --- | ---: | ---: | ---: | ---: |
| Identity source, legacy metadata | 1.0 | 1,310,764 | 1,185,536 | 1,163.80 s |
| Compact comments, legacy metadata | 1.0 | 1,621,661 | 1,356,416 | 1,026.37 s |
| Compact comments, compact metadata | 1.0 | 2,452,634 | 2,297,344 | 1,275.25 s |

All three runs' Grok envelopes omitted the explicit usage-completeness flag.
Their counters are coherent reported observations; complete all-attempt totals
remain unknown. Grok's fresh-input, cached-input, cache-write and output counts
are disjoint. Cached tokens are already included in the reported total and must
not be added a second time. These records do not establish a complete bill.

The first run used one planning invocation and one coding invocation, reporting
one and 23 native turns respectively. It produced two goals and one task.
Planning reported 24,474 tokens; coding reported 1,286,290. The source broker
observed seven accepted/delivered pages: one discovery, four source-view pages,
one mapped preparation and one confirmation. It returned 4,461 source bytes and
19,220 owner-response JSON bytes. Those scopes overlap and exclude commands,
client formatting, framing and conversation history. Original/compact/working
read subdivisions and ordinary-tool source-read counts are not retained.

The compact-comments arm reported 24,055 planning tokens and 1,597,606 coding
tokens across one and 26 turns. It also produced two goals and one task. Its
broker observed six requests: five delivered successful pages and one retrieval
refusal. No successful mapped preparation or confirmation was observed. The
refused request's operation is not retained, so this evidence cannot attribute
it to a particular read or edit operation. The broker returned 4,102 source
bytes and 15,399 owner-response JSON bytes.

Thus the source-only treatment reduced those transfer counters while reporting
310,897 more native tokens (23.7%) and three more coding turns. Its initial
complete prompt was 78,499 bytes, only 98 bytes smaller. This trial establishes
no source-minification efficiency advantage. Unobserved ordinary tool contents
and model variation prevent attributing the token difference to one refusal or
to the view profile alone.

The metadata treatment selected `common-bindings-instruction@1` without fallback.
Its own complete native input was 78,553 bytes; the selected representation was
77,585 bytes, a 968-byte (1.23%) reduction. This is a within-invocation byte
comparison, not a replay of the previous run's prompt. Planning reported 24,609
tokens and coding 2,428,025, with one and 36 native turns respectively. The
broker observed 15 requests: 14 successful responses and one retrieval refusal
whose operation is unknown. Three mapped preparations and three confirmations
were delivered; source bytes were 4,543 and owner-response JSON bytes 33,485.
The plan again contained two goals and one task.

Reported tokens increased by 830,973 (51.2%) relative to compact comments with
legacy metadata. This run demonstrates actual metadata displacement and
successive mapped edits through the full supervisor, but no total-token
efficiency advantage. None of these observations justifies enabling compaction
by default for token savings.

## What the first run reveals

The initial complete coding prompt was 78,597 UTF-8 bytes. Its encoded provider
object occupied 72,152 bytes, partitioned into 40,012 bytes of native context,
29,954 bytes of translated semantic content, a 1,335-byte translation table,
644 bytes of instructions and 207 remaining bytes. Within the semantic content,
raw sources occupied 2,588 bytes and capsules 12,008 bytes. These are nested
partitions; they cannot be added to the outer totals.

The dominant reported category is cached input across a multi-turn coding
conversation. This is evidence of a large cumulative context count, not evidence
that one source snapshot contained a million tokens. The initial payload census
also shows why comment removal alone has limited reach on this small task:
metadata and supervisor evidence occupy much more space than the raw sources.
It does not identify the contents of every later native tool result or prove
which text was reused on each turn.

Container setup took 361.54 seconds. Model planning took 149.77 seconds, while
initial indexed-context preparation took 16.53 seconds. Other recorded stages
include context preparation, doctor work and publication refresh; these named
stage measurements should not be blindly summed into a derived coding duration.

## Follow-up controls and measurement limits

The existing `common-bindings-instruction@1` view deduplicates shared metadata
and an exact admitted instruction copy. It records the original, candidate and
selected complete prompt hashes/bytes at actual dispatch, including selection
or fallback reason. The current CLI supplies no qualified Grok tokenizer to
this selector: it uses a UTF-8 byte and named proxy gate. Byte reduction alone
does not establish token reduction or better task performance.

Exact posthoc input replay is unavailable for the first run because the harness
retains counts and hashes without exporting the supervisor's private state tree
or original prompt bodies. The additional treatment measures selection inside
the live process instead of substituting a synthetic fixture.

`declared-contract@1` is another existing opt-in: it compiles the signed
mechanical planning skeleton without a planner model call. It does not interpret
intent or establish that the implementation satisfies it. Its net effect on
coding behavior requires a separate experiment; it is not varied here.

Each treatment has one observation. Model planning and coding can vary between
runs. Reported counter differences are preliminary observations, not causal or
statistical claims. Missing usage, uncertain helper observations and failed
attempts remain visible rather than becoming zero-cost successes.

The host was not isolated from other workloads. An unrelated benchmark container
was present during the third treatment; its contents were not inspected and it
was not changed. Equal container resource limits therefore do not make these
wall times a controlled runtime comparison.

## Next optimization target

Repeated compact-view editing is now exercised end to end, including three
confirmed mappings in one live task. Its existence alone does not reduce the
larger conversation context. A useful next control is to deliver bounded state
changes and request relevant indexed evidence on demand, while retaining the
admitted goal, constraints, source-generation checks and validation obligations.
Advisory autoencoder projections must remain distinguishable from checked proof
results. Symbolically discharged work should be recorded before deciding whether
another model turn is needed.

Measurement also needs bounded numeric usage observations per turn. The current
stream counts usage events but the retained readout cannot identify each turn's
context growth. The subsequent
[request-attribution change](evidence/source-request-attribution-20261014/README.md)
distinguishes refused reads from refused edit operations for future runs; it
does not reconstruct the missing operation labels or per-turn token values in
these three trials.

## Follow-up: preserve symbolic dispatch and diagnose event growth

The follow-up implementation fixes a conflict between symbolic completion and
model-only controls. A checked `doctor_candidate` or `doctor_contract_candidate`
can now proceed when a supported compact source, metadata or reply treatment is
configured for the model fallback. Previously these settings could reject the
provider-free route during implementation setup. The signed candidate handoff,
source checks, independent validations and completion rules remain in force.
Invalid fallback configurations are still rejected.

The report preserves the requested controls and marks them
`not_applicable_provider_free`. Source-view observations only adopt that status
when a matching successful Doctor materialization, cleanup and absence of coding
provider attempts are observed. A missing or contradictory receipt remains
unavailable. This fixes a route-selection failure; it does not show that Doctor
could solve `largest-eigenval`, whose observed dispatch was a model residual.

Grok stream observations now retain at most the first 128 usage events, with
numeric counters, event ordinals, timing and explicit omission counts. Missing
counter fields remain null. Historical progress records retain their original
shape without invented event histories. Event counters are not summed or used
to replace the existing final-envelope accounting: their cumulative/per-response,
cache and auxiliary-model semantics remain unverified. Progress updates carry
counts only; the final receipt retains the detailed bounded history once, to
avoid growing the worker log through repeated copies.

The context audit found that semantic translation already removes the original
semantic evidence row before inserting translated capsules. There is no
wholesale duplicate capsule copy to remove. The native Source384 and numerical
summaries contain different evidence. Existing compiler retry deltas do not
control the native CLI's internal conversation. A future explicit limit of one
or two selected capsules can use the existing bounded selection API while
retaining the full index and source checks, but that is an advisory-evidence
ablation and may increase retrieval calls. It has not been enabled by this fix.

The [follow-up qualification](evidence/symbolic-dispatch-usage-20261014/README.md)
records 691 distinct passing tests, including the actual Doctor proof-to-completion
join and bounded stream observations through the router and comparison reader.
These tests made no model calls and did not rerun the live benchmark; the token
scores above remain the last measured observations.

## Follow-up: bounded initial capsule selection

The next implementation adds frozen `one-capsule@1` and `two-capsules@1` budgets
for one signed task in the full indexed arm. The legacy maximum of eight remains
the default. Full native indexes, required raw inputs and source checks stay
intact; publication refresh rejects changes to the selected policy. Other arms
and multi-task profiles refuse these explicit budgets before initialization.

In two small authored repositories, one selected capsule reduced the complete
coding prompt by 30.78–31.17%, and two reduced it by 20.45–20.71%. The existing
byte budget selected four capsules in both legacy cases. These measurements
use real indexed preparation and router dispatch with an authored provider,
declared-contract planning and no learned Source384 inference. They measure
UTF-8 bytes, not live tokens or benchmark performance.

The [capsule-budget qualification](evidence/capsule-budget-20261014/README.md)
records 440 passing tests with unchanged sources, exact measurements and scope.
It also checks retrieval of omitted functions through current candidate capsules
and refusal of stale generations. That API rebuilds structural capsules rather
than restoring exact baseline or learned outputs; it requires source views and
has a smaller symbol limit than initial indexing. A one-capsule budget can also
change model-planning evidence. Additional retrieval or coding turns could
offset the initial byte reduction, so total token benefit still requires a live
comparison. The three earlier benchmark scores are unchanged.

## Follow-up: registered capsule comparisons and retrieval observations

`terminal_capsule_budget_matrix` prepares a bounded comparison of `legacy`,
`one-capsule@1` and `two-capsules@1`. It registers every output and receipt path
before preflight, including slots whose preparation fails. Three repeats rotate
each budget through every execution position. This is a deterministic schedule,
not randomization. The default common settings use declared-contract planning
and the existing five-CPU, 20-GiB coding600 resource profile. Preparation runs
Harbor dry-runs serially and launches no providers or containers.

Use a newly built runtime archive containing the current sources. Additional
options are a JSON object of `full_supervisor_benchmark.prepare` arguments;
the matrix owns `arm`, `capsule_budget`, task, archive and output selection.
For example, a model-planning comparison uses a separate `--options` file
containing `{"planning_mode":"model"}`. Provider, source representation,
retrieval policy, checkpoints and all other settings must remain common within
a matrix. Declared-contract results belong to a separate planning experiment
from the historical model-planning results above. Nondefault tasks require
their reviewed task profile. For the example below, save an options file with
`{"task_profile":"/path/to/reviewed-largest-eigenval-profile.json"}`.

```bash
python -m benchmarks.agent_supervisor.container_coding.terminal_capsule_budget_matrix \
  --dataset /path/to/terminal-bench-2 \
  --archive /path/to/current-runtime \
  --task largest-eigenval \
  --options /path/to/capsule-options.json \
  --output /path/to/new-capsule-matrix \
  --repeats 3
```

`matrix.json` contains the ordered `execute_argv` for every slot and a
`collect_argv` containing every registered receipt path. `preflight.json`
records preparation outcomes. Execute prepared slots one at a time using the
existing full-supervisor command. Its shared provider cooldown, resource limits
and one-shot invocation checks remain in effect. A provider deferral is an
unexecuted slot; keep it in the comparison. Do not replay an already invoked
trial to replace an unsuccessful result. The matrix generator itself neither
executes these commands nor enforces scheduling if an operator launches them
independently.

The dedicated `terminal_capsule_budget_comparison` collector reads the receipt
and its frozen `preparation.json` and `config.json` siblings. It permits only
capsule-budget and job/output naming differences, independently retaining the
generic collector's budget-mismatch result. It compares runtime archives,
assets, declared settings, original task integrity, realized capsule counts,
runtime budget declarations and provider identities. Missing receipts, absent
trials, failed attempts, incomplete usage and unavailable official rewards
remain visible. Duplicate observations do not double-charge an invocation.

```bash
python -m benchmarks.agent_supervisor.container_coding.terminal_capsule_budget_comparison \
  --legacy /path/to/legacy/receipt.json \
  --one /path/to/one/receipt.json \
  --two /path/to/two/receipt.json \
  --output /path/to/new-comparison
```

Repeat each budget flag to include all its registered attempts. The matrix's
generated collection command does this automatically. Complete native-token
differences require matching experiment settings, equal distinct trial counts,
official success for all observed trials, no missing registered trials and
complete native usage. Known partial counters remain subtotals. Prompt bytes
and stream snapshots never become token totals. Even a qualified descriptive
difference does not establish causal savings, exhaustive pipeline cost or a
verified bill. The collector does not independently prove preregistration or
that every real-world attempt was supplied.

Repository-source audit summaries now retain optional, content-free retrieval
observations: successful and empty queries/expansions, candidate/capsule counts,
generation-binding observations and stale-generation/source/root refusals.
These summaries survive private journal cleanup. Historical observations remain
unknown for retrieval. A successful expansion records a prepared tool response;
it does not establish worker receipt, model attention, identity with an omitted initial
capsule, a proof, task completion or token savings. Initial context and source
sessions can have different native identities, so consumers must query current
session handles and bind relevant symbols by source identity.

The live token figures earlier in this document remain unchanged. These tools
support the next experiment; their authored qualification is not a new live
Terminal Bench result.

The [qualification](evidence/capsule-comparison-retrieval-20261014/qualification.json)
records 793 distinct passing tests. The combined suite passed 630 checks and
reused five AST test seals; those five then passed against a fresh dedicated
cache. The remaining 158 checks cover the existing capsule budgets and context
path. Tested Python source hashes stayed unchanged during final verification.
The new runner/stdio cases recover an initially omitted symbol with either one
or two initial capsules, read its exact source, refresh after an external edit,
and refuse stale source, generation and root handles. Provider responses are
authored. The retained [census](evidence/capsule-comparison-retrieval-20261014/prompt-census.json)
again measures roughly 31% and 21% smaller initial prompt bytes; it does not
measure full-session provider token savings.

## Follow-up: indexed definition reads in the worker edit loop

For an invocation using repository-source tools, `repository_read` accepts an
optional `symbol_id` from the current `repository_context_query` or
`repository_symbols` result. It resolves the exact native identity without a
list-result limit, checks the retained original file hash, and matches the
native source span to a supported Python definition. The response includes a
verified local `symbol`, such as `Counter.square`, usable by
`repository_replace_symbol`. Workers no longer have to infer edit names from
module-qualified index names or reread every definition in a file to obtain an
edit selector.

The worker passes the read's `generation`, `path`, `symbol`, `source_sha256`
(as `expected_sha256`) and returned `representation` to the existing guarded
replacement tool. The representation matters when a compact request falls
back to original text: replaying that exact definition should preserve the
canonical bytes. Real changes still use a disposable preflight, whole-file hash
checks, canonical replacement and a refreshed native index. Stable native IDs
can survive a body change; they never substitute for a current generation or
file hash. A change elsewhere in the checkout also invalidates a targeted read.

Definition reads include decorators and dedent the selected code. They omit
surrounding imports, enclosing class context and adjacent comments. Compact
reads also omit internal comments and formatting. Ask for `original` to inspect
the selected definition's original text; omit `symbol_id` to read the complete
file. Respect capsule requirements for raw source. Nested functions, conditional
definitions, ambiguous names and unsupported languages require full-file reads
and the existing canonical editing workflow. Comment-only changes still require
a direct original-source edit followed by `repository_refresh`.

Compact definition rendering must reduce content bytes. When an owner-selected
local tokenizer is configured, the server additionally requires a strict
reduction in the complete MCP reply's token count, including identifier and
metadata overhead. It rechecks source freshness after counting. These tools
remain part of the selected repository-source policy; this change neither
intercepts arbitrary terminal reads nor demonstrates live model adoption.

`test_repository_source_symbol_stdio.py` exercises the real supervisor route,
child MCP, native index, targeted read, canonical edit, successor query, stale
read refusal, full-file fallback and caller behavior on authored source. Its
JUnit `symbol_read_wire_comparison` property records complete request/reply
bytes and explicitly labelled fixture-tokenizer counts for full-original,
full-compact and definition-compact reads of the same 32-definition file. That
local read comparison excludes the common preceding query, tool registration,
model reasoning, history replay and subsequent task work. It is not a live
Terminal Bench result or a full-session token-savings claim.

Fresh runtime qualification uses `repository-source-runtime-probe@2`. The
installed-runtime probe exercises the indexed definition read/edit path and
validates its retained retrieval audit, in addition to the existing full-file,
cache and generation checks. Historical `@1` receipts can be reviewed with an
explicit legacy option, but do not qualify the current deployment. Build a new
archive to include the helper and updated source pins before a live comparison.

The [qualification](evidence/symbol-workflow-20261014/qualification.json)
records 703 passing checks: 655 with retained JUnit logs and 48 renderer/session
checks reported by delegated test runs. No failures or skips remained. The
[wire comparison](evidence/symbol-workflow-20261014/wire-comparison.json)
measured 4,463 request/reply bytes for a full original read, 2,029 for a full
compact read and 767 for a compact definition read. The latter is a 62.2%
reduction relative to the full compact operation in this authored fixture.
All 18 changed Python files retained identical hashes during final checks.
No fresh runtime archive, container trial or live provider call was performed.
