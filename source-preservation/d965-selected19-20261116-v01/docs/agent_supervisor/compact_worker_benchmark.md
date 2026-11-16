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

## Exact original line ranges in the source working set (2026-10-15)

The existing `repository_read` tool now accepts `start_line` and `end_line`
together with `representation="original"`. This supports small exact reads of
admitted UTF-8 files, including languages without indexed Python definitions:

```json
{"generation":1,"path":"app.js","representation":"original","start_line":301,"end_line":305}
```

Coordinates are one-based, inclusive physical LF lines. Returned content
preserves original bytes, including CRLF, Unicode separators and an absent final
newline. Replies retain the whole-file `source_sha256` and generation, and add
`scope="line_range"`, `start_line`, `end_line`, `total_lines` and `omitted_lines`.
The rest of the file is omitted explicitly; a range is not a complete program or
an indexed symbol edit handle. Request surrounding lines, a supported definition
or the original file when additional context is needed.

Ranges cannot combine with `symbol_id` or a compact representation. Both bounds
are required. Binary/empty files, invalid coordinates, more than 4,096 selected
lines or more than 65,536 selected UTF-8 bytes refuse without truncation.
Whole-file and definition reads keep their existing behavior. Whole-repository
generation/currentness checks still surround the retained read; changing an
unselected file also requires refresh. This changes model-visible source scope,
not source integrity or executable checkout contents.

This capability is available through the existing explicitly selected repository
source policy and its ordinary tool discovery. It does not intercept shell
reads, remove required initial evidence or guarantee model adoption. A local
complete-MCP-reply comparison excludes request bytes, once-per-invocation
guidance/tool schema costs and later conversation replay; it cannot establish
whole-task savings.

Fresh [qualification](evidence/source-working-set-20261015/qualification.json)
passed all 451 selected checks across 12 test files, with no skips, fresh AST
seals and optional proof reuse off. It used native DuckDB 1.5.5 and real MCP
children on authored sources; runtime source hashes, membership and child import
origins stayed bound to the captured checkouts. Native processes and the owned
resource reservation closed successfully. Earlier failed attempts remain in the
qualification history.

The [authored reply comparison](evidence/source-working-set-20261015/wire-comparison.json)
measured 32,495 bytes for a 600-line JavaScript file and 688 bytes for five exact
lines, including the complete MCP reply envelope and range metadata: 97.9% fewer
reply bytes. Counts from the authored local tokenizer were 5,457 and 122.
These are fixture observations, with no live provider usage or complete-task
token-saving claim. The [JUnit record](evidence/source-working-set-20261015/junit.xml)
retains the original measurement property.

## Retained read scope and complete local exchanges (2026-10-15)

The existing private MCP audit now records content-free read-scope events and
retains a validated `read_observation` after cleanup. It distinguishes full-file,
indexed-definition and exact-line reads, original/compact representations,
failures, generation bindings and exposed content-string bytes. Source paths and
bodies are not retained in these events. Binary whole-file exposure counts its
base64 text rather than decoded source bytes. These counts exclude JSON framing,
requests, tool schemas and guidance.

Historical journals without scope retain an unknown scope. Mixed historical or
unbound observations are explicitly incomplete. Line totals count each range
operation, including repeated reads and different generations; omitted lines
describe unreturned context and do not establish avoided ordinary reads.

The existing worker receipt and Terminal collectors preserve validated read,
retrieval and tokenizer observations alongside native session accounting.
Read-scope totals require complete observations for every selected invocation
and distinct invocation identities. Failed attempts remain in the population.
`line_reads_observed` requires the existing exact-input, native configuration,
initialization and cleanup joins; missing evidence stays unknown. Prepared
source replies do not establish model ingestion, displacement or saved tokens.

The authored worker comparison uses the same canonical source edit and Python
validation for full-file and line-range reads. It records actual pipe writes and
reads from initialization through discovery, a context query, editing, a stale
read refusal and a successor read, together with the identical rendered initial
prompt and guidance. Local tokenizer counts cover that bounded exchange once;
provider-internal history, reasoning, conversation replay and official task
success remain unmeasured. Native provider totals remain a separate measure.

Fresh [qualification](evidence/source-adoption-20261015/qualification.json)
passed all 861 selected checks across 17 test files, with no skips, fresh AST
seals and optional proof reuse off. This includes actual worker/MCP reads,
failed-provider cleanup and collection, native isolated runtime probes, authored
archive compatibility and signed-routing regressions. Earlier failed checks and
the three reviewed corrections remain in the retained history; no production
archive or source-integrity guard was relaxed. Captured source hashes and native
child import origins stayed bound, and the owned processes and reservation
closed successfully. These checks do not qualify a fresh runtime archive,
Docker lifecycle or live model adoption.

The [complete local exchange](evidence/source-adoption-20261015/session-comparison.json)
used nine request frames and eight reply frames in each arm, with the identical
2,886-byte initial prompt and the same validated canonical result. Full-file
reads required 23,949 prompt-plus-wire bytes; line reads required 17,631, a 26.4%
reduction. Authored local-tokenizer counts were 4,648 and 3,050, a 34.4%
reduction. The extra line selectors increased request bytes from 1,494 to 1,596;
reply bytes fell from 19,569 to 13,149. This includes discovery and a failed
read, while provider history and whole-task tokens remain unmeasured. The
[JUnit record](evidence/source-adoption-20261015/junit.xml) retains the original
measurement property.

## Native session reconciliation in comparisons (2026-10-15)

The existing benchmark comparator now uses the existing context-delivery
accounting owner for supervisor native Codex receipts. Distinct router wrappers
can observe the same cumulative session; their snapshots are not additive. A
consistent later snapshot counts once, while a separate failed retry remains
included. Modern Codex receipts with missing identities, conflicting snapshots,
or unknown missing-attempt markers withhold complete totals. Historical wrappers
without identities retain explicitly labelled unreconciled descriptive sums.
Grok-only populations retain the labelled invocation-accounting route in this
increment.

Input totals state their cache scope. Codex cache and reasoning counters remain
subsets, and mixed Codex/Grok observations do not claim cumulative lower bounds.
Shared-session dollar costs remain unknown unless their accounting semantics are
established; raw reported cost subtotals stay explicitly unreconciled. Reusing an
identified supervisor session across retained trials or comparison arms withholds
aggregate totals and tokens per official pass. Baseline session identities were
outside this initial increment; the [baseline and capsule overlap increment](#baseline-and-capsule-session-overlap-2026-10-15)
below extends that coverage.

The comparator recomputes its context observation from retained invocation
receipts rather than accepting a precomputed summary. Read scopes, retrieval
observations and local tokenizer counts remain separate diagnostics. Planning
and coding phase presence does not establish that every campaign call was
retained. No source byte count or local token estimate is added to native usage.

The [authored integration observation](evidence/native-session-accounting-20261015/native-session-comparison.json)
uses the actual local Codex JSONL parser and MCP audit producer. Three wrappers
cover two sessions: a later shared-session total of 170 and a failed retry of
77 yield 247, rather than the erroneous additive wrapper total of 357. Read
exposure is separately recorded as 36 UTF-8 bytes. These are authored provider
facts that qualify accounting behavior, not authenticated remote usage or token
savings. A complete supplied receipt population still does not establish
whole-run coverage, verified billing or a causal comparison.

Fresh [qualification](evidence/native-session-accounting-20261015/qualification.json)
passed 505 checks across 10 normal test modules with fresh AST seals, optional
proof reuse disabled and no skipped cases. The [JUnit record](evidence/native-session-accounting-20261015/junit.xml)
includes the integration measurement. Captured source membership, hashes and
child import origins were verified; owned process groups and the reservation
closed. The first attempt failed collection because an older lifecycle fixture
imports an absent parser API. That failure and the explicit scope revision are
retained; the current usage-growth tests cover the current parser lifecycle.
This qualification covers neither that older fixture nor live provider runs,
runtime archive/Docker lifecycle, CUDA performance or whole-task token savings.

## Baseline and capsule session overlap (2026-10-15)

The existing comparator now shares one overlap guard across reported baseline
session identities and reconciled supervisor session identities. Exact declared
baseline IDs are hashed in the same identity domain; transcript paths, artifact
hashes and `unidentified:` fallbacks never substitute for IDs. Bounded identity
coverage is an observation, not authentication of the supplied receipt or proof
that every campaign call was retained.

Duplicate, malformed or partial declared baseline identity inventories withhold
complete row totals and costs. Current producer receipts also require coherent
top-level and per-session counters and explicit per-session completion. Known
reported subtotals remain labelled observations. Valid IDs from a partial
inventory still participate conservatively in overlap checks. Historical
identity-free receipts retain explicitly unverified descriptive accounting.
The baseline producer continues to deduplicate identical transcript copies;
the comparator withholds totals for duplicated IDs in a retained inventory
rather than guessing how its top-level sum was constructed.

Overlap is recomputed across all deduplicated rows before splitting baseline,
supervisor, capsule-budget or unsuccessful-attempt groups. Reusing a native
session with a different router wrapper ID therefore cannot produce complete
contrasting totals. Capsule `complete_registered_native_usage` now requires a
complete aggregate as well as complete individual rows. Distinct sessions
retain their existing totals and descriptive contrasts.

The [authored baseline observation](evidence/baseline-session-overlap-20261015/baseline-session-overlap.json)
uses the actual baseline producer on authored JSONL shared with the supervisor
fixture. Its row observations remain 170 baseline tokens and 247 supervisor
tokens, including a separate failed retry. The shared session makes both arm
totals and per-pass ratios unavailable. This is overlap detection, not a token
reduction or a live provider measurement.

Fresh [qualification](evidence/baseline-session-overlap-20261015/qualification.json)
passed 564 checks across 12 normal modules, with fresh AST seals, optional proof
reuse disabled and no skipped cases. The [JUnit record](evidence/baseline-session-overlap-20261015/junit.xml)
retains both the original native-session observation and the new baseline
overlap observation. Captured sources, membership and child import origins
were checked; owned processes and the reservation closed. The initial failed
staging attempt and the later 562/563 regression result remain unqualified in
the history. The corrected producer test now checks complete and sparse
counters separately, without weakening the production guard. This qualification
does not establish live provider adoption, whole-task savings, verified billing,
runtime archive/Docker lifecycle or neural-head/CUDA performance.

## Local receipt collection and semantic attempt coverage (2026-10-16)

The existing Terminal producer now retains a possible missing planning attempt
when a normal planner response lacks its qualified final router receipt, as it
already did for interrupted planning. Failed provider receipts remain visible.
The existing missing-attempt marker preserves an earlier failure; an observed
new gap supersedes a prior explicit no-gap declaration.

Final log collection now uses bounded regular, stable file reads and records
content-free scan diagnostics. Oversized, unreadable, changing or linked logs,
linked subdirectories, truncated JSON-looking records and exhausted scan bounds
leave collection incomplete even when other native receipts survived. Optional
Doctor metadata cannot interrupt collection or admit unbound Doctor rows.
Identical native observations deduplicate through the existing receipt owner;
conflicting observations remain available for the existing consumer rejection.

The comparator also reuses the existing semantic-decision coverage owner on
original retained dialogue and owner-binding evidence. Native receipts found
only inside a dialogue reach the existing native counter reconciler. Missing
declared dialogue/owner coverage and malformed or incomplete scan diagnostics
withhold complete public totals, costs and call counts while preserving known
native subtotals and the native core's separately scoped observations. Stored
usage summaries cannot repair those gaps.

Ordinary runs now finalize the existing accounting observation before their
final audit and again after a successful audit. The Harbor adapter routes
recognized Codex receipts through the existing session reconciliation instead
of summing cumulative wrapper snapshots. Ordinary Grok final-envelope accounting
retains its provider-specific scope; missing-attempt markers and invalid or
incomplete collection diagnostics withhold its complete export too. Both paths
share the closed scan diagnostic validator with the comparator.

The adapter also retains independently readable task-report diagnostics when
usage calculation or Harbor export fails. It records the accounting exception
type separately and clears all three Harbor token counters, including a partial
export. Accounting failure does not turn the retained task report into an
unavailable report, and exception messages are not exported.

The adapter's existing lifecycle reconciler still requires its explicit
lifecycle evidence. Current Codex parser receipts do not supply that older
contract in full, so their Harbor export remains unknown. Authored fixtures
with the complete lifecycle contract qualify only that contract; they do not
establish current live-producer compatibility. The newer comparator retains its
own separately scoped native evidence without upgrading the adapter's result.

A clean scan describes the logs visited, not an exhaustive model-call census.
It cannot repair an unknown missing-attempt marker, establish campaign coverage,
or authenticate provider usage. Native provider callback counts describe outer
execution callbacks, which can contain multiple model calls; they are not a
substitute for model-attempt coverage. Expected attempt counts and whole-campaign
completeness remain unknown, including on clean and provider-free paths.

The [authored collection observation](evidence/usage-coverage-20261015/usage-coverage.json)
runs the actual receipt collector and comparator on the existing native JSONL
fixture plus one unavailable oversized log. All three retained invocations and
the failed retry remain visible, with 247 known native tokens and 36 separately
reported source bytes. The complete total and per-pass ratio stay unavailable.
This observation establishes conservative accounting behavior, not that the
unavailable log represents another dispatched call or that tokens were saved.

The [current-source qualification](evidence/usage-coverage-20261015/qualification.json)
passed 732 checks across 19 normal modules, with fresh mandatory AST seals,
optional proof reuse disabled and no skipped cases. Captured runtime membership,
source hashes and actual child import origins were checked; owned native
processes and the reservation closed. The [JUnit record](evidence/usage-coverage-20261015/junit.xml)
retains the native-session, baseline-overlap and collection observations.
The earlier 702/712 and two 730/731 executions remain unqualified in the history.
Their fixes address ordinary accounting/finalization/report retention and stale
authored model/archive fixtures, while preserving production admission guards.
This qualification covers local accounting behavior. Current Codex lifecycle
compatibility, registered whole-run coverage and whole-task token savings remain
separate work.

Documentation evidence uses stable preflight snapshots. An interrupted seal
caused by a concurrent shared-document edit is retained as unqualified; its
partial source copies are not reused for the final seal. Later shared-document
updates remain outside the captured documentation snapshots.

## Codex lifecycle metadata through Harbor (2026-10-16)

The current Codex receipt producer now emits the existing reconciler's explicit
`native_lifecycle_after_latest_usage@1` completion basis, structural malformed
record count and usage-lifecycle completion flag. Usage completion requires a
validated full final snapshot, a positive validated snapshot count and native
completion after the latest work, with the existing malformed-record and
counter-consistency guards satisfied. A completion marker without usage still
records that marker while leaving usage incomplete; missing categories remain
unknown. Quota updates and sparse snapshots do not become validated full
records or inflate the structural malformed count.

Regression detection now retains a bounded per-category high-water value across
sparse, quota and malformed gaps. A later lower valid numeric category sets the
existing regression flag and invalidates the existing completion observation.
For example, full input 50, sparse input 60, then full input 55 cannot regain
completion merely because 55 exceeds the first full snapshot. Monotone recovery
remains supported. Harbor and the comparator receive this correction through
their existing guards; no alternative accounting owner is introduced.

The existing reconciler, native context core, Terminal finalizer and Harbor
adapter retain their receipt binding, alias, coverage and counter policies.
Historical receipts without explicit lifecycle evidence remain unknown in the
strict reconciler. Mixing historical and current re-parsed receipts for the
same rollout digest can trigger its existing alias-conflict guard; this bridge
does not upgrade historical receipts automatically.

The authored parser-to-Harbor fixture retains a shared session's 120-token
snapshot and its later 200-token snapshot, plus a separate completed failed
retry with 80 tokens. It uses the actual JSONL parser and existing consumers:
the wrapper sum of 400 becomes a supplied-session total of 280, with Harbor
input/cache/output counters of 230/115/50. The failed retry stays charged even
though its outer status is failed. An unfinished retry retains observed charges
while withholding complete exports. This is an accounting correction;
whole-campaign completeness and whole-task token savings remain unmeasured.

The [current-source qualification](evidence/codex-lifecycle-20261016/qualification.json)
passed 982 checks across 26 normal modules, with fresh mandatory AST seals,
optional proof reuse disabled and no skipped cases. It covers native parser,
router, telemetry, reconciliation, Terminal finalization and Harbor behavior.
Captured source hashes, membership and actual child import origins were checked;
owned native processes and the reservation closed. The
[JUnit record](evidence/codex-lifecycle-20261016/junit.xml) retains the three prior
accounting observations plus the new
[parser-to-Harbor observation](evidence/codex-lifecycle-20261016/codex-lifecycle-harbor.json).
The initial 974/982 execution remains unqualified. Its existing diagnostic
tests now distinguish recoverable gaps from persistent numeric decreases, and
the router fixture checks the current literal-response and temperature options;
all original cases remain. The production implementation is identical in both
executions.

The next experiment needs a frozen runtime containing this producer revision,
fresh native session state, registered dispatch coverage and unchanged task
acceptance across matched arms. This local qualification does not establish
runtime archive deployment, live provider coverage or whole-task savings.


## Frozen runtime and capsule pilot preparation (2026-10-16)

Benchmark preparation and execution now require the actual archived bytes of all
four declared deadline owners and nine declared usage owners. The usage chain
includes the Codex parser, interaction telemetry, native delivery, delivery
accounting, cumulative reconciliation, semantic decision usage, Terminal usage
finalization and the existing Harbor adapter observers. The worker capability
also pins this declared chain. A refreshed outer archive hash or manifest label
cannot hide stale, missing or duplicate required members. Execution checks this
before creating its invocation marker or launching Harbor. Selected decision
routes retain their additional source checks. This is a declared dependency
check, not a proof of the complete imported dependency closure.

The existing capsule matrix remains the sole owner of registered slots. It now
uses the existing capsule comparator's frozen configuration, asset and control
checks before reporting readiness. All slots are registered first, and the
registry digest is held before preparation begins. Missing, corrupt or mismatched
preparations, unexpected execution receipts and a changed registry prevent
readiness while retaining the original planned population and completed
preflight observations. This does not authenticate preregistration or certify
complete future model dispatch coverage.

The provider-free pilot is fixed to `fix-code-vulnerability`, three repeats of
legacy/one-capsule/two-capsules, explicit model planning and the existing
`codex-gpt-6.1-sol-0.160.0@1` profile. It uses the common
`source384-5cpu-20gib-coding600@1` resource profile. Optional pretrained model,
Torch and native toolchain assets are omitted equally from every arm. The
experiment isolates capsule delivery; it does not qualify 384D/768D/4096D neural
head performance, CodebaseIR training or CUDA inference. Archive input custody
includes all code assets selected by the existing builder, including supported
non-Python assets, rather than treating Python-only test capture as a complete
archive input snapshot.

The complete archive input snapshot is captured before native validation. Its
13,102 Python files, 50 supported non-Python code assets and three existing
extensions are stable across the original capture census. Later preparation
uses these immutable inputs and rechecks their bytes, membership and modes; it
does not chase unrelated later checkout edits or claim live checkout currentness.
Earlier failed preparation attempts remain recorded with their actual causes.

The [qualification](evidence/matched-preflight-20261016/qualification.json) passed
1,429 cases across the same 32 normal modules, with fresh mandatory AST seals,
optional proof reuse off and no skipped cases. The
[JUnit record](evidence/matched-preflight-20261016/junit.xml) retains the four
prior accounting observations. Native source hashes, membership and actual
child import origins were checked. The owned capture, native test and
preparation processes closed, and the unchanged MAIN reservation was released
without requiring the global pool to be empty.

The independently checked provider-free preparation produced a 13,155-member
runtime archive with exact selected-asset bytes, sizes and executable modes,
and nine actual successful Harbor dry-runs. All nine preparations satisfy the
existing frozen control/asset comparison, and the originally held registry
digest remains unchanged. Model planning is explicit in the matrix; omitted
default planning fields are interpreted by the existing planning-policy owner.
No provider, container, task code or benchmark verifier was run. Earlier failed
preflights and native attempts retain their actual statuses and exact inputs;
the earlier successful native-only run is not added to this case count.

This completes local archive and matched-preparation qualification. Registered
whole-run model dispatch coverage, runtime deployment, neural head/CUDA
qualification and equivalent accepted-task token totals remain separate gates.
A matching capsule setup alone does not establish whole-task savings.

## Current-world inline metadata compaction (2026-10-16)

The current-world runner now defaults to a reversible reference-metadata view
for `run(world_task_packet=...)`. Identical scalar schema, version, kind, tier,
repository, tree and path fields may be shared in one inline table. Every task,
source/reference body, identity, coverage field and unresolved obligation stays
recoverable from the single delivered input. The original issued packet and its
late source/owner checks remain the admission authority. This route currently
supports Codex coding. Set `compact_world_prompt=False` to retain native input.

The existing complete-input selector compares both fully rendered alternatives,
including workspace/source-tool guidance, reply instructions and the separate
native reply schema. The compact form must use strictly fewer bytes and fewer
declared local counts or labelled estimates. Ties, absent/failed counters and
larger representations retain native input; explicit budget refusals remain.
The selected measurement reuses this observation without a second token count.
The projection creates no additional store and assumes no retained provider
parent. Historical invocation replay reconstructs the selected bytes without a
tokenizer or live source/intent owners; old records without the new marker keep
native rendering. Replay does not reinstate live dispatch authority.

The [local qualification](evidence/world-prompt-compaction-20261016/qualification.json)
passed 589 cases across 20 normal modules with fresh mandatory AST seals,
optional proof reuse off and no skipped cases. All 13,108 captured Python runtime
sources and 126 normal test/configuration inputs were checked; their source
bytes are retained with the campaign. The actual child import origins were
checked, and the owned processes and reservation closed without requiring an
empty global resource pool. Earlier unsuccessful attempts retain their actual
diagnostics and are not added to the passing case count.

In the authored-provider fixture, complete model input fell from 27,079 to
25,923 bytes, a 4.3% reduction, with exact native restoration. The automatic
UTF-8-based estimate fell from 6,770 to 6,481; the other declared counters are
authored byte counters. These are local input observations, not provider-token
or accepted-task session savings. The
[JUnit record](evidence/world-prompt-compaction-20261016/junit.xml) binds the three
observations to their exact passing test cases.

The next integrations are the existing Terminal worker handoff to live world
owners, the deterministic-first gate before retries, and checked Doctor results
in successor obligations. Each needs current task, source and policy bindings
and the existing independent validation/proof gates. Keep model-driven planning
in the matched comparison. Total native planning, coding and retry input/output
tokens at unchanged acceptance remain the outcome to measure. A byte-reversible
prompt alone does not establish model behavior equivalence or session savings.

For a matched world-input run, qualify the live Terminal handoff first, then
compare `compact_world_prompt=False` with the default using fixed task, packet,
model, source, tool, reply and resource controls. Retain complete native usage,
failed attempts and acceptance outcomes across both arms.
