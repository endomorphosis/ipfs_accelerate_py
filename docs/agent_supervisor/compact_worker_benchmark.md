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
