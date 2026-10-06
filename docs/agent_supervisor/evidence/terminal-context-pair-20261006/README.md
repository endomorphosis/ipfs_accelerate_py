# Terminal context comparison with planning retained

The compact coding transport did not produce a successful token reduction in
this fresh pair. The default `@1` arm completed `largest-eigenval` with official
reward **1.0** and **292,378 total tokens**. The compact `@2` arm received reward
**0.0** and consumed **335,762 tokens**, 43,384 more. Keep `@1` as the default;
`@2` remains an opt-in experiment.

Both arms retained one direct LLM planning session and one coding session through
`llm_router`. They used the same frozen source commit
`82671e6c8411875cb9919e3a71b07831548c0aeb`, archive, public inputs, Codex 0.160.0,
model/reasoning, 5-CPU/16-GiB resource profile, cache policy, retry policy and
timeouts. The independently checked serialized configurations differed only in
coding transport and output directory.

| Attempt | Official reward | Planning tokens | Coding tokens | Total tokens |
| --- | ---: | ---: | ---: | ---: |
| First baseline, strict planning rejection | 0.0 | 23,737 | Not invoked | 23,737 |
| Fresh default `@1` | 1.0 | 25,332 | 267,046 | 292,378 |
| Fresh compact `@2` | 0.0 | 25,346 | 310,416 | 335,762 |

Cached input is included in input and total tokens. Complete native usage is
available for every invoked session. The full observed experiment cost is
**651,877 tokens**, including the failed first planning attempt. Compact-01 was
prepared but never executed; it is not a completed zero-cost trial.

## What failed

The [planning schema fix](../terminal-planning-schema-20261006/README.md) worked
in both fresh arms: the CLI received the same closed canonical generation
schema, and both plans independently qualified through the native planning path.
The first attempt's unexpected task title did not recur.

In the compact arm, the coding CLI returned with exit code 0 and complete native
usage. After 174.44 seconds of coding invocation time, the router rejected its
reserved structured semantic reply with `SemanticTranslationError` at
`semantic_response_decode`, reason `response_envelope_binding_mismatch`.
The metadata does not identify whether envelope keys or a particular binding
mismatched. No raw
response, transcript, model thought, solution or hidden evaluator body was read
for this diagnosis.

The task then remained `in_progress` until the supervisor's work cutoff. Its
final report recorded a native-execution `TimeoutError` after 844.61 seconds.
Both arms stopped their native workers cleanly, with zero remaining processes.
The coding invocation finished before its provider timeout; the later total
budget failure is a separate lifecycle observation.

The compact arm's complete initial coding task input was 57,754 bytes, versus
59,128 in the baseline. These inputs came from independently generated native
plans and exclude provider system context and later internal history. Their
smaller initial size did not establish lower total usage. Native coding usage
recorded seven cumulative counter records in the baseline and eight in the
compact arm; these are not certified API-call counts.

## Evidence and next work

The [results](results.md), [comparison](comparison.json), [receipt review](receipt-review.json)
and [phase ledger](phase-ledger.json) retain all costs and failure outcomes. The
[archive preflight](archive-preflight.json) passed 219 checks; the
[prepared-pair preflight](pair-preflight.json) passed 85 checks. All 51 baseline
final receipt gates passed. The compact arm failed official reward and native
completion gates. The pair therefore does not qualify for a successful saving
claim, and one pair would not establish a repeatable causal reduction even if
both arms had passed.

The [diagnosis and backlog](response-settlement-diagnosis.json) prioritize
qualifying the existing native failure settlement on this exact post-decode
failure, then adding an explicit response contract for reference-bearing coding
replies with exact generated binding fields and unchanged native candidate
validation. Keep bad bindings rejected; do not strip fields or relabel a reserved
reply as prose. Later `main` contains concurrent lifecycle and provider-failure
handling changes; they were not exercised by this frozen pair and must be
considered before implementing another settlement path.

The [roadmap](roadmap.md) keeps LLM planning and describes the next representation,
catalog, capsule, formal-status and context-delta treatments. The
[planning acceptance reference design](next-treatment-design.json) is a proposed
lossless single-field deduplication, with its original source bindings retained.
It must be requalified against the current schema-enabled route. Embeddings can
rank candidate facts; native source, scope, completeness and proof checks still
decide what can be omitted or treated as established.

The [qualification](qualification.json) and [artifact manifest](artifact-manifest.json)
bind this publication. Runtime archives, model weights, raw model data and
benchmark solutions are not part of the evidence files here.
