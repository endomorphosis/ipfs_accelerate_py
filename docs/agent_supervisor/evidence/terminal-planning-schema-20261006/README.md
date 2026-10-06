# Keep planning while constraining its response shape

The first fresh `largest-eigenval` baseline stopped before coding because the
planning response added `tasks[0].title`. The native graph grammar allows a goal
title but forbids that task field. The trial received official reward **0** and
used **23,737 tokens**: 22,198 input (including 12,288 cached input) and 1,539
output. Its complete cost remains in the [failed-attempt accounting](failed-attempt-accounting.json).
The prepared compact arm was never executed. These records do not establish
token savings or a successful matched pair.

The Codex planning route now passes the canonical direct planning grammar to
the pinned CLI's `--output-schema` option. It uses a private temporary schema
file and records its exact byte count and SHA-256. The runner checks those
observed values against its generated schema before validating the response.
The response stays unchanged: unexpected fields are rejected rather than
removed. Ordinary Codex calls retain their previous command arguments.

The generation projection replaces constant strings with typed singleton enums,
uses `$defs` for local references, and removes only schema identity and
application annotations. All required fields and closed object shapes remain.
The original canonical validator still enforces the native response contract.
This is structural generation support for the direct prompt-goal route; generic
requests and the separate IntentIR coverage wrapper do not gain this support.

Both LLM planning and coding continue through `llm_router`. After response shape
validation, the existing native planning path still checks scope, evidence,
freshness and admission. The schema receipt explicitly grants no execution,
proof or completion authority. Passing a JSON schema does not discharge mathematical,
timing or task-correctness obligations.

The pinned Codex 0.160.0 help advertises `--output-schema`. The
[official Structured Outputs documentation](https://developers.openai.com/api/docs/guides/structured-outputs#supported-schemas)
describes closed object schemas, required fields and supported references. The
local provider, projection and roundtrip tests use fake CLI processes; they
do not establish that a fresh benchmark will succeed.

## Qualification and next run

The [qualification](qualification.json), [independent review](independent-review.json)
and [artifact manifest](artifact-manifest.json) bind the implementation and
offline results. Provider and ordinary-route checks passed **58 tests**; schema
projection checks passed **20 tests**; integration and selected regressions
passed **45 tests**. The initial integration run exposed missing schema receipt
fields; that issue was fixed and the successful rerun is retained separately.

The next comparison uses one newly built archive containing this fix for both
full arms: coding transport `supervisor-semantic-router-input@1` versus `@2`,
with one direct planning session and one coding session in each. Match public
inputs, model, CLI, reasoning, resources, cache policy, retries and timeouts.
Keep this failed 23,737-token attempt separate and include it in total experiment
cost. A qualified comparison must independently check successful native plan
admission, schema custody, complete native token usage, coding-input version
verification, official reward and cleanup. A single pair is descriptive evidence
and cannot establish a stable causal saving.

The [context reduction plan](../terminal-context-dictionary-20261006/README.md)
still prioritizes metadata queries, supported formal obligations, semantic
capsules and reversible minification. This change fixes a planning failure and
does not remove the planning session or implement those later treatments.
