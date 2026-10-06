# Qualify failure settlement and declare coding completion replies

The [previous transport pair](../terminal-context-pair-20261006/README.md) exposed
two separate problems: a completed coding CLI returned a reserved reply whose
envelope bindings failed validation, and the admitted task remained in progress
until the overall work cutoff. Its cost and failed reward remain unchanged in
the previous evidence. This follow-up qualifies existing settlement code and
adds an explicit coding completion contract before another live comparison.

## Qualified native failure settlement

No settlement runtime implementation was added. The current native path was
tested against an authored CLI that writes a partial candidate, finishes its
usage receipt and exits zero. The production router then rejects a foreign
reserved reply through `semantic_response_decode` and exits nonzero. Detached,
pooled and signed START/Quack-owner/STOP fixtures exercise the actual native
child custody, lifecycle, event/state and one-use capability checks.

The owner persists `failed_outcome_settled`, blocks the task, releases its claim,
rejects the partial edit and admits no automatic provider retry. Canonical files
remain unchanged and native STOP removes process members. The
[settlement summary](settlement-summary.json) binds 68 custody/lifecycle/admission
controls and three final post-decode entrypoint cases. The signed fixture
reaches terminal failure within its 15-second observation budget after dispatch.
This is offline qualification, rather than a new Terminal Bench timing result.
The fixture's 19-token usage is authored test data, not billed model usage.

## Opt-in ordinary coding completion

Select `--coding-reply-mode ordinary-completion@1` during benchmark preparation
for the full indexed Codex route. The owner then requests ordinary source edits
and this closed final acknowledgment:

```json
{"schema":"supervisor-coding-completion@1","status":"candidate_ready"}
```

The router passes its two-field schema to the native Codex generation option.
All fields are required, both values are fixed enums and extra fields are
forbidden. The observed schema byte count and SHA-256 must match the requested
schema. The final response passes the unchanged strict semantic decoder first,
then the bounded completion validator. Reserved replies with bad bindings remain
rejected; no fields or binding values are repaired. This mode does not select a
residual task family or accept a reference-bearing candidate reply.

The acknowledgment grants no execution, completion, settlement or retry
authority. Native owners still validate and publish actual candidate effects.
LLM planning remains enabled through `llm_router`, with its separate native
planning schema and admission checks. Grok, no-index and provider-free Doctor
routes reject this new benchmark selection before dispatch.

The default reply mode is `legacy`. Selecting or omitting that default preserves
the historical `@1` and `@2` prompt, configuration, metadata and command shapes.
The new reply mode is independent of transport selection; use the same explicit
mode in both arms of a later transport comparison.

## Input accounting and preflight

The mode adds a fixed **506-byte instruction suffix** after the complete native
context, residual advisory, literal public instruction and workspace advisory.
Its native generation schema is **215 bytes**. The receipt binds the input
before and after the suffix, suffix digest/count, generation schema digest/count
and response-validation status. Complete `model_prompt` hashes and counts still
describe the actual provider input. Historical context auditing reconstructs and
checks the suffix even after source changes or removal of the allocated worktree.

Preparation freezes the choice in its complete configuration digest. Setup,
driver, worker and execution checks retain that explicit choice. Thirteen
current archive source bodies, including the strict decoder, CLI metadata,
controls and input auditor, must match before the new mode can be used. An old
or mismatched archive is refused before uploads or provider dispatch.

## Evidence and next qualification

The [qualification](qualification.json), [runtime review](runtime-review.json),
[benchmark plumbing review](benchmark-review.json), [accounting controls](accounting-controls.json)
and [artifact manifest](artifact-manifest.json) bind this change. Offline checks
passed 92 contract/runner/planner/diagnostic cases, 93 benchmark/input-audit cases
and 34 comparison-control regressions, in addition to the settlement cases.
The first benchmark-path run caught the missing controls allowlist entry; its
failure is retained separately from the successful rerun.

Future comparison must verify actual coding receipt mode, native schema custody,
successful response validation and complete final-input auditing in both arms,
along with qualified planning, full native usage, official reward and cleanup.
Configuration selection alone is insufficient. Use the
[updated comparison producer](compare_reply_contract_receipts.py) with
`--require-planning-schema --require-coding-contract --require-qualified` on
fresh final receipts and one frozen runtime. Failed attempts retain their costs.

No real model/provider calls, new benchmark rewards, training or model-weight
changes were made for this qualification. The extra input bytes are measured;
their effect on total tokens remains unmeasured. Keep the
[metadata/capsule/formal-status roadmap](../terminal-context-pair-20261006/roadmap.md)
and its later planning deduplication as separate treatments after reliability
qualification.
