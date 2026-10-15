# Bounded initial capsules

Implementation `bda0e7fa04` adds an opt-in maximum of one or two initial semantic
capsules to the full indexed supervisor. The default remains the existing
maximum of eight, subject to the existing byte limit. The complete native index,
signed instruction, smoke check, task profile and source checks are retained.
This changes advisory selection; it does not grant proof or completion authority.

The [qualification](qualification.json) records **440 passing tests across
14 files** in one final run with unchanged source hashes. The
[ledger](tests-ledger.json) lists every case. No live model, new full-daemon START
or official benchmark trial was used for this qualification.

## Complete coding prompt measurements

The [census](prompt-census.json) measures the actual text reaching the shared
router's authored provider callback. Each profile uses native indexed
preparation, admitted goal/task state, context compilation, semantic transport
`supervisor-semantic-router-input@1`, instruction restoration and router dispatch.
The native worker byte budget remains 32,768. Planning uses
`declared-contract@1`, with zero planning model calls. Learned Source384 inference
is not included. Provider counters from the fixture are excluded from the census.

| Authored functions | Budget | Selected / full capsules | Complete prompt bytes | Reduction from legacy |
| --- | --- | ---: | ---: | ---: |
| 4 | legacy, maximum 8 | 4 / 5 | 47,810 | — |
| 4 | `one-capsule@1` | 1 / 5 | 32,907 | 31.17% |
| 4 | `two-capsules@1` | 2 / 5 | 37,909 | 20.71% |
| 8 | legacy, maximum 8 | 4 / 9 | 48,411 | — |
| 8 | `one-capsule@1` | 1 / 9 | 33,508 | 30.78% |
| 8 | `two-capsules@1` | 2 / 9 | 38,510 | 20.45% |

A module is itself a capsule, so capsule counts differ from function-vector
counts. The existing byte budget selected only four capsules in each legacy
case. Required raw inputs were byte-identical across profiles, and translation
round-tripped each selected context exactly. Translation retained the required
native core while replacing its semantic evidence rows with the translated view.
The source and allocated worktree stayed unchanged. Admitted context reused the
initial indexes without new embedding calls.

Profiles use equal-length fixture paths and the same source/instruction, but
independent native identities. These are small authored-input measurements,
not replays of earlier live prompts. Bytes are not model tokens, and a smaller
initial input may cause additional retrieval or coding turns. The measurements
do not establish total token savings or benchmark correctness.

## Retained context and retrieval

Tests verify all nine retained baseline capsule blocks in an eight-function
fixture, then use the real router, registered child process and Unix socket to
retrieve a function omitted from the initial selection. Edits invalidate prior
generations and continuation cursors; rebuilding produces the changed source
hash. This retrieval-only fixture uses the same larger 65,536-byte budget for
all three arms so it can exercise realized counts of one, two and eight.
The pipeline census above keeps the native 32,768-byte budget.

Candidate retrieval rebuilds current structural capsules. Its identities differ
from baseline identities: querying an omitted baseline ID does not restore that
capsule. The worker must discover the current candidate catalog and use its
identities. Exact omitted baseline bytes and learned Source384 outputs are not
returned by this API. Candidate rebuilding also requires source-view opt-in and
supports 256 symbols, while initial indexing allows 1,024. Larger candidate
scopes refuse; the small tests do not qualify arbitrary repository retrieval.
The socket fixture uses authored same-UID/container admission and does not
qualify a new Docker owner/worker isolation boundary.

## Configuration and lifecycle

Rebuild the runtime archive, then add `--capsule-budget one-capsule@1` or
`--capsule-budget two-capsules@1` to the full-arm benchmark `prepare` command.
`execute` and `collect` use the frozen value. Opt-in profiles require current
archive source hashes and one independent signed task. No-index and multi-task
profiles reject them before expensive initialization. Omitting the option or
using `legacy` preserves existing default fields and flags.

The budget is checked through outer configuration, Harbor, container startup,
initial preparation, context reuse and publication refresh. The publication
reader now rejects changed lexical selection policies as it already did for
retrieval-ranked policies. Reuse rejects missing or inconsistent frozen budget
declarations. Unsupported fallback without the initial context also refuses.
Reports distinguish matching runtime declarations from absent or changed
declarations. A mismatch prevents a complete matched receipt while retaining
observed provider usage. This declaration check alone is not proof of actual
coding dispatch; the census separately checks delivered bytes.

This is not generally a coding-only experiment. The model planner's semantic
summary contains at most two selected capsules, so a one-capsule budget can
reduce planning evidence too. The two-capsule setting usually keeps that existing
prefix, subject to selection and byte bounds. Declared-contract and symbolic
planning return before a planning model invocation. Signed constraints and
separate formalization summaries remain intact.

## Evidence and next comparison

The [development record](development-runs.json) retains failure counts and hashes
of earlier local runs. Initial failures came from fixture imports, counting
module capsules as functions, omitting the signed profile from raw-input counts,
expecting the maximum count despite byte limits, and an overbroad process guard.
Some intermediate runs also recorded concurrent edits by other agents. Only
the final frozen run contributes to the 440-test count.

The [review](independent-review.json) found no blocking issue and records the
retrieval and planning limits above. The manifest binds source pins, commands,
JUnit, measurements and qualification records. Private databases and provider
transcripts are excluded.

The next live comparison should hold the provider, source view, transport,
planning mode, checkpoints and task fixed while varying this budget. Record
official reward, complete prompt bytes, retrieval traffic, coding turns and
bounded usage observations, with cold indexing charged to the supervisor.
Model planning and declared-contract planning require separate comparisons.
The [first fresh live comparison](../capsule-budget-live-20261014/README.md)
now records a passing legacy baseline and a one-capsule setup abort caused by a
Grok authentication refusal. It does not yet establish a capsule-budget token
advantage. The previous [live benchmark scores](../../compact_worker_benchmark.md)
remain historical observations from their separate runtime and treatments.
