# Live indexed supervisor and native Codex comparison

Three fresh Docker attempts on Terminal-Bench `largest-eigenval` passed the
original verifier. The indexed supervisor used 6.84% fewer reported native tokens
than the no-index supervisor, but 2.93 times native Codex's tokens. This single,
sequential pilot does not establish an overall token advantage over native Codex.

| Harness | Reward | Reported native tokens | Agent execution | Setup |
| --- | ---: | ---: | ---: | ---: |
| Native Codex | 1.0 | 113,883 | 181.69 s | 99.44 s |
| No-index supervisor | 1.0 | 358,492 | 443.08 s | 333.58 s |
| Indexed supervisor | 1.0 | 333,973 | 271.76 s | 235.39 s |

All used GPT-6.1-sol, high reasoning effort, Codex CLI 0.160.0, five CPUs,
20 GiB, the original task inputs and verifier, and the same outer time limits.
Both supervisors used `llm_router`. Grok's isolated readiness check refused
authentication, so all three attempts used the authorized Codex route. The native
harness has different internal time allocation; the intentional absence of its
numerical profile remains explicit in the control audit.

Each distinct native session contributes its final cumulative usage once.
Cached input is already included. Auxiliary refresh accounting is incomplete;
these totals are reported native usage, not exhaustive pipeline costs or billing.
See the [full report and sealed evidence](evidence/numerical-live-comparison-20261013/README.md)
for input/cache/output counters, phase costs, source pins, launch bindings and
independent reviews.

## Why the supervisor used more tokens

The retained counters show larger inputs and more model steps, not a discovered
cache double count or router retry loop. Cached input is already included in the
input column below; it must not be added again.

| Harness | Input tokens | Of which cached | Output tokens | Coding usage records / tool calls |
| --- | ---: | ---: | ---: | ---: |
| Native Codex | 107,689 | 86,016 | 6,194 | 6 / 5 |
| No-index supervisor | 346,026 | 283,392 | 12,466 | 13 / 12 |
| Indexed supervisor | 327,785 | 265,600 | 6,188 | 7 / 6 |

The supervisor rows include one additional planning session each. Planning costs
20,965 tokens without indexes and 27,526 with indexes, accounting for only 8.6%
and 12.5% of their excess over native Codex. Most of the difference is in coding.

The indexed coding input was **77,568 UTF-8 bytes**, versus **10,538 bytes** without
indexes. Its cumulative coding input divided by observed usage records is about
43,133 tokens, versus 25,132 without indexes and 17,948 for native Codex. These
averages are diagnostic proxies, not reconstructed per-request counts. Supervisor
per-step rollouts were not retained after container cleanup. The retained native
rollout has six distinct monotone usage records, with request input growing from
14,354 to 23,011 tokens.

The indexed prompt contains capsule metadata and formalization evidence in
addition to task requirements. The retained formalization evidence alone is
14,184 bytes; encoding it as JSON string chunks adds 1,228 bytes. Escaping is a
small contributor, not an explanation for the full increase. Existing compact
transport and instruction-deduplication views were not selected in this pilot;
source views were also disabled. Exact full prompt reconstruction is unavailable,
so the remaining byte distribution must not be inferred from partial artifacts.

Of the indexed arm's 220,090 extra tokens over native Codex, 179,584 are cached
input, 40,512 are uncached input, and output differs by minus six tokens. Cached
context still contributes to the token total. Uncached input is also 2.87 times
native Codex's, so cache alone does not explain away the overhead. These counts
do not measure dollar cost.

The next qualification should measure complete prompt components and consecutive
usage snapshots, then compare explicit context treatments with unchanged task
requirements. Retained owner indexes can support task-relevant retrieval without
requiring every index record in every model input. Fewer bytes in an offline
codec comparison must not be reported as a measured end-to-end token saving.

The runtime now retains bounded consecutive usage observations and tool-output
byte counts before removing a Codex container. Semantic coding receipts also
include a content-free partition of native and encoded prompt bytes. These
diagnostics stay outside the model prompt and do not change the cumulative
totals. They cannot recover missing per-step records from the historical runs.
Completion accounting also clears a prior completion after new work or an abort,
and refuses complete totals after invalid final usage or observed counter resets.
That separate defect was reproduced with authored events; it is not an
established cause of this pilot's token difference. **234 distinct regression
tests passed**, and replaying the retained native rollout preserved its exact
113,883-token total. No new live model calls or token-saving claims accompany
these diagnostics.

## Moving work out of model calls

An explicit `--planning-mode declared-contract@1` now constructs a goal, subgoal
and task directly from an independently signed single-task declaration. It runs
the existing formal-plan compiler, consistency checks and independent admission,
then materializes the admitted plan in `IntentRepository`, with **zero planning
model calls**. The complete public instruction, scope, outputs and pending
checks remain available to the worker. This removes a model call whose job was
to reproduce an already declared task structure.

The mode does not interpret arbitrary prose, prove source correctness or supply
general task decomposition. It rejects explicit Intent requirement contracts and
multi-task profiles, which retain their existing planning routes. `model` remains
the default. Preparation signs the selection; the driver and Harbor adapter
reject later changes and require a fresh runtime archive containing the selected
implementation. Live metadata census, which measures actual model delivery,
cannot select the provider-free mode. Numerical receipts distinguish validated
administrative structure from model delivery and leave source obligations pending.

Two authored repair fixtures now pass through declared planning, real indexes,
independent admission, the Lean/Z3 Doctor and worker candidate materialization,
then pass their public smoke checks without reaching `llm_router.generate_text`.
They cover keyword repair and imported-call alias repair. Doctor discovery now
honors `DOCTOR_COMPOSITION_Z3`, allowing an explicitly selected prover outside
`PATH`; proof validation still runs. The missing-prover control retains the
residual model route without manufacturing a candidate. These are narrow repair
qualifications, not Terminal-Bench scores or whole-program proofs.

The native context integration also compares the existing compact transport
with the legacy transport after real indexing, admission, database materialization
and router input construction, using authored provider responses. Both transports
restore the native prompt exactly. Four- and eight-function fixtures show:

| Observation | Result |
| --- | ---: |
| Planning provider calls, model / declared modes | 1 / 0 |
| Model planning input removed in declared mode | 20,626–20,950 bytes |
| Complete coding input reduction with compact transport | 1,022 bytes |
| Estimated coding input reduction with compact transport | 509–533 tokens (3.45–3.65%) |

Token estimates use the locally cached `o200k_base` tokenizer; its equivalence to
the selected provider's tokenizer is unverified. They exclude CLI scaffolding and
subsequent conversation turns. Isolated capsule fixtures gave larger reductions
of 6.3–9.3%, which do not describe the complete coding prompt. Instruction
deduplication reduced bytes but added 51 estimated tokens in those fixtures, so
it is not selected for the proposed comparison. The new checks made no external
model requests and do not update the historical token scores above.

For a fresh indexed treatment, add the following to the existing reviewed
benchmark **prepare** command:

```text
--planning-mode declared-contract@1
--semantic-transport-schema supervisor-semantic-router-input@2
```

Keep the metadata view at its existing `legacy` setting. To isolate the index
effect, compare full and no-index arms with the same planning mode and transport
`@1`; no-index does not support `@2`. Separately compare full `@1` against full
`@2`, then model against declared planning with the other controls fixed. Preserve
the public inputs, model route, resource profile and checkpoint selections, and
use new run directories and a newly qualified archive. A fresh live run is still
needed to measure total tokens, reward and supervisor START behavior together.

**331 distinct tests passed** across the selected regression and integration
runs. Relevant implementation and tests:
[declared planner](../../ipfs_accelerate_py/agent_supervisor/runtime/declared_contract_planning.py),
[native context comparison](../../benchmarks/agent_supervisor/container_coding/test_terminal_symbolic_efficiency.py)
and [Doctor pipeline](../../benchmarks/agent_supervisor/container_coding/test_terminal_declared_doctor_pipeline.py).

## Indexed coverage and remaining gaps

The indexed arm hydrated three DuckLake catalogs, three links and a world record,
reused four indexed symbols, and delivered four capsules to its worker. Its
8,064-byte numerical summary reached the live planner with eight pending hints.
The model made zero optional numerical retrieval requests. All 105 obligations
remain unresolved; no source correctness proof was established.

The pinned Source384 checkpoint ran without training, but all four learned
candidates were unsupported. IntentIR had no checkpoint selected and the Doctor
abstained. After publication, fresh numerical and Source384 successors were
unavailable. These observations identify decoder coverage and final-state
formalization as remaining work. Source views were disabled, so this pilot does
not test reversible whole-codebase minification.

Both supervisor attempts also lacked final output hashes. The
[output observer](../../benchmarks/agent_supervisor/container_coding/terminal_output_observation.py)
now handles the valid queued-completion lifecycle by joining the owner event to
one later, resolved reconciliation with matching identities. It preserves
settlement, custody, cleanup and output-path checks; it cannot authorize a merge
or prove correctness. **132 distinct tests passed**, including 31 new queued
event cases, before and after integrating concurrent upstream work.

That observer fix was made after the frozen benchmark runs. It does not recover
their missing hashes or establish that it fixes every possible allocation-event
failure. A new run must qualify actual byte capture. The live runtime remains
bound to accelerator `a8f57366ff4110a87f1f13d1e60992f83f689821`, datasets
`28b9ef1f0b9da51e87b1e68ee565b43681397109` and the archive recorded in the report.
