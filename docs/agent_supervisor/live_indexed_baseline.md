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
