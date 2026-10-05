# Regular Codex eigenvalue baseline

Regular Codex finished the original `largest-eigenval` task with official verifier
reward **0.0**. Its native session completed, so all 161,750 observed tokens are
retained as the cost of this unsuccessful trial. The previously retained full
supervisor trial passed with reward **1.0**.

| Arm | Official reward | Observed total tokens | Agent execution seconds | Agent setup seconds | Overall invocation seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Regular Codex | 0.0 | 161,750 | 288.11 | 106.18 | 414.69 |
| Full indexed supervisor | 1.0 | 291,432 | 336.36 | 213.21 | 570.62 |

The supervisor used 129,682 more observed tokens, or about 1.80 times the native
count, and 48.25 more agent seconds. This is a descriptive comparison of one
trial per arm. It does not establish a reliability or efficiency advantage.

| Arm | Input tokens | Cached input, already included | Output tokens | Native sessions / router invocations |
| --- | ---: | ---: | ---: | ---: |
| Regular Codex | 154,292 | 132,608 | 7,458 | 1 |
| Full indexed supervisor | 285,558 | 244,992 | 5,874 | 2 |

Final cumulative session counters are counted once, with explicit native
`task_complete` evidence. Baseline usage also matches Harbor's job counters.
The supervisor total includes both planning and coding. Cached input is already
included in input; session counts are not individual internal model turns.
Dollar costs and verified billing totals remain unknown.

Both trials select `gpt-6.1-sol`, high reasoning and Codex CLI `0.160.0`, with
unchanged original task inputs, five CPUs, 16 GiB RAM, one attempt, one worker,
no retries, a 960-second outer agent allowance and an 1800-second setup cap.
The native session metadata confirms the actual CLI, model and reasoning.
The [inflight review](runtime-review-inflight.json) observed Docker's five-CPU
and 16 GiB RAM limits, with a 32 GiB combined RAM/swap ceiling. Original native
task defaults are one CPU, 2 GiB and 900 agent seconds; this remains a distinct
development resource profile.

Effective work allowances differ: regular Codex can use its entire 960-second
agent window; the supervisor permits 840 seconds of work, reserves 60 seconds
for cleanup and caps planning/coding calls at 90/300 seconds. Native terminal
tools, supervisor transactional worker constraints, index preparation and setup
cache treatment also differ. Both observed agent durations are below their
respective limits; this is not a repeated or randomized campaign.

The strict [control comparison](comparison-preflight.json) reports a mismatch
in the serialized retry object: the same nine exclusion names appear in a
different order. The semantic retry policies are equal, and both set
`max_retries=0`. That observation does not overwrite the strict mismatch or
turn this pilot into a fully qualified comparison campaign.

The run used Harbor's regular native Codex adapter directly, as requested,
without the supervisor, llm_router wrapper, retrieval or autoencoder inference.
The [bounded comparison](comparison.json) binds the canonical receipts,
original result and configurations by SHA-256; the
[versioned inspector](inspect_comparison.py) exports only selected metadata.
The native preparation and execution retain the frozen `e804005fc` source.
[Supervisor recovery evidence](../terminal-supervisor-blockers-20261005/README.md)
retains the earlier source, immutable archive and failed predecessor separately.
Original receipts, raw local logs and session rollouts stay in local artifacts.
No model responses, hidden verifier bodies, credentials, training or model-weight
changes are published by this comparison.
