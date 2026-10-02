# Full-system Terminal-Bench comparison

Status: the isolated native supervisor now runs the original task through
Harbor in `full` and `no-index` arms, alongside a retained native Codex baseline.
This is a one-task qualification pilot; no performance advantage is established.
The full arm builds and hydrates the permitted-input indexes. An exact input
audit verified semantic, retrieval, and world context in the trial 04 coding
invocation; trial 06 lost its capsule audit during teardown and retains that
evidence gap explicitly. Its Doctor
eligibility check currently abstains on this task, so no automatic proof-backed
repair should be inferred from the arm's name. Build-only, warm-index, separate
proof-coordination, and cooperative multi-worker comparisons remain unmeasured.

The retained v8 pilot on `fix-code-vulnerability` measured:

| Arm | Official reward | Native completion | Agent seconds | Observed total tokens |
| --- | ---: | --- | ---: | ---: |
| Native Codex harness | 1.0 | N/A | 52.421 | 286,557 |
| Supervisor, no index (trial 06) | 0.0 | Yes | 116.740 | 266,001 |
| Supervisor, full cold index (trial 06) | 0.0 | No | 249.879 | 350,588 |

These runs share the pinned model/reasoning/CLI, a 300-second outer agent limit,
and one worker in each original 1-CPU/2-GiB task container. The supervisor
reserves cleanup time inside that limit. Its native START/STOP succeeded in
both trial 06 arms with zero remaining native and worker processes. No-index06
published a model-authored patch and completed its canonical task, but failed
one official check. Full06 spent 107.245 seconds on cold context construction;
its coding call timed out at its remaining 56-second allowance. The full token
count includes observed interrupted-session usage, not verified final billing.
Both planner and coding calls are included. Setup is measured separately.

Those legacy runs have no pre-execution frozen comparison controls, so their
control match remains unknown. The collector preserves their measured results
without retrospectively certifying equivalent configurations.

The two trial 06 task containers overlapped during setup, as recorded in the
schedule receipt; this is independent task concurrency, not cooperative agent
parallelism. The workspace artifact
`artifacts/terminal_bench_supervisor/full-integration-20260929/benchmark-comparison-pilot-05/`
retains all ten executed native trials, prior failures, unknown usage and phase
timings. Fewer tokens on an incorrect task do not establish an advantage.

The older two-task Grok pilot is a legacy Markdown worker pilot. Its results
must remain separate from the indexed native-supervisor comparison.

## Planned comparison matrix

New preparations freeze hashes of the validated Harbor configuration, common
resource/time/retry/concurrency settings, model identity, and complete task
inputs. Collection rechecks the original declaration. Changed or malformed
controls fail comparison; missing legacy declarations remain unknown. Task
inventories reject linked directories, broken links, and special files. Verifier
bytes are hashed only for integrity and are never supplied to the model.

Native and supervisor setup allowances are explicitly 1,800 seconds; the agent
execution limit remains 300 seconds. Arm-specific configuration and paths are
bound within each preparation, while only common controls are compared across
arms. A configuration match does not establish runtime enforcement or benchmark
advantage. The recovered implementation passed 92 controls after the restart;
the new native preflight is a dry run with no model calls.

Use identical pinned task versions, initial container files, model, reasoning
effort, available implementation tools, concurrency, attempt count, and total
per-task wall-time budget across these arms:

| Arm | Prompt planning | Native supervisor | Index build | Retrieval | Proof coordination |
| --- | --- | --- | --- | --- | --- |
| Native Codex harness | Native Codex | No | No | Native tools | No |
| Supervisor without indexes | Goals/subgoals/tasks | Yes | No | Native tools | No |
| Supervisor build-only | Goals/subgoals/tasks | Yes | Yes | Disabled | No |
| Full indexed, without proof coordination | Goals/subgoals/tasks | Yes | Yes | Enabled | No |
| Full indexed, cold | Goals/subgoals/tasks | Yes | Yes | Enabled | Yes |
| Full indexed, warm | Goals/subgoals/tasks | Yes | Reopen permitted-input indexes | Enabled | Yes |

Cold runs include scan, embedding, indexing, planning, proof, scheduling,
implementation, validation, and merge costs. Warm runs may reuse only indexes
of the original permitted inputs, never solutions, prior patches, verifier
feedback, or previous model answers. Report warm provisioning costs separately.
Count planner, router retries, worker, and repair calls in total usage.
Report raw provider token categories; unavailable usage or prices remain null.

The first source-bearing task candidate is `fix-code-vulnerability`, with
`polyglot-c-py` as an empty-workspace control. Index the actual container files:
the former task modifies the checked-out source during its Docker build, so
indexing upstream Git blobs would expose different inputs. Official tests and
solutions stay outside all agent/index input roots.

Run all arms on every selected task; retain failures, timeouts, and setup
failures. One repetition is a qualification pilot, not a statistical advantage
claim. Compare official reward first, then total tokens, elapsed time, and cost.
Use serial task trials initially to avoid confusing inter-task concurrency with
parallel supervisor workers. A later matched concurrency sweep can measure both.

## Required evidence before a full-system run qualifies

- Prompt and admitted goal graph, including subgoals, executable tasks, and DAG.
- Actual task-source materialization and fenced supervisor START receipt.
- Live worker/daemon, validation, and merge receipts; no simulated completion.
- Populated backing catalogs for each advertised index, input hashes, index
  identities, reopen/hydration results, and measured initialization costs.
- Retrieval query/result identities and evidence of worker consumption. Merely
  registering catalogs or mirroring events does not establish retrieval use.
- Real prover invocation results with their exact obligations. Coordination
  proofs are not proofs of task-code correctness.
- Native Codex version/config, matched model/tool controls, official task result,
  and complete usage. Cross-provider comparisons must be labeled confounded.

## Fresh-state qualification, 2026-09-28

`index_preflight.py` builds a real native DatabaseRepositoryIndexer AST store,
links its real symbols through SupervisorMetaIndex, projects to real DuckLake,
closes/reopens the stores, and verifies query identities. Missing DuckLake is a
failure. It accepts only an explicit list of permitted source files and requires
a fresh output directory. It does not qualify vector/BM25/world/proof indexes,
hydrate operational authority from history, or claim supervisor consumption.

Example, from the repository root:

```bash
PYTHONPATH=. python3 benchmarks/agent_supervisor/container_coding/index_preflight.py \
  --workspace benchmarks/agent_supervisor/container_coding \
  --file portable_context.py --output /tmp/supervisor-index-qualification
```

Fresh AST/meta-index/DuckLake persistence and retrieval passed on this machine.
The profile bootstrap called its initializer with an unsupported
`repository_root` argument; it now uses observed Git repository identity and
baseline commit and supports isolated profile/lifecycle directories.

Remaining observed startup blockers:

1. A fresh repository has no authorized production scheduler configuration.
   Creating a local signed profile does not establish scheduler activation.
2. The missing `IntentRepository.plan_projection()` has been restored. All 26
   prompt API tests now pass, including durable materialization and separate
   authorized START. Eleven real-storage projection regression cases also pass.
   Completion, task-authority, event replay, and DuckDB ownership contracts
   have also been restored. The combined core regression run passed 100 tests
   with one skipped; an additional router-proposal replay regression passed.
   All 75 goal-authority cases pass across the full and targeted rerun.
   Retry feedback currently passes 44 of 47 cases. Its remaining failures
   expose missing bridge attempt-budget wiring and durable daemon retry
   transitions; these must be repaired before full-system qualification.
3. Default production composition does not supply the mutation bindings and
   runtime intent factory required for durable materialization and START.

No new paid benchmark calls or native Codex baseline measurements were made in
this qualification step. No performance advantage has been established.

## Doctor contract preflight

From the source repository root:

```bash
PYTHONPATH=. python3 scripts/ops/agent_supervisor/deterministic_doctor.py \
  --checkout-root . --runtime-contracts-only inspect
```

This bounded static check reads named source files, reports missing intent
exports and adapter methods, and checks the bridge attempt-budget signature.
It returns exit status 1 on findings. It never imports the target checkout,
invokes a model, opens its database, or claims behavioral qualification.
At that stage the checkout remained failed on the missing bridge attempt-budget
contract. The later database/index integration below restores that contract;
passing the bounded static check alone is still insufficient.

The existing deterministic doctor defaults to report-only. Its write-capable
modes require a supported repair plan, enabled policy, exact target evidence,
and a writer lease. It has no automatic model fallback to synthesize missing
APIs. Restoring these implementations from repository history is source
repair work, not an automatic doctor repair receipt.

The ordinary repository-wide doctor inspection also currently stops on
`PlanningAnalysisSecretError` for `config/lgcvf_r_and_d_authority_public_key.pem`.
The bounded contract preflight does not scan that file. This is not evidence
that the ordinary doctor or its automatic repair pipeline has been qualified.

`objectives/doctor_plan_refill.py` can turn unsupported obligations into bounded
successor proposals; it does not itself authorize mutation or completion.
`todo_daemon/native_doctor_callback.py` applies a predeclared exact edit plan.
Those paths do not provide a working automatic synthesis-and-repair loop for
the missing daemon contracts observed here.

## Write-capable Doctor follow-up

The 2026-09-28 bounded repair probe used native DoctorWorktreeAdapter writes,
Z3 plus Lean through DeterministicDoctorHammer, sealed proof reload, real AST
and DuckLake indexing, and 30 passing regression tests to fix expression-shaped
diagnostic symbols crossing the strict identifier boundary. The candidate was
assistant-authored, not tactician-synthesized. Separate native probes exercised
tactician abstention, program-graph queries, and world-model catalog persistence.
The vector lane and end-to-end automatic repair composition remain unqualified.
See the workspace artifact `doctor-auto-20260928/README.md` for exact scope,
proof assumptions, unsuccessful attempts, and remaining blockers.

## Closed immutable-export repair follow-up

The 2026-09-29 `doctor_literal_repair.py` probe generated two missing retained-callback
exports from pinned historical AST declarations, constrained by the existing
caller and fixture. Both native tactician plans were admitted; actual Z3/Lean
proofs and DoctorWorktreeAdapter isolation gated publication. The combined
regression run passed 65 tests, with eight known unrelated Quack-owner failures
deselected. It hydrated a native lexical TF-IDF index in DuckDB, queried a single
consumer-function row, and registered AST/graph/vector/world catalogs through
DuckLake metadata. This narrow vector probe is not learned semantic retrieval
or a meaningful ranking benchmark. Full-caller vector indexing still fails on
an oversized AST effect reference. The default runtime's end-to-end automatic
repair composition remains unqualified. See workspace artifact
`doctor-indexed-20260928/README.md` for receipts, failed attempts, proof scope,
and explicit exclusions. No performance advantage or baseline comparison has
been measured.

## Semantic context integration (2026-09-29)

`terminal_run.py --semantic-context` and `terminal_worker.py --semantic-context`
select an explicit **legacy-daemon-semantic-context** integration arm. The worker
binds the sibling datasets/kit source roots, builds real producer capsules and
freshness admissions from public input files, persists capsule/world metadata
through DuckLake, and attaches a task/digest-bound payload to the generated
board. The native daemon validates source freshness and consumes required
payload chunks through its normal context compiler. The arm remains
`full_indexed_arm=false`; it must not be reported as the requested complete
system. Use `--prepare-only` on the worker to exercise generation and packaging
without model dispatch.

128 focused tests plus five Harbor adapter tests passed. Larger native Doctor
planning now processes 566 findings without its former bridge/request size
failures, but all 566 abstain without independent expectations. Wire minification
here means lossless JSON whitespace removal, not proof-carrying semantic
reconstruction or measured token savings. Live admitted world snapshots,
proof/synthesis/transaction composition and retry refresh remain open. See the
workspace `integration-20260929/README.md` for the integration matrix and receipts.

## Native composition follow-up (2026-09-29)

The later [supervision integration](SUPERVISION_INTEGRATION.md) adds real scoped
reconstruction and retry refresh, required intent-world context consumed by
the daemon and native planner, and an explicitly bound Doctor composition
through sealed proof, synthesis, graph-derived impact and live transaction.
The transaction now requires executed validation evidence for its validation
obligations. Native retry deltas remain local; stateless providers receive full
verified reconstruction. This supersedes the earlier retry-refresh and
composition wiring gaps, without qualifying generic automatic repair,
accepted post-merge world refresh, production START, semantic-vector retrieval,
or the matched full-system benchmark.

## Database and learned-index follow-up (2026-09-29)

The later integration restores the bridge attempt cap and runner propagation,
ordinary typed-owner reservation/admission, durable pre-effect retry, and exact
previous-attempt diagnostic delivery. Candidate diagnostics preserve unknown
callback custody; active-worker settlement and cross-lane transfer remain open.
The cap measures coordination ordinals, not actual model-token consumption.

Complete permitted-file vectors now build and hydrate without truncating large
AST facts. Both lexical and real locally pinned MiniLM lanes replay all native
fact rows and query real DuckLake metadata. Malformed/unsupported source cannot
be reported as complete coverage. This supersedes the earlier one-function
vector workaround. Learned retrieval is qualified as an isolated local lane;
its worker consumption and task-solving advantage have not been measured.

The full START audit still finds a missing independent admission builder, absent
runtime handler composition, and required native activation/launch ownership.
Explicit admission and optional-analysis hooks now survive public prompt
service construction. These hooks do not supply those missing materials.
No new Terminal-Bench/native Codex score or token-saving result is implied by
the component qualification.
