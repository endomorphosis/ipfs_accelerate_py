# Container coding qualification and live benchmark protocol

The newer joined native integration is documented in
[SUPERVISION_INTEGRATION.md](SUPERVISION_INTEGRATION.md): isolated local profiles,
pending acceptance, semantic capsules, world snapshots, learned retrieval and
actual Doctor transactions. `run_supervision_docker.py` runs its offline native
qualification. The original scripted and live component pilots below remain
separate experiments; none establishes a Terminal-Bench/native Codex advantage.

The [IntentIR planning improvement plan](../../../docs/architecture/TERMINAL_BENCH_INTENT_PLANNING_PLAN.md)
traces the current public instruction entry point and defines requirements,
symbolic planning, and admission milestones. The first local coverage milestone
is implemented: a source-bound datasets requirement ledger, independently
authored artifact/validation groundings, checked provider bindings, and signed
coverage retained through native task contracts. A bounded reviewed-operation
symbolic route now joins the existing obligation compiler, candidate planner,
critic and formal compiler. Task-specific worker requirements and read-only
public-check progress with bounded repair nominations are also implemented.
General prompt interpretation and live benchmark evaluation remain in the backlog.

The proposed [repository proof index and codebase IR plan](../../../docs/architecture/REPOSITORY_PROOF_INDEX_AND_CODEBASE_IR_PLAN.md)
adds current-code matching, bounded per-repository autoencoder training, checked
formal properties and DuckDB/DuckLake evidence storage. Its separate backlog
defines the path from IntentIR and source evidence into symbolic planning and
signed admission; this repository-proof pipeline remains proposed.

The first [codebase IR qualification experiment](../../../docs/agent_supervisor/terminal_codebase_ir_qualification.md)
now joins real supervisor preplanning, 358-function public Bottle autoencoder
training, source-bound Z3/Lean model checks and exact native DuckDB/DuckLake
metadata replay. The completed run retains 8,723 records across 28 families.
Finite loss reduction is measured; the declared stability criterion is unmet.
Learned semantic decoding, the complete logic-family matrix and authoritative
IntentIR-to-code planning remain open.

The subsequent [learned candidate experiment](../../../docs/agent_supervisor/terminal_codebase_decoder_qualification.md)
trains a fresh native production head on exact public source, reconstructs three
function ASTs and checks both header candidates through the reviewed native
model route. Its recorded cross-entropy and fixed finite tail criterion compile
in Lean; this is numerical/model evidence, with broader typed semantics and
optimizer convergence still unproved. Source, weights and complete producer
records replay through a separate native catalog referencing the initial run.

The [conditional evidence and IntentIR control experiment](../../../docs/agent_supervisor/terminal_codebase_intent_qualification.md)
adds exact native proof-key lookup, reviewed model nominations with all behavior
still residual, and frozen materials for a separate authored symbolic control.
The full public instruction stays unresolved and is refused by planning;
repository proof facts are not admitted through the administrative profile.

The [current repository catalog lane](../../../docs/agent_supervisor/terminal_codebase_repository_qualification.md)
rebuilds actual AST/KG/contracts and lexical vectors for the signed source forest,
replays complete frozen learned inventories, and consumes the byte-bound indexed
context in a fresh native symbolic selection. A private same-HEAD dirty edit
refuses the old index, evidence and model; a source-only successor agrees with a
cold rebuild. Behavioral requirements remain residual and successor admission
is not qualified.

Pass `--intent-requirement-contract /path/to/requirements.json` to
`full_supervisor_benchmark prepare`, the container supervisor, or its preparation
command, or supply `intent_requirement_contract` in `FullSupervisorAgent` kwargs.
The `intent-plan-requirement-contract@1` artifact
must name `.supervisor-instruction.md` and bind the exact public instruction.
Selecting it requires checked coverage; a rejected interpretation or missing
binding cannot silently use direct planning. The current profile supports atomic
native goals and retains unsupported compound scope. Optional autoencoder
advice remains separate. Local coverage does not establish correct translation,
task repair, or official reward.

A version 2 requirement artifact adds `symbolic_operations`: reviewed exact
native atom matchers bound to the complete signed task population, outputs,
checks and dependencies. It selects `intent_symbolic` and makes zero planning
provider calls. Selection, schedule, critique and formal effects are recomputed
at admission and retained in native pending contracts. Conditions, alternatives
and prohibitions are unsupported in this symbolic profile. The worker still
needs its qualified execution provider. Omitting a contract selects `direct`;
version 1 selects `intent_coverage`.

The coverage preparation command needs both the accelerate and datasets
packages available to the Harbor Python. Use a newly built runtime archive
containing these modules; the earlier source review snapshot identifies the
pre-implementation checkout.

For the current original-task comparison, use the Harbor environment's Python
with `full_supervisor_benchmark`, an explicit archive from `terminal_deployment`,
and a fresh trial directory:

```bash
PYTHONPATH=. python -m benchmarks.agent_supervisor.container_coding.full_supervisor_benchmark prepare \
  --arm full --dataset /path/to/terminal-bench-2 \
  --archive /path/to/full-runtime-archive --output /path/to/fresh-full-trial
PYTHONPATH=. python -m benchmarks.agent_supervisor.container_coding.full_supervisor_benchmark execute \
  --output /path/to/fresh-full-trial
```

Use `--arm no-index` with its matched source archive for the current combined
ablation, which disables both indexes and the symbolic Doctor;
`native_codex_baseline` uses Harbor's native Codex adapter. Each execution checks
the prepared hashes and runs once. The full arm includes cold index costs and
actual worker context; unsupported Doctor repairs abstain. Official reward,
native completion, cleanup, and all observed provider usage remain separate
measurements. See [INDEXED_ABLATIONS.md](INDEXED_ABLATIONS.md) for the larger
planned matrix and unmeasured parallel-worker comparisons.

Both supervisor model-coding routes now receive the original public instruction
verbatim through an owner-selected, signed source binding, in addition to the
planner's task summary. The router verifies the source and allocated worktree
before calling the provider and records the inclusion outside semantic
minification. This prevents public output requirements from depending on whether
the worker discovers an instruction file. The harness retains bounded before/after
copies of the declared public outputs for diagnosis. Native completion remains
separate from official benchmark correctness.

For requirement-backed manifest version 4, public-instruction version 2 also
delivers exact task requirements, native atoms, source spans, prerequisite
requirements, supported global prohibitions and signed outputs/checks. The
worker verifies the public graph and receipt replay without owner profile keys;
native persistence remains checked by owner launch. The terminal observation
report includes `intent_requirements` and `intent_requirement_repair`: current,
failed, stale, unobserved or unmeasured public evidence, missing outputs, and
validation/repair nominations within signed scope. Nominations require fresh
independent admission before execution. Source meaning and official reward
remain unresolved by these reports.

[SECURITY_CHECKPOINT_PLAN.md](SECURITY_CHECKPOINT_PLAN.md) describes the portable
SecurityIR weight format, Hugging Face publication, pinned ModelManager loading,
formal-candidate training and prover-checked planning milestones. It also defines
separate Doctor/index/autoencoder ablations; the current combined arm cannot
isolate their individual contributions.

The original `run_docker.sh` below is a zero-model-call **component qualification**,
not a Terminal-Bench score or an end-to-end supervisor evaluation. It imports the
production supervisor conflict scheduler, runs seeded Python repairs, checks
their outputs, and asks real Z3 and cvc5 executables to check finite task-state
invariants. The worker is an explicitly labeled scripted oracle. It does not
launch the implementation daemon, call a model, exercise worktree merging,
or demonstrate token savings.

From the ipfs_accelerate repository:

```bash
bash benchmarks/agent_supervisor/container_coding/run_docker.sh
```

The first build downloads a Python base image and Debian solver packages.
Execution has no network, credentials, Docker socket, or writable source mount.
Only the results directory is mounted writable; repair files live in temporary
container storage. The script runs regression checks before 12 trials (three
repetitions of each configuration). Pass a different output directory as the
first argument. Requires Docker; Linux arm64 was used for the initial run.

`artifacts/container_coding_qualification/qualification.json` records:

- Real repair-test results, final source hashes, scheduling waves, elapsed time,
  solver overhead, and individual solver verdicts bound to query hashes.
- Negative controls for unvalidated completion, unsatisfied dependencies, and
  overlapping writes. Unit tests isolate write conflicts from dependencies.
- Scheduler source hash, corpus hash, Python version, and both solver versions.
- Context payload **bytes**, with token savings explicitly `null`. The scripted
  oracle does not consume context, so byte reduction is only a payload-size
  observation, not evidence that a model can solve with the shorter prompt.

The adjacent image ID, source HEAD, and dirty status identify the local run.
The image installs distribution solver versions and uses a moving base tag;
retain the built image by ID for replay. The mounted source is the current
checkout, not a sealed source release. Existing unrelated edits are preserved.

## Experiment to answer the actual efficiency question

The companion `live.py` is a small router-backed coding pilot. It uses
`ipfs_accelerate_py.llm_router` and the efficient-route catalog shared with the
model manager. It selects an available API or authenticated CLI route once and
holds provider/model fixed across the four arms. Codex and Grok CLI adapters
are invoked through `llm_router`; the harness does not invoke either executable
directly. Cross-provider fallback, local model fallback, and response-cache
sharing between calls are excluded. This is still a component harness, not the full implementation daemon.

After building the image above, run from the repository root:

```bash
PYTHONPATH=. python3 benchmarks/agent_supervisor/container_coding/live.py \
  --preflight --provider grok_cli \
  --output artifacts/container_coding_qualification/router-preflight.json
PYTHONPATH=. python3 benchmarks/agent_supervisor/container_coding/live.py \
  --provider grok_cli --output artifacts/container_coding_qualification/router-pilot.json
```

Optional `--provider` and `--model` pin a router route. For example, use
`--provider grok_cli` or `--provider codex_cli` to use existing CLI logins. Configure credentials
through the existing router configuration on the **host**. Neither credentials
nor the host environment are passed to the generated-code containers. The host
calls the router; each arm's generated code executes in a separate networkless
Docker container. No model text is evaluated by the host Python interpreter.
The pilot makes at most 16 router calls and requests at most 512 output tokens
per call with a 90-second provider timeout. CLI adapters may not implement
the output-token cap, so the request count and timeout are the operative bounds. These are request bounds, not a
dollar-budget enforcement mechanism; provider-internal retries may add requests.

Native usage is captured from the selected router provider's chat response
or its thread-local CLI usage receipt when available, including usage from responses that later fail parsing or
validation. Missing usage stays `null` and prevents token-savings claims.
Text-only responses without native usage receipts leave token counts unknown. Savings
are calculated only when both compared arms solve the full corpus and all call
usage is known. The pilot has one repetition, simple tasks, and no repair retry;
use it to validate wiring before running a substantive study. Preflight records
`blocked` and exits 2 if no supported provider is discovered. Installed and
authenticated CLI routes are eligible; mocked/local fallback routes are excluded.

Use the same tasks, model/version, token budget, timeout, CPU allocation, and
initial repository snapshot in every arm. Run several paired repetitions:

| Arm | Workers per task | State checks | Context |
| --- | ---: | --- | --- |
| A | 1 | Existing supervisor validation | Full history |
| B | 3 | Existing supervisor validation | Full history |
| C | 3 | Additional solver-gated state | Full history |
| D | 3 | Additional solver-gated state | Obligation/delta capsule |

Keep benchmark-trial concurrency separate from workers cooperating on one
task. Increasing Harbor trial concurrency alone does not test collaboration.
Do not disable existing supervisor safety/acceptance gates for an ablation.

For live runs, collect provider-native usage across **all** planner, worker,
reviewer, failed, retried, and abandoned calls. Use the supervisor's
`self_improvement/supervisor_token_ledger.py` accounting contract; missing usage
is unknown, not zero. Cached input tokens remain a subset of input usage.
Report success rate, total tokens per successful task, paired wall time,
provider cost, CPU-seconds, solver overhead, retries, conflicts, and duplicate
work. Include failures in each arm's token numerator. Report success alongside
efficiency so dropping difficult tasks cannot look like an improvement.

The supervisor stores authoritative work state. Provers check whether a proposed
transition preserves an explicit invariant; they are not the state database.
Completion still needs the benchmark's independent verifier. The qualification
checks facts in concrete snapshots; it does not prove arbitrary Python programs,
the supervisor implementation, or correctness of a natural-language translation.
Two agreeing SMT solvers are not a kernel-checked Lean proof.

For production state transitions, bind receipts to the task, source tree,
obligation, dependency versions, solver version, and checker policy. Invalidate
on changed dependencies. `unknown`, timeouts, missing tools, stale receipts,
and disagreement must block proof-required transitions. Use existing supervisor
proof routing, receipt-cache, and conformance modules rather than promoting this
small benchmark checker into production authority. Introduce Lean or another
kernel checker only for obligations with a reviewed encoding and measurable
benefit; charge its translation/proof cost to the experiment.

## Terminal-Bench integration boundary

[Harbor](https://docs.harborframework.com/core-concepts/agents/pre-integrated-agents)
supports custom agents and container task environments. Pin a dataset revision
and task subset rather than relying on a moving `latest` alias. The
[Terminal-Bench documentation](https://github.com/harbor-framework/docs/blob/main/examples/terminal-bench.mdx)
describes running it with Harbor.

The next integration is a custom Harbor agent that:

1. Receives the task instruction and a clean task environment; provisions a
   pinned supervisor checkout and the selected provider inside that environment.
2. Uses `SupervisorControlService` to materialize the task workflow, and the
   existing implementation daemon/multi-lane runner for workers, leases,
   isolated edits, validation, and merging. It must not substitute this oracle.
3. Runs under a per-trial timeout and provider budget, retaining durable state
   and usage receipts. It returns the final working tree to Harbor.
4. Lets Harbor's verifier determine task success independently and joins that
   outcome to the supervisor token ledger. It preserves raw trajectories for
   audit and calculates paired comparisons across A–D.

The general installed-supervisor Harbor adapter is **not implemented by the
qualification runner**. A limited two-task workspace adapter is provided below.
The companion router pilot accepts both API configuration and existing Grok/Codex CLI logins; it is not a replacement
for the full daemon/Harbor evaluation. Select a spend limit and enforce it through
the existing usage coordinator before scaling beyond the small pilot.
The four local modes make the scheduling/proof/context plumbing testable now,
but their timings should not be extrapolated to model performance.

## Discovery regression and initial live result

The initial pilot incorrectly intersected router discovery with an API-only
allowlist. The router itself already discovered both authenticated `codex_cli`
and `grok_cli` providers on this machine. The harness now retains those routes,
reports the discovered provider list, and uses the correct CLI allocation path.
A second defect used a prefixed allocation session ID; Grok requires a UUID
because the router forwards this ID to its native session. Each call now gets
a fresh UUID, with regression coverage for both fixes.

CLI usage is reset in the calling thread before each dispatch and read even on
failure, preventing a timeout from inheriting an earlier call's token receipt.
Runs with task failures exit 1; blocked setup exits 2. Preflight readiness is
local discovery evidence, while a completed live run verifies actual dispatch.

Render a live result as Markdown:

```bash
python3 benchmarks/agent_supervisor/container_coding/report.py \
  artifacts/container_coding_qualification/router-grok-pilot-v2.json \
  --output artifacts/container_coding_qualification/router-grok-pilot-v2.md
```

## Native objective and implementation daemon pilot

`prepare_daemon.py` seeds a disposable Git repository with failing tests and
invokes the native objective daemon to create a root goal, two subgoals, and
three task packets. `daemon_run.py` invokes the production implementation
daemon, which owns dispatch, worktrees, validation, and merge processing.

Run from the supervisor source checkout:

```bash
PYTHONPATH=. python3 benchmarks/agent_supervisor/container_coding/prepare_daemon.py \
  --output artifacts/native-daemon-trial
IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER=grok \
IPFS_ACCELERATE_AGENT_GROK_MODEL=grok-4.6 \
python3 benchmarks/agent_supervisor/container_coding/daemon_run.py \
  --repository artifacts/native-daemon-trial/repo \
  --state-root artifacts/native-daemon-trial/state --once
```

Omit `--once` to run continuously; interrupt with Ctrl-C. Each implementation
attempt has a 180-second timeout and each task a two-attempt limit. This is not
a dollar-budget cap. Optional `--plan-with-router` on preparation requests
Grok planning through the LLM router; inspect generation receipts because native
planning can fall back to deterministic plans.

The September 28 pilot generated three goals and three tasks. The native Grok
worker repaired `operations.py`, passed native validation, and merged commit
`9dc779bf47d7ba19950b5e310df9ec3277d99bc7d` through the merge queue. Both independent
Docker unittest cases passed after merging; both failed at baseline. BENCH-002
completed. A subsequent overlapping BENCH-003 attempt timed out and produced
no accepted patch; BENCH-001 and BENCH-003 remain pending. Preparation now tracks
discovery notes and bundles so Git worktrees receive that context. This setup
change was smoke-tested, but its effect on live follow-up completion is unverified.

This pilot uses the explicit legacy-Markdown task-source profile, with the
daemon on the host and a native Docker implementation worker. It is not the
activated Quack-backed prompt facade, a fully containerized supervisor, a
Terminal-Bench score, or a parallel efficiency measurement. Facade initialization
currently fails on an incompatible `repository_root` argument to
`initialize_local_profile`. All three router planning requests in this pilot
fell back to deterministic plans. The bounded daemon passes have exited.

Results and native receipts are under
`artifacts/container_coding_qualification/full_daemon/`, including
`daemon-pass-result.json`, `daemon-third-result.json`, and implementation logs.
The runner now forwards `--max-task-attempts` to its daemon constructors; the
focused regression assertion passes when invoked directly (the repository pytest configuration skips this module). The older runner test module cannot collect
because it imports the unavailable `IMPLEMENTATION_DAEMON_MODULE_SENTINEL`.

## Terminal-Bench workspace adapter

`harbor_agent.py` implements Harbor's `BaseAgent` interface for two selected
Terminal-Bench 2 coding tasks: `cancel-async-tasks` and `polyglot-c-py`.
`terminal_worker.py` generates a native root goal, subgoal, and task packet,
then runs one production implementation-daemon pass with Grok 4.6. Only a
merged deliverable is copied into Harbor's `/app`; Harbor runs the unchanged
official verifier afterward. Native validation uses a separate public-contract
smoke test, which is not benchmark completion authority.

The adapter intentionally accepts only those two tasks and requires an empty
initial `/app`. It does not silently discard files from arbitrary benchmark
environments. The public instruction is preserved, with an explicit mapping
from `/app` to the isolated Git workspace. Official tests and reference solutions
are never supplied to the implementation worker. The benchmark opts into
`IPFS_ACCELERATE_AGENT_GROK_TASK_TOOL_PROFILE=files`, retaining the sealed local
file-tool list and disabling shell, web, MCP, and subagent tools. The default
production terminal-enabled profile is unchanged. Generated context references
are made workspace-relative before board registration and Git commit. The host daemon and its native
Docker worker are separate from the Harbor task container; this profile cannot
measure general interactive terminal capabilities or environment provisioning.

Example from the outer `lift_coding` workspace:

```bash
uv venv .venvs/terminal-bench-harbor --python 3.12
uv pip install --python .venvs/terminal-bench-harbor/bin/python 'harbor==0.23.0'
git clone https://github.com/harbor-framework/terminal-bench-2.git .benchmarks/terminal-bench-2
git -C .benchmarks/terminal-bench-2 checkout 2fd12b88aafdd04a52c298e3940bcb189f9766d6
python3 external/ipfs_accelerate/benchmarks/agent_supervisor/container_coding/terminal_run.py \
  --harbor .venvs/terminal-bench-harbor/bin/harbor \
  --dataset .benchmarks/terminal-bench-2 \
  --jobs-dir artifacts/terminal_bench_supervisor \
  --job-name small-serial-pilot
```

The calling Python needs the supervisor dependencies and the existing Grok
credentials. Harbor uses its separate virtual environment. The launcher uses
`--force-build` because the dataset's published x86-64 images do not execute on
this ARM64 host; it builds the unchanged upstream Dockerfiles for the host.
Each task gets one 300-second implementation attempt; the outer subprocess has
a 480-second limit. These are time/attempt bounds, not a hard dollar cap.

Use a fresh job name for every run. `--concurrency 2` runs independent Harbor
trials concurrently; that measures inter-task concurrency, not the supervisor's
internal multi-agent decomposition. The corrected pilot runs two trials concurrently. Do not claim
speedup or token savings without paired repetitions under matching conditions.
A changed adapter constitutes a different configuration, not another identical
benchmark attempt.

`terminal_report.py JOB_DIRECTORY` creates `summary.json` and `summary.md` with
official rewards, infrastructure errors, environment/agent/verifier timings,
and raw native usage-event counters. Cache/input semantics are preserved rather
than guessed, and normalized tokens/cost remain unavailable. Harbor's zero exit
status can include errored trials; `terminal_run.py` returns 2 for infrastructure
errors, 1 for verifier failures, and 0 only when both tasks pass.

Adapter boundary tests (no model calls):

```bash
PYTHONPATH=external/ipfs_accelerate/benchmarks/agent_supervisor/container_coding \
  .venvs/terminal-bench-harbor/bin/python -m unittest -v test_terminal_adapter
```

An initial unrestricted diagnostic run was invalidated after the async worker
attempted to retrieve upstream tests and a reference solution. Its logs and
`INVALID_FOR_COMPARISON.json` are retained. It must not be used as a baseline
against the file-only rerun. Prior architecture/adapter setup errors also remain
in their original job directories; neither made model calls.

Each trial gets a distinct native goal/task prefix derived from its artifact
path. This avoids reusing the supervisor's persistent attempt counters and
completion receipts across independent trials. A no-dispatch daemon pass is
reported as an infrastructure error, not as a model attempt. Preparation can
be exercised without model calls using `terminal_worker.py --prepare-only`.
