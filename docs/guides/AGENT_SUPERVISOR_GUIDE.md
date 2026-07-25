# Agent Supervisor Guide

The agent supervisor is a bounded control plane for objective-driven software
work. It turns reviewed objectives into typed tasks, runs implementation work
in isolated lanes, records evidence, and exposes the same control contract to
Python, the product CLI, and MCP.

The supervisor is not an authority shortcut. Model output is a proposal;
completion, mutation, merge, and automatic self-improvement remain subject to
repository and state allowlists, identity and policy bindings, deterministic
validation, fresh evidence, leases, fencing, and authorization.

For the design rationale and rollout invariants, see the
[Agent Supervisor Architecture](../architecture/AGENT_SUPERVISOR_ARCHITECTURE.md)
and
[Self-Improvement Plan](../architecture/AGENT_SUPERVISOR_SELF_IMPROVEMENT_PLAN.md).

## Installation and entry points

From a source checkout:

```bash
python -m pip install -e ".[dev]"
```

The preferred operator surface is the unified product CLI:

```bash
ipfs-accelerate agent --help
```

The low-level entry points remain available for objective generation, board
execution, recovery, and migrations:

| Command | Purpose |
| --- | --- |
| `ipfs-accelerate-agent-objective-daemon` | Reconcile an objective heap and generate task, graph, dataset, and bundle projections. |
| `ipfs-accelerate-agent-backlog-refinery` | Create bounded repair or refill work from objective, code, dependency, and retry evidence. |
| `ipfs-accelerate-agent-bundle-supervisor` | Plan or start isolated lanes for bundle shards. |
| `ipfs-accelerate-agent-implementation-daemon` | Drain one Markdown task board. |
| `ipfs-accelerate-agent-implementation-supervisor` | Monitor and recover an implementation daemon. |
| `ipfs-accelerate-agent-artifact-query` | Query bounded JSON or DuckDB evidence. |
| `ipfs-accelerate-agent-merge-resolver` | Inspect a failed merge and construct a bounded repair request. |
| `ipfs-accelerate-agent-llm-merge-resolver-fallback` | Run the packaged merge-repair fallback. |

These scripts are execution engines, not competing control APIs. The unified
service can call their package APIs through registered handlers; it never
turns a typed operation into a shell string.

## One contract on Python, CLI, and MCP

The closed operation vocabulary is:

| Authority | Operations | CLI names |
| --- | --- | --- |
| Read | `capabilities`, `status`, `health`, `metrics`, `goals`, `tasks`, `bundles`, `lanes`, `events`, `receipts`, `cache_inspect`, `artifact_query` | The same names, except `cache` and `artifact` |
| Proposal | `objective_preview`, `plan` | `preview`, `plan` |
| Mutation | `objective_refine`, `objective_reconcile`, `backlog_refill`, `start`, `pause`, `resume`, `drain`, `stop`, `retry`, `cancel`, `quarantine`, `validation_replay` | `refine`, `reconcile`, `refill`, lifecycle names, and `validation-replay` |

Every surface decodes an `OperationRequest`, invokes
`SupervisorControlService.execute()`, and returns an `OperationResult`. The
request and result schema identities come from
`operation_request_json_schema()` and `operation_result_json_schema()`.
Transport adapters cannot silently add authority or narrow the schema.

The three surfaces have deliberately different configuration boundaries:

- Python receives an explicit service and explicit allowlists from its
  embedding process.
- The local CLI creates a service for the absolute repository and state roots
  named by the operator.
- MCP accepts roots in a request only when the server has independently
  allowlisted them. A tool request never configures server authority.

Read and proposal behavior, stable failure records, and authorized mutation
behavior are parity-tested across all three surfaces. A successful result on
one surface therefore has the same canonical record as the same request on the
other surfaces.

## Discover capabilities before use

Discovery and capability inspection are separate:

- A `ControlDiscoveryManifest` describes the complete, static operation and
  schema population for a surface.
- A `CapabilityReport` describes the operations the configured service backend
  can actually execute and their bounds.

Static discovery does not resolve a service, import an optional provider, scan
a repository, or start a process. Runtime capability inspection is also
read-only and reports `optional_providers_loaded` and `processes_started`.

```python
from ipfs_accelerate_py.agent_supervisor import (
    ControlDiscoveryManifest,
    ControlSurface,
    Operation,
    SupervisorControlService,
    operation_request_json_schema,
    operation_result_json_schema,
)

# Static shared schemas; no backend or optional provider is initialized.
manifest = ControlDiscoveryManifest(surface=ControlSurface.PYTHON)
assert set(manifest.operations) == set(Operation)
status_request_schema = operation_request_json_schema(Operation.STATUS)
status_result_schema = operation_result_json_schema(Operation.STATUS)

# Runtime support is explicit and scoped to allowlisted roots.
service = SupervisorControlService(
    repository_allowlist=("/srv/project",),
    state_allowlist=("/var/lib/ipfs-accelerate-agent/project",),
)
capabilities = service.capabilities()
assert capabilities.optional_providers_loaded is False
if not capabilities.supports(Operation.GOALS):
    raise RuntimeError("configured backend does not support objective reads")
```

The default repository backend implements bounded file-backed reads. Proposal
and mutation operations require an embedding runtime to register direct Python
handlers. Always test `CapabilityReport.supports()`; a command name existing
does not prove that the selected backend implements it.

Formal-analysis and prover support has a second, operation-specific handshake.
Use `probe_formal_verification_capabilities()` and the prover matrix before
routing work. Package presence, a model name, or an old successful receipt is
not a current capability. Optional `ipfs_datasets_py`, Leanstral, and external
prover integrations stay lazy until a configured operation invokes them.

## Python control

`SupervisorTarget` binds every constructed request to one repository tree,
objective revision, policy revision, state root, and caller. `SupervisorClient`
constructs reads and proposals from that binding, but deliberately refuses to
manufacture authorization for real mutations.

```python
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor import (
    ControlBounds,
    SupervisorControlService,
    SupervisorTarget,
)

repo = Path.cwd().resolve()
state = (repo / "data" / "agent_supervisor").resolve()

service = SupervisorControlService(
    repository_allowlist=(repo,),
    state_allowlist=(state,),
)
target = SupervisorTarget(
    repository_root=str(repo),
    state_root=str(state),
    repository_id="repo:ipfs-accelerate",
    tree_id="git:CURRENT_TREE_ID",
    objective_id="ASI-G090",
    objective_revision="objectives:CURRENT_CONTENT_ID",
    policy_id="policy:production",
    policy_revision="policy:1",
    caller="operator:release",
)
client = service.client(
    target,
    bounds=ControlBounds(
        max_items=50,
        max_paths=50,
        max_effects=50,
        max_serialized_bytes=262_144,
        max_text_bytes=8_192,
        timeout_ms=30_000,
    ),
)

goals = client.goals(
    objective_path="docs/architecture/"
    "agent_supervisor_self_improvement.objectives.md",
    limit=20,
)
tasks = client.tasks(
    todo_path="docs/architecture/"
    "agent_supervisor_self_improvement.todo.md",
    task_header_prefix="## ASI-",
    limit=20,
)
assert goals.succeeded and tasks.succeeded
```

Paths in operation parameters are root-relative. Objective and task-board
paths are resolved beneath `repository_root`; status, health, metrics, events,
receipts, cache, and lane state are normally resolved beneath `state_root`.
Path traversal and raw SQL are rejected. Reads are paginated with `limit` and
`offset`; request count, byte, depth, text, path, effect, and timeout bounds
travel with the request.

## Unified CLI

The CLI requires all nine target bindings. This is intentional: it never
guesses a tree, objective, policy, caller, repository root, or state root.
Agent output is always canonical JSON; `--output-json` selects compact output.

```bash
REPO_ROOT="$(pwd -P)"
STATE_ROOT="${REPO_ROOT}/data/agent_supervisor"
mkdir -p "${STATE_ROOT}"

ipfs-accelerate agent capabilities \
  --repository-root "${REPO_ROOT}" \
  --state-root "${STATE_ROOT}" \
  --repository-id "repo:ipfs-accelerate" \
  --tree-id "git:CURRENT_TREE_ID" \
  --objective-id "ASI-G090" \
  --objective-revision "objectives:CURRENT_CONTENT_ID" \
  --policy-id "policy:smoke" \
  --policy-revision "policy:1" \
  --caller "operator:local-smoke" \
  --output-json
```

Reuse the same binding flags for reads:

```bash
ipfs-accelerate agent goals \
  --repository-root "${REPO_ROOT}" \
  --state-root "${STATE_ROOT}" \
  --repository-id "repo:ipfs-accelerate" \
  --tree-id "git:CURRENT_TREE_ID" \
  --objective-id "ASI-G090" \
  --objective-revision "objectives:CURRENT_CONTENT_ID" \
  --policy-id "policy:smoke" \
  --policy-revision "policy:1" \
  --caller "operator:local-smoke" \
  --path "docs/architecture/agent_supervisor_self_improvement.objectives.md" \
  --limit 20 --max-items 20 --output-json

ipfs-accelerate agent tasks \
  --repository-root "${REPO_ROOT}" \
  --state-root "${STATE_ROOT}" \
  --repository-id "repo:ipfs-accelerate" \
  --tree-id "git:CURRENT_TREE_ID" \
  --objective-id "ASI-G090" \
  --objective-revision "objectives:CURRENT_CONTENT_ID" \
  --policy-id "policy:smoke" \
  --policy-revision "policy:1" \
  --caller "operator:local-smoke" \
  --path "docs/architecture/agent_supervisor_self_improvement.todo.md" \
  --task-header-prefix "## ASI-" \
  --limit 20 --max-items 20 --output-json
```

`status`, `health`, `metrics`, and `events` support bounded JSON Lines watch
output with `--watch-count 1..100` and
`--watch-interval-ms 0..60000`. Other operations reject watch mode. Stable
exit codes are `0` for success, `1` for operation/internal failure, `2` for an
invalid or denied request, `3` for conflict, and `4` for not found.

Use `--request-file` for production mutations. It preserves the exact typed
authorization and avoids shell quoting a policy decision. Request-file and
request-building flags cannot be mixed.

## MCP control

The MCP category is `agent_supervisor`. It registers one tool per canonical
operation using the operation value as its name, for example
`agent_supervisor/status`, `agent_supervisor/tasks`, and
`agent_supervisor/objective_reconcile`. Each tool accepts exactly:

```json
{
  "request": {
    "operation": "status",
    "...": "the canonical OperationRequest record"
  }
}
```

The MCP server must configure both allowlists before any tool invocation:

```bash
export IPFS_ACCELERATE_AGENT_REPOSITORY_ALLOWLIST="/srv/project"
export IPFS_ACCELERATE_AGENT_STATE_ALLOWLIST="/var/lib/ipfs-accelerate-agent/project"
```

Multiple roots use the platform path separator (`:` on POSIX, `;` on
Windows). An embedded server can instead call
`configure_agent_supervisor_control(service=...)` or supply a
`service_factory`. Supplying neither makes invocation fail closed when either
allowlist is absent.

Listing the category, tools, tags, or schemas is safe. Service resolution
occurs only after a tool receives and successfully decodes a request.
Mutation tools are tagged as authorization-required, dry-run, idempotent,
lease-fenced, and audit-receipt-producing.

## Authorization and mutation workflow

Read operations need a complete target binding and bounds. Proposal operations
have proposal authority only. A dry-run mutation may describe mutation-shaped
effects, but it cannot invoke a mutation backend or claim that an effect was
applied.

Use dry-run first:

```bash
ipfs-accelerate agent pause \
  --repository-root "${REPO_ROOT}" \
  --state-root "${STATE_ROOT}" \
  --repository-id "repo:ipfs-accelerate" \
  --tree-id "git:CURRENT_TREE_ID" \
  --objective-id "ASI-G090" \
  --objective-revision "objectives:CURRENT_CONTENT_ID" \
  --policy-id "policy:production" \
  --policy-revision "policy:1" \
  --caller "operator:release" \
  --target-id "supervisor:self-improvement" \
  --reason "rollout review" \
  --requested-state "paused" \
  --expected-effects-json '[{
    "effect_id":"pause:self-improvement",
    "kind":"lifecycle_transition",
    "resource":"supervisor:self-improvement",
    "paths":["supervisor.json"],
    "description":"Pause dispatch for rollout review"
  }]' \
  --dry-run --output-json
```

A real mutation additionally requires all of the following:

1. An `IdempotencyKey` scoped to the exact operation, caller, repository, and
   objective.
2. A current `AuthorizationDecision` with verdict `permit`, mutation
   authority, the exact repository/state/tree/objective/policy/caller binding,
   and exactly the declared effect IDs.
3. A lease ID and non-negative fencing epoch matching the authorization.
4. At least one declared mutation `ExpectedEffect`.
5. A backend handler for the operation and, by conservative default, a lease
   validator.

Policy should create the decision; do not turn authorization into a static
configuration file. Decisions may carry an expiry and grant IDs. Stale trees,
expired or mismatched decisions, missing effects, path escapes, absent
idempotency, invalid leases, stale fencing epochs, undeclared backend effects,
and excess bounds return stable failures before mutation dispatch.

Successful mutations emit an audit receipt. Replaying the same scoped request
returns the persisted result without dispatching the backend twice. Reusing an
idempotency key for different request content is a conflict.

## Generate objectives and task boards

The objective heap is the source of intent. Graphs, datasets, bundle indexes,
and todo boards are rebuildable projections.

```bash
ipfs-accelerate-agent-objective-daemon \
  --repo-root "${REPO_ROOT}" \
  --objective-path \
    docs/architecture/agent_supervisor_self_improvement.objectives.md \
  --todo-path \
    docs/architecture/agent_supervisor_self_improvement.todo.md \
  --discovery-dir data/agent_supervisor/self_improvement/discovery \
  --bundle-dir data/agent_supervisor/self_improvement/bundles \
  --dataset-dir data/agent_supervisor/self_improvement/datasets \
  --graph-path data/agent_supervisor/self_improvement/objective_graph.json \
  --task-prefix "ASI-"
```

Add `--refine-objective-heap` for bounded child-goal refinement,
`--generate-plan-branches --plan-branch-count 3` for typed alternative plan
proposals, or `--submit-bundles` to submit generated shards to the local queue.
Run without submission first and inspect the graph and bundle index.

Plan isolated lanes before starting them:

```bash
ipfs-accelerate-agent-bundle-supervisor \
  --bundle-index-path \
    data/agent_supervisor/self_improvement/bundles/index.json \
  --repo-root "${REPO_ROOT}" \
  --state-root data/agent_supervisor/self_improvement/state \
  --worktree-root data/agent_supervisor/self_improvement/worktrees \
  --log-dir data/agent_supervisor/self_improvement/logs \
  --once

ipfs-accelerate-agent-bundle-supervisor \
  --bundle-index-path \
    data/agent_supervisor/self_improvement/bundles/index.json \
  --repo-root "${REPO_ROOT}" \
  --state-root data/agent_supervisor/self_improvement/state \
  --worktree-root data/agent_supervisor/self_improvement/worktrees \
  --log-dir data/agent_supervisor/self_improvement/logs \
  --start --max-lanes 4
```

`--max-lanes` is an admission ceiling. Dependencies, conflict paths, leases,
provider capacity, and CPU, memory, disk, GPU, process, model, and artifact
budgets may reduce actual width.

For a single board:

```bash
# Reconcile once without invoking an implementation agent.
ipfs-accelerate-agent-implementation-daemon \
  --once \
  --todo-path \
    docs/architecture/agent_supervisor_self_improvement.todo.md \
  --state-dir data/agent_supervisor/self_improvement/state

# Start implementation only after the dry pass and profile review.
ipfs-accelerate-agent-implementation-daemon \
  --implement --interval 300 \
  --todo-path \
    docs/architecture/agent_supervisor_self_improvement.todo.md \
  --state-dir data/agent_supervisor/self_improvement/state

# Run one monitoring, recovery, and refill pass.
ipfs-accelerate-agent-implementation-supervisor \
  --once \
  --todo-path \
    docs/architecture/agent_supervisor_self_improvement.todo.md \
  --task-prefix "ASI-" \
  --state-dir data/agent_supervisor/self_improvement/state
```

## Bounded backlog refill

The backlog refinery can run four evidence sources:

```bash
ipfs-accelerate-agent-backlog-refinery \
  --repo-root "${REPO_ROOT}" \
  --todo-path \
    docs/architecture/agent_supervisor_self_improvement.todo.md \
  --state-path data/agent_supervisor/self_improvement/state.json \
  --strategy-path data/agent_supervisor/self_improvement/strategy.json \
  --events-path data/agent_supervisor/self_improvement/events.jsonl \
  --objective-path \
    docs/architecture/agent_supervisor_self_improvement.objectives.md \
  --task-prefix "ASI-" \
  --task-header-prefix "## ASI-" \
  --objective-scan --codebase-scan --retry-budget --dependency-guardrail
```

- `--objective-scan` finds unsatisfied objective evidence.
- `--codebase-scan` records bounded static findings.
- `--retry-budget` turns exhausted implementation, validation, or merge
  retries into repair work.
- `--dependency-guardrail` repairs invalid task dependencies.

With no mode flag, all available sources run. Findings are content-identified,
deduplicated, cooled down, and bounded by configured open-work and finding
limits. Refill does not prove completion and does not authorize implementation.

The implementation supervisor can run objective refill after a drained pass:

```bash
ipfs-accelerate-agent-implementation-supervisor \
  --once \
  --todo-path \
    docs/architecture/agent_supervisor_self_improvement.todo.md \
  --task-prefix "ASI-" \
  --state-dir data/agent_supervisor/self_improvement/state \
  --worktree-root data/agent_supervisor/self_improvement/worktrees \
  --objective-refill-scan \
  --objective-path \
    docs/architecture/agent_supervisor_self_improvement.objectives.md \
  --objective-graph-path \
    data/agent_supervisor/self_improvement/objective_graph.json \
  --objective-bundle-dir \
    data/agent_supervisor/self_improvement/bundles \
  --objective-dataset-dir \
    data/agent_supervisor/self_improvement/datasets \
  --objective-discovery-dir \
    data/agent_supervisor/self_improvement/discovery \
  --objective-todo-vector-index-path \
    data/agent_supervisor/self_improvement/bundles/todo_vector_index.json
```

## Context, cache, and resource profiles

An operating profile is a reviewed recipe, not a global singleton. Bind its
values into `ControlBounds`, `context_contracts.ContextBudget`,
`cache_coordinator.CacheQuotaPolicy`, `ResourcePolicy`,
`ResourceLeaseBudget`, formal-verification
`formal_verification_contracts.ResourceBudget`, and daemon CLI limits at the
point each component is constructed. Persist the profile revision in requests,
receipts, cache keys, plans, and epoch bindings.

Two supported starting recipes follow. They are ceilings, not targets.

| Setting | Deterministic smoke | Production starting point |
| --- | ---: | ---: |
| Control items / serialized bytes / text bytes / timeout | 32 / 65,536 / 4,096 / 10 s | 256 / 262,144 / 8,192 / 30 s |
| Context input / output reserve / tool reserve | 2,048 / 512 / 128 tokens | 8,192 / 2,048 / 512 tokens |
| Context items / serialized bytes | 32 / 65,536 | 128 / 262,144 |
| Cache entries / namespace bytes / entry bytes | 64 / 4 MiB / 64 KiB | 512 / 32 MiB / 256 KiB |
| Negative TTL / maximum TTL | 60 s / 1 h | 5 min / 24 h |
| Supervisor lanes | 1 | 4, reduced by admission telemetry |
| Proof/model/artifact concurrency | 1 / 1 / 1 | 2 / 1 / 2, never above the top-level lease |
| Network and optional providers | Disabled | Disabled until capability and policy approval |
| Rollout | `shadow` | Start `shadow`; promote through `assist` only after paired evidence |

### Deterministic smoke profile

Use the smoke profile in CI, migration rehearsals, and recovery checks:

- one lane, one process, one proof route, and no adaptive scheduling;
- frozen tree, objective, policy, capability snapshot, fixture inputs, and
  clock where the test harness supports it;
- local deterministic handlers or fixtures only; network remains false;
- small context and cache bounds, with cache directories isolated per test;
- no implementation mutation unless the test explicitly supplies typed
  authorization, lease validation, and a temporary repository;
- paired rollout requested as `shadow`.

This profile is intentionally too small for broad production goals. A passing
smoke run proves contract wiring and deterministic recovery, not production
capacity or automatic-rollout eligibility.

### Production profile

Start production conservatively:

- retain the default `ControlBounds` and
  `context_contracts.ContextBudget` ceilings shown above;
- negotiate the effective context limit against the current provider report
  and reserve output/tool tokens before compiling context;
- retain namespaced cache identity dimensions and freshness checks; never
  convert a draft, negative, stale, or capability-mismatched hit into
  authoritative evidence;
- cap each coordinator cache namespace at 512 entries, 32 MiB, and 256 KiB per
  entry initially. The analysis-receipt cache retains its stricter defaults of
  128 KiB per entry and 96 KiB per receipt; size from measured eviction and
  reuse rather than disabling quotas;
- begin with four top-level lanes and concurrency two for CPU-proof, one for
  model, and two for artifact classes; lower admission under host pressure,
  provider latency, quota reserve, merge age, or conflicting paths;
- use explicit non-zero wall-time, memory, disk, process, output, token, and
  provider-quota limits in each production plan. A zero in `ResourceBudget`
  means “not declared,” not a safe finite limit;
- leave network access false unless the exact provider operation and egress
  are policy-approved;
- require provider telemetry and current capability probes before admission;
- run in shadow, retain paired reports, and require operator review in assist
  before considering automatic behavior.

Scale one dimension at a time. Increasing context, lane width, prover
portfolio width, model concurrency, or cache TTL changes policy and invalidates
receipts or cache entries bound to the previous profile.

## Shadow, assist, and automatic rollout

`SelfImprovementRolloutMode` has exactly `shadow`, `assist`, and `automatic`.

| Mode | Allowed behavior |
| --- | --- |
| `shadow` | Run bounded candidates and write metrics/reports; do not change dispatch authority, objective state, task completion, or merge decisions. |
| `assist` | Present or queue a validated proposal for an authorized operator; no unreviewed mutation. |
| `automatic` | Apply only the narrowly approved capability after all policy, authorization, assurance, freshness, resource, and paired-rollout gates pass. |

Evaluate baseline and candidate on the closed fixture population: cold, warm,
broad goal, contradictory input, malformed output, stale cache, unavailable
provider, independent parallel work, conflicting parallel work, failed
validation, restart, and drained refill.

```python
from ipfs_accelerate_py.agent_supervisor import (
    PairedRolloutPolicy,
    PairedRolloutReportStore,
    SelfImprovementRolloutMode,
    evaluate_paired_self_improvement_rollout,
)

# `fixtures` must contain one typed PairedRolloutFixture for every required
# PairedFixtureKind, with baseline and candidate measured on identical input.
report = evaluate_paired_self_improvement_rollout(
    fixtures,
    desired_mode=SelfImprovementRolloutMode.ASSIST,
    policy=PairedRolloutPolicy(),
)
PairedRolloutReportStore(
    "data/agent_supervisor/self_improvement/paired_rollout"
).persist(report)

if report.effective_mode is SelfImprovementRolloutMode.SHADOW:
    print(report.reason_codes)
```

The default paired policy requires zero false completions, authority
violations, stale authoritative hits, escaped defects, duplicate executions,
and unauthorized mutations; bounded artifacts; stable restart; no quality,
coverage, accepted-work, defect-detection, false-rejection, or merge-conflict
regression; at least 35% lower median input tokens; at least 70% cache reuse on
the repeated fixtures; and at least 2x independent-lane throughput. A missing
fixture or any failed gate forces the effective mode to `shadow`, even when
`assist` or `automatic` was requested.

Promotion is capability-specific. A report permits policy to consider
promotion; it is not itself an authorization decision or completion proof.

## Metrics and evidence

Expose metrics through the control operation with an explicit state-relative
path:

```bash
ipfs-accelerate agent metrics \
  --repository-root "${REPO_ROOT}" \
  --state-root "${STATE_ROOT}" \
  --repository-id "repo:ipfs-accelerate" \
  --tree-id "git:CURRENT_TREE_ID" \
  --objective-id "ASI-G090" \
  --objective-revision "objectives:CURRENT_CONTENT_ID" \
  --policy-id "policy:production" \
  --policy-revision "policy:1" \
  --caller "operator:release" \
  --path "metrics/supervisor.json" \
  --output-json
```

At minimum, monitor:

- accepted and completed work, false completions, and evidence coverage;
- input tokens, context truncation/inclusion reasons, delta retries, and cost;
- cache lookup, hit, miss, rejection, stale rejection, eviction, and
  single-flight behavior by namespace;
- queue depth, admitted lanes, resource pressure, provider latency/quota, lane
  throughput, conflicts, and starvation age;
- validation/proof outcomes, false rejection, seeded/detected/escaped defects,
  prover quarantine, and freshness;
- mutation dispatches, authorization failures, audit receipts, idempotent
  replays, stale leases/fences, and unauthorized mutations;
- retry counts, heartbeat age, recovery decisions, merge conflicts, and
  terminal acceptance;
- self-refill epoch status, blocker codes, successor counts, replay, and
  healthy exhaustion.

Metrics are observations. Completion requires typed evidence bound to the
current repository tree, objective, policy, command/toolchain, scope, result,
artifact digest, and freshness policy. Keep raw prompts, model outputs, source
bodies, cache values, and large artifacts outside metrics and control results.

Use bounded artifact inspection:

```bash
ipfs-accelerate-agent-artifact-query \
  data/agent_supervisor/self_improvement/bundles/index.json --schema

ipfs-accelerate-agent-artifact-query \
  data/agent_supervisor/self_improvement/bundles/index.json \
  --table bundles --limit 20
```

## Failure recovery

Recovery is evidence-preserving and bounded:

1. Read `health`, `status`, recent `events`, and `receipts`; check heartbeat
   age, lease/fence ownership, tree identity, and the last terminal receipt.
2. Use `pause` to stop new dispatch while retaining resumability, or `drain`
   to finish admitted work without accepting more.
3. Run the implementation supervisor once. It reconciles state, stale
   heartbeats, retry budgets, worktree ownership, and refill hooks.
4. Use `validation_replay` only with the exact validation and tree binding.
   Changed evidence requires a new request, not replay of an old success.
5. Use `retry` only after a transient failure or a recorded changed trigger.
   Idempotency prevents duplicate execution.
6. Use `quarantine` for a task/provider/lane that must not be scheduled until
   reviewed. Use `cancel` or `stop` only with an explicit target and effects.
7. Inspect merge receipts and use the merge resolver for a bounded conflict;
   never mark a task complete because a process exited or a board drained.

Do not delete state to “unstick” the supervisor. JSONL control receipts,
content-addressed artifacts, epoch ledgers, lease records, and materialization
journals are the restart and audit boundary. If state is corrupt, preserve it,
start from a new explicit state root, and reconcile against the unchanged
objective and repository tree.

Stable error codes distinguish invalid requests, denied authority, conflict,
not found, cancellation, timeout, and unavailable operations. Provider
unavailability degrades to a typed local fallback or rejection; it never
grants another provider more authority.

## Self-refill epochs

A drained board is a trigger for reconciliation, not proof that the objective
is complete. `run_self_improvement_epoch()` binds one epoch to:

- the repository and tree;
- objective and task-board content;
- self-improvement policy;
- capability snapshot;
- observation window and operator revision; and
- ledger, strategy, materialization, discovery, and bundle control paths.

The observation provider must return fresh typed observations for the closed
efficiency, planning, validation, cache, throughput, control, and safety
dimensions. Observations are read-only. A proposal provider is called only for
a blocker-free actionable epoch, and admitted successor work is bounded,
quality-checked, deduplicated across lifecycle states, cooled down, and
materialized transactionally.

An identical content-addressed epoch replays before providers run. A healthy
epoch with complete independent evidence records healthy exhaustion, creates
no busywork, and waits for a meaningful trigger: a changed tree, objective,
policy, or capability snapshot; stale evidence; a measured regression; an
operator revision; or the scheduled observation window. An actionable epoch
can create bounded successor goals and tasks. A blocked or incomplete epoch
cannot claim exhaustion.

Keep the epoch ledger and strategy projection under the state root and include
their receipt IDs in operations and dashboards. Never edit the ledger to force
a refill.

## Migration from standalone scripts

Existing deployments can migrate incrementally:

| Standalone pattern | Unified replacement |
| --- | --- |
| Read objective Markdown directly | `goals` with explicit `objective_path`, bounds, and target binding |
| Parse the todo board in an operator script | `tasks` with explicit `todo_path` and `task_header_prefix` |
| Read status, metrics, events, cache, or bundle JSON ad hoc | Corresponding bounded read operation |
| Shell out from Python to a supervisor script | Register a direct package-API handler and call `SupervisorControlService` |
| Let an MCP request choose arbitrary roots | Configure server-side repository/state allowlists |
| Invoke a lifecycle command without a receipt | Typed mutation request with authorization, idempotency, lease/fence, effects, and audit receipt |
| Treat successful process exit as completion | Require terminal acceptance and objective evidence bound to the current tree |
| Refill whenever the board is empty | Objective reconciliation, then a content-addressed self-refill epoch |

Recommended sequence:

1. Inventory current roots, callers, scripts, state files, task prefixes, and
   mutation effects.
2. Freeze repository, objective, policy, and capability identities for a smoke
   run.
3. Move reads first and compare canonical results with the old projections.
4. Add server allowlists and policy-produced authorization; keep all new
   mutation behavior in dry-run and shadow.
5. Register package handlers for the standalone engines. Keep scripts as
   operator break-glass entry points, not as hidden subprocesses behind the
   service.
6. Run the deterministic smoke profile, restart/replay checks, and full
   Python/CLI/MCP parity tests.
7. Run paired production fixtures in shadow, then assist. Consider automatic
   use only after the complete gate passes and policy approves the exact
   capability.
8. Retire duplicate parsing and shell orchestration only after receipts and
   recovery have been verified from a clean restart.

Standalone commands do not automatically gain unified authorization,
idempotency, or MCP allowlists. During migration, protect them with the same
filesystem ownership, process isolation, explicit paths, and operator policy;
do not assume equivalence until their direct package handler is parity-tested.

## Validation

Run the deterministic contract and surface-parity suite:

```bash
python -m pytest \
  test/api/test_agent_supervisor_self_improvement_e2e.py \
  test/api/test_agent_supervisor_control_plane.py \
  test/test_unified_cli_agent_supervisor.py \
  test/mcp_server/test_agent_supervisor_tools.py -q
```

Run provider, external prover, IPFS, P2P, and hardware-dependent tests
separately with their declared capabilities and resource profile. A skipped or
unavailable optional integration must remain a typed non-authority outcome,
not a silent pass.
