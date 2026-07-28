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
[Agent Supervisor philosophy](../architecture/AGENT_SUPERVISOR_PHILOSOPHY.md),
[Architecture](../architecture/AGENT_SUPERVISOR_ARCHITECTURE.md),
[documentation hub](../architecture/agent_supervisor/README.md), and
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

## Mental model (short)

1. **Objectives** state durable intent; **taskboards** are drainable projections.  
2. **Models propose**; validation, leases, and typed evidence **admit**.  
3. **Domain packages** (`control`, `proof`, `runtime`, …) own code—not board prefixes.  
4. Discovery ≠ capability ≠ proof (see [philosophy](../architecture/AGENT_SUPERVISOR_PHILOSOPHY.md)).

Package placement and DAG: [Package map](../architecture/agent_supervisor/PACKAGE_MAP.md).  
Agent fail-closed list: [FOR_AGENTS.md](../architecture/agent_supervisor/FOR_AGENTS.md).

## Implementation provider environment

Implementation daemons select a code-edit provider from the environment.
Defaults used by multi-lane programs in this repo:

| Role | Variable | Recommended value |
| --- | --- | --- |
| Provider selection | `IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER` | `grok` (Grok Build CLI) |
| Grok model | `IPFS_ACCELERATE_AGENT_GROK_MODEL` | `grok-4.5` |
| Grok binary | `IPFS_ACCELERATE_AGENT_GROK_BIN` | path to `grok` (e.g. `~/.local/bin/grok`) |
| Grok permissions | `IPFS_ACCELERATE_AGENT_GROK_PERMISSION_MODE` | `bypassPermissions` for unattended lanes |
| Codex model (when Codex path is used) | `IPFS_ACCELERATE_AGENT_CODEX_MODEL` | `gpt-5.6-terra` |

Example `implementation.env` for a runtime directory:

```bash
export IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER=grok
export IPFS_ACCELERATE_AGENT_GROK_MODEL=grok-4.5
export IPFS_ACCELERATE_AGENT_GROK_PERMISSION_MODE=bypassPermissions
export IPFS_ACCELERATE_AGENT_GROK_BIN=/home/barberb/.local/bin/grok
export IPFS_ACCELERATE_AGENT_CODEX_MODEL=gpt-5.6-terra
```

Notes:

- `provider=grok` forces the Grok Build path when the CLI is available.  
- Codex is used when the provider is `codex` / `openai`, or as a fallback when
  Grok is not selected and `codex` is on `PATH`. Always set
  `IPFS_ACCELERATE_AGENT_CODEX_MODEL` so fallback does not inherit an unrelated
  interactive Codex default.  
- Source the env **before** starting multi-supervisor / daemons; children
  inherit process environment and do not re-read files unless you wire that
  yourself.

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

### Generation-2 stable discovery

Generation 2 publishes only the reviewed provider-free names in
`AGENT_SUPERVISOR_V2_STABLE_EXPORTS`. `V2_STABLE_EXPORTS` is its compatibility
alias, `AGENT_SUPERVISOR_V2_EXPORT_MODULES` maps every name to its owner module,
and `AGENT_SUPERVISOR_V2_PUBLIC_API_VERSION` is `2`. Do not discover v2 by
walking package modules or treating a private implementation symbol as stable.

```python
from ipfs_accelerate_py.agent_supervisor import (
    AGENT_SUPERVISOR_V2_EXPORT_MODULES,
    AGENT_SUPERVISOR_V2_PUBLIC_API_VERSION,
    AGENT_SUPERVISOR_V2_STABLE_EXPORTS,
    OPERATION_CATALOG_V2,
    Operation,
    agent_supervisor_v2_control_surface_publication,
    agent_supervisor_v2_discovery_manifest,
)

assert AGENT_SUPERVISOR_V2_PUBLIC_API_VERSION == 2
assert frozenset(AGENT_SUPERVISOR_V2_EXPORT_MODULES) == frozenset(
    AGENT_SUPERVISOR_V2_STABLE_EXPORTS
)

manifest = agent_supervisor_v2_discovery_manifest()
publication = agent_supervisor_v2_control_surface_publication()
assert manifest.operations == tuple(
    sorted(Operation, key=lambda item: item.value)
)
assert publication.catalog_id == OPERATION_CATALOG_V2.catalog_id
```

For CLI embedding, use `agent_cli_v2_discovery_manifest()` and
`v2_cli_control_surface_publication()`. For MCP registration or a tools/list
response, use the MCP adapter's
`agent_supervisor_v2_discovery_manifest()` and
`mcp_v2_control_surface_publication()`. These are equivalent static
publications of `OPERATION_CATALOG_V2`; actual control still uses the existing
CLI commands and MCP tools, each of which decodes the canonical
`OperationRequest` and dispatches directly to `SupervisorControlService`.

Run discovery before runtime capability inspection. In a fresh interpreter,
package import, stable-manifest inspection, resolving every stable member, and
repeated Python/CLI/MCP discovery must load no optional dataset, model,
analysis, or prover provider; start no process; and preserve object identity
with the declared owner modules. Discovery also must not resolve a service or
backend. A later `capabilities` operation is read-only but may inspect the
explicitly configured backend; it is not part of this cold-import guarantee.

The v1 package exports, discovery functions, CLI command names, MCP tool names,
and serialized v1 audit records remain supported. V2 reuses the canonical
`Operation`, request/result contracts, and service dispatcher; it does not
create look-alike enum members or silently upgrade persisted evidence.

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
The raw codebase inventory remains objective-agnostic and unchanged by refill
policy. Rejected admission candidates remain in the durable details artifact.
Use `--allow-unscoped-codebase-refill` only for an explicitly unscoped legacy
board; it is rejected when an objective heap is configured and is an unsafe
compatibility opt-out, not a scanner-scope flag. Goal-backed admission also
requires explicit statuses, existing parents, acyclic ancestry, and semantic
evidence beyond a path token for broad top-level directory outputs.

## Supervisor self-improvement program

The maintained self-improvement program has separate intent and execution
artifacts:

- [objective heap](../architecture/agent_supervisor_self_improvement.objectives.md);
- [executable task board](../architecture/agent_supervisor_self_improvement.todo.md);
  and
- [architecture and rollout plan](../architecture/AGENT_SUPERVISOR_SELF_IMPROVEMENT_PLAN.md).

Use task prefix `ASI-` when a daemon reads the board. The implementation
supervisor can refill it from the objective heap as the initial work drains:

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

An operating profile is a reviewed configuration plus its sizing evidence, not
a global singleton or a worker-count convention. Bind its values into
`ControlBounds`, `context_contracts.ContextBudget`,
`cache_coordinator.CacheQuotaPolicy`, `ResourcePolicy`,
`ResourceLeaseBudget`, formal-verification
`formal_verification_contracts.ResourceBudget`, and daemon CLI limits at the
point each component is constructed. Persist the profile revision, repository
and tree, host class, provider/capability snapshot, observation window, fixture
population, measured high-watermarks, reserve, and effective ceilings in
requests, receipts, cache keys, plans, and epoch bindings.

For every resource class, measure CPU saturation/time, peak RSS and GPU memory,
processes, temporary and durable bytes, model tokens/quota, provider latency,
queue delay, validation and merge pressure, and accepted throughput. Set its
effective ceiling to the minimum of the hard contract limit, policy, current
provider report, backend limit, and measured host capacity after the reviewed
reserve. Missing or stale telemetry means zero new capacity. A configured lane
or worker count is only the final admission ceiling produced by that
calculation.

The smoke recipe below uses a single serialized lane for deterministic testing.
The production values are contract/context/cache envelopes; production
concurrency must be measured.

| Setting | Deterministic smoke | Production starting point |
| --- | ---: | ---: |
| Control items / serialized bytes / text bytes / timeout | 32 / 65,536 / 4,096 / 10 s | 256 / 262,144 / 8,192 / 30 s |
| Context input / output reserve / tool reserve | 2,048 / 512 / 128 tokens | 8,192 / 2,048 / 512 tokens |
| Context items / serialized bytes | 32 / 65,536 | 128 / 262,144 |
| Coordinator cache entries / namespace bytes / entry bytes | 64 / 4 MiB / 64 KiB | 512 / 32 MiB / 256 KiB |
| Analysis cache entry / receipt bytes | 64 KiB / 48 KiB | 128 KiB / 96 KiB |
| Negative TTL / maximum TTL | 60 s / 1 h | 5 min / 24 h |
| Supervisor lanes | 1, for deterministic serialization | Measured per host/resource class; never inferred from the smoke value |
| Proof/model/artifact concurrency | 1 / 1 / 1 fixture slots | Minimum of measured class capacity, provider capacity, and the top-level/global lease |
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
- derive top-level lane and CPU-proof/model/artifact ceilings from
  representative cold, warm, independent-lane, conflicting-lane, and
  artifact-pressure measurements on the deployment host. Record the measured
  peak and reserve for each class; lower admission under host pressure,
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

### Distributed profile

Size every worker class and provider route independently. Persist per-node
CPU/RSS/GPU/process/disk ceilings and enforce a separate global
`ResourceLeaseBudget` for shared model quota, artifact I/O, validation,
persistence, and merge capacity. Do not add per-node worker maxima together:
provider pressure, conflicting paths, dependency admission, and the global
lease can only reduce aggregate width.

Qualify the measured ceiling with the independent- and conflicting-lane v2
fixtures. Accepted throughput must improve without duplicate compute,
conflict regression, resource-bound violations, or stale-fence publication.
On partition, unknown telemetry, lease loss, or stale fencing epoch, stop new
admission for the affected route; a surviving node does not inherit its
authority or quota.

### Degraded profile

Enter degraded operation whenever a dataset/model/prover provider, network
route, capability, or host resource disappears or its report becomes stale.
Recompute the profile with that capacity set to zero. Consult the canonical
operation descriptor in `OPERATION_CATALOG_V2`:

- `local_read_only` permits only the bounded local read implementation;
- `proposal_only` permits a proposal that still has no mutation authority;
- `fail_closed` rejects the operation; and
- `not_applicable` adds no fallback route.

Record the capability and reason codes and run the unavailable-provider
fixture. Never use import success, a stale cache receipt, or an alternate
provider to increase authority. Return to the production/distributed profile
only after a fresh capability report and the affected shadow checks pass.

### Recovery profile

Pause new admission and preserve journals, receipts, leases, fences, and
content-addressed artifacts. Reserve measured CPU, memory, process, and I/O
capacity for reconciliation; new-work capacity stays zero until repository
tree, state, live fence, active phase, and last terminal receipt agree.

Run the restart fixture and replay the exact request identities. An accepted
mutation replay must return its persisted result without a second backend
effect; a different request under the same idempotency key must conflict.
Incomplete validation or merge work remains bounded and actionable. Do not
delete state to speed recovery or infer completion from process exit.

### Refill profile

Observe a drained board for at least 10 minutes with at most 2,000
milli-percent idle CPU and zero unchanged-state writes. Run a v2 refill epoch
only after a meaningful binding change or an eligible scheduled observation
window. The immutable ceiling is eight goals and twenty-four tasks per epoch
with at least a six-hour cooldown; set lower `SupervisorV2Policy` values when
measured validation and materialization capacity cannot safely process that
population.

Require exact replay to be a no-op, zero duplicate successors, fresh complete
observations, and the healthy-exhaustion trigger guard. Preview before
materialization. A partial, blocked, proposal-only, or stale epoch creates no
work, and board drain does not prove objective completion.

### Rollback profile

Enter rollback on a stale rollout binding, current-tree regression, failed
later evaluation, capability loss, or resource-bound violation. Set the
affected `V2RolloutMode` to `shadow`, stop new automatic admission, and use the
measured recovery reserve to drain or quarantine already admitted work.
Persist `rollback_applied`, bounded reason codes, and the qualifying/current
evaluation identities.

Rollback is not a compensating mutation and does not edit objectives, code,
policy, or completion state. A new qualifying complete v2 evaluation and a
separate later current-tree evaluation are required before policy may restore
automatic operation.

### Migration profile

Keep v1 and v2 discovery and reads side by side in smoke/shadow with v2
mutation capacity set to zero. Measure schema/result parity, state and artifact
growth, restart/replay time, and operator load. Require v2 Python, CLI, and MCP
publications to share `OPERATION_CATALOG_V2`, the canonical `Operation`
objects, request/result schema IDs, behavior IDs, and direct-service dispatcher
identity.

Move preview/dry-run next, then authorize one mutation family at a time.
Retain v1 package imports, operation spellings, reports, and recovery paths
until the migrated family has passed cold discovery, conformance, restart, and
resource-ceiling checks. Any identity, capability, state, or recovery drift
returns it to the v1 adapter or v2 shadow. Preserve v1 evidence as audit data;
produce fresh v2 records rather than rewriting serialized v1 reports.

## Shadow, assist, and automatic rollout

`SelfImprovementRolloutMode` has exactly `shadow`, `assist`, and `automatic`.

| Mode | Allowed behavior |
| --- | --- |
| `shadow` | Run bounded candidates and write metrics/reports; do not change dispatch authority, objective state, task completion, or merge decisions. |
| `assist` | Present or queue a validated proposal for an authorized operator; no unreviewed mutation. |
| `automatic` | Apply only the narrowly approved capability after all policy, authorization, assurance, freshness, resource, and paired-rollout gates pass. |

The generation-2 contract is `V2RolloutMode`, with `off` in addition to those
three values. `V2RolloutPolicy` deliberately excludes `automatic` by default.
`evaluate_v2_self_improvement_rollout()` requires a complete qualifying
evaluation; automatic also requires an explicitly approving policy and a
separate later current-tree evaluation. A stale binding or regression makes
the effective mode `shadow` and records `rollback_applied`. The report remains
evidence, not control authorization or goal-completion proof.

### Prompt bootstrap and rescue rollout

The prompt-workflow gate uses `PromptWorkflowRolloutMode`: `off`, `shadow`,
`assist`, and `automatic`. It freezes one prompt/repository population and
requires identical admitted task CIDs, ready sets, accepted effects, and
terminal outcomes across deterministic and model planning, Markdown/DuckDB/both
task sources, and Python/CLI/script/MCP surfaces.

`prompt_workflow_benchmark.py` owns the closed paired, adversarial, and chaos
receipts. `recompute_prompt_workflow_gate()` rejects any scope, secret,
identity, SQL, process, policy, authority, effect, completion, or
mandatory-evidence escape; requires a typed resume/compensate/quarantine
outcome for every materialize/lifecycle/rescue intent-effect-receipt boundary;
and bounds tokens, model calls, retries, storage, and processes. Optional
dependency loss must degrade explicitly through a deterministic local replay
without eager provider import.

Automatic is a two-observation mode. Keep the complete qualifying evaluation,
then collect a later separate current-root evaluation. An explicit policy must
approve `behavior:prompt-workflow-bootstrap-rescue@1` and the automatic mode.
A stale binding, narrowed population, safety failure, or metric regression
returns only the affected behavior to shadow. A later separate fresh-root
evaluation is required before automatic promotion; the local frozen smoke
population is conformance evidence, not production promotion evidence.

```python
from ipfs_accelerate_py.agent_supervisor.prompt_workflow_benchmark import (
    build_frozen_prompt_workflow_benchmark,
    recompute_prompt_workflow_gate,
)
from ipfs_accelerate_py.agent_supervisor.prompt_workflow_rollout import (
    PromptWorkflowControlRequest,
    PromptWorkflowPublicAPI,
    PromptWorkflowRolloutEvaluation,
    build_default_prompt_workflow_binding,
    build_default_prompt_workflow_policy,
    evaluate_prompt_workflow_rollout,
)

qualification = PromptWorkflowRolloutEvaluation(
    "evaluation:qualification@1",
    "2026-01-01T00:00:00Z",
    build_frozen_prompt_workflow_benchmark(observation_label="qualification"),
)
current = PromptWorkflowRolloutEvaluation(
    "evaluation:current@1",
    "2026-01-02T00:00:00Z",
    build_frozen_prompt_workflow_benchmark(observation_label="current"),
)
binding = build_default_prompt_workflow_binding()
policy = build_default_prompt_workflow_policy(approve_automatic=True)

report = recompute_prompt_workflow_gate(qualification.benchmark)
assert report.passed

decision = evaluate_prompt_workflow_rollout(
    qualification,
    binding=binding,
    policy=policy,
    desired_mode="automatic",
    current_evaluation=current,
)

api = PromptWorkflowPublicAPI(
    qualification,
    binding=binding,
    policy=policy,
    current_evaluation=current,
)
request = PromptWorkflowControlRequest(action="automatic")
python_result = api.python(request)
cli_result = api.cli(request.to_dict())
mcp_result = api.mcp(request.to_dict())
assert python_result.to_dict() == cli_result.to_dict() == mcp_result.to_dict()
assert api.rollback().decision.effective_mode.value == "shadow"
```

Workflow bootstrap through the shared service (preview and mutation remain
separate authority boundaries):

```python
from ipfs_accelerate_py.agent_supervisor.prompt_workflow import (
    PromptSource,
    PromptSupervisorService,
    PromptWorkflowRequest,
)

service = PromptSupervisorService(control_service=control_service)
preview = service.preview(
    PromptWorkflowRequest(
        prompt_source=PromptSource.inline("Improve retry recovery"),
        repository_root=repo,
        directory=repo,
        output_mode="both",
        dry_run=True,
    )
)
# materialize/start require normal authorization, idempotency, and expected
# effects; bootstrap() may compose them as a receipt-linked saga only.
```

```bash
# Prefer --prompt-file or stdin so sensitive prompts avoid process listings.
ipfs-accelerate agent workflow-preview \
  --directory /path/to/repository \
  --prompt-file request.md \
  --output-mode both \
  --markdown-path plan.todo.md \
  --duckdb-path plan.duckdb

python -m ipfs_accelerate_py.agent_supervisor.prompt_workflow \
  workflow-preview --directory /path/to/repository --prompt-file request.md

ipfs-accelerate agent rescue-preview --repository-root /path/to/repository
ipfs-accelerate agent rescue --allow-llm-fallback
```

MCP discovery remains provider-free and generates catalog-equivalent tools such
as `agent_supervisor_workflow_preview`, `agent_supervisor_workflow_materialize`,
`agent_supervisor_restart`, `agent_supervisor_rescue_preview`, and
`agent_supervisor_rescue`. Tool selection never grants mutation authority.

Operator rescue order: status/health → deterministic recovery ladder →
exhaustion receipt → optional closed LLM rescue plan → one permitted action at
a time → post-effect health or quarantine. Roll back automatic/assist to shadow
immediately on parity, safety, or binding regression.

Validation:

```bash
python -m pytest \
  test/api/test_agent_supervisor_prompt_workflow_e2e.py \
  test/api/test_agent_supervisor_prompt_workflow_adversarial.py \
  test/api/test_agent_supervisor_prompt_workflow_chaos.py \
  test/api/test_agent_supervisor_prompt_workflow_rollout.py \
  test/api/test_agent_supervisor_prompt_workflow_public_api.py -q
```

### Proof-directed decision-runtime rollout

The decision-runtime gate uses the separate `DecisionRuntimeRolloutMode`
vocabulary: `off`, `shadow`, `assist`, and `automatic`. It compares the current
and proof-directed live paths on the same frozen decisions. The closed paired
population independently grows irrelevant legal corpus, codebase, SkillCenter
rows, SkillCenter graph, and conversation history by at least 10x.

Build `DecisionRuntimeProducerReceipt` values from context, runtime, cache,
invalidation, plan, proof, validation, and effect producer receipts. Do not
copy dashboard aggregates or token estimates into promotion evidence.
`recompute_proof_dependency_scaling()` derives provider tokens, mandatory
proof closure nodes/bytes, total-corpus nodes/bytes, exact warm reuse,
invalidation true/false positives and false negatives, first-valid plans,
retries, proof/validation cost, effects, and terminal results. For every scale
ablation, the proof-directed provider context, mandatory closure, effects, and
terminal result must remain fixed; only bounded index metadata may grow.

The zero-escape population covers forged CID, canonicalization, schema, stale
root, cross partition, prompt injection, poisoned embedding, inapplicable law,
legal conflict, SecurityIR deny and unknown, intent-authority confusion, dirty
file, changed tool arguments, stale lease, proof replay, graph truncation,
recovery, path and effect escape, and mandatory omission. One escape fails the
whole report. Optional provider loss must replay deterministically through the
local fail-closed path, and public discovery must remain lazy.

Automatic is a two-observation mode. Keep the complete qualifying frozen
evaluation, then collect a later separate current-root evaluation from a
distinct producer population. An explicit policy must approve the exact
behavior and automatic mode. A stale binding, narrowed population, safety
failure, or configured metric regression returns the affected behavior to
shadow.

The module-local facade provides equivalent controls without duplicating
transport policy:

```python
from ipfs_accelerate_py.agent_supervisor.decision_runtime_rollout import (
    DecisionRuntimeControlRequest,
    DecisionRuntimePublicAPI,
)

api = DecisionRuntimePublicAPI(
    qualification,
    binding=binding,
    policy=policy,
    current_evaluation=later_current_root_evaluation,
)

request = DecisionRuntimeControlRequest(action="automatic")
python_result = api.python(request)
cli_result = api.cli(request.to_dict())
mcp_result = api.mcp(request.to_dict())
assert python_result.to_dict() == cli_result.to_dict() == mcp_result.to_dict()

status = api.status()
explanation = api.explanation()
rollback = api.rollback()
assert rollback.decision.effective_mode.value == "shadow"
```

The CLI-shaped action vocabulary and MCP-shaped request object are exactly the
canonical `DecisionRuntimeControlRequest`: `off`, `shadow`, `assist`,
`automatic`, `status`, `explanation`, and `rollback`. These adapters do not
shell out, resolve a provider, or grant authority. A
`DecisionRuntimeRolloutDecision` is rollout evidence only; live mutation still
requires the exact current permit, lease/fence, expected effects, and
post-effect validation.

Evaluate baseline and candidate on the closed fixture population: cold, warm,
broad goal, contradictory input, malformed output, stale cache, unavailable
provider, independent parallel work, conflicting parallel work, failed
validation, restart, and drained refill.

```python
from ipfs_accelerate_py.agent_supervisor import (
    PAIRED_EFFICIENCY_REQUIREMENT_ID,
    SHADOW_FALSE_COMPLETION_REQUIREMENT_ID,
    PairedRolloutRequirementEvidence,
    PairedRolloutPolicy,
    PairedRolloutReportStore,
    REQUIRED_PAIRED_FIXTURE_KINDS,
    evaluate_paired_self_improvement_rollout,
)

def verify_shadow_population(fixtures, *, repository_id, tree_id, report_dir):
    # The harness supplies one paired measurement for every reviewed kind.
    kinds = tuple(item.fixture_kind for item in fixtures)
    assert len(kinds) == len(REQUIRED_PAIRED_FIXTURE_KINDS)
    assert frozenset(kinds) == frozenset(REQUIRED_PAIRED_FIXTURE_KINDS)
    report = evaluate_paired_self_improvement_rollout(
        fixtures,
        policy=PairedRolloutPolicy(),
    )  # omitted desired_mode intentionally defaults to shadow
    store = PairedRolloutReportStore(report_dir)
    store.persist(report)
    recovered = store.load(report.report_id)
    evidence = tuple(
        recovered.evidence_for(
            requirement_id,
            repository_id=repository_id,
            repository_tree=tree_id,
        )
        for requirement_id in (
            SHADOW_FALSE_COMPLETION_REQUIREMENT_ID,
            PAIRED_EFFICIENCY_REQUIREMENT_ID,
        )
    )
    restored = tuple(
        PairedRolloutRequirementEvidence.from_dict(
            item.to_dict(),
            report=recovered,
        )
        for item in evidence
    )
    assert all(item.requirement_satisfied for item in restored)
    assert recovered.effective_mode.value == "shadow"
    assert not recovered.promotion_allowed
    return recovered, restored
```

The default paired policy requires zero false completions, authority
violations, stale authoritative hits, escaped defects, duplicate executions,
and unauthorized mutations; bounded artifacts; stable restart; no quality,
coverage, accepted-work, defect-detection, false-rejection, or merge-conflict
regression; at least 35% lower median input tokens; at least 70% cache reuse on
the repeated fixtures; at least 2x independent-lane throughput; and a planning
improvement of either at least 1,000 basis points in median evidence coverage
or at least 2,000 basis points in aggregate invalid-plan-branch reduction. A
missing fixture or any failed gate forces the effective mode to `shadow`, even
when `assist` or `automatic` was requested.

Three stable objective terms make the rollout and its public adoption boundary
directly auditable:

- `109590900757783560279417463762322084165` is satisfied only when the complete
  seeded population has zero candidate false completions; seeding any false
  completion fails the non-negotiable gate and forces `shadow`.
- `146189916032404266364029134505159070240` is satisfied only when the paired
  token, repeated-cache, planning, and independent-throughput gates all pass.
- `300500866741873729474343907613893393545`, published as
  `PAIRED_ROLLOUT_LAZY_EXPORT_REQUIREMENT_ID`, is satisfied only by a fresh
  interpreter check proving package import and manifest inspection are cold,
  every unique manifest member resolves to its identical provider-free owner,
  and no optional provider is loaded. Its canonical heap owner is ASI-G114.

Use
`report.evidence_for(requirement_id, repository_id=..., repository_tree=...)`
to obtain the bounded, typed projection for either report-backed term. The
canonical ASI-G112/ASI-G113 goal is derived from the requirement and cannot be
supplied by the caller. The ASI-G114 import-isolation term is intentionally not
accepted by `evidence_for`: it is a package-interface property proved by the
fresh-process surface test, not a claim about paired measurements. An unmet
report term returns a negative diagnostic witness; an unsupported term raises
`PairedRolloutValidationError`. Restore serialized report evidence with
`PairedRolloutRequirementEvidence.from_dict(payload, report=report)`, which
re-derives the claim and rejects altered, detached, or unknown data. Neither
evidence route grants mutation authorization or goal-completion authority.

Paired report version 2 adds the explicit invalid-plan-branch measurement and
four component-gate projections. The reader still recomputes and accepts
persisted version-1 reports for audit and recovery, but a version-1 report
cannot claim the paired-efficiency term because it never measured the planning
gate. Run the current shadow population to mint a version-2 witness before
considering assist or automatic use.

The package-root `PAIRED_ROLLOUT_STABLE_EXPORTS` manifest is the authoritative
lazy surface. It groups the bounds and threshold constants; goal and
requirement IDs; schema and version constants; fixture, measurement, policy,
mode, report, evidence, store, and validation-error types; the reviewed
fixture collections; and the evaluator. Importing
`ipfs_accelerate_py.agent_supervisor` or reading the manifest does not load
optional analysis, model, dataset, or prover providers or start a process.
Accessing a listed name loads only the provider-free rollout contract module.
The adjacent `PAIRED_ROLLOUT_LAZY_EXPORT_REQUIREMENT_ID` and
`PAIRED_ROLLOUT_LAZY_EXPORT_GOAL_ID` bind that compatibility contract without
loading the rollout module. A qualifying smoke preflight must run
`test_stable_rollout_exports_remain_lazy_without_optional_providers` in its
fresh child interpreter; importing successfully in an already-warm operator
process or checking only some manifest names is not evidence.
Migrate direct imports from
`ipfs_accelerate_py.agent_supervisor.self_improvement_rollout` to the package
root; migrate a version-1 report by running a fresh version-2 shadow population,
not by editing its serialized form.

In the deterministic smoke profile, call `evidence_for` after each seeded
fault. A seeded false completion affirmatively proves the safety term only
when the complete population forces shadow; failed efficiency gates produce
negative efficiency witnesses. In production, persist both projections with
the current report, profile revision, capability snapshot, tree, objective,
and policy identities; a changed binding or a missing projection requires a
new shadow evaluation before assist or automatic use.

Promotion is capability-specific. A report permits policy to consider
promotion; it is not itself an authorization decision or completion proof.

Before requesting `assist`, and again before `automatic`, the operator's
go/no-go review must confirm:

1. the fresh-process ASI-G114 import-isolation preflight passed for the current
   package and provider inventory, and capability discovery likewise loaded no
   optional provider and started no process;
2. a current version-2 report covers every reviewed fixture in shadow;
3. both strictly restored projections are satisfied and bound to the current
   repository and tree, and are retained with the exact profile, capability
   snapshot, objective, and policy identities;
4. the persisted report reloads to the same content identity and all bounded
   reason codes have been reviewed;
5. the desired mutation separately has authorization, expected effects,
   idempotency, a live lease, and the current fence; and
6. any package, manifest, provider-inventory, tree, policy, capability, or
   profile binding change returns operation to shadow and reruns the applicable
   import-isolation preflight and paired population.

### Requesting ASI-G090 completion

Promotion authority does not complete ASI-G090. After all changes are present
on the candidate tree, run the mandatory command:

```bash
python -m pytest \
  test/api/test_agent_supervisor_self_improvement_e2e.py \
  test/api/test_agent_supervisor_self_improvement_benchmark.py -q
```

Submit the completion request through
`report.evaluate_objective_completion(...)` or
`evaluate_paired_rollout_completion(...)` with all of the following:

1. a fresh recomputed complete report and both strictly restored, satisfied
   G112/G113 requirement projections bound to the current repository/tree;
2. exactly ASI-023 and ASI-024 in terminal-success states and exactly G112,
   G113, and G114 freshly `verified_complete`, each with a passing current-tree
   gate, validation record, and conclusive current proof requirement;
3. exactly one fresh passing current-tree validation receipt for each of the
   five literal G090 clauses, plus an exact coverage row naming concrete
   implementation and that receipt CID;
4. analyzer data with `status: healthy`, `healthy: true`, and
   `safe_for_completion_reasoning: true`, bound to repository, tree,
   `ASI-G090`, `ASI-G090@asi-090`, `paired-rollout-completion@1`, and
   `paired-rollout-completion-policy@1`;
5. exactly two fresh members, each healthy, completion-safe, exhaustive, and
   identically bound, with unique member ID, evidence channel, and receipt
   CID.

Every submitted record is checked. Extra, failed, stale, future, foreign,
duplicated, or detached evidence fails the request; a passing sibling cannot
mask it. The first successful evaluation produces only
`provisionally_complete`. Run a separate later evaluation against the same
still-current proof population before accepting `verified_complete`. If the
tree/profile/provider binding changes or a child reopens, rerun the shadow
population and completion evidence collection.

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
- paired token reduction, repeated-cache reuse, planning coverage improvement,
  invalid-plan-branch reduction, independent throughput, all four component
  gates, and bounded reason codes;
- proof-runtime provider tokens, mandatory-closure versus total-corpus
  nodes/bytes, exact warm reuse, invalidation true/false positives and false
  negatives, first-valid plans, retries, proof/validation cost, effects, and
  terminal parity;
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

Recover proof-runtime benchmark and rollout state by reloading the complete
producer population and recomputing both reports. Never restore automatic from
a serialized summary alone. A stale/corrupt source identity, replayed proof,
changed root, omitted fixture, or later safety/binding regression returns the
affected behavior to shadow and requires a fresh qualification/current-root
pair.

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
| Import an internal v2 module or enumerate package globals | Import only package-root names in `AGENT_SUPERVISOR_V2_STABLE_EXPORTS`; use `AGENT_SUPERVISOR_V2_EXPORT_MODULES` to audit canonical ownership |
| Maintain separate Python, CLI, or MCP operation tables | Discover `OPERATION_CATALOG_V2` through the surface-specific v2 discovery/publication entry point |
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
Keep v1 imports and records intact throughout this sequence. V1 and v2 must
resolve the same canonical operation and request/result objects where their
surfaces overlap; a wrapper must not manufacture a second enum or translate a
v1 receipt into a v2 receipt.

## Validation

Run the proof-runtime scaling, adversarial, rollout, and public-control gate:

```bash
python -m pytest \
  test/api/test_agent_supervisor_decision_runtime_benchmark.py \
  test/api/test_agent_supervisor_decision_runtime_adversarial.py \
  test/api/test_agent_supervisor_decision_runtime_rollout.py \
  test/api/test_agent_supervisor_decision_runtime_public_api.py -q
```

Run the prompt bootstrap/rescue paired, adversarial, chaos, rollout, and
public-control gate:

```bash
python -m pytest \
  test/api/test_agent_supervisor_prompt_workflow_e2e.py \
  test/api/test_agent_supervisor_prompt_workflow_adversarial.py \
  test/api/test_agent_supervisor_prompt_workflow_chaos.py \
  test/api/test_agent_supervisor_prompt_workflow_rollout.py \
  test/api/test_agent_supervisor_prompt_workflow_public_api.py -q
```

Run the generation-2 public-surface, transport-conformance, and rollback gate:

```bash
python -m pytest \
  test/api/test_agent_supervisor_v2_public_api.py \
  test/api/test_agent_supervisor_control_conformance_v2.py \
  test/api/test_agent_supervisor_self_improvement_v2_rollout.py -q
```

The public API test includes the qualifying fresh-interpreter check. Running
the same imports in a warm application is not evidence that package import,
complete stable-member resolution, and repeated discovery start no process,
load no optional dataset/model/prover provider, and preserve canonical
identities.

Run the deterministic contract and surface-parity suite:

```bash
python -m pytest \
  test/api/test_agent_supervisor_self_improvement_e2e.py \
  test/api/test_agent_supervisor_self_improvement_benchmark.py \
  test/api/test_agent_supervisor_control_plane.py \
  test/test_unified_cli_agent_supervisor.py \
  test/mcp_server/test_agent_supervisor_tools.py -q
```

Run provider, external prover, IPFS, P2P, and hardware-dependent tests
separately with their declared capabilities and resource profile. A skipped or
unavailable optional integration must remain a typed non-authority outcome,
not a silent pass.
