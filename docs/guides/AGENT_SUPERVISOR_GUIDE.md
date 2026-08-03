# Agent Supervisor Guide

**Status:** Current

**Owner:** agent-supervisor maintainers

**Audience:** Operators, integrators, developers, and implementation agents

**Sources:** `ipfs_accelerate_py/agent_supervisor/control/`;
`ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py`;
`ipfs_accelerate_py/agent_supervisor/grok_cli_runner.py`;
`ipfs_accelerate_py/agent_supervisor/prompt/prompt_workflow.py`;
`ipfs_accelerate_py/cli.py`; supervisor module `--help` output

**Last-verified:** 2026-08-03 @ `da2c574c3`; operation catalog, module paths,
prompt workflows, and default provider routing rechecked

**Freshness triggers:** operation-catalog, daemon/provider routing, prompt
workflow, CLI/MCP adapter, package-layout, or authorization changes

The agent supervisor is a bounded control plane for objective-driven software
work. It turns reviewed objectives into typed tasks, runs implementation work
in isolated lanes, records evidence, and exposes the same control contract to
Python, the product CLI, and MCP.

The supervisor is not an authority shortcut. Model output is a proposal;
completion, mutation, merge, and automatic self-improvement remain subject to
repository and state allowlists, identity and policy bindings, deterministic
validation, fresh evidence, leases, fencing, and authorization.

## Related documentation

| Need | Document |
| --- | --- |
| Design pillars | [Philosophy](../architecture/AGENT_SUPERVISOR_PHILOSOPHY.md) |
| Control plane contracts | [CONTROL_PLANE.md](../architecture/agent_supervisor/CONTROL_PLANE.md) |
| Execution, lanes, recovery | [EXECUTION_AND_RECOVERY.md](../architecture/agent_supervisor/EXECUTION_AND_RECOVERY.md) |
| Plans, proofs, assurance | [PLANNING_AND_ASSURANCE.md](../architecture/agent_supervisor/PLANNING_AND_ASSURANCE.md) |
| Prompt-first composition | [PROMPT_FIRST_RUNTIME.md](../architecture/agent_supervisor/PROMPT_FIRST_RUNTIME.md) |
| Extend / place code | [Developer guide](../architecture/agent_supervisor/DEVELOPER_GUIDE.md) |
| Package ownership | [Package map](../architecture/agent_supervisor/PACKAGE_MAP.md) |
| Doc hub | [agent_supervisor/](../architecture/agent_supervisor/README.md) |
| Agent fail-closed list | [FOR_AGENTS.md](../architecture/agent_supervisor/FOR_AGENTS.md) |
| Code-tree entry | [Package README](../../ipfs_accelerate_py/agent_supervisor/README.md) |

This guide is the **operator / integration** surface (install, discover,
authorize, profiles, recovery). Prefer the developer guide when changing
package layout or implementing new domain features. Prefer CONTROL_PLANE and
EXECUTION_AND_RECOVERY for deep contracts; this page stays compact.

## Installation and entry points

From a source checkout:

```bash
python -m pip install -e ".[dev]"
```

The preferred operator surface is the unified product CLI:

```bash
python -m ipfs_accelerate_py.cli agent --help
# or, when installed on PATH:
ipfs-accelerate agent --help
```

Low-level execution engines remain available for objective generation, board
execution, recovery, and migrations. They are engines, not competing control
APIs—the unified service calls their package APIs through registered handlers
and never turns a typed operation into a shell string.

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

Prompt workflow (preview / materialize / restart / rescue) is also available as
a domain module entrypoint—see [Prompt workflow](#prompt-workflow-bootstrap-and-rescue).

## Mental model (short)

1. **Objectives** state durable intent; **taskboards** are drainable projections.
2. **Models propose**; validation, leases, and typed evidence **admit**.
3. **Domain packages** (`control`, `proof`, `runtime`, `prompt`, …) own code—not
   board prefixes.
4. Discovery ≠ capability ≠ proof (see
   [philosophy](../architecture/AGENT_SUPERVISOR_PHILOSOPHY.md)).

Package placement and DAG:
[Package map](../architecture/agent_supervisor/PACKAGE_MAP.md).

## Implementation provider environment

Implementation daemons select a code-edit provider from the environment. The
default route is deliberately narrower than a general provider cascade:

| Role | Variable | Recommended value |
| --- | --- | --- |
| Provider selection | `IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER` | Leave unset or set `auto` for the quota-gated default route |
| Primary model | `IPFS_ACCELERATE_AGENT_GROK_MODEL` | Exact `grok-4.5` |
| Grok binary | `IPFS_ACCELERATE_AGENT_GROK_BIN` | path to `grok` (e.g. `~/.local/bin/grok`) |
| Grok permissions | `IPFS_ACCELERATE_AGENT_GROK_PERMISSION_MODE` | `bypassPermissions` for unattended lanes |
| Quota-only fallback | daemon-owned; not an operator model override | Exact `gpt-5.6-terra`, reasoning `medium` |

Example `implementation.env` for a runtime directory:

```bash
# Leave IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER unset (the default is auto).
export IPFS_ACCELERATE_AGENT_GROK_MODEL=grok-4.5
export IPFS_ACCELERATE_AGENT_GROK_PERMISSION_MODE=bypassPermissions
export IPFS_ACCELERATE_AGENT_GROK_BIN="${HOME}/.local/bin/grok"
```

Notes:

- `auto` requires an installed, authenticated Grok CLI before dispatch and
  requires the primary model to be exactly `grok-4.5` when the fallback is
  attached. Predispatch unavailability fails closed.
- The runner may invoke Codex only after Grok's durable terminal record has
  the exact observed 402 balance-exhausted failure, mapped internally to
  `usage_pool_exhausted`. Other typed labels remain diagnostic and do not grant
  fallback authority without a validated durable format. The primary claim
  must agree with a fresh, tool-free `grok-4.5` quota probe. Model output and
  streamed stdout do not grant fallback authority. The fallback command is
  pinned to `gpt-5.6-terra` with `medium` reasoning; general Codex
  model/reasoning environment overrides do not redefine this fallback.
- The quota-routed default always runs Grok in a capability-restricted outer
  container; explicit Grok selection may use the native custom sandbox where
  it is enforceable. Only the active worktree and Grok's ephemeral state are
  writable. Its fixed tool surface permits repository read/search/edit
  operations but no arbitrary shell, web, MCP meta-tools, memory, or
  subagents. Peer-provider credentials, configuration, binaries, and runtime
  sockets are withheld. Grok still requires its own authentication and state,
  so this is not a confidentiality boundary against Grok itself. Before any
  quota fallback, the parent also requires the workspace's complete
  content/mode/symlink fingerprint to remain unchanged. If the capability
  boundary or cleanup watchdog cannot be established, dispatch fails closed
  before provider work starts. Validation commands run later in the
  supervisor, outside the model capability boundary.
- Explicit `provider=grok` forces Grok and deliberately omits fallback.
  Explicit `provider=codex` or `provider=openai` selects Codex directly and is
  not the quota-exhaustion fallback route.
- Source the env **before** starting multi-supervisor / daemons; children
  inherit process environment and do not re-read files unless you wire that
  yourself.

## One contract on Python, CLI, and MCP

The closed operation vocabulary is the 31-member `Operation` catalog. Every
surface decodes an `OperationRequest`, invokes
`SupervisorControlService.execute()`, and returns an `OperationResult`. Schema
identities come from `operation_request_json_schema()` and
`operation_result_json_schema()`. Transport adapters cannot silently add
authority or narrow the schema.

| Authority | Operations | Typical CLI names |
| --- | --- | --- |
| **Read** | `capabilities`, `status`, `health`, `metrics`, `goals`, `tasks`, `bundles`, `lanes`, `events`, `receipts`, `cache_inspect`, `artifact_query` | Same names (`cache` / `artifact` for the last two) |
| **Proposal** | `objective_preview`, `plan`, `workflow_preview`, `rescue_preview` | `preview`, `plan`, `workflow-preview`, `rescue-preview` |
| **Mutation** | `objective_refine`, `objective_reconcile`, `backlog_refill`, `workflow_materialize`, `start`, `pause`, `resume`, `drain`, `stop`, `restart`, `retry`, `cancel`, `quarantine`, `validation_replay`, `rescue` | `refine`, `reconcile`, `refill`, `workflow-create`, lifecycle names, `validation-replay`, `rescue` |

Prompt-control operations (`workflow_preview`, `workflow_materialize`,
`restart`, `rescue_preview`, `rescue`) walk the same binding, lease, effect, and
audit path as every other operation. A prompt or tool name never grants
mutation authority.

Configuration boundaries differ by surface; operation meaning does not:

- **Python** receives an explicit service and allowlists from its embedding process.
- **CLI** creates a service for the absolute repository and state roots the operator names.
- **MCP** accepts roots in a request only when the server has independently allowlisted them.

Read, proposal, failure, and authorized mutation behavior are parity-tested
across all three surfaces. Deep contract detail:
[CONTROL_PLANE.md](../architecture/agent_supervisor/CONTROL_PLANE.md).

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

manifest = ControlDiscoveryManifest(surface=ControlSurface.PYTHON)
assert set(manifest.operations) == set(Operation)
status_request_schema = operation_request_json_schema(Operation.STATUS)
status_result_schema = operation_result_json_schema(Operation.STATUS)

service = SupervisorControlService(
    repository_allowlist=("/srv/project",),
    state_allowlist=("/var/lib/ipfs-accelerate-agent/project",),
)
capabilities = service.capabilities()
assert capabilities.optional_providers_loaded is False
if not capabilities.supports(Operation.GOALS):
    raise RuntimeError("configured backend does not support objective reads")
if not capabilities.supports(Operation.WORKFLOW_PREVIEW):
    raise RuntimeError("prompt workflow preview is not registered on this backend")
```

The default repository backend implements bounded file-backed reads. Proposal
and mutation operations require an embedding runtime to register direct Python
handlers. Always test `CapabilityReport.supports()`; a command name existing
does not prove that the selected backend implements it.

Formal-analysis and prover support has a second, operation-specific handshake.
Use `probe_formal_verification_capabilities()` and the prover matrix before
routing work. Package presence, a model name, or an old successful receipt is
not a current capability.

### Generation-2 stable discovery

Generation 2 publishes only the reviewed provider-free names in
`AGENT_SUPERVISOR_PUBLIC_API_EXPORTS` (aliases:
`AGENT_SUPERVISOR_V2_STABLE_EXPORTS` / `V2_STABLE_EXPORTS`).
`AGENT_SUPERVISOR_V2_EXPORT_MODULES` maps every name to its owner module, and
`AGENT_SUPERVISOR_V2_PUBLIC_API_VERSION` is `2`. Do not discover v2 by walking
package modules or treating a private implementation symbol as stable.

Layout inventories use **product package names**, not board prefixes. Prefer
`AGENT_SUPERVISOR_PUBLIC_API_EXPORTS`, `AGENT_SUPERVISOR_DOMAIN_PACKAGES`,
`AGENT_SUPERVISOR_LANDED_MODULE_TO_PACKAGE`, and the semantic
`AGENT_SUPERVISOR_*_PACKAGES` / `*_STEMS` groups over deprecated
goal-numbered aliases. Board identifiers may still appear as string *values*
for scanners; they are not public API names.

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
assert manifest.operations == tuple(sorted(Operation, key=lambda item: item.value))
assert publication.catalog_id == OPERATION_CATALOG_V2.catalog_id
```

CLI embedding: `agent_cli_v2_discovery_manifest()` /
`v2_cli_control_surface_publication()`. MCP:
`agent_supervisor_v2_discovery_manifest()` /
`mcp_v2_control_surface_publication()`. These are static publications of
`OPERATION_CATALOG_V2`; live control still uses the existing CLI commands and
MCP tools.

## Python control

`SupervisorTarget` binds every constructed request to one repository tree,
objective revision, policy revision, state root, and caller.
`SupervisorClient` constructs reads and proposals from that binding, but
deliberately refuses to manufacture authorization for real mutations.

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
    repository_id="repo:example",
    tree_id="git:CURRENT_TREE_ID",
    objective_id="objective:example",
    objective_revision="objectives:CURRENT_CONTENT_ID",
    policy_id="policy:smoke",
    policy_revision="policy:1",
    caller="operator:local-smoke",
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
    objective_path="docs/architecture/example.objectives.md",
    limit=20,
)
tasks = client.tasks(
    todo_path="docs/architecture/example.todo.md",
    task_header_prefix="## PREFIX-",
    limit=20,
)
assert goals.succeeded and tasks.succeeded
```

Paths in operation parameters are root-relative. Objective and task-board
paths resolve beneath `repository_root`; status, health, metrics, events,
receipts, cache, and lane state normally resolve beneath `state_root`. Path
traversal and raw SQL are rejected. Reads are paginated with `limit` and
`offset`; request bounds travel with the request.

## Unified CLI

The CLI requires all nine target bindings. It never guesses a tree, objective,
policy, caller, repository root, or state root. Output is always canonical
JSON; `--output-json` selects compact output.

```bash
REPO_ROOT="$(pwd -P)"
STATE_ROOT="${REPO_ROOT}/data/agent_supervisor"
mkdir -p "${STATE_ROOT}"

BINDING=(
  --repository-root "${REPO_ROOT}"
  --state-root "${STATE_ROOT}"
  --repository-id "repo:example"
  --tree-id "git:CURRENT_TREE_ID"
  --objective-id "objective:example"
  --objective-revision "objectives:CURRENT_CONTENT_ID"
  --policy-id "policy:smoke"
  --policy-revision "policy:1"
  --caller "operator:local-smoke"
)

python -m ipfs_accelerate_py.cli agent capabilities "${BINDING[@]}" --output-json

python -m ipfs_accelerate_py.cli agent goals "${BINDING[@]}" \
  --path "docs/architecture/example.objectives.md" \
  --limit 20 --max-items 20 --output-json

python -m ipfs_accelerate_py.cli agent tasks "${BINDING[@]}" \
  --path "docs/architecture/example.todo.md" \
  --task-header-prefix "## PREFIX-" \
  --limit 20 --max-items 20 --output-json
```

`status`, `health`, `metrics`, and `events` support bounded JSON Lines watch
output with `--watch-count 1..100` and `--watch-interval-ms 0..60000`. Other
operations reject watch mode. Stable exit codes: `0` success, `1`
operation/internal failure, `2` invalid or denied request, `3` conflict,
`4` not found.

Use `--request-file` for production mutations. It preserves the exact typed
authorization and avoids shell-quoting a policy decision. Request-file and
request-building flags cannot be mixed.

## Prompt workflow (bootstrap and rescue)

Prompt bootstrap turns a free-form prompt plus a directory scan into an admitted
goal/task proposal; rescue diagnoses stuck work and optionally returns a
validated recovery proposal. Both stay inside the shared control catalog.

| Operation | Authority | Role |
| --- | --- | --- |
| `workflow_preview` | proposal | Scan/plan an admitted goal/task proposal without durable board or process effects |
| `workflow_materialize` | mutation | Materialize an admitted plan (Markdown / DuckDB / both) under normal authz |
| `restart` | mutation | Restart a fenced lifecycle under normal authz |
| `rescue_preview` | proposal | Diagnose, run or account for deterministic recovery, optionally propose LLM rescue |
| `rescue` | mutation | Apply one permitted rescue action under normal authz |

Prefer the domain module path for the thin Python entrypoint (not a retired
flat import):

```text
python -m ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow
```

That module is `ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow`. An
ops wrapper also exists at `scripts/ops/agent_supervisor/prompt_workflow.py`.

### Python (service API)

```python
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import (
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
# materialize / start / rescue require normal authorization, idempotency,
# lease/fence, and expected effects; bootstrap may compose them only as a
# receipt-linked saga.
```

### CLI and module entry

Prefer `--prompt-file` or stdin so sensitive prompts avoid process listings.

```bash
# Product CLI (catalog-aligned command names)
python -m ipfs_accelerate_py.cli agent workflow-preview \
  --directory /path/to/repository \
  --prompt-file request.md \
  --output-mode both \
  --markdown-path plan.todo.md \
  --duckdb-path plan.duckdb \
  "${BINDING[@]}"

python -m ipfs_accelerate_py.cli agent rescue-preview \
  --repository-root /path/to/repository \
  "${BINDING[@]}"

# Domain module entry (same closed catalog)
python -m ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow \
  workflow-preview \
  --directory /path/to/repository \
  --prompt-file request.md

python -m ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow \
  rescue-preview \
  --repository-root /path/to/repository
```

MCP discovery remains provider-free and publishes catalog-equivalent tools such
as `agent_supervisor_workflow_preview`, `agent_supervisor_workflow_materialize`,
`agent_supervisor_restart`, `agent_supervisor_rescue_preview`, and
`agent_supervisor_rescue`. Tool selection never grants mutation authority.

### Landed vs planned prompt-first facade

The domain `prompt` package and control operations above are the **current**
operator path. A convenience product facade
(`Supervisor.open().run(prompt)`, single-flag launch without nine bindings) is
still composition work under `entrypoints/`—see
[PROMPT_FIRST_RUNTIME.md](../architecture/agent_supervisor/PROMPT_FIRST_RUNTIME.md)
for the landed resolver/broker/registry matrix versus planned product facades.
Pre-facade friction inventory:
[PROMPT_ENTRYPOINT_BASELINE.md](../architecture/agent_supervisor/PROMPT_ENTRYPOINT_BASELINE.md).

### Operator rescue order

1. `status` / `health` / recent `events` / `receipts`
2. Deterministic recovery ladder (implementation supervisor once, leases, fences)
3. Exhaustion / blocker receipt
4. `rescue_preview` (deterministic first; optional closed LLM proposal only when policy allows)
5. One permitted `rescue` mutation at a time
6. Post-effect health check or quarantine

Roll back automatic/assist rollout modes to shadow immediately on parity,
safety, or binding regression. Deep recovery narrative:
[EXECUTION_AND_RECOVERY.md](../architecture/agent_supervisor/EXECUTION_AND_RECOVERY.md).

## MCP control

The MCP category is `agent_supervisor`. It registers one tool per canonical
operation. Each tool accepts exactly:

```json
{
  "request": {
    "operation": "status",
    "...": "the canonical OperationRequest record"
  }
}
```

Configure both allowlists before any tool invocation:

```bash
export IPFS_ACCELERATE_AGENT_REPOSITORY_ALLOWLIST="/srv/project"
export IPFS_ACCELERATE_AGENT_STATE_ALLOWLIST="/var/lib/ipfs-accelerate-agent/project"
```

Multiple roots use the platform path separator (`:` on POSIX, `;` on Windows).
An embedded server can call `configure_agent_supervisor_control(service=...)`
or supply a `service_factory`. Supplying neither fails closed when either
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
python -m ipfs_accelerate_py.cli agent pause "${BINDING[@]}" \
  --target-id "supervisor:example" \
  --reason "rollout review" \
  --requested-state "paused" \
  --expected-effects-json '[{
    "effect_id":"pause:example",
    "kind":"lifecycle_transition",
    "resource":"supervisor:example",
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
configuration file. Stale trees, expired or mismatched decisions, missing
effects, path escapes, absent idempotency, invalid leases, stale fencing
epochs, undeclared backend effects, and excess bounds return stable failures
before mutation dispatch.

Successful mutations emit an audit receipt. Replaying the same scoped request
returns the persisted result without dispatching the backend twice. Reusing an
idempotency key for different request content is a conflict.

## Generate objectives and task boards

The objective heap is the source of intent. Graphs, datasets, bundle indexes,
and todo boards are rebuildable projections. Paths and prefixes below are
examples—use the program’s own heap, board, and task-header prefix.

```bash
ipfs-accelerate-agent-objective-daemon \
  --repo-root "${REPO_ROOT}" \
  --objective-path docs/architecture/example.objectives.md \
  --todo-path docs/architecture/example.todo.md \
  --discovery-dir data/agent_supervisor/example/discovery \
  --bundle-dir data/agent_supervisor/example/bundles \
  --dataset-dir data/agent_supervisor/example/datasets \
  --graph-path data/agent_supervisor/example/objective_graph.json \
  --task-prefix "PREFIX-"
```

Add `--refine-objective-heap` for bounded child-goal refinement,
`--generate-plan-branches --plan-branch-count 3` for typed alternative plan
proposals, or `--submit-bundles` to submit generated shards to the local queue.
Run without submission first and inspect the graph and bundle index.

Plan isolated lanes before starting them:

```bash
ipfs-accelerate-agent-bundle-supervisor \
  --bundle-index-path data/agent_supervisor/example/bundles/index.json \
  --repo-root "${REPO_ROOT}" \
  --state-root data/agent_supervisor/example/state \
  --worktree-root data/agent_supervisor/example/worktrees \
  --log-dir data/agent_supervisor/example/logs \
  --once

ipfs-accelerate-agent-bundle-supervisor \
  --bundle-index-path data/agent_supervisor/example/bundles/index.json \
  --repo-root "${REPO_ROOT}" \
  --state-root data/agent_supervisor/example/state \
  --worktree-root data/agent_supervisor/example/worktrees \
  --log-dir data/agent_supervisor/example/logs \
  --start --max-lanes 4
```

`--max-lanes` is an admission ceiling. Dependencies, conflict paths, leases,
provider capacity, and resource budgets may reduce actual width.

For a single board:

```bash
# Reconcile once without invoking an implementation agent.
ipfs-accelerate-agent-implementation-daemon \
  --once \
  --todo-path docs/architecture/example.todo.md \
  --state-dir data/agent_supervisor/example/state

# Start implementation only after the dry pass and profile review.
ipfs-accelerate-agent-implementation-daemon \
  --implement --interval 300 \
  --todo-path docs/architecture/example.todo.md \
  --state-dir data/agent_supervisor/example/state

# One monitoring, recovery, and refill pass.
ipfs-accelerate-agent-implementation-supervisor \
  --once \
  --todo-path docs/architecture/example.todo.md \
  --task-prefix "PREFIX-" \
  --state-dir data/agent_supervisor/example/state
```

## Bounded backlog refill

```bash
ipfs-accelerate-agent-backlog-refinery \
  --repo-root "${REPO_ROOT}" \
  --todo-path docs/architecture/example.todo.md \
  --state-path data/agent_supervisor/example/state.json \
  --strategy-path data/agent_supervisor/example/strategy.json \
  --events-path data/agent_supervisor/example/events.jsonl \
  --objective-path docs/architecture/example.objectives.md \
  --task-prefix "PREFIX-" \
  --task-header-prefix "## PREFIX-" \
  --objective-scan --codebase-scan --retry-budget --dependency-guardrail
```

- `--objective-scan` finds unsatisfied objective evidence.
- `--codebase-scan` records bounded static findings.
- `--retry-budget` turns exhausted implementation, validation, or merge
  retries into repair work.
- `--dependency-guardrail` repairs invalid task dependencies.

With no mode flag, all available sources run. Findings are content-identified,
deduplicated, cooled down, and bounded. Refill does not prove completion and
does not authorize implementation. Use `--allow-unscoped-codebase-refill` only
for an explicitly unscoped legacy board; it is rejected when an objective heap
is configured.

## Programs layered on the control plane

Long-running efforts (self-improvement, codebase-proof, domain layout, catalog,
and related work) are **programs** that use the supervisor—they are not
alternate supervisors. Map board prefixes in
[PROGRAMS.md](../architecture/agent_supervisor/PROGRAMS.md). Keep program
objectives, boards, and sealed plans under `docs/architecture/`; protect them
during implementation lanes when they are operator inputs.

Product vocabulary is package and operation names. Board prefixes are
scheduling identity only.

## Context, cache, and resource profiles

An operating profile is a reviewed configuration plus its sizing evidence, not
a global singleton. Bind its values into `ControlBounds`, context budgets,
cache quotas, resource leases, formal-verification budgets, and daemon CLI
limits at construction time. Persist profile revision, tree, host class,
provider/capability snapshot, and measured ceilings in requests and receipts.

| Setting | Deterministic smoke | Production starting point |
| --- | ---: | ---: |
| Control items / serialized bytes / text bytes / timeout | 32 / 65,536 / 4,096 / 10 s | 256 / 262,144 / 8,192 / 30 s |
| Context input / output reserve / tool reserve | 2,048 / 512 / 128 tokens | 8,192 / 2,048 / 512 tokens |
| Coordinator cache entries / namespace bytes / entry bytes | 64 / 4 MiB / 64 KiB | 512 / 32 MiB / 256 KiB |
| Supervisor lanes | 1 (deterministic) | Measured per host; never inferred from smoke |
| Network and optional providers | Disabled | Disabled until capability and policy approval |
| Rollout | `shadow` | Start `shadow`; promote through `assist` only after paired evidence |

**Smoke:** one lane, frozen tree/objective/policy, local fixtures only, no
mutation without typed authorization. Passing smoke proves contract wiring, not
production capacity.

**Production:** measure host high-watermarks, retain cache freshness checks,
require provider telemetry before admission, and scale one dimension at a time.

**Degraded / recovery / rollback:** missing telemetry or provider loss means
zero new capacity; pause admission; preserve journals and receipts; never
delete state to “unstick” the supervisor; return automatic modes to `shadow` on
binding or safety regression.

Deep execution profiles and daemon loops:
[EXECUTION_AND_RECOVERY.md](../architecture/agent_supervisor/EXECUTION_AND_RECOVERY.md).

## Shadow, assist, and automatic rollout

Self-improvement and related gates use explicit modes. Common vocabulary:

| Mode | Allowed behavior |
| --- | --- |
| `off` / `shadow` | Observe or write metrics; no unreviewed mutation |
| `assist` | Present or queue a validated proposal for an authorized operator |
| `automatic` / `enforce` | Apply only the narrowly approved capability after all gates pass |

Automatic / enforce modes are two-observation: keep a complete qualifying
evaluation, then collect a later separate current-tree evaluation. An explicit
policy must approve the exact behavior. Stale binding, narrowed population,
safety failure, or metric regression returns the affected behavior to shadow.
A rollout report is evidence, not control authorization or goal-completion
proof.

Prompt-workflow gates reuse the same discipline (`PromptWorkflowRolloutMode`:
`off`, `shadow`, `assist`, `automatic`) with paired deterministic/model and
surface-parity populations. Endpoint-aware usage modes
(`off` / `observe` / `shadow` / `assist` / `enforce`) are separate; default is
`off` via `IPFS_ACCELERATE_SUPERVISOR_USAGE_MODE`.

## Metrics and evidence

```bash
python -m ipfs_accelerate_py.cli agent metrics "${BINDING[@]}" \
  --path "metrics/supervisor.json" \
  --output-json
```

At minimum, monitor: accepted/completed work and false completions; tokens and
cache behavior; queue depth, lanes, provider latency; validation/proof
outcomes; mutation dispatches and authz failures; retry/recovery/merge
signals; paired rollout reason codes.

Metrics are observations. Completion requires typed evidence bound to the
current repository tree, objective, policy, command/toolchain, scope, result,
artifact digest, and freshness policy. Keep raw prompts, model outputs, source
bodies, cache values, and large artifacts outside metrics and control results.

```bash
ipfs-accelerate-agent-artifact-query \
  data/agent_supervisor/example/bundles/index.json --schema

ipfs-accelerate-agent-artifact-query \
  data/agent_supervisor/example/bundles/index.json \
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
5. Use `retry` only after a transient failure or a recorded changed trigger.
6. Use `quarantine` for a task/provider/lane that must not be scheduled until
   reviewed. Use `cancel` or `stop` only with an explicit target and effects.
7. Prefer `rescue_preview` then a single authorized `rescue` over ad-hoc
   process kills. Inspect merge receipts and use the merge resolver for a
   bounded conflict.
8. Never mark a task complete because a process exited or a board drained.

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
is complete. A self-improvement epoch binds repository/tree, objective and
task-board content, policy, capability snapshot, observation window, and
ledger/strategy paths. Identical content-addressed epochs replay before
providers run. Healthy exhaustion creates no busywork. Actionable epochs may
create bounded successor goals and tasks. Blocked epochs cannot claim
exhaustion. Never edit the ledger to force a refill.

## Migration from standalone scripts

| Standalone pattern | Unified replacement |
| --- | --- |
| Enumerate package globals or flat modules | Import reviewed package-root / domain names only |
| Separate Python / CLI / MCP op tables | Discover `OPERATION_CATALOG_V2` per surface |
| Parse boards in operator scripts | `goals` / `tasks` with explicit paths and bounds |
| Shell out to supervisor scripts | Register a direct package-API handler |
| MCP request chooses arbitrary roots | Server-side repository/state allowlists |
| Lifecycle without a receipt | Typed mutation with authz, idempotency, lease/fence, effects |
| Process exit as completion | Terminal acceptance + objective evidence on current tree |
| Flat `prompt_workflow` import | `agent_supervisor.prompt.prompt_workflow` |

Recommended sequence: inventory roots and callers → freeze identities for smoke
→ move reads first → add allowlists and dry-run mutations → register handlers →
parity tests → shadow then assist → retire duplicate orchestration only after
restart recovery works.

## Validation

Control surface, prompt workflow, and core self-improvement gates (run subsets
matching the change under review):

```bash
python -m pytest \
  test/api/test_agent_supervisor_control_plane.py \
  test/api/test_agent_supervisor_prompt_control_conformance.py \
  test/api/test_agent_supervisor_prompt_workflow_e2e.py \
  test/api/test_agent_supervisor_prompt_workflow_public_api.py \
  test/api/test_agent_supervisor_v2_public_api.py \
  test/api/test_agent_supervisor_control_conformance_v2.py \
  test/test_unified_cli_agent_supervisor.py \
  test/mcp_server/test_agent_supervisor_tools.py -q
```

Primary-doc ticket-prefix guard (no board IDs as product vocabulary on primary
surfaces):

```bash
python scripts/docs/check_agent_supervisor_docs.py
```

Run provider, external prover, IPFS, P2P, and hardware-dependent tests
separately with their declared capabilities. A skipped or unavailable optional
integration must remain a typed non-authority outcome, not a silent pass.
