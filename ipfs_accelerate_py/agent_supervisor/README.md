# agent_supervisor

Autonomous agent supervisor for objective-driven todo execution, control-plane
operations, formal planning/proof, merge lanes, and implementation daemons.

This root package is the **public API and package map** for the ASREF module
layout (`docs/architecture/AGENT_SUPERVISOR_MODULE_REFACTOR_PLAN.md`). Domain
code lives under named subpackages; the root `__init__.py` re-exports only
intentional symbols (control surface, stable generation-2 contracts, and
selected objective/planning helpers). New code must import from domain
packages—not from retired flat module paths.

## Public API (package root)

Import the reviewed control surface and stable manifests from the package root:

```python
from ipfs_accelerate_py.agent_supervisor import (
    Operation,
    OperationRequest,
    OperationResult,
    SupervisorControlService,
    AGENT_SUPERVISOR_V2_STABLE_EXPORTS,
    AGENT_SUPERVISOR_DOMAIN_PACKAGES,
    AGENT_SUPERVISOR_LANDED_MODULE_OWNERS,
)
```

Root `__init__.py` rules:

1. **Eager** exports are provider-free control contracts/services plus a
   reviewed set of objective, proof, and planning helpers used by transports.
2. **Lazy** exports (via `__getattr__`) keep optional providers, rollout
   modules, and heavy runners off the cold-import path.
3. Landed dual-copied modules resolve through their **domain package** path
   (see `AGENT_SUPERVISOR_LANDED_MODULE_OWNERS`). Do not reintroduce thin
   re-export stubs at former flat paths.
4. Prefer domain imports for new callers even when a root re-export exists.

## Package map

Every domain package under `ipfs_accelerate_py/agent_supervisor/`:

| Package | Bundle | Purpose | README |
|---|---|---|---|
| `core/` | `asref/core` | Shared identity, conflict graph, external completion, wrappers | [core/README.md](./core/README.md) |
| `control/` | `asref/control` | Control plane, CLI, contracts, lifecycle, permits, authz | [control/README.md](./control/README.md) |
| `task_sources/` | `asref/task-sources` | Markdown/DuckDB task sources, taskboard, queues, indexes | [task_sources/README.md](./task_sources/README.md) |
| `context/` | `asref/context` | Context compiler/contracts, decision runtime (planned / partial) | *(scaffold or flat until move)* |
| `analysis/` | `asref/analysis` | Analysis pipeline, AST, cache, retrieval, consensus (planned / partial) | *(scaffold or flat until move)* |
| `proof/` | `asref/proof` | Formal verification, provers, attestation (planned / partial) | *(scaffold or flat until move)* |
| `objectives/` | `asref/objectives` | Objective heap, daemon, tracker, backlog, goal completion | [objectives/README.md](./objectives/README.md) |
| `planning/` | `asref/planning` | Adaptive/formal planning, plan metrics/rollout, failure memory | [planning/README.md](./planning/README.md) |
| `prompt/` | `asref/prompt` | Prompt workflow, scanner, plan admission (planned / partial) | *(scaffold or flat until move)* |
| `validation/` | `asref/validation` | Proposal validation, scope adjudication, validation runtime | [validation/README.md](./validation/README.md) |
| `merge/` | `asref/merge` | Merge queue/train/resolver, checkout lock, git GC, leases | [merge/README.md](./merge/README.md) |
| `rescue/` | `asref/rescue` | Rescue orchestrator, failure policy, recovery/watchdog | [rescue/README.md](./rescue/README.md) |
| `runtime/` | `asref/runtime` | Multi-supervisor runner, artifacts, schedulers, CAS | [runtime/README.md](./runtime/README.md) |
| `self_improvement/` | `asref/self-improvement` | Self-improvement completion, v2 contracts/metrics/rollout | [self_improvement/README.md](./self_improvement/README.md) |
| `integrations/` | `asref/integrations` | External tool bridges (LLM merge fallback, goose, datasets) | *(scaffold or flat until move)* |
| `todo_daemon/` | `asref/todo-daemon` | Executable daemons (implementation, loop, git, legal, LLM) | package `__init__` / modules |

Present on disk with modules + README (landed scaffolds): `core`, `control`,
`task_sources`, `objectives`, `planning`, `validation`, `merge`, `rescue`,
`runtime`, `self_improvement`, `todo_daemon`.

Still largely flat at package root until their move batches finish: `context`,
`analysis`, `proof`, `prompt`, `integrations` module sets (see
`docs/architecture/asref/move_map.json`).

## Package dependency DAG (allowed direction)

```text
core
  ↑
control, task_sources, context, analysis, proof
  ↑
objectives, planning, validation, prompt
  ↑
merge, rescue, runtime, self_improvement
  ↑
todo_daemon.*, integrations
```

Forbidden examples: `core` → `todo_daemon`; `proof` → `todo_daemon.implementation`;
cross-package cycles. Each package README lists allowed dependents and
forbidden dependencies.

## Preferred import style

```python
# Control (transport-neutral)
from ipfs_accelerate_py.agent_supervisor.control.control_contracts import (
    Operation,
    OperationRequest,
)
from ipfs_accelerate_py.agent_supervisor.control.control_plane import (
    SupervisorControlService,
)

# Core / task sources / objectives / validation
from ipfs_accelerate_py.agent_supervisor.core.conflict_graph import ConflictGraph
from ipfs_accelerate_py.agent_supervisor.task_sources.task_source import (
    CanonicalTaskSource,
)
from ipfs_accelerate_py.agent_supervisor.objectives.objective_graph import (
    parse_goal_heap,
)
from ipfs_accelerate_py.agent_supervisor.validation.proposal_validation import (
    validate_implementation_proposal,
)

# Daemons (already package-native)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    TodoImplementationDaemon,
)
```

## Landed modules (no flat import for package-root resolution)

These stems already exist under a domain package. Package-root eager/lazy
exports import them from the package path. Prefer the package path in all new
code, tests, scripts, and entry points:

| Stem | Owner package |
|---|---|
| `authorization_logic`, `control_cli`, `control_contracts`, `control_plane`, `execution_permit`, `lifecycle_orchestrator` | `control` |
| `conflict_graph`, `external_completion`, `program_behavior`, `submodule_degradation`, `wrapper_utils` | `core` |
| `dataset_store`, `duckdb_state`, `duckdb_task_source`, `markdown_task_source`, `persistent_task_queue`, `task_identity`, `task_source`, `taskboard_store`, `todo_vector_index` | `task_sources` |
| `backlog_refinery`, `objective_daemon`, `objective_graph` | `objectives` |
| `proposal_validation` | `validation` |
| `plan_failure_memory`, `formal_planning_metrics`, `formal_planning_rollout` | `planning` |
| `checkout_lock`, `git_gc`, `merge_checkpoint`, `merge_conflict_repair`, `merge_resolver` | `merge` |
| `codex_failure_policy`, `rescue_orchestrator` | `rescue` |
| `multi_supervisor_runner` | `runtime` |
| `self_improvement_completion` | `self_improvement` (package file present; root still loads the flat module until `self_improvement/__init__.py` drops its temporary flat `self_improvement.py` re-export, which otherwise pulls `todo_daemon.llm` / optional providers onto cold import) |

Runtime constant: `AGENT_SUPERVISOR_LANDED_MODULE_OWNERS` mirrors this table.

### Import cutover policy

1. After a module lands under package `P`, callers must use
   `ipfs_accelerate_py.agent_supervisor.P.<module>`.
2. **Do not** leave long-lived re-export stubs at
   `ipfs_accelerate_py.agent_supervisor.<module>`.
3. Temporary dual copies at the flat path may remain only until callers and
   console entry points are rewritten; they are not part of the public API.
4. Assert absence after full caller cutover, for example:

```bash
rg -n 'agent_supervisor\.(control_contracts|control_plane|conflict_graph|objective_graph|proposal_validation)\b' \
  --glob '!**/__pycache__/**' \
  --glob '!docs/architecture/asref/**' \
  --glob '!**/control/**' --glob '!**/core/**' \
  --glob '!**/objectives/**' --glob '!**/validation/**'
```

## Console entry points (post-move targets)

| Console script | Canonical target |
|---|---|
| `ipfs-accelerate-agent-objective-daemon` | `objectives.objective_daemon:main` |
| `ipfs-accelerate-agent-backlog-refinery` | `objectives.backlog_refinery:main` |
| `ipfs-accelerate-agent-bundle-supervisor` | `objectives.bundle_supervisor:main` (when landed) |
| `ipfs-accelerate-agent-artifact-query` | `runtime.artifact_store:main` (when landed) |
| `ipfs-accelerate-agent-implementation-daemon` | `todo_daemon.implementation_daemon:main` |
| `ipfs-accelerate-agent-implementation-supervisor` | `todo_daemon.implementation_supervisor:main` |
| `ipfs-accelerate-agent-merge-resolver` | `merge.merge_resolver:main` |
| `ipfs-accelerate-agent-llm-merge-resolver-fallback` | `integrations` / flat until integrations lands |

Unified CLI: `control.control_cli:register_agent_cli` /
`control.control_cli:run_agent_cli`.

During dual-copy windows, `pyproject.toml` / `setup.py` may still point at a
flat module until the owning move task rewrites entry points in the same
change as the final flat-file removal.

## todo_daemon layout

`todo_daemon/` remains the executable daemon package. Planned internal splits
(from the program plan; land via `asref/todo-daemon`):

- `implementation/` — implementation daemon + supervisor
- `loop/` — supervisor loop, runner, engine pieces
- `git/` — worktrees, git utils, auto-commit

Import daemons as:

```python
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import ...
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import ...
```

## Related program docs

- Plan: `docs/architecture/AGENT_SUPERVISOR_MODULE_REFACTOR_PLAN.md`
- Move map: `docs/architecture/asref/move_map.json`
- Import inventory: `docs/architecture/asref/import_inventory.md`
- Nested monorepo products: `docs/NESTED_PACKAGES.md`

## Validation (cutover gate)

```bash
python -m pytest \
  test/api/test_agent_supervisor_todo_daemon_port.py \
  test/api/test_agent_supervisor_control_conformance_v2.py \
  test/api/test_agent_supervisor_objective_graph.py \
  test/api/test_agent_supervisor_proposal_validation.py \
  -q
```

## Status (ASREF-007 / ASREF-G090)

- Root `README.md` package map: present
- Root `__init__.py` explicit exports + package-path resolution for landed
  modules: present
- Domain package READMEs: present for landed packages listed above
- Remaining flat modules: move under their target packages per
  `move_map.json` without reintroducing flat shims
- Entry-point and caller rewrites: complete when each owning move task lands
  and flat dual copies are deleted
