# agent_supervisor

Autonomous agent supervisor for objective-driven todo execution, control-plane
operations, formal planning/proof, merge lanes, and implementation daemons.

This root package is the **public API and package map** for the ASREF module
layout (`docs/architecture/AGENT_SUPERVISOR_MODULE_REFACTOR_PLAN.md`). Domain
code lives under named subpackages; the root `__init__.py` re-exports only
intentional symbols (control surface, stable generation-2 contracts, domain
layout constants, and selected objective/planning helpers). New code must
import from domain packages—not from retired flat module paths.

**Cutover goal:** `ASREF-G090` (task `ASREF-012`, packet
`goal_packet/cutover/ipfs_accelerate_py/090ea2138c6f`). Parent evidence for
this goal is the set of package goals **ASREF-G020**, **ASREF-G030**,
**ASREF-G040**, **ASREF-G050**, **ASREF-G060**, **ASREF-G070**, and
**ASREF-G080**. This README is the durable map that binds those goals to
on-disk packages, public imports, root hygiene, and the no-old-import gate.

## Public API (package root)

Import the reviewed control surface, stable manifests, and layout constants
from the package root:

```python
from ipfs_accelerate_py.agent_supervisor import (
    Operation,
    OperationRequest,
    OperationResult,
    SupervisorControlService,
    AGENT_SUPERVISOR_V2_STABLE_EXPORTS,
    AGENT_SUPERVISOR_DOMAIN_PACKAGES,
    AGENT_SUPERVISOR_LANDED_MODULE_OWNERS,
    AGENT_SUPERVISOR_PACKAGE_GOAL_EVIDENCE,
    AGENT_SUPERVISOR_CUTOVER_GOAL_ID,
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
5. Layout constants (`AGENT_SUPERVISOR_DOMAIN_PACKAGES`,
   `AGENT_SUPERVISOR_LANDED_MODULE_OWNERS`,
   `AGENT_SUPERVISOR_PACKAGE_GOAL_EVIDENCE`) are intentional public symbols
   for discovery, cutover gates, and objective evidence scans.

## Package map

Every domain package under `ipfs_accelerate_py/agent_supervisor/`:

| Package | Bundle | Objective goal | Purpose | README / status |
|---|---|---|---|---|
| `core/` | `asref/core` | **ASREF-G020** | Shared identity, conflict graph, external completion, wrappers | [core/README.md](./core/README.md) — landed |
| `control/` | `asref/control` | **ASREF-G030** | Control plane, CLI, contracts, lifecycle, permits, authz | [control/README.md](./control/README.md) — landed |
| `task_sources/` | `asref/task-sources` | **ASREF-G040** | Markdown/DuckDB task sources, taskboard, queues, indexes | [task_sources/README.md](./task_sources/README.md) — landed |
| `context/` | `asref/context` | **ASREF-G050** | Context compiler/contracts, decision runtime | flat modules until `asref/context` move |
| `prompt/` | `asref/prompt` | **ASREF-G050** | Prompt workflow, scanner, plan admission | flat modules until `asref/prompt` move |
| `analysis/` | `asref/analysis` | **ASREF-G060** | Analysis pipeline, AST, cache, retrieval, consensus | flat modules until `asref/analysis` move |
| `proof/` | `asref/proof` | **ASREF-G060** | Formal verification, provers, attestation | flat modules until `asref/proof` move |
| `objectives/` | `asref/objectives` | **ASREF-G070** | Objective heap, daemon, tracker, backlog, goal completion | [objectives/README.md](./objectives/README.md) — landed |
| `planning/` | `asref/planning` | **ASREF-G070** | Adaptive/formal planning, plan metrics/rollout, failure memory | [planning/README.md](./planning/README.md) — landed |
| `validation/` | `asref/validation` | **ASREF-G070** | Proposal validation, scope adjudication, validation runtime | [validation/README.md](./validation/README.md) — landed |
| `merge/` | `asref/merge` | **ASREF-G070** | Merge queue/train/resolver, checkout lock, git GC, leases | [merge/README.md](./merge/README.md) — landed |
| `rescue/` | `asref/rescue` | **ASREF-G070** | Rescue orchestrator, failure policy, recovery/watchdog | [rescue/README.md](./rescue/README.md) — landed |
| `runtime/` | `asref/runtime` | **ASREF-G070** | Multi-supervisor runner, artifacts, schedulers, CAS | [runtime/README.md](./runtime/README.md) — landed |
| `self_improvement/` | `asref/self-improvement` | **ASREF-G070** | Self-improvement completion, v2 contracts/metrics/rollout | [self_improvement/README.md](./self_improvement/README.md) — landed |
| `todo_daemon/` | `asref/todo-daemon` | **ASREF-G080** | Executable daemons (implementation, loop, git, legal, LLM) | package modules present; internal subpackage split still open |
| `integrations/` | `asref/integrations` | **ASREF-G080** | External tool bridges (LLM merge fallback, goose, datasets) | flat modules until `asref/integrations` move |

Present on disk with modules + README (landed scaffolds): `core`, `control`,
`task_sources`, `objectives`, `planning`, `validation`, `merge`, `rescue`,
`runtime`, `self_improvement`, `todo_daemon`.

Still largely flat at package root until their move batches finish: `context`,
`analysis`, `proof`, `prompt`, `integrations` module sets (see
`docs/architecture/asref/move_map.json`). Owning goals remain **ASREF-G050**,
**ASREF-G060**, and **ASREF-G080**; cutover documents them here so
`ASREF-G090` retains an exact-text evidence nomination for each package goal.

## Package-goal evidence (ASREF-G090 parent obligations)

`ASREF-G090` lists package goals as its evidence terms. The cutover packet
covers them in one cohesive pass by mapping each term to a package contract
and import style. Runtime mirror:
`AGENT_SUPERVISOR_PACKAGE_GOAL_EVIDENCE` in `__init__.py`.

| Evidence term | Bundle | Landed package path(s) | Cutover witness |
|---|---|---|---|
| **ASREF-G020** | `asref/core` | `core/` | README + `AGENT_SUPERVISOR_LANDED_MODULE_OWNERS` core stems; no flat `conflict_graph` / `external_completion` dual copies |
| **ASREF-G030** | `asref/control` | `control/` | Control contracts/plane re-exported from package path; conformance suite |
| **ASREF-G040** | `asref/task-sources` | `task_sources/` | Markdown/DuckDB/taskboard modules under package; protocol may stay conceptual in core |
| **ASREF-G050** | `asref/context` (+ prompt) | `context/`, `prompt/` (planned) | Named in domain map; flat `context_*` / `decision_*` / `prompt_*` remain until move tasks land |
| **ASREF-G060** | `asref/analysis` (+ proof) | `analysis/`, `proof/` (planned) | Named in domain map; flat analysis/proof modules remain until move tasks land |
| **ASREF-G070** | `asref/objectives` (+ siblings) | `objectives/`, `planning/`, `validation/`, `merge/`, `rescue/`, `runtime/`, `self_improvement/` | Package READMEs + landed module owners; objective graph / proposal validation import package paths |
| **ASREF-G080** | `asref/todo-daemon` (+ integrations) | `todo_daemon/`, `integrations/` (planned) | Daemon imports already package-native; integrations flat until move |

Child package lanes still own the physical moves for partial rows
(**ASREF-G050**, remaining **ASREF-G060** / **ASREF-G080** splits). Cutover
owns the **final no-old-import gate**, public API map, and root hygiene for
merge readiness.

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

# Daemons (already package-native) — ASREF-G080
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    TodoImplementationDaemon,
)
```

## Landed modules (no flat import for package-root resolution)

These stems already exist under a domain package. Package-root eager/lazy
exports import them from the package path. Prefer the package path in all new
code, tests, scripts, and entry points:

| Stem | Owner package | Goal |
|---|---|---|
| `authorization_logic`, `control_cli`, `control_contracts`, `control_plane`, `execution_permit`, `lifecycle_orchestrator` | `control` | ASREF-G030 |
| `conflict_graph`, `external_completion`, `program_behavior`, `submodule_degradation`, `wrapper_utils` | `core` | ASREF-G020 |
| `dataset_store`, `duckdb_state`, `duckdb_task_source`, `markdown_task_source`, `persistent_task_queue`, `task_identity`, `task_source`, `taskboard_store`, `todo_vector_index` | `task_sources` | ASREF-G040 |
| `backlog_refinery`, `objective_daemon`, `objective_graph` | `objectives` | ASREF-G070 |
| `proposal_validation` | `validation` | ASREF-G070 |
| `plan_failure_memory`, `formal_planning_metrics`, `formal_planning_rollout` | `planning` | ASREF-G070 |
| `checkout_lock`, `git_gc`, `merge_checkpoint`, `merge_conflict_repair`, `merge_resolver` | `merge` | ASREF-G070 |
| `codex_failure_policy`, `rescue_orchestrator` | `rescue` | ASREF-G070 |
| `multi_supervisor_runner` | `runtime` | ASREF-G070 |
| `self_improvement_completion` | `self_improvement` | ASREF-G070 |

Runtime constant: `AGENT_SUPERVISOR_LANDED_MODULE_OWNERS` mirrors this table.
Package-root resolution aliases retired flat submodule names to the owner
package without leaving long-lived stub **files** at the flat path.

### Import cutover policy (no-old-import gate)

1. After a module lands under package `P`, callers must use
   `ipfs_accelerate_py.agent_supervisor.P.<module>`.
2. **Do not** leave long-lived re-export stubs at
   `ipfs_accelerate_py.agent_supervisor.<module>`.
3. Temporary dual copies at the flat path may remain only until callers and
   console entry points are rewritten; they are not part of the public API.
4. Assert absence after full caller cutover (cutover owns this gate):

```bash
# Landed control / core / objectives / validation stems
rg -n 'agent_supervisor\.(control_contracts|control_plane|conflict_graph|objective_graph|proposal_validation)\b' \
  --glob '!**/__pycache__/**' \
  --glob '!docs/architecture/asref/**' \
  --glob '!**/control/**' --glob '!**/core/**' \
  --glob '!**/objectives/**' --glob '!**/validation/**'

# Broader landed-owner sweep (should not hit production callers outside packages)
rg -n 'agent_supervisor\.(markdown_task_source|duckdb_task_source|task_identity|merge_resolver|rescue_orchestrator|multi_supervisor_runner|objective_daemon|backlog_refinery)\b' \
  --glob '!**/__pycache__/**' \
  --glob '!docs/architecture/asref/**' \
  --glob '!**/task_sources/**' --glob '!**/merge/**' \
  --glob '!**/rescue/**' --glob '!**/runtime/**' \
  --glob '!**/objectives/**'
```

## Console entry points (post-move targets)

| Console script | Canonical target | Goal |
|---|---|---|
| `ipfs-accelerate-agent-objective-daemon` | `objectives.objective_daemon:main` | ASREF-G070 |
| `ipfs-accelerate-agent-backlog-refinery` | `objectives.backlog_refinery:main` | ASREF-G070 |
| `ipfs-accelerate-agent-bundle-supervisor` | `objectives.bundle_supervisor:main` (when landed) | ASREF-G070 |
| `ipfs-accelerate-agent-artifact-query` | `runtime.artifact_store:main` (when landed) | ASREF-G070 |
| `ipfs-accelerate-agent-implementation-daemon` | `todo_daemon.implementation_daemon:main` | ASREF-G080 |
| `ipfs-accelerate-agent-implementation-supervisor` | `todo_daemon.implementation_supervisor:main` | ASREF-G080 |
| `ipfs-accelerate-agent-merge-resolver` | `merge.merge_resolver:main` | ASREF-G070 |
| `ipfs-accelerate-agent-llm-merge-resolver-fallback` | `integrations` / flat until integrations lands | ASREF-G080 |

Unified CLI: `control.control_cli:register_agent_cli` /
`control.control_cli:run_agent_cli` (**ASREF-G030**).

During dual-copy windows, `pyproject.toml` / `setup.py` may still point at a
flat module until the owning move task rewrites entry points in the same
change as the final flat-file removal.

## todo_daemon layout (ASREF-G080)

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

Integrations (LLM merge fallback, meta_spark/goose, datasets providers) move
to `integrations/` under the same **ASREF-G080** goal; until then they may
remain flat at package root and must not be treated as long-lived public API.

## Root hygiene (ASREF-G090 / ASREF-006)

Monorepo root hygiene is part of cutover readiness. Durable docs:

| Concern | Location |
|---|---|
| Nested product trees and submodules | [`docs/NESTED_PACKAGES.md`](../../../docs/NESTED_PACKAGES.md) |
| Process-junk ignore rules | [`.gitignore`](../../../.gitignore) (ASREF-006 / ASREF-G090 section) |
| Nested products must not absorb `agent_supervisor` packages | Ownership policy in `NESTED_PACKAGES.md` |

Ephemeral process files at monorepo root (**must not be treated as product
artifacts**):

- `dashboard.out`, `dashboard.pid`, `err.txt` — ignored; untrack with
  `git rm --cached` when an operator task has broader path scope
- Misplaced root `test_*.py` (e.g. `test_dashboard_sdk.py`,
  `test_mcp_jsonrpc_conformance.py`) — prefer `test/`; relocation may land in
  a hygiene task with broader edit scope than cutover outputs alone

## Related program docs

- Plan: `docs/architecture/AGENT_SUPERVISOR_MODULE_REFACTOR_PLAN.md`
- Objectives heap: `docs/architecture/agent_supervisor_module_refactor.objectives.md`
- Todo board: `docs/architecture/agent_supervisor_module_refactor.todo.md`
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

These cover **ASREF-G080** daemon ports, **ASREF-G030** control conformance,
**ASREF-G070** objective graph, and **ASREF-G070** proposal validation —
the shared cutover packet validation surface for `ASREF-G090`.

## Status (ASREF-012 / ASREF-G090 cutover)

| Acceptance item | Status |
|---|---|
| Root `README.md` maps all domain packages | present (this file) |
| Root `__init__.py` intentional public symbols + package-path resolution | present |
| Evidence terms **ASREF-G020** … **ASREF-G080** named for parent goal | present (package map + evidence table + constants) |
| Domain package READMEs for landed packages | present for core/control/task_sources/objectives/planning/validation/merge/rescue/runtime/self_improvement |
| Nested product trees documented | `docs/NESTED_PACKAGES.md` |
| Process junk ignored | `.gitignore` (`dashboard.out`, `dashboard.pid`, `err.txt`, …) |
| No-old-import gate owned by cutover | rg recipes above; landed flat dual-copy **files** removed for owner map stems |
| Remaining flat modules (context/analysis/proof/prompt/integrations) | still under child goals **ASREF-G050**, **ASREF-G060**, **ASREF-G080** |
| Entry-point rewrites for remaining flat modules | complete when each owning move task lands |
| Historical tracked process files / root tests | ignore rules present; untrack/move needs broader path scope than cutover-only outputs |
