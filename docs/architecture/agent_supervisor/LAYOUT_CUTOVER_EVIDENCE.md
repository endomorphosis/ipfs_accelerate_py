# Domain layout cutover evidence (historical)

This document preserves the **domain-layout program** evidence map used during
the ASREF package cutover (`ASREF-G090` / packet tasks `ASREF-012`–`ASREF-014`).

**For day-to-day development, do not start here.** Use:

- [Package README (developer entry)](../../../ipfs_accelerate_py/agent_supervisor/README.md)
- [Developer guide](DEVELOPER_GUIDE.md)
- [Package map](PACKAGE_MAP.md)
- [Architecture](../AGENT_SUPERVISOR_ARCHITECTURE.md)

Board goal IDs (`ASREF-G0xx`) below are **evidence terms and historical
receipts**. Public Python APIs use semantic names
(`AGENT_SUPERVISOR_CORE_PACKAGES`, not `AGENT_SUPERVISOR_G020_PACKAGES`).

---

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
and import style. Runtime mirrors in `__init__.py`:

- `AGENT_SUPERVISOR_DOMAIN_LAYOUT_GOAL_IDS` — full parent evidence set
- `AGENT_SUPERVISOR_LAYOUT_GOAL_TO_PACKAGES` — goal id → package names
- `AGENT_SUPERVISOR_FOUNDATION_LAYOUT_GOAL_IDS` — **ASREF-013** subset
- `AGENT_SUPERVISOR_OPERATIONS_LAYOUT_GOAL_IDS` — **ASREF-014** subset
- `AGENT_SUPERVISOR_CORE_STEMS` /
  `AGENT_SUPERVISOR_CONTROL_STEMS` /
  `AGENT_SUPERVISOR_TASK_SOURCES_STEMS` /
  `AGENT_SUPERVISOR_CONTEXT_PROMPT_PLANNED_MODULES` — G020–G050 inventories

| Evidence term | Bundle | Landed package path(s) | Cutover witness |
|---|---|---|---|
| **ASREF-G020** | `asref/core` | `core/` | Package + README landed; `AGENT_SUPERVISOR_CORE_STEMS`; no flat dual-copy files for those stems |
| **ASREF-G030** | `asref/control` | `control/` | Package + README landed; control contracts/plane from package path; conformance suite |
| **ASREF-G040** | `asref/task-sources` | `task_sources/` | Package + README landed; `AGENT_SUPERVISOR_TASK_SOURCES_STEMS`; markdown/DuckDB under package |
| **ASREF-G050** | `asref/context` (+ prompt) | `context/`, `prompt/` (planned) | Named in domain map + `AGENT_SUPERVISOR_CONTEXT_PROMPT_PLANNED_MODULES`; flat until move tasks land |
| **ASREF-G060** | `asref/analysis` (+ proof) | `analysis/`, `proof/` (planned) | Named in domain map + `AGENT_SUPERVISOR_PLANNED_MODULE_TO_PACKAGE` analysis/proof stems; flat modules remain until move tasks land |
| **ASREF-G070** | `asref/objectives` (+ siblings) | `objectives/`, `planning/`, `validation/`, `merge/`, `rescue/`, `runtime/`, `self_improvement/` | Package READMEs + landed module owners; objective graph / proposal validation import package paths |
| **ASREF-G080** | `asref/todo-daemon` (+ integrations) | `todo_daemon/`, `integrations/` (planned) | Daemon imports already package-native; integrations flat until move (planned owners listed) |

Child package lanes still own the physical moves for partial rows
(**ASREF-G050**, remaining **ASREF-G060** / **ASREF-G070** flats /
**ASREF-G080** integrations split). Cutover owns the **final no-old-import
gate**, public API map, and root hygiene for merge readiness.

## ASREF-G020 evidence (core)

**Goal:** Create `agent_supervisor/core` with shared contracts, identity,
events, and task-source protocol modules, README, and updated imports.

**Package constants:** `AGENT_SUPERVISOR_CORE_PACKAGES` → `core`.

**Landed stems** (`AGENT_SUPERVISOR_CORE_STEMS`; no flat dual-copy
files):

| Stem | Path | Role |
|---|---|---|
| `conflict_graph` | `core/conflict_graph.py` | Conflict / dependency graph |
| `external_completion` | `core/external_completion.py` | External completion receipts |
| `program_behavior` | `core/program_behavior.py` | Program behavior helpers |
| `submodule_degradation` | `core/submodule_degradation.py` | Submodule degradation policy |
| `wrapper_utils` | `core/wrapper_utils.py` | Shared wrappers |

**Preferred imports (landed — use these, not retired flat paths):**

```python
from ipfs_accelerate_py.agent_supervisor.core.conflict_graph import ConflictGraph
from ipfs_accelerate_py.agent_supervisor.core.external_completion import (
    ExternalCompletion,  # illustrative — use real symbols
)
from ipfs_accelerate_py.agent_supervisor.core.wrapper_utils import ...
```

**DAG:** `core` is the bottom package. It must **not** import `todo_daemon`,
`runtime`, `merge`, `rescue`, or `self_improvement`. Package README:
[core/README.md](./core/README.md).

**No-old-import check for landed G020 stems:**

```bash
rg -n 'agent_supervisor\.(conflict_graph|external_completion|program_behavior|submodule_degradation|wrapper_utils)\b' \
  --glob '!**/__pycache__/**' \
  --glob '!docs/architecture/**' \
  --glob '!**/core/**'
```

**Cutover witness for ASREF-G020:** package on disk with README, stem
inventory, preferred imports, no flat dual-copy files for those stems.

## ASREF-G030 evidence (control)

**Goal:** Create `agent_supervisor/control` for control plane, CLI, contracts,
lifecycle orchestration, and execution permits with README and hard import
updates.

**Package constants:** `AGENT_SUPERVISOR_CONTROL_PACKAGES` → `control`.

**Landed stems** (`AGENT_SUPERVISOR_CONTROL_STEMS`):

| Stem | Path | Role |
|---|---|---|
| `authorization_logic` | `control/authorization_logic.py` | Authz / delegation policy |
| `control_cli` | `control/control_cli.py` | Unified agent CLI adapter |
| `control_contracts` | `control/control_contracts.py` | Operation / request / result types |
| `control_plane` | `control/control_plane.py` | `SupervisorControlService` |
| `execution_permit` | `control/execution_permit.py` | Short-lived mutation permits |
| `lifecycle_orchestrator` | `control/lifecycle_orchestrator.py` | Process lifecycle mutations |

**Preferred imports (landed):**

```python
from ipfs_accelerate_py.agent_supervisor.control.control_contracts import (
    Operation,
    OperationRequest,
    OperationResult,
)
from ipfs_accelerate_py.agent_supervisor.control.control_plane import (
    SupervisorControlService,
)
from ipfs_accelerate_py.agent_supervisor.control.control_cli import (
    register_agent_cli,
    run_agent_cli,
)
```

**DAG:** `control` may depend on `core` only among domain packages. Package
README: [control/README.md](./control/README.md).

**Validation (owning goal + cutover suite):**

```bash
python -m pytest test/api/test_agent_supervisor_control_conformance_v2.py \
  test/api/test_agent_supervisor_lifecycle_orchestrator.py -q
```

**No-old-import check for landed G030 stems:**

```bash
rg -n 'agent_supervisor\.(control_contracts|control_plane|control_cli|authorization_logic|execution_permit|lifecycle_orchestrator)\b' \
  --glob '!**/__pycache__/**' \
  --glob '!docs/architecture/**' \
  --glob '!**/control/**'
```

**Cutover witness for ASREF-G030:** package + README, stem inventory, root
eager re-exports of control surface, conformance suite in cutover gate.

## ASREF-G040 evidence (task_sources)

**Goal:** Create `agent_supervisor/task_sources` for Markdown and DuckDB
projections, taskboard store, persistent queues, and todo vector index with
README and caller updates.

**Package constants:** `AGENT_SUPERVISOR_TASK_SOURCES_PACKAGES` → `task_sources`.

**Landed stems** (`AGENT_SUPERVISOR_TASK_SOURCES_STEMS`):

| Stem | Path | Role |
|---|---|---|
| `markdown_task_source` | `task_sources/markdown_task_source.py` | Markdown board projection |
| `duckdb_task_source` | `task_sources/duckdb_task_source.py` | DuckDB board projection |
| `task_source` | `task_sources/task_source.py` | Backend-neutral protocol (design home: core) |
| `taskboard_store` | `task_sources/taskboard_store.py` | Taskboard store |
| `persistent_task_queue` | `task_sources/persistent_task_queue.py` | Persistent queue |
| `task_identity` | `task_sources/task_identity.py` | Task identity helpers |
| `duckdb_state` | `task_sources/duckdb_state.py` | DuckDB state locking |
| `dataset_store` | `task_sources/dataset_store.py` | Dataset store |
| `todo_vector_index` | `task_sources/todo_vector_index.py` | Todo vector index |

**Protocol refinement:** keep the backend-neutral protocol design in `core`;
implementations live in `task_sources`. Until `task_source` relocates under
`core/`, the protocol module co-resides in `task_sources/` so dual-projection
code stays coherent.

**Preferred imports (landed):**

```python
from ipfs_accelerate_py.agent_supervisor.task_sources.markdown_task_source import (
    MarkdownTaskSource,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_task_source import (
    DuckDBTaskSource,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.task_source import (
    CanonicalTaskSource,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.taskboard_store import (
    TaskboardStore,
)
```

**DAG:** sits above `core`. Package README:
[task_sources/README.md](./task_sources/README.md).

**Validation (owning goal):**

```bash
python -m pytest test/api/test_agent_supervisor_markdown_task_source.py \
  test/api/test_agent_supervisor_duckdb_task_source.py -q
```

**No-old-import check for landed G040 stems:**

```bash
rg -n 'agent_supervisor\.(markdown_task_source|duckdb_task_source|task_identity|taskboard_store|persistent_task_queue|todo_vector_index)\b' \
  --glob '!**/__pycache__/**' \
  --glob '!docs/architecture/**' \
  --glob '!**/task_sources/**'
```

**Cutover witness for ASREF-G040:** package + README, stem inventory,
preferred imports, no flat dual-copy files for owner-map stems.

## ASREF-G050 evidence (context + prompt)

**Goal:** Create `agent_supervisor/context` and `agent_supervisor/prompt` for
context compilation, decision runtime, and prompt workflow/scanner/admission
surfaces with READMEs and import updates.

**Package constants:** `AGENT_SUPERVISOR_CONTEXT_PROMPT_PACKAGES` → `context`, `prompt`.

**Planned modules** (still flat at package root until move tasks land;
`AGENT_SUPERVISOR_CONTEXT_PROMPT_PLANNED_MODULES` /
`AGENT_SUPERVISOR_PLANNED_MODULE_TO_PACKAGE`):

| Stem | Planned owner | Role |
|---|---|---|
| `context_compiler`, `context_contracts` | `context` | Context compilation / contracts |
| `decision_context`, `decision_contracts` | `context` | Decision context surfaces |
| `decision_runtime`, `decision_runtime_benchmark`, `decision_runtime_rollout` | `context` | Decision runtime + rollout |
| `prompt_workflow` | `prompt` | Prompt workflow service |
| `prompt_directory_scanner` | `prompt` | Prompt directory scanner |
| `prompt_plan_admission` | `prompt` | Plan admission |
| `prompt_goal_planner` | `prompt` | Goal planner surface |

**Preferred future imports (after package land):**

```python
from ipfs_accelerate_py.agent_supervisor.context.context_compiler import (
    ContextCompiler,  # illustrative — use real symbols after move
)
from ipfs_accelerate_py.agent_supervisor.context.decision_runtime import (
    DecisionRuntime,  # illustrative
)
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import (
    PromptWorkflowService,  # illustrative
)
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_directory_scanner import (
    PromptDirectoryScanner,  # illustrative
)
```

**Until moves land**, callers may still import flat modules at package root
(e.g. `from ipfs_accelerate_py.agent_supervisor.context.context_compiler import ...`).
Those flat paths are **not** the long-term public API. Do not add package
re-export stubs early.

**DAG:** `prompt` may depend on `context`; not vice versa. Physical package
creation remains `asref/context` / `asref/prompt` move tasks.

**Validation (owning goal, when packages land):**

```bash
python -m pytest test/api/test_agent_supervisor_context_compiler.py \
  test/api/test_agent_supervisor_prompt_workflow_service.py -q
```

**Cutover witness for ASREF-G050:** package map row + planned owners + this
section + exact goal id in `AGENT_SUPERVISOR_DOMAIN_LAYOUT_GOAL_IDS` and
`AGENT_SUPERVISOR_FOUNDATION_LAYOUT_GOAL_IDS`. Child lanes own physical moves.

## ASREF-G060 evidence (analysis + proof)

**Goal:** Create `analysis/` and `proof/` packages grouping analysis
pipeline/cache/AST/retrieval and formal verification/prover modules.

**Package constants:** `AGENT_SUPERVISOR_ANALYSIS_PROOF_PACKAGES` → `analysis`, `proof`.

**Objective evidence modules (still flat until move tasks land):**

| Stem | Planned owner | Role |
|---|---|---|
| `analysis_pipeline` | `analysis` | End-to-end analysis pipeline |
| `analysis_cache`, `analysis_ast_index`, `analysis_retrieval` | `analysis` | Cache / AST / retrieval |
| `analysis_contracts`, `analysis_consensus`, `analysis_transport` | `analysis` | Contracts and transport |
| `formal_verification_provider` | `proof` | Proof provider surface |
| `multi_prover_router`, `multi_prover_resources` | `proof` | Multi-prover portfolio |
| `prover_matrix_registry`, `prover_conformance`, `prover_evidence_store` | `proof` | Prover matrix and evidence |
| `leanstral_proof_provider`, `kernel_verification` | `proof` | Leanstral / kernel lanes |

Full planned stem → owner map: every `analysis` / `proof` entry in
`AGENT_SUPERVISOR_PLANNED_MODULE_TO_PACKAGE` (from
`docs/architecture/asref/move_map.json`).

**Preferred future imports (after package land):**

```python
from ipfs_accelerate_py.agent_supervisor.analysis.analysis_pipeline import (
    AnalysisPipeline,  # name illustrative — use real symbols after move
)
from ipfs_accelerate_py.agent_supervisor.proof.multi_prover_router import (
    MultiProverRouter,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_provider import (
    FormalVerificationProvider,  # illustrative
)
```

**Until moves land**, callers may still import flat modules at package root
(e.g. `from ipfs_accelerate_py.agent_supervisor.analysis.analysis_pipeline import ...`).
Those flat paths are **not** the long-term public API. Forbidden after land:
`core` / `analysis` / `proof` must not import `todo_daemon` (DAG).

**Cutover witness for ASREF-G060:** package map row + planned owners + this
section + exact goal id in `AGENT_SUPERVISOR_DOMAIN_LAYOUT_GOAL_IDS` and
`AGENT_SUPERVISOR_OPERATIONS_LAYOUT_GOAL_IDS`. Physical package creation remains
`asref/analysis` / `asref/proof` move tasks.

## ASREF-G070 evidence (objectives, planning, validation, merge, rescue, runtime, self_improvement)

**Goal:** Domain packages for objectives/ops/runtime with READMEs and hard
import/entry-point updates.

**Package constants:** `AGENT_SUPERVISOR_OPERATIONS_PACKAGES`.

**Landed package scaffolds (on disk with README):**

| Package | Landed stems (no flat dual-copy files) | Status |
|---|---|---|
| `objectives/` | `objective_graph`, `objective_daemon`, `backlog_refinery` | [README](./objectives/README.md) |
| `planning/` | `plan_failure_memory`, `formal_planning_metrics`, `formal_planning_rollout` | [README](./planning/README.md) |
| `validation/` | `proposal_validation` | [README](./validation/README.md) |
| `merge/` | `merge_resolver`, `merge_checkpoint`, `merge_conflict_repair`, `checkout_lock`, `git_gc` | [README](./merge/README.md) |
| `rescue/` | `rescue_orchestrator`, `codex_failure_policy` | [README](./rescue/README.md) |
| `runtime/` | `multi_supervisor_runner` | [README](./runtime/README.md) |
| `self_improvement/` | `self_improvement_completion` | [README](./self_improvement/README.md) |

Runtime list: `AGENT_SUPERVISOR_OPERATIONS_LANDED_STEMS`.

**Preferred imports (landed — use these, not retired flat paths):**

```python
from ipfs_accelerate_py.agent_supervisor.objectives.objective_graph import (
    parse_goal_heap,
)
from ipfs_accelerate_py.agent_supervisor.objectives.objective_daemon import (
    ObjectiveDaemon,  # illustrative of package path
)
from ipfs_accelerate_py.agent_supervisor.validation.proposal_validation import (
    validate_implementation_proposal,
)
from ipfs_accelerate_py.agent_supervisor.merge.merge_resolver import MergeResolver
from ipfs_accelerate_py.agent_supervisor.rescue.rescue_orchestrator import (
    RescueOrchestrator,
)
from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
    MultiSupervisorRunner,
)
```

**Remaining flat under G070** (still package-root files; planned owners listed
in `AGENT_SUPERVISOR_PLANNED_MODULE_TO_PACKAGE`): e.g. `goal_completion`,
`bundle_supervisor`, `adaptive_planner`, `formal_plan_*`, `merge_queue`,
`merge_train`, `artifact_store`, `self_improvement_v2`, …
Child move tasks land those stems; cutover documents the target owners.

**No-old-import check for landed G070 stems:**

```bash
rg -n 'agent_supervisor\.(objective_graph|objective_daemon|proposal_validation|merge_resolver|rescue_orchestrator|multi_supervisor_runner|backlog_refinery)\b' \
  --glob '!**/__pycache__/**' \
  --glob '!docs/architecture/**' \
  --glob '!**/objectives/**' --glob '!**/validation/**' \
  --glob '!**/merge/**' --glob '!**/rescue/**' --glob '!**/runtime/**'
```

**Cutover witness for ASREF-G070:** seven packages on disk with READMEs, landed
owner map, preferred imports above, validation suite objective graph +
proposal validation tests.

## ASREF-G080 evidence (todo_daemon + integrations)

**Goal:** Keep/re-package `todo_daemon` with clear internal layout; move
integration runners into `integrations/`; update console and ops scripts
without long-lived shims.

**Package constants:** `AGENT_SUPERVISOR_INTEGRATIONS_DAEMON_PACKAGES` → `todo_daemon`,
`integrations`.

### todo_daemon (package-native today)

Canonical daemon modules (already under the package):

```python
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    TodoImplementationDaemon,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
    TodoImplementationSupervisor,
)
```

`AGENT_SUPERVISOR_TODO_DAEMON_STEMS` lists the primary entry modules.
Planned internal subpackages (from the program plan; land via
`asref/todo-daemon`):

- `implementation/` — implementation daemon + supervisor (+ runners)
- `loop/` — supervisor loop, runner, engine pieces
- `git/` — worktrees, git utils, auto-commit

Root-level `implementation_daemon_runner` /
`implementation_supervisor_runner` remain flat until the implementation
subpackage move; both are planned owners under `todo_daemon` in
`AGENT_SUPERVISOR_PLANNED_MODULE_TO_PACKAGE`.

### integrations (planned package)

Flat modules to move under `integrations/`:

| Stem | Role |
|---|---|
| `llm_merge_resolver_fallback` | LLM merge-resolver fallback entry |
| `meta_spark_goose_runner` | Goose / meta-spark bridge |
| `ipfs_datasets_analysis_provider` | Datasets analysis provider |
| `ipfs_datasets_logic_provider` | Datasets logic provider |

```python
# After integrations/ lands:
from ipfs_accelerate_py.agent_supervisor.integrations.llm_merge_resolver_fallback import (
    main,  # illustrative
)
```

**Cutover witness for ASREF-G080:** package-native daemon imports, planned
integrations map, console entry targets below, daemon port validation test.

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

Runtime constant: `AGENT_SUPERVISOR_LANDED_MODULE_TO_PACKAGE` mirrors this table.
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

## Root hygiene (ASREF-G090 / ASREF-006 / ASREF-013 / ASREF-014)

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

**ASREF-G020 … ASREF-G080 packages never live at monorepo root** — they stay
under `ipfs_accelerate_py/agent_supervisor/` only (see
`docs/NESTED_PACKAGES.md`).

## Related program docs

- Plan: `docs/architecture/AGENT_SUPERVISOR_MODULE_REFACTOR_PLAN.md`
- Objectives heap: `docs/architecture/agent_supervisor_module_refactor.objectives.md`
- Todo board: `docs/architecture/agent_supervisor_module_refactor.todo.md`
- Move map (ASREF-G010): `docs/architecture/asref/move_map.json`
- Import inventory (ASREF-G010): `docs/architecture/asref/import_inventory.md`
- Nested monorepo products: `docs/NESTED_PACKAGES.md`
- Layout evidence module: `asref_layout_evidence.py` (ASREF-G010 / ASREF-G090 / ASREF-G100)
- Multi-lane launch (ASREF-G100): `scripts/ops/agent_supervisor/asref_multi_lane_launch.py`
- Supervisor launcher (ASREF-G100): `scripts/ops/asref_module_refactor_supervisor.py`

## Validation (cutover gate)

```bash
python -m pytest \
  test/api/test_agent_supervisor_todo_daemon_port.py \
  test/api/test_agent_supervisor_control_conformance_v2.py \
  test/api/test_agent_supervisor_objective_graph.py \
  test/api/test_agent_supervisor_proposal_validation.py \
  test/api/test_agent_supervisor_asref_layout_evidence.py \
  -q
```

These cover **ASREF-G080** daemon ports, **ASREF-G030** control conformance,
**ASREF-G070** objective graph, and **ASREF-G070** proposal validation —
the shared cutover packet validation surface for `ASREF-G090`.
**ASREF-G030** is exercised by the conformance suite; **ASREF-G020** /
**ASREF-G040** are covered by package-path imports used across the same suite;
**ASREF-G050** is named and inventoried (flat until move lanes land).

Layout evidence CLI:

```bash
python -m ipfs_accelerate_py.agent_supervisor.asref_layout_evidence
python scripts/ops/agent_supervisor/asref_multi_lane_launch.py recipe
```

Optional focused checks for child package goals (not required by cutover
packet validation, but useful for G020–G080 lanes):

```bash
# ASREF-G020 / event runtime (collect-only smoke)
python -m pytest test/api/test_agent_supervisor_event_driven_runtime.py -q --collect-only

# ASREF-G040 task sources
python -m pytest test/api/test_agent_supervisor_markdown_task_source.py \
  test/api/test_agent_supervisor_duckdb_task_source.py -q

# ASREF-G050 (when context/prompt tests are exercised)
python -m pytest test/api/test_agent_supervisor_context_compiler.py \
  test/api/test_agent_supervisor_prompt_workflow_service.py -q

# ASREF-G060 (when analysis/proof tests are exercised)
python -m pytest test/api/test_agent_supervisor_analysis_pipeline.py \
  test/api/test_agent_supervisor_multi_prover_router.py -q

# ASREF-G070 recovery path
python -m pytest test/api/test_agent_supervisor_programmatic_recovery.py -q

# ASREF-G080 protected paths
python -m pytest test/api/test_agent_supervisor_implementation_protected_paths.py -q
```

## Status (ASREF-013 / ASREF-G090 cutover — G020–G050 evidence cluster)

| Acceptance item | Status |
|---|---|
| Root `README.md` maps all domain packages | present (this file) |
| Root `__init__.py` intentional public symbols + package-path resolution | present |
| Evidence terms **ASREF-G020**, **ASREF-G030**, **ASREF-G040**, **ASREF-G050** | present (cluster constants, owners map, package map, stem inventories) |
| Evidence terms **ASREF-G060** … **ASREF-G080** named for parent goal | present (ASREF-014 cluster + package map) |
| **ASREF-G020** core package + `AGENT_SUPERVISOR_CORE_STEMS` | present (package on disk; no flat dual-copy files) |
| **ASREF-G030** control package + conformance suite | present (package path re-exports; cutover gate) |
| **ASREF-G040** task_sources package + stem inventory | present (package on disk; dual-projection modules under package) |
| **ASREF-G050** context/prompt planned inventory | present (`AGENT_SUPERVISOR_CONTEXT_PROMPT_PLANNED_MODULES`; flat until move) |
| **ASREF-G060** analysis/proof map + planned owners | present (`AGENT_SUPERVISOR_PLANNED_MODULE_TO_PACKAGE`) |
| **ASREF-G070** seven packages + landed stems + preferred imports | present (packages on disk; landed owners wired) |
| **ASREF-G080** todo_daemon package-native + integrations planned | present (daemon imports; integrations planned owners) |
| Domain package READMEs for landed packages | present for core/control/task_sources/objectives/planning/validation/merge/rescue/runtime/self_improvement |
| Frozen move map + import inventory (**ASREF-G010**) | present under `docs/architecture/asref/` |
| Layout evidence module (**ASREF-G010** / **ASREF-G090** / **ASREF-G100**) | `asref_layout_evidence.py` |
| Multi-lane Grok launch recipe + protected-path wiring (**ASREF-G100**) | present under `scripts/ops/agent_supervisor/` |
| Nested product trees documented | `docs/NESTED_PACKAGES.md` |
| Process junk ignored | `.gitignore` (`dashboard.out`, `dashboard.pid`, `err.txt`, …) |
| No-old-import gate owned by cutover | rg recipes above; landed flat dual-copy **files** removed for G020/G030/G040 owner map stems |
| Remaining flat modules for **ASREF-G050** | inventoried; move owned by context/prompt lanes |
| Remaining flat modules (analysis/proof/integrations + G070 remainder) | still under child goals **ASREF-G060**, **ASREF-G070**, **ASREF-G080** |
| Entry-point rewrites for remaining flat modules | complete when each owning move task lands |
| Historical tracked process files / root tests | ignore rules present; untrack/move needs broader path scope than cutover-only outputs |
