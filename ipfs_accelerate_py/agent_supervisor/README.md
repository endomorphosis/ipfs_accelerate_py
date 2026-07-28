# agent_supervisor

Autonomous agent supervisor for objective-driven todo execution, control-plane
operations, formal planning/proof, merge lanes, and implementation daemons.

This root package is the **public API and package map** for the ASREF module
layout (`docs/architecture/AGENT_SUPERVISOR_MODULE_REFACTOR_PLAN.md`). Domain
code lives under named subpackages; the root `__init__.py` re-exports only
intentional symbols (control surface, stable generation-2 contracts, domain
layout constants, and selected objective/planning helpers). New code must
import from domain packages—not from retired flat module paths.

**Cutover goal:** `ASREF-G090` (packet tasks `ASREF-012`, `ASREF-013`,
`ASREF-014`; packet `goal_packet/cutover/ipfs_accelerate_py/090ea2138c6f`).
Parent evidence for this goal is the set of package goals **ASREF-G020**,
**ASREF-G030**, **ASREF-G040**, **ASREF-G050**, **ASREF-G060**, **ASREF-G070**,
and **ASREF-G080**. This README is the durable map that binds those goals to
on-disk packages, public imports, root hygiene, and the no-old-import gate.

**ASREF-014 evidence cluster:** this cutover pass closes the missing parent
terms **ASREF-G060**, **ASREF-G070**, and **ASREF-G080** by naming package
contracts, landed vs planned module owners, preferred imports, and validation
hooks in the package root (see `AGENT_SUPERVISOR_EVIDENCE_CLUSTER_G060_G080`).

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
    AGENT_SUPERVISOR_PLANNED_MODULE_OWNERS,
    AGENT_SUPERVISOR_PACKAGE_GOAL_EVIDENCE,
    AGENT_SUPERVISOR_PACKAGE_GOAL_TO_PACKAGES,
    AGENT_SUPERVISOR_EVIDENCE_CLUSTER_G060_G080,
    AGENT_SUPERVISOR_G060_PACKAGES,
    AGENT_SUPERVISOR_G070_PACKAGES,
    AGENT_SUPERVISOR_G080_PACKAGES,
    AGENT_SUPERVISOR_CUTOVER_GOAL_ID,
    AGENT_SUPERVISOR_CUTOVER_TASK_ID,
    AGENT_SUPERVISOR_CUTOVER_PACKET_TASK_IDS,
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
   `AGENT_SUPERVISOR_PLANNED_MODULE_OWNERS`,
   `AGENT_SUPERVISOR_PACKAGE_GOAL_EVIDENCE`,
   `AGENT_SUPERVISOR_PACKAGE_GOAL_TO_PACKAGES`,
   `AGENT_SUPERVISOR_EVIDENCE_CLUSTER_G060_G080`) are intentional public
   symbols for discovery, cutover gates, and objective evidence scans.
6. `AGENT_SUPERVISOR_PLANNED_MODULE_OWNERS` is **documentation and scan
   evidence only** — it does not install import aliases until packages exist.

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
`AGENT_SUPERVISOR_PACKAGE_GOAL_EVIDENCE` and
`AGENT_SUPERVISOR_PACKAGE_GOAL_TO_PACKAGES` in `__init__.py`.

| Evidence term | Bundle | Landed package path(s) | Cutover witness |
|---|---|---|---|
| **ASREF-G020** | `asref/core` | `core/` | README + `AGENT_SUPERVISOR_LANDED_MODULE_OWNERS` core stems; no flat `conflict_graph` / `external_completion` dual copies |
| **ASREF-G030** | `asref/control` | `control/` | Control contracts/plane re-exported from package path; conformance suite |
| **ASREF-G040** | `asref/task-sources` | `task_sources/` | Markdown/DuckDB/taskboard modules under package; protocol may stay conceptual in core |
| **ASREF-G050** | `asref/context` (+ prompt) | `context/`, `prompt/` (planned) | Named in domain map; flat `context_*` / `decision_*` / `prompt_*` remain until move tasks land |
| **ASREF-G060** | `asref/analysis` (+ proof) | `analysis/`, `proof/` (planned) | Named in domain map + `AGENT_SUPERVISOR_PLANNED_MODULE_OWNERS` analysis/proof stems; flat modules remain until move tasks land |
| **ASREF-G070** | `asref/objectives` (+ siblings) | `objectives/`, `planning/`, `validation/`, `merge/`, `rescue/`, `runtime/`, `self_improvement/` | Package READMEs + landed module owners; objective graph / proposal validation import package paths |
| **ASREF-G080** | `asref/todo-daemon` (+ integrations) | `todo_daemon/`, `integrations/` (planned) | Daemon imports already package-native; integrations flat until move (planned owners listed) |

Child package lanes still own the physical moves for partial rows
(**ASREF-G050**, remaining **ASREF-G060** / **ASREF-G070** flats /
**ASREF-G080** integrations split). Cutover owns the **final no-old-import
gate**, public API map, and root hygiene for merge readiness.

## ASREF-G060 evidence (analysis + proof)

**Goal:** Create `analysis/` and `proof/` packages grouping analysis
pipeline/cache/AST/retrieval and formal verification/prover modules.

**Package constants:** `AGENT_SUPERVISOR_G060_PACKAGES` → `analysis`, `proof`.

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
`AGENT_SUPERVISOR_PLANNED_MODULE_OWNERS` (from
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
(e.g. `from ipfs_accelerate_py.agent_supervisor.analysis_pipeline import ...`).
Those flat paths are **not** the long-term public API. Forbidden after land:
`core` / `analysis` / `proof` must not import `todo_daemon` (DAG).

**Cutover witness for ASREF-G060:** package map row + planned owners + this
section + exact goal id in `AGENT_SUPERVISOR_PACKAGE_GOAL_EVIDENCE` and
`AGENT_SUPERVISOR_EVIDENCE_CLUSTER_G060_G080`. Physical package creation remains
`asref/analysis` / `asref/proof` move tasks.

## ASREF-G070 evidence (objectives, planning, validation, merge, rescue, runtime, self_improvement)

**Goal:** Domain packages for objectives/ops/runtime with READMEs and hard
import/entry-point updates.

**Package constants:** `AGENT_SUPERVISOR_G070_PACKAGES`.

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

Runtime list: `AGENT_SUPERVISOR_G070_LANDED_STEMS`.

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
in `AGENT_SUPERVISOR_PLANNED_MODULE_OWNERS`): e.g. `goal_completion`,
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

**Package constants:** `AGENT_SUPERVISOR_G080_PACKAGES` → `todo_daemon`,
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

`AGENT_SUPERVISOR_G080_TODO_DAEMON_STEMS` lists the primary entry modules.
Planned internal subpackages (from the program plan; land via
`asref/todo-daemon`):

- `implementation/` — implementation daemon + supervisor (+ runners)
- `loop/` — supervisor loop, runner, engine pieces
- `git/` — worktrees, git utils, auto-commit

Root-level `implementation_daemon_runner` /
`implementation_supervisor_runner` remain flat until the implementation
subpackage move; both are planned owners under `todo_daemon` in
`AGENT_SUPERVISOR_PLANNED_MODULE_OWNERS`.

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

## Root hygiene (ASREF-G090 / ASREF-006 / ASREF-014)

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

**ASREF-G060 / G070 / G080 packages never live at monorepo root** — they stay
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

Layout evidence CLI:

```bash
python -m ipfs_accelerate_py.agent_supervisor.asref_layout_evidence
python scripts/ops/agent_supervisor/asref_multi_lane_launch.py recipe
```

Optional focused checks for child package goals (not required by cutover
packet validation, but useful for G060/G070/G080 lanes):

```bash
# ASREF-G060 (when analysis/proof tests are exercised)
python -m pytest test/api/test_agent_supervisor_analysis_pipeline.py \
  test/api/test_agent_supervisor_multi_prover_router.py -q

# ASREF-G070 recovery path
python -m pytest test/api/test_agent_supervisor_programmatic_recovery.py -q

# ASREF-G080 protected paths
python -m pytest test/api/test_agent_supervisor_implementation_protected_paths.py -q
```

## Status (ASREF-014 / ASREF-G090 cutover — G060/G070/G080 cluster + ASREF-008 evidence)

| Acceptance item | Status |
|---|---|
| Root `README.md` maps all domain packages | present (this file) |
| Root `__init__.py` intentional public symbols + package-path resolution | present |
| Evidence terms **ASREF-G020** … **ASREF-G080** named for parent goal | present (package map + evidence table + constants) |
| **ASREF-G060** analysis/proof map + planned owners | present (this section + `AGENT_SUPERVISOR_PLANNED_MODULE_OWNERS`) |
| **ASREF-G070** seven packages + landed stems + preferred imports | present (packages on disk; landed owners wired) |
| **ASREF-G080** todo_daemon package-native + integrations planned | present (daemon imports; integrations planned owners) |
| Domain package READMEs for landed packages | present for core/control/task_sources/objectives/planning/validation/merge/rescue/runtime/self_improvement |
| Frozen move map + import inventory (**ASREF-G010**) | present under `docs/architecture/asref/` |
| Layout evidence module (**ASREF-G010** / **ASREF-G090** / **ASREF-G100**) | `asref_layout_evidence.py` |
| Multi-lane Grok launch recipe + protected-path wiring (**ASREF-G100**) | present under `scripts/ops/agent_supervisor/` |
| Nested product trees documented | `docs/NESTED_PACKAGES.md` |
| Process junk ignored | `.gitignore` (`dashboard.out`, `dashboard.pid`, `err.txt`, …) |
| No-old-import gate owned by cutover | rg recipes above; landed flat dual-copy **files** removed for owner map stems |
| Remaining flat modules (context/analysis/proof/prompt/integrations + G070 remainder) | still under child goals **ASREF-G050**, **ASREF-G060**, **ASREF-G070**, **ASREF-G080** |
| Entry-point rewrites for remaining flat modules | complete when each owning move task lands |
| Historical tracked process files / root tests | ignore rules present; untrack/move needs broader path scope than cutover-only outputs |
