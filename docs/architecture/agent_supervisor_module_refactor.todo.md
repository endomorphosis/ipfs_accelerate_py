# Agent Supervisor Module Refactor Task Board

This board is the executable projection of the
[Agent Supervisor Module Refactor objective heap](agent_supervisor_module_refactor.objectives.md).
See the human plan
[AGENT_SUPERVISOR_MODULE_REFACTOR_PLAN.md](AGENT_SUPERVISOR_MODULE_REFACTOR_PLAN.md).

Program invariants:

- Branch: `refactor/agent-supervisor-layout` only until cutover.
- Package submodules are Python packages with `README.md`, not git submodules.
- **No thin compatibility wrappers** at old flat `agent_supervisor/*.py` paths.
  Update imports, entry points, scripts, tests, and docs in the same task as
  each `git mv`.
- Protected operator inputs (never claim as editable Outputs of refill tasks):
  - `docs/architecture/agent_supervisor_module_refactor.objectives.md`
  - `docs/architecture/agent_supervisor_module_refactor.todo.md`
  - `docs/architecture/AGENT_SUPERVISOR_MODULE_REFACTOR_PLAN.md`
- Concurrent lanes must honor `Bundle` / `Conflict policy` file ownership.
- Completion requires the task `Validation` command to pass on the current tree.
- Prefer Grok 4.6 (or the configured implementation provider) with one package
  move per task after the freeze map exists.

Seeded bootstrap tasks appear below. The objective daemon may append gap tasks;
do not delete protected headers or rewrite completed history.

## ASREF-001 Create branch and freeze inventory move map

- Status: pending
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: bootstrap
- Depends on:
- Goal id: ASREF-G010
- Outputs: data/agent_supervisor/discovery/asref/move_map.json, data/agent_supervisor/discovery/asref/import_inventory.md, docs/architecture/AGENT_SUPERVISOR_MODULE_REFACTOR_PLAN.md
- Validation: test -f data/agent_supervisor/discovery/asref/move_map.json && test -f data/agent_supervisor/discovery/asref/import_inventory.md
- Board namespace: agent-supervisor-module-refactor-v1
- Bundle: asref/bootstrap
- Parallel lane: asref-bootstrap
- Resource class: cpu-small
- Token class: medium
- Estimated tokens: 8000
- Predicted files: data/agent_supervisor/discovery/asref/move_map.json, data/agent_supervisor/discovery/asref/import_inventory.md, docs/architecture/AGENT_SUPERVISOR_MODULE_REFACTOR_PLAN.md
- Allow concurrent with:
- Conflict policy: Bootstrap only. Do not move production modules yet. Do not edit unrelated packages.
- Preconditions: Working tree can create branch `refactor/agent-supervisor-layout` from `origin/main`. List every `ipfs_accelerate_py/agent_supervisor/**/*.py` module.
- Effects: Create/update branch; write move_map.json mapping each module to target package, bundle, and known import sites; write import_inventory.md including dynamic imports; ensure plan/objectives/todo exist on the branch.
- Acceptance: Branch exists; every production module is assigned exactly one target package; entry points from pyproject.toml and setup.py appear in the inventory; package DAG in the plan remains acyclic; no code moves yet.

## ASREF-002 Seed multi-lane launch recipe for Grok 4.6

- Status: pending
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: autonomous-execution
- Depends on: ASREF-001
- Goal id: ASREF-G100
- Outputs: data/agent_supervisor/bundles/asref, docs/architecture/AGENT_SUPERVISOR_MODULE_REFACTOR_PLAN.md, scripts/ops/agent_supervisor
- Validation: test -d data/agent_supervisor/bundles/asref || test -f docs/architecture/AGENT_SUPERVISOR_MODULE_REFACTOR_PLAN.md
- Board namespace: agent-supervisor-module-refactor-v1
- Bundle: asref/bootstrap
- Parallel lane: asref-bootstrap
- Resource class: cpu-small
- Token class: medium
- Estimated tokens: 6000
- Predicted files: docs/architecture/AGENT_SUPERVISOR_MODULE_REFACTOR_PLAN.md, scripts/ops/agent_supervisor
- Allow concurrent with:
- Conflict policy: Documentation and launch wiring only; no package moves.
- Preconditions: ASREF-001 inventory exists.
- Effects: Document exact objective-daemon and multi-lane supervisor commands for this heap; create empty bundle dir layout; ensure protected-path flags are listed for the three architecture files; note Grok 4.6 provider selection env/flags.
- Acceptance: An operator can launch objective scan and implementation lanes for ASREF without reading other programs; protected paths are explicit; provider selection is documented.

## ASREF-003 Create core package and move shared modules

- Status: pending
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: package-core
- Depends on: ASREF-001
- Goal id: ASREF-G020
- Outputs: ipfs_accelerate_py/agent_supervisor/core/README.md, ipfs_accelerate_py/agent_supervisor/core/__init__.py, ipfs_accelerate_py/agent_supervisor/core
- Validation: python -m pytest test/api/test_agent_supervisor_event_driven_runtime.py -q --collect-only
- Board namespace: agent-supervisor-module-refactor-v1
- Bundle: asref/core
- Parallel lane: asref-core
- Resource class: cpu-small
- Token class: large
- Estimated tokens: 16000
- Predicted files: ipfs_accelerate_py/agent_supervisor/core, test/api/test_agent_supervisor_event_driven_runtime.py
- Allow concurrent with: ASREF-010
- Conflict policy: Owns only core-mapped modules from move_map.json and their import updates. No todo_daemon moves.
- Preconditions: move_map.json lists core modules. Use git mv.
- Effects: Create core package + README; move modules; update all imports/tests/scripts; delete old paths without stubs.
- Acceptance: Core package README states allowed dependents; old flat paths for moved modules are gone; imports resolve; collection/import of event-driven tests succeeds.

## ASREF-004 Create control package and update CLI entry surfaces

- Status: pending
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: package-control
- Depends on: ASREF-003
- Goal id: ASREF-G030
- Outputs: ipfs_accelerate_py/agent_supervisor/control/README.md, ipfs_accelerate_py/agent_supervisor/control
- Validation: python -m pytest test/api/test_agent_supervisor_control_conformance_v2.py test/api/test_agent_supervisor_lifecycle_orchestrator.py -q
- Board namespace: agent-supervisor-module-refactor-v1
- Bundle: asref/control
- Parallel lane: asref-control
- Resource class: cpu-small
- Token class: large
- Estimated tokens: 15000
- Predicted files: ipfs_accelerate_py/agent_supervisor/control, test/api/test_agent_supervisor_control_conformance_v2.py
- Allow concurrent with: ASREF-005, ASREF-006
- Conflict policy: Owns control-mapped modules only. Do not edit core package contents except imports.
- Preconditions: core package landed.
- Effects: git mv control modules; fix imports and docs; run control conformance tests.
- Acceptance: Validation suite passes; no old control_* flat modules remain; README present.

## ASREF-005 Create task_sources package

- Status: pending
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: package-task-sources
- Depends on: ASREF-003
- Goal id: ASREF-G040
- Outputs: ipfs_accelerate_py/agent_supervisor/task_sources/README.md, ipfs_accelerate_py/agent_supervisor/task_sources
- Validation: python -m pytest test/api/test_agent_supervisor_markdown_task_source.py test/api/test_agent_supervisor_duckdb_task_source.py -q
- Board namespace: agent-supervisor-module-refactor-v1
- Bundle: asref/task-sources
- Parallel lane: asref-task-sources
- Resource class: cpu-small
- Token class: large
- Estimated tokens: 14000
- Predicted files: ipfs_accelerate_py/agent_supervisor/task_sources
- Allow concurrent with: ASREF-004, ASREF-006
- Conflict policy: Owns task_sources-mapped modules only.
- Preconditions: core package landed; task_source protocol remains importable from core.
- Effects: Move markdown/duckdb/taskboard/queue modules; update imports; README.
- Acceptance: Dual task-source tests pass; no flat leftovers for moved modules.

## ASREF-006 Monorepo root hygiene pass

- Status: pending
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: root-hygiene
- Depends on: ASREF-001
- Goal id: ASREF-G090
- Outputs: .gitignore, docs/NESTED_PACKAGES.md, docs/architecture/MCP_SERVER_UNIFICATION_PLAN.md
- Validation: test ! -f dashboard.out && test ! -f dashboard.pid && test ! -f err.txt || true
- Board namespace: agent-supervisor-module-refactor-v1
- Bundle: asref/root-hygiene
- Parallel lane: asref-root-hygiene
- Resource class: cpu-small
- Token class: small
- Estimated tokens: 5000
- Predicted files: .gitignore, docs/NESTED_PACKAGES.md, README.md
- Allow concurrent with: ASREF-003, ASREF-004, ASREF-005
- Conflict policy: Root and docs only. Do not move agent_supervisor packages.
- Preconditions: Inventory of root clutter exists in plan or discovery note.
- Effects: Ignore or remove ephemeral runtime files; move misplaced root plans/tests into docs/test; document nested product trees; do not delete nested package checkouts without ownership review.
- Acceptance: Root no longer tracks process junk; nested packages documented; no agent_supervisor import breakage from this task.

## ASREF-007 Final public API README and no-old-import cutover gate

- Status: pending
- Completion: manual
- Is schedulable: false
- Review only: false
- Priority: P0
- Track: cutover
- Depends on: ASREF-003, ASREF-004, ASREF-005, ASREF-006
- Goal id: ASREF-G090
- Outputs: ipfs_accelerate_py/agent_supervisor/README.md, ipfs_accelerate_py/agent_supervisor/__init__.py
- Validation: python -m pytest test/api/test_agent_supervisor_todo_daemon_port.py test/api/test_agent_supervisor_control_conformance_v2.py test/api/test_agent_supervisor_objective_graph.py test/api/test_agent_supervisor_proposal_validation.py -q
- Board namespace: agent-supervisor-module-refactor-v1
- Bundle: asref/cutover
- Parallel lane: asref-cutover
- Resource class: cpu-medium
- Token class: large
- Estimated tokens: 18000
- Predicted files: ipfs_accelerate_py/agent_supervisor/README.md, ipfs_accelerate_py/agent_supervisor/__init__.py, pyproject.toml, setup.py
- Allow concurrent with:
- Conflict policy: Final sweep. Depends on all package moves being complete; re-enable Is schedulable only after ASREF-G070/G080 package tasks land (objective refill will add those).
- Preconditions: All domain packages from the freeze map exist. Entry points updated. No compatibility stubs.
- Effects: Write root package README map; explicit __all__; rg-based old-path purge; full validation suite.
- Acceptance: Validation suite passes; README maps every package; no retired flat import paths remain in code or entry points.
