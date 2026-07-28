# Agent Supervisor Module Refactor Objective Heap

This objective heap is the durable source of intent for reorganizing
`ipfs_accelerate_py.agent_supervisor` into domain package submodules with
per-package READMEs, cleaning monorepo root clutter, and updating every
dependent import, script, entry point, and test to the new paths.

The companion todo board
[`agent_supervisor_module_refactor.todo.md`](./agent_supervisor_module_refactor.todo.md)
is the executable projection consumed by the agent supervisor. The human plan is
[`AGENT_SUPERVISOR_MODULE_REFACTOR_PLAN.md`](./AGENT_SUPERVISOR_MODULE_REFACTOR_PLAN.md).

Program invariants:

- Work lands only on branch `refactor/agent-supervisor-layout` until cutover.
- Package submodules are **Python packages**, not git submodules.
- Each package has `README.md` stating purpose, public surface, and allowed deps.
- Moves use `git mv` and update all callers in the same task; **no thin
  compatibility re-export stubs** at old flat module paths.
- Console scripts, ops scripts, tests, docs, and protected-path lists must
  resolve to new modules before a move goal can close.
- Goals close only with current-tree evidence (files + validation), not todo
  status alone or model chatter.
- Parallel lanes must respect `Bundle:` ownership; do not edit another bundle’s
  `Outputs:` paths without refining the heap first.
- Implementation agents (including Grok 4.6) follow each goal’s `Validation:`
  line before marking work done.

## ASREF-G000 Clear agent_supervisor package layout and monorepo root hygiene

- Status: active
- Parent:
- Fib priority: 1
- Track: agent-supervisor-refactor
- Priority: P0
- Bundle: asref/root
- Goal: Reorganize agent_supervisor into domain package submodules with README contracts, eliminate flat-module sprawl, update every dependent path without shims, and clean monorepo root clutter so autonomous agents and humans can navigate ownership safely.
- Evidence: ASREF-G010, ASREF-G020, ASREF-G030, ASREF-G040, ASREF-G050, ASREF-G060, ASREF-G070, ASREF-G080, ASREF-G090, ASREF-G100
- Outputs: ipfs_accelerate_py/agent_supervisor, docs/architecture/agent_supervisor_module_refactor.objectives.md, docs/architecture/agent_supervisor_module_refactor.todo.md, docs/architecture/AGENT_SUPERVISOR_MODULE_REFACTOR_PLAN.md, pyproject.toml, setup.py, scripts, test/api
- Validation: python -m pytest test/api/test_agent_supervisor_todo_daemon_port.py test/api/test_agent_supervisor_control_conformance_v2.py -q --collect-only && rg -n "agent_supervisor\\.(objective_daemon|backlog_refinery|merge_resolver)\\b" pyproject.toml setup.py || true
- Acceptance: Flat production modules under agent_supervisor root are gone except __init__.py and py.typed; every domain package has README.md; all entry points and scripts import new paths; no old-path re-export stubs remain; monorepo root has no ephemeral process files or misplaced root tests; full agent-supervisor API test selection for the branch passes; this heap’s child goals are complete with current-tree evidence.
- Gap task: Implement the highest-priority incomplete child with focused tests and import-path updates only inside that child’s Outputs.
- Refinement: Bootstrap inventory and move map first; land independent packages in parallel after core; re-pack todo_daemon and public API late; root hygiene can run in parallel after bootstrap; cutover is last.
- Embedding query: agent supervisor package layout refactor domain modules README import paths root hygiene no compatibility wrappers
- AST query: objective_daemon implementation_daemon implementation_supervisor control_plane proposal_validation

## ASREF-G010 Branch bootstrap inventory and frozen move map

- Status: active
- Parent: ASREF-G000
- Fib priority: 2
- Track: bootstrap
- Priority: P0
- Bundle: asref/bootstrap
- Goal: Create branch refactor/agent-supervisor-layout from origin/main, inventory every agent_supervisor module and dynamic import, and freeze a file-to-package move map with conflict domains for parallel lanes.
- Evidence: docs/architecture/AGENT_SUPERVISOR_MODULE_REFACTOR_PLAN.md, docs/architecture/agent_supervisor_module_refactor.objectives.md
- Outputs: docs/architecture/AGENT_SUPERVISOR_MODULE_REFACTOR_PLAN.md, data/agent_supervisor/discovery/asref/move_map.json, data/agent_supervisor/discovery/asref/import_inventory.md
- Validation: test -f docs/architecture/AGENT_SUPERVISOR_MODULE_REFACTOR_PLAN.md && test -f data/agent_supervisor/discovery/asref/move_map.json
- Acceptance: Branch exists; move_map.json lists every top-level and todo_daemon module with target package, owning bundle, and dependent entry-point/script hits; import_inventory.md lists dynamic import sites; plan documents the package DAG and no-shim rule; objectives heap is committed on the branch.
- Gap task: Generate move_map.json and import_inventory.md from a deterministic scan and open the branch if missing.
- Refinement: Do not move code in this goal; only inventory and document.
- Embedding query: branch inventory import graph move map agent supervisor modules conflict domain
- AST query: importlib.__import__ import_module getattr

## ASREF-G020 Core package submodule

- Status: active
- Parent: ASREF-G000
- Fib priority: 3
- Track: package-core
- Priority: P0
- Bundle: asref/core
- Goal: Create agent_supervisor/core with shared contracts, identity, events, and task-source protocol modules, README, and updated imports for all callers.
- Evidence: ipfs_accelerate_py/agent_supervisor/task_identity.py, ipfs_accelerate_py/agent_supervisor/event_log.py, ipfs_accelerate_py/agent_supervisor/task_source.py
- Outputs: ipfs_accelerate_py/agent_supervisor/core/README.md, ipfs_accelerate_py/agent_supervisor/core/__init__.py, ipfs_accelerate_py/agent_supervisor/core
- Validation: python -m pytest test/api/test_agent_supervisor_event_driven_runtime.py -q --collect-only
- Acceptance: Core package exists with README describing allowed dependents; former flat core modules live only under core/; no re-export stub at old paths; imports updated across package, tests, and scripts that referenced moved modules; core introduces no dependency on todo_daemon or runtime.
- Gap task: git mv the core-mapped modules, write README, fix imports, run focused tests.
- Refinement: Land identity/event/task_source first if the full set is too large for one task.
- Embedding query: agent supervisor core package task identity event log task source contracts
- AST query: TaskIdentity EventLog TaskSource

## ASREF-G030 Control package submodule

- Status: active
- Parent: ASREF-G000
- Fib priority: 5
- Track: package-control
- Priority: P0
- Bundle: asref/control
- Goal: Create agent_supervisor/control for control plane, CLI, contracts, lifecycle orchestration, and execution permits with README and hard import updates.
- Evidence: ipfs_accelerate_py/agent_supervisor/control_plane.py, ipfs_accelerate_py/agent_supervisor/control_cli.py, ipfs_accelerate_py/agent_supervisor/control_contracts.py, ipfs_accelerate_py/agent_supervisor/lifecycle_orchestrator.py
- Outputs: ipfs_accelerate_py/agent_supervisor/control/README.md, ipfs_accelerate_py/agent_supervisor/control
- Validation: python -m pytest test/api/test_agent_supervisor_control_conformance_v2.py test/api/test_agent_supervisor_lifecycle_orchestrator.py -q
- Acceptance: Control modules live only under control/; entry points and CLI docs use new imports; conformance and lifecycle tests pass; package depends only on core and declared deps.
- Gap task: Move control_* lifecycle and permit modules; update tests and any MCP control tools.
- Refinement: Keep authorization_logic with control if it is the policy surface.
- Embedding query: control plane CLI lifecycle execution permit agent supervisor package
- AST query: ControlPlane control_cli LifecycleOrchestrator ExecutionPermit

## ASREF-G040 Task sources package submodule

- Status: active
- Parent: ASREF-G000
- Fib priority: 5
- Track: package-task-sources
- Priority: P0
- Bundle: asref/task-sources
- Goal: Create agent_supervisor/task_sources for Markdown and DuckDB projections, taskboard store, persistent queues, and todo vector index with README and caller updates.
- Evidence: ipfs_accelerate_py/agent_supervisor/markdown_task_source.py, ipfs_accelerate_py/agent_supervisor/duckdb_task_source.py, ipfs_accelerate_py/agent_supervisor/taskboard_store.py, ipfs_accelerate_py/agent_supervisor/persistent_task_queue.py
- Outputs: ipfs_accelerate_py/agent_supervisor/task_sources/README.md, ipfs_accelerate_py/agent_supervisor/task_sources
- Validation: python -m pytest test/api/test_agent_supervisor_markdown_task_source.py test/api/test_agent_supervisor_duckdb_task_source.py -q
- Acceptance: Task source modules live only under task_sources/; dual-projection tests pass; no old flat paths; README documents protocol dependence on core.task_source.
- Gap task: Move markdown/duckdb/taskboard/queue/vector modules and fix imports.
- Refinement: Keep protocol in core; implementations in task_sources.
- Embedding query: markdown duckdb task source taskboard persistent queue vector index
- AST query: MarkdownTaskSource DuckDBTaskSource TaskboardStore PersistentTaskQueue

## ASREF-G050 Context and prompt package submodules

- Status: active
- Parent: ASREF-G000
- Fib priority: 8
- Track: package-context-prompt
- Priority: P0
- Bundle: asref/context
- Goal: Create agent_supervisor/context and agent_supervisor/prompt packages for context compilation, decision runtime, and prompt workflow/scanner/admission/planner surfaces with READMEs and import updates.
- Evidence: ipfs_accelerate_py/agent_supervisor/context_compiler.py, ipfs_accelerate_py/agent_supervisor/context_contracts.py, ipfs_accelerate_py/agent_supervisor/prompt_workflow.py, ipfs_accelerate_py/agent_supervisor/prompt_directory_scanner.py
- Outputs: ipfs_accelerate_py/agent_supervisor/context/README.md, ipfs_accelerate_py/agent_supervisor/context, ipfs_accelerate_py/agent_supervisor/prompt/README.md, ipfs_accelerate_py/agent_supervisor/prompt
- Validation: python -m pytest test/api/test_agent_supervisor_context_compiler.py test/api/test_agent_supervisor_prompt_workflow_service.py -q
- Acceptance: context_* and decision_* modules live under context/; prompt_* modules live under prompt/; tests and scripts updated; packages have READMEs; prompt may depend on context but not vice versa.
- Gap task: Split moves into two commits if needed (context first, then prompt) still under this goal’s acceptance.
- Refinement: Prefer sequential context then prompt if one lane cannot hold both.
- Embedding query: context compiler decision runtime prompt workflow directory scanner plan admission
- AST query: ContextCompiler ContextCapsule PromptWorkflowService PromptDirectoryScanner

## ASREF-G060 Analysis and proof package submodules

- Status: active
- Parent: ASREF-G000
- Fib priority: 8
- Track: package-analysis-proof
- Priority: P0
- Bundle: asref/analysis
- Goal: Create agent_supervisor/analysis and agent_supervisor/proof packages grouping analysis pipeline/cache/AST/retrieval and formal verification/prover/leanstral modules with READMEs and import updates.
- Evidence: ipfs_accelerate_py/agent_supervisor/analysis_pipeline.py, ipfs_accelerate_py/agent_supervisor/formal_verification_provider.py, ipfs_accelerate_py/agent_supervisor/multi_prover_router.py
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/README.md, ipfs_accelerate_py/agent_supervisor/analysis, ipfs_accelerate_py/agent_supervisor/proof/README.md, ipfs_accelerate_py/agent_supervisor/proof
- Validation: python -m pytest test/api/test_agent_supervisor_analysis_pipeline.py test/api/test_agent_supervisor_multi_prover_router.py -q
- Acceptance: analysis_* modules under analysis/; formal_*/proof_*/prover_*/leanstral_* and related under proof/; no flat leftovers for those prefixes; tests pass; packages document forbidden imports from todo_daemon.
- Gap task: Move analysis package first if proof set is too large; keep DAG acyclic.
- Refinement: Bundle asref/proof may be split by heap refine if merge conflicts appear.
- Embedding query: analysis pipeline AST cache formal verification prover leanstral proof package
- AST query: AnalysisPipeline AnalysisCache FormalVerificationProvider MultiProverRouter

## ASREF-G070 Objectives planning validation merge rescue runtime packages

- Status: active
- Parent: ASREF-G000
- Fib priority: 13
- Track: package-ops-runtime
- Priority: P0
- Bundle: asref/objectives
- Goal: Create remaining domain packages—objectives, planning, validation, merge, rescue, runtime, self_improvement—each with README and hard import/entry-point updates.
- Evidence: ipfs_accelerate_py/agent_supervisor/objective_graph.py, ipfs_accelerate_py/agent_supervisor/objective_daemon.py, ipfs_accelerate_py/agent_supervisor/proposal_validation.py, ipfs_accelerate_py/agent_supervisor/merge_resolver.py, ipfs_accelerate_py/agent_supervisor/rescue_orchestrator.py, ipfs_accelerate_py/agent_supervisor/multi_supervisor_runner.py
- Outputs: ipfs_accelerate_py/agent_supervisor/objectives, ipfs_accelerate_py/agent_supervisor/planning, ipfs_accelerate_py/agent_supervisor/validation, ipfs_accelerate_py/agent_supervisor/merge, ipfs_accelerate_py/agent_supervisor/rescue, ipfs_accelerate_py/agent_supervisor/runtime, ipfs_accelerate_py/agent_supervisor/self_improvement
- Validation: python -m pytest test/api/test_agent_supervisor_objective_graph.py test/api/test_agent_supervisor_proposal_validation.py test/api/test_agent_supervisor_programmatic_recovery.py -q
- Acceptance: Each listed package exists with README; modules from the frozen move map live only in their packages; pyproject entry points for objective-daemon, backlog-refinery, bundle-supervisor, merge-resolver point at new modules; focused tests pass; no old-path stubs.
- Gap task: Implement one package per task using bundle refinement (asref/objectives, asref/planning, asref/validation, asref/merge, asref/rescue, asref/runtime, asref/self-improvement).
- Refinement: Split this parent into one child goal per package when generating todos; do not move todo_daemon here.
- Embedding query: objective daemon planning validation merge rescue multi supervisor self improvement packages
- AST query: parse_goal_heap ObjectiveDaemon ProposalValidation MergeResolver RescueOrchestrator MultiSupervisorRunner

## ASREF-G080 Todo daemon re-packaging and integrations package

- Status: active
- Parent: ASREF-G000
- Fib priority: 21
- Track: package-todo-daemon
- Priority: P0
- Bundle: asref/todo-daemon
- Goal: Re-package todo_daemon into clear internal subpackages (implementation, loop, git, …), move integration runners (llm merge fallback, meta spark/goose, datasets providers) into integrations/, and update all console scripts and ops scripts without shims.
- Evidence: ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_supervisor.py, ipfs_accelerate_py/agent_supervisor/llm_merge_resolver_fallback.py, scripts/ops/agent_supervisor
- Outputs: ipfs_accelerate_py/agent_supervisor/todo_daemon/README.md, ipfs_accelerate_py/agent_supervisor/todo_daemon, ipfs_accelerate_py/agent_supervisor/integrations/README.md, ipfs_accelerate_py/agent_supervisor/integrations, pyproject.toml, setup.py, scripts/ops/agent_supervisor
- Validation: python -m pytest test/api/test_agent_supervisor_todo_daemon_port.py test/api/test_agent_supervisor_implementation_protected_paths.py -q
- Acceptance: implementation daemon and supervisor import paths updated in entry points and scripts; todo_daemon README maps subpackages; integrations README lists external tool bridges; daemon port tests and protected-path tests pass; no flat integration modules remain at agent_supervisor root.
- Gap task: Move implementation_daemon last among hot files; update scripts/ops entrypoints in the same commit.
- Refinement: Prefer implementation/ subpackage plus loop/ and git/ splits over one giant move.
- Embedding query: implementation daemon supervisor todo_daemon integrations goose meta spark llm merge resolver
- AST query: TodoImplementationDaemon TodoImplementationSupervisor llm_merge_resolver_fallback

## ASREF-G090 Public API package README root hygiene and cutover

- Status: active
- Parent: ASREF-G000
- Fib priority: 34
- Track: cutover
- Priority: P0
- Bundle: asref/cutover
- Goal: Publish agent_supervisor root README and explicit __init__ exports, finish monorepo root hygiene, eliminate every remaining old import path, and prove branch readiness for merge.
- Evidence: ASREF-G020, ASREF-G030, ASREF-G040, ASREF-G050, ASREF-G060, ASREF-G070, ASREF-G080
- Outputs: ipfs_accelerate_py/agent_supervisor/README.md, ipfs_accelerate_py/agent_supervisor/__init__.py, docs/NESTED_PACKAGES.md, .gitignore, docs/architecture/AGENT_SUPERVISOR_MODULE_REFACTOR_PLAN.md
- Validation: python -m pytest test/api/test_agent_supervisor_todo_daemon_port.py test/api/test_agent_supervisor_control_conformance_v2.py test/api/test_agent_supervisor_objective_graph.py test/api/test_agent_supervisor_proposal_validation.py -q
- Acceptance: Root agent_supervisor/README.md maps all packages; __init__.py exports only intentional public symbols; rg finds no imports of retired flat module paths; monorepo root has no dashboard.out/dashboard.pid/err.txt tracked; misplaced root test_*.py live under test/; nested product trees documented; full listed validation suite passes on the branch.
- Gap task: Final import sweep, root hygiene, README, and CI-facing validation.
- Refinement: Root hygiene may land earlier under asref/root-hygiene if parallelized, but cutover still owns the final no-old-import gate.
- Embedding query: public API README root hygiene gitignore nested packages import sweep cutover
- AST query: __all__

## ASREF-G100 Autonomous supervisor execution with Grok 4.6

- Status: active
- Parent: ASREF-G000
- Fib priority: 4
- Track: autonomous-execution
- Priority: P1
- Bundle: asref/bootstrap
- Goal: Wire this objective heap and todo board into a multi-lane implementation supervisor configuration that uses Grok 4.6 (or the configured implementation provider) to drain ASREF tasks with protected-path safety and bundle isolation.
- Evidence: docs/architecture/AGENT_SUPERVISOR_MODULE_REFACTOR_PLAN.md, docs/architecture/agent_supervisor_module_refactor.todo.md, scripts/ops/agent_supervisor
- Outputs: docs/architecture/agent_supervisor_module_refactor.todo.md, data/agent_supervisor/bundles/asref, scripts/ops/agent_supervisor
- Validation: test -f docs/architecture/agent_supervisor_module_refactor.todo.md && test -d data/agent_supervisor/bundles/asref || true
- Acceptance: Objective daemon can scan this heap into the todo board; bundle index assigns lanes by Bundle fields; implementation supervisor launch docs/scripts protect the three architecture files; Grok 4.6 (or successor) is selectable as implementation provider without changing goal text; workers follow Validation lines and the no-shim rule.
- Gap task: Seed todo board, generate first objective scan, and document the exact multi-lane launch command for this program.
- Refinement: Keep provider wiring in integrations/runtime; do not block package moves on provider choice.
- Embedding query: grok 4.6 multi lane implementation supervisor objective bundle protected path asref
- AST query: MultiSupervisorRunner TodoImplementationSupervisor generate_objective_todos

## ASREF-G101 Prove ASREF-G090 for Clear agent_supervisor package layout and monorepo root hygiene

- Status: active
- Parent: ASREF-G000
- Fib priority: 3000
- Track: agent-supervisor-refactor
- Priority: P0
- Bundle: asref/root
- Goal: Create concrete implementation, tests, docs, or interface descriptors proving `ASREF-G090`.
- Evidence: ASREF-G090
- Outputs: ipfs_accelerate_py/agent_supervisor, docs/architecture/agent_supervisor_module_refactor.objectives.md, docs/architecture/agent_supervisor_module_refactor.todo.md, docs/architecture/AGENT_SUPERVISOR_MODULE_REFACTOR_PLAN.md, pyproject.toml, setup.py, scripts, test/api
- Validation: python -m pytest test/api/test_agent_supervisor_todo_daemon_port.py test/api/test_agent_supervisor_control_conformance_v2.py -q --collect-only && rg -n "agent_supervisor\\.(objective_daemon|backlog_refinery|merge_resolver)\\b" pyproject.toml setup.py || true
- Refinement depth: 1
- Embedding query: ASREF-G090
- AST query: ASREF-G090
- Parallel lane: asref/root
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Gap task: Close the missing objective evidence `ASREF-G090` with a narrow, verifiable change.

## ASREF-G102 Prove ASREF-G100 for Clear agent_supervisor package layout and monorepo root hygiene

- Status: active
- Parent: ASREF-G000
- Fib priority: 3001
- Track: agent-supervisor-refactor
- Priority: P0
- Bundle: asref/root
- Goal: Create concrete implementation, tests, docs, or interface descriptors proving `ASREF-G100`.
- Evidence: ASREF-G100
- Outputs: ipfs_accelerate_py/agent_supervisor, docs/architecture/agent_supervisor_module_refactor.objectives.md, docs/architecture/agent_supervisor_module_refactor.todo.md, docs/architecture/AGENT_SUPERVISOR_MODULE_REFACTOR_PLAN.md, pyproject.toml, setup.py, scripts, test/api
- Validation: python -m pytest test/api/test_agent_supervisor_todo_daemon_port.py test/api/test_agent_supervisor_control_conformance_v2.py -q --collect-only && rg -n "agent_supervisor\\.(objective_daemon|backlog_refinery|merge_resolver)\\b" pyproject.toml setup.py || true
- Refinement depth: 1
- Embedding query: ASREF-G100
- AST query: ASREF-G100
- Parallel lane: asref/root
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Gap task: Close the missing objective evidence `ASREF-G100` with a narrow, verifiable change.
