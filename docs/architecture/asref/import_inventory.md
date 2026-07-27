# Agent Supervisor Import Inventory (ASREF-G010)

Tracked copy: `docs/architecture/asref/`. Runtime discovery mirror: `data/agent_supervisor/discovery/asref/` (often gitignored).

Generated: 2026-07-27T07:16:07.547559+00:00
Modules mapped: 187
Dynamic import sites: 4

## Package counts

- `proof`: 36
- `todo_daemon.loop`: 26
- `planning`: 15
- `objectives`: 14
- `analysis`: 13
- `self_improvement`: 10
- `merge`: 9
- `task_sources`: 9
- `runtime`: 8
- `context`: 7
- `control`: 6
- `rescue`: 6
- `core`: 5
- `validation`: 5
- `integrations`: 4
- `prompt`: 4
- `todo_daemon.git`: 4
- `todo_daemon.implementation`: 4
- `(root)`: 1
- `todo_daemon`: 1

## Console entry points

- `ipfs-accelerate-agent-objective-daemon` → `ipfs_accelerate_py.agent_supervisor.objectives.objective_daemon:main` (pyproject.toml)
- `ipfs-accelerate-agent-backlog-refinery` → `ipfs_accelerate_py.agent_supervisor.objectives.backlog_refinery:main` (pyproject.toml)
- `ipfs-accelerate-agent-bundle-supervisor` → `ipfs_accelerate_py.agent_supervisor.bundle_supervisor:main` (pyproject.toml)
- `ipfs-accelerate-agent-artifact-query` → `ipfs_accelerate_py.agent_supervisor.artifact_store:main` (pyproject.toml)
- `ipfs-accelerate-agent-implementation-daemon` → `ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon:main` (pyproject.toml)
- `ipfs-accelerate-agent-implementation-supervisor` → `ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor:main` (pyproject.toml)
- `ipfs-accelerate-agent-merge-resolver` → `ipfs_accelerate_py.agent_supervisor.merge.merge_resolver:main` (pyproject.toml)
- `ipfs-accelerate-agent-llm-merge-resolver-fallback` → `ipfs_accelerate_py.agent_supervisor.llm_merge_resolver_fallback:main` (pyproject.toml)
- `ipfs-accelerate-agent-objective-daemon` → `ipfs_accelerate_py.agent_supervisor.objectives.objective_daemon:main` (setup.py)
- `ipfs-accelerate-agent-backlog-refinery` → `ipfs_accelerate_py.agent_supervisor.objectives.backlog_refinery:main` (setup.py)
- `ipfs-accelerate-agent-bundle-supervisor` → `ipfs_accelerate_py.agent_supervisor.bundle_supervisor:main` (setup.py)
- `ipfs-accelerate-agent-artifact-query` → `ipfs_accelerate_py.agent_supervisor.artifact_store:main` (setup.py)
- `ipfs-accelerate-agent-implementation-daemon` → `ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon:main` (setup.py)
- `ipfs-accelerate-agent-implementation-supervisor` → `ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor:main` (setup.py)
- `ipfs-accelerate-agent-merge-resolver` → `ipfs_accelerate_py.agent_supervisor.merge.merge_resolver:main` (setup.py)
- `ipfs-accelerate-agent-llm-merge-resolver-fallback` → `ipfs_accelerate_py.agent_supervisor.llm_merge_resolver_fallback:main` (setup.py)

## Dynamic import sites

- `test/api/test_agent_supervisor_v2_public_api.py:108` — `ipfs_accelerate_py.mcp_server.tools.agent_supervisor_tools.`
- `test/api/test_agent_supervisor_protocol_verification.py:266` — `ipfs_accelerate_py.agent_supervisor.formal_verification_contracts`
- `test/api/test_agent_supervisor_ir_registry.py:755` — `ipfs_accelerate_py.agent_supervisor.ir_registry`
- `test/api/test_agent_supervisor_prompt_cli.py:93` — `ipfs_accelerate_py.agent_supervisor.prompt_workflow`

## High fan-in modules (by distinct importing files)

- `formal_verification_contracts` → `proof` (40 files) bundle `asref/proof`
- `objective_graph` → `objectives` (23 files) bundle `asref/objectives`
- `goal_completion` → `objectives` (20 files) bundle `asref/objectives`
- `control_contracts` → `control` (16 files) bundle `asref/control`
- `prompt_workflow` → `prompt` (15 files) bundle `asref/prompt`
- `control_plane` → `control` (13 files) bundle `asref/control`
- `bundle_supervisor` → `objectives` (12 files) bundle `asref/objectives`
- `objective_tracker` → `objectives` (11 files) bundle `asref/objectives`
- `backlog_refinery` → `objectives` (10 files) bundle `asref/objectives`
- `resource_scheduler` → `runtime` (10 files) bundle `asref/runtime`
- `scan_receipts` → `objectives` (10 files) bundle `asref/objectives`
- `artifact_store` → `runtime` (9 files) bundle `asref/runtime`
- `formal_planning_contracts` → `planning` (9 files) bundle `asref/planning`
- `objective_daemon` → `objectives` (9 files) bundle `asref/objectives`
- `task_proposal_router` → `planning` (9 files) bundle `asref/planning`
- `analyzer_health` → `analysis` (8 files) bundle `asref/analysis`
- `code_evidence_graph` → `analysis` (8 files) bundle `asref/analysis`
- `code_proof_obligations` → `proof` (8 files) bundle `asref/proof`
- `context_contracts` → `context` (8 files) bundle `asref/context`
- `goal_coverage` → `objectives` (8 files) bundle `asref/objectives`
- `context_compiler` → `context` (7 files) bundle `asref/context`
- `decision_contracts` → `context` (7 files) bundle `asref/context`
- `formal_logic_vocabulary` → `proof` (7 files) bundle `asref/proof`
- `formal_verification_policy` → `proof` (7 files) bundle `asref/proof`
- `ir_registry` → `proof` (7 files) bundle `asref/proof`
- `lease_coordination` → `merge` (7 files) bundle `asref/merge`
- `merge_queue` → `merge` (7 files) bundle `asref/merge`
- `merge_resolver` → `merge` (7 files) bundle `asref/merge`
- `validation_commands` → `validation` (7 files) bundle `asref/validation`
- `validation_scheduler` → `validation` (7 files) bundle `asref/validation`
- `conflict_graph` → `core` (6 files) bundle `asref/core`
- `control_cli` → `control` (6 files) bundle `asref/control`
- `formal_plan_compiler` → `planning` (6 files) bundle `asref/planning`
- `plan_evaluator` → `planning` (6 files) bundle `asref/planning`
- `prover_matrix_registry` → `proof` (6 files) bundle `asref/proof`
- `semantic_dependency_graph` → `analysis` (6 files) bundle `asref/analysis`
- `event_log` → `runtime` (5 files) bundle `asref/runtime`
- `formal_verification_provider` → `proof` (5 files) bundle `asref/proof`
- `merge_train` → `merge` (5 files) bundle `asref/merge`
- `multi_prover_router` → `proof` (5 files) bundle `asref/proof`

## Full module map

| Current | Target package | Bundle | Imports |
|---|---|---|---:|
| `ipfs_accelerate_py/agent_supervisor/__init__.py` | root | `asref/public-api` | — |
| `adaptive_goal_refiner.py` | `objectives` | `asref/objectives` | 3 |
| `adaptive_planner.py` | `planning` | `asref/planning` | 3 |
| `analysis_ast_index.py` | `analysis` | `asref/analysis` | 1 |
| `analysis_cache.py` | `analysis` | `asref/analysis` | 4 |
| `analysis_consensus.py` | `analysis` | `asref/analysis` | 1 |
| `analysis_contracts.py` | `analysis` | `asref/analysis` | 1 |
| `analysis_operation_registry.py` | `analysis` | `asref/analysis` | 1 |
| `analysis_pipeline.py` | `analysis` | `asref/analysis` | 2 |
| `analysis_retrieval.py` | `analysis` | `asref/analysis` | 4 |
| `analysis_transport.py` | `analysis` | `asref/analysis` | 3 |
| `analyzer_health.py` | `analysis` | `asref/analysis` | 8 |
| `artifact_store.py` | `runtime` | `asref/runtime` | 9 |
| `audit_scanner.py` | `analysis` | `asref/analysis` | 3 |
| `authorization_logic.py` | `control` | `asref/control` | 2 |
| `backlog_refinery.py` | `objectives` | `asref/objectives` | 10 |
| `bundle_optimizer.py` | `objectives` | `asref/objectives` | 2 |
| `bundle_supervisor.py` | `objectives` | `asref/objectives` | 12 |
| `cache_coordinator.py` | `analysis` | `asref/analysis` | 2 |
| `checkout_lock.py` | `merge` | `asref/merge` | 3 |
| `code_evidence_graph.py` | `analysis` | `asref/analysis` | 8 |
| `code_proof_obligations.py` | `proof` | `asref/proof` | 8 |
| `codex_failure_policy.py` | `rescue` | `asref/rescue` | 1 |
| `conflict_graph.py` | `core` | `asref/core` | 6 |
| `context_compiler.py` | `context` | `asref/context` | 7 |
| `context_contracts.py` | `context` | `asref/context` | 8 |
| `control_cli.py` | `control` | `asref/control` | 6 |
| `control_contracts.py` | `control` | `asref/control` | 16 |
| `control_plane.py` | `control` | `asref/control` | 13 |
| `dataset_store.py` | `task_sources` | `asref/task-sources` | 4 |
| `decision_context.py` | `context` | `asref/context` | 3 |
| `decision_contracts.py` | `context` | `asref/context` | 7 |
| `decision_runtime.py` | `context` | `asref/context` | 2 |
| `decision_runtime_benchmark.py` | `context` | `asref/context` | 4 |
| `decision_runtime_rollout.py` | `context` | `asref/context` | 2 |
| `duckdb_state.py` | `task_sources` | `asref/task-sources` | 1 |
| `duckdb_task_source.py` | `task_sources` | `asref/task-sources` | 3 |
| `event_log.py` | `runtime` | `asref/runtime` | 5 |
| `execution_permit.py` | `control` | `asref/control` | 1 |
| `external_completion.py` | `core` | `asref/core` | 1 |
| `formal_counterexamples.py` | `proof` | `asref/proof` | 2 |
| `formal_logic_vocabulary.py` | `proof` | `asref/proof` | 7 |
| `formal_plan_compiler.py` | `planning` | `asref/planning` | 6 |
| `formal_plan_conformance.py` | `planning` | `asref/planning` | 3 |
| `formal_plan_context.py` | `planning` | `asref/planning` | 1 |
| `formal_plan_validator.py` | `planning` | `asref/planning` | 4 |
| `formal_planning_adversarial.py` | `planning` | `asref/planning` | 1 |
| `formal_planning_contracts.py` | `planning` | `asref/planning` | 9 |
| `formal_planning_metrics.py` | `planning` | `asref/planning` | 1 |
| `formal_planning_rollout.py` | `planning` | `asref/planning` | 1 |
| `formal_replanner.py` | `planning` | `asref/planning` | 4 |
| `formal_verification_cache.py` | `proof` | `asref/proof` | 3 |
| `formal_verification_capabilities.py` | `proof` | `asref/proof` | 4 |
| `formal_verification_contracts.py` | `proof` | `asref/proof` | 40 |
| `formal_verification_policy.py` | `proof` | `asref/proof` | 7 |
| `formal_verification_provider.py` | `proof` | `asref/proof` | 5 |
| `git_gc.py` | `merge` | `asref/merge` | 3 |
| `goal_completion.py` | `objectives` | `asref/objectives` | 20 |
| `goal_coverage.py` | `objectives` | `asref/objectives` | 8 |
| `goal_development_contracts.py` | `objectives` | `asref/objectives` | 4 |
| `goal_quality.py` | `objectives` | `asref/objectives` | 2 |
| `goal_refinement_verification.py` | `objectives` | `asref/objectives` | 4 |
| `hyperproperty_verification.py` | `proof` | `asref/proof` | 1 |
| `implementation_daemon_runner.py` | `todo_daemon.implementation` | `asref/todo-daemon` | 4 |
| `implementation_supervisor_runner.py` | `todo_daemon.implementation` | `asref/todo-daemon` | 4 |
| `intent_constraint_adapter.py` | `proof` | `asref/proof` | 3 |
| `interface_contract_codegen.py` | `proof` | `asref/proof` | 1 |
| `ipfs_datasets_analysis_provider.py` | `integrations` | `asref/integrations` | 2 |
| `ipfs_datasets_logic_provider.py` | `integrations` | `asref/integrations` | 2 |
| `ir_adapters.py` | `proof` | `asref/proof` | 4 |
| `ir_constraint_compiler.py` | `proof` | `asref/proof` | 3 |
| `ir_registry.py` | `proof` | `asref/proof` | 7 |
| `kernel_verification.py` | `proof` | `asref/proof` | 3 |
| `leanstral_goal_benchmark.py` | `proof` | `asref/proof` | 0 |
| `leanstral_goal_development.py` | `proof` | `asref/proof` | 1 |
| `leanstral_goal_lifecycle.py` | `proof` | `asref/proof` | 0 |
| `leanstral_proof_provider.py` | `proof` | `asref/proof` | 4 |
| `lease_coordination.py` | `merge` | `asref/merge` | 7 |
| `leased_lane.py` | `merge` | `asref/merge` | 2 |
| `legal_constraint_adapter.py` | `proof` | `asref/proof` | 2 |
| `lifecycle_orchestrator.py` | `control` | `asref/control` | 1 |
| `llm_merge_resolver_fallback.py` | `integrations` | `asref/integrations` | 1 |
| `logic_translation_validation.py` | `proof` | `asref/proof` | 1 |
| `markdown_task_source.py` | `task_sources` | `asref/task-sources` | 4 |
| `merge_checkpoint.py` | `merge` | `asref/merge` | 0 |
| `merge_conflict_repair.py` | `merge` | `asref/merge` | 0 |
| `merge_queue.py` | `merge` | `asref/merge` | 7 |
| `merge_resolver.py` | `merge` | `asref/merge` | 7 |
| `merge_train.py` | `merge` | `asref/merge` | 5 |
| `meta_spark_goose_runner.py` | `integrations` | `asref/integrations` | 1 |
| `multi_prover_resources.py` | `proof` | `asref/proof` | 1 |
| `multi_prover_router.py` | `proof` | `asref/proof` | 5 |
| `multi_supervisor_runner.py` | `runtime` | `asref/runtime` | 2 |
| `objective_daemon.py` | `objectives` | `asref/objectives` | 9 |
| `objective_graph.py` | `objectives` | `asref/objectives` | 23 |
| `objective_task_janitor.py` | `objectives` | `asref/objectives` | 3 |
| `objective_tracker.py` | `objectives` | `asref/objectives` | 11 |
| `persistent_task_queue.py` | `task_sources` | `asref/task-sources` | 1 |
| `plan_evaluator.py` | `planning` | `asref/planning` | 6 |
| `plan_failure_memory.py` | `planning` | `asref/planning` | 1 |
| `program_behavior.py` | `core` | `asref/core` | 1 |
| `prompt_directory_scanner.py` | `prompt` | `asref/prompt` | 1 |
| `prompt_goal_planner.py` | `prompt` | `asref/prompt` | 3 |
| `prompt_plan_admission.py` | `prompt` | `asref/prompt` | 1 |
| `prompt_workflow.py` | `prompt` | `asref/prompt` | 15 |
| `proof_attestation.py` | `proof` | `asref/proof` | 4 |
| `proof_carrying_planner.py` | `planning` | `asref/planning` | 1 |
| `proof_context.py` | `proof` | `asref/proof` | 5 |
| `proof_directed_retrieval.py` | `proof` | `asref/proof` | 3 |
| `proof_fallbacks.py` | `proof` | `asref/proof` | 1 |
| `proof_metrics.py` | `proof` | `asref/proof` | 2 |
| `proof_obligation_templates.py` | `proof` | `asref/proof` | 1 |
| `proof_scheduler.py` | `proof` | `asref/proof` | 3 |
| `proof_scope_index.py` | `proof` | `asref/proof` | 4 |
| `proposal_validation.py` | `validation` | `asref/validation` | 4 |
| `protocol_verification.py` | `proof` | `asref/proof` | 1 |
| `prover_conformance.py` | `proof` | `asref/proof` | 2 |
| `prover_evidence_store.py` | `proof` | `asref/proof` | 1 |
| `prover_matrix_registry.py` | `proof` | `asref/proof` | 6 |
| `provider_batch_scheduler.py` | `runtime` | `asref/runtime` | 5 |
| `recovery_diagnostics.py` | `rescue` | `asref/rescue` | 3 |
| `rescue_orchestrator.py` | `rescue` | `asref/rescue` | 1 |
| `rescue_planner.py` | `rescue` | `asref/rescue` | 3 |
| `resource_scheduler.py` | `runtime` | `asref/runtime` | 10 |
| `runtime_cas.py` | `runtime` | `asref/runtime` | 3 |
| `runtime_temporal_monitor.py` | `runtime` | `asref/runtime` | 2 |
| `scan_receipts.py` | `objectives` | `asref/objectives` | 10 |
| `scheduler_metrics.py` | `runtime` | `asref/runtime` | 4 |
| `scope_adjudication.py` | `validation` | `asref/validation` | 1 |
| `security_constraint_adapter.py` | `proof` | `asref/proof` | 3 |
| `self_improvement.py` | `self_improvement` | `asref/self-improvement` | 1 |
| `self_improvement_completion.py` | `self_improvement` | `asref/self-improvement` | 1 |
| `self_improvement_rollout.py` | `self_improvement` | `asref/self-improvement` | 1 |
| `self_improvement_v2.py` | `self_improvement` | `asref/self-improvement` | 4 |
| `self_improvement_v2_rollout.py` | `self_improvement` | `asref/self-improvement` | 1 |
| `semantic_dependency_graph.py` | `analysis` | `asref/analysis` | 6 |
| `submodule_degradation.py` | `core` | `asref/core` | 0 |
| `supervisor_efficiency_metrics.py` | `self_improvement` | `asref/self-improvement` | 3 |
| `supervisor_recovery.py` | `rescue` | `asref/rescue` | 3 |
| `supervisor_state_model.py` | `self_improvement` | `asref/self-improvement` | 1 |
| `supervisor_token_ledger.py` | `self_improvement` | `asref/self-improvement` | 1 |
| `supervisor_v2_benchmark.py` | `self_improvement` | `asref/self-improvement` | 3 |
| `supervisor_v2_contracts.py` | `self_improvement` | `asref/self-improvement` | 5 |
| `supervisor_watchdog.py` | `rescue` | `asref/rescue` | 3 |
| `task_identity.py` | `task_sources` | `asref/task-sources` | 3 |
| `task_proposal_router.py` | `planning` | `asref/planning` | 9 |
| `task_quality.py` | `planning` | `asref/planning` | 3 |
| `task_source.py` | `task_sources` | `asref/task-sources` | 2 |
| `taskboard_store.py` | `task_sources` | `asref/task-sources` | 4 |
| `todo_daemon/__main__.py` | `todo_daemon` | `asref/todo-daemon` | 0 |
| `todo_daemon/app.py` | `todo_daemon.loop` | `asref/todo-daemon` | 0 |
| `todo_daemon/artifacts.py` | `todo_daemon.loop` | `asref/todo-daemon` | 0 |
| `todo_daemon/auto_commit.py` | `todo_daemon.git` | `asref/todo-daemon` | 0 |
| `todo_daemon/cli.py` | `todo_daemon.loop` | `asref/todo-daemon` | 0 |
| `todo_daemon/context.py` | `todo_daemon.loop` | `asref/todo-daemon` | 0 |
| `todo_daemon/core.py` | `todo_daemon.loop` | `asref/todo-daemon` | 0 |
| `todo_daemon/deterministic_fallback.py` | `todo_daemon.loop` | `asref/todo-daemon` | 0 |
| `todo_daemon/diagnostics.py` | `todo_daemon.loop` | `asref/todo-daemon` | 0 |
| `todo_daemon/engine.py` | `todo_daemon.loop` | `asref/todo-daemon` | 0 |
| `todo_daemon/file_replacement.py` | `todo_daemon.loop` | `asref/todo-daemon` | 0 |
| `todo_daemon/git_utils.py` | `todo_daemon.git` | `asref/todo-daemon` | 0 |
| `todo_daemon/history.py` | `todo_daemon.git` | `asref/todo-daemon` | 0 |
| `todo_daemon/implementation_daemon.py` | `todo_daemon.implementation` | `asref/todo-daemon` | 0 |
| `todo_daemon/implementation_supervisor.py` | `todo_daemon.implementation` | `asref/todo-daemon` | 0 |
| `todo_daemon/legal_parser.py` | `todo_daemon.loop` | `asref/todo-daemon` | 0 |
| `todo_daemon/legal_parser_daemon.py` | `todo_daemon.loop` | `asref/todo-daemon` | 0 |
| `todo_daemon/lifecycle_wrapper.py` | `todo_daemon.loop` | `asref/todo-daemon` | 0 |
| `todo_daemon/llm.py` | `todo_daemon.loop` | `asref/todo-daemon` | 0 |
| `todo_daemon/llm_defaults.py` | `todo_daemon.loop` | `asref/todo-daemon` | 0 |
| `todo_daemon/logic_port.py` | `todo_daemon.loop` | `asref/todo-daemon` | 0 |
| `todo_daemon/plans.py` | `todo_daemon.loop` | `asref/todo-daemon` | 0 |
| `todo_daemon/registry.py` | `todo_daemon.loop` | `asref/todo-daemon` | 0 |
| `todo_daemon/runner.py` | `todo_daemon.loop` | `asref/todo-daemon` | 0 |
| `todo_daemon/specs.py` | `todo_daemon.loop` | `asref/todo-daemon` | 0 |
| `todo_daemon/status.py` | `todo_daemon.loop` | `asref/todo-daemon` | 0 |
| `todo_daemon/supervisor.py` | `todo_daemon.loop` | `asref/todo-daemon` | 0 |
| `todo_daemon/supervisor_loop.py` | `todo_daemon.loop` | `asref/todo-daemon` | 0 |
| `todo_daemon/supervisor_runtime.py` | `todo_daemon.loop` | `asref/todo-daemon` | 0 |
| `todo_daemon/task_board.py` | `todo_daemon.loop` | `asref/todo-daemon` | 0 |
| `todo_daemon/typescript.py` | `todo_daemon.loop` | `asref/todo-daemon` | 0 |
| `todo_daemon/worktrees.py` | `todo_daemon.git` | `asref/todo-daemon` | 0 |
| `todo_daemon/wrapper.py` | `todo_daemon.loop` | `asref/todo-daemon` | 0 |
| `todo_vector_index.py` | `task_sources` | `asref/task-sources` | 4 |
| `validation_commands.py` | `validation` | `asref/validation` | 7 |
| `validation_runtime.py` | `validation` | `asref/validation` | 3 |
| `validation_scheduler.py` | `validation` | `asref/validation` | 7 |
| `wrapper_utils.py` | `core` | `asref/core` | 3 |
