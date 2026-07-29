# ASREF dual-layout cutover receipt

Modules cut over: 36

- `authorization_logic` → `control.authorization_logic`
- `backlog_refinery` → `objectives.backlog_refinery`
- `checkout_lock` → `merge.checkout_lock`
- `codex_failure_policy` → `rescue.codex_failure_policy`
- `conflict_graph` → `core.conflict_graph`
- `control_cli` → `control.control_cli`
- `control_contracts` → `control.control_contracts`
- `control_plane` → `control.control_plane`
- `dataset_store` → `task_sources.dataset_store`
- `duckdb_state` → `task_sources.duckdb_state`
- `duckdb_task_source` → `task_sources.duckdb_task_source`
- `execution_permit` → `control.execution_permit`
- `external_completion` → `core.external_completion`
- `formal_planning_metrics` → `planning.formal_planning_metrics`
- `formal_planning_rollout` → `planning.formal_planning_rollout`
- `git_gc` → `merge.git_gc`
- `lifecycle_orchestrator` → `control.lifecycle_orchestrator`
- `markdown_task_source` → `task_sources.markdown_task_source`
- `merge_checkpoint` → `merge.merge_checkpoint`
- `merge_conflict_repair` → `merge.merge_conflict_repair`
- `merge_resolver` → `merge.merge_resolver`
- `multi_supervisor_runner` → `runtime.multi_supervisor_runner`
- `objective_daemon` → `objectives.objective_daemon`
- `objective_graph` → `objectives.objective_graph`
- `persistent_task_queue` → `task_sources.persistent_task_queue`
- `plan_failure_memory` → `planning.plan_failure_memory`
- `program_behavior` → `core.program_behavior`
- `proposal_validation` → `validation.proposal_validation`
- `rescue_orchestrator` → `rescue.rescue_orchestrator`
- `self_improvement_completion` → `self_improvement.self_improvement_completion`
- `submodule_degradation` → `core.submodule_degradation`
- `task_identity` → `task_sources.task_identity`
- `task_source` → `task_sources.task_source`
- `taskboard_store` → `task_sources.taskboard_store`
- `todo_vector_index` → `task_sources.todo_vector_index`
- `wrapper_utils` → `core.wrapper_utils`

Files rewritten: 154
Flats deleted: 36
