# ASREF-G070 Child Goals (module land batches)

Parent: ASREF-G070 / ASREF-011

Proposal-gate limits (`max_patch_bytes` ≈ 2 MiB, dual-copy cutover like
`core`/`control`/`task_sources`) require landing move-map modules in
batches. This attempt lands a first batch covering every package and the
unresolved evidence modules (`objective_graph`, `objective_daemon`).

Rules (same as parent):

- Dual-copy into package dirs; do **not** leave long-lived re-export stubs
  at flat paths when a later cutover removes flats.
- Rewrite same-package relative imports to stay `.owned`; outbound becomes `..other`.
- Stay inside package Outputs (no repo-wide caller rewrite here).
- Do **not** move `todo_daemon` (ASREF-G080).
- Update `pyproject.toml` / `setup.py` entry points only when the
  corresponding modules are landed **and** the task Outputs allow those paths.

## Landed this batch (ASREF-011 attempt)

- `merge/checkout_lock.py` (5994 bytes)
- `merge/git_gc.py` (17130 bytes)
- `merge/merge_checkpoint.py` (6892 bytes)
- `merge/merge_conflict_repair.py` (23629 bytes)
- `merge/merge_resolver.py` (55982 bytes)
- `objectives/backlog_refinery.py` (321465 bytes)
- `objectives/objective_daemon.py` (189864 bytes)
- `objectives/objective_graph.py` (428434 bytes)
- `planning/formal_planning_metrics.py` (33405 bytes)
- `planning/formal_planning_rollout.py` (34769 bytes)
- `planning/plan_failure_memory.py` (30852 bytes)
- `rescue/codex_failure_policy.py` (7374 bytes)
- `rescue/rescue_orchestrator.py` (84029 bytes)
- `runtime/multi_supervisor_runner.py` (57702 bytes)
- `self_improvement/self_improvement_completion.py` (16247 bytes)
- `validation/proposal_validation.py` (178826 bytes)

## Deferred per package (follow-on child tasks)

### asref/objectives

- Bundle: `asref/objectives`
- Outputs: `ipfs_accelerate_py/agent_supervisor/objectives`
- Already landed: backlog_refinery, objective_daemon, objective_graph
- Remaining modules:
  - `adaptive_goal_refiner.py`
  - `bundle_optimizer.py`
  - `bundle_supervisor.py`
  - `goal_completion.py`
  - `goal_coverage.py`
  - `goal_development_contracts.py`
  - `goal_quality.py`
  - `goal_refinement_verification.py`
  - `objective_task_janitor.py`
  - `objective_tracker.py`
  - `scan_receipts.py`
- Validation: `python -m pytest test/api/test_agent_supervisor_objective_graph.py test/api/test_agent_supervisor_proposal_validation.py test/api/test_agent_supervisor_programmatic_recovery.py -q`

### asref/planning

- Bundle: `asref/planning`
- Outputs: `ipfs_accelerate_py/agent_supervisor/planning`
- Already landed: formal_planning_metrics, formal_planning_rollout, plan_failure_memory
- Remaining modules:
  - `adaptive_planner.py`
  - `formal_plan_compiler.py`
  - `formal_plan_conformance.py`
  - `formal_plan_context.py`
  - `formal_plan_validator.py`
  - `formal_planning_adversarial.py`
  - `formal_planning_contracts.py`
  - `formal_replanner.py`
  - `plan_evaluator.py`
  - `proof_carrying_planner.py`
  - `task_proposal_router.py`
  - `task_quality.py`
- Validation: `python -m pytest test/api/test_agent_supervisor_objective_graph.py test/api/test_agent_supervisor_proposal_validation.py test/api/test_agent_supervisor_programmatic_recovery.py -q`

### asref/validation

- Bundle: `asref/validation`
- Outputs: `ipfs_accelerate_py/agent_supervisor/validation`
- Already landed: proposal_validation
- Remaining modules:
  - `scope_adjudication.py`
  - `validation_commands.py`
  - `validation_runtime.py`
  - `validation_scheduler.py`
- Validation: `python -m pytest test/api/test_agent_supervisor_objective_graph.py test/api/test_agent_supervisor_proposal_validation.py test/api/test_agent_supervisor_programmatic_recovery.py -q`

### asref/merge

- Bundle: `asref/merge`
- Outputs: `ipfs_accelerate_py/agent_supervisor/merge`
- Already landed: checkout_lock, git_gc, merge_checkpoint, merge_conflict_repair, merge_resolver
- Remaining modules:
  - `lease_coordination.py`
  - `leased_lane.py`
  - `merge_queue.py`
  - `merge_train.py`
- Validation: `python -m pytest test/api/test_agent_supervisor_objective_graph.py test/api/test_agent_supervisor_proposal_validation.py test/api/test_agent_supervisor_programmatic_recovery.py -q`

### asref/rescue

- Bundle: `asref/rescue`
- Outputs: `ipfs_accelerate_py/agent_supervisor/rescue`
- Already landed: codex_failure_policy, rescue_orchestrator
- Remaining modules:
  - `recovery_diagnostics.py`
  - `rescue_planner.py`
  - `supervisor_recovery.py`
  - `supervisor_watchdog.py`
- Validation: `python -m pytest test/api/test_agent_supervisor_objective_graph.py test/api/test_agent_supervisor_proposal_validation.py test/api/test_agent_supervisor_programmatic_recovery.py -q`

### asref/runtime

- Bundle: `asref/runtime`
- Outputs: `ipfs_accelerate_py/agent_supervisor/runtime`
- Already landed: multi_supervisor_runner
- Remaining modules:
  - `artifact_store.py`
  - `event_log.py`
  - `provider_batch_scheduler.py`
  - `resource_scheduler.py`
  - `runtime_cas.py`
  - `runtime_temporal_monitor.py`
  - `scheduler_metrics.py`
- Validation: `python -m pytest test/api/test_agent_supervisor_objective_graph.py test/api/test_agent_supervisor_proposal_validation.py test/api/test_agent_supervisor_programmatic_recovery.py -q`

### asref/self-improvement

- Bundle: `asref/self-improvement`
- Outputs: `ipfs_accelerate_py/agent_supervisor/self_improvement`
- Already landed: self_improvement_completion
- Remaining modules:
  - `self_improvement.py`
  - `self_improvement_rollout.py`
  - `self_improvement_v2.py`
  - `self_improvement_v2_rollout.py`
  - `supervisor_efficiency_metrics.py`
  - `supervisor_state_model.py`
  - `supervisor_token_ledger.py`
  - `supervisor_v2_benchmark.py`
  - `supervisor_v2_contracts.py`
- Validation: `python -m pytest test/api/test_agent_supervisor_objective_graph.py test/api/test_agent_supervisor_proposal_validation.py test/api/test_agent_supervisor_programmatic_recovery.py -q`

## Entry-point retarget (when Outputs permit)

After modules land under packages, retarget console scripts:

- `ipfs_accelerate_py.agent_supervisor.objectives.objective_daemon:main`
- `ipfs_accelerate_py.agent_supervisor.objectives.backlog_refinery:main`
- `ipfs_accelerate_py.agent_supervisor.objectives.bundle_supervisor:main`
- `ipfs_accelerate_py.agent_supervisor.merge.merge_resolver:main`
- `ipfs_accelerate_py.agent_supervisor.runtime.artifact_store:main`

Flat dual-copies remain until ASREF-G090 cutover removes them and rewrites callers.

