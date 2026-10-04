"""Supervisor consumer of datasets-owned source/model training declarations."""
from ipfs_accelerate_py.agent_supervisor.runtime.security_code_training_profile import (
    build_security_code_learning_campaign, compile_security_code_learning_plan,
)
from tests.unit.logic.formalization.autoencoder.test_security_code_training_profile import profile


def test_native_campaign_preserves_parallel_dependencies_and_unresolved_admission(profile):
    campaign = build_security_code_learning_campaign(profile=profile, repository_tree_id="authored:tree")
    tasks = {task.task_id: task for task in campaign.tasks}
    projections = [task for task in campaign.tasks if task.task_id.startswith("SEC-PROJECT-")]
    assert len(projections) == 7 and len(tasks) == 15
    assert tasks["SEC-SOURCE-CONTEXT"].depends_on == ("SEC-SPLITS",)
    assert tasks["SEC-FORMALIZE"].depends_on == ("SEC-SOURCE-CONTEXT",)
    assert all(task.depends_on == ("SEC-FORMALIZE",) for task in projections if task.task_id != "SEC-PROJECT-CONTRACT")
    assert tasks["SEC-PROJECT-CONTRACT"].depends_on == ("SEC-FORMALIZE", "SEC-PROJECT-PROGRAM")
    assert "security-training/sec-project-program/result.json" in tasks["SEC-PROJECT-CONTRACT"].expected_inputs
    assert len({task.owned_paths for task in projections}) == 7
    assert set(tasks["SEC-FIT"].depends_on) == {"SEC-INITIALIZER", *(task.task_id for task in projections)}
    assert tasks["SEC-EVALUATE"].depends_on == ("SEC-FIT",)
    assert tasks["SEC-RELEASE"].depends_on == ("SEC-EVALUATE",)
    assert campaign.lease_eligible_task_ids == ()
    assert all(not task.is_schedulable and task.status.value == "todo" for task in tasks.values())
    assert all(task.metadata["stage_executor_configured"] is False for task in tasks.values())
    assert all("SystemExit" in task.validation and "pytest" not in task.validation for task in tasks.values())
    assert all(revision.unresolved_dependency_outputs for revision in campaign.task_revisions)
    assert "RESULT(SOURCE-ADMISSION)" in campaign.project_dependencies().unresolved_result_ids


def test_learning_plan_uses_actual_formal_compiler_without_running_tasks(profile):
    planned = compile_security_code_learning_plan(profile=profile, repository_tree_id="authored:tree")
    assert planned["compilation"]["status"] == "compiled"
    assert planned["compilation"]["plan"]
    assert planned["lease_eligible_task_ids"] == [] and not planned["execution_started"]
    assert planned["provider_calls"] == planned["training_steps"] == 0
    assert not planned["proof_authority"] and not planned["publication_authority"]
