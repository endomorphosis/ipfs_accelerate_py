from __future__ import annotations

import pytest
from ipfs_accelerate_py.agent_supervisor.autonomy.contracts import (
    MetaAction,
    RiskClass,
    SupervisorSkill,
)
from ipfs_accelerate_py.agent_supervisor.autonomy.supervisor_skills import (
    ALLOWLISTED_OPERATIONS,
    SUPERVISOR_SKILL_INTERFACE,
    SUPERVISOR_SKILL_REGISTRY_INTERFACE,
    SupervisorSkillError,
    SupervisorSkillRegistry,
)


def _skill(**overrides: object) -> SupervisorSkill:
    values: dict[str, object] = {
        "version": "skill-v1",
        "precondition_ids": ("tree-current",),
        "input_schema_id": "schema-static-analysis",
        "effect_class": "read_only_analysis",
        "steps": (MetaAction.RUN_LOCAL_STATIC_ANALYSIS, MetaAction.RUN_SELECTED_TEST),
        "postcondition_ids": ("question-resolved",),
        "validation_ids": ("validate-tree-binding",),
        "rollback_action_ids": ("release-reservation",),
        "fallback": MetaAction.QUARANTINE_TASK,
        "scope_paths": ("ipfs_accelerate_py/agent_supervisor/autonomy",),
        "scope_symbols": ("DecisionQuestion",),
        "risk_class": RiskClass.R1_READ_ONLY,
    }
    values.update(overrides)
    return SupervisorSkill(**values)


def test_interfaces_are_versioned() -> None:
    assert SUPERVISOR_SKILL_REGISTRY_INTERFACE == "SupervisorSkillRegistry@1"
    assert SUPERVISOR_SKILL_INTERFACE == "SupervisorSkill@1"
    assert SupervisorSkillRegistry.INTERFACE == SUPERVISOR_SKILL_REGISTRY_INTERFACE
    assert MetaAction.RUN_LOCAL_STATIC_ANALYSIS in ALLOWLISTED_OPERATIONS


def test_registry_admits_allowlisted_typed_operations() -> None:
    registry = SupervisorSkillRegistry()
    skill = registry.register(_skill())
    receipt = registry.execute(
        skill.skill_id,
        allowed_paths=("ipfs_accelerate_py/agent_supervisor/autonomy",),
        admitted_preconditions=("tree-current",),
    )
    assert receipt.status == "succeeded"
    assert receipt.applied_steps == skill.steps
    assert receipt.fallback is None


def test_registry_rejects_shell_and_forged_skills() -> None:
    registry = SupervisorSkillRegistry()
    with pytest.raises(SupervisorSkillError, match="forbidden"):
        registry.register(
            {
                **_skill().to_dict(),
                "shell_command": "bash -lc rm -rf /",
            }
        )
    with pytest.raises(SupervisorSkillError, match="forged"):
        registry.register({**_skill().to_dict(), "unsigned_hook": "eval"})
    payload = _skill().to_dict()
    payload["skill_id"] = "not-the-content-id"
    with pytest.raises(SupervisorSkillError, match="identity"):
        registry.register(payload)


def test_out_of_domain_and_missing_preconditions_use_fallback() -> None:
    registry = SupervisorSkillRegistry()
    skill = registry.register(_skill())
    missing = registry.execute(
        skill.skill_id,
        allowed_paths=("ipfs_accelerate_py/agent_supervisor/autonomy",),
        admitted_preconditions=(),
    )
    assert missing.status == "fallback"
    assert missing.fallback is MetaAction.QUARANTINE_TASK
    assert missing.applied_steps == ()
    scoped = registry.execute(
        skill.skill_id,
        allowed_paths=("docs/architecture",),
        admitted_preconditions=("tree-current",),
    )
    assert scoped.status == "fallback"
    assert scoped.applied_steps == ()


def test_failure_rolls_back_and_cancellation_stops() -> None:
    registry = SupervisorSkillRegistry()
    skill = registry.register(_skill())
    failed = registry.execute(
        skill.skill_id,
        allowed_paths=("ipfs_accelerate_py/agent_supervisor/autonomy",),
        admitted_preconditions=("tree-current",),
        fail_step=MetaAction.RUN_SELECTED_TEST,
    )
    assert failed.status == "rolled_back"
    assert failed.applied_steps == ()
    assert failed.rolled_back_steps
    cancelled = registry.execute(
        skill.skill_id,
        allowed_paths=("ipfs_accelerate_py/agent_supervisor/autonomy",),
        admitted_preconditions=("tree-current",),
        cancelled=True,
    )
    assert cancelled.status == "cancelled"
    assert cancelled.cancelled is True
    assert cancelled.applied_steps == ()
