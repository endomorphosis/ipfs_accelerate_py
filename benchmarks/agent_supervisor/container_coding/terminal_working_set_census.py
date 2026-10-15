"""Observe an admitted declared plan without planning or coding provider calls.

The signed declaration supplies the task structure. Recompiling it verifies
provenance, not instruction interpretation or proof of correct implementation.
The normal owner, source, world and detached-workspace checks remain shared
with the existing model-planned metadata census.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from .terminal_live_metadata_census import _identity, _prepare_terminal_census

SCHEMA = "terminal-working-set-census-context@1"


def _verify_declared_planning(*, prepared, planning, verified, task, retained_receipt):
    from ipfs_accelerate_py.agent_supervisor.runtime.declared_contract_planning import (
        PLANNING_MODE, build_declared_contract_plan,
    )

    if (prepared.get("planning_mode") != PLANNING_MODE
            or prepared.get("planning_strategy") != "declared_contract"
            or planning.get("schema") != "terminal-indexed-planner-result@1"
            or planning.get("planning_mode") != PLANNING_MODE
            or planning.get("planning_strategy") != "declared_contract"
            or planning.get("qualified") is not True
            or type(planning.get("provider_calls")) is not int or planning["provider_calls"] != 0
            or planning.get("provider_receipt", "missing") is not None
            or planning.get("execution_receipt", "missing") is not None
            or planning.get("model_request_sha256", "missing") is not None
            or planning.get("provider_observation") != {}
            or planning.get("task_cids") != [task["task_cid"]]
            or type(planning.get("goals")) is not int or planning["goals"] != len(verified["graph"].goals)
            or type(planning.get("tasks")) is not int or planning["tasks"] != 1
            or any(planning.get(key) != prepared[key] for key in ("provider", "model", "reasoning_effort"))
            or any(planning.get(key) is not False for key in (
                "instruction_interpreted", "source_semantics_verified", "semantic_alignment_verified",
                "proof_authority", "execution_authority", "completion_authority", "isolation_verified_here"))):
        raise ValueError("persisted native declared planning provenance differs")
    initial = planning.get("initial_indexed_context")
    if (type(initial) is not dict or any(initial.get(key) is not False for key in (
            "supplied_to_router", "used_for_task_selection", "execution_authority", "completion_authority"))):
        raise ValueError("declared planner initial context must remain unsupplied to a router")
    for stage in ("intent_preplanning", "source_unit_preplanning"):
        if planning.get(stage) != {
                "status": "not_used_for_declared_contract_planning", "supplied_to_router": False,
                "before_goal_decomposition": True, "source_semantics_verified": False,
                "execution_authority": False, "completion_authority": False}:
            raise ValueError("declared planner preplanning provenance differs")
    expected = build_declared_contract_plan(manifest=prepared["manifest"])
    receipt = expected["receipt"]
    # Canonical JSON equality also distinguishes false from integer zero.
    if (expected["graph"] != verified["graph"]
            or _identity(planning.get("declared_contract_planning")) != _identity(receipt)
            or _identity(retained_receipt) != _identity(receipt)):
        raise ValueError("declared planning receipt or admitted graph differs from signed declaration")


@contextmanager
def prepare_terminal_working_set_census(*, state: Path, include_doctor_context: bool = True,
        require_source384: bool = True, worktree_root: Path | None = None,
        worker_writable: bool = False, worker_group_gid: int | None = None,
        coding_reply_mode: str = "ordinary-completion@1"):
    """Yield provider-free census inputs for an existing declared-plan owner.

    The caller supplies the normal isolated router/source broker. The census
    itself admits no new plan, changes no source and invokes no model. Optional
    lexical preparation must be selected explicitly with require_source384=False.
    """
    from .terminal_coding_reply_policy import validate_coding_reply_mode
    validate_coding_reply_mode(coding_reply_mode)
    with _prepare_terminal_census(state=state, include_doctor_context=include_doctor_context,
            require_source384=require_source384, worktree_root=worktree_root,
            worker_writable=worker_writable, worker_group_gid=worker_group_gid,
            semantic_metadata_view="legacy", planning_mode="declared-contract@1") as context:
        context["runner_kwargs"].update(census_only=False, working_set_census_only=True,
            semantic_working_set="capsule-references@1", coding_reply_mode=coding_reply_mode)
        yield context
