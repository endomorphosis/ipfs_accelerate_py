"""SAWM-020 procedure compilation and trajectory normalization."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.mcp_server.mcplusplus.kubo_cid import cid_for_bytes

from ipfs_accelerate_py.agent_supervisor.self_improvement.program_world_procedure_bridge import (
    ProcedureHole,
    ProcedurePromotionProposal,
    ProgramWorldProcedureBridge,
    ProgramWorldProcedureError,
    compile_program_world_procedure_candidate,
    match_program_world_procedure,
)
from ipfs_accelerate_py.agent_supervisor.self_improvement.program_world_trajectory import (
    ProgramWorldTrajectoryError,
    normalize_accepted_trajectory,
)


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


def _accepted_trace(*, task: str = "task", env: str = "env", suffix: str = "a") -> dict[str, object]:
    return {
        "accepted": True,
        "task_cid": _cid(task),
        "environment_cid": _cid(env),
        "policy_cid": _cid("policy"),
        "steps": [
            {
                "action_cid": _cid("act-1"),
                "state_cid": _cid("state-0"),
                "effect_cid": _cid("state-1"),
            },
            {
                "action_cid": _cid("act-2"),
                "state_cid": _cid("state-1"),
                "effect_cid": _cid(f"state-2-{suffix}"),
            },
        ],
    }


def test_unaccepted_trajectory_is_rejected() -> None:
    record = _accepted_trace()
    record["accepted"] = False
    with pytest.raises(ProgramWorldTrajectoryError, match="unaccepted"):
        normalize_accepted_trajectory(record)


def test_observational_fields_cannot_enter_identity() -> None:
    record = _accepted_trace()
    record["steps"][0]["ann_score"] = 0.9  # type: ignore[index]
    with pytest.raises(ProgramWorldTrajectoryError, match="observational"):
        normalize_accepted_trajectory(record)


def test_anti_unification_requires_repeated_same_family() -> None:
    first = _accepted_trace(suffix="shared")
    second = _accepted_trace(task="task-2", suffix="shared")
    candidate = compile_program_world_procedure_candidate(
        [first, second],
        validation_cid=_cid("validation"),
        rollback_cid=_cid("rollback"),
        held_out=[_accepted_trace(task="held", suffix="shared")],
        adversarial=[_accepted_trace(task="adv", suffix="shared")],
    )
    assert candidate.family_cid == normalize_accepted_trajectory(first).family_cid
    assert candidate.validation_cid == _cid("validation")
    assert candidate.rollback_cid == _cid("rollback")


def test_unique_trajectory_stays_episodic() -> None:
    with pytest.raises(ProgramWorldProcedureError, match="episodic"):
        compile_program_world_procedure_candidate(
            [_accepted_trace()],
            validation_cid=_cid("validation"),
            rollback_cid=_cid("rollback"),
        )


def test_stale_procedure_stops() -> None:
    first = _accepted_trace(suffix="shared")
    second = _accepted_trace(task="task-2", suffix="shared")
    bridge = ProgramWorldProcedureBridge(current_generation=2)
    bridge.compile_program_world_procedure_candidate(
        [first, second],
        validation_cid=_cid("validation"),
        rollback_cid=_cid("rollback"),
        generation=1,
    )
    with pytest.raises(ProgramWorldProcedureError, match="stale"):
        match_program_world_procedure(first, bridge=bridge, current_generation=2)


def test_validation_holes_retain_exact_validation() -> None:
    first = _accepted_trace(suffix="shared")
    second = _accepted_trace(task="task-2", suffix="shared")
    with pytest.raises(ProgramWorldProcedureError, match="validation holes"):
        compile_program_world_procedure_candidate(
            [first, second],
            validation_cid=_cid("validation"),
            rollback_cid=_cid("rollback"),
            holes=[{"kind": "validation", "validation_cid": _cid("other")}],
        )
    candidate = compile_program_world_procedure_candidate(
        [first, second],
        validation_cid=_cid("validation"),
        rollback_cid=_cid("rollback"),
        holes=[{"kind": "validation", "validation_cid": _cid("validation")}],
    )
    assert candidate.holes[0] == ProcedureHole(
        kind="validation", validation_cid=_cid("validation")
    )


def test_forbidden_hole_kinds_fail_closed() -> None:
    with pytest.raises(ProgramWorldProcedureError, match="forbidden hole"):
        ProcedureHole(kind="credential", validation_cid=_cid("validation"))


def test_promotion_is_nomination_only() -> None:
    first = _accepted_trace(suffix="shared")
    second = _accepted_trace(task="task-2", suffix="shared")
    bridge = ProgramWorldProcedureBridge()
    candidate = bridge.compile_program_world_procedure_candidate(
        [first, second],
        validation_cid=_cid("validation"),
        rollback_cid=_cid("rollback"),
    )
    proposal = bridge.request_promotion(candidate)
    assert isinstance(proposal, ProcedurePromotionProposal)
    assert proposal.admitted is False
    assert proposal.authority == "ProofCarryingProcedureCompiler"
    with pytest.raises(ProgramWorldProcedureError, match="self-admit"):
        ProcedurePromotionProposal(candidate_cid=candidate.candidate_cid, admitted=True)


def test_match_before_model_routes_repeated_family() -> None:
    first = _accepted_trace(suffix="shared")
    second = _accepted_trace(task="task-2", suffix="shared")
    bridge = ProgramWorldProcedureBridge(current_generation=1)
    compiled = bridge.compile_program_world_procedure_candidate(
        [first, second],
        validation_cid=_cid("validation"),
        rollback_cid=_cid("rollback"),
        generation=1,
    )
    matched = match_program_world_procedure(first, bridge=bridge)
    assert matched is not None
    assert matched.candidate_cid == compiled.candidate_cid
    assert match_program_world_procedure(
        _accepted_trace(suffix="other-family"), bridge=bridge
    ) is None
