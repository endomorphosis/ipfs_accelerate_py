"""Manual remaining SPAR/SAWM/ASEH/PCTDD/DOEP surfaces are overlay code, not extra-gate."""

from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.runtime.remaining_task_runtime import (
    bind_remaining_task,
    execute_remaining_task,
    remaining_task_analytical_candidate,
    remaining_task_overlay_root,
)


def test_remaining_sawm_and_doep_bind_overlay_control_surfaces() -> None:
    overlay = remaining_task_overlay_root()
    sawm = bind_remaining_task("SAWM-039")
    doep = bind_remaining_task("DOEP-125")
    assert sawm is not None
    assert doep is not None
    assert sawm.no_model_route == "SemanticWorldService@1"
    assert (overlay / "ipfs_accelerate_py/agent_supervisor/semantic_state/program_world_service.py").is_file()
    assert bind_remaining_task("PCTDD-006") is None
    assert bind_remaining_task("unrelated") is None


def test_remaining_task_execute_is_never_board_completion() -> None:
    for task_id in (
        "SAWM-016",
        "SAWM-022",
        "SAWM-030",
        "SAWM-031",
        "SAWM-039",
        "SAWM-044",
        "DOEP-111",
        "DOEP-113",
        "DOEP-120",
        "DOEP-125",
        "SPAR-050",
        "ASEH-035",
    ):
        result = execute_remaining_task(task_id)
        assert result is not None, task_id
        assert result["completion_authority"] is False
        assert result["cas_completed"] is False
        assert result["admitted"] is False
        assert result["proposal_only"] is True


def test_missing_overlay_files_do_not_bind(tmp_path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert bind_remaining_task("SAWM-039", overlay=empty) is None
    assert remaining_task_analytical_candidate("SAWM-039", overlay=empty) is None
    assert execute_remaining_task("SAWM-039", overlay=empty) is None
