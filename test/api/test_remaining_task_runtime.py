"""Remaining SPAR/SAWM/ASEH/PCTDD/DOEP theory must bind extra-gate runtime.

Overlay remaining-task modules are the no-model route. Extra-gate nested
accelerate does not contain them. This binder never admits DuckDB completion.
"""

from __future__ import annotations

import os

from ipfs_accelerate_py.agent_supervisor.rescue.sealed_board_supervisor_launch import (
    prepend_remaining_task_overlay_pythonpath,
)
from ipfs_accelerate_py.agent_supervisor.runtime.remaining_task_runtime import (
    bind_remaining_task,
    execute_remaining_task,
    remaining_task_analytical_candidate,
    remaining_task_overlay_root,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_disposition import (
    ImplementationDisposition,
    ImplementationForestRoots,
    implementation_disposition_cid,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.pre_implementation_provider_gate import (
    evaluate_provider_gate,
)


def _forest() -> ImplementationForestRoots:
    cid = implementation_disposition_cid
    return ImplementationForestRoots(
        repository_id="repository:sha256:remaining-task",
        repository_forest_cid=cid({"forest": "remaining"}),
        git_tree_id=cid({"tree": "remaining"}),
        policy_root=cid({"policy": "remaining"}),
        dirty_overlay_cid=cid({"overlay": "remaining"}),
        capability_catalog_root=cid({"capabilities": "remaining"}),
        configuration_root=cid({"config": "remaining"}),
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


def test_remaining_task_execute_is_never_extra_gate_completion() -> None:
    for task_id in ("SAWM-039", "SAWM-016", "SAWM-044", "DOEP-111", "DOEP-125", "SPAR-050", "ASEH-035"):
        result = execute_remaining_task(task_id)
        assert result is not None, task_id
        assert result["completion_authority"] is False
        assert result["cas_completed"] is False
        assert result["admitted"] is False
        assert result["proposal_only"] is True


def test_remaining_task_unique_mapping_blocks_provider(monkeypatch) -> None:
    monkeypatch.delenv("IPFS_ACCELERATE_SUPERVISOR_OVERLAY", raising=False)
    decision = evaluate_provider_gate(
        task_cid=implementation_disposition_cid({"task": "SAWM-039"}),
        task_alias="SAWM-039",
        forest_roots=_forest(),
    )
    assert decision.disposition is ImplementationDisposition.CLOSED_DETERMINISTIC
    assert decision.skip_provider is True
    assert decision.provider_authorized is False
    assert decision.reason_code == "analytical_unique_mapping"


def test_non_remaining_task_does_not_mint_unique_mapping() -> None:
    decision = evaluate_provider_gate(
        task_cid=implementation_disposition_cid({"task": "other"}),
        task_alias="OTHER-001",
        forest_roots=_forest(),
    )
    assert decision.disposition is not ImplementationDisposition.CLOSED_DETERMINISTIC
    assert decision.reason_code == "no_analytical_close"


def test_pythonpath_prepend_puts_overlay_first_for_extra_gate_children(
    tmp_path, monkeypatch
) -> None:
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    nested = tmp_path / "nested"
    nested.mkdir()
    monkeypatch.setenv("PYTHONPATH", str(nested))
    updated = prepend_remaining_task_overlay_pythonpath(str(overlay))
    assert updated.split(os.pathsep)[0] == str(overlay.resolve())
    assert str(nested) in updated.split(os.pathsep)


def test_missing_overlay_files_do_not_bind(tmp_path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert bind_remaining_task("SAWM-039", overlay=empty) is None
    assert remaining_task_analytical_candidate("SAWM-039", overlay=empty) is None
    assert execute_remaining_task("SAWM-039", overlay=empty) is None
