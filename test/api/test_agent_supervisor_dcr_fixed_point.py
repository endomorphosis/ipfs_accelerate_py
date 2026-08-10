"""DCR-094: stable contract fixed point and legacy supersession.

Acceptance:
* Supported repairable findings are proved repaired.
* 13 ambiguous-anchor legacy rows are superseded/deduplicated.
* Unsupported/review-required residuals remain explicitly open.
* Two unchanged epochs emit zero tasks/edits and identical roots.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.evaluation.dcr_fixed_point import (
    CONTRACT_REPAIR_FIXED_POINT_INTERFACE,
    DEFAULT_BACKLOG_PATH,
    DEFAULT_FIXED_POINT_PATH,
    DCR_FIXED_POINT_EVIDENCE,
    DCR_TASK_ID,
    LEGACY_AMBIGUOUS_ANCHOR_COUNT,
    REPAIR_BACKLOG_PROJECTION_INTERFACE,
    ContractRepairFixedPoint,
    FindingStatus,
    FixedPointError,
    materialize_fixed_point,
    reach_contract_repair_fixed_point,
    supersede_legacy_repairs,
)


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parents[4], here.parents[3], Path.cwd()):
        if (candidate / "config" / "deterministic_contract_repair_services.json").is_file():
            return candidate
    return here.parents[4]


@pytest.fixture(scope="module")
def fixed_point() -> ContractRepairFixedPoint:
    return reach_contract_repair_fixed_point(repo_root=_repo_root())


def test_interfaces_and_symbols() -> None:
    assert CONTRACT_REPAIR_FIXED_POINT_INTERFACE == "ContractRepairFixedPoint@1"
    assert REPAIR_BACKLOG_PROJECTION_INTERFACE == "RepairBacklogProjection@1"
    assert ContractRepairFixedPoint.INTERFACE == CONTRACT_REPAIR_FIXED_POINT_INTERFACE
    assert DCR_TASK_ID == "DCR-094"
    assert DCR_FIXED_POINT_EVIDENCE == "dcr/contract-repair-fixed-point@1"
    assert LEGACY_AMBIGUOUS_ANCHOR_COUNT == 13
    assert callable(supersede_legacy_repairs)
    assert callable(reach_contract_repair_fixed_point)


def test_fixed_point_passes_with_identical_epochs(
    fixed_point: ContractRepairFixedPoint,
) -> None:
    assert fixed_point.passed is True
    assert fixed_point.preconditions_ok is True
    assert fixed_point.epoch_roots[0] == fixed_point.epoch_roots[1]
    assert fixed_point.epoch_task_counts == (0, 0)
    assert fixed_point.epoch_edit_counts == (0, 0)
    assert fixed_point.runtime_model_calls == 0
    assert "two_unchanged_epochs" in fixed_point.reason_codes


def test_thirteen_legacy_anchors_superseded(
    fixed_point: ContractRepairFixedPoint,
) -> None:
    superseded = [
        item
        for item in fixed_point.final_findings
        if item.status is FindingStatus.SUPERSEDED
        and item.family == "ambiguous_anchor"
    ]
    assert len(superseded) == LEGACY_AMBIGUOUS_ANCHOR_COUNT
    assert len(fixed_point.supersession_map) >= LEGACY_AMBIGUOUS_ANCHOR_COUNT
    # Historical evidence preserved (duplicates remain as DUPLICATE).
    assert any(
        item.status is FindingStatus.DUPLICATE for item in fixed_point.final_findings
    )


def test_supported_findings_repaired_residuals_open(
    fixed_point: ContractRepairFixedPoint,
) -> None:
    assert len(fixed_point.published_repairs) >= 3
    statuses = {item.status for item in fixed_point.unresolved_typed}
    assert FindingStatus.UNSUPPORTED in statuses
    assert FindingStatus.REVIEW_REQUIRED in statuses
    # No supported repairable left open.
    assert not any(
        item.status is FindingStatus.REPAIRABLE for item in fixed_point.final_findings
    )


def test_cannot_pass_with_divergent_epochs() -> None:
    fp = reach_contract_repair_fixed_point(repo_root=_repo_root())
    with pytest.raises(FixedPointError):
        ContractRepairFixedPoint(
            passed=True,
            preconditions_ok=True,
            initial_findings=fp.initial_findings,
            final_findings=fp.final_findings,
            supersession_map=fp.supersession_map,
            published_repairs=fp.published_repairs,
            unresolved_typed=fp.unresolved_typed,
            epoch_roots=("sha256:" + "a" * 64, "sha256:" + "b" * 64),
            epoch_task_counts=(0, 0),
            epoch_edit_counts=(0, 0),
            backlog=fp.backlog,
            reason_codes=("bad",),
        )


def test_materialize_fixed_point_and_backlog(tmp_path: Path) -> None:
    dest = tmp_path / "fixed-point.json"
    backlog = tmp_path / "ipfs_accelerate_contract_repairs.todo.md"
    payload = materialize_fixed_point(
        repo_root=_repo_root(),
        destination=dest,
        backlog_path=backlog,
    )
    assert dest.is_file()
    assert backlog.is_file()
    on_disk = json.loads(dest.read_text(encoding="utf-8"))
    assert on_disk["interface"] == CONTRACT_REPAIR_FIXED_POINT_INTERFACE
    assert on_disk["result"]["passed"] is True
    assert on_disk["runtime_model_calls"] == 0
    text = backlog.read_text(encoding="utf-8")
    assert "RepairBacklogProjection@1" in text
    assert "Open task count:" in text
    assert payload["result"]["passed"] is True


def test_default_paths() -> None:
    assert DEFAULT_FIXED_POINT_PATH.endswith("fixed-point.json")
    assert DEFAULT_BACKLOG_PATH.endswith("ipfs_accelerate_contract_repairs.todo.md")
