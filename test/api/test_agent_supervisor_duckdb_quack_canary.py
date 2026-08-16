"""Tests for DuckDBQuackCanary@1 (DQP-035)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    duckdb_available,
)
from ipfs_accelerate_py.agent_supervisor.validation.duckdb_quack_canary import (
    DUCKDB_QUACK_CANARY_INTERFACE,
    LANE_COUNT,
    TASKS_PER_LANE,
    CanaryOutcome,
    DuckDBQuackCanary,
    DuckDBQuackCanaryError,
    run_duckdb_quack_canary,
)

pytestmark = pytest.mark.skipif(
    not duckdb_available(),
    reason="DuckDB is required for canary tests",
)


def test_interface_identity() -> None:
    assert DuckDBQuackCanary.INTERFACE == DUCKDB_QUACK_CANARY_INTERFACE
    assert DUCKDB_QUACK_CANARY_INTERFACE == "DuckDBQuackCanary@1"


def test_canary_passes_multi_lane_overlap_and_lineage(tmp_path: Path) -> None:
    report = run_duckdb_quack_canary(tmp_path / "canary")
    assert report.passed is True
    assert report.outcome is CanaryOutcome.PASSED
    assert report.lane_count == LANE_COUNT
    assert report.overlapping_lanes >= 2
    assert report.duplicate_claims == 0
    assert report.stale_writes == 0
    assert report.drained is True
    assert report.export_non_authoritative is True
    assert report.database_authority_intact_after_export_tamper is True
    # Primary tasks + one resume task per lane.
    assert len(report.lineages) == LANE_COUNT * TASKS_PER_LANE + LANE_COUNT
    assert all(item.to_dict()["complete"] for item in report.lineages)
    assert all(lane.restarted and lane.drained for lane in report.lanes)
    payload = report.to_dict()
    assert payload["interface"] == DUCKDB_QUACK_CANARY_INTERFACE
    assert payload["passed"] is True
    assert payload["task_id"] == "DQP-035"


def test_canary_registers_four_strict_lanes(tmp_path: Path) -> None:
    report = DuckDBQuackCanary(tmp_path / "lanes").run()
    assert len(report.lanes) == 4
    for index, lane in enumerate(report.lanes):
        assert lane.lane_index == index
        # Each lane executed its primary tasks plus resume.
        assert len(lane.tasks) >= TASKS_PER_LANE + 1
        assert len(lane.claims) == len(lane.effects)


def test_duplicate_claim_is_rejected(tmp_path: Path) -> None:
    canary = DuckDBQuackCanary(tmp_path / "dup")
    canary._bootstrap_database()
    task = canary._plan_tasks()[0]
    canary._claim(task, fence=1)
    with pytest.raises(DuckDBQuackCanaryError, match="duplicate claim"):
        # Force a second distinct claim id by bumping fence via private claims map.
        with canary._claim_lock:
            canary._claims[task.task_id] = "claim:foreign-owner"
        canary._claim(task, fence=2)


def test_export_tamper_does_not_change_database_authority(tmp_path: Path) -> None:
    report = run_duckdb_quack_canary(tmp_path / "export")
    assert report.database_authority_intact_after_export_tamper is True
    export = Path(report.export_path)
    assert export.exists()
    # Marker may be overwritten by tamper pass; authority flag still true.
    assert report.export_non_authoritative is True


def test_server_restart_resumes_lane_work(tmp_path: Path) -> None:
    report = run_duckdb_quack_canary(tmp_path / "restart")
    assert all(lane.restarted for lane in report.lanes)
    resume_ids = [item.task_id for item in report.lineages if item.task_id.endswith("-resume")]
    assert len(resume_ids) == LANE_COUNT


def test_processes_drain_cleanly(tmp_path: Path) -> None:
    report = run_duckdb_quack_canary(tmp_path / "drain")
    assert report.drained is True
    assert all(lane.drained for lane in report.lanes)
