"""Tests for the multi-daemon, multi-worktree DuckDB/Quack canary (DQP-035).

Acceptance:

* At least two lanes overlap
* No duplicate claim/effect or stale write
* Every change has complete lineage
* Server/worker restart resumes
* Final tasks/goals/daemons/worktrees/events/proofs agree
* Tampered/missing exports do not affect result
* All processes drain cleanly

Evidence subset: real processes, overlap, claim/fence, worktree, mutation,
validation, merge, restart, refill, export, drain, database queries.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    duckdb_available,
)
from ipfs_accelerate_py.agent_supervisor.validation.duckdb_quack_canary import (
    CANARY_VERSION,
    DEFAULT_LANE_COUNT,
    DUCKDB_QUACK_CANARY_INTERFACE,
    EVIDENCE,
    GOAL_ID,
    LANE_FILE_PATHS,
    REQUIRED_EVIDENCE_KEYS,
    STRICT_LANE_IDS,
    TASK_ID,
    CanaryVerdict,
    DuckDBQuackCanary,
    DuckDBQuackCanaryAcceptanceError,
    DuckDBQuackCanaryReport,
    assert_canary_passed,
    count_pairwise_overlaps,
    default_canary_population,
    hermetic_capability_report,
    intervals_overlap,
    run_duckdb_quack_canary,
)

pytestmark = pytest.mark.skipif(
    not duckdb_available(),
    reason="DuckDB is required for DuckDB/Quack canary hermetic tests",
)


# ---------------------------------------------------------------------------
# Unit / contract proofs
# ---------------------------------------------------------------------------


def test_interface_identities_and_defaults() -> None:
    assert DUCKDB_QUACK_CANARY_INTERFACE == "DuckDBQuackCanary@1"
    assert DuckDBQuackCanary.INTERFACE == DUCKDB_QUACK_CANARY_INTERFACE
    assert TASK_ID == "DQP-035"
    assert GOAL_ID == "DQP-G080"
    assert EVIDENCE == "dqp/duckdb-quack-canary@1"
    assert CANARY_VERSION == 1
    assert DEFAULT_LANE_COUNT == 4
    assert len(STRICT_LANE_IDS) == 4
    assert set(LANE_FILE_PATHS) == set(STRICT_LANE_IDS)
    # File-disjoint: every lane owns a unique path.
    assert len(set(LANE_FILE_PATHS.values())) == len(LANE_FILE_PATHS)
    report = hermetic_capability_report()
    assert report.status.value == "compatible"


def test_default_population_is_file_disjoint() -> None:
    population = default_canary_population(lane_count=4)
    tasks = population["tasks"]
    assert len(tasks) == 4
    paths = [tuple(task["allowed_paths"]) for task in tasks]
    assert len({paths[i][0] for i in range(4)}) == 4
    assert all(task["status"] == "ready" for task in tasks)
    assert population["objectives"][0]["goal_alias"] == GOAL_ID


def test_interval_overlap_helpers() -> None:
    assert intervals_overlap(0, 10, 5, 15) is True
    assert intervals_overlap(0, 5, 5, 10) is False
    assert intervals_overlap(0, 5, 6, 10) is False
    assert count_pairwise_overlaps([(0, 10), (5, 15), (20, 30)]) == 1
    assert count_pairwise_overlaps([(0, 10), (1, 2), (3, 4)]) == 3


def test_cold_construction_has_no_side_effects(tmp_path: Path) -> None:
    # Construction before open must not create stores when duckdb is present;
    # open() is the I/O boundary. We only instantiate the class fields.
    missing = tmp_path / "does-not-exist-yet"
    canary = DuckDBQuackCanary(missing)
    assert canary._closed is True
    assert not (missing / "tasks.duckdb").exists()


def test_assert_canary_passed_fails_closed() -> None:
    from ipfs_accelerate_py.agent_supervisor.validation.duckdb_quack_canary import (
        CanaryExportReceipt,
        CanaryFinalAgreement,
    )

    report = DuckDBQuackCanaryReport(
        verdict=CanaryVerdict.FAILED,
        workspace="/tmp/x",
        lane_count=4,
        lanes=(),
        overlap_pair_count=0,
        server_restarted=False,
        worker_restarted=False,
        refill_task_count=0,
        agreement=CanaryFinalAgreement(
            task_count=0,
            completed_task_count=0,
            goal_count=0,
            open_goal_count=0,
            daemon_session_count=0,
            worktree_count=0,
            terminal_worktree_count=0,
            event_count=0,
            mutation_count=0,
            accepted_mutation_count=0,
            merge_accepted_count=0,
            proof_count=0,
            complete_lineage_count=0,
            duplicate_claims=0,
            duplicate_effects=0,
            stale_writes=0,
            agreed=False,
        ),
        export_receipt=CanaryExportReceipt(
            export_path="",
            authority="export_only",
            event_count_before=0,
            event_count_after_tamper=0,
            event_count_after_delete=0,
            task_completed_before=0,
            task_completed_after=0,
            board_export_path="",
            board_tampered=False,
            board_deleted=False,
            unaffected=False,
        ),
        drained=False,
        control_files_used_as_authority=False,
        duckdb_available=True,
        failures=("lanes did not overlap",),
    )
    with pytest.raises(DuckDBQuackCanaryAcceptanceError):
        assert_canary_passed(report)


# ---------------------------------------------------------------------------
# Full multi-daemon canary
# ---------------------------------------------------------------------------


def test_full_multi_daemon_multi_worktree_canary(tmp_path: Path) -> None:
    report = run_duckdb_quack_canary(tmp_path / "canary-e2e", lane_count=4)

    assert isinstance(report, DuckDBQuackCanaryReport)
    assert report.INTERFACE == DUCKDB_QUACK_CANARY_INTERFACE
    assert report.task_id == TASK_ID
    assert report.goal_id == GOAL_ID
    assert report.duckdb_available is True
    assert report.control_files_used_as_authority is False
    assert report.mode == "hermetic"

    # Surface failures clearly on regression.
    if report.verdict is not CanaryVerdict.PASSED:
        pytest.fail(
            "canary failed: "
            + "; ".join(report.failures)
            + f" evidence={dict(report.evidence_subset)}"
        )

    assert_canary_passed(report)
    assert report.passed is True
    assert report.verdict is CanaryVerdict.PASSED

    # --- Acceptance predicates ---
    assert report.lane_count == 4
    assert len(report.lanes) == 4
    assert report.overlap_pair_count >= 1
    assert report.server_restarted is True
    assert report.worker_restarted is True
    assert report.refill_task_count == 1
    assert report.drained is True
    assert report.export_receipt.unaffected is True
    assert report.agreement.agreed is True

    # No duplicate claim/effect or stale write.
    claim_ids = [lane.claim_id for lane in report.lanes]
    assert len(claim_ids) == len(set(claim_ids))
    task_cids = [lane.task_cid for lane in report.lanes]
    assert len(task_cids) == len(set(task_cids))
    for lane in report.lanes:
        assert lane.provider_calls == 1, lane.to_dict()
        assert lane.effect_calls == 1, lane.to_dict()
        assert lane.lineage_count >= 1
        assert lane.mutation_status == "accepted"
        assert lane.merge_status in {"accepted", "settled"}
        assert lane.fencing_token > 0
        assert lane.evidence_digest.startswith("sha256:")

    # File-disjoint paths across lanes.
    paths = [lane.file_path for lane in report.lanes]
    assert len(set(paths)) == len(paths)

    # Restart lane resumed without re-running provider.
    restarted = [lane for lane in report.lanes if lane.restarted]
    assert len(restarted) == 1
    assert restarted[0].provider_duplicated_on_resume is True

    # Evidence subset complete.
    for key in REQUIRED_EVIDENCE_KEYS:
        assert report.evidence_subset.get(key) is True, key

    # Final agreement numbers.
    agreement = report.agreement
    assert agreement.completed_task_count >= 5  # 4 primary + refill
    assert agreement.accepted_mutation_count >= 4
    assert agreement.merge_accepted_count >= 4
    assert agreement.proof_count >= 4
    assert agreement.complete_lineage_count >= 4
    assert agreement.duplicate_claims == 0
    assert agreement.stale_writes == 0
    assert agreement.worktree_count >= 4
    assert agreement.event_count >= 1

    # Report is content-addressed and free of raw token material.
    payload = report.to_dict()
    assert payload["verdict"] == "passed"
    assert payload["failure_count"] == 0
    content_id = report.content_id()
    assert content_id.startswith("sha256:")
    serialized = json.dumps(payload)
    assert "super-secret" not in serialized
    assert "quack_token" not in serialized.lower() or "handle:" in serialized


def test_canary_harness_phased_api(tmp_path: Path) -> None:
    """Exercise the public phased API without the full run() wrapper."""

    with DuckDBQuackCanary(tmp_path / "phased", lane_count=2) as canary:
        owner = canary.start_state_owner()
        assert owner["ready"] is True
        lanes = canary.register_lanes()
        assert len(lanes) == 2
        canary.materialize_tasks()
        receipts = canary.execute_lanes(lanes, restart_worker_lane_index=0)
        assert len(receipts) == 2
        assert any(item.restarted for item in receipts)
        refill = canary.refill_tasks()
        assert refill["status"] == "completed"
        export_receipt = canary.export_and_prove_non_authority()
        assert export_receipt.unaffected is True
        drain = canary.drain()
        assert drain["server_stopped"] is True
        agreement = canary.final_agreement()
        assert agreement.agreed is True
        assert agreement.duplicate_claims == 0


def test_export_tamper_does_not_change_task_status(tmp_path: Path) -> None:
    report = run_duckdb_quack_canary(tmp_path / "export-only", lane_count=2)
    assert_canary_passed(report)
    receipt = report.export_receipt
    assert receipt.event_count_before == receipt.event_count_after_tamper
    assert receipt.event_count_before == receipt.event_count_after_delete
    assert receipt.task_completed_before == receipt.task_completed_after
    assert receipt.board_tampered is True
    assert receipt.board_deleted is True
    assert not Path(receipt.export_path).exists()
    assert not Path(receipt.board_export_path).exists()
