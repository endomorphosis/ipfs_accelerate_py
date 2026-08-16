"""Tests for DuckDBControlPlaneReleaseReceipt@1 (DQP-039)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.validation.duckdb_quack_baseline import (
    SAFETY_FLOOR_KEYS,
)
from ipfs_accelerate_py.agent_supervisor.validation.duckdb_quack_release import (
    DUCKDB_CONTROL_PLANE_RELEASE_INTERFACE,
    EVIDENCE,
    GOAL_ID,
    PRIOR_TASK_IDS,
    REQUIRED_COMPONENT_ROOTS,
    TASK_ID,
    ComponentEvidence,
    ComponentStatus,
    DuckDBControlPlaneReleaseReceipt,
    ReleaseVerdict,
    SafetyFloorObservation,
    board_tasks_terminal,
    evaluate_release,
    hermetic_component_bundle,
    parse_board_task_statuses,
    run_hermetic_release,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
BOARD = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "agent_supervisor_duckdb_quack_control_plane.todo.md"
)
RELEASE_DOC = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "AGENT_SUPERVISOR_DUCKDB_QUACK_RELEASE.md"
)


def test_interface_identity() -> None:
    assert (
        DUCKDB_CONTROL_PLANE_RELEASE_INTERFACE
        == "DuckDBControlPlaneReleaseReceipt@1"
    )
    assert (
        DuckDBControlPlaneReleaseReceipt.INTERFACE
        == DUCKDB_CONTROL_PLANE_RELEASE_INTERFACE
    )
    assert TASK_ID == "DQP-039"
    assert GOAL_ID == "DQP-G090"
    assert EVIDENCE == "dqp/duckdb-quack-release@1"
    assert len(REQUIRED_COMPONENT_ROOTS) >= 17
    assert PRIOR_TASK_IDS[0] == "DQP-000"
    assert PRIOR_TASK_IDS[-1] == "DQP-038"


def test_hermetic_release_passes_with_terminal_tasks() -> None:
    # Force tasks terminal so hermetic unit tests do not depend on live board
    # mutation order while DQP-039 itself is still open.
    receipt = run_hermetic_release(force_tasks_terminal=True)
    assert receipt.verdict is ReleaseVerdict.PASS
    assert receipt.passed is True
    assert receipt.tasks_terminal is True
    assert receipt.safety_floors_zero is True
    assert receipt.quality_non_regressing is True
    assert receipt.rollback_present is True
    assert receipt.legacy_file_decision_read_in_canary is False
    assert receipt.experimental_scope is True
    assert receipt.production_ha_claimed is False
    assert receipt.duckdb_2_0_compatibility_claimed is False
    assert receipt.missing_roots == ()
    assert not receipt.modules_missing
    assert set(receipt.components) == set(REQUIRED_COMPONENT_ROOTS)
    assert all(
        status == ComponentStatus.PASS.value
        for status in receipt.components.values()
    )
    payload = receipt.to_dict()
    assert payload["interface"] == DUCKDB_CONTROL_PLANE_RELEASE_INTERFACE
    assert payload["experimental_scope"] is True
    assert payload["production_ha_claimed"] is False
    assert payload["duckdb_2_0_compatibility_claimed"] is False
    assert payload["task_id"] == "DQP-039"
    assert "identity_id" in payload


def test_missing_component_blocks_or_fails() -> None:
    receipt = run_hermetic_release(
        force_tasks_terminal=True,
        exclude_roots=("chaos",),
    )
    assert receipt.passed is False
    assert "chaos" in receipt.missing_roots
    assert any(r.startswith("missing:chaos") for r in receipt.reason_codes)


def test_stale_evidence_blocks() -> None:
    receipt = run_hermetic_release(
        force_tasks_terminal=True,
        stale_roots=("canary",),
    )
    assert receipt.passed is False
    assert receipt.components["canary"] == ComponentStatus.STALE.value
    assert any(r.startswith("stale:canary") for r in receipt.reason_codes)


def test_synthetic_evidence_fails() -> None:
    receipt = run_hermetic_release(
        force_tasks_terminal=True,
        synthetic_roots=("shadow",),
    )
    assert receipt.verdict is ReleaseVerdict.FAIL
    assert receipt.components["shadow"] == ComponentStatus.SYNTHETIC.value


def test_skipped_evidence_blocks() -> None:
    receipt = run_hermetic_release(
        force_tasks_terminal=True,
        skipped_roots=("backup",),
    )
    assert receipt.passed is False
    assert receipt.components["backup"] == ComponentStatus.SKIPPED.value


def test_failed_component_fails_release() -> None:
    receipt = run_hermetic_release(
        force_tasks_terminal=True,
        fail_roots=("canary",),
    )
    assert receipt.verdict is ReleaseVerdict.FAIL
    assert receipt.components["canary"] == ComponentStatus.FAIL.value


def test_safety_floor_nonzero_fails() -> None:
    floors = SafetyFloorObservation(
        floors={
            **{key: 0 for key in SAFETY_FLOOR_KEYS},
            "unauthorized_sql": 1,
        }
    )
    receipt = run_hermetic_release(
        force_tasks_terminal=True,
        safety_floors=floors,
    )
    assert receipt.verdict is ReleaseVerdict.FAIL
    assert receipt.safety_floors_zero is False
    assert "safety_floor_nonzero" in receipt.reason_codes


def test_legacy_file_decision_read_in_canary_fails() -> None:
    receipt = run_hermetic_release(
        force_tasks_terminal=True,
        legacy_file_decision_read_in_canary=True,
    )
    assert receipt.verdict is ReleaseVerdict.FAIL
    assert "legacy_file_decision_read_in_canary" in receipt.reason_codes


def test_quality_regression_fails() -> None:
    receipt = run_hermetic_release(
        force_tasks_terminal=True,
        quality_non_regressing=False,
    )
    assert receipt.verdict is ReleaseVerdict.FAIL
    assert "quality_regression" in receipt.reason_codes


def test_never_fabricates_missing_roots() -> None:
    components = hermetic_component_bundle(exclude_roots=("rollback", "cutover"))
    receipt = evaluate_release(
        components=components,
        tree_id="tree:t",
        schema_checksum="sha256:" + ("ee" * 32),
        store_generation=1,
        quack_profile="profile:p",
        extension_fingerprint="sha256:" + ("ff" * 32),
        git_identity="git:x",
        tasks_terminal=True,
    )
    assert "rollback" in receipt.missing_roots
    assert "cutover" in receipt.missing_roots
    # Verifier must not invent pass statuses for missing roots.
    assert receipt.components["rollback"] == ComponentStatus.MISSING.value
    assert receipt.components["cutover"] == ComponentStatus.MISSING.value


def test_board_parser_and_live_board_when_available() -> None:
    assert BOARD.is_file()
    text = BOARD.read_text(encoding="utf-8")
    statuses = parse_board_task_statuses(text)
    assert "DQP-000" in statuses
    assert statuses["DQP-000"] == "completed"
    # DQP-039 itself may still be todo at evaluation time; prior tasks should
    # be complete after the release tail lands.
    terminal, incomplete = board_tasks_terminal(text)
    # Allow DQP-039 still open; prior must be complete for a true release.
    open_prior = [t for t in incomplete if t != "DQP-039"]
    assert open_prior == [], f"prior tasks still open: {open_prior}"
    # When prior are complete, force_tasks_terminal path already covered pass.
    if terminal:
        receipt = run_hermetic_release(board_text=text)
        assert receipt.tasks_terminal is True


def test_release_doc_exists_and_states_non_claims() -> None:
    assert RELEASE_DOC.is_file()
    text = RELEASE_DOC.read_text(encoding="utf-8")
    for phrase in (
        "experimental",
        "production",
        "high availability",
        "2.0",
        "rollback",
        "loopback",
        "beta",
    ):
        assert phrase.lower() in text.lower(), phrase


def test_tree_mismatch_fails_component() -> None:
    components = hermetic_component_bundle(tree_id="tree:other")
    # Re-bind one component to wrong tree while envelope expects tree:main.
    items = []
    for item in components:
        if item.root == "schema":
            items.append(
                ComponentEvidence(
                    root=item.root,
                    identity=item.identity,
                    age_seconds=item.age_seconds,
                    passed=True,
                    tree_id="tree:wrong",
                    schema_checksum=item.schema_checksum,
                    store_generation=item.store_generation,
                    profile_id=item.profile_id,
                )
            )
        else:
            items.append(item)
    receipt = evaluate_release(
        components=tuple(items),
        tree_id="tree:other",
        schema_checksum=items[0].schema_checksum,
        store_generation=1,
        quack_profile=items[0].profile_id,
        extension_fingerprint="sha256:" + ("11" * 32),
        git_identity="git:x",
        tasks_terminal=True,
    )
    assert receipt.passed is False
    assert receipt.components["schema"] == ComponentStatus.FAIL.value
