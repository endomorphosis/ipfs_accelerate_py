"""Tests for DatabaseRolloutPolicy@1 / DatabaseCutoverReceipt@1 (DQP-038)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.self_improvement.database_rollout import (
    DATABASE_CUTOVER_RECEIPT_INTERFACE,
    DATABASE_ROLLOUT_POLICY_INTERFACE,
    EVIDENCE,
    GOAL_ID,
    REQUIRED_EVIDENCE_ROOTS,
    TASK_ID,
    CutoverVerdict,
    DatabaseCutoverReceipt,
    DatabaseRollout,
    DatabaseRolloutError,
    DatabaseRolloutPolicy,
    DenialReason,
    EvidenceBundle,
    EvidenceItem,
    RolloutStage,
    hermetic_passing_evidence,
    run_staged_cutover_to_default,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.task_source import (
    StateAuthorityMode,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CLI = REPO_ROOT / "scripts" / "ops" / "agent_supervisor" / "duckdb_quack_control_plane.py"
GUIDE = REPO_ROOT / "docs" / "guides" / "AGENT_SUPERVISOR_DUCKDB_QUACK_GUIDE.md"


def test_interface_identities() -> None:
    assert DATABASE_ROLLOUT_POLICY_INTERFACE == "DatabaseRolloutPolicy@1"
    assert DATABASE_CUTOVER_RECEIPT_INTERFACE == "DatabaseCutoverReceipt@1"
    assert DatabaseRollout.INTERFACE == DATABASE_ROLLOUT_POLICY_INTERFACE
    assert DatabaseCutoverReceipt.INTERFACE == DATABASE_CUTOVER_RECEIPT_INTERFACE
    assert TASK_ID == "DQP-038"
    assert GOAL_ID == "DQP-G080"
    assert EVIDENCE == "dqp/database-rollout@1"


def test_staged_cutover_to_default_under_valid_evidence() -> None:
    controller, receipts = run_staged_cutover_to_default()
    assert controller.stage is RolloutStage.DEFAULT
    assert controller.authority_mode == StateAuthorityMode.QUACK_AUTHORITATIVE.value
    assert all(item.promoted for item in receipts)
    assert receipts[-1].to_stage is RolloutStage.DEFAULT
    assert receipts[-1].history_preserved is True
    assert receipts[-1].dual_write_accepted is False
    payload = receipts[-1].to_dict()
    assert payload["interface"] == DATABASE_CUTOVER_RECEIPT_INTERFACE
    assert payload["dual_write_accepted"] is False
    assert payload["promoted"] is True


def test_default_denied_without_canary_evidence() -> None:
    controller = DatabaseRollout(initial_stage=RolloutStage.CANARY)
    # Missing canary root.
    items = [
        EvidenceItem(
            root=root,
            identity=f"e:{root}",
            age_seconds=10,
            passed=True,
            tree_id="tree:x",
            schema_checksum="sha256:" + ("bb" * 32),
            profile_id="profile:p",
        )
        for root in REQUIRED_EVIDENCE_ROOTS
        if root != "canary"
    ]
    evidence = EvidenceBundle(
        items=tuple(items),
        tree_id="tree:x",
        schema_checksum="sha256:" + ("bb" * 32),
        store_generation=1,
        quack_profile="profile:p",
        beta_waiver=True,
        backup_age_seconds=10,
    )
    receipt = controller.transition(RolloutStage.DEFAULT, evidence)
    assert receipt.verdict is CutoverVerdict.DENIED
    assert receipt.promoted is False
    assert any("missing_evidence:canary" in r for r in receipt.denial_reasons)
    assert controller.stage is RolloutStage.CANARY


def test_stale_evidence_denies_default() -> None:
    controller = DatabaseRollout(initial_stage=RolloutStage.CANARY)
    evidence = hermetic_passing_evidence(age_seconds=10**9)
    receipt = controller.transition(RolloutStage.DEFAULT, evidence)
    assert receipt.promoted is False
    assert any(r.startswith("stale_evidence") for r in receipt.denial_reasons)


def test_server_unavailable_denies_canary() -> None:
    controller = DatabaseRollout(initial_stage=RolloutStage.ASSIST)
    evidence = hermetic_passing_evidence()
    # Rebuild with server down.
    evidence = EvidenceBundle(
        items=evidence.items,
        tree_id=evidence.tree_id,
        schema_checksum=evidence.schema_checksum,
        store_generation=evidence.store_generation,
        quack_profile=evidence.quack_profile,
        server_available=False,
        beta_waiver=True,
        backup_age_seconds=10,
    )
    receipt = controller.transition(RolloutStage.CANARY, evidence)
    assert receipt.promoted is False
    assert DenialReason.SERVER_UNAVAILABLE.value in receipt.denial_reasons


def test_remote_endpoint_prohibited() -> None:
    controller = DatabaseRollout(initial_stage=RolloutStage.CANARY)
    evidence = hermetic_passing_evidence()
    evidence = EvidenceBundle(
        items=evidence.items,
        tree_id=evidence.tree_id,
        schema_checksum=evidence.schema_checksum,
        store_generation=evidence.store_generation,
        quack_profile=evidence.quack_profile,
        remote_endpoint=True,
        beta_waiver=True,
        backup_age_seconds=10,
    )
    receipt = controller.transition(RolloutStage.DEFAULT, evidence)
    assert receipt.promoted is False
    assert DenialReason.REMOTE_PROHIBITED.value in receipt.denial_reasons


def test_rollback_preserves_history_no_dual_write() -> None:
    controller, receipts = run_staged_cutover_to_default()
    history_len = len(controller._history)
    evidence = hermetic_passing_evidence()
    receipt = controller.transition(
        RolloutStage.ROLLBACK, evidence, force_rollback=True
    )
    assert receipt.verdict is CutoverVerdict.ROLLED_BACK
    assert receipt.history_preserved is True
    assert receipt.dual_write_accepted is False
    assert controller.stage is RolloutStage.ROLLBACK
    assert controller.authority_mode == StateAuthorityMode.EMBEDDED_MAINTENANCE.value
    assert len(controller._history) > history_len
    assert controller._legacy_export_only is True


def test_kill_switch_blocks_promotion() -> None:
    policy = DatabaseRolloutPolicy(kill_switch_engaged=True)
    controller = DatabaseRollout(policy=policy, initial_stage=RolloutStage.CANARY)
    receipt = controller.transition(RolloutStage.DEFAULT, hermetic_passing_evidence())
    assert receipt.promoted is False
    assert DenialReason.KILL_SWITCH.value in receipt.denial_reasons


def test_policy_rejects_dual_write_flag() -> None:
    with pytest.raises(DatabaseRolloutError, match="dual writes"):
        DatabaseRolloutPolicy(allow_legacy_dual_write=True)


def test_illegal_transition_denied() -> None:
    controller = DatabaseRollout(initial_stage=RolloutStage.OFF)
    receipt = controller.transition(RolloutStage.DEFAULT, hermetic_passing_evidence())
    assert receipt.promoted is False
    assert DenialReason.ILLEGAL_TRANSITION.value in receipt.denial_reasons


def test_operator_guide_states_beta_limitations() -> None:
    assert GUIDE.is_file()
    text = GUIDE.read_text(encoding="utf-8")
    for phrase in (
        "beta",
        "single-failure-domain",
        "loopback",
        "backup",
        "restore",
        "rollback",
        "health",
        "upgrade",
    ):
        assert phrase.lower() in text.lower(), phrase


def test_cli_stages_and_guide_check() -> None:
    stages = subprocess.run(
        [sys.executable, str(CLI), "stages"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert stages.returncode == 0, stages.stderr
    payload = json.loads(stages.stdout)
    assert "default" in payload["stages"]
    assert "canary" in payload["stages"]

    guide = subprocess.run(
        [sys.executable, str(CLI), "guide-check"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert guide.returncode == 0, guide.stderr
    assert json.loads(guide.stdout)["ok"] is True


def test_cli_walk_promote_to_default() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "promote",
            "--from-stage",
            "off",
            "--to",
            "default",
            "--walk",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    payload = json.loads(result.stdout)
    assert payload["receipts"][-1]["to_stage"] == "default"
    assert payload["receipts"][-1]["promoted"] is True


def test_cli_rollback() -> None:
    result = subprocess.run(
        [sys.executable, str(CLI), "rollback", "--from-stage", "canary"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["verdict"] == "rolled_back"
    assert payload["history_preserved"] is True
