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
    DEFAULT_MAX_BACKUP_AGE_SECONDS,
    EVIDENCE,
    GOAL_ID,
    REASON_BACKUP_AGE,
    REASON_BETA_WAIVER,
    REASON_DEFAULT_REQUIRES_CANARY,
    REASON_HISTORY_PRESERVED,
    REASON_KILL_SWITCH,
    REASON_LEGACY_EXPORT,
    REASON_MISSING_EVIDENCE,
    REASON_NO_LEGACY_DUAL_WRITE,
    REASON_PARTIAL_ROLLOUT,
    REASON_PROMOTION_DENIED,
    REASON_REMOTE_PROHIBITION,
    REASON_ROLLBACK,
    REASON_SERVER_UNAVAILABLE,
    REASON_STALE_EVIDENCE,
    REQUIRED_EVIDENCE_KINDS,
    TASK_ID,
    DatabaseCutoverReceipt,
    DatabaseRollout,
    DatabaseRolloutBinding,
    DatabaseRolloutError,
    DatabaseRolloutPolicy,
    DatabaseRolloutStage,
    EvidenceStatus,
    OperatorAction,
    RolloutDisposition,
    RolloutEvidenceCell,
    authority_mode_for_stage,
    closed_rollout_stages,
    default_passing_evidence,
    dual_observation_allowed,
    run_full_cutover,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_capabilities import (
    DEFAULT_QUACK_BETA_LIMITATIONS,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.task_source import (
    StateAuthorityMode,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
OPS_SCRIPT = (
    REPO_ROOT / "scripts/ops/agent_supervisor/duckdb_quack_control_plane.py"
)
GUIDE_PATH = (
    REPO_ROOT / "docs/guides/AGENT_SUPERVISOR_DUCKDB_QUACK_GUIDE.md"
)


def _promote_to(
    rollout: DatabaseRollout,
    stage: DatabaseRolloutStage,
    *,
    evidence: dict | None = None,
) -> DatabaseCutoverReceipt:
    cells = evidence or default_passing_evidence(
        tree_id=rollout.binding.tree_id
    )
    last: DatabaseCutoverReceipt | None = None
    for step in DatabaseRolloutStage:
        if step.rank == 0:
            continue
        if step.rank > stage.rank:
            break
        last = rollout.promote(
            step,
            evidence=cells,
            server_available=True,
            backup_age_seconds=0,
            beta_waiver_recorded=True,
        )
        assert last.disposition is RolloutDisposition.PROMOTE, last.to_dict()
    assert last is not None
    return last


# ---------------------------------------------------------------------------
# Identity / vocabulary
# ---------------------------------------------------------------------------


def test_interface_identities() -> None:
    assert DATABASE_ROLLOUT_POLICY_INTERFACE == "DatabaseRolloutPolicy@1"
    assert DATABASE_CUTOVER_RECEIPT_INTERFACE == "DatabaseCutoverReceipt@1"
    assert DatabaseRollout.INTERFACE == DATABASE_ROLLOUT_POLICY_INTERFACE
    assert DatabaseCutoverReceipt.INTERFACE == DATABASE_CUTOVER_RECEIPT_INTERFACE
    assert DatabaseRolloutPolicy.INTERFACE == DATABASE_ROLLOUT_POLICY_INTERFACE
    assert TASK_ID == "DQP-038"
    assert GOAL_ID == "DQP-G080"
    assert EVIDENCE == "dqp/database-rollout@1"


def test_closed_stage_ladder_and_authority_binding() -> None:
    assert closed_rollout_stages() == (
        "off",
        "observe",
        "shadow",
        "assist",
        "canary",
        "default",
    )
    assert authority_mode_for_stage("off") is StateAuthorityMode.EMBEDDED_MAINTENANCE
    assert authority_mode_for_stage("observe") is StateAuthorityMode.EMBEDDED_MAINTENANCE
    assert authority_mode_for_stage("shadow") is StateAuthorityMode.QUACK_SHADOW
    assert authority_mode_for_stage("assist") is StateAuthorityMode.QUACK_SHADOW
    assert authority_mode_for_stage("canary") is StateAuthorityMode.QUACK_AUTHORITATIVE
    assert authority_mode_for_stage("default") is StateAuthorityMode.QUACK_AUTHORITATIVE
    assert dual_observation_allowed("shadow") is True
    assert dual_observation_allowed("assist") is True
    assert dual_observation_allowed("canary") is False
    assert dual_observation_allowed("default") is False


def test_policy_refuses_permanent_dual_write_and_history_deletion() -> None:
    policy = DatabaseRolloutPolicy(
        accept_legacy_dual_writes=True,  # type: ignore[arg-type]
        allow_history_deletion=True,  # type: ignore[arg-type]
    )
    assert policy.accept_legacy_dual_writes is False
    assert policy.allow_history_deletion is False
    payload = policy.to_dict()
    assert payload["accept_legacy_dual_writes"] is False
    assert payload["allow_history_deletion"] is False


def test_default_policy_excludes_default_stage() -> None:
    policy = DatabaseRolloutPolicy.default()
    assert DatabaseRolloutStage.DEFAULT not in policy.allowed_stages
    assert policy.allows("canary") is True
    assert policy.allows("default") is False


# ---------------------------------------------------------------------------
# Promotion / release gate
# ---------------------------------------------------------------------------


def test_full_cutover_defaults_new_programs_to_quack() -> None:
    rollout, receipt = run_full_cutover()
    assert receipt.disposition is RolloutDisposition.PROMOTE
    assert receipt.to_stage is DatabaseRolloutStage.DEFAULT
    assert receipt.accepted is True
    assert receipt.release_gate_valid is True
    assert receipt.history_preserved is True
    assert receipt.legacy_dual_write_accepted is False
    assert rollout.new_programs_default_to_quack() is True
    assert rollout.default_stage_for_new_program() is DatabaseRolloutStage.DEFAULT
    record = rollout.register_new_program("program:local-1")
    assert record["defaults_to_quack"] is True
    assert record["stage"] == "default"
    assert record["authority_mode"] == StateAuthorityMode.QUACK_AUTHORITATIVE.value
    payload = receipt.to_dict()
    assert payload["interface"] == DATABASE_CUTOVER_RECEIPT_INTERFACE
    assert payload["task_id"] == "DQP-038"
    assert payload["new_program_defaults_to_quack"] is True


def test_new_programs_do_not_default_without_release_gate() -> None:
    rollout = DatabaseRollout()
    assert rollout.new_programs_default_to_quack() is False
    record = rollout.register_new_program("program:early")
    assert record["defaults_to_quack"] is False
    assert record["stage"] == "off"
    # Even at canary, default cutover has not happened.
    _promote_to(rollout, DatabaseRolloutStage.CANARY)
    assert rollout.stage is DatabaseRolloutStage.CANARY
    assert rollout.new_programs_default_to_quack() is False
    assert rollout.default_stage_for_new_program() is DatabaseRolloutStage.OFF


def test_promotion_is_one_stage_at_a_time() -> None:
    rollout = DatabaseRollout(DatabaseRolloutPolicy.with_default_cutover())
    denied = rollout.promote("canary", evidence=default_passing_evidence())
    assert denied.disposition is RolloutDisposition.DENY
    assert REASON_PARTIAL_ROLLOUT in denied.reason_codes
    assert REASON_PROMOTION_DENIED in denied.reason_codes
    assert rollout.stage is DatabaseRolloutStage.OFF


def test_promotion_denial_stale_evidence() -> None:
    rollout = DatabaseRollout(DatabaseRolloutPolicy.with_default_cutover())
    _promote_to(rollout, DatabaseRolloutStage.ASSIST)
    stale = default_passing_evidence()
    stale["canary_e2e"]["status"] = EvidenceStatus.STALE.value
    stale["canary_e2e"]["passed"] = False
    denied = rollout.promote("canary", evidence=stale)
    assert denied.disposition is RolloutDisposition.DENY
    assert REASON_STALE_EVIDENCE in denied.reason_codes or REASON_PROMOTION_DENIED in denied.reason_codes


def test_promotion_denial_missing_evidence() -> None:
    rollout = DatabaseRollout(DatabaseRolloutPolicy.with_default_cutover())
    _promote_to(rollout, DatabaseRolloutStage.ASSIST)
    denied = rollout.promote("canary", evidence={})
    assert denied.disposition is RolloutDisposition.DENY
    assert REASON_MISSING_EVIDENCE in denied.reason_codes


def test_promotion_denial_server_unavailable() -> None:
    rollout = DatabaseRollout(DatabaseRolloutPolicy.with_default_cutover())
    _promote_to(rollout, DatabaseRolloutStage.ASSIST)
    denied = rollout.promote(
        "canary",
        evidence=default_passing_evidence(),
        server_available=False,
    )
    assert denied.disposition is RolloutDisposition.DENY
    assert REASON_SERVER_UNAVAILABLE in denied.reason_codes


def test_promotion_denial_backup_age() -> None:
    rollout = DatabaseRollout(DatabaseRolloutPolicy.with_default_cutover())
    _promote_to(rollout, DatabaseRolloutStage.ASSIST)
    denied = rollout.promote(
        "canary",
        evidence=default_passing_evidence(),
        backup_age_seconds=DEFAULT_MAX_BACKUP_AGE_SECONDS + 1,
    )
    assert denied.disposition is RolloutDisposition.DENY
    assert REASON_BACKUP_AGE in denied.reason_codes


def test_promotion_denial_partial_rollout() -> None:
    rollout = DatabaseRollout(DatabaseRolloutPolicy.with_default_cutover())
    _promote_to(rollout, DatabaseRolloutStage.ASSIST)
    denied = rollout.promote(
        "canary",
        evidence=default_passing_evidence(),
        partial_rollout=True,
    )
    assert denied.disposition is RolloutDisposition.DENY
    assert REASON_PARTIAL_ROLLOUT in denied.reason_codes


def test_promotion_denial_remote_prohibition() -> None:
    rollout = DatabaseRollout(DatabaseRolloutPolicy.with_default_cutover())
    _promote_to(rollout, DatabaseRolloutStage.ASSIST)
    denied = rollout.promote(
        "canary",
        evidence=default_passing_evidence(),
        remote_bind_requested=True,
    )
    assert denied.disposition is RolloutDisposition.DENY
    assert REASON_REMOTE_PROHIBITION in denied.reason_codes


def test_promotion_denial_beta_waiver() -> None:
    rollout = DatabaseRollout(DatabaseRolloutPolicy.with_default_cutover())
    _promote_to(rollout, DatabaseRolloutStage.ASSIST)
    denied = rollout.promote(
        "canary",
        evidence=default_passing_evidence(),
        beta_waiver_recorded=False,
    )
    assert denied.disposition is RolloutDisposition.DENY
    assert REASON_BETA_WAIVER in denied.reason_codes


def test_promotion_denial_legacy_dual_write_acceptance() -> None:
    rollout = DatabaseRollout(DatabaseRolloutPolicy.with_default_cutover())
    _promote_to(rollout, DatabaseRolloutStage.ASSIST)
    denied = rollout.promote(
        "canary",
        evidence=default_passing_evidence(),
        dual_write_accepted=True,
    )
    assert denied.disposition is RolloutDisposition.DENY
    assert REASON_LEGACY_EXPORT in denied.reason_codes


def test_default_requires_canary() -> None:
    # Force a policy that somehow tries to jump with canary flag cleared.
    rollout = DatabaseRollout(DatabaseRolloutPolicy.with_default_cutover())
    _promote_to(rollout, DatabaseRolloutStage.ASSIST)
    # Manually mark canary incomplete after promoting to canary without flag.
    receipt = rollout.promote(
        "canary",
        evidence=default_passing_evidence(),
    )
    assert receipt.disposition is RolloutDisposition.PROMOTE
    rollout._canary_completed = False  # noqa: SLF001 — adversarial
    denied = rollout.promote("default", evidence=default_passing_evidence())
    assert denied.disposition is RolloutDisposition.DENY
    assert REASON_DEFAULT_REQUIRES_CANARY in denied.reason_codes


def test_kill_switch_denies_promotion() -> None:
    rollout = DatabaseRollout(DatabaseRolloutPolicy.with_default_cutover())
    rollout.engage_kill_switch()
    denied = rollout.promote("observe")
    assert denied.disposition is RolloutDisposition.DENY
    assert REASON_KILL_SWITCH in denied.reason_codes
    assert rollout.stage is DatabaseRolloutStage.OFF


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------


def test_rollback_switches_route_without_deleting_history() -> None:
    rollout, _ = run_full_cutover()
    history_before = len(rollout.history)
    assert history_before > 0
    receipt = rollout.rollback(
        DatabaseRolloutStage.ASSIST,
        delete_history=True,
        accept_legacy_dual_writes=True,
    )
    assert receipt.disposition is RolloutDisposition.ROLLBACK
    assert receipt.to_stage is DatabaseRolloutStage.ASSIST
    assert receipt.history_preserved is True
    assert receipt.legacy_dual_write_accepted is False
    assert REASON_ROLLBACK in receipt.reason_codes
    assert REASON_HISTORY_PRESERVED in receipt.reason_codes
    assert REASON_NO_LEGACY_DUAL_WRITE in receipt.reason_codes
    assert len(rollout.history) > history_before
    assert any(item.get("event") == "rollback" for item in rollout.history)
    # History entries from the cutover are still present.
    assert any(item.get("event") == "promote" for item in rollout.history)
    assert rollout.authority_mode is StateAuthorityMode.QUACK_SHADOW
    assert rollout.new_programs_default_to_quack() is False


def test_rollback_one_stage_by_default() -> None:
    rollout = DatabaseRollout(DatabaseRolloutPolicy.with_default_cutover())
    _promote_to(rollout, DatabaseRolloutStage.SHADOW)
    receipt = rollout.rollback()
    assert receipt.to_stage is DatabaseRolloutStage.OBSERVE
    assert receipt.history_preserved is True
    assert receipt.legacy_dual_write_accepted is False


def test_kill_switch_forces_off_preserving_history() -> None:
    rollout, _ = run_full_cutover()
    prior = len(rollout.history)
    receipt = rollout.engage_kill_switch()
    assert receipt.operator_action is OperatorAction.KILL_SWITCH
    assert receipt.to_stage is DatabaseRolloutStage.OFF
    assert receipt.history_preserved is True
    assert receipt.kill_switch_engaged is True
    assert len(rollout.history) > prior
    assert rollout.new_programs_default_to_quack() is False


# ---------------------------------------------------------------------------
# Evidence cells / binding
# ---------------------------------------------------------------------------


def test_release_gate_evaluation_payload() -> None:
    binding = DatabaseRolloutBinding(
        repository_id="repository:test",
        tree_id="tree:test",
        store_generation=3,
    )
    rollout = DatabaseRollout(
        DatabaseRolloutPolicy.with_default_cutover(), binding=binding
    )
    _promote_to(rollout, DatabaseRolloutStage.ASSIST)
    gate = rollout.evaluate_release_gate(
        default_passing_evidence(tree_id="tree:test"),
        target_stage="canary",
    )
    assert gate.valid is True
    assert set(cell.kind for cell in gate.evidence) == set(REQUIRED_EVIDENCE_KINDS)
    assert gate.backup_age_ok is True
    assert gate.remote_bind_prohibited is True
    assert "one_quack_server_is_one_failure_domain" in gate.beta_limitations
    assert "loopback_bind_required_unless_separately_reviewed" in gate.beta_limitations
    payload = gate.to_dict()
    assert payload["binding"]["store_generation"] == 3


def test_evidence_cell_admitted_requires_measured_pass() -> None:
    cell = RolloutEvidenceCell(
        kind="canary_e2e",
        status=EvidenceStatus.MEASURED,
        evidence_id="evidence:canary",
        passed=True,
    )
    assert cell.admitted is True
    bad = RolloutEvidenceCell(
        kind="canary_e2e",
        status=EvidenceStatus.SYNTHETIC,
        evidence_id="evidence:fake",
        passed=True,
    )
    assert bad.admitted is False


def test_unknown_stage_fails_closed() -> None:
    with pytest.raises(DatabaseRolloutError):
        authority_mode_for_stage("automatic")


# ---------------------------------------------------------------------------
# Ops script + guide
# ---------------------------------------------------------------------------


def test_ops_script_recipe_and_procedures() -> None:
    assert OPS_SCRIPT.is_file()
    result = subprocess.run(
        [sys.executable, str(OPS_SCRIPT), "recipe"],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["interface"] == "DatabaseRolloutPolicy@1"
    assert payload["receipt_interface"] == "DatabaseCutoverReceipt@1"
    assert payload["ladder"] == [
        "off",
        "observe",
        "shadow",
        "assist",
        "canary",
        "default",
    ]
    assert payload["single_failure_domain"] is True
    assert payload["loopback_required_by_default"] is True
    for name in ("health", "backup", "restore", "upgrade"):
        assert name in payload["procedures"]
        assert payload["procedures"][name]["steps"]
    assert set(DEFAULT_QUACK_BETA_LIMITATIONS).issubset(
        set(payload["beta_limitations"])
    )


def test_ops_script_limitations_and_health() -> None:
    for command in ("limitations", "health", "backup", "restore", "upgrade", "stages"):
        result = subprocess.run(
            [sys.executable, str(OPS_SCRIPT), command, "--json"],
            check=False,
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0, (command, result.stderr)
        payload = json.loads(result.stdout)
        assert payload


def test_ops_script_promote_and_rollback() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(OPS_SCRIPT),
            "--allow-default",
            "--target-stage",
            "observe",
            "promote",
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["disposition"] == "promote"
    assert payload["to_stage"] == "observe"
    assert payload["history_preserved"] is True
    assert payload["legacy_dual_write_accepted"] is False


def test_ops_script_refuses_argv_credentials() -> None:
    result = subprocess.run(
        [sys.executable, str(OPS_SCRIPT), "status", "--token", "secret"],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode != 0
    combined = (result.stdout + result.stderr).lower()
    assert "token" in combined or "credential" in combined or "refusing" in combined


def test_guide_states_limitations_and_procedures() -> None:
    assert GUIDE_PATH.is_file()
    text = GUIDE_PATH.read_text(encoding="utf-8")
    # Beta / single-failure-domain / loopback
    assert "beta" in text.lower()
    assert "one_quack_server_is_one_failure_domain" in text
    assert "single failure domain" in text.lower() or "single-failure-domain" in text.lower()
    assert "loopback" in text.lower()
    assert "loopback_bind_required_unless_separately_reviewed" in text
    # Exact procedures
    assert "Exact health procedure" in text
    assert "Exact backup procedure" in text
    assert "Exact restore procedure" in text
    assert "Exact upgrade procedure" in text
    assert "ControlPlaneBackup@1" in text
    assert "RestoreReceipt@1" in text
    assert "DEFAULT_MAX_BACKUP_AGE_SECONDS" in text or "30 days" in text
    # Acceptance themes
    assert "DatabaseRolloutPolicy@1" in text
    assert "DatabaseCutoverReceipt@1" in text
    assert "never deleted" in text.lower() or "without deleting history" in text.lower()
    assert "dual write" in text.lower() or "dual-write" in text.lower()
    assert "release gate" in text.lower()
    assert "duckdb-1.5.x-quack-pinned" in text


def test_status_snapshot_includes_beta_limitations() -> None:
    status = DatabaseRollout().status()
    assert status["stage"] == "off"
    assert status["history_preserved"] is True
    assert status["accept_legacy_dual_writes"] is False
    assert "one_quack_server_is_one_failure_domain" in status["beta_limitations"]
