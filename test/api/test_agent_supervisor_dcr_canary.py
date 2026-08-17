"""DCR-102: fixture-apply then auto-safe canary mode.

Acceptance:
* Mode progression report_only → fixture_apply → auto_safe only.
* Always-abstain families never apply.
* Safety-floor breach disables apply (report-only evidence).
* Circuit breakers and rollback drills recorded.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.evaluation.dcr_canary import (
    ALWAYS_ABSTAIN_FAMILIES,
    AUTO_SAFE_ADMISSION_INTERFACE,
    DEFAULT_CANARY_REPORT_PATH,
    DEFAULT_POLICY_PATH,
    DETERMINISTIC_REPAIR_POLICY_INTERFACE,
    DCR_CANARY_EVIDENCE,
    DCR_TASK_ID,
    AutoSafeAdmission,
    CanaryError,
    DeterministicRepairPolicy,
    RepairExecutionMode,
    default_policy,
    materialize_policy_and_canary,
    run_fixture_apply_canary,
)


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parents[4], here.parents[3], Path.cwd()):
        if (candidate / "config" / "deterministic_contract_repair_services.json").is_file():
            return candidate
    return here.parents[4]


@pytest.fixture(scope="module")
def canary_report():
    return run_fixture_apply_canary(repo_root=_repo_root())


def test_interfaces_and_mode_progression() -> None:
    assert DETERMINISTIC_REPAIR_POLICY_INTERFACE == "DeterministicRepairPolicy@1"
    assert AUTO_SAFE_ADMISSION_INTERFACE == "AutoSafeAdmission@1"
    assert AutoSafeAdmission.INTERFACE == AUTO_SAFE_ADMISSION_INTERFACE
    assert DCR_TASK_ID == "DCR-102"
    assert DCR_CANARY_EVIDENCE == "dcr/canary-admission@1"
    assert RepairExecutionMode.progression() == (
        RepairExecutionMode.REPORT_ONLY,
        RepairExecutionMode.FIXTURE_APPLY,
        RepairExecutionMode.AUTO_SAFE,
    )
    assert RepairExecutionMode.REPORT_ONLY.can_advance_to(
        RepairExecutionMode.FIXTURE_APPLY
    )
    assert not RepairExecutionMode.REPORT_ONLY.can_advance_to(
        RepairExecutionMode.AUTO_SAFE
    )


def test_illegal_skip_transition_rejected() -> None:
    policy = default_policy(mode=RepairExecutionMode.REPORT_ONLY)
    with pytest.raises(CanaryError):
        policy.advance(RepairExecutionMode.AUTO_SAFE)


def test_canary_reaches_auto_safe(canary_report) -> None:
    assert canary_report.passed is True
    assert canary_report.policy.mode is RepairExecutionMode.AUTO_SAFE
    assert canary_report.admission.admitted is True
    assert canary_report.shadow_precondition_ok is True
    assert canary_report.benchmark_precondition_ok is True
    assert "advance:fixture_apply" in canary_report.mode_transitions
    assert "advance:auto_safe" in canary_report.mode_transitions
    assert canary_report.runtime_model_calls == 0
    assert canary_report.provider_calls == 0


def test_always_abstain_families_never_apply(canary_report) -> None:
    for repair in canary_report.admission.repairs:
        if repair.family in ALWAYS_ABSTAIN_FAMILIES:
            assert repair.applied is False
            assert repair.admitted is False


def test_allowlisted_fixture_applies_and_rollback_drill(canary_report) -> None:
    applied = [r for r in canary_report.admission.repairs if r.applied]
    assert applied
    assert any(r.rolled_back for r in canary_report.admission.repairs)
    assert canary_report.admission.rollback_drill_ok is True
    assert canary_report.admission.circuit_breaker.tripped is False


def test_safety_breach_forces_report_only() -> None:
    policy = default_policy(mode=RepairExecutionMode.AUTO_SAFE)
    assert policy.apply_enabled is True
    disabled = policy.disable_apply_on_breach()
    assert disabled.mode is RepairExecutionMode.REPORT_ONLY
    assert disabled.apply_enabled is False


def test_report_only_cannot_enable_apply() -> None:
    with pytest.raises(CanaryError):
        DeterministicRepairPolicy(
            mode=RepairExecutionMode.REPORT_ONLY,
            allowlisted_operators=("operator:x@1",),
            always_abstain_families=tuple(sorted(ALWAYS_ABSTAIN_FAMILIES)),
            circuit_breaker={"max_error_rate_numerator": 0, "max_error_rate_denominator": 1,
                             "max_apply_rate_per_window": 1, "max_rollback_events": 0,
                             "review_window_repairs": 1},
            safety_floors={"false_completion": 0},
            apply_enabled=True,
        )


def test_materialize_policy_and_canary(tmp_path: Path) -> None:
    policy_dest = tmp_path / "deterministic_contract_repair_policy.json"
    report_dest = tmp_path / "canary-report.json"
    payload = materialize_policy_and_canary(
        repo_root=_repo_root(),
        policy_destination=policy_dest,
        report_destination=report_dest,
    )
    assert policy_dest.is_file()
    assert report_dest.is_file()
    policy = json.loads(policy_dest.read_text(encoding="utf-8"))
    assert policy["interface"] == DETERMINISTIC_REPAIR_POLICY_INTERFACE
    assert policy["mode"] == "auto_safe"
    on_disk = json.loads(report_dest.read_text(encoding="utf-8"))
    assert on_disk["task_id"] == DCR_TASK_ID
    assert on_disk["result"]["passed"] is True
    assert payload["result"]["passed"] is True


def test_default_paths() -> None:
    assert DEFAULT_POLICY_PATH.endswith("deterministic_contract_repair_policy.json")
    assert DEFAULT_CANARY_REPORT_PATH.endswith("canary-report.json")
