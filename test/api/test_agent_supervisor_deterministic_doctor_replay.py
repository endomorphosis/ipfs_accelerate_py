"""LPR-042: replay joined release receipts with identity-equivalent CIDs.

Proves that:

* two full ``validate_deterministic_doctor_release`` runs seal identically;
* ``replay_release_receipt`` recomputes the same receipt_id;
* doctor dual-run report_ids and VFS content identities remain stable;
* forged or mutated receipts fail closed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.validation import (
    deterministic_doctor_release as release,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def first_receipt() -> release.DeterministicDoctorReleaseReceipt:
    return release.validate_deterministic_doctor_release(_REPO_ROOT)


@pytest.fixture(scope="module")
def second_receipt() -> release.DeterministicDoctorReleaseReceipt:
    return release.validate_deterministic_doctor_release(_REPO_ROOT)


def test_dual_full_release_identity_equivalent(
    first_receipt: release.DeterministicDoctorReleaseReceipt,
    second_receipt: release.DeterministicDoctorReleaseReceipt,
) -> None:
    assert first_receipt.valid is True
    assert second_receipt.valid is True
    assert first_receipt.receipt_id == second_receipt.receipt_id
    assert first_receipt.doctor_report_id == second_receipt.doctor_report_id
    assert (
        first_receipt.vfs_equivalence_content_id
        == second_receipt.vfs_equivalence_content_id
    )
    assert first_receipt.two_profile_content_id == second_receipt.two_profile_content_id

    first = first_receipt.to_dict()
    second = second_receipt.to_dict()
    # Check names and statuses must match exactly across dual runs.
    assert set(first["checks"]) == set(second["checks"])
    for name, item in first["checks"].items():
        assert item["status"] == second["checks"][name]["status"], name


def test_replay_release_receipt_round_trip(
    first_receipt: release.DeterministicDoctorReleaseReceipt,
) -> None:
    payload = first_receipt.to_dict()
    replay = release.replay_release_receipt(payload)
    assert replay["valid"] is True
    assert replay["identity_ok"] is True
    assert replay["claimed_receipt_id"] == payload["receipt_id"]
    assert replay["recomputed_receipt_id"] == payload["receipt_id"]
    assert replay["mutation_authorized"] is False
    assert replay["completion_authoritative"] is False

    # Typed receipt path.
    typed_replay = release.replay_release_receipt(first_receipt)
    assert typed_replay["identity_ok"] is True
    assert typed_replay["claimed_receipt_id"] == first_receipt.receipt_id


def test_forged_receipt_fails_closed(
    first_receipt: release.DeterministicDoctorReleaseReceipt,
) -> None:
    payload = first_receipt.to_dict()
    forged = dict(payload)
    forged["doctor_report_id"] = "sha256:" + ("0" * 64)
    assert release.verify_sealed(forged) is False
    replay = release.replay_release_receipt(forged)
    assert replay["identity_ok"] is False
    assert replay["valid"] is False


def test_seal_payload_is_deterministic() -> None:
    body = {
        "task_id": "LPR-042",
        "goal_id": "LPR-G110",
        "valid": True,
        "nested": {"b": 2, "a": 1},
        "list": [3, 1, 2],
    }
    first = release.seal_payload(body)
    second = release.seal_payload(body)
    assert first["receipt_id"] == second["receipt_id"]
    assert release.verify_sealed(first) is True


def test_doctor_report_dual_run_stable_across_release_passes(
    first_receipt: release.DeterministicDoctorReleaseReceipt,
    second_receipt: release.DeterministicDoctorReleaseReceipt,
) -> None:
    first_doctor = first_receipt.checks["doctor_fixture_dual_run"]["evidence"]
    second_doctor = second_receipt.checks["doctor_fixture_dual_run"]["evidence"]
    assert first_doctor["report_id"] == second_doctor["report_id"]
    assert first_doctor["identity_equivalent"] is True
    assert second_doctor["identity_equivalent"] is True
    assert first_doctor["llm_zero"] is True


def test_vfs_content_ids_stable_across_release_passes(
    first_receipt: release.DeterministicDoctorReleaseReceipt,
    second_receipt: release.DeterministicDoctorReleaseReceipt,
) -> None:
    first_vfs = first_receipt.checks["vfs_profiles_dual_run"]["evidence"]
    second_vfs = second_receipt.checks["vfs_profiles_dual_run"]["evidence"]
    assert first_vfs["equivalence_content_id"] == second_vfs["equivalence_content_id"]
    assert first_vfs["conformance_content_id"] == second_vfs["conformance_content_id"]
    assert first_vfs["equivalence_identity_equivalent"] is True
    assert first_vfs["vfs_identity_equivalent"] is True
    assert first_vfs["non_vfs_identity_equivalent"] is True


def test_policy_binding_stable() -> None:
    first = release.default_release_policy().policy_binding_id
    second = release.default_release_policy().policy_binding_id
    assert first == second
    assert first.startswith("sha256:")
