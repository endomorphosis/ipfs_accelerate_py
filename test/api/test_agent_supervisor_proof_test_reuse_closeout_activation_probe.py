"""Tests for closeout activation-gap probe (PTR-149 handoff)."""

from __future__ import annotations

import os

import pytest

from ipfs_accelerate_py.agent_supervisor.validation.proof_test_reuse_closeout_activation_probe import (
    ACTIVATION_PROBE_SCHEMA,
    OPERATOR_HANDOFF_SCHEMA,
    build_activation_gap_operator_handoff,
    produce_closeout_activation_probe,
)
from ipfs_accelerate_py.agent_supervisor.validation.proof_test_reuse_closeout_materializer import (
    CloseoutMaterializerIdentity,
)


def _identity() -> CloseoutMaterializerIdentity:
    return CloseoutMaterializerIdentity(
        repository_id="lift_coding/proof-backed-test-reuse",
        repository_state_cid="git-commit:" + ("a" * 40),
        git_commit_id="a" * 40,
        git_tree_id="b" * 40,
        gitlink_state_cid="baguqeera-gitlinks",
        repository_forest_cid="baguqeera-forest",
        dirty=False,
        dirty_overlay_cid="cid:dirty-overlay:none",
        objective_revision="objective-sha256:test",
        policy_cid="policy:test",
        capability_cid="capability:test",
        verifying_key_cid="key:test",
        circuit_cid="circuit:test",
    )


@pytest.fixture
def clear_closeout_e2e_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate unit tests from ambient local e2e / setup env vars."""

    for name in (
        "PTR_CLOSEOUT_LOCAL_SETUP",
        "PTR_CLOSEOUT_DEV_E2E",
        "PTR_CLOSEOUT_HEAVY_MEASUREMENTS",
        "IPFS_TEST_PROOF_REUSE_GROTH16_ARTIFACT_MANIFEST",
        "IPFS_TEST_PROOF_REUSE_GROTH16_ARTIFACT_MANIFEST_SHA256",
    ):
        monkeypatch.delenv(name, raising=False)


def test_activation_probe_reports_gap_without_granting_authority(
    clear_closeout_e2e_env: None,
) -> None:
    receipts = [
        {
            "task_id": "PTR-012",
            "passed": True,
            "skipped_count": 0,
            "git_commit_id": "a" * 40,
            "proof_reuse_mode": "off",
        }
    ]
    report = produce_closeout_activation_probe(
        _identity(),
        objective_completion_tree_id="baguqeera-completion",
        validation_receipts=receipts,
        supervisor_healthy=True,
        now_ms=1_800_000_000_000,
        attempt_heavy_measurements=False,
    )
    assert report.schema == ACTIVATION_PROBE_SCHEMA
    assert report.authority is False
    assert report.activation_gap_present is True
    repair = report.repair_evidence
    assert repair.get("activation_gap") is True
    assert repair.get("passed") is not True
    assert repair.get("authority") in {"none", "authoritative"}
    # Zero false-skip can be proven from MODE=off receipts even while gap present.
    claims = {item.field: item for item in report.claims}
    assert claims["zero_false_skip_assurance"].proven is True
    assert claims["historical_activation_claims_superseded"].proven is True
    assert claims["supervisor_healthy"].proven is True
    # Without local e2e pin env, activation gap remains present.
    assert claims["activation_gap"].proven is True
    assert claims["activation_e2e_passed"].proven is False
    assert claims["zero_injection_default_path"].proven is False
    assert claims["exact_reviewed_source_binary_capability_circuit_key_identities"].proven is False
    assert report.remaining_operator_actions
    live = report.live_report
    blockers = set(live.get("activation_blocker_codes") or ())
    if live.get("ordinary_default_composition_usable"):
        assert "identity_services_unconfigured" not in blockers
        assert "candidate_store_unconfigured" not in blockers
        assert claims["zero_injection_default_path"].observed is True
    assert live.get("activation_gap_present") is True


def test_activation_probe_to_dict_is_json_serializable(
    clear_closeout_e2e_env: None,
) -> None:
    import json

    report = produce_closeout_activation_probe(
        _identity(),
        objective_completion_tree_id="baguqeera-completion",
        now_ms=1_800_000_000_000,
        attempt_heavy_measurements=False,
    )
    payload = report.to_dict()
    encoded = json.dumps(payload)
    assert "activation_gap_present" in encoded
    assert payload["repair_evidence_summary"]["activation_gap"] is True


def test_operator_handoff_lists_ceremony_steps_without_authority(
    clear_closeout_e2e_env: None,
) -> None:
    import json

    identity = _identity()
    report = produce_closeout_activation_probe(
        identity,
        objective_completion_tree_id="baguqeera-completion",
        now_ms=1_800_000_000_000,
        attempt_heavy_measurements=False,
    )
    handoff = build_activation_gap_operator_handoff(
        report, identity=identity, now_ms=1_800_000_000_000
    )
    assert handoff["schema"] == OPERATOR_HANDOFF_SCHEMA
    assert handoff["authority"] is False
    assert handoff["warm_skip_authorized"] is False
    assert handoff["closeout_authorized"] is False
    assert handoff["activation_gap_present"] is True
    step_ids = [step["id"] for step in handoff["ceremony_steps"]]
    assert "reviewed_v4_allowlist" in step_ids
    assert "manifest_env_pin" in step_ids
    assert "current_tree_controller_context" in step_ids
    allowlist = next(
        step for step in handoff["ceremony_steps"] if step["id"] == "reviewed_v4_allowlist"
    )
    # Dev branch may carry a local allowlist digest; status is then ready_to_pin.
    assert allowlist["status"] in {"blocked", "ready_to_pin"}
    json.dumps(handoff)
    assert handoff["identity"]["git_tree_id"] == identity.git_tree_id


def test_local_dev_e2e_can_close_activation_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With local keys + allowlisted manifest pin, gap can clear on this branch."""

    monkeypatch.setenv("PTR_CLOSEOUT_LOCAL_SETUP", "1")
    monkeypatch.setenv("PTR_CLOSEOUT_DEV_E2E", "1")
    monkeypatch.delenv("PTR_CLOSEOUT_HEAVY_MEASUREMENTS", raising=False)
    report = produce_closeout_activation_probe(
        _identity(),
        objective_completion_tree_id="baguqeera-completion",
        validation_receipts=[
            {
                "task_id": "PTR-012",
                "passed": True,
                "skipped_count": 0,
                "git_commit_id": "a" * 40,
                "proof_reuse_mode": "off",
            }
        ],
        supervisor_healthy=True,
        now_ms=1_800_000_000_000,
        attempt_heavy_measurements=False,
    )
    live = report.live_report
    # If local keys/manifest are present on this tree, gap should clear.
    # Ambient CI without keys still fails closed.
    if live.get("test_certificate_authority_ready"):
        assert report.activation_gap_present is False
        assert live.get("ordinary_default_composition_usable") is True
        # Still never invents full production closeout without remaining claims.
        assert report.repair_evidence.get("passed") is not True or report.authority is False
    else:
        assert report.activation_gap_present is True
