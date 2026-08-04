"""Tests for closeout activation-gap probe (PTR-149 handoff)."""

from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.validation.proof_test_reuse_closeout_activation_probe import (
    ACTIVATION_PROBE_SCHEMA,
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


def test_activation_probe_reports_gap_without_granting_authority() -> None:
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
    # Activation gap remains present: reviewed authority still absent.
    assert claims["activation_gap"].proven is True
    assert claims["activation_e2e_passed"].proven is False
    # Ordinary default composition may be usable (mode+cache_root) but must not
    # prove zero-injection production path while the gap is present.
    assert claims["zero_injection_default_path"].proven is False
    # Exact reviewed production identity pins remain operator-owned.
    assert claims["exact_reviewed_source_binary_capability_circuit_key_identities"].proven is False
    assert report.remaining_operator_actions
    assert any(
        "reviewed" in action.lower()
        or "identity" in action.lower()
        or "v4" in action.lower()
        or "gap" in action.lower()
        or "warm-skip" in action.lower()
        for action in report.remaining_operator_actions
    )
    # Live composition probe should not invent identity unconfigured when the
    # factory path can inject services under SHADOW + cache_root.
    live = report.live_report
    blockers = set(live.get("activation_blocker_codes") or ())
    # Identity wiring is no longer a hard ambient blocker of the live probe path.
    # Remaining blockers should center on reviewed certificate authority / gap.
    if live.get("ordinary_default_composition_usable"):
        assert "identity_services_unconfigured" not in blockers
        assert "candidate_store_unconfigured" not in blockers
        assert claims["zero_injection_default_path"].observed is True
    assert live.get("activation_gap_present") is True


def test_activation_probe_to_dict_is_json_serializable() -> None:
    import json

    report = produce_closeout_activation_probe(
        _identity(),
        objective_completion_tree_id="baguqeera-completion",
        now_ms=1_800_000_000_000,
    )
    payload = report.to_dict()
    encoded = json.dumps(payload)
    assert "activation_gap_present" in encoded
    assert payload["repair_evidence_summary"]["activation_gap"] is True
