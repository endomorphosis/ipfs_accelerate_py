from __future__ import annotations

import json
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.adversarial import (
    MUTANT_FAMILIES,
    ResidualAdversarialAdapter,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.contracts import (
    ExpertDisposition,
    ResidualIntelligenceError,
)

REPORT = (
    Path(__file__).resolve().parents[3]
    / "docs/architecture/residual_intelligence_inventory/adversarial_campaign_report.json"
)


def receipt_for(family: str, *, escaped: bool = False) -> dict[str, object]:
    disposition = ExpertDisposition.VALIDATION_REQUIRED.value
    evidence: dict[str, object] = {}
    if family == "abstention":
        disposition = ExpertDisposition.ABSTAIN.value
        evidence["false_nonabstention_detected"] = True
    elif family in {"injection", "leakage", "privacy", "authority", "completion"}:
        disposition = ExpertDisposition.REJECT_INPUT.value
        evidence[
            {
                "injection": "prompt_injection_detected",
                "leakage": "leakage_blocked",
                "privacy": "privacy_boundary_preserved",
                "authority": "authority_not_granted",
                "completion": "completion_not_granted",
            }[family]
        ] = True
    elif family in {"staleness", "quantization", "disagreement"}:
        evidence[
            {
                "staleness": "stale_input_rejected",
                "quantization": "quantization_regression_detected",
                "disagreement": "teacher_disagreement_escalated",
            }[family]
        ] = True
    return {
        "family": family,
        "escaped": escaped,
        "receipt_id": f"receipt:{family}",
        "observed_disposition": disposition,
        "evidence": evidence,
    }


def receipts(*, escaped: bool = False):
    return [receipt_for(family, escaped=escaped) for family in MUTANT_FAMILIES]


def test_campaign_covers_all_families_and_rejects_escapes() -> None:
    campaign = ResidualAdversarialAdapter().run("tree:final", receipts())
    assert campaign.critical_zero_escape is True
    assert campaign.completion_authoritative is False
    assert {item.family for item in campaign.results} == set(MUTANT_FAMILIES)
    with pytest.raises(ResidualIntelligenceError, match="escaped"):
        ResidualAdversarialAdapter().run("tree:final", receipts(escaped=True))
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    assert payload["critical_zero_escape"] is True
    assert payload["mutant_families"] == list(MUTANT_FAMILIES)
    assert payload["completion_authoritative"] is False


@pytest.mark.parametrize(
    ("family", "disposition", "message"),
    [
        ("abstention", "ACCEPT", "unsafe disposition"),
        ("injection", "VALIDATION_REQUIRED", "unsafe disposition"),
        ("staleness", "ABSTAIN", "unsafe disposition"),
        ("quantization", "ACCEPT", "unsafe disposition"),
        ("disagreement", "ACCEPT", "unsafe disposition"),
        ("authority", "ACCEPT", "unsafe disposition"),
        ("completion", "ACCEPT", "unsafe disposition"),
    ],
)
def test_critical_boundaries_fail_closed_on_unsafe_disposition(
    family: str, disposition: str, message: str
) -> None:
    receipt = receipt_for(family)
    receipt["observed_disposition"] = disposition
    with pytest.raises(ResidualIntelligenceError, match=message):
        ResidualAdversarialAdapter().run("tree:final", [receipt])


@pytest.mark.parametrize(
    ("family", "evidence_key"),
    [
        ("abstention", "false_nonabstention_detected"),
        ("injection", "prompt_injection_detected"),
        ("leakage", "leakage_blocked"),
        ("staleness", "stale_input_rejected"),
        ("quantization", "quantization_regression_detected"),
        ("disagreement", "teacher_disagreement_escalated"),
    ],
)
def test_sensitive_mutants_require_positive_safety_evidence(
    family: str, evidence_key: str
) -> None:
    receipt = receipt_for(family)
    receipt["evidence"] = {evidence_key: False}
    with pytest.raises(ResidualIntelligenceError, match="lacks required evidence"):
        ResidualAdversarialAdapter().run("tree:final", [receipt])


def test_receipts_are_closed_and_do_not_coerce_boolean_or_duplicate_families() -> None:
    malformed = receipt_for("family")
    malformed["escaped"] = "false"
    with pytest.raises(ResidualIntelligenceError, match="boolean"):
        ResidualAdversarialAdapter().run("tree:final", [malformed])
    unknown = receipt_for("family")
    unknown["completion"] = False
    with pytest.raises(ResidualIntelligenceError, match="unknown fields"):
        ResidualAdversarialAdapter().run("tree:final", [unknown])
    duplicate = receipts()
    duplicate[-1] = receipt_for("family")
    with pytest.raises(ResidualIntelligenceError, match="duplicated"):
        ResidualAdversarialAdapter().run("tree:final", duplicate)


def test_adapter_routes_execution_to_existing_authority() -> None:
    class FakeAssurance:
        def __init__(self) -> None:
            self.calls: list[tuple[object, object, dict[str, object]]] = []

        def execute_mutation_campaign(self, plan, verification_policy, **kwargs):
            self.calls.append((plan, verification_policy, kwargs))
            return {"candidate_reports": receipts()}

    authority = FakeAssurance()
    campaign = ResidualAdversarialAdapter(authority).run_authority_campaign(
        "tree:final", {"plan": "bound"}, {"policy": "bound"}, metadata={"scope": "test"}
    )
    assert campaign.critical_zero_escape is True
    assert authority.calls == [
        ({"plan": "bound"}, {"policy": "bound"}, {"metadata": {"scope": "test"}})
    ]
