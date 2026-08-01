from __future__ import annotations

import json
from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.cve_security_gate import (
    CVESecurityGateFindingCode,
    SecurityCorrelationFinding,
)
from ipfs_accelerate_py.agent_supervisor.cve_security_receipts import (
    BoundedSecurityDecisionReceipt,
    CVESecurityReceiptError,
    MAX_IDENTIFIERS_PER_FIELD,
    MAX_RECEIPT_UTF8_BYTES,
    SecurityReceiptRole,
    emit_cve_security_decision_receipt,
)
from ipfs_accelerate_py.agent_supervisor.proof.ir_constraint_compiler import (
    CVESecurityEnforcementEvidence,
    CVESecurityEnforcementStage,
)
from test.api.test_agent_supervisor_cve_security_enforcement import (
    _base_admission,
    _gate_result,
)


def _evidence(
    *,
    stage: CVESecurityEnforcementStage = (
        CVESecurityEnforcementStage.POST_GENERATION
    ),
    tree_id: str | None = None,
    parent_evidence_id: str = "evidence:pre-execution",
    authority: str = "authoritative",
    expires_at_ms: int | None = 61_000,
    stale_decision: bool = False,
) -> CVESecurityEnforcementEvidence:
    admission = _base_admission()
    return CVESecurityEnforcementEvidence(
        stage=stage,
        repository_tree_id=tree_id or admission.repository_tree_id,
        gate_result=_gate_result(admission, stale_decision=stale_decision),
        parent_evidence_id=parent_evidence_id,
        authority=authority,
        expires_at_ms=expires_at_ms,
    )


def _receipt(**updates: object) -> BoundedSecurityDecisionReceipt:
    inputs: dict[str, object] = {
        "cve_ids": ("CVE-2024-0002", "CVE-2024-0001"),
        "cwe_ids": ("CWE-89", "CWE-79"),
        "source_cids": ("bafy-source-two", "bafy-source-one"),
        "semantic_roots": {
            "cvefixes_release": "release:sha256:fixture",
            "intent_ir": "intent:sha256:fixture",
        },
    }
    inputs.update(updates)
    return emit_cve_security_decision_receipt(_evidence(), **inputs)


def test_receipt_is_canonical_and_links_bounded_decision_provenance() -> None:
    receipt = _receipt()
    payload = receipt.to_dict()

    assert payload["stage"] == "post_generation"
    assert payload["repository_tree_id"] == _base_admission().repository_tree_id
    assert payload["outcome"] == "pass"
    assert payload["record_role"] == "evidence"
    assert payload["intent_evidence"]["mapping_ids"]
    assert payload["intent_evidence"]["source_ids"]
    assert payload["intent_evidence"]["request_ids"]
    assert payload["intent_evidence"]["decision_ids"]
    assert payload["code_evidence"]["mapping_ids"]
    assert payload["matched_policy_ids"] == ["policy:action:write"]
    assert payload["cve_ids"] == ["CVE-2024-0001", "CVE-2024-0002"]
    assert payload["cwe_ids"] == ["CWE-79", "CWE-89"]
    assert payload["source_cids"] == ["bafy-source-one", "bafy-source-two"]
    assert payload["semantic_roots"]["security_ir"].startswith(
        f"{payload['security_root_artifact_id']}:"
    )
    assert payload["policy_receipt_id"]
    assert payload["gate_result_id"]
    assert payload["enforcement_evidence_id"]
    assert payload["reason_codes"]
    assert len(receipt.canonical_bytes) <= MAX_RECEIPT_UTF8_BYTES
    assert receipt.canonical_bytes == json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()

    permuted = _receipt(
        cve_ids=("CVE-2024-0001", "CVE-2024-0002", "CVE-2024-0001"),
        cwe_ids=("CWE-79", "CWE-89"),
        source_cids=("bafy-source-one", "bafy-source-two"),
        semantic_roots={
            "intent_ir": "intent:sha256:fixture",
            "cvefixes_release": "release:sha256:fixture",
        },
    )
    assert permuted.receipt_id == receipt.receipt_id
    assert permuted.canonical_bytes == receipt.canonical_bytes


def test_round_trip_recomputes_identity_and_rejects_tampering() -> None:
    receipt = _receipt()
    restored = BoundedSecurityDecisionReceipt.from_dict(receipt.to_dict())

    assert restored == receipt
    assert restored.receipt_id == receipt.receipt_id
    assert restored.cache_key == receipt.cache_key

    tampered = receipt.to_dict()
    tampered["repository_tree_id"] = "tree:sha256:tampered"
    with pytest.raises(CVESecurityReceiptError, match="cache key mismatch"):
        BoundedSecurityDecisionReceipt.from_dict(tampered)

    authority_tampered = receipt.to_dict()
    authority_tampered["grants_execution_authority"] = True
    with pytest.raises(
        CVESecurityReceiptError, match="grants_execution_authority mismatch"
    ):
        BoundedSecurityDecisionReceipt.from_dict(authority_tampered)


def test_event_is_redacted_and_omits_request_and_sensitive_details() -> None:
    evidence = _evidence()
    sensitive_marker = "=".join(
        ("password", "must-not-enter-the-observability-event")
    )
    gate = replace(
        evidence.gate_result,
        findings=(
            SecurityCorrelationFinding(
                code=CVESecurityGateFindingCode.BROADENED_CODE_EFFECT,
                intent_mapping_ids=(
                    evidence.gate_result.intent_mappings[0].mapping_id,
                ),
                code_mapping_ids=(
                    evidence.gate_result.code_mappings[0].mapping_id,
                ),
                details={
                    "code_body": "dangerous_generated_code()",
                    "diagnostic": sensitive_marker,
                },
            ),
        ),
    )
    receipt = emit_cve_security_decision_receipt(
        replace(evidence, gate_result=gate),
        cve_ids=("CVE-2024-0001",),
        cwe_ids=("CWE-79",),
        source_cids=("bafy-source",),
    )
    encoded = receipt.canonical_bytes.decode()
    event = receipt.to_event_fields()

    assert event["redacted"] is True
    assert event["contains_code_body"] is False
    assert event["contains_secrets"] is False
    assert event["counterexamples"][0]["details_redacted"] is True
    assert "correlation_finding.details" in event["redacted_fields"]
    assert "dangerous_generated_code" not in encoded
    assert sensitive_marker not in encoded
    assert '"current_state":' not in encoded
    assert '"expected_effect":' not in encoded
    assert '"data_flow":' not in encoded


def test_receipt_distinguishes_trusted_evidence_from_execution_authority() -> None:
    receipt = emit_cve_security_decision_receipt(
        _evidence(authority="verified_input")
    )
    payload = receipt.to_dict()

    assert receipt.record_role is SecurityReceiptRole.EVIDENCE
    assert receipt.evidence_authority == "verified_input"
    assert receipt.grants_execution_authority is False
    assert receipt.authorizes_completion is False
    assert payload["evidence_is_authority"] is False
    assert payload["grants_execution_authority"] is False
    assert payload["authorizes_completion"] is False


def test_cache_key_invalidates_on_every_declared_dependency_class() -> None:
    baseline = _receipt()
    variants = (
        emit_cve_security_decision_receipt(
            _evidence(stage=CVESecurityEnforcementStage.MERGE_ADMISSION)
        ),
        emit_cve_security_decision_receipt(
            _evidence(tree_id="tree:sha256:merged")
        ),
        emit_cve_security_decision_receipt(
            _evidence(parent_evidence_id="evidence:changed-parent")
        ),
        emit_cve_security_decision_receipt(
            _evidence(authority="verified")
        ),
        emit_cve_security_decision_receipt(
            _evidence(expires_at_ms=62_000)
        ),
        _receipt(cve_ids=("CVE-2025-9999",)),
        _receipt(cwe_ids=("CWE-22",)),
        _receipt(source_cids=("bafy-changed-source",)),
        _receipt(
            semantic_roots={
                "cvefixes_release": "release:sha256:changed",
                "intent_ir": "intent:sha256:fixture",
            }
        ),
    )

    dependency_fields = set(baseline.declared_dependencies)
    assert {
        "stage",
        "repository_tree_id",
        "outcome",
        "security_roots",
        "policy_receipt_id",
        "gate_result_id",
        "enforcement_evidence_id",
        "parent_evidence_id",
        "evidence_authority",
        "evaluated_at_ms",
        "expires_at_ms",
        "intent_evidence",
        "code_evidence",
        "matched_policy_ids",
        "cve_ids",
        "cwe_ids",
        "source_cids",
        "reason_codes",
        "counterexample_ids",
    } == dependency_fields
    assert all(item.cache_key != baseline.cache_key for item in variants)


def test_bounds_and_sensitive_value_checks_fail_closed() -> None:
    with pytest.raises(CVESecurityReceiptError, match="item input bound"):
        emit_cve_security_decision_receipt(
            _evidence(),
            cve_ids=tuple(
                f"CVE-2024-{index:04d}"
                for index in range(MAX_IDENTIFIERS_PER_FIELD + 1)
            ),
        )

    sensitive_source = "=".join(("api_key", "should-never-be-logged"))
    with pytest.raises(CVESecurityReceiptError, match="credential or secret"):
        emit_cve_security_decision_receipt(
            _evidence(),
            source_cids=(sensitive_source,),
        )

    with pytest.raises(CVESecurityReceiptError, match="UTF-8 bytes"):
        emit_cve_security_decision_receipt(
            _evidence(),
            source_cids=("x" * 1_025,),
        )


def test_detached_decision_and_conflicting_security_root_fail_closed() -> None:
    with pytest.raises(CVESecurityReceiptError, match="timestamp is detached"):
        emit_cve_security_decision_receipt(_evidence(stale_decision=True))

    with pytest.raises(CVESecurityReceiptError, match="security_ir root differs"):
        emit_cve_security_decision_receipt(
            _evidence(),
            semantic_roots={"security_ir": "security:sha256:other"},
        )
