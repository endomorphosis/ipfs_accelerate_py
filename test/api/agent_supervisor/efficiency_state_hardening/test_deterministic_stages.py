"""ASEH-022: identity-bound deterministic ladder evidence and receipts."""

from __future__ import annotations

from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import HarnessError
from ipfs_accelerate_py.agent_supervisor.semantic_state.deterministic_stages import (
    DETERMINISTIC_RECEIPT_SCHEMA,
    IDENTITY_FIELDS,
    evaluate_deterministic_stages,
    evaluate_exact_receipt_freshness,
    evaluate_impact_analysis,
    evaluate_incremental_proof,
    evaluate_selected_tests,
    evaluate_static_contract_checks,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.routing import (
    DeterministicEvidenceState,
    LadderStage,
    evaluate_identity_bound_deterministic_ladder,
)


def _identity(**overrides: str) -> dict[str, str]:
    result = {field: f"id:{field}" for field in IDENTITY_FIELDS}
    result.update(overrides)
    return result


def _cache(**overrides: Any) -> dict[str, Any]:
    identity = _identity()
    result: dict[str, Any] = {
        "receipt_available": True,
        "receipt_admitted": True,
        "receipt_identity": identity,
        "current_identity": dict(identity),
        "receipt_lease_id": "lease:7",
        "current_lease_id": "lease:7",
        "receipt_fence": 4,
        "current_fence": 4,
        "policy_expected_id": "id:policy_id",
        "policy_current_id": "id:policy_id",
    }
    result.update(overrides)
    return result


def test_known_cache_is_exact_reusable_and_receipt_is_stable() -> None:
    first = evaluate_exact_receipt_freshness(_cache())
    second = evaluate_exact_receipt_freshness(_cache())
    assert first.disposition == DeterministicEvidenceState.RESOLVES.value
    assert first.reason_codes == ("exact_authoritative_receipt_fresh",)
    assert first.receipt_id == second.receipt_id
    assert first.to_dict()["schema"] == DETERMINISTIC_RECEIPT_SCHEMA


@pytest.mark.parametrize(
    ("overrides", "reason"),
    (
        ({"receipt_identity": _identity(tree_id="old-tree")}, "identity_mismatch:tree_id"),
        ({"receipt_lease_id": "lease:old"}, "stale_lease_or_fence"),
        ({"receipt_fence": 3}, "stale_lease_or_fence"),
        ({"policy_current_id": "policy:new"}, "policy_pointer_cas_mismatch"),
    ),
)
def test_stale_tree_lease_fence_and_policy_cas_reject_cache_deterministically(
    overrides: dict[str, Any], reason: str
) -> None:
    receipt = evaluate_exact_receipt_freshness(_cache(**overrides))
    assert receipt.disposition == DeterministicEvidenceState.RESOLVES.value
    assert reason in receipt.reason_codes


def test_documentation_only_and_dependency_impact_cases_are_deterministic() -> None:
    docs = evaluate_impact_analysis(
        {
            "changed_paths": ["docs/guide.md"],
            "affected_symbols": [],
            "dependency_cone_complete": True,
            "impact_safe": False,
        }
    )
    dependency = evaluate_impact_analysis(
        {
            "changed_paths": ["pkg/worker.py"],
            "affected_symbols": ["pkg.worker.run"],
            "dependency_cone_complete": True,
            "impact_safe": True,
        }
    )
    incomplete = evaluate_impact_analysis(
        {
            "changed_paths": ["pkg/worker.py"],
            "affected_symbols": ["pkg.worker.run"],
            "dependency_cone_complete": False,
            "impact_safe": False,
        }
    )
    assert docs.reason_codes == ("documentation_only_no_symbol_impact",)
    assert dependency.reason_codes == ("complete_dependency_impact_safe",)
    assert incomplete.disposition == DeterministicEvidenceState.UNRESOLVED.value


def test_schema_type_static_lint_contract_and_policy_checks_resolve() -> None:
    passed = evaluate_static_contract_checks(
        {
            "schema_valid": True,
            "type_valid": True,
            "static_valid": True,
            "lint_valid": True,
            "contracts_valid": True,
            "policy_cas_current": True,
        }
    )
    failed = evaluate_static_contract_checks(
        {
            "schema_valid": False,
            "type_valid": True,
            "static_valid": True,
            "lint_valid": True,
            "contracts_valid": True,
            "policy_cas_current": False,
        }
    )
    assert passed.reason_codes == ("schema_type_static_lint_contracts_passed",)
    assert set(failed.reason_codes) == {
        "check_failed:policy_cas_current",
        "check_failed:schema_valid",
    }


def test_known_test_selection_is_closed_and_bad_selection_is_rejected() -> None:
    passed = evaluate_selected_tests(
        {
            "known_tests": ["test_a", "test_b"],
            "selected_tests": ["test_b"],
            "selection_complete": True,
            "selected_tests_passed": True,
        }
    )
    unknown = evaluate_selected_tests(
        {
            "known_tests": ["test_a"],
            "selected_tests": ["test_unknown"],
            "selection_complete": True,
            "selected_tests_passed": True,
        }
    )
    assert passed.reason_codes == ("known_selected_tests_passed",)
    assert unknown.reason_codes == ("unknown_selected_test:test_unknown",)


def test_incremental_proof_requires_full_identity_match() -> None:
    identity = _identity()
    passed = evaluate_incremental_proof(
        {
            "proof_available": True,
            "proof_identity": identity,
            "current_identity": dict(identity),
            "proof_passed": True,
        }
    )
    mismatched = evaluate_incremental_proof(
        {
            "proof_available": True,
            "proof_identity": _identity(environment_id="old-environment"),
            "current_identity": dict(identity),
            "proof_passed": True,
        }
    )
    assert passed.reason_codes == ("incremental_proof_identity_matched",)
    assert mismatched.reason_codes == ("proof_identity_mismatch:environment_id",)


def test_stage_decision_projects_receipts_to_the_existing_canonical_ladder() -> None:
    decision = evaluate_deterministic_stages(
        receipt_freshness={**_cache(), "receipt_available": False},
        impact={
            "changed_paths": ["pkg/worker.py"],
            "affected_symbols": ["pkg.worker.run"],
            "dependency_cone_complete": False,
            "impact_safe": False,
        },
        static={
            "schema_valid": True,
            "type_valid": True,
            "static_valid": True,
            "lint_valid": True,
            "contracts_valid": True,
            "policy_cas_current": True,
        },
    )
    ladder = evaluate_identity_bound_deterministic_ladder(
        receipt_freshness={**_cache(), "receipt_available": False},
        impact={
            "changed_paths": ["pkg/worker.py"],
            "affected_symbols": ["pkg.worker.run"],
            "dependency_cone_complete": False,
            "impact_safe": False,
        },
        static={
            "schema_valid": True,
            "type_valid": True,
            "static_valid": True,
            "lint_valid": True,
            "contracts_valid": True,
            "policy_cas_current": True,
        },
    )
    assert len(decision.receipt_ids) == 5
    assert decision.evidence.schema_type_static == DeterministicEvidenceState.RESOLVES.value
    assert ladder.selected_stage == LadderStage.SCHEMA_TYPE_STATIC_LINT_AND_CONTRACT_CHECKS.value


def test_closed_inputs_and_receipt_integrity_fail_closed() -> None:
    payload = _cache()
    payload["frontier_model_available"] = True
    with pytest.raises(HarnessError, match="fields must be exactly"):
        evaluate_exact_receipt_freshness(payload)
    receipt = evaluate_exact_receipt_freshness(_cache()).to_dict()
    from ipfs_accelerate_py.agent_supervisor.semantic_state.deterministic_stages import (
        DeterministicStageReceipt,
    )

    assert DeterministicStageReceipt.from_dict(receipt).to_dict() == receipt
    receipt["receipt_id"] = "sha256:bad"
    with pytest.raises(HarnessError, match="does not rehash"):
        # Constructing through the public decision is not needed; the receipt
        # object itself protects its digest-bound identity.
        DeterministicStageReceipt(
            stage=receipt["stage"],
            disposition=receipt["disposition"],
            decisive=receipt["decisive"],
            reason_codes=tuple(receipt["reason_codes"]),
            subject=receipt["subject"],
            receipt_id=receipt["receipt_id"],
        )
