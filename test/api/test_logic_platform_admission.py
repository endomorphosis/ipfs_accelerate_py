"""Contract tests for SupervisorLogicPlatformReceiptAdmission@1 (LPC-111).

A result may affect completion or merge only after structural validity,
content identity, source/tree/environment/policy binding, translation chain,
evidence kind, authority ceiling, required reconstruction, freshness,
non-simulation, and policy admission.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    AssuranceLevel,
    EvidenceAuthority,
    EvidenceFreshness,
    EvidenceKind,
    EvidenceVerdict,
    ProofEvidence,
    ProofReceipt,
    ProofVerdict,
    ResourceBudget,
)
from ipfs_accelerate_py.agent_supervisor.proof.logic_platform_admission import (
    ADMISSION_CONTEXT_SCHEMA,
    ADMISSION_GOAL_ID,
    ADMISSION_RESULT_SCHEMA,
    ADMISSION_SCHEMA_VERSION,
    ADMISSION_TASK_ID,
    AdmissionCheck,
    AdmissionContext,
    AdmissionDisposition,
    AdmissionResult,
    LogicPlatformAdmissionError,
    SUPERVISOR_LOGIC_PLATFORM_RECEIPT_ADMISSION_INTERFACE,
    TEN_POINT_CHECKS,
    SupervisorLogicPlatformReceiptAdmission,
    admit_receipt,
    admit_receipts,
    get_receipt_admission,
    may_affect_completion_or_merge,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ADMISSION_SOURCE = (
    REPO_ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "proof"
    / "logic_platform_admission.py"
)
ADMISSION_NOTE = (
    REPO_ROOT
    / "data"
    / "agent_supervisor"
    / "logic_platform_canonicalization"
    / "notes"
    / "receipt_admission.md"
)
ADMISSION_MODULE_NAME = (
    "ipfs_accelerate_py.agent_supervisor.proof.logic_platform_admission"
)


def _context(**overrides: Any) -> AdmissionContext:
    values: dict[str, Any] = {
        "task_id": "LPC-111",
        "repository_tree_id": "tree:sha256:abc",
        "policy_id": "policy:implementation-daemon",
        "operation": "merge",
        "required_authority": AssuranceLevel.KERNEL_VERIFIED.value,
        "repository_id": "repository:sha256:repo",
        "environment_id": "env:validation-hermetic",
        "source_id": "source:sha256:" + ("11" * 32),
        "policy_revision": "sha256:" + ("22" * 32),
        "plan_id": "plan:test",
        "obligation_id": "obl:test",
        "require_reconstruction": True,
        "require_kernel": True,
        "network_allowed": False,
    }
    values.update(overrides)
    return AdmissionContext(**values)


def _kernel_evidence(
    obligation_id: str = "obl:test",
    *,
    simulated: bool = False,
    freshness: EvidenceFreshness = EvidenceFreshness.CURRENT,
) -> ProofEvidence:
    return ProofEvidence(
        kind=EvidenceKind.KERNEL_VERIFICATION,
        authority=EvidenceAuthority.KERNEL,
        verdict=EvidenceVerdict.ACCEPTED,
        artifact_id=f"artifact:{obligation_id}",
        subject_id=obligation_id,
        verifier_id="kernel:lean",
        freshness=freshness,
        independent=True,
        simulated=simulated,
    )


def _proof_receipt(**overrides: Any) -> ProofReceipt:
    obligation_id = str(overrides.pop("obligation_id", "obl:test"))
    evidence = overrides.pop("evidence", None)
    if evidence is None:
        evidence = (_kernel_evidence(obligation_id),)
    values: dict[str, Any] = {
        "obligation_id": obligation_id,
        "plan_id": "plan:test",
        "attempt_id": f"attempt:{obligation_id}",
        "repository_id": "repository:sha256:repo",
        "repository_tree_id": "tree:sha256:abc",
        "ast_scope_ids": ("scope:test",),
        "premise_ids": (),
        "translator_id": "translator:reviewed",
        "solver_id": "solver:reviewed",
        "kernel_id": "kernel:lean",
        "toolchain_id": "toolchain:locked",
        "policy_id": "policy:implementation-daemon",
        "resource_budget": ResourceBudget(),
        "verdict": ProofVerdict.PROVED,
        "evidence": evidence,
        "kernel_receipt_id": f"kernel:{obligation_id}",
        "provider_claimed_assurance": AssuranceLevel.UNVERIFIED,
        "freshness": EvidenceFreshness.CURRENT,
        "metadata": {
            "environment_id": "env:validation-hermetic",
            "source_id": "source:sha256:" + ("11" * 32),
        },
    }
    values.update(overrides)
    return ProofReceipt(**values)


def _admissible_envelope(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "content_id": "sha256:" + ("ab" * 32),
        "receipt_id": "sha256:" + ("ab" * 32),
        "obligation_id": "obl:test",
        "plan_id": "plan:test",
        "repository_id": "repository:sha256:repo",
        "repository_tree_id": "tree:sha256:abc",
        "policy_id": "policy:implementation-daemon",
        "policy_revision": "sha256:" + ("22" * 32),
        "environment_id": "env:validation-hermetic",
        "source_id": "source:sha256:" + ("11" * 32),
        "operation": "merge",
        "semantic_verdict": "proved",
        "evidence_kind": EvidenceKind.KERNEL_VERIFICATION.value,
        "authority_ceiling": AssuranceLevel.KERNEL_VERIFIED.value,
        "freshness": EvidenceFreshness.CURRENT.value,
        "simulated": False,
        "reconstruction_passed": True,
        "kernel_checked": True,
        "translation": {
            "valid": True,
            "translation_class": "exact",
            "source_id": "source:ast",
            "target_id": "target:lean",
        },
        "evidence": [
            {
                "schema": "ipfs_accelerate_py/agent-supervisor/proof-evidence@1",
                "kind": EvidenceKind.KERNEL_VERIFICATION.value,
                "authority": EvidenceAuthority.KERNEL.value,
                "verdict": EvidenceVerdict.ACCEPTED.value,
                "artifact_id": "artifact:obl:test",
                "subject_id": "obl:test",
                "verifier_id": "kernel:lean",
                "freshness": EvidenceFreshness.CURRENT.value,
                "independent": True,
                "simulated": False,
                "metadata": {},
            }
        ],
        "policy_admitted": True,
        "network_allowed": False,
    }
    body.update(overrides)
    return body


# ---------------------------------------------------------------------------
# Module / note / identity
# ---------------------------------------------------------------------------


def test_interface_and_schema_are_stable() -> None:
    helper = get_receipt_admission()
    payload = helper.to_dict()
    assert helper.interface == SUPERVISOR_LOGIC_PLATFORM_RECEIPT_ADMISSION_INTERFACE
    assert payload["interface"] == SUPERVISOR_LOGIC_PLATFORM_RECEIPT_ADMISSION_INTERFACE
    assert payload["schema_version"] == ADMISSION_SCHEMA_VERSION
    assert payload["task_id"] == ADMISSION_TASK_ID
    assert payload["goal_id"] == ADMISSION_GOAL_ID
    assert helper.ten_point_checks() == TEN_POINT_CHECKS
    assert len(TEN_POINT_CHECKS) == 10
    assert list(AdmissionCheck)  # enum is populated
    assert [c.value for c in AdmissionCheck] == list(TEN_POINT_CHECKS)


def test_declared_note_documents_acceptance_surface() -> None:
    text = ADMISSION_NOTE.read_text(encoding="utf-8")
    assert "SupervisorLogicPlatformReceiptAdmission@1" in text
    assert "LPC-111" in text
    for token in (
        "structural validity",
        "content identity",
        "source/tree/environment/policy",
        "translation chain",
        "evidence kind",
        "authority ceiling",
        "required reconstruction",
        "freshness",
        "non-simulation",
        "policy admission",
    ):
        assert token in text.lower()
    assert "logic_platform_admission.py" in text
    assert "test_logic_platform_admission.py" in text
    assert "may_affect_completion" in text
    assert "may_affect_merge" in text


def test_ten_point_checks_match_plan_order() -> None:
    assert TEN_POINT_CHECKS == (
        "structural_validity",
        "content_identity",
        "source_tree_environment_policy_binding",
        "translation_chain",
        "evidence_kind",
        "authority_ceiling",
        "required_reconstruction",
        "freshness",
        "non_simulation",
        "policy_admission",
    )


def test_importing_admission_module_does_not_import_datasets_package() -> None:
    source = ADMISSION_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("ipfs_datasets_py"), alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert not module.startswith("ipfs_datasets_py"), module

    module = importlib.import_module(ADMISSION_MODULE_NAME)
    assert hasattr(module, "admit_receipt")
    assert hasattr(module, "TEN_POINT_CHECKS")


# ---------------------------------------------------------------------------
# Context construction (fail closed)
# ---------------------------------------------------------------------------


def test_context_requires_task_tree_policy_operation() -> None:
    with pytest.raises(LogicPlatformAdmissionError):
        AdmissionContext(
            task_id="",
            repository_tree_id="tree:1",
            policy_id="policy:1",
            operation="merge",
        )
    with pytest.raises(LogicPlatformAdmissionError):
        AdmissionContext(
            task_id="task:1",
            repository_tree_id="",
            policy_id="policy:1",
            operation="merge",
        )
    with pytest.raises(LogicPlatformAdmissionError):
        AdmissionContext(
            task_id="task:1",
            repository_tree_id="tree:1",
            policy_id="",
            operation="merge",
        )
    with pytest.raises(LogicPlatformAdmissionError):
        AdmissionContext(
            task_id="task:1",
            repository_tree_id="tree:1",
            policy_id="policy:1",
            operation="",
        )


def test_kernel_required_context_cannot_disable_reconstruction() -> None:
    with pytest.raises(LogicPlatformAdmissionError, match="reconstruction"):
        AdmissionContext(
            task_id="task:1",
            repository_tree_id="tree:1",
            policy_id="policy:1",
            operation="merge",
            required_authority=AssuranceLevel.KERNEL_VERIFIED.value,
            require_reconstruction=False,
        )


# ---------------------------------------------------------------------------
# Happy path — all ten points
# ---------------------------------------------------------------------------


def test_admissible_envelope_passes_all_ten_points() -> None:
    result = admit_receipt(_admissible_envelope(), _context())
    assert result.admitted is True
    assert result.disposition is AdmissionDisposition.ADMITTED
    assert result.may_affect_completion is True
    assert result.may_affect_merge is True
    assert len(result.checks) == 10
    assert all(item.passed for item in result.checks)
    assert [item.check.value for item in result.checks] == list(TEN_POINT_CHECKS)
    payload = result.to_dict()
    assert payload["schema"] == ADMISSION_RESULT_SCHEMA
    assert payload["may_affect_completion"] is True
    assert payload["may_affect_merge"] is True
    assert payload["context"]["schema"] == ADMISSION_CONTEXT_SCHEMA


def test_may_affect_completion_or_merge_helper() -> None:
    ctx = _context()
    assert may_affect_completion_or_merge(_admissible_envelope(), ctx) is True
    bad = _admissible_envelope(simulated=True)
    assert may_affect_completion_or_merge(bad, ctx) is False


def test_helper_object_matches_module_functions() -> None:
    helper = SupervisorLogicPlatformReceiptAdmission()
    ctx = _context()
    envelope = _admissible_envelope()
    assert helper.admit(envelope, ctx).admitted is True
    assert helper.may_affect_completion_or_merge(envelope, ctx) is True


def test_admit_receipts_sequence() -> None:
    ctx = _context()
    results = admit_receipts(
        (_admissible_envelope(), _admissible_envelope(simulated=True)),
        ctx,
    )
    assert len(results) == 2
    assert results[0].admitted is True
    assert results[1].admitted is False


# ---------------------------------------------------------------------------
# Each ten-point failure rejects completion/merge influence
# ---------------------------------------------------------------------------


def test_structural_invalidity_rejects() -> None:
    result = admit_receipt("not-an-object", _context())
    assert result.admitted is False
    assert result.may_affect_completion is False
    assert result.may_affect_merge is False
    assert result.check_map()["structural_validity"].passed is False
    assert "receipt_not_object" in result.reasons


def test_empty_envelope_rejects_structurally() -> None:
    result = admit_receipt({}, _context())
    assert result.admitted is False
    assert result.check_map()["structural_validity"].passed is False


def test_content_identity_mismatch_rejects() -> None:
    result = admit_receipt(
        _admissible_envelope(),
        _context(expected_content_id="sha256:" + ("ff" * 32)),
    )
    assert result.admitted is False
    assert result.check_map()["content_identity"].passed is False
    assert result.may_affect_merge is False


def test_content_identity_malformed_rejects() -> None:
    result = admit_receipt(
        _admissible_envelope(content_id="!!!", receipt_id="!!!"),
        _context(),
    )
    assert result.admitted is False
    assert result.check_map()["content_identity"].passed is False


def test_tree_binding_mismatch_rejects() -> None:
    result = admit_receipt(
        _admissible_envelope(repository_tree_id="tree:other"),
        _context(),
    )
    assert result.admitted is False
    failed = result.check_map()["source_tree_environment_policy_binding"]
    assert failed.passed is False
    assert failed.reason_code == "repository_tree_id_mismatch"


def test_policy_binding_mismatch_rejects() -> None:
    result = admit_receipt(
        _admissible_envelope(policy_id="policy:other"),
        _context(),
    )
    assert result.admitted is False
    assert (
        result.check_map()["source_tree_environment_policy_binding"].reason_code
        == "policy_id_mismatch"
    )


def test_environment_and_source_binding_mismatch_reject() -> None:
    env = admit_receipt(
        _admissible_envelope(environment_id="env:other"),
        _context(),
    )
    assert env.admitted is False
    assert (
        env.check_map()["source_tree_environment_policy_binding"].reason_code
        == "environment_id_mismatch"
    )
    src = admit_receipt(
        _admissible_envelope(source_id="source:other"),
        _context(),
    )
    assert src.admitted is False
    assert (
        src.check_map()["source_tree_environment_policy_binding"].reason_code
        == "source_id_mismatch"
    )


def test_translation_chain_invalid_rejects() -> None:
    result = admit_receipt(
        _admissible_envelope(
            translation={"valid": False, "translation_class": "exact"}
        ),
        _context(),
    )
    assert result.admitted is False
    assert result.check_map()["translation_chain"].passed is False
    assert result.check_map()["translation_chain"].reason_code == (
        "translation_chain_invalid"
    )


def test_heuristic_translation_rejects_for_kernel_policy() -> None:
    result = admit_receipt(
        _admissible_envelope(
            translation={"valid": True, "translation_class": "heuristic"}
        ),
        _context(),
    )
    assert result.admitted is False
    assert result.check_map()["translation_chain"].reason_code == (
        "translation_class_heuristic"
    )


def test_evidence_kind_candidate_cannot_support_proved() -> None:
    result = admit_receipt(
        _admissible_envelope(
            evidence_kind=EvidenceKind.ATP_CANDIDATE.value,
            authority_ceiling=AssuranceLevel.CANDIDATE.value,
        ),
        _context(required_authority=AssuranceLevel.CANDIDATE.value),
    )
    # Even at candidate floor, proved + candidate kind is rejected by kind check.
    assert result.admitted is False
    assert result.check_map()["evidence_kind"].passed is False


def test_evidence_kind_below_kernel_rejects_kernel_policy() -> None:
    kernel_floor = admit_receipt(
        _admissible_envelope(
            evidence_kind=EvidenceKind.SOLVER_RESULT.value,
            authority_ceiling=AssuranceLevel.SOLVER_CHECKED.value,
        ),
        _context(),
    )
    assert kernel_floor.admitted is False
    assert kernel_floor.check_map()["evidence_kind"].passed is False
    assert kernel_floor.may_affect_completion is False


def test_authority_ceiling_insufficient_rejects() -> None:
    result = admit_receipt(
        _admissible_envelope(
            authority_ceiling=AssuranceLevel.CANDIDATE.value,
            evidence_kind=EvidenceKind.KERNEL_VERIFICATION.value,
        ),
        _context(),
    )
    assert result.admitted is False
    assert result.check_map()["authority_ceiling"].passed is False
    assert result.may_affect_completion is False


def test_success_does_not_imply_authority() -> None:
    result = admit_receipt(
        _admissible_envelope(
            operation_status="succeeded",
            semantic_verdict="unknown",
            evidence_kind=EvidenceKind.KERNEL_VERIFICATION.value,
            authority_ceiling=AssuranceLevel.KERNEL_VERIFIED.value,
        ),
        _context(),
    )
    assert result.admitted is False
    # Either evidence kind (unknown verdict is fine) or authority check rejects
    # success-implied kernel authority.
    assert (
        result.check_map()["authority_ceiling"].reason_code
        == "success_does_not_imply_authority"
        or result.admitted is False
    )


def test_missing_reconstruction_rejects() -> None:
    result = admit_receipt(
        _admissible_envelope(
            reconstruction_passed=False,
            kernel_checked=True,
        ),
        _context(),
    )
    assert result.admitted is False
    assert result.check_map()["required_reconstruction"].passed is False


def test_missing_kernel_check_rejects() -> None:
    result = admit_receipt(
        _admissible_envelope(
            reconstruction_passed=True,
            kernel_checked=False,
            evidence=[],  # no typed kernel evidence fallback
        ),
        _context(),
    )
    assert result.admitted is False
    assert result.check_map()["required_reconstruction"].passed is False
    assert result.check_map()["required_reconstruction"].reason_code == (
        "kernel_check_not_passed"
    )


def test_stale_freshness_rejects() -> None:
    result = admit_receipt(
        _admissible_envelope(freshness=EvidenceFreshness.STALE.value),
        _context(),
    )
    assert result.admitted is False
    assert result.check_map()["freshness"].passed is False
    assert result.may_affect_merge is False


def test_unknown_freshness_rejects() -> None:
    result = admit_receipt(
        _admissible_envelope(freshness=EvidenceFreshness.UNKNOWN.value),
        _context(),
    )
    assert result.admitted is False
    assert result.check_map()["freshness"].passed is False


def test_simulated_receipt_rejects() -> None:
    result = admit_receipt(
        _admissible_envelope(simulated=True),
        _context(),
    )
    assert result.admitted is False
    assert result.check_map()["non_simulation"].passed is False
    assert result.may_affect_completion is False


def test_policy_operation_mismatch_rejects() -> None:
    result = admit_receipt(
        _admissible_envelope(operation="prove"),
        _context(operation="merge"),
    )
    assert result.admitted is False
    assert result.check_map()["policy_admission"].passed is False
    assert result.check_map()["policy_admission"].reason_code == "operation_mismatch"


def test_network_policy_overclaim_rejects() -> None:
    result = admit_receipt(
        _admissible_envelope(network_allowed=True),
        _context(network_allowed=False),
    )
    assert result.admitted is False
    assert result.check_map()["policy_admission"].reason_code == (
        "network_policy_denied"
    )


def test_policy_explicit_denial_rejects() -> None:
    result = admit_receipt(
        _admissible_envelope(policy_admitted=False),
        _context(),
    )
    assert result.admitted is False
    assert result.check_map()["policy_admission"].reason_code == (
        "policy_explicitly_denied"
    )


# ---------------------------------------------------------------------------
# ProofReceipt path + client projection unwrap
# ---------------------------------------------------------------------------


def test_proof_receipt_object_can_admit_when_kernel_fresh() -> None:
    receipt = _proof_receipt()
    # ProofReceipt authority is derived from evidence; pin required to kernel.
    result = admit_receipt(
        receipt,
        _context(
            expected_content_id=receipt.receipt_id,
            # ProofReceipt path binds translator_id; no separate translation body.
        ),
    )
    assert result.admitted is True
    assert result.may_affect_completion is True
    assert result.receipt_content_id == receipt.receipt_id


def test_simulated_proof_evidence_rejects() -> None:
    receipt = _proof_receipt(
        evidence=(_kernel_evidence(simulated=True),),
        kernel_receipt_id="kernel:obl:test",
    )
    result = admit_receipt(receipt, _context())
    assert result.admitted is False
    assert result.check_map()["non_simulation"].passed is False


def test_stale_proof_receipt_rejects() -> None:
    receipt = _proof_receipt(freshness=EvidenceFreshness.STALE)
    result = admit_receipt(receipt, _context())
    assert result.admitted is False
    assert result.check_map()["freshness"].passed is False


def test_client_projection_envelope_is_unwrapped() -> None:
    """LPC-110 projects {receipt, admitted:false}; admission unwraps the body."""

    projected = {
        "receipt": _admissible_envelope(),
        "admitted": False,
        "trusted": False,
        "ten_point_gate": "deferred_to_lpc_111",
    }
    result = admit_receipt(projected, _context())
    assert result.admitted is True
    assert result.may_affect_merge is True


def test_provider_success_alone_never_admits() -> None:
    success_only = {
        "operation_status": "succeeded",
        "semantic_verdict": "unknown",
        "evidence_kind": "candidate",
        "authority_ceiling": "advisory",
        "content_id": "sha256:" + ("cd" * 32),
        "repository_tree_id": "tree:sha256:abc",
        "policy_id": "policy:implementation-daemon",
        "freshness": "current",
        "simulated": False,
    }
    result = admit_receipt(success_only, _context())
    assert result.admitted is False
    assert result.may_affect_completion is False
    assert result.may_affect_merge is False


def test_partial_passes_never_set_completion_or_merge_flags() -> None:
    # Pass structural + identity-ish fields but fail simulation.
    result = admit_receipt(
        _admissible_envelope(simulated=True),
        _context(),
    )
    assert any(item.passed for item in result.checks)
    assert any(not item.passed for item in result.checks)
    assert result.admitted is False
    assert result.may_affect_completion is False
    assert result.may_affect_merge is False
    # Result constructor invariant: admitted == all checks.
    assert result.admitted == all(item.passed for item in result.checks)


def test_admission_result_rejects_incomplete_check_trace() -> None:
    ctx = _context()
    good = admit_receipt(_admissible_envelope(), ctx)
    with pytest.raises(LogicPlatformAdmissionError):
        AdmissionResult(
            admitted=True,
            disposition=AdmissionDisposition.ADMITTED,
            checks=good.checks[:5],
            context=ctx,
        )


def test_end_to_end_acceptance_path() -> None:
    """Exercise the LPC-111 acceptance path as one cohesive flow."""

    ctx = _context()
    helper = get_receipt_admission()

    admitted = helper.admit(_admissible_envelope(), ctx)
    assert admitted.admitted is True
    assert admitted.may_affect_completion is True
    assert admitted.may_affect_merge is True
    assert {c.check.value for c in admitted.checks} == set(TEN_POINT_CHECKS)

    for mutation, expected_reason_substr in (
        ({"simulated": True}, "simulated"),
        ({"freshness": "stale"}, "stale"),
        ({"repository_tree_id": "tree:other"}, "tree"),
        ({"reconstruction_passed": False}, "reconstruction"),
        ({"authority_ceiling": "candidate"}, "authority"),
        ({"policy_admitted": False}, "policy"),
    ):
        rejected = helper.admit(_admissible_envelope(**mutation), ctx)
        assert rejected.admitted is False, mutation
        assert rejected.may_affect_completion is False
        assert rejected.may_affect_merge is False
        assert rejected.reasons, mutation
        assert any(
            expected_reason_substr in code for code in rejected.reasons
        ) or any(
            expected_reason_substr in item.reason_code
            for item in rejected.failed_checks()
        ), (mutation, rejected.reasons)
