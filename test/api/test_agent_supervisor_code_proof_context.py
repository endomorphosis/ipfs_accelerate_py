"""CBP-060: obligation-first context capsule tests."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.code_claim_contracts import (
    ClaimFamily,
    ClaimStatus,
    CodeClaimRecord,
    EvidenceTier,
    build_invalidation_selectors,
)
from ipfs_accelerate_py.agent_supervisor.code_proof_context import (
    CODE_PROOF_CONTEXT_INTERFACE,
    UNTRUSTED_DATA_LABEL,
    CodeProofContextRequest,
    compile_code_proof_context_capsule,
)
from ipfs_accelerate_py.agent_supervisor.code_proof_query import build_code_proof_query
from ipfs_accelerate_py.agent_supervisor.context_compiler import (
    compile_code_proof_context_capsule as compile_via_compiler_module,
)
from ipfs_accelerate_py.agent_supervisor.context_contracts import ContextBudget, ContextTier
from ipfs_accelerate_py.agent_supervisor.decision_context import (
    CODE_PROOF_OBLIGATION_FIRST_CORE_FIELDS,
)
from ipfs_accelerate_py.agent_supervisor.formal_verification_contracts import (
    AssuranceLevel,
)


def _budget(max_input_tokens: int = 3_000) -> ContextBudget:
    return ContextBudget(
        max_input_tokens=max_input_tokens,
        reserved_output_tokens=400,
        reserved_tool_tokens=100,
        max_items=48,
        max_item_bytes=16_384,
        max_serialized_bytes=400_000,
        max_depth=10,
        max_text_bytes=16_384,
    )


def _claim(
    property_id: str,
    status: ClaimStatus,
    *,
    obligation_id: str = "obligation:1",
) -> CodeClaimRecord:
    selectors = build_invalidation_selectors(
        repository_tree_id="git-tree:ctx",
        scope_ids=("scope:a",),
        premise_ids=("premise:a",),
        assumption_ids=("assumption:a",),
        toolchain_id="toolchain:t",
        policy_id="policy:p",
        catalog_version="1",
        property_id=property_id,
        producer_id="test",
        required_assurance=AssuranceLevel.KERNEL_VERIFIED,
    )
    satisfied = status is ClaimStatus.SATISFIED
    return CodeClaimRecord(
        claim_family=ClaimFamily.API_CONTRACT,
        status=status,
        property_id=property_id,
        obligation_id=obligation_id,
        repository_id="repo:ctx",
        repository_tree_id="git-tree:ctx",
        scope_ids=("scope:a",),
        premise_ids=("premise:a",),
        assumption_ids=("assumption:a",),
        producer_id="test",
        toolchain_id="toolchain:t",
        policy_id="policy:p",
        catalog_version="1",
        required_assurance=AssuranceLevel.KERNEL_VERIFIED,
        derived_assurance=(
            AssuranceLevel.KERNEL_VERIFIED if satisfied else AssuranceLevel.UNVERIFIED
        ),
        invalidation_selectors=selectors,
        evidence_ids=("evidence:k",) if satisfied else (),
        evidence_tiers=(EvidenceTier.KERNEL_PROOF,) if satisfied else (),
        receipt_id="receipt:k1" if satisfied else "",
        statement=property_id,
    )


def _request(**kwargs):
    claims = kwargs.pop(
        "claims",
        (
            _claim("property:open", ClaimStatus.OPEN, obligation_id="obligation:open"),
            _claim("property:sat", ClaimStatus.SATISFIED, obligation_id="obligation:sat"),
            _claim("property:ref", ClaimStatus.REFUTED, obligation_id="obligation:ref"),
        ),
    )
    values = dict(
        repository_id="repo:ctx",
        tree_id="git-tree:ctx",
        task_id="CBP-060-DEMO",
        acceptance_ids=("accept:open-cleared", "accept:no-regression"),
        claims=claims,
        changed_paths=("src/worker.py",),
        changed_symbols=("Worker.run",),
        specification_handles=("spec:api-contract@1",),
        failure_traces=({"summary": "prior attempt failed validation", "code": "E1"},),
        optional_source_snippets=(
            {"path": "src/worker.py", "text": "class Worker: ...", "handle": "h:worker"},
        ),
        budget=_budget(),
    )
    values.update(kwargs)
    return CodeProofContextRequest(**values)


def test_invariant_core_includes_task_open_obligations_assumptions_counterexamples_slice() -> None:
    capsule = compile_code_proof_context_capsule(_request())
    assert capsule.task_id == "CBP-060-DEMO"
    assert "accept:open-cleared" in capsule.acceptance_ids
    assert "obligation:open" in capsule.open_obligation_ids
    assert capsule.counterexample_ids  # refuted claim
    kinds = {ref.kind for ref in capsule.capsule.evidence}
    assert "open_obligation" in kinds
    assert "assumptions" in kinds
    assert "counterexample" in kinds
    assert "dependency_ast_slice" in kinds
    assert "specification_handle" in kinds
    assert "failure_trace" in kinds
    # required open obligations are invariant tier
    open_refs = [r for r in capsule.capsule.evidence if r.kind == "open_obligation"]
    assert open_refs
    assert all(r.tier is ContextTier.INVARIANT for r in open_refs)
    assert all(r.metadata.get("required") is True for r in open_refs)


def test_satisfied_proofs_are_digest_handles_only() -> None:
    capsule = compile_code_proof_context_capsule(_request())
    assert "receipt:k1" in capsule.satisfied_receipt_handles
    sat_refs = [r for r in capsule.capsule.evidence if r.kind == "satisfied_proof_handle"]
    assert sat_refs
    assert all(r.metadata.get("digest_only") is True for r in sat_refs)
    assert all(r.metadata.get("no_body") is True for r in sat_refs)
    # not invariant-required
    assert all(r.metadata.get("required") is False for r in sat_refs)


def test_optional_source_is_untrusted_data_and_not_instructions() -> None:
    capsule = compile_code_proof_context_capsule(_request())
    src_refs = [r for r in capsule.capsule.evidence if r.kind == "optional_source"]
    assert src_refs
    for ref in src_refs:
        assert ref.metadata.get("data_label") == UNTRUSTED_DATA_LABEL
        assert ref.metadata.get("instruction_injection") is False
        assert ref.metadata.get("treat_as") == "data_not_instructions"
        assert ref.metadata.get("required") is False


def test_solver_traces_excluded_by_default() -> None:
    capsule = compile_code_proof_context_capsule(_request())
    kinds = {ref.kind for ref in capsule.capsule.evidence}
    assert "solver_trace" not in kinds
    assert capsule.metadata.get("solver_traces_excluded_by_default") is True

    with_trace = compile_code_proof_context_capsule(
        _request(include_solver_traces=True)
    )
    kinds2 = {ref.kind for ref in with_trace.capsule.evidence}
    assert "solver_trace" in kinds2


def test_token_budget_and_omitted_manifest_auditable() -> None:
    capsule = compile_code_proof_context_capsule(_request())
    assert "max_input_tokens" in capsule.token_budget
    assert "input_tokens" in capsule.token_budget
    assert capsule.token_budget["input_tokens"] >= 1
    # omitted_handles is a tuple (may be empty under roomy budget)
    assert isinstance(capsule.omitted_handles, tuple)
    assert capsule.to_dict()["interface"] == CODE_PROOF_CONTEXT_INTERFACE
    assert capsule.capsule_id


def test_required_claim_coverage_cannot_be_deferred() -> None:
    capsule = compile_code_proof_context_capsule(_request())
    acceptance = capsule.capsule.acceptance
    assert acceptance.get("cannot_defer_required_claims") is True
    assert "obligation:open" in acceptance.get("required_claim_coverage", [])
    # no required evidence uses expansion tier
    for ref in capsule.capsule.evidence:
        if ref.metadata.get("required"):
            assert ref.tier is not ContextTier.EXPANSION


def test_compiler_module_wrapper_matches() -> None:
    request = _request()
    a = compile_code_proof_context_capsule(request)
    b = compile_via_compiler_module(request)
    assert a.task_id == b.task_id
    assert a.open_obligation_ids == b.open_obligation_ids


def test_decision_context_marks_obligations_as_core() -> None:
    assert "obligations" in CODE_PROOF_OBLIGATION_FIRST_CORE_FIELDS
    assert "acceptance" in CODE_PROOF_OBLIGATION_FIRST_CORE_FIELDS
