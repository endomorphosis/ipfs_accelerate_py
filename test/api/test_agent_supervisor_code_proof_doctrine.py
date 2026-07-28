"""CBP-010: fail-closed codebase-proof doctrine guards.

These tests pin the normative CBP rules used by the agent supervisor:

1. Candidate assurance cannot satisfy kernel-required policy.
2. Private witness markers are rejected from public receipt JSON.
3. Simulated attestation/ZK evidence cannot produce AssuranceLevel.ATTESTED.
4. The sealed CBP plan names formal_verification_cache as the sole memoization
   trust boundary (read-only check).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.formal_verification_contracts import (
    AssuranceLevel,
    CodeProofObligation,
    ContractValidationError,
    EvidenceAuthority,
    EvidenceFreshness,
    EvidenceKind,
    EvidenceVerdict,
    ProofEvidence,
    ProofPlan,
    ProofPlanStep,
    ProofReceipt,
    ProofStage,
    ProofVerdict,
    ResourceBudget,
    _contains_private_material,
    assurance_satisfies,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CBP_PLAN = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "AGENT_SUPERVISOR_CODEBASE_PROOF_CONTEXT_PLAN.md"
)


def _budget() -> ResourceBudget:
    return ResourceBudget(
        wall_time_ms=30_000,
        cpu_time_ms=20_000,
        memory_bytes=512 * 1024 * 1024,
        disk_bytes=64 * 1024 * 1024,
        max_processes=4,
        max_premises=32,
        max_output_bytes=1_000_000,
        model_token_limit=4_096,
        provider_quota=1,
        network_allowed=False,
    )


def _obligation() -> CodeProofObligation:
    return CodeProofObligation(
        repository_id="repo:cbp-doctrine",
        repository_tree_id="git-tree:cbp-doctrine",
        ast_scope_ids=("scope:a",),
        statement="Doctrine probe obligation.",
        premise_ids=("premise:a",),
        template_id="lease-fencing",
        template_version="2",
        template_semantic_hash="sha256:template",
        invariant_class="lease_safety",
        task_id="CBP-010",
        required_assurance=AssuranceLevel.KERNEL_VERIFIED,
        fallback_checks=("pytest:test_agent_supervisor_code_proof_doctrine",),
        metadata={"suite": "cbp-doctrine"},
    )


def _plan(obligation: CodeProofObligation) -> ProofPlan:
    return ProofPlan(
        repository_tree_id=obligation.repository_tree_id,
        obligation_ids=(obligation.obligation_id,),
        steps=(
            ProofPlanStep(
                step_id="kernel",
                obligation_id=obligation.obligation_id,
                stage=ProofStage.KERNEL_VERIFY,
                provider_id="supervisor:lean-kernel",
                required_assurance=AssuranceLevel.KERNEL_VERIFIED,
                resource_class="kernel",
            ),
        ),
        policy_id="policy:cbp-doctrine",
        resource_budget=_budget(),
        max_parallel=1,
        task_id="CBP-010",
    )


def _kernel(obligation_id: str) -> ProofEvidence:
    return ProofEvidence(
        kind=EvidenceKind.KERNEL_VERIFICATION,
        authority=EvidenceAuthority.KERNEL,
        verdict=EvidenceVerdict.ACCEPTED,
        artifact_id="artifact:kernel",
        subject_id=obligation_id,
        verifier_id="kernel:lean-4.19",
        freshness=EvidenceFreshness.CURRENT,
        independent=True,
        simulated=False,
    )


def _candidate(obligation_id: str) -> ProofEvidence:
    return ProofEvidence(
        kind=EvidenceKind.SMT_CANDIDATE,
        authority=EvidenceAuthority.SMT,
        verdict=EvidenceVerdict.ACCEPTED,
        artifact_id="artifact:candidate",
        subject_id=obligation_id,
        verifier_id="provider:claimed",
        freshness=EvidenceFreshness.CURRENT,
        independent=True,
        simulated=False,
    )


def _receipt(
    obligation: CodeProofObligation,
    plan: ProofPlan,
    evidence: tuple[ProofEvidence, ...],
) -> ProofReceipt:
    return ProofReceipt(
        obligation_id=obligation.obligation_id,
        plan_id=plan.plan_id,
        attempt_id="attempt:cbp-010",
        repository_id=obligation.repository_id,
        repository_tree_id=obligation.repository_tree_id,
        ast_scope_ids=obligation.ast_scope_ids,
        premise_ids=obligation.premise_ids,
        translator_id="translator:python-to-lean@1",
        solver_id="solver:z3@4.13",
        kernel_id="kernel:lean-4.19",
        toolchain_id="toolchain:nix-lock-sha256",
        theorem_registry_id="registry:reviewed-v3",
        policy_id="policy:cbp-doctrine",
        resource_budget=_budget(),
        verdict=ProofVerdict.PROVED,
        evidence=evidence,
        provider_id="provider:hammer",
        provider_claimed_assurance=AssuranceLevel.ATTESTED,
        started_at="2026-07-28T00:00:00Z",
        finished_at="2026-07-28T00:00:01Z",
        resource_usage={"wall_time_ms": 1_000, "peak_memory_bytes": 1_000_000},
    )


def test_candidate_assurance_cannot_satisfy_kernel_required_policy() -> None:
    obligation = _obligation()
    plan = _plan(obligation)
    receipt = _receipt(obligation, plan, (_candidate(obligation.obligation_id),))

    assert receipt.authoritative_assurance is AssuranceLevel.CANDIDATE
    assert not receipt.satisfies(AssuranceLevel.KERNEL_VERIFIED)
    assert not AssuranceLevel.CANDIDATE.satisfies(AssuranceLevel.KERNEL_VERIFIED)
    assert not assurance_satisfies(
        AssuranceLevel.CANDIDATE, AssuranceLevel.KERNEL_VERIFIED
    )
    assert assurance_satisfies(
        AssuranceLevel.KERNEL_VERIFIED, AssuranceLevel.CANDIDATE
    )


def test_private_witness_markers_rejected_from_public_receipt_json() -> None:
    assert _contains_private_material({"private_witness": "secret-gold"})
    assert _contains_private_material({"nested": {"hidden_witness": True}})
    assert _contains_private_material({"api_key": "x"})
    assert not _contains_private_material({"public_statement": "ok", "digest": "abc"})

    obligation = _obligation()
    plan = _plan(obligation)
    receipt = _receipt(obligation, plan, (_kernel(obligation.obligation_id),))
    payload = receipt.to_dict()
    payload["metadata"] = {"private_witness": "must-not-ship"}

    with pytest.raises(ContractValidationError, match="private"):
        ProofReceipt.from_dict(payload)


def test_simulated_attestation_cannot_produce_attested_assurance() -> None:
    obligation = _obligation()
    plan = _plan(obligation)
    kernel_receipt_id = "baguqeera-kernel-receipt-doctrine"
    simulated = ProofEvidence(
        kind=EvidenceKind.CRYPTOGRAPHIC_ATTESTATION,
        authority=EvidenceAuthority.ATTESTATION_VERIFIER,
        verdict=EvidenceVerdict.ACCEPTED,
        artifact_id="artifact:simulated-zkp",
        subject_id=kernel_receipt_id,
        verifier_id="simulated-zkp-v0.1",
        independent=True,
        simulated=True,
    )
    receipt = ProofReceipt(
        obligation_id=obligation.obligation_id,
        plan_id=plan.plan_id,
        attempt_id="attempt:sim",
        repository_id=obligation.repository_id,
        repository_tree_id=obligation.repository_tree_id,
        ast_scope_ids=obligation.ast_scope_ids,
        premise_ids=obligation.premise_ids,
        translator_id="translator:python-to-lean@1",
        solver_id="solver:z3@4.13",
        kernel_id="kernel:lean-4.19",
        toolchain_id="toolchain:nix-lock-sha256",
        theorem_registry_id="registry:reviewed-v3",
        policy_id="policy:cbp-doctrine",
        resource_budget=_budget(),
        verdict=ProofVerdict.PROVED,
        evidence=(_kernel(obligation.obligation_id), simulated),
        provider_id="provider:sim",
        provider_claimed_assurance=AssuranceLevel.ATTESTED,
        started_at="2026-07-28T00:00:00Z",
        finished_at="2026-07-28T00:00:01Z",
        resource_usage={"wall_time_ms": 10, "peak_memory_bytes": 1_000},
        kernel_receipt_id=kernel_receipt_id,
    )

    assert receipt.authoritative_assurance is AssuranceLevel.KERNEL_VERIFIED
    assert receipt.authoritative_assurance is not AssuranceLevel.ATTESTED
    assert not receipt.satisfies(AssuranceLevel.ATTESTED)


def test_sealed_plan_names_formal_verification_cache_as_trust_boundary() -> None:
    assert CBP_PLAN.is_file(), f"missing sealed CBP plan: {CBP_PLAN}"
    text = CBP_PLAN.read_text(encoding="utf-8")
    assert "formal_verification_cache" in text
    assert "TrustAwareProofCache" in text or "trust-aware" in text.lower()
    assert "sole" in text.lower() or "one cache trust boundary" in text.lower()
    assert "rederive" in text.lower() or "re-derive" in text.lower()
    assert "sim" in text.lower()  # simulated ZK ≠ ATTESTED doctrine present
