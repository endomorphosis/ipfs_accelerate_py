from __future__ import annotations

import json

import pytest

from ipfs_accelerate_py.agent_supervisor.formal_verification_contracts import (
    AssuranceLevel,
    AttemptStatus,
    CodeProofObligation,
    ContractValidationError,
    EvidenceAuthority,
    EvidenceFreshness,
    EvidenceKind,
    EvidenceVerdict,
    ProofAttempt,
    ProofEvidence,
    ProofPlan,
    ProofPlanStep,
    ProofReceipt,
    ProofStage,
    ProofVerdict,
    ResourceBudget,
    assurance_satisfies,
    canonical_json,
    derive_assurance,
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


def _obligation(**changes: object) -> CodeProofObligation:
    values = {
        "repository_id": "repo:complaint-generator",
        "repository_tree_id": "git-tree:abc123",
        "ast_scope_ids": ("scope:z", "scope:a"),
        "statement": "A current fencing token is required for every lease mutation.",
        "premise_ids": ("premise:lease-state", "premise:token-order"),
        "template_id": "lease-fencing",
        "template_version": "2",
        "template_semantic_hash": "sha256:template",
        "invariant_class": "lease_safety",
        "task_id": "REF-245",
        "required_assurance": AssuranceLevel.KERNEL_VERIFIED,
        "fallback_checks": ("pytest:test_stale_token",),
        "metadata": {"risk": "critical", "ordinal": 2},
    }
    values.update(changes)
    return CodeProofObligation(**values)


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
                depends_on=("solve",),
                required_assurance=AssuranceLevel.KERNEL_VERIFIED,
                resource_class="kernel",
            ),
            ProofPlanStep(
                step_id="solve",
                obligation_id=obligation.obligation_id,
                stage=ProofStage.SOLVE,
                provider_id="provider:hammer",
                resource_class="solver",
            ),
        ),
        policy_id="policy:formal-v1",
        resource_budget=_budget(),
        max_parallel=2,
        task_id="REF-245",
    )


def _candidate(
    obligation_id: str,
    *,
    kind: EvidenceKind = EvidenceKind.SMT_CANDIDATE,
    authority: EvidenceAuthority = EvidenceAuthority.PROVIDER,
    simulated: bool = False,
    freshness: EvidenceFreshness = EvidenceFreshness.CURRENT,
) -> ProofEvidence:
    return ProofEvidence(
        kind=kind,
        authority=authority,
        verdict=EvidenceVerdict.ACCEPTED,
        artifact_id="artifact:candidate",
        subject_id=obligation_id,
        verifier_id="provider:claimed-verifier",
        freshness=freshness,
        independent=True,
        simulated=simulated,
    )


def _kernel(obligation_id: str) -> ProofEvidence:
    return ProofEvidence(
        kind=EvidenceKind.KERNEL_VERIFICATION,
        authority=EvidenceAuthority.KERNEL,
        verdict=EvidenceVerdict.ACCEPTED,
        artifact_id="artifact:checked-lean-source",
        subject_id=obligation_id,
        verifier_id="kernel:lean-4.19",
        freshness=EvidenceFreshness.CURRENT,
        independent=True,
        simulated=False,
    )


def _receipt(
    obligation: CodeProofObligation,
    plan: ProofPlan,
    evidence: tuple[ProofEvidence, ...],
    **changes: object,
) -> ProofReceipt:
    values = {
        "obligation_id": obligation.obligation_id,
        "plan_id": plan.plan_id,
        "attempt_id": "attempt:one",
        "repository_id": obligation.repository_id,
        "repository_tree_id": obligation.repository_tree_id,
        "ast_scope_ids": obligation.ast_scope_ids,
        "premise_ids": obligation.premise_ids,
        "translator_id": "translator:python-to-lean@1",
        "solver_id": "solver:z3@4.13",
        "kernel_id": "kernel:lean-4.19",
        "toolchain_id": "toolchain:nix-lock-sha256",
        "theorem_registry_id": "registry:reviewed-v3",
        "policy_id": "policy:formal-v1",
        "resource_budget": _budget(),
        "verdict": ProofVerdict.PROVED,
        "evidence": evidence,
        "provider_id": "provider:hammer",
        "provider_claimed_assurance": AssuranceLevel.ATTESTED,
        "started_at": "2026-07-22T01:00:00Z",
        "finished_at": "2026-07-22T01:00:01Z",
        "resource_usage": {"wall_time_ms": 1_000, "peak_memory_bytes": 1_000_000},
    }
    values.update(changes)
    return ProofReceipt(**values)


def test_obligation_has_canonical_json_round_trip_and_content_identity() -> None:
    first = _obligation(
        ast_scope_ids=("scope:z", "scope:a"),
        metadata={"z": ["last"], "a": {"b": 2}},
    )
    second = _obligation(
        ast_scope_ids=("scope:a", "scope:z"),
        metadata={"a": {"b": 2}, "z": ["last"]},
    )

    assert first.to_json() == second.to_json()
    assert first.obligation_id == second.obligation_id
    assert first.obligation_id.startswith("baguqeera")
    assert json.loads(first.to_json())["required_assurance"] == "kernel_verified"
    assert CodeProofObligation.from_dict(first.to_dict()) == first
    assert _obligation(statement="A different theorem.").obligation_id != first.obligation_id


def test_all_primary_contracts_have_stable_ids_and_round_trip() -> None:
    obligation = _obligation()
    plan = _plan(obligation)
    evidence = _candidate(obligation.obligation_id)
    attempt = ProofAttempt(
        plan_id=plan.plan_id,
        step_id="solve",
        obligation_id=obligation.obligation_id,
        repository_tree_id=obligation.repository_tree_id,
        provider_id="provider:hammer",
        stage=ProofStage.SOLVE,
        status=AttemptStatus.SUCCEEDED,
        evidence=(evidence,),
        input_ids=("input:b", "input:a"),
        output_ids=("artifact:candidate",),
        provider_claimed_assurance=AssuranceLevel.KERNEL_VERIFIED,
    )
    receipt = _receipt(obligation, plan, (evidence,), attempt_id=attempt.attempt_id)

    assert ProofPlan.from_dict(plan.to_dict()).plan_id == plan.plan_id
    assert ProofAttempt.from_dict(attempt.to_dict()).attempt_id == attempt.attempt_id
    assert ProofReceipt.from_dict(receipt.to_dict()).receipt_id == receipt.receipt_id
    assert len({obligation.obligation_id, plan.plan_id, attempt.attempt_id, receipt.receipt_id}) == 4
    for contract in (obligation, plan, attempt, receipt):
        assert contract.to_json() == canonical_json(contract.to_dict())
        assert contract.content_id == contract.cid == contract.identity


def test_plan_rejects_unknown_dependencies_and_cycles() -> None:
    obligation = _obligation()
    common = {
        "repository_tree_id": obligation.repository_tree_id,
        "obligation_ids": (obligation.obligation_id,),
        "policy_id": "policy:v1",
        "resource_budget": _budget(),
    }
    with pytest.raises(ContractValidationError, match="unknown dependencies"):
        ProofPlan(
            **common,
            steps=(
                ProofPlanStep(
                    step_id="solve",
                    obligation_id=obligation.obligation_id,
                    stage=ProofStage.SOLVE,
                    provider_id="provider",
                    depends_on=("missing",),
                ),
            ),
        )
    with pytest.raises(ContractValidationError, match="acyclic"):
        ProofPlan(
            **common,
            steps=(
                ProofPlanStep(
                    step_id="a",
                    obligation_id=obligation.obligation_id,
                    stage=ProofStage.SOLVE,
                    provider_id="provider",
                    depends_on=("b",),
                ),
                ProofPlanStep(
                    step_id="b",
                    obligation_id=obligation.obligation_id,
                    stage=ProofStage.KERNEL_VERIFY,
                    provider_id="kernel",
                    depends_on=("a",),
                ),
            ),
        )


def test_receipt_binds_every_trust_relevant_input() -> None:
    obligation = _obligation()
    plan = _plan(obligation)
    receipt = _receipt(obligation, plan, (_kernel(obligation.obligation_id),))
    payload = receipt.to_dict()

    assert payload["repository_tree_id"] == obligation.repository_tree_id
    assert payload["ast_scope_ids"] == sorted(obligation.ast_scope_ids)
    assert payload["premise_ids"] == sorted(obligation.premise_ids)
    assert payload["translator_id"].startswith("translator:")
    assert payload["solver_id"].startswith("solver:")
    assert payload["kernel_id"].startswith("kernel:")
    assert payload["toolchain_id"].startswith("toolchain:")
    assert payload["theorem_registry_id"].startswith("registry:")
    assert payload["policy_id"] == "policy:formal-v1"
    assert payload["resource_budget"]["wall_time_ms"] == 30_000
    assert payload["resource_budget"]["network_allowed"] is False
    assert receipt.authoritative_assurance is AssuranceLevel.KERNEL_VERIFIED

    changed = _receipt(
        obligation,
        plan,
        (_kernel(obligation.obligation_id),),
        policy_id="policy:formal-v2",
    )
    assert changed.receipt_id != receipt.receipt_id


@pytest.mark.parametrize(
    ("kind", "authority"),
    (
        (EvidenceKind.LLM_OUTPUT, EvidenceAuthority.LLM),
        (EvidenceKind.ATP_CANDIDATE, EvidenceAuthority.ATP),
        (EvidenceKind.SMT_CANDIDATE, EvidenceAuthority.SMT),
    ),
)
def test_candidate_sources_cannot_become_kernel_verified(
    kind: EvidenceKind, authority: EvidenceAuthority
) -> None:
    obligation = _obligation()
    plan = _plan(obligation)
    evidence = _candidate(
        obligation.obligation_id,
        kind=kind,
        authority=authority,
    )
    receipt = _receipt(obligation, plan, (evidence,))

    assert receipt.authoritative_assurance is AssuranceLevel.CANDIDATE
    assert not receipt.satisfies(AssuranceLevel.KERNEL_VERIFIED)


def test_provider_claims_are_audited_but_never_authoritative() -> None:
    obligation = _obligation()
    plan = _plan(obligation)
    forged_kernel = _candidate(
        obligation.obligation_id,
        kind=EvidenceKind.KERNEL_VERIFICATION,
        authority=EvidenceAuthority.PROVIDER,
    )
    attempt = ProofAttempt(
        plan_id=plan.plan_id,
        step_id="kernel",
        obligation_id=obligation.obligation_id,
        repository_tree_id=obligation.repository_tree_id,
        provider_id="provider:untrusted",
        stage=ProofStage.KERNEL_VERIFY,
        status=AttemptStatus.SUCCEEDED,
        evidence=(forged_kernel,),
        provider_claimed_assurance=AssuranceLevel.ATTESTED,
    )
    receipt = _receipt(obligation, plan, (forged_kernel,))

    assert attempt.provider_claimed_assurance is AssuranceLevel.ATTESTED
    assert attempt.authoritative_assurance is AssuranceLevel.CANDIDATE
    assert receipt.authoritative_assurance is AssuranceLevel.CANDIDATE
    with pytest.raises(TypeError):
        _receipt(  # type: ignore[call-arg]
            obligation,
            plan,
            (forged_kernel,),
            assurance=AssuranceLevel.ATTESTED,
        )

    tampered = receipt.to_dict()
    tampered["authoritative_assurance"] = "attested"
    with pytest.raises(ContractValidationError, match="does not match derived"):
        ProofReceipt.from_dict(tampered)


def test_independent_exact_kernel_evidence_derives_kernel_assurance() -> None:
    obligation = _obligation()
    plan = _plan(obligation)
    receipt = _receipt(obligation, plan, (_kernel(obligation.obligation_id),))

    assert receipt.authoritative_assurance is AssuranceLevel.KERNEL_VERIFIED
    assert receipt.satisfies(AssuranceLevel.SOLVER_CHECKED)
    assert assurance_satisfies(
        AssuranceLevel.KERNEL_VERIFIED, AssuranceLevel.CANDIDATE
    )
    assert not assurance_satisfies(
        AssuranceLevel.SOLVER_CHECKED, AssuranceLevel.KERNEL_VERIFIED
    )

    wrong_subject = _kernel("different-obligation")
    assert (
        derive_assurance(
            (wrong_subject,),
            obligation_id=obligation.obligation_id,
            kernel_id="kernel:lean-4.19",
        )
        is AssuranceLevel.CANDIDATE
    )


def test_attestation_requires_kernel_receipt_and_real_independent_verification() -> None:
    obligation = _obligation()
    plan = _plan(obligation)
    kernel_receipt_id = "baguqeera-kernel-receipt"
    attestation = ProofEvidence(
        kind=EvidenceKind.CRYPTOGRAPHIC_ATTESTATION,
        authority=EvidenceAuthority.ATTESTATION_VERIFIER,
        verdict=EvidenceVerdict.ACCEPTED,
        artifact_id="artifact:groth16-proof",
        subject_id=kernel_receipt_id,
        verifier_id="verifier:groth16-production",
        independent=True,
        simulated=False,
    )
    receipt = _receipt(
        obligation,
        plan,
        (_kernel(obligation.obligation_id), attestation),
        kernel_receipt_id=kernel_receipt_id,
    )

    assert receipt.authoritative_assurance is AssuranceLevel.ATTESTED

    no_kernel = _receipt(
        obligation,
        plan,
        (attestation,),
        kernel_receipt_id=kernel_receipt_id,
    )
    assert no_kernel.authoritative_assurance is AssuranceLevel.CANDIDATE

    simulated = ProofEvidence(
        kind=EvidenceKind.CRYPTOGRAPHIC_ATTESTATION,
        authority=EvidenceAuthority.ATTESTATION_VERIFIER,
        verdict=EvidenceVerdict.ACCEPTED,
        artifact_id="artifact:simulated-proof",
        subject_id=kernel_receipt_id,
        verifier_id="simulated-zkp-v0.1",
        independent=True,
        simulated=True,
    )
    simulated_receipt = _receipt(
        obligation,
        plan,
        (_kernel(obligation.obligation_id), simulated),
        kernel_receipt_id=kernel_receipt_id,
    )
    assert simulated_receipt.authoritative_assurance is AssuranceLevel.KERNEL_VERIFIED
    assert simulated_receipt.authoritative_assurance is not AssuranceLevel.ATTESTED


def test_stale_cache_evidence_cannot_satisfy_kernel_or_attested_assurance() -> None:
    obligation = _obligation()
    plan = _plan(obligation)
    stale_kernel = ProofEvidence(
        kind=EvidenceKind.KERNEL_VERIFICATION,
        authority=EvidenceAuthority.KERNEL,
        verdict=EvidenceVerdict.ACCEPTED,
        artifact_id="cache:old-kernel-receipt",
        subject_id=obligation.obligation_id,
        verifier_id="kernel:lean-4.19",
        freshness=EvidenceFreshness.STALE,
        independent=True,
    )
    receipt = _receipt(
        obligation,
        plan,
        (stale_kernel,),
        freshness=EvidenceFreshness.STALE,
    )

    assert receipt.authoritative_assurance is AssuranceLevel.CANDIDATE
    assert "stale_or_unknown_receipt" in receipt.assurance_assessment.reason_codes
    assert not receipt.satisfies(AssuranceLevel.KERNEL_VERIFIED)


def test_canonical_contracts_reject_floats_and_forged_content_ids() -> None:
    with pytest.raises(ContractValidationError, match="floats"):
        _obligation(metadata={"confidence": 0.99})

    payload = _obligation().to_dict()
    payload["content_id"] = "forged"
    with pytest.raises(ContractValidationError, match="identity"):
        CodeProofObligation.from_dict(payload)
