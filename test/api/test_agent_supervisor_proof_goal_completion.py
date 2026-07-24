from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ipfs_accelerate_py.agent_supervisor.code_proof_obligations import (
    ImplementationEvidenceKind,
    ImplementationObligationSet,
    ImplementationResultBinding,
    ImplementationResultEvidence,
    compile_candidate_proof_scopes,
    derive_fresh_implementation_obligations,
)
from ipfs_accelerate_py.agent_supervisor.formal_verification_contracts import (
    AssuranceLevel,
    CodeProofObligation,
    EvidenceAuthority,
    EvidenceFreshness,
    EvidenceKind,
    EvidenceVerdict,
    ProofEvidence,
    ProofReceipt,
    ProofVerdict,
    ResourceBudget,
)
from ipfs_accelerate_py.agent_supervisor.formal_planning_contracts import (
    PlanAssurance,
    PlanConsistencyLevel,
)
from ipfs_accelerate_py.agent_supervisor.goal_completion import (
    CompletionEvidence,
    GoalState,
    evaluate_code_proof_goal_completion,
    evaluate_completion_gate,
    evaluate_goal_completion,
    validate_completion_evidence,
)


NOW = datetime(2026, 7, 23, 12, 0, tzinfo=timezone.utc)
TREE = "git-tree:ref-267"
REPOSITORY = "repo:complaint-generator"
CRITERION = "Lease mutations require a current fencing token."


def _obligation(
    *,
    required_assurance: AssuranceLevel = AssuranceLevel.KERNEL_VERIFIED,
) -> CodeProofObligation:
    return CodeProofObligation(
        repository_id=REPOSITORY,
        repository_tree_id=TREE,
        ast_scope_ids=("scope:lease-mutation",),
        statement=CRITERION,
        template_id="lease-fencing",
        template_version="1",
        template_semantic_hash="sha256:lease-template",
        task_id="REF-267",
        required_assurance=required_assurance,
    )


def _kernel_evidence(
    obligation: CodeProofObligation,
    *,
    verdict: EvidenceVerdict = EvidenceVerdict.ACCEPTED,
    failure_code: str = "",
) -> ProofEvidence:
    return ProofEvidence(
        kind=EvidenceKind.KERNEL_VERIFICATION,
        authority=EvidenceAuthority.KERNEL,
        verdict=verdict,
        artifact_id="artifact:kernel-reconstruction",
        subject_id=obligation.obligation_id,
        verifier_id="kernel:lean-4.19",
        independent=True,
        metadata={"failure_code": failure_code} if failure_code else {},
    )


def _receipt(
    obligation: CodeProofObligation,
    *,
    evidence: tuple[ProofEvidence, ...] | None = None,
    verdict: ProofVerdict = ProofVerdict.PROVED,
    freshness: EvidenceFreshness = EvidenceFreshness.CURRENT,
    provider_claimed_assurance: AssuranceLevel = AssuranceLevel.ATTESTED,
) -> ProofReceipt:
    return ProofReceipt(
        obligation_id=obligation.obligation_id,
        plan_id="plan:ref-267",
        attempt_id="attempt:ref-267",
        repository_id=REPOSITORY,
        repository_tree_id=TREE,
        ast_scope_ids=obligation.ast_scope_ids,
        premise_ids=(),
        translator_id="translator:python-lean@1",
        solver_id="solver:z3@4",
        kernel_id="kernel:lean-4.19",
        toolchain_id="toolchain:locked",
        policy_id="policy:critical-proof",
        resource_budget=ResourceBudget(wall_time_ms=30_000),
        verdict=verdict,
        evidence=evidence
        if evidence is not None
        else (_kernel_evidence(obligation),),
        provider_id="provider:untrusted",
        provider_claimed_assurance=provider_claimed_assurance,
        freshness=freshness,
        finished_at=NOW.isoformat(),
    )


def _completion_evidence(
    receipt: ProofReceipt,
    obligation: CodeProofObligation,
    *,
    validation_passed: bool = True,
) -> CompletionEvidence:
    return CompletionEvidence.from_proof_receipt(
        acceptance_criterion=CRITERION,
        obligation=obligation,
        proof_receipt=receipt,
        validation_receipt={"attempted": True, "passed": validation_passed},
        validation_passed=validation_passed,
        observed_at=NOW,
    )


def test_trusted_receipt_maps_every_completion_identity_and_round_trips() -> None:
    obligation = _obligation()
    receipt = _receipt(obligation)
    evidence = _completion_evidence(receipt, obligation)
    payload = evidence.to_dict()

    assert evidence.obligation_id == obligation.obligation_id
    assert evidence.proof_receipt_id == receipt.receipt_id
    assert evidence.required_assurance is AssuranceLevel.KERNEL_VERIFIED
    assert evidence.authoritative_assurance is AssuranceLevel.KERNEL_VERIFIED
    assert evidence.proof_verdict is ProofVerdict.PROVED
    assert evidence.proof_freshness is EvidenceFreshness.CURRENT
    assert evidence.repository_tree == TREE
    assert evidence.provenance_cid == receipt.receipt_id
    assert payload["proof_receipt"] == receipt.to_dict()
    assert CompletionEvidence.from_dict(payload) == evidence

    result = validate_completion_evidence(
        evidence,
        repository_id=REPOSITORY,
        repository_tree=TREE,
        now=NOW,
    )
    assert result.valid is True
    assert result.validation_succeeded is True
    assert result.assurance_satisfied is True


def test_provider_claim_and_passing_validation_cannot_upgrade_inconclusive_proof() -> None:
    obligation = _obligation()
    candidate = ProofEvidence(
        kind=EvidenceKind.SMT_CANDIDATE,
        authority=EvidenceAuthority.PROVIDER,
        verdict=EvidenceVerdict.ACCEPTED,
        artifact_id="artifact:candidate",
        subject_id=obligation.obligation_id,
        verifier_id="provider:self-asserted",
        independent=True,
    )
    receipt = _receipt(obligation, evidence=(candidate,))
    evidence = _completion_evidence(receipt, obligation)

    result = validate_completion_evidence(evidence, repository_tree=TREE, now=NOW)
    decision = evaluate_goal_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        acceptance_criteria=(CRITERION,),
        evidence=(evidence,),
        tasks_complete=True,
        repository_tree=TREE,
        now=NOW,
        require_completion_gate=False,
    )

    assert result.validation_succeeded is True
    assert result.assurance_satisfied is False
    assert "inconclusive_proof" in result.reason_codes
    assert "required_assurance_not_satisfied" in result.reason_codes
    assert decision.verified is False
    assert decision.state is GoalState.PROVISIONALLY_COMPLETE


def test_assurance_is_evaluated_independently_from_failed_validation() -> None:
    obligation = _obligation()
    evidence = _completion_evidence(
        _receipt(obligation),
        obligation,
        validation_passed=False,
    )

    result = validate_completion_evidence(evidence, repository_tree=TREE, now=NOW)

    assert result.validation_succeeded is False
    assert result.assurance_satisfied is True
    assert result.valid is False
    assert "failed_validation" in result.reason_codes
    assert "required_assurance_not_satisfied" not in result.reason_codes


@pytest.mark.parametrize(
    ("receipt", "reason_code"),
    [
        (
            lambda obligation: _receipt(
                obligation,
                freshness=EvidenceFreshness.STALE,
            ),
            "stale_proof_receipt",
        ),
        (
            lambda obligation: _receipt(
                obligation,
                evidence=(
                    _kernel_evidence(
                        obligation,
                        verdict=EvidenceVerdict.UNSUPPORTED,
                        failure_code="unsupported_kernel",
                    ),
                ),
                verdict=ProofVerdict.UNSUPPORTED,
            ),
            "unsupported_proof",
        ),
        (
            lambda obligation: _receipt(
                obligation,
                evidence=(),
                verdict=ProofVerdict.INCONCLUSIVE,
            ),
            "inconclusive_proof",
        ),
    ],
)
def test_nonqualifying_proof_outcomes_fail_closed(receipt, reason_code: str) -> None:
    obligation = _obligation()
    evidence = _completion_evidence(receipt(obligation), obligation)

    result = validate_completion_evidence(evidence, repository_tree=TREE, now=NOW)

    assert result.assurance_satisfied is False
    assert reason_code in result.reason_codes


def test_embedded_receipt_and_summary_tampering_are_rejected() -> None:
    obligation = _obligation()
    payload = _completion_evidence(_receipt(obligation), obligation).to_dict()
    payload["authoritative_assurance"] = "attested"
    payload["assurance"] = "attested"
    evidence = CompletionEvidence.from_dict(payload)

    result = validate_completion_evidence(evidence, repository_tree=TREE, now=NOW)

    assert result.assurance_satisfied is False
    assert "proof_assurance_mismatch" in result.reason_codes


def test_parent_aggregates_and_rejects_stale_descendant_proof_requirement() -> None:
    obligation = _obligation()
    child_evidence = _completion_evidence(_receipt(obligation), obligation)
    child_result = validate_completion_evidence(
        child_evidence, repository_tree=TREE, now=NOW
    )
    assert child_result.valid is True
    # Exercise the persisted decision shape consumed by parent aggregation.
    requirement = {
        "goal_id": "G11.S7.1",
        "acceptance_criterion": CRITERION,
        "obligation_id": obligation.obligation_id,
        "proof_receipt_id": child_evidence.proof_receipt_id,
        "required_assurance": "kernel_verified",
        "authoritative_assurance": "kernel_verified",
        "proof_verdict": "proved",
        "freshness": "stale",
        "assurance_satisfied": True,
        "reason_codes": [],
    }
    child = {
        "goal_id": "G11.S7.1",
        "state": "verified_complete",
        "verified": True,
        "completion_gate": {"passed": True},
        "proof_requirements": [requirement],
    }

    gate = evaluate_completion_gate(
        acceptance_criteria=(),
        child_goals=(child,),
        repository_tree=TREE,
        now=NOW,
    )
    child_check = next(check for check in gate.checks if check.name == "child_goals")

    assert child_check.passed is False
    assert child_check.reason_code == "child_proof_stale"
    assert child_check.evidence["proof_requirements"] == [requirement]
    assert child_check.evidence["unsatisfied_proof_requirements"][0][
        "failure_codes"
    ] == ["child_proof_stale", "child_required_assurance_not_satisfied"]


@pytest.mark.parametrize(
    ("verdict", "freshness", "contradicted", "expected_code"),
    [
        ("unsupported", "current", False, "child_proof_unsupported"),
        ("inconclusive", "current", False, "child_proof_inconclusive"),
        ("disproved", "current", True, "child_proof_contradicted"),
        ("proved", "unknown", False, "child_proof_stale"),
    ],
)
def test_parent_cannot_hide_nonqualifying_nested_descendant_proof(
    verdict: str,
    freshness: str,
    contradicted: bool,
    expected_code: str,
) -> None:
    requirement = {
        "goal_id": "G11.S7.1.1",
        "acceptance_criterion": CRITERION,
        "obligation_id": "obligation:nested",
        "proof_receipt_id": "receipt:nested",
        "required_assurance": "kernel_verified",
        "authoritative_assurance": "kernel_verified",
        "proof_verdict": verdict,
        "freshness": freshness,
        "assurance_satisfied": True,
        "contradicted": contradicted,
    }
    nested = {
        "goal_id": "G11.S7.1.1",
        "state": "verified_complete",
        "verified": True,
        "completion_gate": {"passed": True},
        "proof_requirements": [requirement],
    }
    child = {
        "goal_id": "G11.S7.1",
        "state": "verified_complete",
        "verified": True,
        "completion_gate": {
            "passed": True,
            "evaluated_evidence": {"child_goals": [nested]},
        },
    }

    gate = evaluate_completion_gate(
        acceptance_criteria=(),
        child_goals=(child,),
        repository_tree=TREE,
        now=NOW,
    )
    child_check = next(check for check in gate.checks if check.name == "child_goals")

    assert child_check.passed is False
    assert child_check.reason_code == expected_code
    assert child_check.evidence["unsatisfied_proof_requirements"][0]["goal_id"] == (
        "G11.S7.1.1"
    )


def test_legacy_assurance_claim_is_readable_but_never_optimistically_upgraded() -> None:
    legacy = CompletionEvidence.from_dict(
        {
            "version": 0,
            "criterion": CRITERION,
            "task_id": "REF-legacy",
            "validation": {"attempted": True, "passed": True},
            "tree_identity": TREE,
            "fresh": True,
            "generated_at": NOW.isoformat(),
            "receipt_cid": "bafy-legacy-validation",
            "proof_obligation_id": "forged-obligation",
            "proof_receipt_id": "forged-receipt",
            "required_assurance": "kernel_verified",
            "assurance": "attested",
            "proof_verdict": "proved",
            "proof_freshness": "current",
        }
    )

    result = validate_completion_evidence(legacy, repository_tree=TREE, now=NOW)

    assert legacy.metadata["source_schema_version"] == 0
    assert legacy.acceptance_criterion == CRITERION
    assert legacy.assurance_satisfied is False
    assert result.validation_succeeded is True
    assert result.assurance_satisfied is False
    assert "untrusted_proof_reference" in result.reason_codes


def test_formal_completion_reference_uses_authoritative_not_provider_assurance() -> None:
    obligation = _obligation()
    receipt = _receipt(
        obligation,
        evidence=(),
        provider_claimed_assurance=AssuranceLevel.ATTESTED,
    )

    reference = receipt.completion_reference(AssuranceLevel.KERNEL_VERIFIED)

    assert reference["authoritative_assurance"] == "unverified"
    assert reference["authoritative_verdict"] == "inconclusive"
    assert reference["assurance_satisfied"] is False
    assert receipt.satisfies_completion(AssuranceLevel.KERNEL_VERIFIED) is False


def _fresh_implementation_manifest(
    *,
    repository_tree: str = TREE,
    assumptions: tuple[str, ...] = ("assumption:lease-owner-is-current",),
) -> ImplementationObligationSet:
    scope_set = compile_candidate_proof_scopes(
        (
            {
                "new_path": "src/lease.py",
                "status": "add",
                "after_source": """\
class Lease:
    def mutate(self, token: int) -> bool:
        self.token = token
        return validate_fencing_token(token)
""",
            },
        )
    )
    bounds = {
        "python_versions": ["3.11", "3.12"],
        "tests": ["test/test_lease.py"],
        "runtime_scenarios": ["expired_token", "current_token"],
    }
    evidence_common = {
        "accepted_plan_id": "plan:ref-267",
        "repository_id": REPOSITORY,
        "repository_tree_id": repository_tree,
        "scope_ids": scope_set.scope_ids,
        "passed": True,
        "observed_at": NOW,
        "validation_bounds": bounds,
        "assumption_ids": assumptions,
    }
    test_evidence = ImplementationResultEvidence(
        kind=ImplementationEvidenceKind.TEST,
        producer_id="pytest:locked",
        artifact_id="artifact:test-run:lease-fencing",
        **evidence_common,
    )
    runtime_evidence = ImplementationResultEvidence(
        kind=ImplementationEvidenceKind.RUNTIME,
        producer_id="runtime-harness:lease-contention",
        artifact_id="artifact:runtime:lease-contention",
        **evidence_common,
    )
    return derive_fresh_implementation_obligations(
        scope_set,
        accepted_plan_id="plan:ref-267",
        repository_id=REPOSITORY,
        repository_tree_id=repository_tree,
        assumption_ids=assumptions,
        validation_bounds=bounds,
        test_evidence=(test_evidence,),
        runtime_evidence=(runtime_evidence,),
        planned_effect_ids=("effect:reject-stale-fencing-token",),
        task_id="REF-267",
    )


def _bound_code_receipt(
    obligation: CodeProofObligation,
    binding: ImplementationResultBinding,
    *,
    freshness: EvidenceFreshness = EvidenceFreshness.CURRENT,
    contradicted: bool = False,
) -> ProofReceipt:
    if contradicted:
        evidence = (
            ProofEvidence(
                kind=EvidenceKind.SOLVER_RESULT,
                authority=EvidenceAuthority.SOLVER,
                verdict=EvidenceVerdict.REJECTED,
                artifact_id="artifact:verified-counterexample",
                subject_id=obligation.obligation_id,
                verifier_id="solver:z3@4",
                independent=True,
                metadata={"counterexample_verified": True},
            ),
        )
        verdict = ProofVerdict.DISPROVED
    else:
        evidence = (_kernel_evidence(obligation),)
        verdict = ProofVerdict.PROVED
    return ProofReceipt(
        obligation_id=obligation.obligation_id,
        plan_id=binding.accepted_plan_id,
        attempt_id="attempt:ref-267:fresh-code-proof",
        repository_id=binding.repository_id,
        repository_tree_id=binding.repository_tree_id,
        ast_scope_ids=obligation.ast_scope_ids,
        premise_ids=obligation.premise_ids,
        translator_id="translator:python-lean@1",
        solver_id="solver:z3@4",
        kernel_id="kernel:lean-4.19",
        toolchain_id="toolchain:locked",
        policy_id="policy:critical-proof",
        resource_budget=ResourceBudget(wall_time_ms=30_000),
        verdict=verdict,
        evidence=evidence,
        freshness=freshness,
        finished_at=NOW.isoformat(),
        metadata=binding.receipt_metadata(obligation=obligation),
    )


def test_verified_decomposition_receipt_cannot_be_reused_as_code_proof() -> None:
    manifest = _fresh_implementation_manifest()
    binding = manifest.binding
    obligation = manifest.obligations[0]
    decomposition_receipt_id = "receipt:verified-decomposition"
    plan_assurance = PlanAssurance(
        plan_id=binding.accepted_plan_id,
        consistency=PlanConsistencyLevel.KERNEL_VERIFIED,
        consistency_receipt_ids=(decomposition_receipt_id,),
        bounds={"decomposition_depth": 4},
    )
    planning_receipt = {
        "receipt_id": decomposition_receipt_id,
        "obligation_id": obligation.obligation_id,
        "kind": "verified_decomposition",
        "verdict": "proved",
        "repository_tree_id": TREE,
    }

    decision = evaluate_code_proof_goal_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        binding=binding,
        required_obligations=manifest,
        receipts=(planning_receipt,),
        plan_assurance=plan_assurance,
    )

    assert decision.verified is False
    assert decision.state is GoalState.PROVISIONALLY_COMPLETE
    assert decision.satisfied_obligation_ids == ()
    assert set(decision.unsatisfied_obligation_ids) == set(manifest.obligation_ids)
    assert "plan_receipt_reused_as_code_proof" in decision.reason_codes
    assert "code_proof_obligations_unsatisfied" in decision.reason_codes


@pytest.mark.parametrize(
    ("receipt", "claim_kind"),
    [
        (
            {
                "receipt_id": "receipt:model-self-proof",
                "kind": "model_generated_proof_claim",
                "verdict": "proved",
                "provider_claimed_assurance": "kernel_verified",
                "passed": True,
            },
            "model_generated_proof_claim",
        ),
        (
            {
                "receipt_id": "receipt:type-check",
                "kind": "type_check",
                "verdict": "passed",
                "passed": True,
                "errors": 0,
            },
            "type_check",
        ),
    ],
)
def test_noncanonical_model_claim_or_type_check_alone_cannot_prove_code(
    receipt: dict[str, object],
    claim_kind: str,
) -> None:
    manifest = _fresh_implementation_manifest()
    binding = manifest.binding
    obligation = manifest.obligations[0]
    receipt["obligation_id"] = obligation.obligation_id

    decision = evaluate_code_proof_goal_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        binding=binding,
        required_obligations=manifest,
        receipts=(receipt,),
    )

    assert decision.verified is False, claim_kind
    assert decision.state is GoalState.PROVISIONALLY_COMPLETE
    assert set(decision.unsatisfied_obligation_ids) == set(manifest.obligation_ids)
    assert "invalid_code_proof_receipt" in decision.reason_codes
    assert "code_proof_obligations_unsatisfied" in decision.reason_codes
    assert decision.receipt_results[0]["valid"] is False


def test_cached_receipt_for_old_tree_cannot_cover_fresh_obligation_set() -> None:
    manifest = _fresh_implementation_manifest()
    old_manifest = _fresh_implementation_manifest(repository_tree="git-tree:ref-266")
    cached_receipts = tuple(
        _bound_code_receipt(obligation, old_manifest.binding)
        for obligation in old_manifest.obligations
    )

    decision = evaluate_code_proof_goal_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        binding=manifest.binding,
        required_obligations=manifest,
        receipts=cached_receipts,
    )

    assert decision.verified is False
    assert decision.reopened is True
    assert decision.stale is True
    assert decision.satisfied_obligation_ids == ()
    assert "receipt_not_required_by_fresh_obligation_set" in decision.reason_codes
    assert any(
        "receipt_not_required_by_fresh_obligation_set"
        in result["reason_codes"]
        for result in decision.receipt_results
    )


def test_fresh_fully_bound_code_receipt_verifies_provisional_goal() -> None:
    manifest = _fresh_implementation_manifest()
    binding = manifest.binding
    receipts = tuple(
        _bound_code_receipt(obligation, binding)
        for obligation in manifest.obligations
    )

    decision = evaluate_code_proof_goal_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        binding=binding,
        required_obligations=manifest,
        receipts=receipts,
    )

    assert decision.verified is True
    assert decision.state is GoalState.VERIFIED_COMPLETE
    assert decision.binding_id == binding.binding_id
    assert decision.all_obligations_satisfied is True
    assert set(decision.satisfied_obligation_ids) == set(manifest.obligation_ids)
    assert decision.unsatisfied_obligation_ids == ()
    assert all(result["valid"] is True for result in decision.receipt_results)
    assert {
        result["binding_id"] for result in decision.receipt_results
    } == {binding.binding_id}


@pytest.mark.parametrize(
    "current_state",
    [GoalState.PROVISIONALLY_COMPLETE, GoalState.VERIFIED_COMPLETE],
)
@pytest.mark.parametrize(
    ("receipt_kwargs", "expected_reason", "expected_flag"),
    [
        (
            {"freshness": EvidenceFreshness.STALE},
            "stale_proof_receipt",
            "stale",
        ),
        (
            {"contradicted": True},
            "code_proof_not_proved",
            "contradictory",
        ),
    ],
)
def test_completed_goal_reopens_when_required_code_binding_is_stale_or_contradictory(
    current_state: GoalState,
    receipt_kwargs: dict[str, object],
    expected_reason: str,
    expected_flag: str,
) -> None:
    manifest = _fresh_implementation_manifest()
    binding = manifest.binding
    target = manifest.obligations[0]
    receipts = tuple(
        _bound_code_receipt(
            obligation,
            binding,
            **(receipt_kwargs if obligation is target else {}),
        )
        for obligation in manifest.obligations
    )

    decision = evaluate_code_proof_goal_completion(
        current_state=current_state,
        binding=binding,
        required_obligations=manifest,
        receipts=receipts,
    )

    assert decision.state is GoalState.REOPENED
    assert decision.reopened is True
    assert decision.verified is False
    assert decision.unsatisfied_obligation_ids == (target.obligation_id,)
    assert getattr(decision, expected_flag) is True
    assert expected_reason in decision.reason_codes
    assert "code_proof_completion_reopened" in decision.reason_codes
