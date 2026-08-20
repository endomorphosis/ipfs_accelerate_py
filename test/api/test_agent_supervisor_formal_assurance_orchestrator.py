"""FACP-050: proof cache and solver orchestration."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.formal_assurance_orchestrator import (
    ANALYZER_VERSION,
    BUNDLE,
    GOAL_ID,
    INTERFACE,
    PROHIBITED_ASSURANCE_STAGES,
    PROOF_CACHE_KEY_SCHEMA,
    PROOF_ROUTER_SCHEMA,
    RESULT_SCHEMA,
    SCHEMA,
    SOLVER_CONFLICT_SCHEMA,
    TASK_ID,
    TOOLCHAIN_ID,
    CacheReuseDerivation,
    CacheReuseKind,
    CapsuleBoundProofCache,
    CapsuleBoundProofCacheKey,
    ConflictKind,
    EscalationStage,
    FormalAssuranceOrchestrator,
    OrchestratorError,
    OrchestratorVerdict,
    StageAttempt,
    StageOutcome,
    build_capsule_bound_cache_key,
    build_conflict,
    escalation_ladder,
    next_stronger_stage,
    orchestrate_obligation,
    stage_for_property,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_cache import (
    CacheLookupStatus,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    AssuranceLevel,
)
from ipfs_accelerate_py.agent_supervisor.proof.multi_prover_router import (
    AttemptOutcome,
    PropertyKind,
    PropertyObligation,
    ProverOutput,
    ProverRole,
)


def _obligation(
    kind: PropertyKind = PropertyKind.FINITE_CONSTRAINT,
    *,
    obligation_id: str = "obligation:facp050",
    premises: tuple[str, ...] = ("assumption:env-bound", "assumption:spec-closed"),
) -> PropertyObligation:
    return PropertyObligation(
        obligation_id=obligation_id,
        property_kind=kind,
        statement=f"reviewed {kind.value} obligation",
        premise_ids=premises,
    )


def _key(
    *,
    capsule_cid: str = "semantic-capsule:sha256:facp050-capsule",
    tactic_id: str = "tactic:default",
    solver_id: str = "solver:z3",
    translation_receipt_cid: str = "",
    assumptions: tuple[str, ...] = ("assumption:env-bound", "assumption:spec-closed"),
    claim_id: str = "claim:facp050",
    code_id: str = "code:facp050",
) -> CapsuleBoundProofCacheKey:
    return build_capsule_bound_cache_key(
        claim_id=claim_id,
        spec_id="spec:facp050",
        code_id=code_id,
        assumptions=assumptions,
        environment_id="environment:facp050",
        solver_id=solver_id,
        revision_id="revision:facp050",
        tactic_id=tactic_id,
        capsule_cid=capsule_cid,
        translation_receipt_cid=translation_receipt_cid,
        property_kind=PropertyKind.FINITE_CONSTRAINT,
    )


def _authoritative_runner(
    verified_stage: EscalationStage = EscalationStage.SMT,
    *,
    outcome: StageOutcome = StageOutcome.VERIFIED,
):
    def runner(
        stage: EscalationStage,
        obligation: PropertyObligation,
        cache_key: CapsuleBoundProofCacheKey,
    ) -> StageAttempt:
        del obligation, cache_key
        if stage is verified_stage:
            return StageAttempt(
                stage=stage,
                outcome=outcome,
                verifier=f"{stage.value}-authority",
                cost=stage.default_cost,
                detail=f"{stage.value} conclusive",
                authoritative=True,
            )
        return StageAttempt(
            stage=stage,
            outcome=StageOutcome.UNKNOWN,
            verifier=f"{stage.value}-probe",
            cost=stage.default_cost,
            detail="inconclusive",
            authoritative=False,
        )

    return runner


def test_evidence_envelope_and_ladder_are_stable() -> None:
    assert TASK_ID == "FACP-050"
    assert GOAL_ID == "FACP-G640"
    assert BUNDLE == "facp/proof/orchestration"
    assert SCHEMA == "facp/proof-orchestration@1"
    assert PROOF_ROUTER_SCHEMA == "facp/proof-router@1"
    assert PROOF_CACHE_KEY_SCHEMA == "facp/proof-cache-key@1"
    assert SOLVER_CONFLICT_SCHEMA == "facp/solver-conflict@1"
    assert RESULT_SCHEMA == "facp/proof-orchestration-result@1"
    assert INTERFACE == "FormalAssuranceOrchestrator@1"
    assert ANALYZER_VERSION
    assert TOOLCHAIN_ID
    assert "llm" in PROHIBITED_ASSURANCE_STAGES

    ladder = escalation_ladder()
    assert ladder[0] is EscalationStage.SCHEMA
    assert ladder[-1] is EscalationStage.HUMAN
    assert [stage.value for stage in ladder] == [
        "schema",
        "abstract_interpretation",
        "datalog",
        "egraph",
        "smt",
        "alloy",
        "tla",
        "specialized",
        "lean",
        "human",
    ]
    assert stage_for_property(PropertyKind.FINITE_CONSTRAINT) is EscalationStage.SMT
    assert stage_for_property(PropertyKind.AUTHORIZATION) is EscalationStage.DATALOG
    assert next_stronger_stage(EscalationStage.SMT) is EscalationStage.ALLOY
    assert next_stronger_stage(EscalationStage.HUMAN) is None


def test_every_result_names_assumptions_verifier_and_toolchain() -> None:
    orch = FormalAssuranceOrchestrator()
    result = orch.execute(obligation=_obligation(), cache_key=_key())

    payload = result.to_dict()
    assert payload["assumptions"] == [
        "assumption:env-bound",
        "assumption:spec-closed",
    ]
    assert payload["verifier"]
    assert payload["toolchain"] == TOOLCHAIN_ID
    assert payload["schema"] == RESULT_SCHEMA
    assert payload["router_schema"] == PROOF_ROUTER_SCHEMA
    assert result.verdict is OrchestratorVerdict.NONVERIFIED
    assert result.nonverified is True
    assert result.assurance is AssuranceLevel.UNVERIFIED


def test_unknown_and_unavailable_remain_nonverified() -> None:
    def runner(stage, obligation, cache_key):
        outcome = (
            StageOutcome.UNKNOWN
            if stage is EscalationStage.SMT
            else StageOutcome.UNAVAILABLE
        )
        return StageAttempt(
            stage=stage,
            outcome=outcome,
            verifier=f"{stage.value}-backend",
            cost=stage.default_cost,
            detail=outcome.value,
            # Attempted promotion must be stripped.
            authoritative=True,
        )

    result = FormalAssuranceOrchestrator().execute(
        obligation=_obligation(),
        cache_key=_key(),
        stage_runner=runner,
        max_stage=EscalationStage.ALLOY,
    )

    assert result.verdict is OrchestratorVerdict.NONVERIFIED
    assert result.assurance is AssuranceLevel.UNVERIFIED
    assert all(not attempt.authoritative for attempt in result.attempts)
    assert all(
        attempt.outcome in (StageOutcome.UNKNOWN, StageOutcome.UNAVAILABLE)
        for attempt in result.attempts
    )


def test_candidate_verified_report_cannot_self_promote() -> None:
    def runner(stage, obligation, cache_key):
        if stage is EscalationStage.SCHEMA:
            return StageAttempt(
                stage=stage,
                outcome=StageOutcome.VERIFIED,
                verifier="schema-candidate",
                cost=1,
                detail="schema claimed verified",
                authoritative=False,
            )
        if stage is EscalationStage.SMT:
            return StageAttempt(
                stage=stage,
                outcome=StageOutcome.CANDIDATE,
                verifier="smt-candidate",
                cost=8,
                detail="candidate only",
                authoritative=False,
            )
        return StageAttempt(
            stage=stage,
            outcome=StageOutcome.UNAVAILABLE,
            verifier=stage.value,
            cost=stage.default_cost,
        )

    result = FormalAssuranceOrchestrator().execute(
        obligation=_obligation(),
        cache_key=_key(),
        stage_runner=runner,
        max_stage=EscalationStage.SMT,
    )

    assert result.verdict is OrchestratorVerdict.NONVERIFIED
    assert result.assurance is AssuranceLevel.UNVERIFIED


def test_authoritative_stage_can_verify_and_measure_cost() -> None:
    orch = FormalAssuranceOrchestrator()
    result = orch.execute(
        obligation=_obligation(),
        cache_key=_key(),
        stage_runner=_authoritative_runner(EscalationStage.SMT),
    )

    assert result.verdict is OrchestratorVerdict.VERIFIED
    assert result.assurance is AssuranceLevel.SOLVER_CHECKED
    assert result.verifier == "smt-authority"
    assert result.toolchain == TOOLCHAIN_ID
    assert result.cumulative_cost == EscalationStage.SMT.default_cost
    assert result.stage is EscalationStage.SMT


def test_stronger_escalation_includes_reason_and_cost() -> None:
    def runner(stage, obligation, cache_key):
        if stage is EscalationStage.SMT:
            return StageAttempt(
                stage=stage,
                outcome=StageOutcome.UNKNOWN,
                verifier="z3",
                cost=8,
                detail="timeout",
            )
        if stage is EscalationStage.LEAN:
            return StageAttempt(
                stage=stage,
                outcome=StageOutcome.VERIFIED,
                verifier="lean-kernel",
                cost=40,
                detail="reconstructed",
                authoritative=True,
            )
        return StageAttempt(
            stage=stage,
            outcome=StageOutcome.UNAVAILABLE,
            verifier=stage.value,
            cost=stage.default_cost,
        )

    result = FormalAssuranceOrchestrator().execute(
        obligation=_obligation(),
        cache_key=_key(),
        stage_runner=runner,
    )

    assert result.verdict is OrchestratorVerdict.VERIFIED
    assert result.assurance is AssuranceLevel.KERNEL_VERIFIED
    assert result.escalations
    first = result.escalations[0]
    assert first.from_stage is EscalationStage.SMT
    assert first.to_stage.rank > EscalationStage.SMT.rank
    assert first.reason
    assert first.cost > 0
    assert first.cumulative_cost >= first.cost
    assert first.to_dict()["schema"] == "facp/proof-escalation@1"

    receipt = FormalAssuranceOrchestrator().escalate(
        EscalationStage.SMT,
        reason="smt inconclusive; need kernel reconstruction",
        cumulative_cost=8,
        to_stage=EscalationStage.LEAN,
    )
    assert receipt.cost == EscalationStage.LEAN.default_cost
    assert receipt.cumulative_cost == 8 + EscalationStage.LEAN.default_cost
    with pytest.raises(OrchestratorError, match="strictly stronger"):
        FormalAssuranceOrchestrator().escalate(
            EscalationStage.LEAN,
            reason="noop",
            to_stage=EscalationStage.SMT,
        )


def test_llm_is_never_an_assurance_stage() -> None:
    orch = FormalAssuranceOrchestrator()
    plan = orch.plan_route(_obligation())
    assert "llm" not in plan["ladder"]
    assert "llm" in plan["prohibited_stages"]
    with pytest.raises(OrchestratorError):
        orch.escalate(EscalationStage.SMT, reason="ask model", to_stage="llm")


def test_cache_reuse_requires_unchanged_or_equivalent_derivation() -> None:
    orch = FormalAssuranceOrchestrator()
    key = _key()
    proved = orch.execute(
        obligation=_obligation(),
        cache_key=key,
        stage_runner=_authoritative_runner(),
    )
    assert proved.verdict is OrchestratorVerdict.VERIFIED

    hit = orch.execute(obligation=_obligation(), cache_key=key)
    assert hit.verdict is OrchestratorVerdict.CACHE_HIT
    assert hit.cache_reuse is not None
    assert hit.cache_reuse.kind is CacheReuseKind.UNCHANGED
    assert hit.cache_reuse.explanation
    assert hit.cache_reuse.prior_closure_id == key.semantic_closure_id
    assert hit.assumptions
    assert hit.verifier
    assert hit.toolchain

    # Changed capsule closure without equivalence → recompute, never silent reuse.
    changed = _key(capsule_cid="semantic-capsule:sha256:changed-closure")
    recomputed = orch.execute(
        obligation=_obligation(),
        cache_key=changed,
        stage_runner=_authoritative_runner(),
    )
    assert recomputed.verdict is OrchestratorVerdict.VERIFIED
    assert recomputed.cache_reuse is None
    assert recomputed.cache_key.capsule_cid.endswith("changed-closure")


def test_equivalent_cache_reuse_with_derivation_path() -> None:
    cache = CapsuleBoundProofCache()
    orch = FormalAssuranceOrchestrator(cache=cache)
    prior_key = _key(tactic_id="tactic:alpha")
    proved = orch.execute(
        obligation=_obligation(),
        cache_key=prior_key,
        stage_runner=_authoritative_runner(),
    )
    assert proved.verdict is OrchestratorVerdict.VERIFIED

    current_key = _key(tactic_id="tactic:beta")
    assert current_key.key_id != prior_key.key_id
    assert current_key.semantic_closure_id == prior_key.semantic_closure_id

    derivation = CacheReuseDerivation(
        kind=CacheReuseKind.EQUIVALENT,
        explanation=(
            "tactic rewrite is proof-irrelevant; claim/spec/code/assumptions/"
            "environment/capsule closure unchanged"
        ),
        prior_key_id=prior_key.key_id,
        current_key_id=current_key.key_id,
        prior_closure_id=prior_key.semantic_closure_id,
        current_closure_id=current_key.semantic_closure_id,
        path=("tactic_id", "proof_irrelevance"),
    )
    status, reused, recorded = cache.lookup(current_key, equivalence=derivation)
    assert status is CacheLookupStatus.HIT
    assert reused is not None
    assert reused.verdict is OrchestratorVerdict.CACHE_HIT
    assert recorded is derivation

    via_execute = orch.execute(
        obligation=_obligation(),
        cache_key=current_key,
        equivalence=derivation,
    )
    assert via_execute.verdict is OrchestratorVerdict.CACHE_HIT
    assert via_execute.cache_reuse is not None
    assert via_execute.cache_reuse.kind is CacheReuseKind.EQUIVALENT


def test_changed_closure_rejects_equivalence_without_closure_path() -> None:
    orch = FormalAssuranceOrchestrator()
    prior_key = _key()
    orch.execute(
        obligation=_obligation(),
        cache_key=prior_key,
        stage_runner=_authoritative_runner(),
    )
    current_key = _key(capsule_cid="semantic-capsule:sha256:other")
    bad = CacheReuseDerivation(
        kind=CacheReuseKind.EQUIVALENT,
        explanation="claimed equivalent despite capsule change",
        prior_key_id=prior_key.key_id,
        current_key_id=current_key.key_id,
        prior_closure_id=prior_key.semantic_closure_id,
        current_closure_id=current_key.semantic_closure_id,
        path=("tactic_id",),
    )
    status, reused, _ = orch.cache.lookup(current_key, equivalence=bad)
    assert status is CacheLookupStatus.REJECTED
    assert reused is None

    conflicted = orch.execute(
        obligation=_obligation(),
        cache_key=current_key,
        equivalence=bad,
        stage_runner=_authoritative_runner(),
    )
    assert conflicted.verdict is OrchestratorVerdict.CONFLICT
    assert conflicted.conflict is not None
    assert conflicted.conflict.kind is ConflictKind.CACHE_CLOSURE_MISMATCH
    assert conflicted.conflict.to_dict()["schema"] == SOLVER_CONFLICT_SCHEMA


def test_stage_disagreement_creates_conflict_record() -> None:
    def runner(stage, obligation, cache_key):
        if stage is EscalationStage.SMT:
            return StageAttempt(
                stage=stage,
                outcome=StageOutcome.VERIFIED,
                verifier="z3",
                cost=8,
                detail="proved",
                authoritative=True,
            )
        if stage is EscalationStage.ALLOY:
            return StageAttempt(
                stage=stage,
                outcome=StageOutcome.DISPROVED,
                verifier="alloy",
                cost=12,
                detail="counterexample",
                authoritative=True,
            )
        return StageAttempt(
            stage=stage,
            outcome=StageOutcome.UNAVAILABLE,
            verifier=stage.value,
            cost=stage.default_cost,
        )

    # Force both conclusive stages by not stopping early.
    result = FormalAssuranceOrchestrator().execute(
        obligation=_obligation(),
        cache_key=_key(),
        stage_runner=runner,
        stop_on_verified=False,
        max_stage=EscalationStage.ALLOY,
    )

    assert result.verdict is OrchestratorVerdict.CONFLICT
    assert result.conflict is not None
    assert result.conflict.kind is ConflictKind.AUTHORITY_DISAGREEMENT
    assert result.assumptions
    assert result.verifier
    assert result.toolchain
    assert result.assurance is AssuranceLevel.UNVERIFIED
    assert result.nonverified is True


def test_build_conflict_helper_names_identity_bindings() -> None:
    left = StageAttempt(
        stage=EscalationStage.SMT,
        outcome=StageOutcome.VERIFIED,
        verifier="z3",
        cost=8,
        authoritative=True,
    )
    right = StageAttempt(
        stage=EscalationStage.LEAN,
        outcome=StageOutcome.DISPROVED,
        verifier="lean",
        cost=40,
        authoritative=True,
    )
    conflict = build_conflict(
        obligation_id="obligation:facp050",
        left=left,
        right=right,
        assumptions=("assumption:env-bound",),
    )
    payload = conflict.to_dict()
    assert payload["schema"] == SOLVER_CONFLICT_SCHEMA
    assert payload["verdict"] == OrchestratorVerdict.CONFLICT.value
    assert payload["assumptions"] == ["assumption:env-bound"]
    assert payload["verifier"]
    assert payload["toolchain"]


def test_cache_key_projects_into_legacy_proof_cache_key() -> None:
    key = _key(translation_receipt_cid="translation-receipt:sha256:tvc")
    legacy = key.to_legacy_proof_cache_key()
    assert legacy.key_id.startswith("proof-cache-key:sha256:")
    assert key.schema == PROOF_CACHE_KEY_SCHEMA
    assert key.to_dict()["key_id"] == key.key_id
    round_trip = CapsuleBoundProofCacheKey.from_dict(key.to_dict())
    assert round_trip.key_id == key.key_id


def test_plan_route_emits_proof_router_evidence() -> None:
    plan = FormalAssuranceOrchestrator().plan_route(
        _obligation(PropertyKind.AUTHORIZATION)
    )
    assert plan["schema"] == PROOF_ROUTER_SCHEMA
    assert plan["entry_stage"] == EscalationStage.DATALOG.value
    assert plan["ladder"][0] == "datalog"
    assert plan["task_id"] == TASK_ID
    assert plan["goal_id"] == GOAL_ID
    assert "plan_id" in plan


def test_portfolio_composition_keeps_candidates_nonverified() -> None:
    orch = FormalAssuranceOrchestrator()

    def planning_runner(request, cancel):
        del cancel
        if request.lane.role is ProverRole.KERNEL:
            return ProverOutput(AttemptOutcome.UNKNOWN, "kernel could not reconstruct")
        return ProverOutput(
            AttemptOutcome.VERIFIED,
            evidence={"premises": list(request.obligation.premise_ids)},
        )

    result = orch.execute_with_portfolio(
        obligation=_obligation(PropertyKind.TYPED_PLANNING),
        cache_key=_key(),
        portfolio_runner=planning_runner,
    )
    assert result.verdict is OrchestratorVerdict.NONVERIFIED
    assert result.assurance is AssuranceLevel.UNVERIFIED
    assert result.assumptions
    assert result.verifier
    assert result.toolchain


def test_portfolio_disagreement_creates_conflict() -> None:
    orch = FormalAssuranceOrchestrator()

    def disagreeing_runner(request, cancel):
        del cancel
        # Authoritative lanes report opposite conclusive outcomes.
        if request.lane.prover_id == "z3":
            return ProverOutput(AttemptOutcome.VERIFIED)
        if request.lane.prover_id == "cvc5":
            return ProverOutput(AttemptOutcome.COUNTEREXAMPLE)
        return ProverOutput(AttemptOutcome.UNKNOWN)

    result = orch.execute_with_portfolio(
        obligation=_obligation(PropertyKind.FINITE_CONSTRAINT),
        cache_key=_key(),
        portfolio_runner=disagreeing_runner,
    )
    assert result.verdict is OrchestratorVerdict.CONFLICT
    assert result.conflict is not None
    assert result.conflict.kind is ConflictKind.AUTHORITY_DISAGREEMENT
    assert result.nonverified is True
    assert result.assurance is AssuranceLevel.UNVERIFIED
    assert result.assumptions
    assert result.verifier
    assert result.toolchain


def test_orchestrate_obligation_convenience_entrypoint() -> None:
    result = orchestrate_obligation(
        obligation=_obligation(),
        cache_key=_key(),
        stage_runner=_authoritative_runner(),
    )
    assert result.verdict is OrchestratorVerdict.VERIFIED
    assert result.to_dict()["bundle"] == BUNDLE


def test_nonconclusive_results_are_not_cached_as_proofs() -> None:
    cache = CapsuleBoundProofCache()
    orch = FormalAssuranceOrchestrator(cache=cache)
    key = _key()
    result = orch.execute(obligation=_obligation(), cache_key=key)
    assert result.verdict is OrchestratorVerdict.NONVERIFIED
    status, reused, _ = cache.lookup(key)
    assert status is CacheLookupStatus.MISS
    assert reused is None


def test_schema_stage_cannot_claim_authority() -> None:
    def runner(stage, obligation, cache_key):
        return StageAttempt(
            stage=stage,
            outcome=StageOutcome.VERIFIED,
            verifier="schema",
            cost=1,
            authoritative=True,
        )

    with pytest.raises(OrchestratorError, match="cannot claim authoritative"):
        FormalAssuranceOrchestrator().execute(
            obligation=_obligation(PropertyKind.RUNTIME_TRACE),
            cache_key=_key(),
            stage_runner=runner,
            max_stage=EscalationStage.ABSTRACT_INTERPRETATION,
        )
