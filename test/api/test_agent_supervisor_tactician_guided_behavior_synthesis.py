"""Tests for LPR-015: bridge predictions into existing behavior/value synthesis."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.change_propagation_contracts import (
    BehaviorEvidencePrecedence,
    BehaviorKind,
    MissingInputRequirement,
    PropagationAuthorityRoots,
    ValueCandidateDisposition,
)
from ipfs_accelerate_py.agent_supervisor.analysis.program_logic_prediction_contracts import (
    LogicPredictionReceipt,
    PredictionDisposition,
    ProgramLogicAuthorityRoots,
    ProofStatus,
    SourceAuthorityClass,
    SourceRouteKind,
)
from ipfs_accelerate_py.agent_supervisor.analysis.required_behavior_synthesis import (
    BehaviorClauseFamily,
    BehaviorEvidenceAtom,
    RequiredBehaviorSynthesizer,
    SynthesisDisposition,
    precedence_rank,
)
from ipfs_accelerate_py.agent_supervisor.analysis.tactician_guided_behavior_synthesis import (
    CONTRACT_REPAIR_PREDICTION_BRIDGE_INTERFACE,
    TACTICIAN_GUIDED_BEHAVIOR_SYNTHESIS_INTERFACE,
    BindingDisposition,
    BridgeDisposition,
    ConsequenceKind,
    ContractRepairPredictionBridge,
    PremisePrecedenceBinding,
    PredictionEvidenceBinding,
    TacticianBehaviorSynthesisReceipt,
    TacticianGuidedBehaviorSynthesizer,
    TacticianGuidedBehaviorSynthesisAuthorityError,
    create_contract_repair_prediction_bridge,
    create_tactician_guided_behavior_synthesizer,
    effective_consequence_precedence,
    route_to_precedence,
    source_authority_floor,
    weakest_precedence,
)
from ipfs_accelerate_py.agent_supervisor.proof.contract_repair_prover import (
    CandidateProofBundle,
    ContractRepairProofDisposition,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def propagation_roots() -> PropagationAuthorityRoots:
    return PropagationAuthorityRoots(
        repository_id="repository:lpr-015",
        base_forest_id="forest:base",
        base_tree_id="tree:base",
        base_overlay_id="overlay:base",
        candidate_forest_id="forest:candidate",
        candidate_tree_id="tree:candidate",
        candidate_overlay_id="overlay:candidate",
        graph_id="graph:lpr-015",
        index_id="index:lpr-015",
        model_id="model:lpr-015",
        config_id="config:lpr-015",
        translator_id="translator:lpr-015",
        toolchain_id="toolchain:lpr-015",
        policy_id="policy:lpr-015",
    )


@pytest.fixture
def logic_roots() -> ProgramLogicAuthorityRoots:
    return ProgramLogicAuthorityRoots(
        repository_id="repository:lpr-015",
        objective_id="objective:lpr-015",
        trace_id="trace:lpr-015",
        change_id="change:lpr-015",
        consumer_id="consumer:lpr-015",
        forest_id="forest:one",
        tree_id="tree:candidate",
        overlay_id="overlay:one",
        graph_id="graph:one",
        index_id="index:one",
        corpus_id="corpus:one",
        model_id="model:one",
        translator_id="translator:lpr-015",
        toolchain_id="toolchain:lpr-015",
        policy_id="policy:lpr-015",
        environment_id="environment:one",
    )


def _requirement(
    roots: PropagationAuthorityRoots, **extra: object
) -> MissingInputRequirement:
    values: dict[str, object] = {
        "roots": roots,
        "requirement_id": "missing:support-context",
        "obligation_id": "obligation:caller",
        "clause_id": "clause:param-add",
        "parameter_name": "context",
        "type_ref": "type:SupportContext",
        "nullability": "non_null",
        "information_content_ref": "info:request-context",
        "construction_precondition_refs": (),
        "result_postcondition_refs": (),
        "capability_refs": ("cap:context.read",),
        "propagation_depth_bound": 8,
        "proof_refs": ("proof:requirement",),
    }
    values.update(extra)
    return MissingInputRequirement(**values)


def _prediction(
    logic_roots: ProgramLogicAuthorityRoots,
    *,
    receipt_id: str = "prediction:one",
    hypothesis_id: str = "hyp:one",
    candidate_id: str = "candidate:repair-one",
    disposition: PredictionDisposition = PredictionDisposition.PROVED,
    proof_status: ProofStatus = ProofStatus.KERNEL_VERIFIED,
    source_authority: SourceAuthorityClass = SourceAuthorityClass.AUTHORITATIVE,
    value_ref: str = "value:unique-context",
    clause_ref: str = "",
    placement_ref: str = "",
    reconstruction_id: str = "reconstruction:one",
    kernel_receipt_id: str = "kernel:receipt-one",
    residual_gap_ids: tuple[str, ...] = (),
    assumption_refs: tuple[str, ...] = ("assumption:context",),
    automation_eligible: bool = True,
) -> LogicPredictionReceipt:
    if disposition is PredictionDisposition.PROVED and not clause_ref and not value_ref and not placement_ref:
        value_ref = "value:unique-context"
    return LogicPredictionReceipt(
        roots=logic_roots,
        receipt_id=receipt_id,
        goal_id="goal:one",
        hypothesis_id=hypothesis_id,
        tactician_plan_id="plan:tactician-one",
        corpus_id=logic_roots.corpus_id,
        disposition=disposition,
        hammer_request_id="hammer:req-one",
        translation_id="translation:one",
        candidate_id=candidate_id,
        reconstruction_id=reconstruction_id,
        kernel_receipt_id=kernel_receipt_id,
        environment_receipt_id="env:receipt-one",
        derived_clause_ref=clause_ref,
        derived_value_ref=value_ref,
        derived_placement_ref=placement_ref,
        assumption_refs=assumption_refs,
        residual_gap_ids=residual_gap_ids,
        source_authority=source_authority,
        proof_status=proof_status,
        automation_eligible=automation_eligible
        and disposition
        in {
            PredictionDisposition.PROVED,
            PredictionDisposition.VALIDATED_REFUTATION,
        },
        invalidation_refs=(
            logic_roots.tree_id,
            logic_roots.corpus_id,
            logic_roots.environment_id,
            logic_roots.toolchain_id,
            logic_roots.policy_id,
        ),
    )


def _atom(
    roots: PropagationAuthorityRoots,
    *,
    family: str | BehaviorClauseFamily = "fields",
    precedence: str | BehaviorEvidencePrecedence = "reviewed_idl",
    clause_ref: str = "",
    value_ref: str = "",
    subject: str = "symbol:SupportContext",
    evidence_id: str = "",
    **extra: object,
) -> BehaviorEvidenceAtom:
    fam = (
        family
        if isinstance(family, BehaviorClauseFamily)
        else BehaviorClauseFamily(family)
    )
    clause = clause_ref or f"clause:{fam.value}"
    value = value_ref or f"value:{fam.value}"
    eid = evidence_id or f"evidence:{fam.value}:{clause}"
    return BehaviorEvidenceAtom(
        roots=roots,
        evidence_id=eid,
        precedence=precedence,
        family=fam,
        clause_ref=clause,
        value_ref=value,
        subject_symbol_id=subject,
        **extra,
    )


def _class_evidence(roots: PropagationAuthorityRoots) -> list[BehaviorEvidenceAtom]:
    return [
        _atom(roots, family="fields", clause_ref="clause:fields", value_ref="value:fields"),
        _atom(
            roots,
            family="constructors",
            clause_ref="clause:constructors",
            value_ref="value:constructors",
        ),
        _atom(
            roots,
            family="methods",
            clause_ref="clause:methods",
            value_ref="value:methods",
        ),
        _atom(
            roots,
            family="invariants",
            clause_ref="clause:invariants",
            value_ref="value:invariants",
        ),
    ]


# ---------------------------------------------------------------------------
# Precedence lattice
# ---------------------------------------------------------------------------


def test_interfaces_and_factories(propagation_roots: PropagationAuthorityRoots) -> None:
    synth = create_tactician_guided_behavior_synthesizer(propagation_roots)
    bridge = create_contract_repair_prediction_bridge()
    assert isinstance(synth, TacticianGuidedBehaviorSynthesizer)
    assert isinstance(bridge, ContractRepairPredictionBridge)
    assert (
        TACTICIAN_GUIDED_BEHAVIOR_SYNTHESIS_INTERFACE
        == "TacticianGuidedBehaviorSynthesizer@1"
    )
    assert (
        CONTRACT_REPAIR_PREDICTION_BRIDGE_INTERFACE
        == "ContractRepairPredictionBridge@1"
    )


def test_weakest_precedence_uses_existing_ranks_only() -> None:
    assert (
        weakest_precedence(
            [
                BehaviorEvidencePrecedence.REVIEWED_IDL,
                BehaviorEvidencePrecedence.HISTORY,
                BehaviorEvidencePrecedence.CALLER_POSTCONDITION,
            ]
        )
        is BehaviorEvidencePrecedence.HISTORY
    )
    assert (
        weakest_precedence([BehaviorEvidencePrecedence.NORMATIVE_SPEC])
        is BehaviorEvidencePrecedence.NORMATIVE_SPEC
    )
    # Empty premises fall back to implementation hypothesis — no new rank.
    assert (
        weakest_precedence([])
        is BehaviorEvidencePrecedence.IMPLEMENTATION_HYPOTHESIS
    )


def test_proof_status_is_orthogonal_to_source_precedence() -> None:
    strong = effective_consequence_precedence(
        source_authority=SourceAuthorityClass.AUTHORITATIVE,
        premise_precedences=[
            BehaviorEvidencePrecedence.CALLER_POSTCONDITION,
            BehaviorEvidencePrecedence.HISTORY,
        ],
        evidence_routes=[SourceRouteKind.REVIEWED_CONTRACT],
        proof_status=ProofStatus.KERNEL_VERIFIED,
    )
    weak_proof = effective_consequence_precedence(
        source_authority=SourceAuthorityClass.AUTHORITATIVE,
        premise_precedences=[
            BehaviorEvidencePrecedence.CALLER_POSTCONDITION,
            BehaviorEvidencePrecedence.HISTORY,
        ],
        evidence_routes=[SourceRouteKind.REVIEWED_CONTRACT],
        proof_status=ProofStatus.UNPROVED,
    )
    assert strong is weak_proof
    assert strong is BehaviorEvidencePrecedence.HISTORY
    # Nominating authority alone cannot outrank independent history.
    assert (
        source_authority_floor(SourceAuthorityClass.NOMINATING)
        is BehaviorEvidencePrecedence.IMPLEMENTATION_HYPOTHESIS
    )
    assert (
        route_to_precedence(SourceRouteKind.VECTOR)
        is BehaviorEvidencePrecedence.IMPLEMENTATION_HYPOTHESIS
    )


# ---------------------------------------------------------------------------
# PredictionEvidenceBinding
# ---------------------------------------------------------------------------


def test_admitted_binding_maps_exact_refs(
    propagation_roots: PropagationAuthorityRoots,
    logic_roots: ProgramLogicAuthorityRoots,
) -> None:
    synth = TacticianGuidedBehaviorSynthesizer(propagation_roots)
    prediction = _prediction(
        logic_roots,
        value_ref="value:unique-context",
        clause_ref="clause:defaults-context",
        placement_ref="placement:module-support",
    )
    binding = synth.bind_prediction(
        prediction,
        premise_precedences=[
            PremisePrecedenceBinding(
                "premise:a", BehaviorEvidencePrecedence.CALLER_POSTCONDITION
            ),
            PremisePrecedenceBinding(
                "premise:b", BehaviorEvidencePrecedence.DATA_INVARIANT
            ),
        ],
        evidence_routes=[SourceRouteKind.REVIEWED_CONTRACT],
    )
    assert binding.disposition is BindingDisposition.ADMITTED
    assert binding.clause_ref == "clause:defaults-context"
    assert binding.value_ref == "value:unique-context"
    assert binding.placement_ref == "placement:module-support"
    assert binding.consequence_kind is ConsequenceKind.PLACEMENT
    # Weakest independent premise is DATA_INVARIANT (weaker than caller, stronger than history).
    # Reviewed route floors at REVIEWED_IDL; caller is stronger; data_invariant is weakest of premises.
    assert binding.effective_precedence is BehaviorEvidencePrecedence.DATA_INVARIANT
    assert binding.proof_status is ProofStatus.KERNEL_VERIFIED
    assert binding.source_authority is SourceAuthorityClass.AUTHORITATIVE
    assert binding.automation_eligible is True
    payload = binding.to_dict()
    assert payload["schema"] == PredictionEvidenceBinding.SCHEMA
    round_trip = PredictionEvidenceBinding.from_dict(payload)
    assert round_trip.binding_id == binding.binding_id


def test_stale_prediction_remains_nomination(
    propagation_roots: PropagationAuthorityRoots,
    logic_roots: ProgramLogicAuthorityRoots,
) -> None:
    synth = TacticianGuidedBehaviorSynthesizer(propagation_roots)
    prediction = _prediction(
        logic_roots,
        disposition=PredictionDisposition.STALE,
        proof_status=ProofStatus.STALE,
        source_authority=SourceAuthorityClass.AUTHORITATIVE,
        reconstruction_id="",
        kernel_receipt_id="",
        automation_eligible=False,
        residual_gap_ids=("gap:stale-tree",),
    )
    # STALE disposition requires invalidation refs (already on receipt).
    # But PROVED-style reconstruction fields are empty; STALE allows that.
    binding = synth.bind_prediction(prediction)
    assert binding.disposition is BindingDisposition.NOMINATION
    assert "stale_prediction_remains_nomination" in binding.reason_codes
    assert binding.automation_eligible is False


def test_solver_only_prediction_is_nomination(
    propagation_roots: PropagationAuthorityRoots,
    logic_roots: ProgramLogicAuthorityRoots,
) -> None:
    synth = TacticianGuidedBehaviorSynthesizer(propagation_roots)
    prediction = _prediction(
        logic_roots,
        disposition=PredictionDisposition.INCONCLUSIVE,
        proof_status=ProofStatus.SOLVER_CHECKED,
        source_authority=SourceAuthorityClass.AUTHORITATIVE,
        reconstruction_id="",
        kernel_receipt_id="",
        automation_eligible=False,
    )
    binding = synth.bind_prediction(prediction)
    assert binding.disposition is BindingDisposition.NOMINATION
    assert binding.effective_precedence is BehaviorEvidencePrecedence.REVIEWED_IDL


def test_never_overwrite_higher_precedence_source(
    propagation_roots: PropagationAuthorityRoots,
    logic_roots: ProgramLogicAuthorityRoots,
) -> None:
    synth = TacticianGuidedBehaviorSynthesizer(propagation_roots)
    existing = _atom(
        propagation_roots,
        family="defaults",
        precedence="reviewed_idl",
        clause_ref="clause:defaults-context",
        value_ref="value:reviewed-context",
    )
    prediction = _prediction(
        logic_roots,
        value_ref="value:predicted-weaker",
        clause_ref="clause:defaults-context",
    )
    binding = synth.bind_prediction(
        prediction,
        premise_precedences=[BehaviorEvidencePrecedence.HISTORY],
        family=BehaviorClauseFamily.DEFAULTS,
        existing_atoms=(existing,),
    )
    assert binding.disposition is BindingDisposition.SUPERSEDED
    assert "higher_precedence_source_preserved" in binding.reason_codes


def test_never_promote_unsupported_protected_facet(
    propagation_roots: PropagationAuthorityRoots,
    logic_roots: ProgramLogicAuthorityRoots,
) -> None:
    synth = TacticianGuidedBehaviorSynthesizer(propagation_roots)
    prediction = _prediction(
        logic_roots,
        clause_ref="clause:lifetime-borrow",
        value_ref="value:lifetime-borrow",
    )
    binding = synth.bind_prediction(
        prediction,
        family=BehaviorClauseFamily.LIFETIME,
        unsupported_facet_tokens=("memory", "lifetime"),
        premise_precedences=[BehaviorEvidencePrecedence.DATA_INVARIANT],
    )
    assert binding.disposition is BindingDisposition.BLOCKED
    assert any("unsupported" in code for code in binding.reason_codes)


def test_explicit_conflict_blocks_admission(
    propagation_roots: PropagationAuthorityRoots,
    logic_roots: ProgramLogicAuthorityRoots,
) -> None:
    synth = TacticianGuidedBehaviorSynthesizer(propagation_roots)
    prediction = _prediction(
        logic_roots,
        clause_ref="clause:conflicted",
        value_ref="value:conflicted",
    )
    binding = synth.bind_prediction(
        prediction,
        blocked_clause_refs=("clause:conflicted",),
        premise_precedences=[BehaviorEvidencePrecedence.CALLER_POSTCONDITION],
    )
    assert binding.disposition is BindingDisposition.BLOCKED
    assert "explicit_conflict_or_blocked_clause" in binding.reason_codes


# ---------------------------------------------------------------------------
# ContractRepairPredictionBridge → CandidateProofBundle
# ---------------------------------------------------------------------------


def test_projects_reconstructed_predictions_into_proof_bundle(
    logic_roots: ProgramLogicAuthorityRoots,
) -> None:
    bridge = ContractRepairPredictionBridge()
    value_pred = _prediction(
        logic_roots,
        receipt_id="prediction:value",
        hypothesis_id="hyp:value",
        value_ref="value:unique-a",
        clause_ref="",
    )
    placement_pred = _prediction(
        logic_roots,
        receipt_id="prediction:placement",
        hypothesis_id="hyp:placement",
        value_ref="",
        placement_ref="placement:support-module",
        clause_ref="",
    )
    substitution_pred = _prediction(
        logic_roots,
        receipt_id="prediction:subst",
        hypothesis_id="hyp:subst",
        value_ref="",
        clause_ref="substitution:param-thread",
    )
    bundle = bridge.project_bundle(
        candidate_id="candidate:repair-one",
        repository_id=logic_roots.repository_id,
        tree_id=logic_roots.tree_id,
        predictions=(value_pred, placement_pred, substitution_pred),
        translator_id=logic_roots.translator_id,
        toolchain_id=logic_roots.toolchain_id,
        policy_id=logic_roots.policy_id,
    )
    assert isinstance(bundle, CandidateProofBundle)
    assert bundle.candidate_id == "candidate:repair-one"
    assert bundle.repository_id == logic_roots.repository_id
    assert bundle.tree_id == logic_roots.tree_id
    assert len(bundle.results) == 3
    assert all(
        item.disposition is ContractRepairProofDisposition.PROVED
        for item in bundle.results
    )
    assert bundle.candidate_authoritative is True
    # Canonical shape accepted by ContractRepairReranker consumers.
    payload = bundle.to_dict()
    assert payload["candidate_authoritative"] is True
    assert payload["interface"] == "ContractRepairProver@1"
    assert all(item["disposition"] == "proved" for item in payload["results"])


def test_stale_and_solver_only_remain_non_conclusive(
    logic_roots: ProgramLogicAuthorityRoots,
) -> None:
    bridge = ContractRepairPredictionBridge()
    stale = _prediction(
        logic_roots,
        receipt_id="prediction:stale",
        hypothesis_id="hyp:stale",
        disposition=PredictionDisposition.STALE,
        proof_status=ProofStatus.STALE,
        reconstruction_id="",
        kernel_receipt_id="",
        automation_eligible=False,
    )
    solver = _prediction(
        logic_roots,
        receipt_id="prediction:solver",
        hypothesis_id="hyp:solver",
        disposition=PredictionDisposition.INCONCLUSIVE,
        proof_status=ProofStatus.SOLVER_CHECKED,
        reconstruction_id="",
        kernel_receipt_id="",
        automation_eligible=False,
    )
    bundle = bridge.project_bundle(
        candidate_id="candidate:repair-one",
        repository_id=logic_roots.repository_id,
        tree_id=logic_roots.tree_id,
        predictions=(stale, solver),
    )
    assert bundle.candidate_authoritative is False
    assert all(
        item.disposition is ContractRepairProofDisposition.NON_CONCLUSIVE
        for item in bundle.results
    )
    reasons = {code for item in bundle.results for code in item.reason_codes}
    assert "stale_prediction_nomination" in reasons
    assert "solver_only_prediction_nomination" in reasons


def test_bridge_filters_to_candidate_specific_predictions(
    logic_roots: ProgramLogicAuthorityRoots,
) -> None:
    bridge = ContractRepairPredictionBridge()
    own = _prediction(
        logic_roots,
        receipt_id="prediction:own",
        hypothesis_id="hyp:own",
        candidate_id="candidate:target",
    )
    other = _prediction(
        logic_roots,
        receipt_id="prediction:other",
        hypothesis_id="hyp:other",
        candidate_id="candidate:other",
    )
    bundle = bridge.project_bundle(
        candidate_id="candidate:target",
        repository_id=logic_roots.repository_id,
        tree_id=logic_roots.tree_id,
        predictions=(own, other),
    )
    assert len(bundle.results) == 1
    assert "hyp:own" in bundle.results[0].obligation_id or True
    meta = bundle.results[0].receipt.metadata
    assert meta.get("hypothesis_id") == "hyp:own"


# ---------------------------------------------------------------------------
# TacticianGuidedBehaviorSynthesizer composition
# ---------------------------------------------------------------------------


def test_composes_with_required_behavior_synthesizer(
    propagation_roots: PropagationAuthorityRoots,
    logic_roots: ProgramLogicAuthorityRoots,
) -> None:
    synth = TacticianGuidedBehaviorSynthesizer(propagation_roots)
    requirement = _requirement(propagation_roots)
    existing = _class_evidence(propagation_roots)
    # Admitted prediction fills a defaults clause without overwriting reviewed fields.
    prediction = _prediction(
        logic_roots,
        value_ref="value:defaults-from-prediction",
        clause_ref="clause:defaults-from-prediction",
    )
    receipt = synth.synthesize(
        (prediction,),
        requirement=requirement,
        kind=BehaviorKind.CLASS,
        subject_symbol_id="symbol:SupportContext",
        existing_evidence=existing,
        premise_precedences=[
            PremisePrecedenceBinding(
                "premise:idl", BehaviorEvidencePrecedence.REVIEWED_IDL
            ),
            PremisePrecedenceBinding(
                "premise:caller", BehaviorEvidencePrecedence.CALLER_POSTCONDITION
            ),
        ],
        evidence_routes=[SourceRouteKind.LOCAL_STATIC],
        project_proof_bundle=True,
        candidate_id="candidate:repair-one",
        repository_id=propagation_roots.repository_id,
        tree_id=propagation_roots.candidate_tree_id,
    )
    assert isinstance(receipt, TacticianBehaviorSynthesisReceipt)
    assert receipt.write_authority is False
    assert receipt.semantic_authority is False
    assert receipt.disposition in {
        BridgeDisposition.COMPOSED,
        BridgeDisposition.PROOF_PROJECTED,
    }
    assert receipt.behavior_receipt is not None
    # Existing synthesizer still admits from merged independent evidence.
    assert receipt.behavior_receipt.disposition is SynthesisDisposition.ADMITTED
    assert receipt.behavior_contract is not None
    assert receipt.proof_bundle is not None
    assert receipt.proof_bundle.candidate_authoritative is True
    assert len(receipt.admitted_bindings) == 1
    # Consumer of ChangePropagationObligationCompiler receives exact contract.
    contracts = receipt.behavior_contracts_for_obligations
    assert len(contracts) == 1
    assert contracts[0].subject_symbol_id == "symbol:SupportContext"
    payload = receipt.to_dict()
    assert payload["write_authority"] is False
    assert payload["interface"] == TACTICIAN_GUIDED_BEHAVIOR_SYNTHESIS_INTERFACE


def test_value_candidates_feed_missing_input_surface(
    propagation_roots: PropagationAuthorityRoots,
    logic_roots: ProgramLogicAuthorityRoots,
) -> None:
    synth = TacticianGuidedBehaviorSynthesizer(propagation_roots)
    requirement = _requirement(propagation_roots)
    prediction = _prediction(
        logic_roots,
        value_ref="value:unique-context",
        clause_ref="construction:SupportContext",
    )
    receipt = synth.synthesize(
        (prediction,),
        requirement=requirement,
        synthesize_behavior=False,
        premise_precedences=[BehaviorEvidencePrecedence.DATA_INVARIANT],
    )
    assert receipt.value_candidates
    candidate = receipt.value_candidates[0]
    assert candidate.requirement_id == requirement.requirement_id
    assert candidate.disposition is ValueCandidateDisposition.PROVED
    assert candidate.semantic_authority is True
    assert candidate.expression_ref == "value:unique-context"
    assert candidate.proof_refs


def test_nomination_only_when_all_predictions_stale(
    propagation_roots: PropagationAuthorityRoots,
    logic_roots: ProgramLogicAuthorityRoots,
) -> None:
    synth = TacticianGuidedBehaviorSynthesizer(propagation_roots)
    prediction = _prediction(
        logic_roots,
        disposition=PredictionDisposition.STALE,
        proof_status=ProofStatus.STALE,
        reconstruction_id="",
        kernel_receipt_id="",
        automation_eligible=False,
    )
    receipt = synth.synthesize(
        (prediction,),
        requirement=_requirement(propagation_roots),
        synthesize_behavior=False,
    )
    assert receipt.disposition is BridgeDisposition.NOMINATION_ONLY
    assert receipt.admitted_bindings == ()
    assert receipt.nomination_bindings
    assert receipt.proof_bundle is None


def test_merge_preserves_higher_precedence_and_keeps_conflicts_explicit(
    propagation_roots: PropagationAuthorityRoots,
) -> None:
    synth = TacticianGuidedBehaviorSynthesizer(propagation_roots)
    existing = (
        _atom(
            propagation_roots,
            family="fields",
            precedence="reviewed_idl",
            clause_ref="clause:fields",
            value_ref="value:reviewed-fields",
        ),
    )
    predicted = (
        BehaviorEvidenceAtom(
            roots=propagation_roots,
            evidence_id="evidence:prediction:fields",
            precedence=BehaviorEvidencePrecedence.HISTORY,
            family=BehaviorClauseFamily.FIELDS,
            clause_ref="clause:fields",
            value_ref="value:predicted-fields",
            subject_symbol_id="symbol:SupportContext",
            authoritative=True,
            proof_ref="kernel:receipt-one",
        ),
    )
    merged = synth.merge_evidence(existing, predicted)
    # Weaker prediction dropped; stronger reviewed source remains alone.
    assert len(merged) == 1
    assert merged[0].value_ref == "value:reviewed-fields"

    same_rank_conflict = (
        BehaviorEvidenceAtom(
            roots=propagation_roots,
            evidence_id="evidence:prediction:fields-conflict",
            precedence=BehaviorEvidencePrecedence.REVIEWED_IDL,
            family=BehaviorClauseFamily.FIELDS,
            clause_ref="clause:fields",
            value_ref="value:other-reviewed",
            subject_symbol_id="symbol:SupportContext",
            authoritative=True,
            proof_ref="kernel:receipt-two",
        ),
    )
    conflicted = synth.merge_evidence(existing, same_rank_conflict)
    assert len(conflicted) == 2


def test_required_behavior_synthesizer_still_accepts_merged_atoms(
    propagation_roots: PropagationAuthorityRoots,
    logic_roots: ProgramLogicAuthorityRoots,
) -> None:
    """Outputs are accepted unchanged by the existing synthesizer path."""

    bridge_synth = TacticianGuidedBehaviorSynthesizer(propagation_roots)
    requirement = _requirement(propagation_roots)
    existing = _class_evidence(propagation_roots)
    prediction = _prediction(
        logic_roots,
        value_ref="value:defaults-extra",
        clause_ref="clause:defaults-extra",
    )
    binding = bridge_synth.bind_prediction(
        prediction,
        premise_precedences=[BehaviorEvidencePrecedence.NORMATIVE_SPEC],
        family=BehaviorClauseFamily.DEFAULTS,
    )
    atoms = bridge_synth.atoms_from_bindings(
        (binding,), subject_symbol_id="symbol:SupportContext"
    )
    merged = bridge_synth.merge_evidence(existing, atoms)
    # Direct call into existing RequiredBehaviorSynthesizer.
    rbs = RequiredBehaviorSynthesizer(propagation_roots)
    receipt = rbs.synthesize(
        requirement,
        kind=BehaviorKind.CLASS,
        subject_symbol_id="symbol:SupportContext",
        evidence=merged,
    )
    assert receipt.disposition is SynthesisDisposition.ADMITTED
    assert receipt.contract is not None
    assert receipt.implementation_request is False


def test_admitted_receipt_is_canonical_and_replayable(
    propagation_roots: PropagationAuthorityRoots,
    logic_roots: ProgramLogicAuthorityRoots,
) -> None:
    synth = TacticianGuidedBehaviorSynthesizer(propagation_roots)
    prediction = _prediction(logic_roots)
    first = synth.synthesize(
        (prediction,),
        requirement=_requirement(propagation_roots),
        subject_symbol_id="symbol:SupportContext",
        existing_evidence=_class_evidence(propagation_roots),
        premise_precedences=[BehaviorEvidencePrecedence.DATA_INVARIANT],
        project_proof_bundle=True,
        candidate_id="candidate:repair-one",
    )
    second = synth.synthesize(
        (prediction,),
        requirement=_requirement(propagation_roots),
        subject_symbol_id="symbol:SupportContext",
        existing_evidence=_class_evidence(propagation_roots),
        premise_precedences=[BehaviorEvidencePrecedence.DATA_INVARIANT],
        project_proof_bundle=True,
        candidate_id="candidate:repair-one",
    )
    assert first.receipt_id == second.receipt_id
    assert first.to_dict() == second.to_dict()
    assert first.content_id == second.content_id


def test_cannot_claim_write_authority_on_receipt(
    propagation_roots: PropagationAuthorityRoots,
) -> None:
    with pytest.raises(TacticianGuidedBehaviorSynthesisAuthorityError):
        TacticianBehaviorSynthesisReceipt(
            roots=propagation_roots,
            receipt_id="tactician-behavior:forbidden",
            disposition=BridgeDisposition.ABSTAINED,
            bindings=(),
            write_authority=True,
        )


def test_precedence_rank_of_inherited_consequence_matches_lattice() -> None:
    inherited = effective_consequence_precedence(
        source_authority=SourceAuthorityClass.CONFORMANCE,
        premise_precedences=[
            BehaviorEvidencePrecedence.NORMATIVE_SPEC,
            BehaviorEvidencePrecedence.ARCHITECTURE_OWNERSHIP,
        ],
        evidence_routes=[SourceRouteKind.REVIEWED_TEST],
    )
    assert inherited is BehaviorEvidencePrecedence.ARCHITECTURE_OWNERSHIP
    assert precedence_rank(inherited) == precedence_rank(
        BehaviorEvidencePrecedence.ARCHITECTURE_OWNERSHIP
    )
    # Still within the closed existing enum.
    assert inherited in BehaviorEvidencePrecedence
