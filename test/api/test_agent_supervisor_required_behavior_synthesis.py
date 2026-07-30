"""Adversarial conformance tests for required-behavior synthesis (RPR-034)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.change_propagation_contracts import (
    BehaviorEvidencePrecedence,
    BehaviorKind,
    ContractClauseDelta,
    DeltaDisposition,
    DeltaKind,
    MissingInputRequirement,
    ProgramContractDelta,
    PropagationAuthorityRoots,
    RequiredBehaviorContract,
)
from ipfs_accelerate_py.agent_supervisor.analysis.required_behavior_synthesis import (
    PRECEDENCE_GROUPS,
    PRECEDENCE_RANK,
    PRODUCER_ID,
    BehaviorClauseFamily,
    BehaviorEvidenceAtom,
    BehaviorGap,
    BehaviorGapKind,
    RequiredBehaviorSynthesisAuthorityError,
    RequiredBehaviorSynthesisError,
    RequiredBehaviorSynthesisReceipt,
    RequiredBehaviorSynthesizer,
    SynthesisDisposition,
    all_clause_families,
    all_precedence_levels,
    coerce_precedence,
    is_authoritative,
    precedence_rank,
    synthesize_required_behavior,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def roots() -> PropagationAuthorityRoots:
    return PropagationAuthorityRoots(
        repository_id="repository:rpr-034",
        base_forest_id="forest:base",
        base_tree_id="tree:base",
        base_overlay_id="overlay:base",
        candidate_forest_id="forest:candidate",
        candidate_tree_id="tree:candidate",
        candidate_overlay_id="overlay:candidate",
        graph_id="graph:rpr-034",
        index_id="index:rpr-034",
        model_id="model:rpr-034",
        config_id="config:rpr-034",
        translator_id="translator:rpr-034",
        toolchain_id="toolchain:rpr-034",
        policy_id="policy:rpr-034",
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


def _atom(
    roots: PropagationAuthorityRoots,
    *,
    family: str | BehaviorClauseFamily,
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


def _class_evidence(roots: PropagationAuthorityRoots) -> list[dict[str, object]]:
    """Compact recipe covering the structural + lifecycle families for a class."""
    recipes = [
        ("fields", "reviewed_idl", "field:trace_id", "shape:trace_id:str"),
        ("variants", "reviewed_idl", "variant:live", "shape:variant:live"),
        ("generics", "normative_spec", "generic:T", "shape:generic:T"),
        ("invariants", "data_invariant", "inv:non_empty_trace", "inv:non_empty"),
        ("defaults", "reviewed_idl", "default:trace_id=uuid4", "default:uuid4"),
        ("constructors", "reviewed_idl", "ctor:SupportContext", "ctor:total"),
        ("factories", "normative_spec", "factory:from_request", "factory:total"),
        ("totality", "normative_spec", "total:from_request", "total:true"),
        ("methods", "caller_postcondition", "method:with_span", "method:pure"),
        ("state_machine", "callee_precondition", "sm:idle|active|closed", "sm:3"),
        ("transitions", "callee_precondition", "tx:idle->active", "tx:start"),
        ("idempotence", "normative_spec", "idemp:with_span", "idemp:true"),
        ("ownership", "architecture_ownership", "own:caller", "own:unique"),
        ("lifetime", "architecture_ownership", "life:request", "life:scoped"),
        ("mutation", "data_invariant", "mut:interior_mutable", "mut:false"),
        ("concurrency", "normative_spec", "conc:send_sync", "conc:safe"),
        ("cache", "history", "cache:none", "cache:none"),
        ("disposal", "architecture_ownership", "dispose:close", "dispose:idempotent"),
        ("serialization", "reviewed_idl", "ser:json", "ser:json@1"),
        ("persistence", "migration_manifest", "persist:none", "persist:ephemeral"),
        ("versioning", "migration_manifest", "ver:1", "ver:1"),
        ("migrations", "migration_manifest", "mig:v0_to_v1", "mig:additive"),
        ("equality", "normative_spec", "eq:by_trace_id", "eq:value"),
        ("hash", "normative_spec", "hash:trace_id", "hash:stable"),
        ("errors", "caller_postcondition", "err:ContextError", "err:closed"),
        ("cancellation", "callee_precondition", "cancel:cooperative", "cancel:ok"),
        ("effects", "caller_postcondition", "effect:none", "effect:pure"),
        # value_ref must agree with MissingInputRequirement.capability_refs
        # when both contribute CAPABILITIES at the same precedence rank.
        ("capabilities", "callee_precondition", "cap:context.read", "cap:context.read"),
        ("authorization", "callee_precondition", "auth:session", "auth:required"),
        ("trust", "architecture_ownership", "trust:boundary", "trust:internal"),
        ("privacy", "normative_spec", "privacy:no_pii", "privacy:redact"),
        ("resources", "callee_precondition", "res:memory_bound", "res:1mb"),
        ("degradation", "history", "deg:fail_closed", "deg:fail_closed"),
        ("compatibility", "migration_manifest", "compat:adapter_v0", "compat:ok"),
        ("tests", "normative_spec", "test:conformance_suite", "test:required"),
        ("telemetry", "architecture_ownership", "tel:trace_span", "tel:open_telemetry"),
    ]
    return [
        {
            "family": family,
            "precedence": precedence,
            "clause_ref": clause,
            "value_ref": value,
            "subject_symbol_id": "symbol:SupportContext",
            "evidence_id": f"evidence:{family}",
            "proof_ref": f"proof:{family}" if precedence != "history" else "",
        }
        for family, precedence, clause, value in recipes
    ]


# ---------------------------------------------------------------------------
# Precedence and vocabulary
# ---------------------------------------------------------------------------


def test_precedence_rank_matches_plan_order() -> None:
    ordered = all_precedence_levels()
    assert ordered[0] is BehaviorEvidencePrecedence.REVIEWED_IDL
    assert ordered[-1] is BehaviorEvidencePrecedence.IMPLEMENTATION_HYPOTHESIS
    assert precedence_rank(BehaviorEvidencePrecedence.REVIEWED_IDL) < precedence_rank(
        BehaviorEvidencePrecedence.NORMATIVE_SPEC
    )
    assert precedence_rank(BehaviorEvidencePrecedence.CALLER_POSTCONDITION) < precedence_rank(
        BehaviorEvidencePrecedence.DATA_INVARIANT
    )
    assert not is_authoritative(BehaviorEvidencePrecedence.IMPLEMENTATION_HYPOTHESIS)
    assert is_authoritative(BehaviorEvidencePrecedence.REVIEWED_IDL)
    # Acceptance groups cover every enum member exactly once.
    grouped = {
        member
        for _, members in PRECEDENCE_GROUPS
        for member in members
    }
    assert grouped == set(PRECEDENCE_RANK)


def test_precedence_aliases_map_acceptance_sources() -> None:
    assert coerce_precedence("schema") is BehaviorEvidencePrecedence.REVIEWED_IDL
    assert coerce_precedence("public_stub") is BehaviorEvidencePrecedence.REVIEWED_IDL
    assert coerce_precedence("conformance_test") is BehaviorEvidencePrecedence.NORMATIVE_SPEC
    assert coerce_precedence("observation") is BehaviorEvidencePrecedence.IMPLEMENTATION_HYPOTHESIS
    assert coerce_precedence("llm") is BehaviorEvidencePrecedence.IMPLEMENTATION_HYPOTHESIS


def test_all_clause_families_cover_acceptance_dimensions() -> None:
    families = {item.value for item in all_clause_families()}
    required = {
        "fields",
        "variants",
        "generics",
        "invariants",
        "defaults",
        "constructors",
        "factories",
        "totality",
        "methods",
        "state_machine",
        "transitions",
        "idempotence",
        "ownership",
        "lifetime",
        "mutation",
        "concurrency",
        "cache",
        "disposal",
        "serialization",
        "persistence",
        "versioning",
        "migrations",
        "equality",
        "hash",
        "errors",
        "cancellation",
        "effects",
        "capabilities",
        "authorization",
        "trust",
        "privacy",
        "resources",
        "degradation",
        "compatibility",
        "tests",
        "telemetry",
    }
    assert required <= families


# ---------------------------------------------------------------------------
# Happy path admission
# ---------------------------------------------------------------------------


def test_reviewed_idl_admits_required_behavior_contract(
    roots: PropagationAuthorityRoots,
) -> None:
    requirement = _requirement(roots)
    receipt = RequiredBehaviorSynthesizer(roots).synthesize(
        requirement,
        kind=BehaviorKind.CLASS,
        subject_symbol_id="symbol:SupportContext",
        evidence=_class_evidence(roots),
    )

    assert receipt.disposition is SynthesisDisposition.ADMITTED
    assert receipt.admitted
    assert not receipt.has_gap
    assert receipt.implementation_request is False
    assert receipt.producer_id == PRODUCER_ID
    assert receipt.gap is None
    assert isinstance(receipt.contract, RequiredBehaviorContract)
    assert receipt.contract.kind is BehaviorKind.CLASS
    assert (
        receipt.contract.evidence_precedence
        is BehaviorEvidencePrecedence.REVIEWED_IDL
    )
    assert receipt.contract.implementation_hypothesis is False
    assert receipt.contract.field_refs
    assert receipt.contract.constructor_refs
    assert receipt.contract.method_refs
    assert receipt.contract.invariant_refs
    assert receipt.contract.state_transition_refs
    assert receipt.contract.effect_refs
    assert receipt.contract.capability_refs
    assert receipt.contract.authorization_refs
    assert receipt.contract.resource_refs
    assert receipt.contract.proof_refs
    # Canonical RPR-022 identity round-trip.
    assert (
        RequiredBehaviorContract.from_dict(receipt.contract.to_record())
        == receipt.contract
    )
    rebuilt = RequiredBehaviorSynthesisReceipt.from_dict(receipt.to_record())
    assert rebuilt.content_id == receipt.content_id
    assert rebuilt.contract is not None
    assert rebuilt.contract.content_id == receipt.contract.content_id


def test_synthesis_is_deterministic(roots: PropagationAuthorityRoots) -> None:
    requirement = _requirement(roots)
    evidence = _class_evidence(roots)
    left = synthesize_required_behavior(
        roots,
        requirement,
        kind="class",
        subject_symbol_id="symbol:SupportContext",
        evidence=evidence,
    )
    right = synthesize_required_behavior(
        roots,
        requirement,
        kind="class",
        subject_symbol_id="symbol:SupportContext",
        evidence=list(reversed(evidence)),
    )
    assert left.content_id == right.content_id
    assert left.receipt_id == right.receipt_id
    assert left.contract is not None and right.contract is not None
    assert left.contract.content_id == right.contract.content_id


def test_higher_precedence_overrides_lower_without_conflict(
    roots: PropagationAuthorityRoots,
) -> None:
    requirement = _requirement(roots)
    evidence = [
        _atom(
            roots,
            family="fields",
            precedence="history",
            clause_ref="field:trace_id",
            value_ref="shape:old",
            evidence_id="evidence:fields:history",
        ),
        _atom(
            roots,
            family="fields",
            precedence="reviewed_idl",
            clause_ref="field:trace_id",
            value_ref="shape:current",
            evidence_id="evidence:fields:idl",
        ),
        _atom(
            roots,
            family="constructors",
            precedence="reviewed_idl",
            clause_ref="ctor:SupportContext",
            value_ref="ctor:total",
        ),
    ]
    receipt = RequiredBehaviorSynthesizer(roots).synthesize(
        requirement,
        kind=BehaviorKind.CLASS,
        subject_symbol_id="symbol:SupportContext",
        evidence=evidence,
    )
    assert receipt.admitted
    assert receipt.contract is not None
    binding = next(b for b in receipt.clause_bindings if b.family is BehaviorClauseFamily.FIELDS)
    assert binding.value_ref == "shape:current"
    assert binding.precedence is BehaviorEvidencePrecedence.REVIEWED_IDL


# ---------------------------------------------------------------------------
# Gaps: conflict, insufficient, implementation-only
# ---------------------------------------------------------------------------


def test_same_rank_conflict_yields_behavior_gap_without_implementation_request(
    roots: PropagationAuthorityRoots,
) -> None:
    requirement = _requirement(roots)
    evidence = [
        _atom(
            roots,
            family="fields",
            precedence="reviewed_idl",
            clause_ref="field:trace_id",
            value_ref="shape:a",
            evidence_id="evidence:fields:a",
        ),
        _atom(
            roots,
            family="fields",
            precedence="schema",  # alias of reviewed_idl
            clause_ref="field:trace_id",
            value_ref="shape:b",
            evidence_id="evidence:fields:b",
        ),
        _atom(
            roots,
            family="constructors",
            precedence="reviewed_idl",
            clause_ref="ctor:SupportContext",
            value_ref="ctor:total",
        ),
    ]
    receipt = RequiredBehaviorSynthesizer(roots).synthesize(
        requirement,
        kind=BehaviorKind.CLASS,
        subject_symbol_id="symbol:SupportContext",
        evidence=evidence,
    )
    assert receipt.disposition is SynthesisDisposition.BEHAVIOR_GAP
    assert receipt.has_gap
    assert receipt.contract is None
    assert receipt.implementation_request is False
    assert isinstance(receipt.gap, BehaviorGap)
    assert receipt.gap.kind is BehaviorGapKind.CONFLICTING_EVIDENCE
    assert receipt.gap.implementation_request is False
    assert "evidence:fields:a" in receipt.gap.conflicting_evidence_ids
    assert "evidence:fields:b" in receipt.gap.conflicting_evidence_ids
    # Gap itself is content-addressed and fail-closed on forged identity.
    gap_payload = receipt.gap.to_record()
    assert BehaviorGap.from_dict(gap_payload).content_id == receipt.gap.content_id
    gap_payload["content_id"] = "behavior-gap:" + ("0" * 64)
    with pytest.raises(RequiredBehaviorSynthesisAuthorityError):
        BehaviorGap.from_dict(gap_payload)


def test_insufficient_evidence_is_a_typed_gap(
    roots: PropagationAuthorityRoots,
) -> None:
    requirement = _requirement(roots)
    # Only non-structural capability evidence from the requirement itself.
    receipt = RequiredBehaviorSynthesizer(roots).synthesize(
        requirement,
        kind=BehaviorKind.CLASS,
        subject_symbol_id="symbol:SupportContext",
        evidence=(),
        include_requirement_atoms=True,
    )
    assert receipt.has_gap
    assert receipt.gap is not None
    assert receipt.gap.kind in {
        BehaviorGapKind.INSUFFICIENT_EVIDENCE,
        BehaviorGapKind.KIND_REQUIREMENT_UNMET,
    }
    assert receipt.gap.missing_families
    assert receipt.implementation_request is False
    assert receipt.contract is None


def test_implementation_hypothesis_cannot_admit_behavior(
    roots: PropagationAuthorityRoots,
) -> None:
    requirement = _requirement(roots)
    evidence = [
        _atom(
            roots,
            family="fields",
            precedence="implementation_hypothesis",
            clause_ref="field:guessed",
            value_ref="shape:guessed",
            authoritative=False,
        ),
        _atom(
            roots,
            family="methods",
            precedence="observation",
            clause_ref="method:guessed",
            value_ref="method:guessed",
            authoritative=False,
        ),
        _atom(
            roots,
            family="constructors",
            precedence="llm",
            clause_ref="ctor:guessed",
            value_ref="ctor:guessed",
            authoritative=False,
        ),
    ]
    receipt = RequiredBehaviorSynthesizer(roots).synthesize(
        requirement,
        kind=BehaviorKind.CLASS,
        subject_symbol_id="symbol:SupportContext",
        evidence=evidence,
        include_requirement_atoms=False,
    )
    assert receipt.has_gap
    assert receipt.gap is not None
    assert receipt.gap.kind is BehaviorGapKind.IMPLEMENTATION_ONLY
    assert receipt.implementation_request is False
    assert receipt.contract is None


def test_implementation_hypothesis_cannot_carry_proof_ref(
    roots: PropagationAuthorityRoots,
) -> None:
    with pytest.raises(RequiredBehaviorSynthesisAuthorityError, match="proof"):
        _atom(
            roots,
            family="fields",
            precedence="implementation_hypothesis",
            clause_ref="field:x",
            value_ref="shape:x",
            authoritative=False,
            proof_ref="proof:illegal",
        )


def test_gap_cannot_set_implementation_request(
    roots: PropagationAuthorityRoots,
) -> None:
    with pytest.raises(RequiredBehaviorSynthesisAuthorityError, match="implementation"):
        BehaviorGap(
            roots=roots,
            gap_id="behavior-gap:test",
            kind=BehaviorGapKind.INSUFFICIENT_EVIDENCE,
            subject_symbol_id="symbol:SupportContext",
            requirement_id="missing:support-context",
            reason="test",
            implementation_request=True,
        )


# ---------------------------------------------------------------------------
# Assumptions and unsupported clauses
# ---------------------------------------------------------------------------


def test_assumptions_and_unsupported_clauses_are_stated(
    roots: PropagationAuthorityRoots,
) -> None:
    requirement = _requirement(roots)
    evidence = [
        _atom(
            roots,
            family="fields",
            precedence="reviewed_idl",
            clause_ref="field:trace_id",
            value_ref="shape:trace_id",
        ),
        _atom(
            roots,
            family="constructors",
            precedence="reviewed_idl",
            clause_ref="ctor:SupportContext",
            value_ref="ctor:total",
        ),
        _atom(
            roots,
            family="concurrency",
            precedence="history",
            clause_ref="conc:assumed_single_threaded",
            value_ref="conc:assumed",
            assumption=True,
            authoritative=False,
        ),
        _atom(
            roots,
            family="persistence",
            precedence="normative_spec",
            clause_ref="persist:native_blob",
            value_ref="persist:unsupported",
            unsupported=True,
            authoritative=False,
        ),
    ]
    receipt = RequiredBehaviorSynthesizer(roots).synthesize(
        requirement,
        kind=BehaviorKind.CLASS,
        subject_symbol_id="symbol:SupportContext",
        evidence=evidence,
        include_requirement_atoms=False,
    )
    assert receipt.admitted
    assert "conc:assumed_single_threaded" in receipt.assumptions
    assert "persist:native_blob" in receipt.unsupported_clauses
    # Unsupported persistence must not appear in field/serialization buckets.
    assert receipt.contract is not None
    assert "persist:native_blob" not in receipt.contract.field_refs


# ---------------------------------------------------------------------------
# Kind coverage and interfaces
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "families"),
    [
        (BehaviorKind.METHOD, ("methods",)),
        (BehaviorKind.FACTORY, ("factories", "totality")),
        (BehaviorKind.SCHEMA, ("fields", "invariants", "defaults")),
        (BehaviorKind.STATE_TRANSITION, ("state_machine", "transitions")),
        (BehaviorKind.ADAPTER, ("methods", "compatibility")),
        (BehaviorKind.SERIALIZER, ("serialization", "fields")),
        (BehaviorKind.PROVIDER, ("methods", "constructors")),
        (BehaviorKind.DATA_STRUCTURE, ("fields", "variants", "invariants")),
    ],
)
def test_kind_specific_structural_requirements(
    roots: PropagationAuthorityRoots,
    kind: BehaviorKind,
    families: tuple[str, ...],
) -> None:
    requirement = _requirement(roots, type_ref=f"type:{kind.value}")
    evidence = [
        _atom(
            roots,
            family=family,
            precedence="reviewed_idl",
            clause_ref=f"clause:{family}",
            value_ref=f"value:{family}",
            subject=f"symbol:{kind.value}",
            evidence_id=f"evidence:{kind.value}:{family}",
        )
        for family in families
    ]
    receipt = RequiredBehaviorSynthesizer(roots).synthesize(
        requirement,
        kind=kind,
        subject_symbol_id=f"symbol:{kind.value}",
        evidence=evidence,
        include_requirement_atoms=False,
    )
    assert receipt.admitted, receipt.gap.reason if receipt.gap else ""
    assert receipt.contract is not None
    assert receipt.contract.kind is kind


def test_contract_delta_contributes_caller_postcondition_evidence(
    roots: PropagationAuthorityRoots,
) -> None:
    requirement = _requirement(roots)
    clause = ContractClauseDelta(
        clause_id="clause:param-add",
        kind=DeltaKind.PARAMETER_ADD,
        disposition=DeltaDisposition.BREAKING,
        subject_symbol_id="symbol:SupportContext",
        consumer_domain="domain:python-callers",
        before_contract_ref="contract:before",
        after_contract_ref="contract:after",
        reason="support context required",
    )
    delta = ProgramContractDelta(
        roots=roots,
        change_set_id="changeset:one",
        subject_symbol_id="symbol:SupportContext",
        before_contract_ref="contract:before",
        after_contract_ref="contract:after",
        clauses=(clause,),
        proof_refs=("proof:delta",),
    )
    evidence = [
        _atom(
            roots,
            family="constructors",
            precedence="reviewed_idl",
            clause_ref="ctor:SupportContext",
            value_ref="ctor:total",
        ),
        _atom(
            roots,
            family="fields",
            precedence="reviewed_idl",
            clause_ref="field:trace_id",
            value_ref="shape:trace_id",
        ),
    ]
    receipt = RequiredBehaviorSynthesizer(roots).synthesize(
        requirement,
        kind=BehaviorKind.CLASS,
        subject_symbol_id="symbol:SupportContext",
        evidence=evidence,
        contract_delta=delta,
        include_requirement_atoms=False,
    )
    assert receipt.admitted
    assert receipt.contract_delta_id == delta.content_id
    # Delta evidence is ingested even when a higher-precedence field atom wins
    # the FIELDS family; the proof ref and evidence id remain on the receipt.
    assert "evidence:delta:clause:param-add" in receipt.evidence_ids
    assert "proof:delta" in receipt.proof_refs


def test_stale_value_provenance_roots_fail_closed(
    roots: PropagationAuthorityRoots,
) -> None:
    requirement = _requirement(roots)

    @dataclass(frozen=True)
    class _ForeignRoots:
        repository_id: str = "repository:other"
        tree_id: str = "tree:foreign"

    @dataclass(frozen=True)
    class _ForeignGraph:
        roots: _ForeignRoots = _ForeignRoots()
        graph_id: str = "value-provenance:foreign"

    with pytest.raises(RequiredBehaviorSynthesisAuthorityError, match="repository_id|tree_id"):
        RequiredBehaviorSynthesizer(roots).synthesize(
            requirement,
            kind=BehaviorKind.CLASS,
            subject_symbol_id="symbol:SupportContext",
            evidence=[
                _atom(
                    roots,
                    family="fields",
                    precedence="reviewed_idl",
                    clause_ref="field:x",
                    value_ref="shape:x",
                ),
                _atom(
                    roots,
                    family="constructors",
                    precedence="reviewed_idl",
                    clause_ref="ctor:x",
                    value_ref="ctor:x",
                ),
            ],
            value_provenance=_ForeignGraph(),
            include_requirement_atoms=False,
        )


def test_mismatched_requirement_roots_fail_closed(
    roots: PropagationAuthorityRoots,
) -> None:
    other = PropagationAuthorityRoots(
        repository_id="repository:rpr-034",
        base_forest_id="forest:base",
        base_tree_id="tree:base",
        base_overlay_id="overlay:base",
        candidate_forest_id="forest:candidate",
        candidate_tree_id="tree:other-candidate",
        candidate_overlay_id="overlay:candidate",
        graph_id="graph:rpr-034",
        index_id="index:rpr-034",
        model_id="model:rpr-034",
        config_id="config:rpr-034",
        translator_id="translator:rpr-034",
        toolchain_id="toolchain:rpr-034",
        policy_id="policy:rpr-034",
    )
    requirement = _requirement(other)
    with pytest.raises(RequiredBehaviorSynthesisAuthorityError, match="roots"):
        RequiredBehaviorSynthesizer(roots).synthesize(
            requirement,
            kind=BehaviorKind.CLASS,
            subject_symbol_id="symbol:SupportContext",
            evidence=(),
        )


def test_body_and_secret_payloads_are_rejected(
    roots: PropagationAuthorityRoots,
) -> None:
    with pytest.raises(RequiredBehaviorSynthesisError, match="bodies|secrets"):
        BehaviorEvidenceAtom.from_mapping(
            roots,
            {
                "family": "fields",
                "precedence": "reviewed_idl",
                "clause_ref": "field:x",
                "value_ref": "shape:x",
                "subject_symbol_id": "symbol:SupportContext",
                "source_body": "class SupportContext: pass",
            },
        )


def test_does_not_redefine_required_behavior_contract() -> None:
    """Conflict policy: import and return the canonical RPR-022 contract."""
    from ipfs_accelerate_py.agent_supervisor.analysis import (
        required_behavior_synthesis as module,
    )
    from ipfs_accelerate_py.agent_supervisor.analysis.change_propagation_contracts import (
        RequiredBehaviorContract as Canonical,
    )

    assert module.RequiredBehaviorContract is Canonical
    assert module.BehaviorEvidencePrecedence is BehaviorEvidencePrecedence


def test_memory_facet_and_program_contract_refs_are_recorded(
    roots: PropagationAuthorityRoots,
) -> None:
    requirement = _requirement(roots)

    @dataclass(frozen=True)
    class _Facet:
        content_id: str = "facet:memory:support"

    @dataclass(frozen=True)
    class _ProgramContract:
        content_id: str = "program-contract:support@1"

    receipt = RequiredBehaviorSynthesizer(roots).synthesize(
        requirement,
        kind=BehaviorKind.CLASS,
        subject_symbol_id="symbol:SupportContext",
        evidence=[
            _atom(
                roots,
                family="fields",
                precedence="reviewed_idl",
                clause_ref="field:trace_id",
                value_ref="shape:trace_id",
            ),
            _atom(
                roots,
                family="constructors",
                precedence="reviewed_idl",
                clause_ref="ctor:SupportContext",
                value_ref="ctor:total",
            ),
        ],
        memory_facet=_Facet(),
        program_contract=_ProgramContract(),
        include_requirement_atoms=False,
    )
    assert receipt.admitted
    assert receipt.memory_facet_ref == "facet:memory:support"
    assert receipt.program_contract_ref == "program-contract:support@1"
