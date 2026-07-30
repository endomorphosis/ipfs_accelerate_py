"""Tests for VFS-019 code-contract → ipfs_datasets_py IR translation."""

from __future__ import annotations

import json
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.code_contract_logic import (
    CODE_CONTRACT_LOGIC_VERSION,
    LOGIC_FAMILY,
    LOGIC_TRANSLATION_EVIDENCE,
    RULESET_ID,
    RULESET_VERSION,
    TRANSLATOR_ID,
    TRANSLATOR_VERSION,
    ArgumentSort,
    CallSliceBinding,
    CodeContractLogicError,
    ConformanceReceipt,
    ContractPredicate,
    LogicAssumption,
    PredicateArgument,
    PredicateRelation,
    RejectionCode,
    SupportedPredicateKind,
    TranslationRejectedError,
    TranslationRequest,
    TranslationResult,
    TranslationStatus,
    UnsupportedResidual,
    extract_assumptions_from_contract,
    extract_predicates_from_contract,
    extract_reachability_predicates,
    make_predicate,
    pinned_translator_identity,
    project_reviewed_formula,
    reconstruct_predicate_from_claim,
    round_trip_predicates,
    translate,
    translate_contract,
    translator_identity,
    verify_conformance_receipt,
)
from ipfs_accelerate_py.agent_supervisor.program_contracts import (
    Assumption,
    AtomicityMode,
    AtomicitySpec,
    AuthorizationMode,
    AuthorizationSpec,
    CapabilityMode,
    CapabilitySpec,
    ConfidenceClass,
    ConsistencyMode,
    ConsistencySpec,
    ContractSourceKind,
    DegradationMode,
    EffectKind,
    EffectPolarity,
    ErrorSpec,
    ExpectedProgramContract,
    FallbackSpec,
    IdempotenceMode,
    IdempotenceSpec,
    InterfaceIdentity,
    OrderingMode,
    OrderingSpec,
    ParameterKind,
    ParameterSpec,
    ProgramContractRole,
    ReturnSpec,
    SemanticAspect,
    SideEffectSpec,
    SourceReference,
    SupportStatus,
    SymbolIdentity,
    SyncAsyncSpec,
    SyncMode,
    TypeConstructor,
    TypeShape,
    UnsupportedSemantics,
)
from ipfs_datasets_py.logic.ir_core.claims import IRClaim


POLICY = "policy:vfs-019-test@1"
SHA_A = "a" * 64
SOURCE_CID = "baguqeer" + "a" * 50


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def symbol() -> SymbolIdentity:
    return SymbolIdentity(
        repository_id="repo:vfs",
        tree_id="tree:abc",
        module_path="ipfs_kit_py/vfs.py",
        symbol_name="read_bytes",
        language="python",
        blob_cid="baguqeer" + "b" * 50,
    )


def interface() -> InterfaceIdentity:
    return InterfaceIdentity(
        interface_name="VFS.read_bytes",
        surface="python",
        version="1",
    )


def source(
    *,
    kind: ContractSourceKind = ContractSourceKind.REVIEWED_INTERFACE,
    artifact_id: str = "artifact:idl",
    locator: str = "VFS.read_bytes",
) -> SourceReference:
    return SourceReference(
        source_kind=kind,
        role=ProgramContractRole.EXPECTED,
        artifact_id=artifact_id,
        locator=locator,
        extractor_rule="idl_v1",
        confidence=ConfidenceClass.HIGH,
        sha256=f"sha256:{SHA_A}",
    )


def string_type(*, nullable: bool = False) -> TypeShape:
    return TypeShape(
        constructor=TypeConstructor.STRING,
        name="str",
        nullable=nullable,
    )


def bytes_type() -> TypeShape:
    return TypeShape(constructor=TypeConstructor.BYTES, name="bytes")


def expected_contract(**kwargs: Any) -> ExpectedProgramContract:
    return ExpectedProgramContract(
        symbol=kwargs.pop("symbol", symbol()),
        interface=kwargs.pop("interface", interface()),
        policy_revision=kwargs.pop("policy_revision", POLICY),
        sources=kwargs.pop("sources", (source(),)),
        inputs=kwargs.pop(
            "inputs",
            (
                ParameterSpec(
                    name="path",
                    type_shape=string_type(),
                    kind=ParameterKind.POSITIONAL,
                    position=0,
                ),
            ),
        ),
        returns=kwargs.pop(
            "returns",
            ReturnSpec(type_shape=bytes_type(), description="file bytes"),
        ),
        errors=kwargs.pop(
            "errors",
            (
                ErrorSpec(error_name="PathEscapeError", code="PATH_ESCAPE"),
                ErrorSpec(error_name="NotFound", code="NOT_FOUND"),
            ),
        ),
        sync_async=kwargs.pop("sync_async", SyncAsyncSpec(mode=SyncMode.SYNC)),
        side_effects=kwargs.pop(
            "side_effects",
            (
                SideEffectSpec(
                    effect_kind=EffectKind.FILESYSTEM,
                    polarity=EffectPolarity.ALLOWED,
                    target="path",
                ),
                SideEffectSpec(
                    effect_kind=EffectKind.WRITE,
                    polarity=EffectPolarity.FORBIDDEN,
                ),
            ),
        ),
        capabilities=kwargs.pop(
            "capabilities",
            (
                CapabilitySpec(
                    capability_name="vfs.read",
                    mode=CapabilityMode.REQUIRED,
                    version="1",
                ),
            ),
        ),
        authorization=kwargs.pop(
            "authorization",
            AuthorizationSpec(
                mode=AuthorizationMode.PATH_SCOPE,
                scopes=("repo:read",),
                policies=("path-scope-v1",),
            ),
        ),
        idempotence=kwargs.pop(
            "idempotence", IdempotenceSpec(mode=IdempotenceMode.PURE)
        ),
        ordering=kwargs.pop(
            "ordering", OrderingSpec(mode=OrderingMode.UNORDERED)
        ),
        atomicity=kwargs.pop(
            "atomicity", AtomicitySpec(mode=AtomicityMode.ATOMIC)
        ),
        consistency=kwargs.pop(
            "consistency", ConsistencySpec(mode=ConsistencyMode.STRONG)
        ),
        fallback=kwargs.pop(
            "fallback",
            FallbackSpec(
                mode=DegradationMode.FAIL_CLOSED,
                description="fail closed",
            ),
        ),
        assumptions=kwargs.pop(
            "assumptions",
            (
                Assumption(
                    statement="path is repository-relative",
                    aspect=SemanticAspect.INPUTS,
                    confidence=ConfidenceClass.HIGH,
                ),
            ),
        ),
        unsupported=kwargs.pop("unsupported", ()),
        summary=kwargs.pop("summary", "VFS read contract"),
        **kwargs,
    )


def closed_slice(
    *,
    nodes: tuple[str, ...] = ("node:a", "node:b", "node:c"),
    depth_bound: int = 3,
) -> CallSliceBinding:
    return CallSliceBinding(
        slice_cid="slice:closed-1",
        node_ids=nodes,
        edge_ids=("edge:a-b", "edge:b-c"),
        complete=True,
        dependency_complete=True,
        truncated=False,
        presented_as_closed=True,
        depth_bound=depth_bound,
        forest_id="forest:test",
        graph_id="graph:test",
    )


def partial_slice() -> CallSliceBinding:
    return CallSliceBinding(
        slice_cid="slice:partial-1",
        node_ids=("node:a", "node:b"),
        edge_ids=("edge:a-b",),
        complete=False,
        dependency_complete=False,
        truncated=True,
        presented_as_closed=False,
        depth_bound=1,
        forest_id="forest:test",
        graph_id="graph:test",
    )


# ---------------------------------------------------------------------------
# Valid translations
# ---------------------------------------------------------------------------


def test_translate_contract_valid_emits_claims_and_receipt() -> None:
    contract = expected_contract()
    result = translate_contract(contract)

    assert result.status is TranslationStatus.TRANSLATED
    assert result.successful
    assert result.claims
    assert all(isinstance(c, IRClaim) for c in result.claims)
    assert all(c.domain == LOGIC_FAMILY for c in result.claims)
    assert result.receipt.round_trip_ok
    assert result.receipt.status is TranslationStatus.TRANSLATED
    assert result.receipt.evidence == LOGIC_TRANSLATION_EVIDENCE
    assert result.receipt.translator_identity == pinned_translator_identity()
    assert result.receipt.source_contract_cid == contract.content_id
    assert result.assumptions
    # Every claim obligation carries assumption CIDs and source refs.
    for claim in result.claims:
        assert claim.obligations
        for obligation in claim.obligations:
            assert obligation.logic_family == LOGIC_FAMILY
            assert obligation.source_refs
            for assumption_id in obligation.assumption_ids:
                assert any(
                    a.assumption_id == assumption_id for a in claim.assumptions
                )


def test_translate_covers_all_supported_predicate_kinds() -> None:
    contract = expected_contract()
    slice_ = closed_slice()
    result = translate_contract(
        contract,
        call_slice=slice_,
        reachability_pairs=(("node:a", "node:c"),),
    )
    kinds = {p.kind for p in result.predicates}
    expected_kinds = set(SupportedPredicateKind)
    assert expected_kinds.issubset(kinds)


def test_round_trip_reconstruct_matches_statement_atoms() -> None:
    contract = expected_contract()
    result = translate_contract(contract)
    ok, reconstructed = round_trip_predicates(result.predicates, result.claims)
    assert ok
    assert len(reconstructed) == len(result.predicates)
    for predicate, claim in zip(result.predicates, result.claims):
        rebuilt = reconstruct_predicate_from_claim(claim)
        assert rebuilt.statement_atom() == predicate.statement_atom()


def test_identity_stable_across_serialize() -> None:
    contract = expected_contract()
    result = translate_contract(contract)
    payload = result.to_dict()
    again = TranslationResult.from_dict(payload)
    assert again.result_cid == result.result_cid
    assert again.receipt.receipt_cid == result.receipt.receipt_cid
    assert [c.digest for c in again.claims] == [c.digest for c in result.claims]


def test_extract_predicates_type_and_nullability() -> None:
    contract = expected_contract(
        inputs=(
            ParameterSpec(
                name="path",
                type_shape=string_type(nullable=True),
                kind=ParameterKind.POSITIONAL,
                position=0,
            ),
        )
    )
    assumptions = extract_assumptions_from_contract(contract)
    preds, _ = extract_predicates_from_contract(
        contract,
        assumption_cids=tuple(a.assumption_cid for a in assumptions),
    )
    nulls = [p for p in preds if p.kind is SupportedPredicateKind.NULLABILITY]
    assert nulls
    assert any(
        a.name == "nullable" and a.value is True
        for p in nulls
        for a in p.arguments
    )


def test_vocabulary_projection_for_authorization_and_reachability() -> None:
    auth = make_predicate(
        SupportedPredicateKind.AUTHORIZATION,
        "sym:read",
        "path_scope",
        "repo:read",
        source_cid=SOURCE_CID,
    )
    formula = project_reviewed_formula(auth)
    assert formula is not None
    assert formula.predicate is not None

    reach = make_predicate(
        SupportedPredicateKind.BOUNDED_REACHABILITY,
        "node:a",
        "node:b",
        4,
        source_cid=SOURCE_CID,
        closed=True,
    )
    formula2 = project_reviewed_formula(reach)
    assert formula2 is not None


def test_translate_with_closed_call_slice_reachability() -> None:
    contract = expected_contract()
    result = translate_contract(
        contract,
        call_slice=closed_slice(),
        reachability_pairs=(("node:a", "node:c"),),
    )
    assert result.status is TranslationStatus.TRANSLATED
    reach = [
        p
        for p in result.predicates
        if p.kind is SupportedPredicateKind.BOUNDED_REACHABILITY
    ]
    assert reach
    assert all(p.closed for p in reach)
    assert result.receipt.call_slice_cid == "slice:closed-1"


def test_verify_conformance_receipt_accepts_current_pin() -> None:
    result = translate_contract(expected_contract())
    verified = verify_conformance_receipt(result.receipt)
    assert verified.receipt_cid == result.receipt.receipt_cid
    # Dict form also works.
    verified2 = verify_conformance_receipt(result.receipt.to_dict())
    assert verified2.receipt_cid == result.receipt.receipt_cid


def test_direct_predicate_translation_request() -> None:
    assumption = LogicAssumption(
        statement="caller is authorized",
        binders=(("actor", ArgumentSort.SCOPE),),
        source_cid=SOURCE_CID,
    )
    pred = make_predicate(
        SupportedPredicateKind.IDEMPOTENCE,
        "sym:write",
        "idempotent",
        source_cid=SOURCE_CID,
        assumption_cids=(assumption.assumption_cid,),
    )
    request = TranslationRequest(
        source_contract_cid=SOURCE_CID,
        predicates=(pred,),
        assumptions=(assumption,),
    )
    result = translate(request)
    assert result.status is TranslationStatus.TRANSLATED
    assert len(result.claims) == 1
    claim = result.claims[0]
    assert claim.assumptions[0].assumption_id == assumption.assumption_cid
    assert claim.obligations[0].assumption_ids == (assumption.assumption_cid,)


# ---------------------------------------------------------------------------
# Invalid / rejection cases
# ---------------------------------------------------------------------------


def test_reject_sort_mismatch() -> None:
    with pytest.raises(TranslationRejectedError) as exc:
        PredicateArgument(name="bound", sort=ArgumentSort.BOUND, value="not-int")
    assert exc.value.code is RejectionCode.SORT_MISMATCH

    with pytest.raises(TranslationRejectedError) as exc2:
        ContractPredicate(
            kind=SupportedPredicateKind.TYPE,
            relation=PredicateRelation.HAS_TYPE,
            arguments=(
                PredicateArgument("symbol", ArgumentSort.SYMBOL, "s"),
                PredicateArgument("slot", ArgumentSort.STRING, "in:x"),
                # wrong sort: boolean instead of type
                PredicateArgument("type", ArgumentSort.BOOLEAN, True),
            ),
            source_cid=SOURCE_CID,
        )
    assert exc2.value.code is RejectionCode.SORT_MISMATCH


def test_reject_unbound_axiom() -> None:
    pred = make_predicate(
        SupportedPredicateKind.ORDERING,
        "sym:a",
        "total",
        source_cid=SOURCE_CID,
        assumption_cids=("assumption:missing",),
    )
    request = TranslationRequest(
        source_contract_cid=SOURCE_CID,
        predicates=(pred,),
        assumptions=(),
    )
    result = translate(request)
    assert result.status is TranslationStatus.REJECTED
    assert RejectionCode.UNBOUND_AXIOM.value in result.rejection_codes


def test_reject_name_capture_duplicate_binders() -> None:
    with pytest.raises(TranslationRejectedError) as exc:
        LogicAssumption(
            statement="two binders same name",
            binders=(
                ("x", ArgumentSort.SYMBOL),
                ("x", ArgumentSort.TYPE),
            ),
            source_cid=SOURCE_CID,
        )
    assert exc.value.code is RejectionCode.NAME_CAPTURE


def test_reject_name_capture_value_collides_with_binder_sort() -> None:
    assumption = LogicAssumption(
        statement="bound x as symbol",
        binders=(("x", ArgumentSort.SYMBOL),),
        source_cid=SOURCE_CID,
    )
    # Free TYPE-sorted argument whose value equals binder name "x".
    pred = ContractPredicate(
        kind=SupportedPredicateKind.TYPE,
        relation=PredicateRelation.HAS_TYPE,
        arguments=(
            PredicateArgument("symbol", ArgumentSort.SYMBOL, "fn"),
            PredicateArgument("slot", ArgumentSort.STRING, "return"),
            PredicateArgument("type", ArgumentSort.TYPE, "x"),
        ),
        source_cid=SOURCE_CID,
        assumption_cids=(assumption.assumption_cid,),
    )
    result = translate(
        TranslationRequest(
            source_contract_cid=SOURCE_CID,
            predicates=(pred,),
            assumptions=(assumption,),
        )
    )
    assert result.status is TranslationStatus.REJECTED
    assert RejectionCode.NAME_CAPTURE.value in result.rejection_codes


def test_reject_partial_slice_presented_as_closed_on_construction() -> None:
    with pytest.raises(TranslationRejectedError) as exc:
        CallSliceBinding(
            slice_cid="slice:bad",
            node_ids=("n1",),
            complete=False,
            dependency_complete=False,
            truncated=True,
            presented_as_closed=True,
            depth_bound=1,
        )
    assert exc.value.code is RejectionCode.PARTIAL_SLICE_AS_CLOSED


def test_reject_closed_reachability_on_partial_slice() -> None:
    slice_ = partial_slice()
    pred = make_predicate(
        SupportedPredicateKind.BOUNDED_REACHABILITY,
        "node:a",
        "node:b",
        2,
        source_cid=SOURCE_CID,
        closed=True,
    )
    result = translate(
        TranslationRequest(
            source_contract_cid=SOURCE_CID,
            predicates=(pred,),
            call_slice=slice_,
        )
    )
    assert result.status is TranslationStatus.REJECTED
    assert RejectionCode.PARTIAL_SLICE_AS_CLOSED.value in result.rejection_codes


def test_reject_closed_reachability_without_slice_binding() -> None:
    pred = make_predicate(
        SupportedPredicateKind.BOUNDED_REACHABILITY,
        "node:a",
        "node:b",
        2,
        source_cid=SOURCE_CID,
        closed=True,
    )
    result = translate(
        TranslationRequest(
            source_contract_cid=SOURCE_CID,
            predicates=(pred,),
        )
    )
    assert result.status is TranslationStatus.REJECTED
    assert RejectionCode.PARTIAL_SLICE_AS_CLOSED.value in result.rejection_codes


def test_open_reachability_on_partial_slice_translates() -> None:
    slice_ = partial_slice()
    pred = make_predicate(
        SupportedPredicateKind.BOUNDED_REACHABILITY,
        "node:a",
        "node:b",
        1,
        source_cid=SOURCE_CID,
        closed=False,
    )
    result = translate(
        TranslationRequest(
            source_contract_cid=SOURCE_CID,
            predicates=(pred,),
            call_slice=slice_,
        )
    )
    assert result.status is TranslationStatus.TRANSLATED
    assert result.predicates[0].closed is False


def test_reject_silent_approximation_residual_on_supported() -> None:
    pred = make_predicate(
        SupportedPredicateKind.EFFECT,
        "sym:a",
        "write",
        "allowed",
        source_cid=SOURCE_CID,
        residual="dropped async effect",
    )
    result = translate(
        TranslationRequest(
            source_contract_cid=SOURCE_CID,
            predicates=(pred,),
            allow_approximation=False,
        )
    )
    assert result.status is TranslationStatus.REJECTED
    assert RejectionCode.SILENT_APPROXIMATION.value in result.rejection_codes


def test_reject_changed_translator_ruleset_reuse() -> None:
    result = translate(
        TranslationRequest(
            source_contract_cid=SOURCE_CID,
            predicates=(
                make_predicate(
                    SupportedPredicateKind.ORDERING,
                    "sym:a",
                    "total",
                    source_cid=SOURCE_CID,
                ),
            ),
            translator_version="999-not-current",
        )
    )
    assert result.status is TranslationStatus.REJECTED
    assert RejectionCode.TRANSLATOR_RULESET_REUSE.value in result.rejection_codes

    # A valid receipt cannot be verified under a foreign pin.
    good = translate_contract(expected_contract())
    with pytest.raises(TranslationRejectedError) as exc:
        verify_conformance_receipt(
            good.receipt,
            expected_translator_identity=translator_identity(
                translator_version="other"
            ),
        )
    assert exc.value.code is RejectionCode.TRANSLATOR_RULESET_REUSE


def test_reject_forged_receipt_translator_identity() -> None:
    good = translate_contract(expected_contract())
    payload = good.receipt.to_dict()
    payload["translator_identity"] = translator_identity(
        translator_version="forged"
    )
    # Keep declared version as current so internal mismatch is detected.
    with pytest.raises(TranslationRejectedError) as exc:
        ConformanceReceipt.from_dict(payload)
    assert exc.value.code is RejectionCode.TRANSLATOR_RULESET_REUSE


def test_reject_empty_translation() -> None:
    result = translate(
        TranslationRequest(
            source_contract_cid=SOURCE_CID,
            predicates=(),
        )
    )
    assert result.status is TranslationStatus.REJECTED
    assert RejectionCode.EMPTY_TRANSLATION.value in result.rejection_codes


# ---------------------------------------------------------------------------
# Ambiguous + unsupported
# ---------------------------------------------------------------------------


def test_ambiguous_predicate_marked() -> None:
    pred = make_predicate(
        SupportedPredicateKind.ERROR,
        "sym:a",
        "E1",
        False,
        source_cid=SOURCE_CID,
        ambiguity="two candidate error taxonomies",
    )
    result = translate(
        TranslationRequest(
            source_contract_cid=SOURCE_CID,
            predicates=(pred,),
        )
    )
    assert result.status is TranslationStatus.AMBIGUOUS
    assert RejectionCode.AMBIGUOUS_PREDICATE.value in result.rejection_codes
    assert result.claims == ()


def test_conflicting_duplicate_atoms_are_ambiguous() -> None:
    base_args = ("sym:a", "pure")
    p1 = make_predicate(
        SupportedPredicateKind.IDEMPOTENCE,
        *base_args,
        source_cid=SOURCE_CID,
        closed=True,
    )
    p2 = make_predicate(
        SupportedPredicateKind.IDEMPOTENCE,
        *base_args,
        source_cid=SOURCE_CID + "x",
        closed=False,
    )
    result = translate(
        TranslationRequest(
            source_contract_cid=SOURCE_CID,
            predicates=(p1, p2),
        )
    )
    assert result.status is TranslationStatus.AMBIGUOUS


def test_unsupported_only_status() -> None:
    residual = UnsupportedResidual(
        aspect="effects",
        reason="unbounded external service effects",
        residual="EXTERNAL_SERVICE/*",
        source_cid=SOURCE_CID,
        predicate_kind=SupportedPredicateKind.EFFECT.value,
    )
    result = translate(
        TranslationRequest(
            source_contract_cid=SOURCE_CID,
            predicates=(),
            unsupported=(residual,),
        )
    )
    assert result.status is TranslationStatus.UNSUPPORTED
    assert result.unsupported
    assert result.receipt.round_trip_ok is False


def test_unsupported_type_shape_emitted_explicitly() -> None:
    contract = expected_contract(
        inputs=(
            ParameterSpec(
                name="handle",
                type_shape=TypeShape(
                    constructor=TypeConstructor.UNSUPPORTED,
                    name="OpaqueHandle",
                    support=SupportStatus.UNSUPPORTED,
                ),
                kind=ParameterKind.POSITIONAL,
                position=0,
            ),
        ),
        returns=ReturnSpec(type_shape=bytes_type()),
    )
    assumptions = extract_assumptions_from_contract(contract)
    preds, unsupported = extract_predicates_from_contract(
        contract,
        assumption_cids=tuple(a.assumption_cid for a in assumptions),
    )
    assert unsupported
    assert all(
        p.kind is not SupportedPredicateKind.TYPE
        or "handle" not in str(p.arguments)
        for p in preds
        if p.kind is SupportedPredicateKind.TYPE
    )
    # Full translation still succeeds for remaining supported aspects.
    result = translate_contract(contract)
    assert result.status is TranslationStatus.TRANSLATED
    assert result.unsupported


def test_contract_unsupported_semantics_propagated() -> None:
    contract = expected_contract(
        unsupported=(
            UnsupportedSemantics(
                aspect=SemanticAspect.RESOURCE_BOUNDS,
                reason="no finite resource model",
                residual="max_fds",
            ),
        )
    )
    result = translate_contract(contract)
    assert result.status is TranslationStatus.TRANSLATED
    assert any(
        u.aspect == SemanticAspect.RESOURCE_BOUNDS.value
        for u in result.unsupported
    )


# ---------------------------------------------------------------------------
# Source / assumption CID binding
# ---------------------------------------------------------------------------


def test_claims_bind_source_and_assumption_cids() -> None:
    contract = expected_contract()
    result = translate_contract(contract)
    assumption_cids = {a.assumption_cid for a in result.assumptions}
    assert assumption_cids
    for claim in result.claims:
        assert claim.source_refs
        for obligation in claim.obligations:
            for aid in obligation.assumption_ids:
                assert aid in assumption_cids
            assert obligation.source_refs
        atom = json.loads(claim.statement)
        assert atom["source_cid"]
        assert "kind" in atom
        assert "relation" in atom


def test_receipt_binds_request_and_claim_digests() -> None:
    result = translate_contract(expected_contract())
    receipt = result.receipt
    assert receipt.request_cid == result.request_cid
    assert receipt.claim_digests
    assert set(receipt.claim_digests) == {c.digest for c in result.claims}
    assert receipt.predicate_cids
    payload = receipt.to_dict()
    assert payload["logic_version"] == CODE_CONTRACT_LOGIC_VERSION
    assert payload["translator_id"] == TRANSLATOR_ID
    assert payload["ruleset_id"] == RULESET_ID
    assert payload["ruleset_version"] == RULESET_VERSION
    assert payload["translator_version"] == TRANSLATOR_VERSION


# ---------------------------------------------------------------------------
# Helpers and identity pins
# ---------------------------------------------------------------------------


def test_translator_identity_deterministic() -> None:
    a = translator_identity()
    b = pinned_translator_identity()
    assert a == b
    assert a != translator_identity(ruleset_version="2")


def test_make_predicate_arity_enforced() -> None:
    with pytest.raises(TranslationRejectedError):
        make_predicate(
            SupportedPredicateKind.ORDERING,
            "only-one-arg",
            source_cid=SOURCE_CID,
        )


def test_predicate_content_id_forged_rejected() -> None:
    pred = make_predicate(
        SupportedPredicateKind.ORDERING,
        "sym:a",
        "total",
        source_cid=SOURCE_CID,
    )
    payload = pred.to_dict()
    payload["content_id"] = "baguqeer" + "f" * 50
    with pytest.raises(CodeContractLogicError):
        ContractPredicate.from_dict(payload)


def test_reachability_extraction_respects_closedness() -> None:
    closed = closed_slice()
    preds = extract_reachability_predicates(
        closed,
        source_cid=SOURCE_CID,
        pairs=(("node:a", "node:c"),),
    )
    assert preds and preds[0].closed is True

    open_preds = extract_reachability_predicates(
        partial_slice(),
        source_cid=SOURCE_CID,
        pairs=(("node:a", "node:b"),),
    )
    assert open_preds and open_preds[0].closed is False


def test_logic_assumption_to_ir_round_trip() -> None:
    assumption = LogicAssumption(
        statement="finite call depth",
        binders=(("depth", ArgumentSort.BOUND),),
        source_cid=SOURCE_CID,
    )
    ir = assumption.to_ir_assumption()
    assert ir.assumption_id == assumption.assumption_cid
    assert ir.schema_version
    again = LogicAssumption.from_dict(assumption.to_dict())
    assert again.assumption_cid == assumption.assumption_cid


def test_all_predicate_kinds_constructible() -> None:
    builders = {
        SupportedPredicateKind.TYPE: ("s", "return", "bytes"),
        SupportedPredicateKind.NULLABILITY: ("s", "return", False),
        SupportedPredicateKind.ERROR: ("s", "E", False),
        SupportedPredicateKind.EFFECT: ("s", "read", "allowed"),
        SupportedPredicateKind.AUTHORIZATION: ("s", "path_scope", "repo:read"),
        SupportedPredicateKind.STATE_TRANSITION: ("s", "pre", "post"),
        SupportedPredicateKind.ORDERING: ("s", "total"),
        SupportedPredicateKind.IDEMPOTENCE: ("s", "pure"),
        SupportedPredicateKind.BOUNDED_REACHABILITY: ("n1", "n2", 3),
    }
    for kind, args in builders.items():
        pred = make_predicate(kind, *args, source_cid=SOURCE_CID, closed=False)
        assert pred.kind is kind
        assert pred.relation is not None

