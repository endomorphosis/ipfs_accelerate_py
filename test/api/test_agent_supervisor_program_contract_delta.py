"""Tests for exact before/after semantic program contract deltas (RPR-026)."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.change_propagation_contracts import (
    ChangeSetKind,
    DeltaDisposition,
    DeltaKind,
    ProgramChangeSet,
    PropagationAuthorityRoots,
)
from ipfs_accelerate_py.agent_supervisor.analysis.program_contract_delta import (
    DOMAIN_MEMORY,
    DOMAIN_PUBLIC_API,
    DOMAIN_PYTHON_CALLERS,
    DOMAIN_REGISTRATION,
    DOMAIN_SCHEMA_CONSUMERS,
    CrossRootContractDeltaError,
    IncompleteContractDeltaError,
    MovePair,
    NonSemanticChurnKind,
    PathChurnClassification,
    ProgramContractDeltaAnalyzer,
    ProgramContractDeltaError,
    ProgramContractDeltaRequest,
    RenamePair,
    StaleContractDeltaError,
    StructuralSurfaceChange,
    StructuralSurfaceKind,
    normalize_change_partition,
)
from ipfs_accelerate_py.agent_supervisor.program_contracts import (
    Applicability,
    AuthorizationMode,
    AuthorizationSpec,
    CapabilityMode,
    CapabilitySpec,
    ConfidenceClass,
    ConsistencyMode,
    ConsistencySpec,
    ContractSourceKind,
    EffectKind,
    EffectPolarity,
    ErrorSpec,
    ExpectedProgramContract,
    IdempotenceMode,
    IdempotenceSpec,
    InterfaceIdentity,
    Optionality,
    OrderingMode,
    OrderingSpec,
    ParameterKind,
    ParameterSpec,
    ProgramContractRole,
    ResourceBounds,
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


POLICY = "policy:rpr-026@1"
REPO = "repository:fixture"
BASE_TREE = "tree:base"
CAND_TREE = "tree:candidate"
SHA = "a" * 64


@pytest.fixture
def roots() -> PropagationAuthorityRoots:
    return PropagationAuthorityRoots(
        repository_id=REPO,
        base_forest_id="forest:base",
        base_tree_id=BASE_TREE,
        base_overlay_id="overlay:base",
        candidate_forest_id="forest:candidate",
        candidate_tree_id=CAND_TREE,
        candidate_overlay_id="overlay:candidate",
        graph_id="graph:one",
        index_id="index:one",
        model_id="model:one",
        config_id="config:one",
        translator_id="translator:one",
        toolchain_id="toolchain:one",
        policy_id="policy:one",
    )


def _symbol(
    *,
    tree_id: str,
    name: str = "process",
    module: str = "pkg/process.py",
    qualified: str = "",
) -> SymbolIdentity:
    return SymbolIdentity(
        repository_id=REPO,
        tree_id=tree_id,
        module_path=module,
        symbol_name=name,
        qualified_name=qualified or f"{module}:{name}",
        language="python",
        span_start=1,
        span_end=20,
        blob_cid="baguqeera" + ("1" if tree_id == BASE_TREE else "2") * 50,
    )


def _interface(**overrides) -> InterfaceIdentity:
    payload = {
        "interface_name": "pkg.process",
        "surface": "python",
        "version": "1.0",
        "method": "process",
        "protocol": "local",
        "media_type": "application/json",
        "path_or_uri": "pkg.process",
    }
    payload.update(overrides)
    return InterfaceIdentity(**payload)


def _source(
    *,
    kind: ContractSourceKind = ContractSourceKind.REVIEWED_INTERFACE,
    artifact_id: str = "artifact:idl",
    locator: str = "idl#process",
) -> SourceReference:
    return SourceReference(
        source_kind=kind,
        role=ProgramContractRole.EXPECTED,
        artifact_id=artifact_id,
        locator=locator,
        extractor_rule="idl_v1",
        confidence=ConfidenceClass.HIGH,
        sha256=f"sha256:{SHA}",
        span_start=1,
        span_end=10,
    )


def _type(name: str = "str", constructor: TypeConstructor = TypeConstructor.STRING, **kw) -> TypeShape:
    return TypeShape(constructor=constructor, name=name, **kw)


def _param(
    name: str,
    *,
    position: int | None = 0,
    optionality: Optionality = Optionality.REQUIRED,
    kind: ParameterKind = ParameterKind.POSITIONAL,
    type_shape: TypeShape | None = None,
    default_summary: str = "",
) -> ParameterSpec:
    return ParameterSpec(
        name=name,
        type_shape=type_shape or _type(),
        kind=kind,
        optionality=optionality,
        position=position,
        default_summary=default_summary,
    )


def _expected(
    *,
    tree_id: str,
    inputs: tuple[ParameterSpec, ...] | None = None,
    returns: ReturnSpec | None = None,
    symbol: SymbolIdentity | None = None,
    interface: InterfaceIdentity | None = None,
    **kwargs,
) -> ExpectedProgramContract:
    return ExpectedProgramContract(
        symbol=symbol or _symbol(tree_id=tree_id),
        interface=interface or _interface(),
        policy_revision=kwargs.pop("policy_revision", POLICY),
        sources=kwargs.pop(
            "sources",
            (
                _source(),
                _source(
                    kind=ContractSourceKind.PUBLIC_SIGNATURE,
                    artifact_id="artifact:sig",
                    locator="def process",
                ),
            ),
        ),
        inputs=inputs
        if inputs is not None
        else (
            _param("a", position=0),
            _param("b", position=1),
        ),
        returns=returns
        if returns is not None
        else ReturnSpec(type_shape=_type("int", TypeConstructor.INT)),
        errors=kwargs.pop(
            "errors",
            (ErrorSpec(error_name="ValueError", code="VALUE"),),
        ),
        sync_async=kwargs.pop("sync_async", SyncAsyncSpec(mode=SyncMode.SYNC)),
        side_effects=kwargs.pop(
            "side_effects",
            (
                SideEffectSpec(
                    effect_kind=EffectKind.NONE,
                    polarity=EffectPolarity.ALLOWED,
                ),
            ),
        ),
        capabilities=kwargs.pop(
            "capabilities",
            (
                CapabilitySpec(
                    capability_name="pkg.process",
                    mode=CapabilityMode.REQUIRED,
                ),
            ),
        ),
        authorization=kwargs.pop(
            "authorization",
            AuthorizationSpec(mode=AuthorizationMode.NONE),
        ),
        idempotence=kwargs.pop(
            "idempotence", IdempotenceSpec(mode=IdempotenceMode.PURE)
        ),
        ordering=kwargs.pop(
            "ordering", OrderingSpec(mode=OrderingMode.UNORDERED)
        ),
        consistency=kwargs.pop(
            "consistency", ConsistencySpec(mode=ConsistencyMode.STRONG)
        ),
        resource_bounds=kwargs.pop("resource_bounds", None),
        applicability=kwargs.pop("applicability", None),
        unsupported=kwargs.pop("unsupported", ()),
        summary=kwargs.pop("summary", "process(a, b) -> int"),
        **kwargs,
    )


def _change_set(
    roots: PropagationAuthorityRoots,
    *paths: str,
    generated: tuple[str, ...] = (),
    tombstones: tuple[str, ...] = (),
) -> ProgramChangeSet:
    return ProgramChangeSet(
        roots=roots,
        kind=ChangeSetKind.REVIEWED_BASE_CANDIDATE,
        producer_id="producer:normalized-diff",
        changed_paths=paths or ("pkg/process.py",),
        tombstone_paths=tombstones,
        generated_manifest_ids=generated,
        evidence_refs=("evidence:diff",),
    )


def _request(
    roots: PropagationAuthorityRoots,
    before: ExpectedProgramContract,
    after: ExpectedProgramContract,
    *,
    consumer_domain: str = DOMAIN_PYTHON_CALLERS,
    change_set: ProgramChangeSet | None = None,
    **kwargs,
) -> ProgramContractDeltaRequest:
    return ProgramContractDeltaRequest(
        roots=roots,
        change_set=change_set or _change_set(roots, "pkg/process.py"),
        before=before,
        after=after,
        consumer_domain=consumer_domain,
        subject_symbol_id=kwargs.pop("subject_symbol_id", "symbol:process"),
        **kwargs,
    )


@pytest.fixture
def analyzer() -> ProgramContractDeltaAnalyzer:
    return ProgramContractDeltaAnalyzer()


def test_interface_constants(analyzer: ProgramContractDeltaAnalyzer) -> None:
    assert analyzer.INTERFACE == "ProgramContractDeltaAnalyzer@1"
    assert analyzer.VERSION == "1"


def test_formatting_and_generated_churn_do_not_manufacture_deltas(
    roots: PropagationAuthorityRoots, analyzer: ProgramContractDeltaAnalyzer
) -> None:
    before = _expected(tree_id=BASE_TREE)
    after = _expected(tree_id=CAND_TREE)
    change_set = _change_set(
        roots,
        "pkg/README.md",
        "pkg/generated/client_pb2.py",
        "pkg/.editorconfig",
        generated=("pkg/generated/client_pb2.py",),
    )
    result = analyzer.analyze(
        _request(
            roots,
            before,
            after,
            change_set=change_set,
            path_churn=(
                PathChurnClassification(
                    path="pkg/.editorconfig",
                    kind=NonSemanticChurnKind.FORMATTING,
                ),
            ),
        )
    )
    assert result.pure_non_semantic is True
    assert result.deltas == ()
    assert result.partition.is_purely_non_semantic
    kinds = {item.kind for item in result.partition.non_semantic}
    assert NonSemanticChurnKind.COMMENT in kinds
    assert NonSemanticChurnKind.GENERATED in kinds
    assert NonSemanticChurnKind.FORMATTING in kinds


def test_move_and_rename_partitioned_separately_from_semantic(
    roots: PropagationAuthorityRoots, analyzer: ProgramContractDeltaAnalyzer
) -> None:
    before = _expected(
        tree_id=BASE_TREE,
        symbol=_symbol(tree_id=BASE_TREE, module="pkg/old.py", name="process"),
    )
    after = _expected(
        tree_id=CAND_TREE,
        symbol=_symbol(tree_id=CAND_TREE, module="pkg/new.py", name="process"),
    )
    change_set = _change_set(roots, "pkg/old.py", "pkg/new.py")
    partition = normalize_change_partition(
        change_set,
        move_pairs=(MovePair(before_path="pkg/old.py", after_path="pkg/new.py"),),
    )
    assert partition.is_purely_non_semantic
    assert all(item.kind is NonSemanticChurnKind.MOVE for item in partition.non_semantic)

    result = analyzer.analyze(
        _request(
            roots,
            before,
            after,
            change_set=change_set,
            move_pairs=(MovePair(before_path="pkg/old.py", after_path="pkg/new.py"),),
        )
    )
    # Pure move with equivalent contracts may still emit a symbol_move identity clause.
    assert result.pure_non_semantic is True
    if result.primary_delta is not None:
        kinds = {c.kind for c in result.primary_delta.clauses}
        assert DeltaKind.SYMBOL_MOVE in kinds
        assert all(
            c.disposition is not DeltaDisposition.BREAKING
            or c.kind is DeltaKind.SYMBOL_MOVE
            for c in result.primary_delta.clauses
        )


def test_required_third_argument_is_breaking_for_callers(
    roots: PropagationAuthorityRoots, analyzer: ProgramContractDeltaAnalyzer
) -> None:
    before = _expected(
        tree_id=BASE_TREE,
        inputs=(_param("a", position=0), _param("b", position=1)),
    )
    after = _expected(
        tree_id=CAND_TREE,
        inputs=(
            _param("a", position=0),
            _param("b", position=1),
            _param("context", position=2),
        ),
    )
    result = analyzer.analyze(_request(roots, before, after))
    delta = result.primary_delta
    assert delta is not None
    adds = [c for c in delta.clauses if c.kind is DeltaKind.PARAMETER_ADD]
    assert len(adds) == 1
    assert adds[0].disposition is DeltaDisposition.BREAKING
    assert "context" in adds[0].reason
    assert adds[0].consumer_domain == DOMAIN_PYTHON_CALLERS
    assert delta.breaking_clauses


def test_optional_parameter_add_compatible_for_callers_behavioral_for_schema(
    roots: PropagationAuthorityRoots, analyzer: ProgramContractDeltaAnalyzer
) -> None:
    before = _expected(
        tree_id=BASE_TREE,
        inputs=(_param("a", position=0),),
    )
    after = _expected(
        tree_id=CAND_TREE,
        inputs=(
            _param("a", position=0),
            _param(
                "timeout",
                position=1,
                optionality=Optionality.OPTIONAL,
                default_summary="30",
            ),
        ),
    )
    callers = analyzer.analyze(
        _request(roots, before, after, consumer_domain=DOMAIN_PYTHON_CALLERS)
    )
    schema = analyzer.analyze(
        _request(roots, before, after, consumer_domain=DOMAIN_SCHEMA_CONSUMERS)
    )
    caller_add = next(
        c for c in callers.all_clauses if c.kind is DeltaKind.PARAMETER_ADD
    )
    schema_add = next(
        c for c in schema.all_clauses if c.kind is DeltaKind.PARAMETER_ADD
    )
    assert caller_add.disposition is DeltaDisposition.COMPATIBLE
    assert schema_add.disposition is DeltaDisposition.BEHAVIORAL
    assert caller_add.consumer_domain != schema_add.consumer_domain


def test_parameter_remove_rename_reorder_default_keyword_variance(
    roots: PropagationAuthorityRoots, analyzer: ProgramContractDeltaAnalyzer
) -> None:
    before = _expected(
        tree_id=BASE_TREE,
        inputs=(
            _param("a", position=0, type_shape=_type("int", TypeConstructor.INT)),
            _param("b", position=1),
            _param(
                "flag",
                position=2,
                kind=ParameterKind.POSITIONAL,
                default_summary="False",
            ),
        ),
    )
    after = _expected(
        tree_id=CAND_TREE,
        inputs=(
            _param("b", position=0),  # reorder + keep
            _param(
                "alpha",  # rename of a at new slot — matched by leftover name miss
                position=1,
                type_shape=_type("str"),  # variance narrowing from int perspective: a was int
            ),
            _param(
                "flag",
                position=2,
                kind=ParameterKind.KEYWORD,
                default_summary="True",
            ),
            # c removed (b kept, a effectively renamed/removed depending on match)
        ),
    )
    # Clearer fixture: explicit remove + rename + reorder + default + keyword + variance
    before = _expected(
        tree_id=BASE_TREE,
        inputs=(
            _param("x", position=0, type_shape=_type("str")),
            _param("y", position=1, type_shape=_type("str")),
            _param(
                "z",
                position=2,
                kind=ParameterKind.POSITIONAL,
                default_summary="0",
                type_shape=_type("int", TypeConstructor.INT),
            ),
            _param("doomed", position=3),
        ),
    )
    after = _expected(
        tree_id=CAND_TREE,
        inputs=(
            _param("y", position=0, type_shape=_type("str")),  # reorder of y
            _param(
                "x_new",  # rename of x at position 1? better: same position rename
                position=1,
                type_shape=_type("str"),
            ),
            _param(
                "z",
                position=2,
                kind=ParameterKind.KEYWORD,
                default_summary="1",
                type_shape=_type("int", TypeConstructor.INT, nullable=True),
            ),
        ),
    )
    # Force rename match: same position different name for x -> x_new at pos 0
    before = _expected(
        tree_id=BASE_TREE,
        inputs=(
            _param("x", position=0, type_shape=_type("str")),
            _param("y", position=1, type_shape=_type("str")),
            _param(
                "z",
                position=2,
                kind=ParameterKind.POSITIONAL,
                default_summary="0",
                type_shape=_type("int", TypeConstructor.INT),
            ),
            _param("doomed", position=3),
        ),
    )
    after = _expected(
        tree_id=CAND_TREE,
        inputs=(
            _param("x_renamed", position=0, type_shape=_type("bytes", TypeConstructor.BYTES)),
            _param("y", position=2, type_shape=_type("str")),  # reorder
            _param(
                "z",
                position=1,
                kind=ParameterKind.KEYWORD,
                default_summary="1",
                type_shape=_type("int", TypeConstructor.INT, nullable=True),
            ),
        ),
    )
    result = analyzer.analyze(_request(roots, before, after))
    kinds = {c.kind for c in result.all_clauses}
    assert DeltaKind.PARAMETER_REMOVE in kinds
    assert DeltaKind.PARAMETER_RENAME in kinds
    assert DeltaKind.PARAMETER_REORDER in kinds
    assert DeltaKind.PARAMETER_DEFAULT in kinds
    assert DeltaKind.PARAMETER_KEYWORD in kinds
    assert DeltaKind.PARAMETER_VARIANCE in kinds or DeltaKind.NULLABILITY_CHANGE in kinds


def test_result_generic_nullability_schema_serialization_protocol(
    roots: PropagationAuthorityRoots, analyzer: ProgramContractDeltaAnalyzer
) -> None:
    before = _expected(
        tree_id=BASE_TREE,
        returns=ReturnSpec(
            type_shape=_type(
                "Result",
                TypeConstructor.OBJECT,
                fields=(("value", _type("int", TypeConstructor.INT)),),
            )
        ),
        interface=_interface(
            protocol="http",
            media_type="application/json",
            version="1.0",
            path_or_uri="/v1/process",
        ),
    )
    after = _expected(
        tree_id=CAND_TREE,
        returns=ReturnSpec(
            type_shape=_type(
                "Result",
                TypeConstructor.OBJECT,
                nullable=True,
                fields=(
                    ("value", _type("int", TypeConstructor.INT)),
                    ("meta", _type("str")),
                ),
            )
        ),
        interface=_interface(
            protocol="grpc",
            media_type="application/protobuf",
            version="2.0",
            path_or_uri="/v2/process",
        ),
    )
    result = analyzer.analyze(
        _request(roots, before, after, consumer_domain=DOMAIN_SCHEMA_CONSUMERS)
    )
    kinds = {c.kind for c in result.all_clauses}
    assert DeltaKind.RESULT_CHANGE in kinds or DeltaKind.NULLABILITY_CHANGE in kinds
    assert DeltaKind.SCHEMA_CHANGE in kinds
    assert DeltaKind.SERIALIZATION_CHANGE in kinds
    assert DeltaKind.PROTOCOL_CHANGE in kinds


def test_sync_async_cancellation_error_effect_capability_auth(
    roots: PropagationAuthorityRoots, analyzer: ProgramContractDeltaAnalyzer
) -> None:
    before = _expected(
        tree_id=BASE_TREE,
        sync_async=SyncAsyncSpec(mode=SyncMode.SYNC),
        errors=(ErrorSpec(error_name="ValueError", code="VALUE"),),
        side_effects=(
            SideEffectSpec(
                effect_kind=EffectKind.NONE, polarity=EffectPolarity.ALLOWED
            ),
            SideEffectSpec(
                effect_kind=EffectKind.NETWORK, polarity=EffectPolarity.FORBIDDEN
            ),
        ),
        capabilities=(
            CapabilitySpec(capability_name="pkg.process", mode=CapabilityMode.REQUIRED),
        ),
        authorization=AuthorizationSpec(
            mode=AuthorizationMode.NONE, scopes=()
        ),
    )
    after = _expected(
        tree_id=CAND_TREE,
        sync_async=SyncAsyncSpec(mode=SyncMode.ASYNC, awaitable=True),
        errors=(
            ErrorSpec(error_name="ValueError", code="VALUE"),
            ErrorSpec(error_name="TimeoutError", code="TIMEOUT"),
        ),
        side_effects=(
            SideEffectSpec(
                effect_kind=EffectKind.NETWORK, polarity=EffectPolarity.REQUIRED
            ),
        ),
        capabilities=(
            CapabilitySpec(capability_name="pkg.process", mode=CapabilityMode.REQUIRED),
            CapabilitySpec(capability_name="net.http", mode=CapabilityMode.REQUIRED),
        ),
        authorization=AuthorizationSpec(
            mode=AuthorizationMode.TOKEN, scopes=("write",)
        ),
    )
    result = analyzer.analyze(
        _request(
            roots,
            before,
            after,
            cancellation_before="none",
            cancellation_after="cooperative-cancel",
        )
    )
    kinds = {c.kind for c in result.all_clauses}
    assert DeltaKind.SYNC_ASYNC_CHANGE in kinds
    assert DeltaKind.CANCELLATION_CHANGE in kinds
    assert DeltaKind.ERROR_CHANGE in kinds
    assert DeltaKind.EFFECT_CHANGE in kinds
    assert DeltaKind.CAPABILITY_CHANGE in kinds
    assert DeltaKind.AUTHORIZATION_CHANGE in kinds
    for kind in (
        DeltaKind.SYNC_ASYNC_CHANGE,
        DeltaKind.CANCELLATION_CHANGE,
        DeltaKind.ERROR_CHANGE,
        DeltaKind.EFFECT_CHANGE,
        DeltaKind.CAPABILITY_CHANGE,
        DeltaKind.AUTHORIZATION_CHANGE,
    ):
        assert any(
            c.kind is kind and c.disposition is DeltaDisposition.BREAKING
            for c in result.all_clauses
        )


def test_lifecycle_state_consistency_resource_memory_visibility_registration(
    roots: PropagationAuthorityRoots, analyzer: ProgramContractDeltaAnalyzer
) -> None:
    before = _expected(
        tree_id=BASE_TREE,
        idempotence=IdempotenceSpec(mode=IdempotenceMode.PURE),
        ordering=OrderingSpec(mode=OrderingMode.UNORDERED),
        consistency=ConsistencySpec(mode=ConsistencyMode.STRONG),
        resource_bounds=ResourceBounds(max_memory_bytes=1024),
        applicability=Applicability(surfaces=("public",), always=True),
    )
    after = _expected(
        tree_id=CAND_TREE,
        idempotence=IdempotenceSpec(mode=IdempotenceMode.NON_IDEMPOTENT),
        ordering=OrderingSpec(mode=OrderingMode.SEQUENTIAL),
        consistency=ConsistencySpec(mode=ConsistencyMode.EVENTUAL),
        resource_bounds=ResourceBounds(max_memory_bytes=256),
        applicability=Applicability(surfaces=("internal",), always=False),
    )
    result = analyzer.analyze(
        _request(
            roots,
            before,
            after,
            consumer_domain=DOMAIN_PUBLIC_API,
            memory_facet_before_ref="facet:mem-a",
            memory_facet_after_ref="facet:mem-b",
            registration_changed=True,
            reexport_paths=("pkg/__init__.py",),
        )
    )
    kinds = {c.kind for c in result.all_clauses}
    assert DeltaKind.LIFECYCLE_CHANGE in kinds
    assert DeltaKind.TEMPORAL_STATE_CHANGE in kinds
    assert DeltaKind.CONSISTENCY_CHANGE in kinds
    assert DeltaKind.RESOURCE_CHANGE in kinds
    assert DeltaKind.MEMORY_FACET_CHANGE in kinds
    assert DeltaKind.VISIBILITY_CHANGE in kinds
    assert DeltaKind.SYMBOL_REGISTRATION in kinds
    assert DeltaKind.SYMBOL_REEXPORT in kinds


def test_class_method_field_factory_intro_remove(
    roots: PropagationAuthorityRoots, analyzer: ProgramContractDeltaAnalyzer
) -> None:
    before = _expected(tree_id=BASE_TREE)
    after = _expected(tree_id=CAND_TREE)
    result = analyzer.analyze(
        _request(
            roots,
            before,
            after,
            surface_changes=(
                StructuralSurfaceChange(
                    surface=StructuralSurfaceKind.CLASS,
                    introduced=True,
                    symbol_id="symbol:Service",
                    reason="new service class",
                ),
                StructuralSurfaceChange(
                    surface=StructuralSurfaceKind.METHOD,
                    introduced=True,
                    symbol_id="symbol:Service.run",
                ),
                StructuralSurfaceChange(
                    surface=StructuralSurfaceKind.FIELD,
                    introduced=True,
                    symbol_id="symbol:Service.state",
                ),
                StructuralSurfaceChange(
                    surface=StructuralSurfaceKind.FACTORY,
                    introduced=True,
                    symbol_id="symbol:Service.create",
                ),
                StructuralSurfaceChange(
                    surface=StructuralSurfaceKind.METHOD,
                    introduced=False,
                    symbol_id="symbol:Legacy.run",
                    reason="legacy method removed",
                ),
            ),
        )
    )
    kinds = {c.kind for c in result.all_clauses}
    assert DeltaKind.CLASS_INTRO in kinds
    assert DeltaKind.METHOD_INTRO in kinds
    assert DeltaKind.FIELD_INTRO in kinds
    assert DeltaKind.FACTORY_INTRO in kinds
    assert DeltaKind.METHOD_REMOVE in kinds
    assert any(
        c.kind is DeltaKind.METHOD_REMOVE and c.disposition is DeltaDisposition.BREAKING
        for c in result.all_clauses
    )


def test_unsupported_aspect_emits_unsupported_disposition(
    roots: PropagationAuthorityRoots, analyzer: ProgramContractDeltaAnalyzer
) -> None:
    before = _expected(
        tree_id=BASE_TREE,
        unsupported=(
            UnsupportedSemantics(
                aspect=SemanticAspect.INPUTS,
                reason="dynamic kwargs unmodeled",
            ),
        ),
    )
    after = _expected(
        tree_id=CAND_TREE,
        inputs=(_param("a", position=0), _param("b", position=1), _param("c", position=2)),
    )
    result = analyzer.analyze(_request(roots, before, after))
    unsup = [
        c
        for c in result.all_clauses
        if c.disposition is DeltaDisposition.UNSUPPORTED
    ]
    assert unsup
    assert any(c.kind is DeltaKind.PARAMETER_VARIANCE for c in unsup)


def test_stale_incomplete_cross_root_fail_closed(
    roots: PropagationAuthorityRoots, analyzer: ProgramContractDeltaAnalyzer
) -> None:
    before = _expected(tree_id=BASE_TREE)
    after = _expected(tree_id=CAND_TREE)

    with pytest.raises(StaleContractDeltaError):
        analyzer.analyze(_request(roots, before, after, before_stale=True))

    with pytest.raises(IncompleteContractDeltaError):
        analyzer.analyze(_request(roots, before, after, incomplete=True))

    wrong_after = _expected(
        tree_id="tree:other",
        symbol=_symbol(tree_id="tree:other"),
    )
    with pytest.raises(CrossRootContractDeltaError, match="candidate_tree_id"):
        analyzer.analyze(_request(roots, before, wrong_after))

    other_roots = PropagationAuthorityRoots(
        repository_id=REPO,
        base_forest_id="forest:base",
        base_tree_id=BASE_TREE,
        base_overlay_id="overlay:base",
        candidate_forest_id="forest:candidate",
        candidate_tree_id=CAND_TREE,
        candidate_overlay_id="overlay:candidate-2",
        graph_id="graph:one",
        index_id="index:one",
        model_id="model:one",
        config_id="config:one",
        translator_id="translator:one",
        toolchain_id="toolchain:one",
        policy_id="policy:one",
    )
    with pytest.raises(CrossRootContractDeltaError, match="change set roots"):
        analyzer.analyze(
            ProgramContractDeltaRequest(
                roots=other_roots,
                change_set=_change_set(roots, "pkg/process.py"),
                before=before,
                after=after,
                consumer_domain=DOMAIN_PYTHON_CALLERS,
                subject_symbol_id="symbol:process",
            )
        )


def test_consumer_domain_required(
    roots: PropagationAuthorityRoots,
) -> None:
    before = _expected(tree_id=BASE_TREE)
    after = _expected(tree_id=CAND_TREE)
    with pytest.raises(ProgramContractDeltaError, match="consumer_domain"):
        ProgramContractDeltaRequest(
            roots=roots,
            change_set=_change_set(roots, "pkg/process.py"),
            before=before,
            after=after,
            consumer_domain="",
        )


def test_delta_is_canonical_rpr022_record(
    roots: PropagationAuthorityRoots, analyzer: ProgramContractDeltaAnalyzer
) -> None:
    before = _expected(
        tree_id=BASE_TREE,
        inputs=(_param("a", position=0),),
    )
    after = _expected(
        tree_id=CAND_TREE,
        inputs=(_param("a", position=0), _param("b", position=1)),
    )
    result = analyzer.analyze(_request(roots, before, after))
    delta = result.primary_delta
    assert delta is not None
    assert delta.roots.content_id == roots.content_id
    assert delta.change_set_id == _change_set(roots, "pkg/process.py").content_id
    assert delta.before_contract_ref == before.expected_contract_id
    assert delta.after_contract_ref == after.expected_contract_id
    round_trip = type(delta).from_dict(delta.to_record())
    assert round_trip == delta


def test_path_churn_unknown_path_fails_closed(
    roots: PropagationAuthorityRoots,
) -> None:
    change_set = _change_set(roots, "pkg/process.py")
    with pytest.raises(ProgramContractDeltaError, match="unknown path"):
        normalize_change_partition(
            change_set,
            path_churn=(
                PathChurnClassification(
                    path="pkg/unrelated.py",
                    kind=NonSemanticChurnKind.FORMATTING,
                ),
            ),
        )


def test_rename_pair_records_symbol_rename(
    roots: PropagationAuthorityRoots, analyzer: ProgramContractDeltaAnalyzer
) -> None:
    before = _expected(
        tree_id=BASE_TREE,
        symbol=_symbol(tree_id=BASE_TREE, name="old_name"),
    )
    after = _expected(
        tree_id=CAND_TREE,
        symbol=_symbol(tree_id=CAND_TREE, name="new_name"),
    )
    result = analyzer.analyze(
        _request(
            roots,
            before,
            after,
            rename_pairs=(RenamePair(before_name="old_name", after_name="new_name"),),
        )
    )
    assert any(c.kind is DeltaKind.SYMBOL_RENAME for c in result.all_clauses)


def test_memory_domain_marks_facet_change_breaking(
    roots: PropagationAuthorityRoots, analyzer: ProgramContractDeltaAnalyzer
) -> None:
    before = _expected(tree_id=BASE_TREE)
    after = _expected(tree_id=CAND_TREE)
    result = analyzer.analyze(
        _request(
            roots,
            before,
            after,
            consumer_domain=DOMAIN_MEMORY,
            memory_facet_before_ref="facet:a",
            memory_facet_after_ref="facet:b",
        )
    )
    mem = [c for c in result.all_clauses if c.kind is DeltaKind.MEMORY_FACET_CHANGE]
    assert mem
    assert mem[0].disposition is DeltaDisposition.BREAKING


def test_registration_domain_breaking_on_registration_change(
    roots: PropagationAuthorityRoots, analyzer: ProgramContractDeltaAnalyzer
) -> None:
    before = _expected(tree_id=BASE_TREE)
    after = _expected(tree_id=CAND_TREE)
    result = analyzer.analyze(
        _request(
            roots,
            before,
            after,
            consumer_domain=DOMAIN_REGISTRATION,
            registration_changed=True,
        )
    )
    reg = [c for c in result.all_clauses if c.kind is DeltaKind.SYMBOL_REGISTRATION]
    assert reg
    assert reg[0].disposition is DeltaDisposition.BREAKING


def test_compare_convenience_matches_analyze(
    roots: PropagationAuthorityRoots, analyzer: ProgramContractDeltaAnalyzer
) -> None:
    before = _expected(
        tree_id=BASE_TREE,
        inputs=(_param("a", position=0),),
    )
    after = _expected(
        tree_id=CAND_TREE,
        inputs=(_param("a", position=0), _param("b", position=1)),
    )
    via_compare = analyzer.compare(
        roots=roots,
        change_set=_change_set(roots, "pkg/process.py"),
        before=before,
        after=after,
        consumer_domain=DOMAIN_PYTHON_CALLERS,
        subject_symbol_id="symbol:process",
    )
    via_analyze = analyzer.analyze(_request(roots, before, after))
    assert via_compare.primary_delta is not None
    assert via_analyze.primary_delta is not None
    assert (
        via_compare.primary_delta.content_id == via_analyze.primary_delta.content_id
    )


def test_expectation_sources_required_not_self_authored(
    roots: PropagationAuthorityRoots,
) -> None:
    # ExpectedProgramContract itself rejects observation-only sources so the
    # candidate implementation cannot self-author expected behavior.
    from ipfs_accelerate_py.agent_supervisor.program_contracts import (
        CircularExpectationError,
        ForgedSourceError,
    )

    with pytest.raises((CircularExpectationError, ForgedSourceError)):
        ExpectedProgramContract(
            symbol=_symbol(tree_id=BASE_TREE),
            interface=_interface(),
            policy_revision=POLICY,
            sources=(
                SourceReference(
                    source_kind=ContractSourceKind.IMPLEMENTATION_OBSERVATION,
                    role=ProgramContractRole.EXPECTED,
                    artifact_id="artifact:impl",
                    locator="body",
                    extractor_rule="scan",
                    confidence=ConfidenceClass.LOW,
                    sha256=f"sha256:{SHA}",
                ),
            ),
            inputs=(_param("a", position=0),),
            returns=ReturnSpec(type_shape=_type("int", TypeConstructor.INT)),
        )
