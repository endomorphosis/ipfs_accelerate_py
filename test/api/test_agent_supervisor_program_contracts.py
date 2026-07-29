"""Tests for versioned expected/observed program contract IR (VFS-014)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from ipfs_accelerate_py.agent_supervisor.program_contracts import (
    CONTRACT_VERSION,
    MAX_CLAUSE_BYTES,
    MAX_COLLECTION_ITEMS,
    PROGRAM_CONTRACT_VERSION,
    SOURCE_PRECEDENCE,
    Applicability,
    Assumption,
    AtomicityMode,
    AtomicitySpec,
    AuthorizationMode,
    AuthorizationSpec,
    CapabilityMode,
    CapabilitySpec,
    CircularExpectationError,
    ConfidenceClass,
    ConflictKind,
    ConsistencyMode,
    ConsistencySpec,
    ContractBoundsError,
    ContractConflict,
    ContractConflictError,
    ContractRefinement,
    ContractSourceKind,
    DegradationMode,
    EffectKind,
    EffectPolarity,
    ErrorSpec,
    ExpectedProgramContract,
    FallbackSpec,
    ForgedIdentityError,
    ForgedSourceError,
    IdempotenceMode,
    IdempotenceSpec,
    InterfaceIdentity,
    ObservedProgramContract,
    Optionality,
    OrderingMode,
    OrderingSpec,
    ParameterKind,
    ParameterSpec,
    ProgramContractBundle,
    ProgramContractError,
    ProgramContractRole,
    RefinementRelation,
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
    UnsupportedVersionError,
    all_expectation_source_kinds,
    all_semantic_aspects,
    canonical_program_contract_json_bytes,
    compare_expected_contracts,
    compare_type_shapes,
    detect_source_conflicts,
    may_define_expectation,
    program_contract_content_identity,
    reject_observation_as_expectation,
    select_dominant_sources,
    source_precedence_rank,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
POLICY = "policy:vfs-assurance@1"
REPO = "repository:ipfs_kit_py"
TREE = "tree:abc123"
OBS_ID = "observation:fixture-1"


def symbol(
    *,
    name: str = "read_bytes",
    module: str = "ipfs_kit_py/vfs.py",
    tree_id: str = TREE,
) -> SymbolIdentity:
    return SymbolIdentity(
        repository_id=REPO,
        tree_id=tree_id,
        module_path=module,
        symbol_name=name,
        language="python",
        span_start=10,
        span_end=40,
        blob_cid="baguqeera" + "1" * 50,
    )


def interface(
    *,
    name: str = "vfs.read",
    surface: str = "mcp++",
    method: str = "read",
) -> InterfaceIdentity:
    return InterfaceIdentity(
        interface_name=name,
        surface=surface,
        version="1.0",
        method=method,
        protocol="mcp",
        path_or_uri="mcp://vfs/read",
    )


def source(
    *,
    kind: ContractSourceKind = ContractSourceKind.REVIEWED_INTERFACE,
    role: ProgramContractRole = ProgramContractRole.EXPECTED,
    artifact_id: str = "artifact:idl",
    locator: str = "tools/list#vfs.read",
) -> SourceReference:
    return SourceReference(
        source_kind=kind,
        role=role,
        artifact_id=artifact_id,
        locator=locator,
        extractor_rule="mcp_idl_v1",
        confidence=ConfidenceClass.HIGH,
        sha256=f"sha256:{SHA_A}",
        span_start=1,
        span_end=20,
    )


def observation_source(
    *, artifact_id: str = "artifact:runtime"
) -> SourceReference:
    return SourceReference(
        source_kind=ContractSourceKind.IMPLEMENTATION_OBSERVATION,
        role=ProgramContractRole.OBSERVED,
        artifact_id=artifact_id,
        locator="callsite:vfs.read",
        extractor_rule="static_effect_scan_v1",
        confidence=ConfidenceClass.MEDIUM,
        sha256=f"sha256:{SHA_B}",
    )


def string_type() -> TypeShape:
    return TypeShape(constructor=TypeConstructor.STRING, name="str")


def bytes_type() -> TypeShape:
    return TypeShape(constructor=TypeConstructor.BYTES, name="bytes")


def int_type() -> TypeShape:
    return TypeShape(constructor=TypeConstructor.INT, name="int")


def path_param() -> ParameterSpec:
    return ParameterSpec(
        name="path",
        type_shape=string_type(),
        kind=ParameterKind.POSITIONAL,
        optionality=Optionality.REQUIRED,
        position=0,
        description="Repository-relative path",
    )


def expected_contract(
    *,
    returns: ReturnSpec | None = None,
    inputs: tuple[ParameterSpec, ...] | None = None,
    sources: tuple[SourceReference, ...] | None = None,
    side_effects: tuple[SideEffectSpec, ...] = (),
    resource_bounds: ResourceBounds | None = None,
    conflicts: tuple[ContractConflict, ...] = (),
    summary: str = "VFS read returns bytes for an authorized path.",
    **kwargs,
) -> ExpectedProgramContract:
    return ExpectedProgramContract(
        symbol=kwargs.pop("symbol", symbol()),
        interface=kwargs.pop("interface", interface()),
        policy_revision=kwargs.pop("policy_revision", POLICY),
        sources=sources
        or (
            source(),
            source(
                kind=ContractSourceKind.PUBLIC_SIGNATURE,
                artifact_id="artifact:types",
                locator="def read_bytes",
            ),
        ),
        inputs=inputs if inputs is not None else (path_param(),),
        returns=returns
        if returns is not None
        else ReturnSpec(type_shape=bytes_type(), description="file bytes"),
        errors=(
            ErrorSpec(
                error_name="PathEscapeError",
                code="PATH_ESCAPE",
                retriable=False,
            ),
            ErrorSpec(
                error_name="NotFound",
                code="NOT_FOUND",
                retriable=False,
            ),
        ),
        sync_async=SyncAsyncSpec(mode=SyncMode.SYNC),
        side_effects=side_effects
        or (
            SideEffectSpec(
                effect_kind=EffectKind.FILESYSTEM,
                polarity=EffectPolarity.ALLOWED,
                target="path",
                description="read-only filesystem access",
            ),
            SideEffectSpec(
                effect_kind=EffectKind.WRITE,
                polarity=EffectPolarity.FORBIDDEN,
            ),
        ),
        capabilities=(
            CapabilitySpec(
                capability_name="vfs.read",
                mode=CapabilityMode.REQUIRED,
                version="1",
            ),
        ),
        authorization=AuthorizationSpec(
            mode=AuthorizationMode.PATH_SCOPE,
            scopes=("repo:read",),
            policies=("path-scope-v1",),
        ),
        idempotence=IdempotenceSpec(mode=IdempotenceMode.PURE),
        ordering=OrderingSpec(mode=OrderingMode.UNORDERED),
        atomicity=AtomicitySpec(mode=AtomicityMode.ATOMIC),
        consistency=ConsistencySpec(mode=ConsistencyMode.STRONG),
        resource_bounds=resource_bounds
        or ResourceBounds(
            max_wall_time_ms=5_000,
            max_output_bytes=16 * 1024 * 1024,
            max_memory_bytes=64 * 1024 * 1024,
        ),
        fallback=FallbackSpec(
            mode=DegradationMode.FAIL_CLOSED,
            description="Missing backend fails closed",
        ),
        applicability=Applicability(always=True),
        assumptions=(
            Assumption(
                statement="path is repository-relative and normalized",
                aspect=SemanticAspect.INPUTS,
                confidence=ConfidenceClass.HIGH,
            ),
        ),
        unsupported=(),
        conflicts=conflicts,
        summary=summary,
        **kwargs,
    )


def observed_contract(
    *,
    returns: ReturnSpec | None = None,
    side_effects: tuple[SideEffectSpec, ...] = (),
    summary: str = "Implementation returned bytes for the fixture path.",
) -> ObservedProgramContract:
    return ObservedProgramContract(
        symbol=symbol(),
        interface=interface(),
        policy_revision=POLICY,
        repository_observation_id=OBS_ID,
        sources=(observation_source(),),
        inputs=(path_param(),),
        returns=returns
        if returns is not None
        else ReturnSpec(type_shape=bytes_type()),
        errors=(
            ErrorSpec(error_name="NotFound", code="NOT_FOUND"),
        ),
        sync_async=SyncAsyncSpec(mode=SyncMode.SYNC),
        side_effects=side_effects
        or (
            SideEffectSpec(
                effect_kind=EffectKind.FILESYSTEM,
                polarity=EffectPolarity.OBSERVED,
                target="path",
            ),
        ),
        capabilities=(
            CapabilitySpec(
                capability_name="vfs.read",
                mode=CapabilityMode.OBSERVED,
            ),
        ),
        authorization=AuthorizationSpec(
            mode=AuthorizationMode.PATH_SCOPE,
            scopes=("repo:read",),
        ),
        idempotence=IdempotenceSpec(mode=IdempotenceMode.PURE),
        ordering=OrderingSpec(mode=OrderingMode.UNORDERED),
        atomicity=AtomicitySpec(mode=AtomicityMode.ATOMIC),
        consistency=ConsistencySpec(mode=ConsistencyMode.STRONG),
        resource_bounds=ResourceBounds(
            max_wall_time_ms=120,
            max_output_bytes=1024,
        ),
        fallback=FallbackSpec(mode=DegradationMode.FAIL_CLOSED),
        summary=summary,
        producer_id="static-observer",
        producer_version="1.0.0",
    )


# ---------------------------------------------------------------------------
# Vocabulary and precedence
# ---------------------------------------------------------------------------


def test_contract_version_and_semantic_aspect_coverage() -> None:
    assert PROGRAM_CONTRACT_VERSION == 1
    assert CONTRACT_VERSION == PROGRAM_CONTRACT_VERSION
    aspects = {item.value for item in all_semantic_aspects()}
    assert aspects == {
        "identity",
        "source_precedence",
        "inputs",
        "outputs",
        "sync_async",
        "errors",
        "side_effects",
        "capabilities",
        "authorization",
        "idempotence",
        "ordering",
        "atomicity",
        "consistency",
        "resource_bounds",
        "fallback_degradation",
    }
    assert all_expectation_source_kinds() == SOURCE_PRECEDENCE
    ranks = [source_precedence_rank(kind) for kind in SOURCE_PRECEDENCE]
    assert ranks == sorted(ranks)
    assert ranks == list(range(len(SOURCE_PRECEDENCE)))
    assert may_define_expectation(ContractSourceKind.REVIEWED_INTERFACE)
    assert not may_define_expectation(
        ContractSourceKind.IMPLEMENTATION_OBSERVATION
    )


def test_source_precedence_orders_expectation_authority() -> None:
    sources = (
        source(kind=ContractSourceKind.NORMATIVE_DOCUMENTATION, artifact_id="a"),
        source(kind=ContractSourceKind.REVIEWED_INTERFACE, artifact_id="b"),
        source(kind=ContractSourceKind.CONTRACT_TEST, artifact_id="c"),
    )
    dominant = select_dominant_sources(sources)
    assert len(dominant) == 1
    assert dominant[0].source_kind is ContractSourceKind.REVIEWED_INTERFACE
    assert (
        source_precedence_rank(ContractSourceKind.PUBLIC_SIGNATURE)
        < source_precedence_rank(ContractSourceKind.COMPATIBILITY_MANIFEST)
    )


# ---------------------------------------------------------------------------
# Strict serialization / round-trip / immutability
# ---------------------------------------------------------------------------


def test_round_trip_identity_and_immutability_for_core_records() -> None:
    expected = expected_contract()
    observed = observed_contract()
    # Refinement needs two real contracts in a bundle — build a refined peer.
    refined = expected_contract(
        resource_bounds=ResourceBounds(
            max_wall_time_ms=1_000,
            max_output_bytes=1024,
            max_memory_bytes=8 * 1024 * 1024,
        ),
        summary="Tighter VFS read contract.",
        sources=(
            source(artifact_id="artifact:idl-tight", locator="tight"),
        ),
    )
    refinement = ContractRefinement(
        base_contract_id=expected.expected_contract_id,
        refined_contract_id=refined.expected_contract_id,
        relation=RefinementRelation.STRICT_SUBTYPE,
        aspects=(SemanticAspect.RESOURCE_BOUNDS,),
        summary="tighter resource bounds",
    )
    conflict = ContractConflict(
        kind=ConflictKind.SOURCE_DISAGREEMENT,
        aspect=SemanticAspect.OUTPUTS,
        left_source_id=source().source_id,
        right_source_id=source(
            kind=ContractSourceKind.PUBLIC_SIGNATURE,
            artifact_id="artifact:types",
        ).source_id,
        summary="IDL says bytes; signature annotation is str",
        left_summary="bytes",
        right_summary="str",
    )
    bundle = ProgramContractBundle(
        repository_id=REPO,
        tree_id=TREE,
        policy_revision=POLICY,
        expected=(expected, refined),
        observed=(observed,),
        refinements=(refinement,),
        conflicts=(conflict,),
        summary="VFS read expected/observed pair",
    )

    records = (
        symbol(),
        interface(),
        source(),
        string_type(),
        path_param(),
        ReturnSpec(type_shape=bytes_type()),
        ErrorSpec(error_name="E", code="E"),
        SideEffectSpec(
            effect_kind=EffectKind.READ, polarity=EffectPolarity.ALLOWED
        ),
        CapabilitySpec(capability_name="c", mode=CapabilityMode.REQUIRED),
        AuthorizationSpec(mode=AuthorizationMode.NONE),
        IdempotenceSpec(mode=IdempotenceMode.IDEMPOTENT),
        OrderingSpec(mode=OrderingMode.TOTAL),
        AtomicitySpec(mode=AtomicityMode.ATOMIC),
        ConsistencySpec(mode=ConsistencyMode.EVENTUAL),
        ResourceBounds(max_calls=1),
        FallbackSpec(mode=DegradationMode.RETRY),
        SyncAsyncSpec(mode=SyncMode.ASYNC, awaitable=True),
        Applicability(conditions=("env=prod",), always=False),
        Assumption(
            statement="hermetic fixture",
            aspect=SemanticAspect.SIDE_EFFECTS,
        ),
        UnsupportedSemantics(
            aspect=SemanticAspect.ORDERING,
            reason="dynamic scheduling not modeled",
        ),
        refinement,
        conflict,
        expected,
        observed,
        bundle,
    )
    for record in records:
        restored = type(record).from_json(record.to_json())
        assert restored == record
        assert restored.content_id == record.content_id
        assert restored.canonical_bytes() == record.canonical_bytes()

    with pytest.raises(FrozenInstanceError):
        expected.summary = "forged"  # type: ignore[misc]


def test_serialization_is_deterministic_and_order_independent_where_declared() -> None:
    left_fields = TypeShape(
        constructor=TypeConstructor.OBJECT,
        fields=(
            ("b", int_type()),
            ("a", string_type()),
        ),
    )
    right_fields = TypeShape(
        constructor=TypeConstructor.OBJECT,
        fields=(
            ("a", string_type()),
            ("b", int_type()),
        ),
    )
    assert left_fields.canonical_bytes() == right_fields.canonical_bytes()
    assert left_fields.type_id == right_fields.type_id

    payload_a = {"z": [1, 2], "a": {"y": True, "x": None}}
    payload_b = {"a": {"x": None, "y": True}, "z": [1, 2]}
    assert canonical_program_contract_json_bytes(payload_a) == (
        canonical_program_contract_json_bytes(payload_b)
    )
    assert program_contract_content_identity(payload_a) == (
        program_contract_content_identity(payload_b)
    )


# ---------------------------------------------------------------------------
# Compatibility / version
# ---------------------------------------------------------------------------


def test_unsupported_contract_version_is_rejected() -> None:
    payload = symbol().to_record()
    payload["contract_version"] = 99
    with pytest.raises(UnsupportedVersionError):
        SymbolIdentity.from_dict(payload)

    payload = expected_contract().to_record()
    payload["schema"] = "ipfs_accelerate_py/agent-supervisor/program-contract/expected@9"
    with pytest.raises(ProgramContractError):
        ExpectedProgramContract.from_dict(payload)


def test_unknown_fields_are_rejected() -> None:
    payload = source().to_record()
    payload["body"] = "source text must never be embedded"
    with pytest.raises(ProgramContractError):
        SourceReference.from_dict(payload)


# ---------------------------------------------------------------------------
# Forged identity / forged source
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory,id_name",
    [
        (lambda: symbol(), "symbol_id"),
        (lambda: interface(), "interface_id"),
        (lambda: source(), "source_id"),
        (lambda: string_type(), "type_id"),
        (lambda: path_param(), "parameter_id"),
        (lambda: expected_contract(), "expected_contract_id"),
        (lambda: observed_contract(), "observed_contract_id"),
    ],
)
def test_forged_record_identities_are_rejected(factory, id_name) -> None:
    record = factory()
    payload = record.to_record()
    payload[id_name] = "baguqeeraforgedidentity0000000000000000000000001"
    with pytest.raises(ForgedIdentityError):
        type(record).from_dict(payload)


def test_nested_identity_forgery_is_rejected() -> None:
    expected = expected_contract()
    payload = expected.to_record()
    payload["symbol"]["symbol_name"] = "attacker_renamed"
    with pytest.raises(ForgedIdentityError):
        ExpectedProgramContract.from_dict(payload)


def test_forged_source_roles_and_observation_as_expectation_are_rejected() -> None:
    with pytest.raises(ForgedSourceError):
        SourceReference(
            source_kind=ContractSourceKind.IMPLEMENTATION_OBSERVATION,
            role=ProgramContractRole.EXPECTED,
            artifact_id="artifact:impl",
        )

    # Observation-role sources cannot be attached to expected contracts.
    with pytest.raises(ForgedSourceError):
        ExpectedProgramContract(
            symbol=symbol(),
            interface=interface(),
            policy_revision=POLICY,
            sources=(
                SourceReference(
                    source_kind=ContractSourceKind.IMPLEMENTATION_OBSERVATION,
                    role=ProgramContractRole.OBSERVED,
                    artifact_id="artifact:impl",
                ),
            ),
            summary="self validating",
        )

    observed = observed_contract()
    with pytest.raises(CircularExpectationError):
        reject_observation_as_expectation(observed)
    with pytest.raises(CircularExpectationError):
        observed.as_expectation_source()

    # Observed contracts require an implementation_observation source kind.
    with pytest.raises(ForgedSourceError):
        ObservedProgramContract(
            symbol=symbol(),
            interface=interface(),
            policy_revision=POLICY,
            repository_observation_id=OBS_ID,
            sources=(
                source(role=ProgramContractRole.OBSERVED),
            ),
            summary="missing implementation observation kind",
        )


def test_forged_derived_projections_are_rejected() -> None:
    expected = expected_contract()
    payload = expected.to_record()
    payload["primary_source_kind"] = ContractSourceKind.COMPATIBILITY_MANIFEST.value
    with pytest.raises(ForgedIdentityError):
        ExpectedProgramContract.from_dict(payload)

    payload = expected.to_record()
    payload["has_conflicts"] = True
    with pytest.raises(ForgedIdentityError):
        ExpectedProgramContract.from_dict(payload)


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------


def test_bounds_and_oversized_text_are_rejected() -> None:
    with pytest.raises(ContractBoundsError):
        Assumption(
            statement="x" * (MAX_CLAUSE_BYTES + 1),
            aspect=SemanticAspect.ERRORS,
        )

    with pytest.raises(ContractBoundsError):
        ResourceBounds(max_wall_time_ms=-1)

    with pytest.raises(ContractBoundsError):
        TypeShape(
            constructor=TypeConstructor.OBJECT,
            fields=tuple(
                (f"f{i}", string_type()) for i in range(MAX_COLLECTION_ITEMS + 1)
            ),
        )


def test_span_and_duplicate_constraints() -> None:
    with pytest.raises(ProgramContractError):
        SymbolIdentity(
            repository_id=REPO,
            tree_id=TREE,
            module_path="m.py",
            symbol_name="s",
            span_start=10,
            span_end=5,
        )
    with pytest.raises(ProgramContractError):
        TypeShape(
            constructor=TypeConstructor.OBJECT,
            fields=(
                ("a", string_type()),
                ("a", int_type()),
            ),
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_values_are_rejected(value: float) -> None:
    with pytest.raises(ValueError):
        canonical_program_contract_json_bytes({"bound": value})


# ---------------------------------------------------------------------------
# Conflicts
# ---------------------------------------------------------------------------


def test_conflicts_are_represented_and_cannot_be_silently_resolved() -> None:
    left = source(artifact_id="artifact:idl-a", locator="a")
    right = source(
        kind=ContractSourceKind.REVIEWED_INTERFACE,
        artifact_id="artifact:idl-b",
        locator="b",
    )
    conflicts = detect_source_conflicts(
        (left, right),
        aspect=SemanticAspect.OUTPUTS,
        left_summary="bytes",
        right_summary="str",
        disagree=True,
    )
    assert len(conflicts) == 1
    assert conflicts[0].kind is ConflictKind.PRECEDENCE_COLLISION
    assert conflicts[0].resolved is False

    with pytest.raises(ContractConflictError):
        ContractConflict(
            kind=ConflictKind.TYPE_MISMATCH,
            aspect=SemanticAspect.OUTPUTS,
            left_source_id=left.source_id,
            right_source_id=right.source_id,
            summary="resolved away",
            resolved=True,
        )

    expected = expected_contract(conflicts=conflicts)
    assert expected.has_conflicts
    bundle = ProgramContractBundle(
        repository_id=REPO,
        tree_id=TREE,
        policy_revision=POLICY,
        expected=(expected,),
        conflicts=conflicts,
    )
    assert bundle.has_conflicts


def test_equal_precedence_no_conflict_when_not_disagreeing() -> None:
    left = source(artifact_id="a")
    right = source(artifact_id="b", locator="other")
    assert (
        detect_source_conflicts(
            (left, right),
            aspect=SemanticAspect.INPUTS,
            left_summary="ok",
            right_summary="ok",
            disagree=False,
        )
        == ()
    )


# ---------------------------------------------------------------------------
# Subtyping / refinements
# ---------------------------------------------------------------------------


def test_type_shape_subtyping_lattice() -> None:
    any_t = TypeShape(constructor=TypeConstructor.ANY)
    never = TypeShape(constructor=TypeConstructor.NEVER)
    string = string_type()
    bytes_t = bytes_type()
    enum_ab = TypeShape(
        constructor=TypeConstructor.ENUM,
        enum_values=("a", "b"),
    )
    enum_a = TypeShape(
        constructor=TypeConstructor.ENUM,
        enum_values=("a",),
    )
    arr_str = TypeShape(constructor=TypeConstructor.ARRAY, item=string)
    arr_any = TypeShape(constructor=TypeConstructor.ARRAY, item=any_t)
    obj_wide = TypeShape(
        constructor=TypeConstructor.OBJECT,
        fields=(("a", string), ("b", int_type())),
    )
    obj_narrow = TypeShape(
        constructor=TypeConstructor.OBJECT,
        fields=(("a", string),),
    )

    assert never.is_subtype_of(string)
    assert string.is_subtype_of(any_t)
    assert not string.is_subtype_of(bytes_t)
    assert enum_a.is_subtype_of(enum_ab)
    assert not enum_ab.is_subtype_of(enum_a)
    assert arr_str.is_subtype_of(arr_any)
    # Width: more fields is subtype of fewer fields (structural)
    assert obj_wide.is_subtype_of(obj_narrow)
    assert not obj_narrow.is_subtype_of(obj_wide)

    assert compare_type_shapes(string, string) is RefinementRelation.EQUIVALENT
    assert compare_type_shapes(enum_a, enum_ab) is RefinementRelation.STRICT_SUBTYPE
    assert compare_type_shapes(enum_ab, enum_a) is RefinementRelation.STRICT_SUPERTYPE
    assert compare_type_shapes(string, bytes_t) is RefinementRelation.INCOMPATIBLE


def test_resource_bounds_and_idempotence_refinement() -> None:
    loose = ResourceBounds(max_wall_time_ms=10_000, max_output_bytes=1_000_000)
    tight = ResourceBounds(max_wall_time_ms=1_000, max_output_bytes=1_000)
    assert tight.is_refinement_of(loose)
    assert not loose.is_refinement_of(tight)

    pure = IdempotenceSpec(mode=IdempotenceMode.PURE)
    idem = IdempotenceSpec(mode=IdempotenceMode.IDEMPOTENT)
    non = IdempotenceSpec(mode=IdempotenceMode.NON_IDEMPOTENT)
    assert pure.is_refinement_of(idem)
    assert not non.is_refinement_of(pure)

    atomic = AtomicitySpec(mode=AtomicityMode.ATOMIC)
    best = AtomicitySpec(mode=AtomicityMode.BEST_EFFORT)
    assert atomic.is_refinement_of(best)


def test_expected_contract_refinement_and_subtyping_comparison() -> None:
    base = expected_contract(
        resource_bounds=ResourceBounds(
            max_wall_time_ms=10_000,
            max_output_bytes=16 * 1024 * 1024,
        ),
        summary="base",
    )
    refined = expected_contract(
        resource_bounds=ResourceBounds(
            max_wall_time_ms=1_000,
            max_output_bytes=1024,
        ),
        summary="refined tighter bounds",
        sources=(source(artifact_id="artifact:refined"),),
    )
    assert refined.is_refinement_of(base)
    assert compare_expected_contracts(refined, base) is (
        RefinementRelation.STRICT_SUBTYPE
    )

    # Incompatible returns break refinement.
    bad = expected_contract(
        returns=ReturnSpec(type_shape=string_type()),
        summary="wrong return type",
        sources=(source(artifact_id="artifact:bad"),),
    )
    assert not bad.is_refinement_of(base)
    assert compare_expected_contracts(bad, base) is RefinementRelation.INCOMPATIBLE


def test_effect_allowance_and_sync_compatibility() -> None:
    allowed = SideEffectSpec(
        effect_kind=EffectKind.FILESYSTEM,
        polarity=EffectPolarity.ALLOWED,
        target="path",
    )
    observed = SideEffectSpec(
        effect_kind=EffectKind.FILESYSTEM,
        polarity=EffectPolarity.OBSERVED,
        target="path",
    )
    write = SideEffectSpec(
        effect_kind=EffectKind.WRITE,
        polarity=EffectPolarity.OBSERVED,
    )
    forbidden_write = SideEffectSpec(
        effect_kind=EffectKind.WRITE,
        polarity=EffectPolarity.FORBIDDEN,
    )
    assert observed.is_allowed_by(allowed)
    assert not write.is_allowed_by(forbidden_write)

    sync = SyncAsyncSpec(mode=SyncMode.SYNC)
    dual = SyncAsyncSpec(mode=SyncMode.DUAL)
    async_mode = SyncAsyncSpec(mode=SyncMode.ASYNC, awaitable=True)
    assert sync.is_compatible_with(dual)
    assert not sync.is_compatible_with(async_mode)


# ---------------------------------------------------------------------------
# Expectations vs observations separation
# ---------------------------------------------------------------------------


def test_expectations_and_observations_are_separate_roles() -> None:
    expected = expected_contract()
    observed = observed_contract()
    assert expected.role is ProgramContractRole.EXPECTED
    assert observed.role is ProgramContractRole.OBSERVED
    assert observed.binds_same_subject(expected)
    assert expected.primary_source_kind is ContractSourceKind.REVIEWED_INTERFACE
    assert "implementation_observation" not in {
        s.source_kind.value for s in expected.sources
    }
    assert all(
        s.source_kind is ContractSourceKind.IMPLEMENTATION_OBSERVATION
        for s in observed.sources
    )

    # Observed payload cannot be decoded as expected even if schema is rewritten.
    payload = observed.to_dict()
    payload["schema"] = ExpectedProgramContract.SCHEMA
    payload["role"] = ProgramContractRole.EXPECTED.value
    with pytest.raises((ForgedSourceError, CircularExpectationError, ProgramContractError)):
        ExpectedProgramContract.from_dict(payload)


def test_bundle_keeps_expected_and_observed_separated() -> None:
    expected = expected_contract()
    observed = observed_contract()
    bundle = ProgramContractBundle(
        repository_id=REPO,
        tree_id=TREE,
        policy_revision=POLICY,
        expected=(expected,),
        observed=(observed,),
    )
    assert bundle.expected_for(symbol(), interface()) == (expected,)
    assert bundle.observed_for(symbol(), interface()) == (observed,)
    assert not bundle.has_conflicts

    with pytest.raises(ProgramContractError):
        ProgramContractBundle(
            repository_id=REPO,
            tree_id="tree:other",
            policy_revision=POLICY,
            expected=(expected,),
        )


def test_unsupported_semantics_and_assumptions_and_applicability() -> None:
    unsupported = UnsupportedSemantics(
        aspect=SemanticAspect.ORDERING,
        reason="concurrent reordering not modeled",
        residual="may reorder concurrent writers",
    )
    expected = expected_contract(
        summary="with unsupported ordering",
        sources=(source(artifact_id="artifact:partial"),),
    )
    # Rebuild with unsupported via manual construction
    expected = ExpectedProgramContract(
        symbol=symbol(),
        interface=interface(),
        policy_revision=POLICY,
        sources=(source(),),
        inputs=(path_param(),),
        returns=ReturnSpec(type_shape=bytes_type()),
        unsupported=(unsupported,),
        assumptions=(
            Assumption(
                statement="single-threaded caller",
                aspect=SemanticAspect.ORDERING,
            ),
        ),
        applicability=Applicability(
            surfaces=("mcp++", "python"),
            environments=("test",),
            always=False,
        ),
        summary="partial semantics",
    )
    assert (
        expected.aspect_support(SemanticAspect.ORDERING)
        is SupportStatus.UNSUPPORTED
    )
    assert expected.applicability is not None
    assert expected.applicability.always is False
    assert expected.assumptions[0].aspect is SemanticAspect.ORDERING


def test_array_type_requires_item_when_supported() -> None:
    with pytest.raises(ProgramContractError):
        TypeShape(constructor=TypeConstructor.ARRAY)
    ok = TypeShape(
        constructor=TypeConstructor.ARRAY,
        support=SupportStatus.UNSUPPORTED,
    )
    assert ok.support is SupportStatus.UNSUPPORTED


def test_refinement_requires_bundle_membership() -> None:
    expected = expected_contract()
    refined = expected_contract(
        summary="peer",
        sources=(source(artifact_id="artifact:peer"),),
    )
    refinement = ContractRefinement(
        base_contract_id=expected.expected_contract_id,
        refined_contract_id=refined.expected_contract_id,
        relation=RefinementRelation.COMPATIBLE,
        aspects=(SemanticAspect.INPUTS,),
    )
    with pytest.raises(ProgramContractError):
        ProgramContractBundle(
            repository_id=REPO,
            tree_id=TREE,
            policy_revision=POLICY,
            expected=(expected,),  # refined missing
            refinements=(refinement,),
        )
    bundle = ProgramContractBundle(
        repository_id=REPO,
        tree_id=TREE,
        policy_revision=POLICY,
        expected=(expected, refined),
        refinements=(refinement,),
    )
    assert len(bundle.refinements) == 1


def test_input_contravariance_helper() -> None:
    required = ParameterSpec(
        name="path",
        type_shape=TypeShape(
            constructor=TypeConstructor.ENUM,
            enum_values=("/a", "/b"),
        ),
        kind=ParameterKind.POSITIONAL,
    )
    any_acceptor = ParameterSpec(
        name="path",
        type_shape=TypeShape(constructor=TypeConstructor.ANY),
        kind=ParameterKind.POSITIONAL,
    )
    # Contravariance: acceptor must accept all required values.
    assert any_acceptor.is_input_compatible_with(required)
    # Covariance: produced enum is a subtype of ANY (top type).
    assert required.type_shape.is_subtype_of(
        TypeShape(constructor=TypeConstructor.ANY)
    )
    assert required.is_output_compatible_with(
        ParameterSpec(
            name="path",
            type_shape=TypeShape(constructor=TypeConstructor.ANY),
        )
    )
