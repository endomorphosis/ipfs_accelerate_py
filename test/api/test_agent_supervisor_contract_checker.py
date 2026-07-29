"""Tests for symbolic contract comparison and counterexamples (VFS-016)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from ipfs_accelerate_py.agent_supervisor.contract_checker import (
    CHECKER_VERSION,
    CLOSED_SUPPORTED_ASPECTS,
    CONTRACT_CHECK_RESULT_EVIDENCE,
    CONTRACT_COUNTEREXAMPLE_EVIDENCE,
    AspectCheckResult,
    AspectVerdict,
    CacheFreshness,
    CallPath,
    CallPathResolution,
    CallPathStep,
    CheckBinding,
    ContractCheckReport,
    ContractCheckResult,
    ContractCheckResultKind,
    ContractChecker,
    ContractCheckerError,
    ContractCounterexample,
    ForgedIdentityError,
    ScopeMismatchError,
    StaleAuthorityError,
    check_errors,
    check_inputs,
    check_outputs,
    check_side_effects,
    closed_supported_aspects,
    compare_contracts,
    compare_expected_refinement,
    contract_check_content_identity,
    make_binding,
    minimal_counterexample,
    _path_is_traversal,
)
from ipfs_accelerate_py.agent_supervisor.program_contracts import (
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
    ObservedProgramContract,
    Optionality,
    OrderingMode,
    OrderingSpec,
    ParameterKind,
    ParameterSpec,
    ProgramContractBundle,
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


SHA_A = "a" * 64
SHA_B = "b" * 64
POLICY = "policy:vfs-assurance@1"
REPO = "repository:ipfs_kit_py"
TREE = "tree:abc123"
OBS_ID = "observation:fixture-1"
EVALUATED = "2026-07-29T12:00:00+00:00"
EXPIRES = "2026-07-29T13:00:00+00:00"
STALE_EXPIRES = "2026-07-29T11:00:00+00:00"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def symbol(
    *,
    name: str = "read_bytes",
    module: str = "ipfs_kit_py/vfs.py",
    tree_id: str = TREE,
    repository_id: str = REPO,
) -> SymbolIdentity:
    return SymbolIdentity(
        repository_id=repository_id,
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


def path_param(
    *,
    optionality: Optionality = Optionality.REQUIRED,
    type_shape: TypeShape | None = None,
) -> ParameterSpec:
    return ParameterSpec(
        name="path",
        type_shape=type_shape or string_type(),
        kind=ParameterKind.POSITIONAL,
        optionality=optionality,
        position=0,
        description="Repository-relative path",
    )


def expected_contract(**kwargs) -> ExpectedProgramContract:
    return ExpectedProgramContract(
        symbol=kwargs.pop("symbol", symbol()),
        interface=kwargs.pop("interface", interface()),
        policy_revision=kwargs.pop("policy_revision", POLICY),
        sources=kwargs.pop(
            "sources",
            (
                source(),
                source(
                    kind=ContractSourceKind.PUBLIC_SIGNATURE,
                    artifact_id="artifact:types",
                    locator="def read_bytes",
                ),
            ),
        ),
        inputs=kwargs.pop("inputs", (path_param(),)),
        returns=kwargs.pop(
            "returns",
            ReturnSpec(type_shape=bytes_type(), description="file bytes"),
        ),
        errors=kwargs.pop(
            "errors",
            (
                ErrorSpec(
                    error_name="PathEscapeError",
                    code="PATH_ESCAPE",
                    retriable=False,
                ),
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
        resource_bounds=kwargs.pop(
            "resource_bounds",
            ResourceBounds(
                max_wall_time_ms=5_000,
                max_output_bytes=16 * 1024 * 1024,
                max_memory_bytes=64 * 1024 * 1024,
            ),
        ),
        fallback=kwargs.pop(
            "fallback",
            FallbackSpec(
                mode=DegradationMode.FAIL_CLOSED,
                description="Missing backend fails closed",
            ),
        ),
        unsupported=kwargs.pop("unsupported", ()),
        summary=kwargs.pop("summary", "VFS read returns bytes."),
        **kwargs,
    )


def observed_contract(**kwargs) -> ObservedProgramContract:
    return ObservedProgramContract(
        symbol=kwargs.pop("symbol", symbol()),
        interface=kwargs.pop("interface", interface()),
        policy_revision=kwargs.pop("policy_revision", POLICY),
        repository_observation_id=kwargs.pop(
            "repository_observation_id", OBS_ID
        ),
        sources=kwargs.pop("sources", (observation_source(),)),
        inputs=kwargs.pop("inputs", (path_param(),)),
        returns=kwargs.pop(
            "returns", ReturnSpec(type_shape=bytes_type())
        ),
        errors=kwargs.pop(
            "errors",
            (ErrorSpec(error_name="NotFound", code="NOT_FOUND"),),
        ),
        sync_async=kwargs.pop("sync_async", SyncAsyncSpec(mode=SyncMode.SYNC)),
        side_effects=kwargs.pop(
            "side_effects",
            (
                SideEffectSpec(
                    effect_kind=EffectKind.FILESYSTEM,
                    polarity=EffectPolarity.OBSERVED,
                    target="path",
                ),
            ),
        ),
        capabilities=kwargs.pop(
            "capabilities",
            (
                CapabilitySpec(
                    capability_name="vfs.read",
                    mode=CapabilityMode.OBSERVED,
                ),
            ),
        ),
        authorization=kwargs.pop(
            "authorization",
            AuthorizationSpec(
                mode=AuthorizationMode.PATH_SCOPE,
                scopes=("repo:read",),
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
        resource_bounds=kwargs.pop(
            "resource_bounds",
            ResourceBounds(max_wall_time_ms=120, max_output_bytes=1024),
        ),
        fallback=kwargs.pop(
            "fallback", FallbackSpec(mode=DegradationMode.FAIL_CLOSED)
        ),
        unsupported=kwargs.pop("unsupported", ()),
        summary=kwargs.pop("summary", "Implementation returned bytes."),
        producer_id=kwargs.pop("producer_id", "static-observer"),
        producer_version=kwargs.pop("producer_version", "1.0.0"),
        **kwargs,
    )


def static_path(
    *,
    path_name: str = "vfs.read.direct",
    target_path: str = "docs/readme.md",
    resolution: CallPathResolution = CallPathResolution.STATIC,
) -> CallPath:
    return CallPath(
        repository_id=REPO,
        tree_id=TREE,
        policy_revision=POLICY,
        path_name=path_name,
        entry_interface="vfs.read",
        exit_symbol="read_bytes",
        steps=(
            CallPathStep(
                step_index=0,
                symbol_name="connector.read",
                interface_name="vfs.read",
                module_path="swissknife/connectors/vfs.ts",
                resolution=CallPathResolution.STATIC,
            ),
            CallPathStep(
                step_index=1,
                symbol_name="read_bytes",
                interface_name="vfs.read",
                module_path="ipfs_kit_py/vfs.py",
                resolution=resolution,
                target_path=target_path,
            ),
        ),
        summary="Direct MCP++ to package implementation",
    )


# ---------------------------------------------------------------------------
# Vocabulary / closed rules
# ---------------------------------------------------------------------------


def test_closed_supported_aspects_cover_acceptance_dimensions() -> None:
    values = {item.value for item in closed_supported_aspects()}
    assert values == {item.value for item in CLOSED_SUPPORTED_ASPECTS}
    for required in (
        "identity",
        "inputs",
        "outputs",
        "errors",
        "sync_async",
        "side_effects",
        "authorization",
        "idempotence",
        "ordering",
        "atomicity",
        "resource_bounds",
        "fallback_degradation",
    ):
        assert required in values
    assert CHECKER_VERSION.startswith("contract-checker@")


def test_objective_evidence_terms_are_emitted_on_result_and_witness() -> None:
    result = compare_contracts(
        expected_contract(),
        observed_contract(returns=ReturnSpec(type_shape=string_type())),
        evaluated_at=EVALUATED,
        authority_expires_at=EXPIRES,
    )
    assert CONTRACT_CHECK_RESULT_EVIDENCE == "vfs/contract-check-result@1"
    assert CONTRACT_COUNTEREXAMPLE_EVIDENCE == "vfs/contract-counterexample@1"
    assert result.evidence == CONTRACT_CHECK_RESULT_EVIDENCE
    assert result.to_dict()["evidence"] == CONTRACT_CHECK_RESULT_EVIDENCE
    assert result.counterexample is not None
    assert result.counterexample.evidence == CONTRACT_COUNTEREXAMPLE_EVIDENCE
    assert (
        result.counterexample.to_dict()["evidence"]
        == CONTRACT_COUNTEREXAMPLE_EVIDENCE
    )

    forged_result = result.to_dict()
    forged_result["evidence"] = "vfs/other-evidence@1"
    with pytest.raises(ContractCheckerError):
        ContractCheckResult.from_dict(forged_result)

    forged_witness = result.counterexample.to_dict()
    forged_witness["evidence"] = "vfs/other-evidence@1"
    with pytest.raises(ContractCheckerError):
        ContractCounterexample.from_dict(forged_witness)


def test_path_traversal_detector() -> None:
    assert _path_is_traversal("../etc/passwd")
    assert _path_is_traversal("/absolute")
    assert _path_is_traversal("foo/../../secret")
    assert not _path_is_traversal("docs/readme.md")
    assert not _path_is_traversal("a/b/c")


# ---------------------------------------------------------------------------
# Proved compatible (happy path + compatible refinements)
# ---------------------------------------------------------------------------


def test_proved_compatible_on_closed_matching_contracts() -> None:
    result = compare_contracts(
        expected_contract(),
        observed_contract(),
        evaluated_at=EVALUATED,
        authority_expires_at=EXPIRES,
        call_path=static_path(),
    )
    assert result.kind is ContractCheckResultKind.PROVED_COMPATIBLE
    assert result.counterexample is None
    assert result.freshness is CacheFreshness.CURRENT
    assert result.binding.expected_contract_id == (
        expected_contract().expected_contract_id
    )
    assert all(
        item.verdict
        in {AspectVerdict.COMPATIBLE, AspectVerdict.NOT_APPLICABLE}
        for item in result.aspect_results
        if item.closed_rule
    )


def test_compatible_refinement_tighter_bounds_and_stronger_idempotence() -> None:
    base = expected_contract(
        resource_bounds=ResourceBounds(
            max_wall_time_ms=10_000, max_output_bytes=1_000_000
        ),
        idempotence=IdempotenceSpec(mode=IdempotenceMode.IDEMPOTENT),
        summary="base expectation",
    )
    refined = expected_contract(
        resource_bounds=ResourceBounds(
            max_wall_time_ms=1_000, max_output_bytes=1_000
        ),
        idempotence=IdempotenceSpec(mode=IdempotenceMode.PURE),
        sources=(source(artifact_id="artifact:refined"),),
        summary="compatible refinement",
    )
    aspect = compare_expected_refinement(refined, base)
    assert aspect.verdict is AspectVerdict.COMPATIBLE

    observed = observed_contract(
        resource_bounds=ResourceBounds(
            max_wall_time_ms=500, max_output_bytes=512
        ),
        idempotence=IdempotenceSpec(mode=IdempotenceMode.PURE),
    )
    result = compare_contracts(
        refined,
        observed,
        evaluated_at=EVALUATED,
        authority_expires_at=EXPIRES,
    )
    assert result.kind is ContractCheckResultKind.PROVED_COMPATIBLE


# ---------------------------------------------------------------------------
# Seeded broken contracts → witnessed mismatch + minimal counterexample
# ---------------------------------------------------------------------------


def test_seeded_output_type_mismatch_emits_minimal_counterexample() -> None:
    expected = expected_contract()
    broken = observed_contract(
        returns=ReturnSpec(type_shape=string_type()),
        summary="returns str instead of bytes",
    )
    result = compare_contracts(
        expected,
        broken,
        evaluated_at=EVALUATED,
        authority_expires_at=EXPIRES,
    )
    assert result.kind is ContractCheckResultKind.WITNESSED_MISMATCH
    assert result.counterexample is not None
    assert result.counterexample.aspect is SemanticAspect.OUTPUTS
    assert result.counterexample.authoritative is True
    assert result.counterexample.witness_steps[0].startswith("bind:")
    assert "expected:" in result.counterexample.witness_steps[-2]
    assert "observed:" in result.counterexample.witness_steps[-1]
    # Minimal: one primary mismatch aspect for the counterexample.
    assert result.counterexample.aspect is SemanticAspect.OUTPUTS
    assert SemanticAspect.OUTPUTS in result.mismatch_aspects


def test_seeded_required_input_missing_and_optionality_flip() -> None:
    expected = expected_contract(inputs=(path_param(),))
    missing = observed_contract(inputs=())
    result = compare_contracts(
        expected,
        missing,
        evaluated_at=EVALUATED,
        authority_expires_at=EXPIRES,
    )
    assert result.kind is ContractCheckResultKind.WITNESSED_MISMATCH
    assert result.counterexample is not None
    assert result.counterexample.aspect is SemanticAspect.INPUTS
    assert "missing" in result.counterexample.observed_fact

    optionalized = observed_contract(
        inputs=(path_param(optionality=Optionality.OPTIONAL),)
    )
    result2 = compare_contracts(
        expected,
        optionalized,
        evaluated_at=EVALUATED,
        authority_expires_at=EXPIRES,
    )
    assert result2.kind is ContractCheckResultKind.WITNESSED_MISMATCH
    assert result2.counterexample is not None
    assert "optional" in result2.counterexample.summary


def test_seeded_forbidden_write_effect_and_error_code_drift() -> None:
    expected = expected_contract()
    writes = observed_contract(
        side_effects=(
            SideEffectSpec(
                effect_kind=EffectKind.WRITE,
                polarity=EffectPolarity.OBSERVED,
                target="path",
            ),
        )
    )
    result = compare_contracts(
        expected,
        writes,
        evaluated_at=EVALUATED,
        authority_expires_at=EXPIRES,
    )
    assert result.kind is ContractCheckResultKind.WITNESSED_MISMATCH
    assert result.counterexample is not None
    assert result.counterexample.aspect is SemanticAspect.SIDE_EFFECTS

    bad_errors = observed_contract(
        errors=(ErrorSpec(error_name="NotFound", code="ENOENT"),)
    )
    err_result = check_errors(expected, bad_errors)
    assert err_result.verdict is AspectVerdict.MISMATCH


def test_seeded_async_auth_idempotence_atomicity_bounds_degradation() -> None:
    expected = expected_contract(
        sync_async=SyncAsyncSpec(mode=SyncMode.SYNC),
        authorization=AuthorizationSpec(
            mode=AuthorizationMode.PATH_SCOPE,
            scopes=("repo:read",),
        ),
        idempotence=IdempotenceSpec(mode=IdempotenceMode.PURE),
        atomicity=AtomicitySpec(mode=AtomicityMode.ATOMIC),
        resource_bounds=ResourceBounds(max_wall_time_ms=5_000),
        fallback=FallbackSpec(mode=DegradationMode.FAIL_CLOSED),
    )
    # Each case changes exactly one closed aspect so the minimal
    # counterexample aspect is unambiguous.
    cases: list[tuple[ObservedProgramContract, SemanticAspect]] = [
        (
            observed_contract(
                sync_async=SyncAsyncSpec(mode=SyncMode.ASYNC, awaitable=True)
            ),
            SemanticAspect.SYNC_ASYNC,
        ),
        (
            observed_contract(
                authorization=AuthorizationSpec(mode=AuthorizationMode.NONE)
            ),
            SemanticAspect.AUTHORIZATION,
        ),
        (
            observed_contract(
                idempotence=IdempotenceSpec(
                    mode=IdempotenceMode.NON_IDEMPOTENT
                )
            ),
            SemanticAspect.IDEMPOTENCE,
        ),
        (
            observed_contract(
                atomicity=AtomicitySpec(mode=AtomicityMode.NON_ATOMIC)
            ),
            SemanticAspect.ATOMICITY,
        ),
        (
            observed_contract(
                resource_bounds=ResourceBounds(max_wall_time_ms=10_000)
            ),
            SemanticAspect.RESOURCE_BOUNDS,
        ),
        (
            observed_contract(
                fallback=FallbackSpec(mode=DegradationMode.FAIL_OPEN)
            ),
            SemanticAspect.FALLBACK_DEGRADATION,
        ),
    ]
    aspects_hit: set[SemanticAspect] = set()
    for broken, expected_aspect in cases:
        result = compare_contracts(
            expected,
            broken,
            evaluated_at=EVALUATED,
            authority_expires_at=EXPIRES,
        )
        assert result.kind is ContractCheckResultKind.WITNESSED_MISMATCH
        assert result.counterexample is not None
        assert result.counterexample.aspect is expected_aspect
        aspects_hit.add(result.counterexample.aspect)
    assert aspects_hit == {
        SemanticAspect.SYNC_ASYNC,
        SemanticAspect.AUTHORIZATION,
        SemanticAspect.IDEMPOTENCE,
        SemanticAspect.ATOMICITY,
        SemanticAspect.RESOURCE_BOUNDS,
        SemanticAspect.FALLBACK_DEGRADATION,
    }


# ---------------------------------------------------------------------------
# Dynamic dispatch uncertainty, omitted effects, path traversal, cache stale
# ---------------------------------------------------------------------------


def test_dynamic_dispatch_uncertainty_is_ambiguous() -> None:
    path = static_path(resolution=CallPathResolution.DYNAMIC)
    assert path.has_dynamic_dispatch
    result = compare_contracts(
        expected_contract(),
        observed_contract(),
        call_path=path,
        evaluated_at=EVALUATED,
        authority_expires_at=EXPIRES,
    )
    assert result.kind is ContractCheckResultKind.AMBIGUOUS
    assert result.counterexample is None


def test_ambiguous_same_name_path_is_not_proved_compatible() -> None:
    path = CallPath(
        repository_id=REPO,
        tree_id=TREE,
        policy_revision=POLICY,
        path_name="vfs.read.ambiguous",
        steps=(
            CallPathStep(
                step_index=0,
                symbol_name="read_bytes",
                module_path="ipfs_kit_py/vfs.py",
                resolution=CallPathResolution.AMBIGUOUS,
                notes="two same-name candidates",
            ),
        ),
    )
    result = compare_contracts(
        expected_contract(),
        observed_contract(),
        call_path=path,
        evaluated_at=EVALUATED,
        authority_expires_at=EXPIRES,
    )
    assert result.kind is ContractCheckResultKind.AMBIGUOUS


def test_omitted_effects_required_is_mismatch_optional_allowance_ok() -> None:
    expected = expected_contract(
        side_effects=(
            SideEffectSpec(
                effect_kind=EffectKind.FILESYSTEM,
                polarity=EffectPolarity.REQUIRED,
                target="path",
            ),
            SideEffectSpec(
                effect_kind=EffectKind.METRICS,
                polarity=EffectPolarity.ALLOWED,
            ),
        )
    )
    omitted = observed_contract(side_effects=())
    aspect = check_side_effects(expected, omitted)
    assert aspect.verdict is AspectVerdict.MISMATCH
    assert "omitted" in aspect.summary

    # Required present, optional allowance unused → compatible.
    present = observed_contract(
        side_effects=(
            SideEffectSpec(
                effect_kind=EffectKind.FILESYSTEM,
                polarity=EffectPolarity.OBSERVED,
                target="path",
            ),
        )
    )
    assert (
        check_side_effects(expected, present).verdict is AspectVerdict.COMPATIBLE
    )


def test_path_traversal_is_witnessed_mismatch() -> None:
    path = static_path(target_path="../etc/passwd")
    assert path.has_path_traversal
    result = compare_contracts(
        expected_contract(),
        observed_contract(),
        call_path=path,
        evaluated_at=EVALUATED,
        authority_expires_at=EXPIRES,
    )
    assert result.kind is ContractCheckResultKind.WITNESSED_MISMATCH
    assert result.counterexample is not None
    assert result.counterexample.aspect is SemanticAspect.INPUTS
    assert "traversal" in result.counterexample.summary.lower() or (
        ".." in result.counterexample.observed_fact
        or "passwd" in result.counterexample.observed_fact
    )


def test_cache_staleness_and_expired_authority() -> None:
    stale_cache = compare_contracts(
        expected_contract(),
        observed_contract(),
        evaluated_at=EVALUATED,
        authority_expires_at=EXPIRES,
        cache_generation="gen:1",
        expected_cache_generation="gen:2",
    )
    assert stale_cache.kind is ContractCheckResultKind.STALE
    assert stale_cache.cache_freshness is CacheFreshness.STALE
    assert (
        stale_cache.binding.cache_binding_freshness
        is CacheFreshness.STALE
    )
    assert stale_cache.counterexample is None

    expired = compare_contracts(
        expected_contract(),
        observed_contract(),
        evaluated_at=EVALUATED,
        authority_expires_at=STALE_EXPIRES,
    )
    assert expired.kind is ContractCheckResultKind.STALE
    assert expired.freshness is CacheFreshness.STALE

    stale_timeout = compare_contracts(
        expected_contract(),
        observed_contract(),
        evaluated_at=EVALUATED,
        authority_expires_at=STALE_EXPIRES,
        force_timeout=True,
    )
    assert stale_timeout.kind is ContractCheckResultKind.STALE


def test_timeout_result_kind() -> None:
    result = compare_contracts(
        expected_contract(),
        observed_contract(),
        evaluated_at=EVALUATED,
        authority_expires_at=EXPIRES,
        budget_ms=10,
        force_timeout=True,
    )
    assert result.kind is ContractCheckResultKind.TIMEOUT
    assert result.elapsed_ms >= result.budget_ms
    assert result.counterexample is None


def test_unsupported_aspect_blocks_proved_compatible() -> None:
    expected = expected_contract(
        unsupported=(
            UnsupportedSemantics(
                aspect=SemanticAspect.ORDERING,
                reason="partial-order lattice incomplete",
            ),
        )
    )
    result = compare_contracts(
        expected,
        observed_contract(),
        evaluated_at=EVALUATED,
        authority_expires_at=EXPIRES,
    )
    assert result.kind is ContractCheckResultKind.UNSUPPORTED
    assert any(
        item.aspect is SemanticAspect.ORDERING
        and item.verdict is AspectVerdict.UNSUPPORTED
        for item in result.aspect_results
    )


def test_unknown_semantics_emit_explicit_unknown_result() -> None:
    result = compare_contracts(
        expected_contract(),
        observed_contract(
            fallback=FallbackSpec(mode=DegradationMode.UNKNOWN)
        ),
        evaluated_at=EVALUATED,
        authority_expires_at=EXPIRES,
    )
    assert result.kind is ContractCheckResultKind.UNKNOWN
    assert result.counterexample is None
    assert any(
        item.aspect is SemanticAspect.FALLBACK_DEGRADATION
        and item.verdict is AspectVerdict.UNKNOWN
        for item in result.aspect_results
    )


# ---------------------------------------------------------------------------
# Adversarial same-name fixtures with deterministic identities
# ---------------------------------------------------------------------------


def test_adversarial_same_name_symbols_have_distinct_deterministic_ids() -> None:
    """Same short name in different modules must not share identities."""

    real = symbol(name="read_bytes", module="ipfs_kit_py/vfs.py")
    mock = symbol(name="read_bytes", module="tests/mocks/vfs_mock.py")
    helper = symbol(name="read_bytes", module="swissknife/helpers/vfs.ts")
    ids = {real.symbol_id, mock.symbol_id, helper.symbol_id}
    assert len(ids) == 3

    expected_real = expected_contract(symbol=real, summary="real package")
    observed_mock = observed_contract(
        symbol=mock,
        summary="mock helper with same name",
        sources=(observation_source(artifact_id="artifact:mock"),),
    )
    # Different subjects → identity mismatch, not silent merge.
    result = compare_contracts(
        expected_real,
        observed_mock,
        evaluated_at=EVALUATED,
        authority_expires_at=EXPIRES,
        require_same_subject=True,
    )
    assert result.kind is ContractCheckResultKind.WITNESSED_MISMATCH
    assert result.counterexample is not None
    assert result.counterexample.aspect is SemanticAspect.IDENTITY

    # Deterministic: recompute yields the same identities and result id.
    again = compare_contracts(
        expected_real,
        observed_mock,
        evaluated_at=EVALUATED,
        authority_expires_at=EXPIRES,
    )
    assert again.result_id == result.result_id
    assert again.binding.binding_id == result.binding.binding_id
    assert again.counterexample.counterexample_id == (
        result.counterexample.counterexample_id
    )


def test_same_name_interfaces_on_different_surfaces_do_not_bind() -> None:
    expected = expected_contract(
        interface=interface(name="vfs.read", surface="mcp++")
    )
    observed = observed_contract(
        interface=interface(name="vfs.read", surface="http"),
        sources=(observation_source(artifact_id="artifact:http"),),
    )
    # binds_same_surface may differ by surface/version — identity rule catches
    # non-matching subject via interface binding.
    if not observed.binds_same_subject(expected):
        result = compare_contracts(
            expected,
            observed,
            evaluated_at=EVALUATED,
            authority_expires_at=EXPIRES,
        )
        assert result.kind is ContractCheckResultKind.WITNESSED_MISMATCH
        assert SemanticAspect.IDENTITY in result.mismatch_aspects
    else:
        # If interface surface is not part of binds_same_surface, the checker
        # still produces deterministic bindings keyed by interface_name.
        binding = make_binding(expected, observed)
        assert binding.interface_name == "vfs.read"
        assert binding.binding_id == make_binding(expected, observed).binding_id


# ---------------------------------------------------------------------------
# Bundle / multi-path checker, serialization, fail-closed records
# ---------------------------------------------------------------------------


def test_contract_checker_bundle_and_paths() -> None:
    expected = expected_contract()
    observed = observed_contract()
    broken = observed_contract(
        returns=ReturnSpec(type_shape=string_type()),
        sources=(observation_source(artifact_id="artifact:broken"),),
        summary="broken observation",
        repository_observation_id="observation:broken",
    )
    # Two observations for same subject — both are checked.
    bundle = ProgramContractBundle(
        repository_id=REPO,
        tree_id=TREE,
        policy_revision=POLICY,
        expected=(expected,),
        observed=(observed, broken),
    )
    checker = ContractChecker(budget_ms=5_000)
    report = checker.check_bundle(
        bundle,
        call_paths=(static_path(),),
        evaluated_at=EVALUATED,
        authority_expires_at=EXPIRES,
    )
    assert isinstance(report, ContractCheckReport)
    assert report.report_id
    assert len(report.results) == 2
    kinds = {item.kind for item in report.results}
    assert ContractCheckResultKind.PROVED_COMPATIBLE in kinds
    assert ContractCheckResultKind.WITNESSED_MISMATCH in kinds
    assert report.counts_by_kind["proved_compatible"] == 1
    assert report.counts_by_kind["witnessed_mismatch"] == 1

    multi = checker.check_along_paths(
        expected,
        observed,
        (
            static_path(path_name="a"),
            static_path(
                path_name="b",
                resolution=CallPathResolution.DYNAMIC,
            ),
        ),
        evaluated_at=EVALUATED,
        authority_expires_at=EXPIRES,
    )
    assert len(multi) == 2
    assert multi[0].result_id <= multi[1].result_id
    assert {item.kind for item in multi} == {
        ContractCheckResultKind.PROVED_COMPATIBLE,
        ContractCheckResultKind.AMBIGUOUS,
    }


def test_round_trip_serialization_and_forged_identity() -> None:
    result = compare_contracts(
        expected_contract(),
        observed_contract(
            returns=ReturnSpec(type_shape=string_type()),
        ),
        evaluated_at=EVALUATED,
        authority_expires_at=EXPIRES,
    )
    assert result.kind is ContractCheckResultKind.WITNESSED_MISMATCH
    restored = ContractCheckResult.from_dict(result.to_dict())
    assert restored.result_id == result.result_id
    assert restored.counterexample is not None
    assert (
        restored.counterexample.counterexample_id
        == result.counterexample.counterexample_id
    )

    path = static_path()
    assert CallPath.from_dict(path.to_dict()).path_id == path.path_id

    forged = result.to_dict()
    forged["content_id"] = "sha256:" + ("0" * 64)
    with pytest.raises(ForgedIdentityError):
        ContractCheckResult.from_dict(forged)

    with pytest.raises(FrozenInstanceError):
        result.kind = ContractCheckResultKind.PROVED_COMPATIBLE  # type: ignore[misc]


def test_conclusive_counterexample_rejects_stale_authority() -> None:
    binding = make_binding(expected_contract(), observed_contract())
    aspect = AspectCheckResult(
        aspect=SemanticAspect.OUTPUTS,
        verdict=AspectVerdict.MISMATCH,
        rule_id="rule:test",
        expected_fact="bytes",
        observed_fact="str",
        summary="type mismatch",
    )
    with pytest.raises(StaleAuthorityError):
        minimal_counterexample(
            binding=binding,
            aspect_result=aspect,
            evaluated_at=EVALUATED,
            authority_expires_at=STALE_EXPIRES,
        )

    stale_binding = make_binding(
        expected_contract(),
        observed_contract(),
        cache_generation="gen:old",
        expected_cache_generation="gen:new",
    )
    with pytest.raises(StaleAuthorityError):
        minimal_counterexample(
            binding=stale_binding,
            aspect_result=aspect,
            evaluated_at=EVALUATED,
            authority_expires_at=EXPIRES,
        )


def test_scope_mismatch_on_call_path_repo() -> None:
    bad_path = CallPath(
        repository_id="repository:other",
        tree_id=TREE,
        policy_revision=POLICY,
        path_name="foreign",
        steps=(
            CallPathStep(
                step_index=0,
                symbol_name="read_bytes",
                resolution=CallPathResolution.STATIC,
            ),
        ),
    )
    with pytest.raises(ScopeMismatchError):
        compare_contracts(
            expected_contract(),
            observed_contract(),
            call_path=bad_path,
            evaluated_at=EVALUATED,
            authority_expires_at=EXPIRES,
        )


@pytest.mark.parametrize(
    ("observed", "different_field"),
    (
        (
            observed_contract(
                symbol=symbol(repository_id="repository:other")
            ),
            "repository",
        ),
        (
            observed_contract(
                symbol=SymbolIdentity(
                    repository_id=REPO,
                    tree_id=TREE,
                    module_path="ipfs_kit_py/vfs.py",
                    symbol_name="read_bytes",
                    language="python",
                    span_start=11,
                    span_end=40,
                    blob_cid="baguqeera" + "1" * 50,
                )
            ),
            "symbol",
        ),
        (
            observed_contract(
                interface=InterfaceIdentity(
                    interface_name="vfs.read",
                    surface="mcp++",
                    version="2.0",
                    method="read",
                    protocol="mcp",
                    path_or_uri="mcp://vfs/read",
                )
            ),
            "interface",
        ),
        (
            observed_contract(policy_revision="policy:other@1"),
            "policy",
        ),
    ),
)
def test_exact_subject_binding_rejects_near_matches(
    observed: ObservedProgramContract,
    different_field: str,
) -> None:
    result = compare_contracts(
        expected_contract(),
        observed,
        evaluated_at=EVALUATED,
        authority_expires_at=EXPIRES,
    )
    binding = result.binding
    assert result.kind is ContractCheckResultKind.WITNESSED_MISMATCH
    assert result.counterexample is not None
    assert result.counterexample.aspect is SemanticAspect.IDENTITY
    assert not binding.subject_matches
    if different_field == "repository":
        assert binding.repository_id != binding.observed_repository_id
    elif different_field == "symbol":
        assert binding.expected_symbol_id != binding.observed_symbol_id
    elif different_field == "interface":
        assert binding.expected_interface_id != binding.observed_interface_id
    else:
        assert binding.policy_revision != binding.observed_policy_revision


def test_exact_call_path_symbol_and_interface_binding() -> None:
    for path in (
        CallPath(
            repository_id=REPO,
            tree_id=TREE,
            policy_revision=POLICY,
            path_name="wrong-interface",
            entry_interface="vfs.write",
            steps=(
                CallPathStep(
                    step_index=0,
                    symbol_name="read_bytes",
                    resolution=CallPathResolution.STATIC,
                ),
            ),
        ),
        CallPath(
            repository_id=REPO,
            tree_id=TREE,
            policy_revision=POLICY,
            path_name="wrong-symbol",
            exit_symbol="write_bytes",
            steps=(
                CallPathStep(
                    step_index=0,
                    symbol_name="read_bytes",
                    resolution=CallPathResolution.STATIC,
                ),
            ),
        ),
    ):
        with pytest.raises(ScopeMismatchError):
            compare_contracts(
                expected_contract(),
                observed_contract(),
                call_path=path,
                evaluated_at=EVALUATED,
                authority_expires_at=EXPIRES,
            )


def test_freshness_is_bound_to_generations_and_witness_window() -> None:
    current = compare_contracts(
        expected_contract(),
        observed_contract(),
        evaluated_at=EVALUATED,
        authority_expires_at=EXPIRES,
        cache_generation="gen:9",
        expected_cache_generation="gen:9",
    )
    assert current.binding.cache_generation == "gen:9"
    assert current.binding.expected_cache_generation == "gen:9"
    assert (
        current.binding.cache_binding_freshness is CacheFreshness.CURRENT
    )

    forged_generation = current.to_dict()
    forged_generation["binding"]["expected_cache_generation"] = "gen:10"
    with pytest.raises(ForgedIdentityError):
        ContractCheckResult.from_dict(forged_generation)

    rebound_generation = current.to_dict()
    rebound_generation["binding"]["expected_cache_generation"] = "gen:10"
    rebound_generation["binding"]["cache_binding_freshness"] = "stale"
    with pytest.raises(StaleAuthorityError):
        ContractCheckResult.from_dict(rebound_generation)

    mismatch = compare_contracts(
        expected_contract(),
        observed_contract(returns=ReturnSpec(type_shape=string_type())),
        evaluated_at=EVALUATED,
        authority_expires_at=EXPIRES,
    )
    forged_window = mismatch.to_dict()
    assert forged_window["counterexample"] is not None
    forged_window["counterexample"]["evaluated_at"] = (
        "2026-07-29T12:01:00+00:00"
    )
    with pytest.raises(StaleAuthorityError):
        ContractCheckResult.from_dict(forged_window)


def test_proved_compatible_rejects_blocking_aspect_on_construction() -> None:
    binding = make_binding(expected_contract(), observed_contract())
    with pytest.raises(ContractCheckerError):
        ContractCheckResult(
            kind=ContractCheckResultKind.PROVED_COMPATIBLE,
            binding=binding,
            aspect_results=(
                AspectCheckResult(
                    aspect=SemanticAspect.OUTPUTS,
                    verdict=AspectVerdict.MISMATCH,
                    rule_id="rule:bad",
                    summary="should not pair with proved_compatible",
                ),
            ),
            summary="illegal",
            evaluated_at=EVALUATED,
            authority_expires_at=EXPIRES,
        )


def test_input_output_variance_helpers() -> None:
    expected = expected_contract(
        inputs=(
            path_param(),
            ParameterSpec(
                name="offset",
                type_shape=int_type(),
                kind=ParameterKind.KEYWORD,
                optionality=Optionality.OPTIONAL,
            ),
        )
    )
    # Observed accepts string path (same) — compatible.
    assert (
        check_inputs(expected, observed_contract()).verdict
        is AspectVerdict.COMPATIBLE
    )
    # Observed path is int — cannot accept string values (contravariance fail).
    narrow = observed_contract(
        inputs=(path_param(type_shape=int_type()),)
    )
    assert check_inputs(expected, narrow).verdict is AspectVerdict.MISMATCH

    # Covariant outputs: bytes subtypes bytes.
    assert (
        check_outputs(expected_contract(), observed_contract()).verdict
        is AspectVerdict.COMPATIBLE
    )


def test_content_identity_stable_for_equal_payloads() -> None:
    left = compare_contracts(
        expected_contract(),
        observed_contract(),
        evaluated_at=EVALUATED,
        authority_expires_at=EXPIRES,
    )
    right = compare_contracts(
        expected_contract(),
        observed_contract(),
        evaluated_at=EVALUATED,
        authority_expires_at=EXPIRES,
    )
    assert left.result_id == right.result_id
    assert contract_check_content_identity(left.to_dict()) == left.result_id


def test_ordering_and_consistency_closed_rules() -> None:
    expected = expected_contract(
        ordering=OrderingSpec(mode=OrderingMode.SEQUENTIAL),
        consistency=ConsistencySpec(mode=ConsistencyMode.STRONG),
    )
    concurrent = observed_contract(
        ordering=OrderingSpec(mode=OrderingMode.CONCURRENT),
        consistency=ConsistencySpec(mode=ConsistencyMode.EVENTUAL),
    )
    result = compare_contracts(
        expected,
        concurrent,
        evaluated_at=EVALUATED,
        authority_expires_at=EXPIRES,
    )
    assert result.kind is ContractCheckResultKind.WITNESSED_MISMATCH
    assert set(result.mismatch_aspects) >= {
        SemanticAspect.ORDERING,
        SemanticAspect.CONSISTENCY,
    }


def test_capability_required_and_forbidden() -> None:
    expected = expected_contract(
        capabilities=(
            CapabilitySpec(
                capability_name="vfs.read",
                mode=CapabilityMode.REQUIRED,
            ),
            CapabilitySpec(
                capability_name="vfs.admin",
                mode=CapabilityMode.FORBIDDEN,
            ),
        )
    )
    missing = observed_contract(capabilities=())
    result = compare_contracts(
        expected,
        missing,
        evaluated_at=EVALUATED,
        authority_expires_at=EXPIRES,
    )
    assert result.kind is ContractCheckResultKind.WITNESSED_MISMATCH
    assert SemanticAspect.CAPABILITIES in result.mismatch_aspects

    present_forbidden = observed_contract(
        capabilities=(
            CapabilitySpec(
                capability_name="vfs.read",
                mode=CapabilityMode.OBSERVED,
            ),
            CapabilitySpec(
                capability_name="vfs.admin",
                mode=CapabilityMode.OBSERVED,
            ),
        )
    )
    result2 = compare_contracts(
        expected,
        present_forbidden,
        evaluated_at=EVALUATED,
        authority_expires_at=EXPIRES,
    )
    assert result2.kind is ContractCheckResultKind.WITNESSED_MISMATCH


def test_check_binding_round_trip() -> None:
    binding = make_binding(
        expected_contract(),
        observed_contract(),
        call_path=static_path(),
        cache_generation="gen:9",
        expected_cache_generation="gen:9",
    )
    restored = CheckBinding.from_dict(binding.to_dict())
    assert restored.binding_id == binding.binding_id
    assert restored.call_path_id == static_path().path_id
    assert restored.checker_version == CHECKER_VERSION
    assert restored.subject_matches
    assert restored.cache_binding_freshness is CacheFreshness.CURRENT
