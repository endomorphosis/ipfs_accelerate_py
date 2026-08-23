from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.contracts import (
    ArtifactBindings,
    ArtifactState,
    EffectClass,
    FORBIDDEN_HOLE_TYPES,
    HoleType,
    ProcedureHole,
    ProcedureSafetyError,
    ProviderClass,
    parse_procedure_artifact,
)
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.hole_resolution import (
    ALLOWED_HOLE_TYPES,
    DETERMINISTIC_PROVIDER_CLASSES,
    HOLE_RESOLVER_REVISION,
    MODEL_PROVIDER_CLASSES,
    PROVIDER_ROUTE_ORDER,
    CompiledHoleContext,
    HoleCandidate,
    HoleContextReference,
    HoleProviderOutcome,
    HoleProviderResult,
    HoleRequest,
    HoleResolution,
    HoleResolutionAction,
    HoleResolutionError,
    HoleResolutionReason,
    HoleResolutionValidator,
    HoleResolver,
    HoleTypeError,
    HoleValidationError,
    HoleValidationReceipt,
    ProviderCapacitySnapshot,
    default_hole_context_compiler,
    model_route_for_provider_class,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import ModelRoute


def _bindings(**changes: object) -> ArtifactBindings:
    values: dict[str, object] = {
        "repository_id": "repo-main",
        "repository_commit": "commit-abc123",
        "tree_id": "tree-abc123",
        "objective_id": "PCPC-G030",
        "task_id": "PCPC-020",
        "contract_revision": "procedure-contracts-v1",
        "policy_revision": "authority-policy-v1",
        "environment_id": "python312-linux-lock1",
    }
    values.update(changes)
    return ArtifactBindings(**values)


def _capacity(
    *classes: ProviderClass,
    available: bool = True,
    remaining_calls: int = 4,
    max_context_bytes: int = 65_536,
    max_tokens: int = 8_192,
) -> tuple[ProviderCapacitySnapshot, ...]:
    return tuple(
        ProviderCapacitySnapshot(
            provider_class=item,
            available=available,
            remaining_calls=remaining_calls,
            max_context_bytes=max_context_bytes,
            max_tokens=max_tokens,
            provider_id=f"provider.{item.value}",
        )
        for item in classes
    )


def _reference(**changes: object) -> HoleContextReference:
    values: dict[str, object] = {
        "reference_id": "evidence.symbols",
        "content_id": "cid-symbols-1",
        "tree_id": "tree-abc123",
        "byte_count": 64,
        "token_count": 16,
        "required": True,
        "summary": "allowed symbols",
    }
    values.update(changes)
    return HoleContextReference(**values)


def _output_for(hole_type: HoleType) -> dict[str, object]:
    if hole_type is HoleType.SELECT_ONE_OF_ALLOWED_SYMBOLS:
        return {"schema_ref": "schema.symbol-out", "selected": "pkg.mod:symbol_a"}
    if hole_type is HoleType.GENERATE_DOCSTRING:
        return {"schema_ref": "schema.docstring-out", "docstring": "Return the selected symbol."}
    if hole_type is HoleType.PROPOSE_BOUNDED_PATCH:
        return {"schema_ref": "schema.patch-out", "template_id": "template.import-purity"}
    if hole_type is HoleType.CLASSIFY_FAILURE:
        return {"schema_ref": "schema.failure-out", "failure_class": "missing-import"}
    if hole_type is HoleType.CHOOSE_APPROVED_REPAIR_TEMPLATE:
        return {"schema_ref": "schema.template-out", "template_id": "template.import-purity"}
    if hole_type is HoleType.SUGGEST_MISSING_TEST_CASE:
        return {"schema_ref": "schema.test-out", "test_name": "test_import_is_pure"}
    return {"schema_ref": "schema.lemma-out", "lemma_name": "import-is-pure"}


def _input_for(hole_type: HoleType) -> dict[str, object]:
    if hole_type is HoleType.SELECT_ONE_OF_ALLOWED_SYMBOLS:
        return {"allowed_values": ("pkg.mod:symbol_a", "pkg.mod:symbol_b")}
    if hole_type is HoleType.GENERATE_DOCSTRING:
        return {"symbol": "pkg.mod:symbol_a"}
    if hole_type is HoleType.PROPOSE_BOUNDED_PATCH:
        return {"template_ids": ("template.import-purity",)}
    if hole_type is HoleType.CLASSIFY_FAILURE:
        return {"failure_signature": "import-side-effect"}
    if hole_type is HoleType.CHOOSE_APPROVED_REPAIR_TEMPLATE:
        return {"template_ids": ("template.import-purity",)}
    if hole_type is HoleType.SUGGEST_MISSING_TEST_CASE:
        return {"uncovered_obligation": "import-is-pure"}
    return {"obligation": "import-is-pure"}


def _request(**changes: object) -> HoleRequest:
    hole_type = changes.get("hole_type", HoleType.SELECT_ONE_OF_ALLOWED_SYMBOLS)
    if not isinstance(hole_type, HoleType):
        hole_type = HoleType(hole_type) if hole_type in ALLOWED_HOLE_TYPES else hole_type
    values: dict[str, object] = {
        "bindings": _bindings(),
        "hole_id": "hole.select-symbol",
        "hole_type": HoleType.SELECT_ONE_OF_ALLOWED_SYMBOLS,
        "input_schema_ref": "schema.symbol-in",
        "output_schema_ref": "schema.symbol-out",
        "allowed_provider_classes": (
            ProviderClass.EXACT_CACHE,
            ProviderClass.REMOTE_STANDARD_MODEL,
        ),
        "context_budget_bytes": 32_768,
        "validation_observation_ids": ("observation.tests",),
        "fallback_step_id": "step.fallback",
        "maximum_attempts": 2,
        "input_payload": _input_for(HoleType.SELECT_ONE_OF_ALLOWED_SYMBOLS),
        "context_references": (_reference(),),
        "authority_requirement_ids": ("authority.execute",),
        "effect_classes": (EffectClass.OBSERVE, EffectClass.MODEL_REQUEST),
        "token_budget": 4_096,
    }
    values.update(changes)
    return HoleRequest(**values)


class RecordingProvider:
    def __init__(self, result: object) -> None:
        self.calls = 0
        self.result = result

    def propose(self, request: HoleRequest, compiled: CompiledHoleContext) -> object:
        self.calls += 1
        if callable(self.result):
            return self.result(request, compiled)
        return self.result


def _proposed(hole_type: HoleType = HoleType.SELECT_ONE_OF_ALLOWED_SYMBOLS, **changes: object) -> HoleProviderResult:
    values: dict[str, object] = {
        "outcome": HoleProviderOutcome.PROPOSED,
        "output": _output_for(hole_type),
        "token_count": 8,
    }
    values.update(changes)
    return HoleProviderResult(**values)


def _resolver(
    *,
    providers: dict[ProviderClass, RecordingProvider] | None = None,
    capacity: tuple[ProviderCapacitySnapshot, ...] | None = None,
    current_tree_id: str = "tree-abc123",
    compiler=None,
) -> HoleResolver:
    default_providers = {
        ProviderClass.EXACT_CACHE: RecordingProvider(_proposed()),
    }
    selected = providers if providers is not None else default_providers
    classes = tuple(selected) or (
        ProviderClass.EXACT_CACHE,
        ProviderClass.REMOTE_STANDARD_MODEL,
    )
    return HoleResolver(
        compiler or default_hole_context_compiler(),
        providers=selected,
        capacity=capacity if capacity is not None else _capacity(*classes),
        current_tree_id=current_tree_id,
    )


def test_allowed_hole_types_round_trip_and_remain_candidates() -> None:
    for hole_type in HoleType:
        request = _request(
            hole_id=f"hole.{hole_type.value.lower()}",
            hole_type=hole_type,
            input_schema_ref=f"schema.{hole_type.value.lower()}-in",
            output_schema_ref=f"schema.{hole_type.value.lower()}-out",
            input_payload=_input_for(hole_type),
            allowed_provider_classes=(ProviderClass.DECLARATIVE_RULE,),
        )
        decoded = HoleRequest.from_dict(request.to_dict())
        assert decoded == request
        assert decoded.state is ArtifactState.CANDIDATE
        assert decoded.can_authorize is False
        parsed = parse_procedure_artifact(request.to_dict())
        assert isinstance(parsed, HoleRequest)
        assert parsed.content_id == request.content_id


@pytest.mark.parametrize("forbidden", sorted(FORBIDDEN_HOLE_TYPES))
def test_prohibited_hole_types_cannot_be_requested(forbidden: str) -> None:
    with pytest.raises(HoleTypeError, match="forbidden hole types"):
        _request(hole_type=forbidden)


def test_unknown_hole_type_is_rejected() -> None:
    with pytest.raises(HoleTypeError, match="outside the allowed"):
        _request(hole_type="INVENTED_HOLE")


def test_only_declared_providers_are_invoked() -> None:
    cache = RecordingProvider(_proposed())
    remote = RecordingProvider(_proposed())
    resolver = _resolver(
        providers={
            ProviderClass.EXACT_CACHE: cache,
            ProviderClass.REMOTE_STANDARD_MODEL: remote,
        },
        capacity=_capacity(ProviderClass.EXACT_CACHE, ProviderClass.REMOTE_STANDARD_MODEL),
    )
    request = _request(allowed_provider_classes=(ProviderClass.EXACT_CACHE,))
    resolution = resolver.resolve(request)

    assert resolution.action is HoleResolutionAction.PROPOSE
    assert resolution.provider_class == ProviderClass.EXACT_CACHE.value
    assert cache.calls == 1
    assert remote.calls == 0
    candidate = resolver.last_candidate(request)
    assert candidate is not None
    assert candidate.state is ArtifactState.CANDIDATE
    assert candidate.validated is False
    assert resolution.remains_candidate is True
    assert resolution.can_authorize is False


def test_undeclared_provider_cannot_be_called_even_when_injected() -> None:
    remote = RecordingProvider(_proposed())
    resolver = _resolver(
        providers={ProviderClass.REMOTE_STANDARD_MODEL: remote},
        capacity=_capacity(ProviderClass.REMOTE_STANDARD_MODEL),
    )
    request = _request(allowed_provider_classes=(ProviderClass.EXACT_CACHE,))
    resolution = resolver.resolve(request)

    assert remote.calls == 0
    assert resolution.action is HoleResolutionAction.FALLBACK
    assert resolution.reason_code in {
        HoleResolutionReason.CAPACITY_MISSING,
        HoleResolutionReason.PROVIDER_UNAVAILABLE,
        HoleResolutionReason.FALLBACK_REQUIRED,
    }


def test_cache_hit_skips_remote_model() -> None:
    cache = RecordingProvider(_proposed())
    remote = RecordingProvider(_proposed())
    resolver = _resolver(
        providers={
            ProviderClass.EXACT_CACHE: cache,
            ProviderClass.REMOTE_STANDARD_MODEL: remote,
        },
        capacity=_capacity(ProviderClass.EXACT_CACHE, ProviderClass.REMOTE_STANDARD_MODEL),
    )
    resolution = resolver.resolve(_request())

    assert resolution.action is HoleResolutionAction.PROPOSE
    assert resolution.provider_class == ProviderClass.EXACT_CACHE.value
    assert cache.calls == 1
    assert remote.calls == 0


def test_model_runs_only_after_deterministic_cache_miss() -> None:
    cache = RecordingProvider(
        HoleProviderResult(outcome=HoleProviderOutcome.MISSED, failure_code="cache-miss")
    )
    remote = RecordingProvider(_proposed())
    resolver = _resolver(
        providers={
            ProviderClass.EXACT_CACHE: cache,
            ProviderClass.REMOTE_STANDARD_MODEL: remote,
        },
        capacity=_capacity(ProviderClass.EXACT_CACHE, ProviderClass.REMOTE_STANDARD_MODEL),
    )
    resolution = resolver.resolve(_request())

    assert cache.calls == 1
    assert remote.calls == 1
    assert resolution.action is HoleResolutionAction.PROPOSE
    assert resolution.provider_class == ProviderClass.REMOTE_STANDARD_MODEL.value
    assert PROVIDER_ROUTE_ORDER.index(ProviderClass.EXACT_CACHE) < PROVIDER_ROUTE_ORDER.index(
        ProviderClass.REMOTE_STANDARD_MODEL
    )


def test_context_budget_overflow_is_refused() -> None:
    resolver = _resolver()
    request = _request(context_budget_bytes=64)
    resolution = resolver.resolve(request)
    assert resolution.action is HoleResolutionAction.REFUSE
    assert resolution.reason_code is HoleResolutionReason.CONTEXT_BUDGET_EXCEEDED


def test_token_budget_overflow_is_refused() -> None:
    resolver = _resolver()
    request = _request(token_budget=1)
    resolution = resolver.resolve(request)
    assert resolution.action is HoleResolutionAction.REFUSE
    assert resolution.reason_code is HoleResolutionReason.TOKEN_BUDGET_EXCEEDED


def test_attempt_bound_falls_back_without_another_provider_call() -> None:
    cache = RecordingProvider(
        HoleProviderResult(outcome=HoleProviderOutcome.MISSED, failure_code="cache-miss")
    )
    resolver = _resolver(
        providers={ProviderClass.EXACT_CACHE: cache},
        capacity=_capacity(ProviderClass.EXACT_CACHE),
    )
    request = _request(
        allowed_provider_classes=(ProviderClass.EXACT_CACHE,),
        maximum_attempts=1,
    )
    first = resolver.resolve(request)
    second = resolver.resolve(request)
    assert first.action is HoleResolutionAction.FALLBACK
    assert first.attempts_used == 1
    assert cache.calls == 1
    assert second.action is HoleResolutionAction.FALLBACK
    assert second.reason_code is HoleResolutionReason.ATTEMPT_BUDGET_EXCEEDED
    assert cache.calls == 1


def test_identical_failure_suppresses_another_call() -> None:
    cache = RecordingProvider(
        HoleProviderResult(outcome=HoleProviderOutcome.FAILED, failure_code="no-match")
    )
    resolver = _resolver(
        providers={ProviderClass.EXACT_CACHE: cache},
        capacity=_capacity(ProviderClass.EXACT_CACHE),
    )
    request = _request(allowed_provider_classes=(ProviderClass.EXACT_CACHE,), maximum_attempts=2)
    first = resolver.resolve(request)
    second = resolver.resolve(request)
    assert first.action is HoleResolutionAction.FALLBACK
    assert cache.calls == 1
    assert second.action is HoleResolutionAction.SUPPRESS
    assert second.reason_code is HoleResolutionReason.IDENTICAL_FAILURE
    assert cache.calls == 1


def test_no_new_evidence_suppresses_another_call_after_candidate() -> None:
    cache = RecordingProvider(_proposed())
    resolver = _resolver(
        providers={ProviderClass.EXACT_CACHE: cache},
        capacity=_capacity(ProviderClass.EXACT_CACHE),
    )
    request = _request(allowed_provider_classes=(ProviderClass.EXACT_CACHE,))
    first = resolver.resolve(request)
    second = resolver.resolve(request)
    assert first.action is HoleResolutionAction.PROPOSE
    assert first.state is ArtifactState.CANDIDATE
    assert cache.calls == 1
    assert second.action is HoleResolutionAction.SUPPRESS
    assert second.reason_code is HoleResolutionReason.NO_NEW_EVIDENCE
    assert second.candidate_cid == first.candidate_cid
    assert cache.calls == 1


def test_new_evidence_allows_another_provider_call() -> None:
    cache = RecordingProvider(_proposed())
    resolver = _resolver(
        providers={ProviderClass.EXACT_CACHE: cache},
        capacity=_capacity(ProviderClass.EXACT_CACHE),
    )
    first_request = _request(
        allowed_provider_classes=(ProviderClass.EXACT_CACHE,),
        context_references=(_reference(content_id="cid-symbols-1"),),
    )
    second_request = _request(
        allowed_provider_classes=(ProviderClass.EXACT_CACHE,),
        context_references=(_reference(content_id="cid-symbols-2"),),
    )
    first = resolver.resolve(first_request)
    second = resolver.resolve(second_request)
    assert first.action is HoleResolutionAction.PROPOSE
    assert second.action is HoleResolutionAction.PROPOSE
    assert cache.calls == 2
    assert first.candidate_cid != second.candidate_cid


def test_stale_context_is_refused_before_provider_call() -> None:
    cache = RecordingProvider(_proposed())
    resolver = _resolver(
        providers={ProviderClass.EXACT_CACHE: cache},
        capacity=_capacity(ProviderClass.EXACT_CACHE),
        current_tree_id="tree-old",
    )
    resolution = resolver.resolve(_request())
    assert resolution.action is HoleResolutionAction.REFUSE
    assert resolution.reason_code is HoleResolutionReason.STALE_CONTEXT
    assert cache.calls == 0


def test_stale_context_reference_tree_is_refused() -> None:
    cache = RecordingProvider(_proposed())
    resolver = _resolver(
        providers={ProviderClass.EXACT_CACHE: cache},
        capacity=_capacity(ProviderClass.EXACT_CACHE),
    )
    request = _request(context_references=(_reference(tree_id="tree-other"),))
    resolution = resolver.resolve(request)
    assert resolution.action is HoleResolutionAction.REFUSE
    assert resolution.reason_code is HoleResolutionReason.STALE_CONTEXT
    assert cache.calls == 0


def test_injection_and_authority_fields_are_refused() -> None:
    cache = RecordingProvider(
        HoleProviderResult(
            outcome=HoleProviderOutcome.PROPOSED,
            output={
                "schema_ref": "schema.symbol-out",
                "selected": "pkg.mod:symbol_a",
                "grant_authority": True,
            },
        )
    )
    resolver = _resolver(
        providers={ProviderClass.EXACT_CACHE: cache},
        capacity=_capacity(ProviderClass.EXACT_CACHE),
    )
    resolution = resolver.resolve(_request(allowed_provider_classes=(ProviderClass.EXACT_CACHE,)))
    assert resolution.action is HoleResolutionAction.REFUSE
    assert resolution.reason_code is HoleResolutionReason.AUTHORITY_FLOW_REJECTED


def test_secret_and_callback_injection_is_rejected() -> None:
    cache = RecordingProvider(
        {
            "outcome": "proposed",
            "output": {
                "schema_ref": "schema.symbol-out",
                "selected": "pkg.mod:symbol_a",
                "api_key": "redacted",
            },
        }
    )
    resolver = _resolver(
        providers={ProviderClass.EXACT_CACHE: cache},
        capacity=_capacity(ProviderClass.EXACT_CACHE),
    )
    resolution = resolver.resolve(_request(allowed_provider_classes=(ProviderClass.EXACT_CACHE,)))
    assert resolution.action is HoleResolutionAction.REFUSE
    assert resolution.reason_code is HoleResolutionReason.INJECTION_REJECTED


def test_effect_flow_cannot_introduce_merge_or_rollback() -> None:
    cache = RecordingProvider(
        HoleProviderResult(
            outcome=HoleProviderOutcome.PROPOSED,
            output={
                "schema_ref": "schema.symbol-out",
                "selected": "pkg.mod:symbol_a",
                "effect_classes": ("merge",),
            },
        )
    )
    resolver = _resolver(
        providers={ProviderClass.EXACT_CACHE: cache},
        capacity=_capacity(ProviderClass.EXACT_CACHE),
    )
    resolution = resolver.resolve(_request(allowed_provider_classes=(ProviderClass.EXACT_CACHE,)))
    assert resolution.action is HoleResolutionAction.REFUSE
    assert resolution.reason_code is HoleResolutionReason.EFFECT_FLOW_REJECTED


def test_schema_mismatch_and_out_of_vocabulary_selection_are_rejected() -> None:
    cache = RecordingProvider(
        HoleProviderResult(
            outcome=HoleProviderOutcome.PROPOSED,
            output={"schema_ref": "schema.symbol-out", "selected": "pkg.mod:not-allowed"},
        )
    )
    resolver = _resolver(
        providers={ProviderClass.EXACT_CACHE: cache},
        capacity=_capacity(ProviderClass.EXACT_CACHE),
    )
    resolution = resolver.resolve(_request(allowed_provider_classes=(ProviderClass.EXACT_CACHE,)))
    assert resolution.action is HoleResolutionAction.REFUSE
    assert resolution.reason_code is HoleResolutionReason.SCHEMA_MISMATCH


def test_outputs_remain_candidates_until_independent_validation() -> None:
    resolver = _resolver()
    request = _request(allowed_provider_classes=(ProviderClass.EXACT_CACHE,))
    resolution = resolver.resolve(request)
    candidate = resolver.last_candidate(request)
    assert candidate is not None
    validator = HoleResolutionValidator(current_tree_id="tree-abc123")
    pending = validator.validate_candidate(request, candidate)
    assert pending.accepted is False
    assert pending.reason_code is HoleResolutionReason.VALIDATION_REQUIRED
    assert pending.remains_candidate is True
    assert pending.can_authorize is False
    assert pending.state is ArtifactState.CANDIDATE

    admitted = validator.validate_candidate(
        request, candidate, observations=("observation.tests",)
    )
    assert admitted.accepted is True
    assert admitted.remains_candidate is True
    assert admitted.can_authorize is False
    assert candidate.state is ArtifactState.CANDIDATE
    assert resolution.state is ArtifactState.CANDIDATE
    parsed = parse_procedure_artifact(admitted.to_dict())
    assert isinstance(parsed, HoleValidationReceipt)
    assert parsed.accepted is True
    assert parsed.can_authorize is False


def test_validation_receipt_cannot_promote_or_authorize() -> None:
    with pytest.raises(HoleValidationError, match="cannot authorize"):
        HoleValidationReceipt(
            bindings=_bindings(),
            request_cid="cid-request",
            candidate_cid="cid-candidate",
            hole_id="hole.select-symbol",
            accepted=True,
            reason_code=HoleResolutionReason.CANDIDATE_PROPOSED,
            observation_ids=("observation.tests",),
            can_authorize=True,
        )
    with pytest.raises(HoleValidationError, match="remain candidates"):
        HoleValidationReceipt(
            bindings=_bindings(),
            request_cid="cid-request",
            candidate_cid="cid-candidate",
            hole_id="hole.select-symbol",
            accepted=True,
            reason_code=HoleResolutionReason.CANDIDATE_PROPOSED,
            observation_ids=("observation.tests",),
            state=ArtifactState.PROMOTED,
        )


def test_candidate_cannot_self_validate_or_leave_candidate_tier() -> None:
    request = _request(allowed_provider_classes=(ProviderClass.EXACT_CACHE,))
    with pytest.raises(HoleValidationError, match="remain candidates"):
        HoleCandidate(
            bindings=request.bindings,
            request_cid=request.content_id,
            hole_id=request.hole_id,
            hole_type=request.hole_type,
            output_schema_ref=request.output_schema_ref,
            provider_class=ProviderClass.EXACT_CACHE,
            output=_output_for(request.hole_type),
            context_receipt_cid="cid-receipt",
            evidence_fingerprint="cid-evidence",
            state=ArtifactState.VERIFIED,
        )
    with pytest.raises(HoleValidationError, match="self-validate"):
        HoleCandidate(
            bindings=request.bindings,
            request_cid=request.content_id,
            hole_id=request.hole_id,
            hole_type=request.hole_type,
            output_schema_ref=request.output_schema_ref,
            provider_class=ProviderClass.EXACT_CACHE,
            output=_output_for(request.hole_type),
            context_receipt_cid="cid-receipt",
            evidence_fingerprint="cid-evidence",
            validated=True,
        )


def test_fallback_when_providers_miss() -> None:
    cache = RecordingProvider(
        HoleProviderResult(outcome=HoleProviderOutcome.MISSED, failure_code="cache-miss")
    )
    resolver = _resolver(
        providers={ProviderClass.EXACT_CACHE: cache},
        capacity=_capacity(ProviderClass.EXACT_CACHE),
    )
    resolution = resolver.resolve(
        _request(allowed_provider_classes=(ProviderClass.EXACT_CACHE,))
    )
    assert resolution.action is HoleResolutionAction.FALLBACK
    assert resolution.fallback_step_id == "step.fallback"
    assert resolution.reason_code is HoleResolutionReason.FALLBACK_REQUIRED
    assert resolution.resolver_revision == HOLE_RESOLVER_REVISION


def test_unavailable_capacity_does_not_call_provider() -> None:
    cache = RecordingProvider(_proposed())
    resolver = _resolver(
        providers={ProviderClass.EXACT_CACHE: cache},
        capacity=_capacity(ProviderClass.EXACT_CACHE, available=False, remaining_calls=0),
    )
    resolution = resolver.resolve(
        _request(allowed_provider_classes=(ProviderClass.EXACT_CACHE,))
    )
    assert cache.calls == 0
    assert resolution.action is HoleResolutionAction.FALLBACK
    assert resolution.reason_code is HoleResolutionReason.PROVIDER_UNAVAILABLE


def test_from_procedure_hole_preserves_bounds_and_validation() -> None:
    hole = ProcedureHole(
        hole_id="hole.classify",
        hole_type=HoleType.CLASSIFY_FAILURE,
        input_schema_ref="schema.failure-in",
        output_schema_ref="schema.failure-out",
        allowed_provider_classes=(ProviderClass.DETERMINISTIC_CLASSIFIER,),
        context_budget_bytes=32_768,
        authority_requirement_ids=("authority.execute",),
        effect_classes=(EffectClass.OBSERVE, EffectClass.MODEL_REQUEST),
        validation_observation_ids=("observation.tests",),
        fallback_step_id="step.fallback",
        maximum_attempts=2,
    )
    request = HoleRequest.from_procedure_hole(
        hole,
        _bindings(),
        input_payload=_input_for(HoleType.CLASSIFY_FAILURE),
        token_budget=4_096,
    )
    assert request.hole_type is HoleType.CLASSIFY_FAILURE
    assert request.maximum_attempts == 2
    assert request.context_budget_bytes == 32_768
    assert request.validation_observation_ids == ("observation.tests",)
    classifier = RecordingProvider(_proposed(HoleType.CLASSIFY_FAILURE))
    resolver = _resolver(
        providers={ProviderClass.DETERMINISTIC_CLASSIFIER: classifier},
        capacity=_capacity(ProviderClass.DETERMINISTIC_CLASSIFIER),
    )
    resolution = resolver.resolve(request)
    assert resolution.action is HoleResolutionAction.PROPOSE
    assert resolution.provider_class == ProviderClass.DETERMINISTIC_CLASSIFIER.value


def test_hole_request_rejects_non_model_effects() -> None:
    with pytest.raises(HoleResolutionError, match="observe/model_request"):
        _request(effect_classes=(EffectClass.MERGE,))


def test_model_route_mapping_uses_provider_route_api() -> None:
    assert model_route_for_provider_class(ProviderClass.EXACT_CACHE) is ModelRoute.DETERMINISTIC_ONLY
    assert model_route_for_provider_class(ProviderClass.LOCAL_SMALL_MODEL) is ModelRoute.SMALL_LOCAL_MODEL
    assert model_route_for_provider_class(ProviderClass.REMOTE_STANDARD_MODEL) is ModelRoute.MEDIUM_MODEL
    assert model_route_for_provider_class(ProviderClass.REMOTE_STRONG_MODEL) is ModelRoute.FRONTIER_MODEL
    assert model_route_for_provider_class(ProviderClass.HUMAN) is ModelRoute.HUMAN_REVIEW_REQUIRED
    assert DETERMINISTIC_PROVIDER_CLASSES.isdisjoint(MODEL_PROVIDER_CLASSES)


def test_resolution_and_candidate_are_immutable_and_parseable() -> None:
    resolver = _resolver()
    request = _request(allowed_provider_classes=(ProviderClass.EXACT_CACHE,))
    resolution = resolver.resolve(request)
    candidate = resolver.last_candidate(request)
    assert candidate is not None
    assert HoleResolution.from_dict(resolution.to_dict()) == resolution
    assert HoleCandidate.from_dict(candidate.to_dict()) == candidate
    assert parse_procedure_artifact(resolution.to_dict()) == resolution
    with pytest.raises(FrozenInstanceError):
        resolution.action = HoleResolutionAction.REFUSE  # type: ignore[misc]


def test_input_payload_rejects_authority_and_unsafe_fields() -> None:
    with pytest.raises(HoleResolutionError, match="forbidden"):
        _request(input_payload={"allowed_values": ("a",), "skip_validation": True})
    with pytest.raises(ProcedureSafetyError):
        _request(input_payload={"allowed_values": ("a",), "api_key": "secret"})
