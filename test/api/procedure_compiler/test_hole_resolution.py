from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from ipfs_accelerate_py.agent_supervisor.context.context_contracts import (
    ContextReference,
    ContextTier,
)
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.contracts import (
    ArtifactBindings,
    ArtifactState,
    EffectClass,
    FORBIDDEN_HOLE_TYPES,
    HoleType,
    ProcedureContractError,
    ProcedureHole,
    ProviderClass,
    parse_procedure_artifact,
)
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.hole_resolution import (
    DETERMINISTIC_PROVIDER_CLASSES,
    MODEL_PROVIDER_CLASSES,
    PROVIDER_ROUTE_ORDER,
    RESOLVER_REVISION,
    VALIDATOR_REVISION,
    CompiledHoleContext,
    HoleAction,
    HoleAttemptLog,
    HoleCandidate,
    HoleContextError,
    HoleProviderResult,
    HoleProviderStatus,
    HoleReason,
    HoleRequest,
    HoleResolution,
    HoleResolutionError,
    HoleResolutionValidator,
    HoleResolver,
    HoleTypeError,
    HoleValidationError,
    HoleValidationReceipt,
    HoleValidationStatus,
    IndependentHoleObservation,
    allowed_hole_types,
    assert_allowed_hole_type,
    forbidden_hole_types,
    route_provider_classes,
    scan_injection,
)


def _bindings() -> ArtifactBindings:
    return ArtifactBindings(
        repository_id="repo-main",
        repository_commit="commit-1",
        tree_id="tree-1",
        objective_id="PCPC-G000",
        task_id="PCPC-020",
        contract_revision="procedure-contracts-v1",
        policy_revision="authority-policy-v1",
        environment_id="python312-linux-lock1",
    )


def _hole(**changes: object) -> ProcedureHole:
    values: dict[str, object] = {
        "hole_id": "hole.classify-failure",
        "hole_type": HoleType.CLASSIFY_FAILURE,
        "input_schema_ref": "schema.failure-in",
        "output_schema_ref": "schema.failure-out",
        "allowed_provider_classes": (
            ProviderClass.EXACT_CACHE,
            ProviderClass.REMOTE_STANDARD_MODEL,
        ),
        "context_budget_bytes": 8_192,
        "authority_requirement_ids": ("authority.execute-tests",),
        "effect_classes": (EffectClass.OBSERVE, EffectClass.MODEL_REQUEST),
        "validation_observation_ids": ("observation.tests",),
        "fallback_step_id": "fallback-step",
        "maximum_attempts": 2,
    }
    values.update(changes)
    return ProcedureHole(**values)


def _request(**changes: object) -> HoleRequest:
    values: dict[str, object] = {
        "bindings": _bindings(),
        "hole": _hole(),
        "request_id": "hole-request-1",
        "input_payload": {"failure_class": "import-side-effect"},
        "context_reference_ids": ("evidence-failure",),
        "evidence_cids": ("evidence-failure",),
        "attempt_index": 1,
        "step_id": "step.hole",
        "context_tree_id": "tree-1",
    }
    values.update(changes)
    return HoleRequest(**values)


def _reference(**changes: object) -> ContextReference:
    values: dict[str, object] = {
        "reference_id": "evidence-failure",
        "kind": "hole-evidence",
        "tier": ContextTier.EVIDENCE,
        "referenced_content_id": "evidence-failure",
        "repository_id": "repo-main",
        "tree_id": "tree-1",
        "summary": "admitted failure signature",
        "byte_count": 64,
        "token_count": 16,
    }
    values.update(changes)
    return ContextReference(**values)


def _compiled() -> CompiledHoleContext:
    return CompiledHoleContext(
        capsule_id="capsule-1",
        receipt_cid="receipt-1",
        repository_id="repo-main",
        tree_id="tree-1",
        input_tokens": 32,
        evidence_bytes": 64,
        evidence_cids=("evidence-failure",),
        reference_ids=("evidence-failure",),
    )


@dataclass
class ScriptedProvider:
    provider_class: ProviderClass
    results: list[HoleProviderResult] = field(default_factory=list)
    calls: int = 0

    def propose(self, request: HoleRequest, compiled: CompiledHoleContext) -> HoleProviderResult:
        del request, compiled
        self.calls += 1
        if not self.results:
            return HoleProviderResult(status=HoleProviderStatus.MISS)
        return self.results.pop(0)


def _candidate_result(output: dict[str, object] | None = None) -> HoleProviderResult:
    return HoleProviderResult(
        status=HoleProviderStatus.CANDIDATE,
        output=output or {"label": "import-side-effect"},
        output_schema_ref="schema.failure-out",
    )


def _failure_result(signature: str = "schema-mismatch") -> HoleProviderResult:
    return HoleProviderResult(status=HoleProviderStatus.FAILURE, failure_signature=signature)


def _miss() -> HoleProviderResult:
    return HoleProviderResult(status=HoleProviderStatus.MISS)


def _resolve(
    request: HoleRequest | None = None,
    *,
    providers: list[ScriptedProvider] | None = None,
    resolver: HoleResolver | None = None,
    **kwargs: object,
) -> HoleResolution:
    active = resolver or HoleResolver()
    return active.resolve(
        request or _request(),
        providers=providers or [],
        context_references=kwargs.pop("context_references", (_reference(),)),
        **kwargs,
    )


def test_allowed_and_forbidden_hole_types_are_closed() -> None:
    assert {item.value for item in allowed_hole_types()} == {item.value for item in HoleType}
    assert forbidden_hole_types() == FORBIDDEN_HOLE_TYPES
    for hole_type in HoleType:
        assert assert_allowed_hole_type(hole_type) is hole_type
        assert assert_allowed_hole_type(hole_type.value) is hole_type
    for forbidden in FORBIDDEN_HOLE_TYPES:
        with pytest.raises(HoleTypeError, match="forbidden hole type"):
            assert_allowed_hole_type(forbidden)
    with pytest.raises(HoleTypeError, match="unknown hole type"):
        assert_allowed_hole_type("AUTHORITY_BY_PROMPT")


def test_hole_request_round_trip_and_generic_candidate_envelope() -> None:
    request = _request()
    decoded = HoleRequest.from_dict(request.to_dict())
    assert decoded == request
    assert decoded.content_id == request.content_id
    assert decoded.can_authorize is False
    artifact = request.to_artifact()
    parsed = parse_procedure_artifact(artifact.to_dict())
    assert parsed.state is ArtifactState.CANDIDATE
    assert parsed.facts["can_authorize"] is False
    payload = request.to_dict()
    payload["claim_completion"] = True
    with pytest.raises(ProcedureContractError, match="unsupported fields"):
        HoleRequest.from_dict(payload)


@pytest.mark.parametrize("hole_type", list(HoleType))
def test_every_allowed_typed_hole_can_call_an_approved_provider(hole_type: HoleType) -> None:
    provider = ScriptedProvider(ProviderClass.EXACT_CACHE, [_candidate_result()])
    resolution = _resolve(
        _request(
            hole=_hole(
                hole_type=hole_type,
                allowed_provider_classes=(ProviderClass.EXACT_CACHE,),
                effect_classes=(EffectClass.OBSERVE,),
            )
        ),
        providers=[provider],
    )
    assert resolution.action is HoleAction.CANDIDATE
    assert resolution.candidate is not None
    assert resolution.candidate.state is ArtifactState.CANDIDATE
    assert resolution.candidate.hole_type is hole_type
    assert resolution.provider_called is True
    assert provider.calls == 1


def test_unapproved_provider_is_not_called_even_when_installed() -> None:
    remote = ScriptedProvider(ProviderClass.REMOTE_STRONG_MODEL, [_candidate_result()])
    resolution = _resolve(
        _request(
            hole=_hole(allowed_provider_classes=(ProviderClass.EXACT_CACHE,)),
        ),
        providers=[remote],
    )
    assert remote.calls == 0
    assert resolution.action is HoleAction.FALLBACK
    assert resolution.reason_code is HoleReason.FALLBACK
    assert resolution.fallback_step_id == "fallback-step"
    assert resolution.provider_called is False


def test_prompt_cannot_select_a_provider_or_skip_the_route() -> None:
    remote = ScriptedProvider(ProviderClass.REMOTE_STANDARD_MODEL, [_candidate_result()])
    cache = ScriptedProvider(ProviderClass.EXACT_CACHE, [_candidate_result()])
    request = _request(preferred_provider_class=ProviderClass.REMOTE_STANDARD_MODEL)
    resolution = _resolve(request, providers=[remote, cache])
    assert resolution.action is HoleAction.REFUSE
    assert resolution.reason_code is HoleReason.PROMPT_SELECTED_PROVIDER
    assert remote.calls == 0
    assert cache.calls == 0


def test_model_calls_only_after_deterministic_cache_and_rule_routes() -> None:
    assert PROVIDER_ROUTE_ORDER[:3] == (
        ProviderClass.EXACT_CACHE,
        ProviderClass.DECLARATIVE_RULE,
        ProviderClass.DETERMINISTIC_CLASSIFIER,
    )
    assert set(PROVIDER_ROUTE_ORDER[:3]) == set(DETERMINISTIC_PROVIDER_CLASSES)
    cache = ScriptedProvider(ProviderClass.EXACT_CACHE, [_miss()])
    rule = ScriptedProvider(ProviderClass.DECLARATIVE_RULE, [_miss()])
    remote = ScriptedProvider(ProviderClass.REMOTE_STANDARD_MODEL, [_candidate_result()])
    # Reverse installation order must not change the sealed route.
    resolution = _resolve(
        _request(
            hole=_hole(
                allowed_provider_classes=(
                    ProviderClass.EXACT_CACHE,
                    ProviderClass.DECLARATIVE_RULE,
                    ProviderClass.REMOTE_STANDARD_MODEL,
                )
            )
        ),
        providers=[remote, rule, cache],
    )
    assert (cache.calls, rule.calls, remote.calls) == (1, 1, 1)
    assert resolution.attempted_provider_classes == (
        ProviderClass.EXACT_CACHE,
        ProviderClass.DECLARATIVE_RULE,
        ProviderClass.REMOTE_STANDARD_MODEL,
    )
    assert resolution.called_provider_class is ProviderClass.REMOTE_STANDARD_MODEL
    assert resolution.candidate is not None
    assert resolution.candidate.state is ArtifactState.CANDIDATE


def test_outputs_remain_candidates_until_independent_validation() -> None:
    provider = ScriptedProvider(ProviderClass.EXACT_CACHE, [_candidate_result()])
    resolution = _resolve(
        _request(hole=_hole(allowed_provider_classes=(ProviderClass.EXACT_CACHE,))),
        providers=[provider],
    )
    candidate = resolution.candidate
    assert candidate is not None
    assert candidate.state is ArtifactState.CANDIDATE
    assert candidate.can_authorize is False
    assert candidate.can_promote is False
    assert candidate.can_establish_completion is False
    assert candidate.can_suppress_validation is False
    with pytest.raises(HoleResolutionError, match="remain candidates"):
        HoleCandidate(
            bindings=_bindings(),
            request_cid=candidate.request_cid,
            hole_id=candidate.hole_id,
            hole_type=candidate.hole_type,
            provider_class=candidate.provider_class,
            output_schema_ref=candidate.output_schema_ref,
            output=dict(candidate.output),
            state=ArtifactState.VERIFIED,
        )

    validator = HoleResolutionValidator()
    refused = validator.validate(candidate, hole=_hole())
    assert refused.status is HoleValidationStatus.REFUSED
    assert refused.reason_code == HoleReason.VALIDATION_REQUIRED.value
    assert refused.accepted is False
    assert refused.state is ArtifactState.CANDIDATE

    accepted = validator.validate(
        candidate,
        hole=_hole(),
        observations=(
            IndependentHoleObservation(
                observation_id="observation.tests",
                producer_id="test-runner@1",
                admitted=True,
                evidence_cid="test-receipt-1",
            ),
        ),
        provider_id="exact-cache",
    )
    assert accepted.accepted is True
    assert accepted.state is ArtifactState.CANDIDATE
    assert accepted.independent is True
    assert accepted.can_authorize is False
    assert accepted.can_promote is False
    assert accepted.validator_revision == VALIDATOR_REVISION
    artifact = accepted.to_artifact()
    assert artifact.state is ArtifactState.CANDIDATE
    assert artifact.facts["can_suppress_validation"] is False


def test_provider_self_report_cannot_satisfy_validation() -> None:
    provider = ScriptedProvider(ProviderClass.EXACT_CACHE, [_candidate_result()])
    resolution = _resolve(
        _request(hole=_hole(allowed_provider_classes=(ProviderClass.EXACT_CACHE,))),
        providers=[provider],
    )
    candidate = resolution.candidate
    assert candidate is not None
    receipt = HoleResolutionValidator().validate(
        candidate,
        hole=_hole(),
        observations=(
            IndependentHoleObservation(
                observation_id="observation.tests",
                producer_id=ProviderClass.EXACT_CACHE.value,
                admitted=True,
                evidence_cid="self-report",
            ),
        ),
        provider_id=ProviderClass.EXACT_CACHE.value,
    )
    assert receipt.status is HoleValidationStatus.REFUSED
    assert receipt.reason_code == HoleReason.VALIDATION_REQUIRED.value


def test_identical_failure_suppresses_another_provider_call() -> None:
    provider = ScriptedProvider(
        ProviderClass.REMOTE_STANDARD_MODEL,
        [_failure_result("schema-mismatch"), _candidate_result()],
    )
    resolver = HoleResolver()
    request = _request(
        hole=_hole(
            allowed_provider_classes=(ProviderClass.REMOTE_STANDARD_MODEL,),
            maximum_attempts=2,
        )
    )
    first = _resolve(request, providers=[provider], resolver=resolver)
    assert first.action is HoleAction.FALLBACK
    assert provider.calls == 1
    second = _resolve(request, providers=[provider], resolver=resolver)
    assert second.action is HoleAction.SUPPRESS
    assert second.reason_code is HoleReason.IDENTICAL_FAILURE
    assert second.provider_called is False
    assert provider.calls == 1


def test_no_new_evidence_suppresses_another_model_call_until_evidence_arrives() -> None:
    provider = ScriptedProvider(
        ProviderClass.REMOTE_STANDARD_MODEL,
        [_failure_result("timeout"), _candidate_result()],
    )
    resolver = HoleResolver()
    hole = _hole(
        allowed_provider_classes=(ProviderClass.REMOTE_STANDARD_MODEL,),
        maximum_attempts=2,
    )
    first = _resolve(_request(hole=hole), providers=[provider], resolver=resolver)
    assert first.action is HoleAction.FALLBACK
    suppressed = _resolve(
        _request(hole=hole, request_id="hole-request-2"),
        providers=[provider],
        resolver=resolver,
    )
    assert suppressed.action is HoleAction.SUPPRESS
    assert suppressed.reason_code in {
        HoleReason.IDENTICAL_FAILURE,
        HoleReason.NO_NEW_EVIDENCE,
    }
    assert provider.calls == 1

    retried = _resolve(
        _request(
            hole=hole,
            request_id="hole-request-3",
            attempt_index=2,
            evidence_cids=("evidence-failure", "evidence-retry"),
        ),
        providers=[provider],
        resolver=resolver,
        context_references=(
            _reference(),
            _reference(
                reference_id="evidence-retry",
                referenced_content_id="evidence-retry",
                summary="new failure observation",
            ),
        ),
    )
    assert retried.action is HoleAction.CANDIDATE
    assert provider.calls == 2
    assert retried.candidate is not None
    assert retried.candidate.state is ArtifactState.CANDIDATE


def test_context_and_attempt_bounds_fail_closed() -> None:
    with pytest.raises(HoleResolutionError, match="attempt_index exceeds"):
        _request(attempt_index=3, hole=_hole(maximum_attempts=2))
    huge = _request(
        hole=_hole(
            allowed_provider_classes=(ProviderClass.EXACT_CACHE,),
            context_budget_bytes=32,
            effect_classes=(EffectClass.OBSERVE,),
        ),
        input_payload={"note": "n" * 40},
        context_reference_ids=(),
        evidence_cids=(),
        context_tree_id="tree-1",
    )
    resolution = _resolve(huge, providers=[], context_references=())
    assert resolution.action is HoleAction.REFUSE
    assert resolution.reason_code is HoleReason.CONTEXT_BUDGET_EXCEEDED


def test_stale_context_is_refused() -> None:
    with pytest.raises(HoleContextError, match="stale context"):
        _request(context_tree_id="tree-stale")
    provider = ScriptedProvider(ProviderClass.EXACT_CACHE, [_candidate_result()])
    stale_tree = _resolve(
        _request(hole=_hole(allowed_provider_classes=(ProviderClass.EXACT_CACHE,))),
        providers=[provider],
        current_tree_id="tree-other",
    )
    assert stale_tree.action is HoleAction.REFUSE
    assert stale_tree.reason_code is HoleReason.STALE_CONTEXT
    assert provider.calls == 0

    stale_reference = _resolve(
        _request(hole=_hole(allowed_provider_classes=(ProviderClass.EXACT_CACHE,))),
        providers=[provider],
        context_references=(_reference(tree_id="tree-other"),),
    )
    assert stale_reference.action is HoleAction.REFUSE
    assert stale_reference.reason_code is HoleReason.STALE_CONTEXT
    assert provider.calls == 0


def test_injection_and_authority_or_effect_escalation_are_refused() -> None:
    with pytest.raises(HoleResolutionError, match="effect classes"):
        _request(hole=_hole(effect_classes=(EffectClass.MERGE,)))
    with pytest.raises(HoleResolutionError, match="forbidden injection"):
        _request(input_payload={"authority_decision": "approve"})
    assert scan_injection({"disable_validation": True}) is not None
    provider = ScriptedProvider(
        ProviderClass.EXACT_CACHE,
        [_candidate_result({"authority_decision": True, "label": "ok"})],
    )
    resolution = _resolve(
        _request(hole=_hole(allowed_provider_classes=(ProviderClass.EXACT_CACHE,))),
        providers=[provider],
    )
    assert resolution.action is HoleAction.REFUSE
    assert resolution.reason_code is HoleReason.INJECTION
    assert resolution.can_authorize is False
    assert resolution.can_grant_authority is False
    assert resolution.can_promote is False


def test_effect_and_authority_flow_cannot_leave_the_hole_envelope() -> None:
    provider = ScriptedProvider(ProviderClass.EXACT_CACHE, [_candidate_result()])
    request = _request(
        hole=_hole(
            allowed_provider_classes=(ProviderClass.EXACT_CACHE,),
            effect_classes=(EffectClass.OBSERVE, EffectClass.MODEL_REQUEST),
            authority_requirement_ids=("authority.execute-tests",),
        )
    )
    resolution = _resolve(request, providers=[provider])
    assert set(request.hole.effect_classes) <= {EffectClass.OBSERVE, EffectClass.MODEL_REQUEST}
    assert resolution.can_authorize is False
    assert resolution.can_grant_authority is False
    assert resolution.can_establish_completion is False
    assert resolution.can_suppress_validation is False
    assert resolution.resolver_revision == RESOLVER_REVISION
    with pytest.raises(HoleResolutionError, match="cannot authorize"):
        HoleResolution(
            bindings=_bindings(),
            request_cid=resolution.request_cid,
            hole_id=resolution.hole_id,
            action=HoleAction.FALLBACK,
            reason_code=HoleReason.FALLBACK,
            fallback_step_id="fallback-step",
            can_authorize=True,
        )
    with pytest.raises(HoleValidationError, match="cannot authorize"):
        HoleValidationReceipt(
            bindings=_bindings(),
            candidate_cid="candidate-1",
            hole_id="hole.classify-failure",
            status=HoleValidationStatus.ACCEPTED,
            observation_ids=("observation.tests",),
            validator_id="validator",
            can_authorize=True,
        )
    with pytest.raises(HoleValidationError, match="cannot be produced by the provider"):
        HoleValidationReceipt(
            bindings=_bindings(),
            candidate_cid="candidate-1",
            hole_id="hole.classify-failure",
            status=HoleValidationStatus.ACCEPTED,
            observation_ids=("observation.tests",),
            validator_id="same-actor",
            provider_id="same-actor",
        )


def test_fallback_is_used_when_approved_providers_miss() -> None:
    cache = ScriptedProvider(ProviderClass.EXACT_CACHE, [_miss()])
    resolution = _resolve(
        _request(hole=_hole(allowed_provider_classes=(ProviderClass.EXACT_CACHE,))),
        providers=[cache],
    )
    assert resolution.action is HoleAction.FALLBACK
    assert resolution.reason_code is HoleReason.FALLBACK
    assert resolution.fallback_step_id == "fallback-step"
    assert resolution.candidate is None
    assert resolution.state is ArtifactState.CANDIDATE


def test_remote_provider_uses_provider_route_admission() -> None:
    remote = ScriptedProvider(ProviderClass.REMOTE_STANDARD_MODEL, [_candidate_result()])
    blocked = _resolve(
        _request(
            hole=_hole(allowed_provider_classes=(ProviderClass.REMOTE_STANDARD_MODEL,)),
        ),
        providers=[remote],
        provider_route_healthy=False,
    )
    assert blocked.action is HoleAction.FALLBACK
    assert blocked.reason_code is HoleReason.PROVIDER_ROUTE_UNAVAILABLE
    assert remote.calls == 0

    admitted = _resolve(
        _request(
            hole=_hole(allowed_provider_classes=(ProviderClass.REMOTE_STANDARD_MODEL,)),
        ),
        providers=[remote],
        provider_route_healthy=True,
    )
    assert admitted.action is HoleAction.CANDIDATE
    assert remote.calls == 1
    assert admitted.called_provider_class is ProviderClass.REMOTE_STANDARD_MODEL


def test_route_order_helper_is_deterministic_to_remote() -> None:
    ordered = route_provider_classes(
        (
            ProviderClass.REMOTE_STRONG_MODEL,
            ProviderClass.EXACT_CACHE,
            ProviderClass.HUMAN,
            ProviderClass.LOCAL_SMALL_MODEL,
        )
    )
    assert ordered == (
        ProviderClass.EXACT_CACHE,
        ProviderClass.LOCAL_SMALL_MODEL,
        ProviderClass.REMOTE_STRONG_MODEL,
        ProviderClass.HUMAN,
    )
    assert ProviderClass.REMOTE_STRONG_MODEL in MODEL_PROVIDER_CLASSES


def test_context_compiler_binds_current_tree_into_the_candidate() -> None:
    provider = ScriptedProvider(ProviderClass.EXACT_CACHE, [_candidate_result()])
    resolution = _resolve(
        _request(hole=_hole(allowed_provider_classes=(ProviderClass.EXACT_CACHE,))),
        providers=[provider],
    )
    assert resolution.compiled_context_cid
    assert resolution.candidate is not None
    assert resolution.candidate.compiled_context_cid == resolution.compiled_context_cid


def test_hole_resolution_round_trip_keeps_candidate_tier() -> None:
    provider = ScriptedProvider(ProviderClass.EXACT_CACHE, [_candidate_result()])
    resolution = _resolve(
        _request(hole=_hole(allowed_provider_classes=(ProviderClass.EXACT_CACHE,))),
        providers=[provider],
    )
    decoded = HoleResolution.from_dict(resolution.to_dict())
    assert decoded == resolution
    assert decoded.candidate is not None
    assert decoded.candidate.state is ArtifactState.CANDIDATE
    envelope = resolution.to_artifact()
    parsed = parse_procedure_artifact(envelope.to_dict())
    assert parsed.state is ArtifactState.CANDIDATE
    assert parsed.facts["can_authorize"] is False


def test_successful_model_call_requires_new_evidence_before_another_call() -> None:
    provider = ScriptedProvider(
        ProviderClass.REMOTE_STANDARD_MODEL,
        [_candidate_result(), _candidate_result()],
    )
    resolver = HoleResolver()
    hole = _hole(
        allowed_provider_classes=(ProviderClass.REMOTE_STANDARD_MODEL,),
        maximum_attempts=2,
    )
    first = _resolve(_request(hole=hole), providers=[provider], resolver=resolver)
    assert first.action is HoleAction.CANDIDATE
    assert provider.calls == 1
    suppressed = _resolve(
        _request(hole=hole, request_id="hole-request-2", attempt_index=2),
        providers=[provider],
        resolver=resolver,
    )
    assert suppressed.action is HoleAction.SUPPRESS
    assert suppressed.reason_code is HoleReason.NO_NEW_EVIDENCE
    assert suppressed.provider_called is False
    assert provider.calls == 1


def test_attempt_log_is_shared_across_resolver_calls() -> None:
    log = HoleAttemptLog()
    resolver = HoleResolver(attempt_log=log)
    provider = ScriptedProvider(
        ProviderClass.REMOTE_STANDARD_MODEL,
        [_failure_result("empty-output")],
    )
    request = _request(
        hole=_hole(allowed_provider_classes=(ProviderClass.REMOTE_STANDARD_MODEL,)),
    )
    _resolve(request, providers=[provider], resolver=resolver)
    assert log.records()
    assert log.records()[0].failure_signature == "empty-output"
    assert log.records()[0].model_called is True
