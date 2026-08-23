from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.contracts import (
    ArtifactBindings,
    ArtifactState,
    EffectClass,
    HoleType,
    ProviderClass,
)
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.distillation import (
    REQUIRED_CACHE_KEY_FIELDS,
    CorpusPartition,
    DeclarativeHoleRule,
    DeclarativeHoleRuleKind,
    DistillationAdmissionError,
    DistillationReason,
    ExactHoleCache,
    HeldOutResolverEvaluation,
    LocalHoleResolver,
)
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.hole_resolution import (
    LOCAL_HOLE_ROUTE_ORDER,
    PROVIDER_ROUTE_ORDER,
    CompiledHoleContext,
    HoleContextReference,
    HoleProviderOutcome,
    HoleProviderResult,
    HoleRequest,
    HoleResolutionAction,
    HoleResolutionReason,
    HoleResolutionValidator,
    HoleResolver,
    ProviderCapacitySnapshot,
    default_hole_context_compiler,
    provider_port_claims_authority,
)


def _bindings(**changes: object) -> ArtifactBindings:
    values: dict[str, object] = {
        "repository_id": "repo-main",
        "repository_commit": "commit-abc123",
        "tree_id": "tree-abc123",
        "objective_id": "PCPC-G030",
        "task_id": "PCPC-022",
        "contract_revision": "procedure-contracts-v1",
        "policy_revision": "authority-policy-v1",
        "environment_id": "python312-linux-lock1",
    }
    values.update(changes)
    return ArtifactBindings(**values)


def _reference(**changes: object) -> HoleContextReference:
    values: dict[str, object] = {
        "reference_id": "evidence.symbols",
        "content_id": "cid-symbols-1",
        "tree_id": "tree-abc123",
        "byte_count": 64,
        "token_count": 16,
        "required": True,
        "summary": "allowed-symbols",
    }
    values.update(changes)
    return HoleContextReference(**values)


def _output(selected: str = "pkg.mod:symbol_a") -> dict[str, str]:
    return {"schema_ref": "schema.symbol-out", "selected": selected}


def _request(**changes: object) -> HoleRequest:
    bindings = changes.get("bindings", _bindings())
    if not isinstance(bindings, ArtifactBindings):
        bindings = _bindings()
    values: dict[str, object] = {
        "bindings": bindings,
        "hole_id": "hole.select-symbol",
        "hole_type": HoleType.SELECT_ONE_OF_ALLOWED_SYMBOLS,
        "input_schema_ref": "schema.symbol-in",
        "output_schema_ref": "schema.symbol-out",
        "allowed_provider_classes": LOCAL_HOLE_ROUTE_ORDER,
        "context_budget_bytes": 32_768,
        "validation_observation_ids": ("observation.tests", "proof.hole-outcome"),
        "fallback_step_id": "step.fallback",
        "maximum_attempts": 4,
        "input_payload": {"allowed_values": ("pkg.mod:symbol_a", "pkg.mod:symbol_b")},
        "context_references": (_reference(tree_id=bindings.tree_id),),
        "authority_requirement_ids": ("authority.execute",),
        "effect_classes": (EffectClass.OBSERVE, EffectClass.MODEL_REQUEST),
        "token_budget": 4_096,
    }
    values.update(changes)
    return HoleRequest(**values)


def _capacity(*classes: ProviderClass) -> tuple[ProviderCapacitySnapshot, ...]:
    selected = classes or LOCAL_HOLE_ROUTE_ORDER
    return tuple(
        ProviderCapacitySnapshot(
            provider_class=item,
            available=True,
            remaining_calls=4,
            max_context_bytes=65_536,
            max_tokens=8_192,
            provider_id=f"provider.{item.value}",
        )
        for item in selected
    )


def _singleton_rule() -> DeclarativeHoleRule:
    return DeclarativeHoleRule(
        rule_id="rule.select-singleton",
        hole_type=HoleType.SELECT_ONE_OF_ALLOWED_SYMBOLS,
        kind=DeclarativeHoleRuleKind.SELECT_SINGLETON,
    )


def _failure_map_rule() -> DeclarativeHoleRule:
    return DeclarativeHoleRule(
        rule_id="rule.classify-import",
        hole_type=HoleType.CLASSIFY_FAILURE,
        kind=DeclarativeHoleRuleKind.CLOSED_MAP,
        input_field="failure_signature",
        output_field="failure_class",
        mapping={"import-side-effect": "missing-import"},
    )


class RecordingProvider:
    def __init__(self, result: object) -> None:
        self.calls = 0
        self.result = result

    def propose(self, request: HoleRequest, compiled: CompiledHoleContext) -> object:
        self.calls += 1
        if callable(self.result):
            return self.result(request, compiled)
        return self.result


class AuthorityClaimingProvider:
    can_skip_validation = True
    claims_correctness = True
    calls = 0

    def propose(self, request: HoleRequest, compiled: CompiledHoleContext) -> HoleProviderResult:
        self.calls += 1
        return HoleProviderResult(
            outcome=HoleProviderOutcome.PROPOSED,
            output=_output(),
        )


def _compiled(request: HoleRequest) -> CompiledHoleContext:
    return HoleResolver(
        default_hole_context_compiler(),
        current_tree_id=request.bindings.tree_id,
    ).compile_context(request)


def _local(
    *,
    rules: tuple[DeclarativeHoleRule, ...] | DeclarativeHoleRule = (),
    remote: RecordingProvider | None = None,
) -> LocalHoleResolver:
    return LocalHoleResolver(rules=rules, remote=remote)


def _train_cache_and_models(
    local: LocalHoleResolver,
    request: HoleRequest,
    compiled: CompiledHoleContext,
    output: dict[str, str],
) -> None:
    local.ingest(
        request,
        output,
        compiled=compiled,
        partition=CorpusPartition.TRAINING,
        accepted=True,
        example_id="ex.train.symbol-a",
        candidate_cid="cid-candidate-train",
        validation_cid="cid-validation-train",
        remember_cache=True,
    )


def test_route_hierarchy_is_exact_cache_rule_classifier_local_remote() -> None:
    assert LOCAL_HOLE_ROUTE_ORDER == (
        ProviderClass.EXACT_CACHE,
        ProviderClass.DECLARATIVE_RULE,
        ProviderClass.DETERMINISTIC_CLASSIFIER,
        ProviderClass.LOCAL_SMALL_MODEL,
        ProviderClass.REMOTE_STANDARD_MODEL,
    )
    assert PROVIDER_ROUTE_ORDER[:5] == LOCAL_HOLE_ROUTE_ORDER
    assert LocalHoleResolver.route_order == LOCAL_HOLE_ROUTE_ORDER
    for earlier, later in zip(LOCAL_HOLE_ROUTE_ORDER, LOCAL_HOLE_ROUTE_ORDER[1:]):
        assert PROVIDER_ROUTE_ORDER.index(earlier) < PROVIDER_ROUTE_ORDER.index(later)


def test_cache_hit_skips_rule_classifier_local_and_remote() -> None:
    remote = RecordingProvider(
        HoleProviderResult(outcome=HoleProviderOutcome.PROPOSED, output=_output())
    )
    local = _local(rules=_singleton_rule(), remote=remote)
    request = _request()
    resolver = local.to_hole_resolver(
        capacity=_capacity(),
        current_tree_id=request.bindings.tree_id,
        remote=remote,
    )
    compiled = resolver.compile_context(request)
    _train_cache_and_models(local, request, compiled, _output())
    resolution = resolver.resolve(request)

    assert resolution.action is HoleResolutionAction.PROPOSE
    assert resolution.provider_class == ProviderClass.EXACT_CACHE.value
    assert local.cache.calls == 1
    assert local.rule_provider.calls == 0
    assert local.classifier.calls == 0
    assert local.calls == 0
    assert remote.calls == 0
    assert resolution.remains_candidate is True
    assert resolution.can_skip_validation is False
    assert resolution.can_authorize is False


def test_rule_runs_after_exact_cache_miss() -> None:
    remote = RecordingProvider(
        HoleProviderResult(outcome=HoleProviderOutcome.PROPOSED, output=_output())
    )
    local = _local(rules=_singleton_rule(), remote=remote)
    request = _request(
        input_payload={"allowed_values": ("pkg.mod:symbol_a",)},
    )
    resolver = local.to_hole_resolver(
        capacity=_capacity(),
        current_tree_id=request.bindings.tree_id,
        remote=remote,
    )
    resolution = resolver.resolve(request)

    assert local.cache.calls == 1
    assert local.rule_provider.calls == 1
    assert local.classifier.calls == 0
    assert local.calls == 0
    assert remote.calls == 0
    assert resolution.action is HoleResolutionAction.PROPOSE
    assert resolution.provider_class == ProviderClass.DECLARATIVE_RULE.value
    candidate = resolver.last_candidate(request)
    assert candidate is not None
    assert candidate.output["selected"] == "pkg.mod:symbol_a"
    assert candidate.validated is False
    assert candidate.state is ArtifactState.CANDIDATE


def test_classifier_runs_after_rule_miss() -> None:
    remote = RecordingProvider(
        HoleProviderResult(outcome=HoleProviderOutcome.PROPOSED, output=_output())
    )
    local = _local(rules=_singleton_rule(), remote=remote)
    request = _request()
    local.classifier.ingest(request, _output(), partition=CorpusPartition.TRAINING)
    resolver = local.to_hole_resolver(
        capacity=_capacity(),
        current_tree_id=request.bindings.tree_id,
        remote=remote,
    )
    resolution = resolver.resolve(request)

    assert local.cache.calls == 1
    assert local.rule_provider.calls == 1
    assert local.classifier.calls == 1
    assert local.calls == 0
    assert remote.calls == 0
    assert resolution.provider_class == ProviderClass.DETERMINISTIC_CLASSIFIER.value
    assert resolution.action is HoleResolutionAction.PROPOSE


def test_local_model_runs_after_classifier_miss() -> None:
    remote = RecordingProvider(
        HoleProviderResult(outcome=HoleProviderOutcome.PROPOSED, output=_output())
    )
    local = _local(rules=_singleton_rule(), remote=remote)
    train = _request()
    compiled = _compiled(train)
    local.ingest(train, _output(), compiled=compiled, validation_cid="cid-validation-train")
    query = _request(
        hole_id="hole.select-symbol.local",
        input_payload={"allowed_values": ("pkg.mod:symbol_a", "pkg.mod:symbol_c")},
        context_references=(_reference(content_id="cid-symbols-local"),),
    )
    resolver = local.to_hole_resolver(
        capacity=_capacity(),
        current_tree_id=query.bindings.tree_id,
        remote=remote,
    )
    resolution = resolver.resolve(query)

    assert local.cache.calls == 1
    assert local.rule_provider.calls == 1
    assert local.classifier.calls == 1
    assert local.calls == 1
    assert remote.calls == 0
    assert resolution.provider_class == ProviderClass.LOCAL_SMALL_MODEL.value
    assert resolution.action is HoleResolutionAction.PROPOSE
    assert local.last_confidence_millis == 1000


def test_remote_runs_after_local_model_miss() -> None:
    remote = RecordingProvider(
        HoleProviderResult(outcome=HoleProviderOutcome.PROPOSED, output=_output())
    )
    local = _local(rules=_singleton_rule(), remote=remote)
    request = _request(
        allowed_provider_classes=(
            ProviderClass.LOCAL_SMALL_MODEL,
            ProviderClass.REMOTE_STANDARD_MODEL,
        )
    )
    resolver = local.to_hole_resolver(
        capacity=_capacity(ProviderClass.LOCAL_SMALL_MODEL, ProviderClass.REMOTE_STANDARD_MODEL),
        current_tree_id=request.bindings.tree_id,
        remote=remote,
    )
    resolution = resolver.resolve(request)

    assert local.calls == 1
    assert remote.calls == 1
    assert resolution.provider_class == ProviderClass.REMOTE_STANDARD_MODEL.value
    assert resolution.action is HoleResolutionAction.PROPOSE


@pytest.mark.parametrize(
    "field,value",
    [
        ("repository_id", "repo-other"),
        ("repository_commit", "commit-other"),
        ("tree_id", "tree-other"),
        ("objective_id", "PCPC-G999"),
        ("task_id", "PCPC-099"),
        ("contract_revision", "procedure-contracts-v2"),
        ("policy_revision", "authority-policy-v2"),
        ("environment_id", "python312-linux-lock2"),
    ],
)
def test_cache_key_changes_with_every_binding_dimension(field: str, value: str) -> None:
    cache = ExactHoleCache()
    request = _request()
    compiled = _compiled(request)
    cache.remember(
        request,
        _output(),
        compiled=compiled,
        validation_cid="cid-validation-train",
    )
    mutated = _request(bindings=_bindings(**{field: value}))
    mutated_compiled = HoleResolver(
        default_hole_context_compiler(),
        current_tree_id=mutated.bindings.tree_id,
    ).compile_context(mutated)
    assert cache.cache_key(request, compiled) != cache.cache_key(mutated, mutated_compiled)
    assert cache.lookup(mutated, mutated_compiled) is None
    result = cache.propose(mutated, mutated_compiled)
    assert result.outcome is HoleProviderOutcome.MISSED


def test_cache_key_changes_with_payload_schema_context_and_observations() -> None:
    cache = ExactHoleCache()
    request = _request()
    compiled = _compiled(request)
    original = cache.remember(
        request,
        _output(),
        compiled=compiled,
        validation_cid="cid-validation-train",
    )
    payload = _request(
        input_payload={"allowed_values": ("pkg.mod:symbol_a", "pkg.mod:symbol_z")}
    )
    schema = _request(output_schema_ref="schema.symbol-out.v2")
    context = _request(context_references=(_reference(content_id="cid-symbols-2"),))
    observations = _request(validation_observation_ids=("observation.other", "proof.hole-outcome"))
    hole_type_change = _request(
        hole_type=HoleType.CHOOSE_APPROVED_REPAIR_TEMPLATE,
        input_schema_ref="schema.template-in",
        output_schema_ref="schema.template-out",
        input_payload={"template_ids": ("template.import-purity", "template.other")},
    )
    for mutated in (payload, schema, context, observations, hole_type_change):
        mutated_compiled = _compiled(mutated) if mutated.bindings.tree_id == "tree-abc123" else compiled
        assert cache.cache_key(mutated, mutated_compiled) != original
        assert cache.lookup(mutated, mutated_compiled) is None
    assert set(REQUIRED_CACHE_KEY_FIELDS) == {
        "repository_id",
        "repository_commit",
        "tree_id",
        "objective_id",
        "task_id",
        "contract_revision",
        "policy_revision",
        "environment_id",
        "hole_type",
        "input_schema_ref",
        "output_schema_ref",
        "input_payload",
        "context_reference_ids",
        "context_content_ids",
        "evidence_fingerprint",
        "validation_observation_ids",
    }


def test_incomplete_cache_key_is_refused() -> None:
    cache = ExactHoleCache()
    request = _request(context_references=())
    with pytest.raises(DistillationAdmissionError, match="identity dimensions") as caught:
        cache.cache_key(request)
    assert caught.value.reason_code is DistillationReason.INCOMPLETE_CACHE_KEY
    compiled = _compiled(_request())
    with pytest.raises(DistillationAdmissionError, match="validation provenance") as missing:
        cache.remember(
            _request(),
            _output(),
            compiled=compiled,
            validation_cid="",
        )
    assert missing.value.reason_code is DistillationReason.MISSING_VALIDATION


def test_rule_is_deterministic_for_identical_requests() -> None:
    rule = _singleton_rule()
    first = _request(input_payload={"allowed_values": ("pkg.mod:symbol_a",)})
    second = _request(input_payload={"allowed_values": ("pkg.mod:symbol_a",)})
    compiled = _compiled(first)
    first_output = rule.match_output(first)
    second_output = rule.match_output(second)
    assert first_output == second_output
    assert first_output is not None
    assert first_output["selected"] == "pkg.mod:symbol_a"
    first_result = rule.propose(first, compiled)
    second_result = rule.propose(second, compiled)
    assert first_result.outcome is HoleProviderOutcome.PROPOSED
    assert first_result.output == second_result.output
    other = _request(input_payload={"allowed_values": ("pkg.mod:symbol_b",)})
    other_output = rule.match_output(other)
    assert other_output is not None
    assert other_output["selected"] == "pkg.mod:symbol_b"
    with pytest.raises(FrozenInstanceError):
        rule.kind = DeclarativeHoleRuleKind.EXACT_OUTPUT  # type: ignore[misc]


def test_rule_bundle_is_order_independent_and_ambiguous_closed() -> None:
    first = DeclarativeHoleRule(
        rule_id="rule.b",
        hole_type=HoleType.SELECT_ONE_OF_ALLOWED_SYMBOLS,
        kind=DeclarativeHoleRuleKind.EXACT_OUTPUT,
        match={"allowed_values_count": 2},
        output={"selected": "pkg.mod:symbol_a"},
    )
    second = DeclarativeHoleRule(
        rule_id="rule.a",
        hole_type=HoleType.SELECT_ONE_OF_ALLOWED_SYMBOLS,
        kind=DeclarativeHoleRuleKind.EXACT_OUTPUT,
        match={"allowed_values_count": 2},
        output={"selected": "pkg.mod:symbol_a"},
    )
    local_left = _local(rules=(first, second))
    local_right = _local(rules=(second, first))
    request = _request()
    compiled = _compiled(request)
    left = local_left.rule_provider.propose(request, compiled)
    right = local_right.rule_provider.propose(request, compiled)
    assert left.outcome is HoleProviderOutcome.MISSED
    assert left.failure_code == "rule-ambiguous"
    assert right.outcome is HoleProviderOutcome.MISSED
    assert right.failure_code == "rule-ambiguous"


def test_closed_map_rule_is_deterministic() -> None:
    rule = _failure_map_rule()
    request = _request(
        hole_id="hole.classify",
        hole_type=HoleType.CLASSIFY_FAILURE,
        input_schema_ref="schema.failure-in",
        output_schema_ref="schema.failure-out",
        input_payload={"failure_signature": "import-side-effect"},
        allowed_provider_classes=(ProviderClass.DECLARATIVE_RULE,),
    )
    compiled = _compiled(request)
    first = rule.propose(request, compiled)
    second = rule.propose(request, compiled)
    assert first.output == second.output
    assert first.output["failure_class"] == "missing-import"
    miss = rule.propose(
        _request(
            hole_id="hole.classify.other",
            hole_type=HoleType.CLASSIFY_FAILURE,
            input_schema_ref="schema.failure-in",
            output_schema_ref="schema.failure-out",
            input_payload={"failure_signature": "unknown-signature"},
            allowed_provider_classes=(ProviderClass.DECLARATIVE_RULE,),
        ),
        compiled,
    )
    assert miss.outcome is HoleProviderOutcome.MISSED


def test_held_out_examples_cannot_train_resolvers() -> None:
    local = _local()
    request = _request()
    compiled = _compiled(request)
    for partition in (
        CorpusPartition.HELD_OUT,
        CorpusPartition.NEGATIVE,
        CorpusPartition.BOUNDARY,
        CorpusPartition.ADVERSARIAL,
    ):
        with pytest.raises(DistillationAdmissionError, match="cannot train") as caught:
            local.ingest(
                request,
                _output(),
                compiled=compiled,
                partition=partition,
                validation_cid="cid-validation-heldout",
            )
        assert caught.value.reason_code is DistillationReason.HELD_OUT_TRAINING_REJECTED
        with pytest.raises(DistillationAdmissionError, match="cannot train") as cache_caught:
            local.cache.remember(
                request,
                _output(),
                compiled=compiled,
                partition=partition,
                validation_cid="cid-validation-heldout",
            )
        assert cache_caught.value.reason_code is DistillationReason.HELD_OUT_TRAINING_REJECTED
        with pytest.raises(DistillationAdmissionError, match="cannot train") as classifier_caught:
            local.classifier.ingest(request, _output(), partition=partition)
        assert classifier_caught.value.reason_code is DistillationReason.HELD_OUT_TRAINING_REJECTED


def test_held_out_evaluation_fails_closed_and_does_not_claim_correctness() -> None:
    local = _local()
    train = _request()
    compiled = _compiled(train)
    local.ingest(
        train,
        _output("pkg.mod:symbol_a"),
        compiled=compiled,
        validation_cid="cid-validation-train",
        remember_cache=True,
    )
    held_out = _request(
        hole_id="hole.select-symbol.heldout",
        input_payload={"allowed_values": ("pkg.mod:symbol_c", "pkg.mod:symbol_d")},
        context_references=(_reference(content_id="cid-symbols-heldout"),),
    )
    held_compiled = _compiled(held_out)
    evaluation = local.evaluate_held_out(
        ((held_out, held_compiled, _output("pkg.mod:symbol_c")),)
    )
    assert evaluation.evaluated_count == 1
    assert evaluation.missed_count == 1
    assert evaluation.matched_count == 0
    assert evaluation.proposed_count == 0
    assert evaluation.claims_correctness is False
    assert evaluation.can_skip_validation is False
    assert evaluation.can_authorize is False
    assert evaluation.accuracy_is_authority is False
    assert local.cache.lookup(held_out, held_compiled) is None
    resolver = local.to_hole_resolver(
        capacity=_capacity(),
        current_tree_id=held_out.bindings.tree_id,
    )
    held_resolution = resolver.resolve(held_out)
    assert held_resolution.action is HoleResolutionAction.FALLBACK
    assert resolver.last_candidate(held_out) is None
    assert held_resolution.can_skip_validation is False
    assert held_resolution.remains_candidate is True
    with pytest.raises(DistillationAdmissionError, match="cannot claim correctness"):
        HeldOutResolverEvaluation(
            evaluated_count=1,
            proposed_count=1,
            missed_count=0,
            matched_count=1,
            claims_correctness=True,
        )


def test_confidence_is_not_authority_and_cannot_skip_validation() -> None:
    local = _local()
    request = _request(allowed_provider_classes=(ProviderClass.LOCAL_SMALL_MODEL,))
    compiled = _compiled(request)
    local.ingest(
        request,
        _output(),
        compiled=compiled,
        validation_cid="cid-validation-train",
    )
    proposed = local.propose(request, compiled)
    assert proposed.outcome is HoleProviderOutcome.PROPOSED
    assert local.last_confidence_millis == 1000
    assert local.can_skip_validation is False
    assert local.claims_correctness is False
    assert "skip_validation" not in proposed.output
    assert "claim_correctness" not in proposed.output
    resolver = local.to_hole_resolver(
        capacity=_capacity(ProviderClass.LOCAL_SMALL_MODEL),
        current_tree_id=request.bindings.tree_id,
    )
    resolution = resolver.resolve(request)
    candidate = resolver.last_candidate(request)
    assert candidate is not None
    validator = HoleResolutionValidator(current_tree_id=request.bindings.tree_id)
    pending = validator.validate_candidate(request, candidate)
    assert pending.accepted is False
    assert pending.reason_code is HoleResolutionReason.VALIDATION_REQUIRED
    assert pending.remains_candidate is True
    assert pending.can_authorize is False
    assert resolution.can_skip_validation is False
    assert resolution.remains_candidate is True
    admitted = validator.validate_candidate(
        request,
        candidate,
        observations=("observation.tests", "proof.hole-outcome"),
    )
    assert admitted.accepted is True
    assert admitted.remains_candidate is True
    assert admitted.can_authorize is False
    assert candidate.validated is False


def test_no_route_suppresses_validation_or_claims_correctness() -> None:
    remote = RecordingProvider(
        HoleProviderResult(outcome=HoleProviderOutcome.PROPOSED, output=_output())
    )
    local = _local(rules=_singleton_rule(), remote=remote)
    request = _request()
    resolver = local.to_hole_resolver(
        capacity=_capacity(),
        current_tree_id=request.bindings.tree_id,
        remote=remote,
    )
    compiled = resolver.compile_context(request)
    _train_cache_and_models(local, request, compiled, _output())
    for port in local.provider_ports().values():
        assert provider_port_claims_authority(port) is False
        assert getattr(port, "can_skip_validation", False) is False
        assert getattr(port, "claims_correctness", False) is False
        assert getattr(port, "can_authorize", False) is False
    resolution = resolver.resolve(request)
    candidate = resolver.last_candidate(request)
    assert candidate is not None
    validator = HoleResolutionValidator(current_tree_id=request.bindings.tree_id)
    pending = validator.validate_candidate(request, candidate)
    assert pending.accepted is False
    assert pending.reason_code is HoleResolutionReason.VALIDATION_REQUIRED
    assert resolution.can_skip_validation is False
    assert resolution.can_grant_authority is False
    assert resolution.can_promote is False
    assert candidate.state is ArtifactState.CANDIDATE
    assert candidate.validated is False


def test_authority_claiming_provider_is_refused_before_proposal() -> None:
    evil = AuthorityClaimingProvider()
    resolver = HoleResolver(
        default_hole_context_compiler(),
        providers={ProviderClass.EXACT_CACHE: evil},
        capacity=_capacity(ProviderClass.EXACT_CACHE),
        current_tree_id="tree-abc123",
    )
    resolution = resolver.resolve(
        _request(allowed_provider_classes=(ProviderClass.EXACT_CACHE,))
    )
    assert evil.calls == 0
    assert resolution.action is HoleResolutionAction.REFUSE
    assert resolution.reason_code is HoleResolutionReason.AUTHORITY_FLOW_REJECTED
    assert resolution.can_skip_validation is False


def test_classifier_ambiguous_training_is_a_miss() -> None:
    local = _local()
    request = _request()
    compiled = _compiled(request)
    local.classifier.ingest(request, _output("pkg.mod:symbol_a"))
    local.classifier.ingest(request, _output("pkg.mod:symbol_b"))
    result = local.classifier.propose(request, compiled)
    assert result.outcome is HoleProviderOutcome.MISSED
    assert result.failure_code == "classifier-ambiguous"


def test_local_resolver_ports_preserve_route_order_for_hole_resolver() -> None:
    local = _local(rules=_singleton_rule())
    ports = local.provider_ports()
    assert tuple(item for item in LOCAL_HOLE_ROUTE_ORDER if item in ports) == (
        ProviderClass.EXACT_CACHE,
        ProviderClass.DECLARATIVE_RULE,
        ProviderClass.DETERMINISTIC_CLASSIFIER,
        ProviderClass.LOCAL_SMALL_MODEL,
    )
    assert list(ports) == [
        ProviderClass.EXACT_CACHE,
        ProviderClass.DECLARATIVE_RULE,
        ProviderClass.DETERMINISTIC_CLASSIFIER,
        ProviderClass.LOCAL_SMALL_MODEL,
    ]
