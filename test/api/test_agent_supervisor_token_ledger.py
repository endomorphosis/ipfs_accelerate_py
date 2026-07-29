from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.self_improvement.supervisor_token_ledger import (
    ACCEPTED_CRITERION_TOKEN_GOAL_ID,
    ACCEPTED_CRITERION_TOKEN_REQUIREMENT_ID,
    TOKEN_LEDGER_AUTHORIZES_USAGE,
    TOKEN_LEDGER_IS_COMPLETION_EVIDENCE,
    TOKEN_LEDGER_IS_CORRECTNESS_EVIDENCE,
    TOKEN_LEDGER_REWRITES_PROVIDER_SETTLEMENT,
    CacheDecision,
    FallbackTokenizerCalibration,
    ProviderModelEnvelope,
    ProviderTokenUsage,
    SupervisorTokenLedger,
    TerminalCriterionAttribution,
    TerminalDisposition,
    TokenAttribution,
    TokenLedgerValidationError,
    TokenizerCalibrationSample,
    UsageSource,
    ValidationResult,
    adapt_efficiency_metrics_from_reconciled_events,
    adapt_efficiency_receipt,
    calibrate_fallback_tokenizer,
    consume_reconciled_endpoint_events_exactly_once,
    provider_usage_from_reconciled_endpoint_event,
    token_ledger_authority_bounds,
)
from ipfs_accelerate_py.agent_supervisor.self_improvement.supervisor_efficiency_metrics import (
    build_efficiency_baseline_fixtures,
)
from ipfs_accelerate_py.agent_supervisor.self_improvement.supervisor_v2_contracts import (
    AuthorityClass,
    ResultBinding,
    SemanticDependencyIdentity,
    StageEvent,
    StageEventKind,
)
from ipfs_accelerate_py.endpoint_usage import (
    EndpointUsageScope,
    ProtocolKind,
    UsageEvent,
    UsageEventKind,
    UsageVector,
    credential_configuration_pseudonym,
    stable_id,
)


def _binding(*, task_id: str = "ASI-094", tree_id: str = "tree:ledger") -> ResultBinding:
    return ResultBinding(
        repository_id="repository:supervisor",
        tree_id=tree_id,
        objective_id="ASI-G210",
        objective_revision="objective:g210@1",
        task_id=task_id,
        task_revision=f"{task_id}@1",
        policy_id="policy:token-ledger",
        policy_revision="policy:token-ledger@1",
        producer_id="producer:token-ledger",
        producer_revision="producer:token-ledger@1",
        capability_id="capability:provider-accounting",
        capability_revision="capability:provider-accounting@1",
        environment_id="environment:test",
        environment_revision="environment:test@1",
        semantic_dependencies=(
            SemanticDependencyIdentity(
                namespace="repository",
                key="source-tree",
                revision=tree_id,
                digest="sha256:" + "a" * 64,
            ),
        ),
    )


def _envelope(model_id: str = "model:reasoner") -> ProviderModelEnvelope:
    return ProviderModelEnvelope(
        provider_id="provider:example",
        model_id=model_id,
        model_revision=f"{model_id}@2026-07",
        tokenizer_id="tokenizer:provider-native",
        envelope_revision="envelope:2026-07",
        max_context_tokens=16_384,
    )


def _event(
    binding: ResultBinding,
    *,
    stage: str,
    attempt: int,
    kind: StageEventKind,
    sequence: int = 0,
) -> StageEvent:
    return StageEvent(
        binding=binding,
        stage=stage,
        attempt=attempt,
        sequence=sequence,
        kind=kind,
        authority=AuthorityClass.VALIDATION,
        occurred_at=f"2026-07-26T12:00:{attempt:02d}.000000Z",
        reason_code="" if kind is StageEventKind.COMPLETED else "attempt-ended",
    )


def _ledger() -> SupervisorTokenLedger:
    binding = _binding()
    envelope = _envelope()
    calibration = calibrate_fallback_tokenizer(
        envelope,
        (
            TokenizerCalibrationSample("sample:a", utf8_bytes=40, provider_tokens=10),
            TokenizerCalibrationSample("sample:b", utf8_bytes=80, provider_tokens=20),
        ),
        calibration_revision="calibration:1",
    )
    failed = _event(
        binding,
        stage="inference",
        attempt=1,
        kind=StageEventKind.FAILED,
    )
    accepted = _event(
        binding,
        stage="inference",
        attempt=2,
        kind=StageEventKind.COMPLETED,
    )
    abandoned = _event(
        binding,
        stage="analysis",
        attempt=1,
        kind=StageEventKind.CANCELLED,
    )
    rejected_terminal = TerminalCriterionAttribution(
        binding=binding,
        terminal_event_id=failed.event_id,
        criterion_id="criterion:provider-attribution",
        disposition=TerminalDisposition.REJECTED,
        validation_result=ValidationResult.FAILED,
        reason_code="malformed-provider-output",
    )
    accepted_terminal = TerminalCriterionAttribution(
        binding=binding,
        terminal_event_id=accepted.event_id,
        criterion_id="criterion:provider-attribution",
        disposition=TerminalDisposition.ACCEPTED,
        validation_result=ValidationResult.PASSED,
        evidence_gain=3,
    )
    abandoned_terminal = TerminalCriterionAttribution(
        binding=binding,
        terminal_event_id=abandoned.event_id,
        criterion_id="criterion:optional-analysis",
        disposition=TerminalDisposition.ABANDONED,
        validation_result=ValidationResult.NOT_RUN,
        reason_code="provider-unavailable",
    )
    failed_usage = ProviderTokenUsage(
        measurement_id="request:failed",
        envelope=envelope,
        source=UsageSource.PROVIDER_NATIVE,
        input_tokens=100,
        output_tokens=20,
        tool_tokens=5,
        failed_attempt_tokens=125,
        cost_microunits=1_000,
    )
    accepted_usage = ProviderTokenUsage(
        measurement_id="request:accepted",
        envelope=envelope,
        source=UsageSource.PROVIDER_NATIVE,
        input_tokens=60,
        output_tokens=20,
        reused_tokens=30,
        speculative_tokens=5,
        tool_tokens=10,
        retry_tokens=90,
        cost_microunits=500,
    )
    fallback_usage = ProviderTokenUsage(
        measurement_id="request:fallback",
        envelope=envelope,
        source=UsageSource.CALIBRATED_FALLBACK,
        input_tokens=8,
        output_tokens=2,
        failed_attempt_tokens=10,
        cost_microunits=25,
        calibration_id=calibration.calibration_id,
    )
    return SupervisorTokenLedger(
        binding=binding,
        lifecycle_events=(failed, accepted, abandoned),
        terminal_attributions=(
            rejected_terminal,
            accepted_terminal,
            abandoned_terminal,
        ),
        attributions=(
            TokenAttribution(
                binding=binding,
                event_id=failed.event_id,
                stage=failed.stage,
                attempt=1,
                context_id="context:initial",
                cache_decision=CacheDecision.MISS,
                validation_result=ValidationResult.FAILED,
                terminal_attribution_id=(
                    rejected_terminal.terminal_attribution_id
                ),
                usage=failed_usage,
            ),
            TokenAttribution(
                binding=binding,
                event_id=accepted.event_id,
                stage=accepted.stage,
                attempt=2,
                context_id="context:retry-delta",
                cache_decision=CacheDecision.HIT,
                validation_result=ValidationResult.PASSED,
                terminal_attribution_id=(
                    accepted_terminal.terminal_attribution_id
                ),
                usage=accepted_usage,
            ),
            TokenAttribution(
                binding=binding,
                event_id=abandoned.event_id,
                stage=abandoned.stage,
                attempt=1,
                context_id="context:analysis",
                cache_decision=CacheDecision.BYPASS,
                validation_result=ValidationResult.NOT_RUN,
                terminal_attribution_id=(
                    abandoned_terminal.terminal_attribution_id
                ),
                usage=fallback_usage,
            ),
        ),
        calibrations=(calibration,),
    )


def test_ledger_attributes_every_native_counter_and_charges_failed_work() -> None:
    ledger = _ledger()
    report = ledger.report

    assert ACCEPTED_CRITERION_TOKEN_REQUIREMENT_ID.isdecimal()
    assert ACCEPTED_CRITERION_TOKEN_GOAL_ID == "ASI-G210"
    assert report.lifecycle_event_count == report.attribution_count == 3
    assert report.input_tokens == 168
    assert report.output_tokens == 42
    assert report.reused_tokens == 30
    assert report.speculative_tokens == 5
    assert report.tool_tokens == 15
    assert report.retry_tokens == 90
    assert report.failed_attempt_tokens == 135
    assert report.provider_native_tokens == 215
    assert report.fallback_tokens == 10
    assert report.rejected_tokens == 125
    assert report.abandoned_tokens == 10
    assert report.total_tokens == 225
    assert report.total_cost_microunits == 1_525
    assert report.accepted_criterion_count == 1
    assert report.cost_per_accepted_criterion_microunits == 1_525
    assert report.tokens_per_accepted_criterion == 225
    assert report.accepted_evidence_gain == 3
    assert report.evidence_gain_per_thousand_tokens == pytest.approx(
        3 * 1_000 / 225
    )

    by_id = {item.criterion_id: item for item in report.criterion_costs}
    assert by_id["criterion:provider-attribution"].accepted
    assert by_id["criterion:provider-attribution"].attempt_count == 2
    assert by_id["criterion:provider-attribution"].total_tokens == 215
    assert not by_id["criterion:optional-analysis"].accepted
    assert by_id["criterion:optional-analysis"].total_tokens == 10


def test_calibration_is_exactly_scoped_and_replayable_without_text() -> None:
    envelope = _envelope()
    calibration = FallbackTokenizerCalibration(
        envelope=envelope,
        calibration_revision="calibration:weighted",
        samples=(
            TokenizerCalibrationSample("one", 9, 3),
            TokenizerCalibrationSample("two", 12, 4),
        ),
    )

    assert calibration.sample_count == 2
    assert calibration.token_numerator == 7
    assert calibration.byte_denominator == 21
    assert calibration.estimate_text("x" * 10) == 4
    assert calibration.maximum_absolute_error_bps <= 2_500
    assert calibration.supports(envelope)
    assert not calibration.supports(_envelope("model:foreign"))
    payload = calibration.to_dict()
    assert all(
        set(sample) <= {
            "schema",
            "contract_version",
            "sample_id",
            "utf8_bytes",
            "provider_tokens",
            "content_id",
        }
        for sample in payload["samples"]
    )
    assert (
        FallbackTokenizerCalibration.from_json(calibration.to_json())
        == calibration
    )


def test_contracts_are_immutable_content_addressed_and_round_trip() -> None:
    ledger = _ledger()
    payload = ledger.to_dict(include_ledger_id=True)
    restored = SupervisorTokenLedger.from_dict(payload)

    assert restored == ledger
    assert restored.ledger_id == ledger.ledger_id
    assert restored.report.content_id == ledger.report.content_id
    with pytest.raises(FrozenInstanceError):
        ledger.binding = _binding(task_id="foreign")  # type: ignore[misc]

    tampered = json.loads(ledger.to_json())
    tampered["report"]["input_tokens"] += 1
    with pytest.raises(
        TokenLedgerValidationError, match="report does not reconcile"
    ):
        SupervisorTokenLedger.from_dict(tampered)

    unknown = ledger.to_dict()
    unknown["prompt"] = "must never cross the ledger"
    with pytest.raises(TokenLedgerValidationError, match="unsupported fields"):
        SupervisorTokenLedger.from_dict(unknown)


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("missing", "exactly once"),
        ("duplicated", "exactly once"),
        ("duplicate_measurement", "duplicated measurements"),
        ("terminally_unattributed", "terminally unattributed"),
        ("foreign_binding", "foreign-bound"),
        ("unused_terminal", "unused records"),
    ],
)
def test_reconciliation_rejects_missing_duplicated_and_foreign_usage(
    mutation: str, message: str
) -> None:
    ledger = _ledger()
    events = list(ledger.lifecycle_events)
    terminals = list(ledger.terminal_attributions)
    attributions = list(ledger.attributions)

    if mutation == "missing":
        attributions.pop()
    elif mutation == "duplicated":
        attributions.append(attributions[0])
    elif mutation == "duplicate_measurement":
        attributions[1] = replace(
            attributions[1],
            usage=replace(
                attributions[1].usage,
                measurement_id=attributions[0].usage.measurement_id,
            ),
        )
    elif mutation == "terminally_unattributed":
        attributions[0] = replace(
            attributions[0], terminal_attribution_id="terminal:missing"
        )
    elif mutation == "foreign_binding":
        attributions[0] = replace(
            attributions[0], binding=_binding(task_id="ASI-foreign")
        )
    elif mutation == "unused_terminal":
        terminals.append(
            replace(
                terminals[-1],
                criterion_id="criterion:unused",
                reason_code="unused-terminal",
            )
        )

    with pytest.raises(TokenLedgerValidationError, match=message):
        SupervisorTokenLedger(
            binding=ledger.binding,
            lifecycle_events=tuple(events),
            terminal_attributions=tuple(terminals),
            attributions=tuple(attributions),
            calibrations=ledger.calibrations,
        )


def test_rejects_negative_overlapping_and_misclassified_counters() -> None:
    envelope = _envelope()
    with pytest.raises(TokenLedgerValidationError, match="input_tokens"):
        ProviderTokenUsage(
            measurement_id="negative",
            envelope=envelope,
            source=UsageSource.PROVIDER_NATIVE,
            input_tokens=-1,
        )
    with pytest.raises(TokenLedgerValidationError, match="reused_tokens"):
        ProviderTokenUsage(
            measurement_id="reuse",
            envelope=envelope,
            source=UsageSource.PROVIDER_NATIVE,
            input_tokens=1,
            reused_tokens=2,
        )
    with pytest.raises(TokenLedgerValidationError, match="speculative_tokens"):
        ProviderTokenUsage(
            measurement_id="speculative",
            envelope=envelope,
            source=UsageSource.PROVIDER_NATIVE,
            output_tokens=1,
            speculative_tokens=2,
        )
    with pytest.raises(TokenLedgerValidationError, match="calibration_id"):
        ProviderTokenUsage(
            measurement_id="uncalibrated",
            envelope=envelope,
            source=UsageSource.CALIBRATED_FALLBACK,
            input_tokens=1,
        )

    ledger = _ledger()
    accepted = next(
        item
        for item in ledger.attributions
        if item.validation_result is ValidationResult.PASSED
    )
    with pytest.raises(
        TokenLedgerValidationError, match="failed-attempt.*incomplete"
    ):
        SupervisorTokenLedger(
            binding=ledger.binding,
            lifecycle_events=ledger.lifecycle_events,
            terminal_attributions=ledger.terminal_attributions,
            attributions=tuple(
                replace(
                    item,
                    usage=replace(item.usage, failed_attempt_tokens=1),
                )
                if item.event_id == accepted.event_id
                else item
                for item in ledger.attributions
            ),
            calibrations=ledger.calibrations,
        )


def test_rejects_foreign_fallback_calibration_and_forged_terminal_claims() -> None:
    ledger = _ledger()
    fallback = next(
        item
        for item in ledger.attributions
        if item.usage.source is UsageSource.CALIBRATED_FALLBACK
    )
    foreign = FallbackTokenizerCalibration(
        envelope=_envelope("model:foreign"),
        calibration_revision="foreign:1",
        samples=(TokenizerCalibrationSample("foreign", 4, 1),),
    )
    with pytest.raises(TokenLedgerValidationError, match="foreign"):
        SupervisorTokenLedger(
            binding=ledger.binding,
            lifecycle_events=ledger.lifecycle_events,
            terminal_attributions=ledger.terminal_attributions,
            attributions=tuple(
                replace(
                    item,
                    usage=replace(
                        item.usage,
                        calibration_id=foreign.calibration_id,
                    ),
                )
                if item.event_id == fallback.event_id
                else item
                for item in ledger.attributions
            ),
            calibrations=(foreign,),
        )

    accepted = next(item for item in ledger.terminal_attributions if item.accepted)
    with pytest.raises(TokenLedgerValidationError, match="passed validation"):
        replace(accepted, validation_result=ValidationResult.FAILED)
    rejected = next(
        item
        for item in ledger.terminal_attributions
        if item.disposition is TerminalDisposition.REJECTED
    )
    with pytest.raises(TokenLedgerValidationError, match="evidence gain"):
        replace(rejected, evidence_gain=1)


def test_v1_adapter_preserves_retry_and_failed_attempt_charges() -> None:
    receipt = build_efficiency_baseline_fixtures()["repaired"]
    envelope = replace(
        _envelope(),
        max_context_tokens=max(16_384, receipt.tokens.input_tokens),
    )
    ledger = adapt_efficiency_receipt(
        receipt,
        binding=_binding(),
        envelope=envelope,
        criterion_id="criterion:repaired-output",
    )
    report = ledger.report

    assert report.input_tokens == receipt.tokens.input_tokens
    assert report.output_tokens == receipt.tokens.output_tokens
    assert report.reused_tokens == receipt.tokens.reused_tokens
    assert report.retry_tokens == sum(
        item.tokens.total_tokens for item in receipt.retries
    )
    assert report.failed_attempt_tokens > 0
    assert report.accepted_criterion_count == 1
    assert report.total_cost_microunits == receipt.total_cost_microunits
    assert len(ledger.lifecycle_events) == receipt.attempt


def _endpoint_scope() -> EndpointUsageScope:
    provider_id = stable_id("provider", "example-ai")
    return EndpointUsageScope(
        provider_id=provider_id,
        protocol=ProtocolKind.HTTPS,
        operation="text.chat",
        deployment_id=stable_id("deployment", provider_id, "chat"),
        credential_pseudonym=credential_configuration_pseudonym(
            "env:EXAMPLE_API_KEY", key_id="ledger-default"
        ),
    )


def _endpoint_event(
    *,
    sequence: int,
    input_tokens: int,
    output_tokens: int,
    cost_micros: int,
    request_id: str = "request:endpoint",
) -> UsageEvent:
    return UsageEvent(
        kind=UsageEventKind.COMMIT,
        scope_id=_endpoint_scope().scope_id,
        request_id=request_id,
        sequence=sequence,
        occurred_at=f"2026-07-28T12:00:{sequence:02d}Z",
        units=UsageVector.of(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_micros=cost_micros,
            currency="USD",
        ),
    )


def test_ledger_consumes_reconciled_endpoint_events_exactly_once() -> None:
    binding = _binding()
    envelope = _envelope()
    failed = _event(
        binding, stage="inference", attempt=1, kind=StageEventKind.FAILED
    )
    accepted = _event(
        binding, stage="inference", attempt=2, kind=StageEventKind.COMPLETED
    )
    rejected_terminal = TerminalCriterionAttribution(
        binding=binding,
        terminal_event_id=failed.event_id,
        criterion_id="criterion:endpoint-bridge",
        disposition=TerminalDisposition.REJECTED,
        validation_result=ValidationResult.FAILED,
        reason_code="provider-error",
    )
    accepted_terminal = TerminalCriterionAttribution(
        binding=binding,
        terminal_event_id=accepted.event_id,
        criterion_id="criterion:endpoint-bridge",
        disposition=TerminalDisposition.ACCEPTED,
        validation_result=ValidationResult.PASSED,
        evidence_gain=2,
    )
    endpoint_failed = _endpoint_event(
        sequence=1, input_tokens=100, output_tokens=20, cost_micros=1_000
    )
    endpoint_accepted = _endpoint_event(
        sequence=2, input_tokens=60, output_tokens=20, cost_micros=500
    )
    ledger = adapt_efficiency_metrics_from_reconciled_events(
        binding=binding,
        lifecycle_events=(failed, accepted),
        terminal_attributions=(rejected_terminal, accepted_terminal),
        endpoint_events=(endpoint_failed, endpoint_accepted),
        envelope=envelope,
        context_ids=("context:failed", "context:accepted"),
        cache_decisions=(CacheDecision.MISS, CacheDecision.MISS),
    )
    report = ledger.report

    assert all(
        item.usage.source is UsageSource.RECONCILED_ENDPOINT
        for item in ledger.attributions
    )
    assert report.input_tokens == 160
    assert report.output_tokens == 40
    assert report.total_cost_microunits == 1_500
    assert report.failed_attempt_tokens == 120
    assert report.rejected_tokens == 120
    assert report.accepted_criterion_count == 1
    assert report.accepted_evidence_gain == 2
    assert report.retry_tokens == 80
    event_ids = [item.usage.endpoint_event_id for item in ledger.attributions]
    assert len(event_ids) == len(set(event_ids)) == 2

    with pytest.raises(
        TokenLedgerValidationError, match="exactly once"
    ):
        consume_reconciled_endpoint_events_exactly_once(
            (endpoint_failed, endpoint_failed)
        )

    # Duplicate endpoint binding inside a constructed ledger fails closed.
    dup_usage = provider_usage_from_reconciled_endpoint_event(
        endpoint_failed,
        envelope=envelope,
        measurement_id="dup",
        failed_attempt_tokens=120,
    )
    with pytest.raises(TokenLedgerValidationError, match="exactly once"):
        SupervisorTokenLedger(
            binding=binding,
            lifecycle_events=(failed, accepted),
            terminal_attributions=(rejected_terminal, accepted_terminal),
            attributions=(
                TokenAttribution(
                    binding=binding,
                    event_id=failed.event_id,
                    stage=failed.stage,
                    attempt=1,
                    context_id="context:failed",
                    cache_decision=CacheDecision.MISS,
                    validation_result=ValidationResult.FAILED,
                    terminal_attribution_id=(
                        rejected_terminal.terminal_attribution_id
                    ),
                    usage=dup_usage,
                ),
                TokenAttribution(
                    binding=binding,
                    event_id=accepted.event_id,
                    stage=accepted.stage,
                    attempt=2,
                    context_id="context:accepted",
                    cache_decision=CacheDecision.MISS,
                    validation_result=ValidationResult.PASSED,
                    terminal_attribution_id=(
                        accepted_terminal.terminal_attribution_id
                    ),
                    usage=ProviderTokenUsage(
                        measurement_id="accepted",
                        envelope=envelope,
                        source=UsageSource.RECONCILED_ENDPOINT,
                        input_tokens=60,
                        output_tokens=20,
                        retry_tokens=80,
                        cost_microunits=500,
                        endpoint_event_id=dup_usage.endpoint_event_id,
                    ),
                ),
            ),
        )


def test_token_ledger_cannot_authorize_usage_or_claim_completion() -> None:
    bounds = token_ledger_authority_bounds()
    assert bounds == {
        "authorizes_usage": False,
        "rewrites_provider_settlement": False,
        "is_completion_evidence": False,
        "is_correctness_evidence": False,
    }
    assert not TOKEN_LEDGER_AUTHORIZES_USAGE
    assert not TOKEN_LEDGER_REWRITES_PROVIDER_SETTLEMENT
    assert not TOKEN_LEDGER_IS_COMPLETION_EVIDENCE
    assert not TOKEN_LEDGER_IS_CORRECTNESS_EVIDENCE

    usage = provider_usage_from_reconciled_endpoint_event(
        _endpoint_event(
            sequence=1, input_tokens=10, output_tokens=5, cost_micros=25
        ),
        envelope=_envelope(),
    )
    assert usage.source is UsageSource.RECONCILED_ENDPOINT
    assert usage.input_tokens == 10
    assert usage.output_tokens == 5
    assert usage.cost_microunits == 25
    # Projection never invents settlement authority flags on the measurement.
    payload = usage.to_dict()
    assert "authorizes_usage" not in payload
    assert payload["endpoint_event_id"]

