"""PDR-071 extensions: cancelled tokens and causal-span joins on the ledger."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.self_improvement.supervisor_token_ledger import (
    CacheDecision,
    ProviderModelEnvelope,
    ProviderTokenUsage,
    SupervisorTokenLedger,
    TerminalCriterionAttribution,
    TerminalDisposition,
    TokenAttribution,
    TokenLedgerValidationError,
    UsageSource,
    ValidationResult,
    attributions_for_span,
    bind_attribution_to_span,
    provider_native_token_totals_for_span,
)
from ipfs_accelerate_py.agent_supervisor.self_improvement.supervisor_v2_contracts import (
    AuthorityClass,
    ResultBinding,
    SemanticDependencyIdentity,
    StageEvent,
    StageEventKind,
)


def _binding(task_id: str = "PDR-071") -> ResultBinding:
    return ResultBinding(
        repository_id="repository:supervisor",
        tree_id="tree:span-ledger",
        objective_id="PDR-G080",
        objective_revision="objective:g080@1",
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
                revision="tree:span-ledger",
                digest="sha256:" + "c" * 64,
            ),
        ),
    )


def _envelope() -> ProviderModelEnvelope:
    return ProviderModelEnvelope(
        provider_id="provider:example",
        model_id="model:reasoner",
        model_revision="model:reasoner@2026-07",
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
) -> StageEvent:
    return StageEvent(
        binding=binding,
        stage=stage,
        attempt=attempt,
        sequence=0,
        kind=kind,
        authority=AuthorityClass.VALIDATION,
        occurred_at=f"2026-08-03T12:00:{attempt:02d}.000000Z",
        reason_code="" if kind is StageEventKind.COMPLETED else "ended",
    )


def _ledger(
    *,
    explicit_cancelled: bool = True,
    span_id: str = "span:attempt-1",
) -> SupervisorTokenLedger:
    binding = _binding()
    envelope = _envelope()
    accepted = _event(
        binding, stage="inference", attempt=1, kind=StageEventKind.COMPLETED
    )
    cancelled = _event(
        binding, stage="inference", attempt=2, kind=StageEventKind.CANCELLED
    )
    accepted_terminal = TerminalCriterionAttribution(
        binding=binding,
        terminal_event_id=accepted.event_id,
        criterion_id="criterion:main",
        disposition=TerminalDisposition.ACCEPTED,
        validation_result=ValidationResult.PASSED,
        evidence_gain=2,
    )
    cancelled_terminal = TerminalCriterionAttribution(
        binding=binding,
        terminal_event_id=cancelled.event_id,
        criterion_id="criterion:follow-up",
        disposition=TerminalDisposition.ABANDONED,
        validation_result=ValidationResult.NOT_RUN,
        reason_code="cancelled",
    )
    accepted_usage = ProviderTokenUsage(
        measurement_id="request:accepted",
        envelope=envelope,
        source=UsageSource.PROVIDER_NATIVE,
        input_tokens=50,
        output_tokens=15,
        reused_tokens=10,
        cost_microunits=200,
    )
    cancelled_usage = ProviderTokenUsage(
        measurement_id="request:cancelled",
        envelope=envelope,
        source=UsageSource.PROVIDER_NATIVE,
        input_tokens=30,
        output_tokens=5,
        tool_tokens=5,
        retry_tokens=40,
        failed_attempt_tokens=40,
        cancelled_tokens=40 if explicit_cancelled else 0,
        cost_microunits=75,
    )
    return SupervisorTokenLedger(
        binding=binding,
        lifecycle_events=(accepted, cancelled),
        terminal_attributions=(accepted_terminal, cancelled_terminal),
        attributions=(
            TokenAttribution(
                binding=binding,
                event_id=accepted.event_id,
                stage=accepted.stage,
                attempt=1,
                context_id="context:accepted",
                cache_decision=CacheDecision.HIT,
                validation_result=ValidationResult.PASSED,
                terminal_attribution_id=(
                    accepted_terminal.terminal_attribution_id
                ),
                usage=accepted_usage,
                span_id=span_id,
            ),
            TokenAttribution(
                binding=binding,
                event_id=cancelled.event_id,
                stage=cancelled.stage,
                attempt=2,
                context_id="context:cancelled",
                cache_decision=CacheDecision.MISS,
                validation_result=ValidationResult.NOT_RUN,
                terminal_attribution_id=(
                    cancelled_terminal.terminal_attribution_id
                ),
                usage=cancelled_usage,
                span_id=span_id,
            ),
        ),
    )


def test_cancelled_tokens_are_classified_without_double_charge() -> None:
    ledger = _ledger(explicit_cancelled=True)
    report = ledger.report

    assert report.input_tokens == 80
    assert report.output_tokens == 20
    assert report.tool_tokens == 5
    assert report.total_tokens == 105
    assert report.retry_tokens == 40
    assert report.cancelled_tokens == 40
    assert report.abandoned_tokens == 40
    assert report.failed_attempt_tokens == 40
    # Cancelled is a classification of the same total, not an extra charge.
    assert report.cancelled_tokens <= report.total_tokens
    assert report.total_cost_microunits == 275


def test_cancelled_tokens_derived_from_cancelled_lifecycle_events() -> None:
    ledger = _ledger(explicit_cancelled=False)
    assert ledger.report.cancelled_tokens == 40


def test_partial_cancelled_tokens_on_cancelled_event_are_rejected() -> None:
    binding = _binding()
    envelope = _envelope()
    cancelled = _event(
        binding, stage="inference", attempt=1, kind=StageEventKind.CANCELLED
    )
    terminal = TerminalCriterionAttribution(
        binding=binding,
        terminal_event_id=cancelled.event_id,
        criterion_id="criterion:only",
        disposition=TerminalDisposition.ABANDONED,
        validation_result=ValidationResult.NOT_RUN,
        reason_code="cancelled",
    )
    usage = ProviderTokenUsage(
        measurement_id="request:partial-cancel",
        envelope=envelope,
        source=UsageSource.PROVIDER_NATIVE,
        input_tokens=10,
        output_tokens=0,
        failed_attempt_tokens=10,
        cancelled_tokens=4,
    )
    with pytest.raises(TokenLedgerValidationError, match="cancelled_tokens"):
        SupervisorTokenLedger(
            binding=binding,
            lifecycle_events=(cancelled,),
            terminal_attributions=(terminal,),
            attributions=(
                TokenAttribution(
                    binding=binding,
                    event_id=cancelled.event_id,
                    stage=cancelled.stage,
                    attempt=1,
                    context_id="context:partial",
                    cache_decision=CacheDecision.MISS,
                    validation_result=ValidationResult.NOT_RUN,
                    terminal_attribution_id=terminal.terminal_attribution_id,
                    usage=usage,
                ),
            ),
        )


def test_span_binding_selects_attributions_exactly_once() -> None:
    ledger = _ledger(span_id="span:run-A")
    selected = attributions_for_span(ledger, "span:run-A")
    assert len(selected) == 2
    assert {item.event_id for item in selected} == {
        item.event_id for item in ledger.lifecycle_events
    }
    assert attributions_for_span(ledger, "span:other") == ()

    totals = provider_native_token_totals_for_span(ledger, "span:run-A")
    assert totals["provider_native_input_tokens"] == 80
    assert totals["provider_native_output_tokens"] == 20
    assert totals["provider_native_reused_tokens"] == 10
    assert totals["provider_native_retry_tokens"] == 40
    assert totals["provider_native_cancelled_tokens"] == 40
    assert totals["model_call_count"] == 2


def test_bind_attribution_to_span_is_exclusive() -> None:
    ledger = _ledger(span_id="")
    unbound = ledger.attributions[0]
    assert unbound.span_id == ""
    bound = bind_attribution_to_span(unbound, "span:new")
    assert bound.span_id == "span:new"
    assert bound.event_id == unbound.event_id
    assert bind_attribution_to_span(bound, "span:new").content_id == bound.content_id
    with pytest.raises(TokenLedgerValidationError, match="different span"):
        bind_attribution_to_span(bound, "span:conflict")


def test_cancelled_tokens_cannot_exceed_total() -> None:
    envelope = _envelope()
    with pytest.raises(TokenLedgerValidationError, match="cancelled_tokens"):
        ProviderTokenUsage(
            measurement_id="request:overflow",
            envelope=envelope,
            source=UsageSource.PROVIDER_NATIVE,
            input_tokens=5,
            output_tokens=0,
            cancelled_tokens=6,
        )


def test_report_round_trip_preserves_cancelled_tokens() -> None:
    ledger = _ledger()
    restored = SupervisorTokenLedger.from_dict(ledger.to_dict())
    assert restored.report.cancelled_tokens == ledger.report.cancelled_tokens
    assert restored.report.content_id == ledger.report.content_id
    assert restored.ledger_id == ledger.ledger_id
