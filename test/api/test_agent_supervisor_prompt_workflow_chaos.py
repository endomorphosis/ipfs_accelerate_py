"""Chaos injection at every materialization/lifecycle/rescue boundary."""

from __future__ import annotations

from dataclasses import replace

from ipfs_accelerate_py.agent_supervisor.prompt_workflow_benchmark import (
    REQUIRED_CHAOS_BOUNDARIES,
    ChaosBoundary,
    FaultOutcome,
    PromptWorkflowBenchmark,
    TerminalOutcome,
    build_frozen_prompt_workflow_benchmark,
    recompute_prompt_workflow_gate,
)


def test_every_intent_effect_receipt_boundary_has_deterministic_recovery():
    benchmark = build_frozen_prompt_workflow_benchmark()
    report = recompute_prompt_workflow_gate(benchmark)

    assert set(report.chaos_boundaries_passed) == {
        item.value for item in REQUIRED_CHAOS_BOUNDARIES
    }
    assert report.chaos_passed
    assert report.bounds_passed
    assert report.passed

    materialize = {
        b for b in REQUIRED_CHAOS_BOUNDARIES if b.value.startswith("materialize-")
    }
    lifecycle = {
        b for b in REQUIRED_CHAOS_BOUNDARIES if b.value.startswith("lifecycle-")
    }
    rescue = {
        b for b in REQUIRED_CHAOS_BOUNDARIES if b.value.startswith("rescue-")
    }
    assert len(materialize) == 6
    assert len(lifecycle) == 6
    assert len(rescue) == 6

    chaos = [
        receipt
        for receipt in benchmark.receipts
        if receipt.chaos_boundary is not None
    ]
    assert len(chaos) == len(REQUIRED_CHAOS_BOUNDARIES)
    outcomes = {receipt.fault_outcome for receipt in chaos}
    assert outcomes == set(FaultOutcome)
    for receipt in chaos:
        assert receipt.fault_outcome is not None
        assert receipt.metrics.escape_count == 0
        assert receipt.metrics.terminal_result in {
            TerminalOutcome.HEALTHY.value,
            TerminalOutcome.QUARANTINED.value,
            TerminalOutcome.DEGRADED.value,
        }


def test_chaos_escape_or_missing_boundary_fails_closed():
    benchmark = build_frozen_prompt_workflow_benchmark()
    target = next(
        receipt
        for receipt in benchmark.receipts
        if receipt.chaos_boundary is ChaosBoundary.LIFECYCLE_AFTER_EFFECT
    )
    escaped = PromptWorkflowBenchmark(
        tuple(
            replace(
                receipt,
                metrics=replace(
                    receipt.metrics,
                    escape_count=1,
                    terminal_result=TerminalOutcome.ACCEPTED.value,
                ),
            )
            if receipt.receipt_id == target.receipt_id
            else receipt
            for receipt in benchmark.receipts
        )
    )
    failed = recompute_prompt_workflow_gate(escaped)
    assert not failed.passed
    assert (
        f"chaos-escape:{ChaosBoundary.LIFECYCLE_AFTER_EFFECT.value}"
        in failed.failure_codes
    )

    omitted = PromptWorkflowBenchmark(
        tuple(
            receipt
            for receipt in benchmark.receipts
            if receipt.chaos_boundary
            is not ChaosBoundary.RESCUE_BEFORE_INTENT
        )
    )
    missing = recompute_prompt_workflow_gate(omitted)
    assert not missing.passed
    assert (
        f"missing-chaos-boundary:{ChaosBoundary.RESCUE_BEFORE_INTENT.value}"
        in missing.failure_codes
    )


def test_resource_bounds_cover_tokens_model_calls_retries_storage_processes():
    benchmark = build_frozen_prompt_workflow_benchmark()
    report = recompute_prompt_workflow_gate(benchmark)
    assert report.model_calls >= 1  # model planning path is present
    assert report.total_tokens >= 0
    assert report.retries >= 0
    assert report.storage_bytes > 0
    assert report.process_count >= 0
    assert report.bounds_passed

    target = next(receipt for receipt in benchmark.receipts if receipt.is_paired_path)
    unbounded = PromptWorkflowBenchmark(
        tuple(
            replace(
                receipt,
                metrics=replace(
                    receipt.metrics,
                    model_calls=10_001,
                    provider_input_tokens=50_000_001,
                    retries=10_001,
                    storage_bytes=512 * 1024 * 1024 + 1,
                    process_count=1_001,
                ),
            )
            if receipt.receipt_id == target.receipt_id
            else receipt
            for receipt in benchmark.receipts
        )
    )
    failed = recompute_prompt_workflow_gate(unbounded)
    assert not failed.passed
    assert "resource-bounds-exceeded" in failed.failure_codes
