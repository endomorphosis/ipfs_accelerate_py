"""End-to-end paired gate for prompt bootstrap and rescue."""

from __future__ import annotations

from dataclasses import replace

from ipfs_accelerate_py.agent_supervisor.prompt_workflow_benchmark import (
    REQUIRED_PLANNING_MODES,
    REQUIRED_TASK_SOURCES,
    REQUIRED_TRANSPORTS,
    PlanningMode,
    PromptWorkflowBenchmark,
    TaskSourceBackend,
    TransportSurface,
    build_frozen_prompt_workflow_benchmark,
    recompute_prompt_workflow_gate,
    verify_prompt_workflow_gate_report,
)


def test_frozen_fixtures_agree_across_backends_planning_and_transports():
    benchmark = build_frozen_prompt_workflow_benchmark()
    report = recompute_prompt_workflow_gate(benchmark)

    assert report.passed
    assert report.task_cid_parity_passed
    assert report.ready_set_parity_passed
    assert report.effect_parity_passed
    assert report.terminal_parity_passed
    assert report.transport_parity_passed
    assert set(report.task_sources_passed) == {
        item.value for item in REQUIRED_TASK_SOURCES
    }
    assert set(report.planning_modes_passed) == {
        item.value for item in REQUIRED_PLANNING_MODES
    }
    assert set(report.transports_passed) == {
        item.value for item in REQUIRED_TRANSPORTS
    }
    assert report.admitted_task_cid_count == 3
    assert report.ready_task_cid_count == 1
    assert report.accepted_effect_count == 2
    assert verify_prompt_workflow_gate_report(report, benchmark)

    paired = [r for r in benchmark.receipts if r.is_paired_path]
    reference = next(
        r
        for r in paired
        if r.planning_mode is PlanningMode.DETERMINISTIC
        and r.task_source is TaskSourceBackend.MARKDOWN
        and r.transport is TransportSurface.PYTHON
    )
    for receipt in paired:
        assert (
            receipt.metrics.admitted_task_cids
            == reference.metrics.admitted_task_cids
        )
        assert (
            receipt.metrics.ready_task_cids == reference.metrics.ready_task_cids
        )
        assert (
            receipt.metrics.accepted_effect_ids
            == reference.metrics.accepted_effect_ids
        )
        assert (
            receipt.metrics.terminal_result == reference.metrics.terminal_result
        )
        if receipt.planning_mode is PlanningMode.DETERMINISTIC:
            assert receipt.metrics.model_calls == 0
        else:
            assert receipt.metrics.model_calls >= 1


def test_gate_fails_when_duckdb_or_mcp_diverges_from_markdown_python():
    benchmark = build_frozen_prompt_workflow_benchmark()
    target = next(
        r
        for r in benchmark.receipts
        if r.is_paired_path
        and r.task_source is TaskSourceBackend.DUCKDB
        and r.transport is TransportSurface.MCP
        and r.planning_mode is PlanningMode.DETERMINISTIC
    )
    diverged_metrics = replace(
        target.metrics,
        admitted_task_cids=target.metrics.admitted_task_cids[:-1],
        ready_task_cids=(),
        terminal_result="rejected",
    )
    changed = PromptWorkflowBenchmark(
        tuple(
            replace(receipt, metrics=diverged_metrics)
            if receipt.receipt_id == target.receipt_id
            else receipt
            for receipt in benchmark.receipts
        )
    )
    report = recompute_prompt_workflow_gate(changed)
    assert not report.passed
    assert "task-cid-parity" in report.failure_codes
    assert "ready-set-parity" in report.failure_codes
    assert "terminal-parity" in report.failure_codes


def test_python_cli_script_mcp_surfaces_are_required_and_parity_bound():
    benchmark = build_frozen_prompt_workflow_benchmark()
    surfaces = {
        (r.planning_mode, r.task_source, r.transport)
        for r in benchmark.receipts
        if r.is_paired_path
    }
    assert surfaces == {
        (mode, source, transport)
        for mode in REQUIRED_PLANNING_MODES
        for source in REQUIRED_TASK_SOURCES
        for transport in REQUIRED_TRANSPORTS
    }
    # script is a first-class surface, not collapsed into CLI.
    assert any(
        r.transport is TransportSurface.SCRIPT and r.is_paired_path
        for r in benchmark.receipts
    )
    assert any(
        r.transport is TransportSurface.MCP and r.is_paired_path
        for r in benchmark.receipts
    )
