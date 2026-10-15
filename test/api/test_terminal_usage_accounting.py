"""Final-report accounting uses native sessions and retains missing coverage.

Counter receipts here are authored fixtures, not provider or benchmark results.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from benchmarks.agent_supervisor.container_coding import (
    full_supervisor_harbor_agent as adapter,
)
from benchmarks.agent_supervisor.container_coding import (
    terminal_container_supervisor as driver,
)
from ipfs_accelerate_py.agent_supervisor.runtime import (
    cumulative_usage_reconciliation as accounting,
)
from test.api.test_terminal_prompt_tokenizer_forwarding import supervisor  # noqa: F401


def _receipt(*, invocation="first", thread="native-session-123456789", count=1,
             inputs=100, outputs=20, complete=True, legacy=False):
    native = {
        "schema": "native-codex-cumulative-usage@1", "thread_id": thread,
        "rollout_sha256": hashlib.sha256(f"{thread}:{count}:{inputs}:{outputs}".encode()).hexdigest(),
        "observed_token_count_records": count,
        "usage": {"input_tokens": inputs, "output_tokens": outputs,
                  "cached_input_tokens": inputs // 2, "reasoning_output_tokens": outputs // 2,
                  "total_tokens": inputs + outputs},
        "cache_included_in_input": True, "usage_available": True,
        "task_complete_observed": complete,
    }
    if not legacy:
        native.update(completion_basis=accounting.COMPLETION_BASIS,
                      lifecycle_malformed_records=0, usage_lifecycle_complete=complete)
    return {
        "schema": "router-implementation-invocation@1", "invocation_id": invocation,
        "provider": "codex_cli", "model": "gpt-5.6-sol", "workspace": "/isolated/checkout",
        "phase": "coding", "status": "provider_returned",
        "usage": {"thread_id": thread, "model_id": "gpt-5.6-sol"},
        "native_rollout_usage": native,
    }


def _finalize(report):
    before = deepcopy(report)
    driver._final_usage_accounting(report)
    result = report["provider_usage_accounting"]
    assert {key: value for key, value in report.items() if key != "provider_usage_accounting"} == before
    assert result["schema"] == "terminal-provider-usage-accounting@1"
    assert result["campaign_usage_complete"] is None
    for key in ("billing_total_verified", "production_token_savings_established",
                "execution_authority", "completion_authority"):
        assert result[key] is False
    return result


def test_final_report_and_harbor_count_one_native_session_across_allocations():
    early = _receipt(complete=False)
    latest = _receipt(invocation="second", count=3, inputs=160, outputs=40)
    report = {"task_completed": False, "provider_invocations": [early, latest, deepcopy(latest)]}
    result = _finalize(report)
    assert result["known_unreceipted_provider_attempt"] is False
    assert result["report_provider_total_tokens"] == 200
    reconciled = result["reconciliation"]
    assert reconciled["native_sessions_observed"] == 1
    assert reconciled["repeated_receipt_observations"] == 1
    assert reconciled["known_counter_subtotals"]["cached_input_tokens"] == 80
    measured = adapter.measured_usage(report)
    assert measured["input_tokens"] == 160
    assert measured["output_tokens"] == 40
    assert measured["cached_input_tokens"] == 80
    assert measured["total_tokens"] == 200
    assert measured["provider_calls"] == 2
    assert measured["observed_complete_sessions"] is True
    assert measured["billing_total_verified"] is False


def test_fresh_failed_retry_is_charged_but_prevents_exact_report_total():
    first = _receipt()
    retry = _receipt(invocation="retry", thread="fresh-retry-session-987654321",
                     inputs=70, outputs=10, complete=False)
    retry.update(status="failed", phase="expansion")
    report = {"provider_invocations": [first, retry], "task_completed": True}
    result = _finalize(report)
    assert result["reconciliation"]["native_sessions_observed"] == 2
    assert result["reconciliation"]["known_provider_token_subtotal"] == 200
    assert result["report_provider_total_tokens"] is None
    measured = adapter.measured_usage(report)
    assert measured["total_tokens"] is None
    assert measured["observed_complete_sessions"] is False


def test_complete_provider_categories_do_not_invent_missing_cache_categories():
    first = _receipt()
    second = _receipt(invocation="other", thread="other-session-987654321")
    del second["native_rollout_usage"]["usage"]["cached_input_tokens"]
    report = {"provider_invocations": [first, second]}
    result = _finalize(report)
    assert result["report_provider_total_tokens"] == 240
    assert result["reconciliation"]["known_counter_subtotals"]["cached_input_tokens"] == 50
    measured = adapter.measured_usage(report)
    assert measured["input_tokens"] == 200
    assert measured["output_tokens"] == 40
    assert measured["total_tokens"] == 240
    assert measured["cached_input_tokens"] is None


@pytest.mark.parametrize("marker", [{"phase": "planning", "usage": None},
                                   {"phase": "coding", "usage": None}, {}, None])
def test_known_unreceipted_attempt_preserves_receipt_subtotal_and_unknown_report(marker):
    report = {"provider_invocations": [_receipt()], "unreceipted_provider_attempt": marker}
    result = _finalize(report)
    assert result["known_unreceipted_provider_attempt"] is True
    assert result["report_provider_total_tokens"] is None
    # Supplied receipts are complete; the additional attempt has no invented ID.
    assert result["reconciliation"]["supplied_receipt_usage_complete"] is True
    assert result["reconciliation"]["provider_total_tokens"] == 120
    assert result["reconciliation"]["observations_received"] == 1
    measured = adapter.measured_usage(report)
    assert measured["total_tokens"] is None
    assert measured["all_invocations_receipted"] is False


def test_old_completion_flag_keeps_harbor_totals_and_final_completeness_unknown():
    report = {"provider_invocations": [_receipt(legacy=True)], "task_completed": True}
    result = _finalize(report)
    assert result["reconciliation"]["known_provider_token_subtotal"] == 120
    assert result["report_provider_total_tokens"] is None
    measured = adapter.measured_usage(report)
    assert measured["total_tokens"] is None
    assert measured["observed_complete_sessions"] is False


@pytest.mark.parametrize("native,reason", [
    (None, "native_usage_unavailable"),
    ({"schema": "native-grok-final-usage@1", "usage": {"total_tokens": 999}},
     "unsupported_native_usage_schema"),
])
def test_missing_and_unsupported_usage_are_unknown_in_both_report_surfaces(native, reason):
    row = _receipt()
    row["native_rollout_usage"] = native
    report = {"provider_invocations": [row]}
    result = _finalize(report)
    assert result["report_provider_total_tokens"] is None
    assert result["reconciliation"]["known_provider_token_subtotal"] is None
    assert reason in {item["reason"] for item in result["reconciliation"]["issues"]}
    assert adapter.measured_usage(report)["total_tokens"] is None


def test_provider_free_candidate_does_not_fabricate_zero_usage():
    report = {"provider_invocations": [], "task_completed": True,
              "implementation_route": "doctor_candidate"}
    result = _finalize(report)
    assert result["report_provider_total_tokens"] is None
    assert result["reconciliation"]["known_provider_token_subtotal"] is None
    measured = adapter.measured_usage(report)
    assert measured["provider_calls"] == 0
    assert measured["total_tokens"] is None
    assert measured["observed_complete_sessions"] is False


def test_accounting_exception_cannot_suppress_report_or_export_exception_details(monkeypatch):
    def broken(_):
        raise RuntimeError("private prompt and credentials must not escape")

    monkeypatch.setattr(accounting, "reconcile_router_cumulative_usage", broken)
    report = {"task_completed": True, "provider_invocations": [_receipt()]}
    result = _finalize(report)
    assert result["report_provider_total_tokens"] is None
    assert result["error_type"] == "RuntimeError"
    assert "private prompt" not in json.dumps(result)


def test_accounting_ignores_untrusted_cached_summary_and_raw_receipt_fields():
    row = _receipt()
    row.update(prompt="sensitive prompt", credential="sensitive secret", tool_tokens=9999)
    report = {"provider_invocations": [row], "provider_usage_accounting": {
        "report_provider_total_tokens": 1, "reconciliation": {"provider_total_tokens": 1}}}
    driver._final_usage_accounting(report)
    result = report["provider_usage_accounting"]
    assert result["report_provider_total_tokens"] == 120
    assert "sensitive" not in json.dumps(result)
    # The adapter derives counters from retained receipts, not a cached summary.
    report["provider_usage_accounting"] = {"report_provider_total_tokens": 1}
    assert adapter.measured_usage(report)["total_tokens"] == 120


@pytest.mark.parametrize("conflict", [False, True])
def test_driver_retains_changed_log_receipts_and_deduplicates_exact_observations(supervisor, monkeypatch, conflict):  # noqa: F811
    state, _, _ = supervisor
    original_plan = driver.preparation.plan
    first = _receipt(invocation="logged-coding")
    second = (_receipt(invocation="logged-coding", count=2, inputs=130, outputs=25)
              if conflict else deepcopy(first))

    def plan(**kwargs):
        result = original_plan(**kwargs)
        (state / "retained-coding.log").write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n")
        return result

    monkeypatch.setattr(driver.preparation, "plan", plan)
    report = driver.run(instruction=Path("/fixture/instruction"), state=state, arm="no-index")
    coding = [row for row in report["provider_invocations"] if row["invocation_id"] == "logged-coding"]
    assert len(coding) == (2 if conflict else 1)
    session = report["provider_usage_accounting"]["reconciliation"]["sessions"][0]
    if conflict:
        assert session["status"] == "conflict"
        assert "invocation_identity_alias_conflict" in session["reasons"]
        assert session["known_provider_token_subtotal"] is None
    else:
        assert session["known_provider_token_subtotal"] == 120


def test_harbor_keeps_readable_report_when_accounting_fails(tmp_path, monkeypatch):
    from harbor.models.agent.context import AgentContext

    report = {"task_completed": True, "seconds": 1.5, "phases": {"prepare_seconds": 0.25},
              "provider_invocations": [_receipt()], "remaining_processes": 0,
              "error": {"type": "AuthoredDiagnostic"}}

    class Environment:
        async def upload_file(self, source, target):
            pass

        async def exec(self, **kwargs):
            return SimpleNamespace(return_code=0, stdout="", stderr="")

        async def download_file(self, source, target):
            Path(target).write_text(json.dumps(report if source.endswith("-result.json") else {}))

    def broken(_):
        raise RuntimeError("private counter exception must not be exported")

    public_inputs = AsyncMock(return_value={"status": "captured"})
    public_outputs = AsyncMock(return_value={"status": "captured"})
    monkeypatch.setattr(adapter, "capture_public_inputs", public_inputs)
    monkeypatch.setattr(adapter, "export_public_outputs", public_outputs)
    monkeypatch.setattr(adapter, "measured_usage", broken)
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "manifest.json").write_text(json.dumps({"archive_sha256": "fixture", "learned_requirements": []}))
    agent = adapter.FullSupervisorAgent(logs_dir=tmp_path / "logs", model_name="gpt-5.6-sol",
                                        runtime_archive=str(archive), arm="no-index")
    context = AgentContext()
    asyncio.run(agent.run("Public task fixture", Environment(), context))
    assert context.metadata["task_completed"] is True
    assert context.metadata["supervisor_seconds"] == report["seconds"]
    assert context.metadata["phases"] == report["phases"]
    assert context.metadata["error"] == report["error"]
    assert context.metadata["remaining_processes"] == 0
    assert context.metadata["usage_accounting_error_type"] == "RuntimeError"
    assert "report_unavailable" not in context.metadata
    assert context.n_input_tokens is context.n_output_tokens is None
    assert "private counter exception" not in json.dumps(context.metadata)
    public_outputs.assert_awaited_once()


def test_driver_retains_unhashable_invocation_identity_as_unknown_evidence(supervisor, monkeypatch):  # noqa: F811
    state, _, _ = supervisor
    original_plan = driver.preparation.plan
    malformed = _receipt()
    malformed["invocation_id"] = ["malformed"]

    def plan(**kwargs):
        result = original_plan(**kwargs)
        (state / "malformed-coding.log").write_text(json.dumps(malformed) + "\n")
        return result

    monkeypatch.setattr(driver.preparation, "plan", plan)
    report = driver.run(instruction=Path("/fixture/instruction"), state=state, arm="no-index")
    assert any(row.get("invocation_id") == ["malformed"] for row in report["provider_invocations"])
    result = report["provider_usage_accounting"]
    assert result["report_provider_total_tokens"] is None
    assert "missing_or_invalid_invocation_identity" in {
        issue["reason"] for issue in result["reconciliation"]["issues"]}
