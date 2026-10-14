"""Authored stop receipts, cleanup and Harbor propagation; no model calls."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchmarks.agent_supervisor.container_coding.terminal_provider_stop import (
    MAX_INVOCATIONS, MAX_RETRY_AFTER_SECONDS, provider_stop_observation,
)


def failed_receipt(**changes):
    row = {
        "schema": "router-implementation-invocation@1", "invocation_id": "authored-planning-call",
        "phase": "planning", "purpose": "planning", "provider": "codex_cli", "model": "gpt-6.1-sol",
        "status": "failed", "failure_phase": "provider_invocation",
        "provider_failure": {"phase": "provider_invocation", "reason_code": "rate_limit"},
        "external_container_boundary": {"identity": "authored-container"},
        "usage": {"exit_code": 1}, "completion_authority": False,
        "native_rollout_usage": {"schema": "native-codex-cumulative-usage@1",
            "task_complete_observed": True, "task_complete_event_observed": True,
            "usage_available": False, "usage": {}, "cache_included_in_input": True},
    }
    row.update(changes)
    return row


@pytest.mark.parametrize("reason,state", [("rate_limit", "rate_limited"),
    ("billing", "provider_blocked"), ("authentication", "provider_blocked"), ("quota", "provider_blocked")])
def test_closed_stop_preserves_reason_without_usage_or_retry_authority(reason, state):
    row = failed_receipt(provider_failure={"phase": "provider_invocation", "reason_code": reason})
    report = {"provider_invocations": [row], "provider_stop": {"raw_message": "PRIVATE_FORGED"}}
    original = deepcopy(report)
    value = provider_stop_observation(report)
    assert value == {"schema": "terminal-provider-stop@1", "state": state, "reason_code": reason,
        "provider": "codex_cli", "model": "gpt-6.1-sol", "phase": "planning",
        "retry_after_seconds": None, "observation_valid": True, "retry_authority": False,
        "automatic_replay_allowed": False, "task_completion_authority": False}
    assert report == original
    assert "PRIVATE" not in json.dumps(value)


@pytest.mark.parametrize("delay", [0.25, 1, 60, MAX_RETRY_AFTER_SECONDS])
def test_only_bounded_positive_numeric_retry_hints_are_retained(delay):
    row = failed_receipt()
    row["provider_failure"]["retry_after_seconds"] = delay
    value = provider_stop_observation({"provider_invocations": [row]})
    assert value["observation_valid"] is True
    assert value["retry_after_seconds"] == delay
    assert value["state"] == "rate_limited"


@pytest.mark.parametrize("hint", [None, True, False, "60", 0, -1, float("nan"),
    float("inf"), -float("inf"), [], {}, MAX_RETRY_AFTER_SECONDS + 1, 10 ** 1000])
def test_malformed_hint_blocks_without_early_cooldown_or_serialized_private_data(hint):
    row = failed_receipt()
    row["provider_failure"]["retry_after_seconds"] = hint
    value = provider_stop_observation({"provider_invocations": [row]})
    assert value["observation_valid"] is False
    assert value["state"] == "provider_blocked"
    assert value["retry_after_seconds"] is None
    json.dumps(value, allow_nan=False)


@pytest.mark.parametrize("hint", [{"retry_after_unbounded": False},
    {"retry_after_unbounded": "true"}, {"retry_after_unbounded": 1},
    {"retry_after_unbounded": True, "retry_after_seconds": 30}])
def test_conflicting_or_false_unbounded_hint_is_invalid(hint):
    row = failed_receipt()
    row["provider_failure"].update(hint)
    value = provider_stop_observation({"provider_invocations": [row]})
    assert value["state"] == "provider_blocked"
    assert value["observation_valid"] is False
    assert value["retry_after_seconds"] is None


def test_unbounded_hint_keeps_stop_without_shortening_it():
    row = failed_receipt()
    row["provider_failure"]["retry_after_unbounded"] = True
    value = provider_stop_observation({"provider_invocations": [row]})
    assert value["state"] == "provider_blocked"
    assert value["observation_valid"] is True
    assert value["retry_after_unbounded"] is True
    assert value["retry_after_seconds"] is None


@pytest.mark.parametrize("field,value", [("provider", ""), ("provider", "PRIVATE unsupported"),
    ("provider", None), ("provider", []), ("model", ""), ("model", "PRIVATE\nmodel"),
    ("model", "x" * 257), ("model", None), ("phase", "PRIVATE_PHASE"), ("phase", []),
    ("failure_phase", "argument_validation"), ("schema", "unrecognized-schema")])
def test_bad_route_or_scope_does_not_drop_a_recognized_failed_stop(field, value):
    row = failed_receipt(**{field: value})
    stop = provider_stop_observation({"provider_invocations": [row]})
    assert stop["state"] == "provider_blocked"
    assert stop["observation_valid"] is False
    assert "PRIVATE" not in json.dumps(stop)


@pytest.mark.parametrize("diagnostic", [
    {"phase": "other", "reason_code": "rate_limit"},
    {"phase": "provider_invocation", "reason_code": "rate_limit", "message": "PRIVATE"},
    {"phase": "provider_invocation", "reason_code": "rate_limit", "http_status": True},
    {"phase": "provider_invocation", "reason_code": "billing", "retry_after_seconds": 10},
])
def test_malformed_closed_diagnostic_stays_blocked(diagnostic):
    stop = provider_stop_observation({"provider_invocations": [failed_receipt(provider_failure=diagnostic)]})
    assert stop["state"] == "provider_blocked"
    assert stop["observation_valid"] is False
    assert "PRIVATE" not in json.dumps(stop)


@pytest.mark.parametrize("report", [{}, {"provider_stop": {"state": "rate_limited"}},
    {"provider_invocations": [failed_receipt(status="provider_returned")]},
    {"provider_invocations": [failed_receipt(provider_failure={"reason_code": "unknown"})]},
    {"provider_invocations": [failed_receipt(provider_failure={"reason_code": "timeout"})]},
    {"provider_invocations": [failed_receipt(provider_failure="PRIVATE rate limit")]},
    {"provider_invocations": [{"error": "PRIVATE rate limit"}, None]},
])
def test_unrelated_errors_or_a_supplied_projection_are_not_provider_stop_evidence(report):
    assert provider_stop_observation(report) is None


@pytest.mark.parametrize("report", [[], {"provider_invocations": None},
    {"provider_invocations": {}}, {"provider_invocations": [None] * (MAX_INVOCATIONS + 1)}])
def test_invalid_populations_raise_instead_of_silently_scanning_a_prefix(report):
    with pytest.raises(ValueError):
        provider_stop_observation(report)


def test_multiple_failures_preserve_largest_cooldown_and_any_harder_stop():
    rows = [failed_receipt(), failed_receipt(phase="coding", invocation_id="authored-coding-call")]
    for row, delay in zip(rows, (12, 100)):
        row["provider_failure"]["retry_after_seconds"] = delay
    stop = provider_stop_observation({"provider_invocations": rows})
    assert stop["retry_after_seconds"] == 100
    assert stop["phase"] == "coding"
    blocked = failed_receipt(provider_failure={"phase": "provider_invocation", "reason_code": "billing"})
    for population in ([*rows, blocked], [blocked, *rows]):
        assert provider_stop_observation({"provider_invocations": population})["state"] == "provider_blocked"
    for changed in (failed_receipt(provider="grok_cli", model="grok-4.7"), failed_receipt(model="different")):
        value = provider_stop_observation({"provider_invocations": [*rows, changed]})
        assert value["state"] == "provider_blocked"
        assert value["observation_valid"] is False
        assert value["provider"] is value["model"] is value["retry_after_seconds"] is None


@pytest.mark.parametrize("extra", [None, {}, failed_receipt(status="provider_returned"),
    failed_receipt(provider_failure={"phase": "provider_invocation", "reason_code": "unknown"}),
    failed_receipt(provider_failure={"phase": "provider_invocation", "reason_code": "billing"})])
def test_malformed_or_conflicting_population_cannot_weaken_known_stop(extra):
    row = failed_receipt()
    row["provider_failure"]["retry_after_seconds"] = 10
    for rows in ([row, extra], [extra, row]):
        value = provider_stop_observation({"provider_invocations": rows})
        assert value["state"] == "provider_blocked"
        assert value["observation_valid"] is False
        assert value["retry_after_seconds"] is None


def test_unreceipted_attempt_keeps_cooldown_blocked_and_exact_duplicates_are_idempotent():
    row = failed_receipt()
    report = {"provider_invocations": [row, deepcopy(row)]}
    assert provider_stop_observation(report) == provider_stop_observation({"provider_invocations": [row]})
    report["unreceipted_provider_attempt"] = {"phase": "coding", "usage": None}
    value = provider_stop_observation(report)
    assert value["observation_valid"] is False
    assert value["state"] == "provider_blocked"


@pytest.fixture
def planning_failure_report(tmp_path, monkeypatch):
    from benchmarks.agent_supervisor.container_coding import terminal_container_supervisor as driver
    from benchmarks.agent_supervisor.container_coding.benchmark_provider_profile import resolve_provider_profile
    route = resolve_provider_profile(None)
    receipt = failed_receipt(model=route["model"])
    monkeypatch.setattr(driver, "ROOT", tmp_path)
    monkeypatch.setattr(driver, "WORKTREES", tmp_path / "worktrees")
    monkeypatch.setattr(driver.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(driver.os, "umask", lambda mask: None)
    monkeypatch.setattr(driver.signal, "signal", lambda *args: None)
    monkeypatch.setattr(driver.signal, "setitimer", lambda *args: None)
    monkeypatch.setattr(driver, "_final_context_audit", lambda *args, **kwargs: None)
    commands, planner_calls = [], []
    def run(argv, **kwargs):
        commands.append(argv)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(driver.subprocess, "run", run)
    class FailedPlanner:
        returncode = 1
        def __init__(self, argv, **kwargs):
            planner_calls.append(argv)
        def communicate(self, *args, **kwargs):
            return json.dumps(receipt) + "\n", ""
    monkeypatch.setattr(driver.subprocess, "Popen", FailedPlanner)
    monkeypatch.setattr(driver.preparation, "prepare", lambda **kwargs: {"intent_preplanning": {}})
    def plan(*, provider_callable, **kwargs):
        return provider_callable("Authored public instruction", repository=tmp_path,
            provider=route["provider"], model=route["model"], reasoning_effort=route["reasoning_effort"],
            timeout=1, max_new_tokens=128, trace_path=tmp_path / "authored-planner.log")
    monkeypatch.setattr(driver.preparation, "plan", plan)
    monkeypatch.setattr(driver.preparation, "context", lambda **kwargs: pytest.fail("coding context must not run"))
    state = tmp_path / "state" / "authored"
    report = driver.run(instruction=tmp_path / "unused.md", state=state, arm="no-index")
    assert len(planner_calls) == 1
    assert len(commands) == 2 and commands[-1][-1] == "--cleanup"
    assert report["worker_cleanup_returncode"] == 0
    assert report["task_completed"] is False
    assert report["provider_invocations"] == [receipt]
    assert report["error_phase"] == "planning"
    assert "coding_dispatch_possible" not in report
    assert json.loads((state.parent / "authored-result.json").read_text()) == report
    return report


def test_failed_planning_is_retained_with_explicit_stop_and_cleanup(planning_failure_report):
    stop = planning_failure_report["provider_stop"]
    assert stop["state"] == "rate_limited"
    assert stop["phase"] == "planning"
    assert stop["retry_authority"] is stop["automatic_replay_allowed"] is False


def test_closed_session_without_counters_remains_unknown_accounting():
    from benchmarks.agent_supervisor.container_coding.full_supervisor_harbor_agent import measured_usage
    value = measured_usage({"provider_invocations": [failed_receipt()]})
    assert value["provider_calls"] == 1
    assert value["observed_complete_sessions"] is True
    assert value["native_scoped_usage_complete_observed"] is True
    assert value["harbor_usage_exportable"] is value["usage_accounting_complete"] is False
    for name in ("input_tokens", "cached_input_tokens", "output_tokens", "total_tokens"):
        assert value[name] is None


@pytest.mark.parametrize("counts", [(0, 0, 0, 0), (100, 80, 15, 115)])
def test_complete_native_accounting_is_distinct_from_task_success(counts):
    from benchmarks.agent_supervisor.container_coding.full_supervisor_harbor_agent import measured_usage
    row = failed_receipt()
    row["native_rollout_usage"].update(usage_available=True, usage=dict(zip(
        ("input_tokens", "cached_input_tokens", "output_tokens", "total_tokens"), counts)))
    value = measured_usage({"provider_invocations": [row]})
    assert value["usage_accounting_complete"] is True
    assert value["provider_calls"] == 1
    assert value["total_tokens"] == counts[-1]
    assert row["status"] == "failed"


def test_harbor_rederives_stop_and_keeps_usage_unknown_after_planning_failure(
        planning_failure_report, tmp_path, monkeypatch):
    from harbor.models.agent.context import AgentContext
    from benchmarks.agent_supervisor.container_coding import full_supervisor_harbor_agent as adapter
    from benchmarks.agent_supervisor.container_coding.test_terminal_deployment import _supervisor_manifest
    report = deepcopy(planning_failure_report)
    report["provider_stop"] = {"state": "FORGED", "retry_authority": True}
    monkeypatch.setattr(adapter, "capture_public_inputs", AsyncMock(return_value={"status": "captured"}))
    monkeypatch.setattr(adapter, "export_public_outputs", AsyncMock(return_value={"status": "captured"}))
    calls = []
    class Environment:
        async def upload_file(self, source, target):
            pass
        async def exec(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(stdout="", stderr="", return_code=1)
        async def download_file(self, source, target):
            if not source.endswith("-result.json"):
                raise FileNotFoundError(source)
            Path(target).write_text(json.dumps(report))
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "manifest.json").write_text(json.dumps(_supervisor_manifest({
        "archive_sha256": "authored", "learned_requirements": []})))
    agent = adapter.FullSupervisorAgent(logs_dir=tmp_path / "logs", model_name="gpt-6.1-sol",
        runtime_archive=str(archive), arm="no-index")
    context = AgentContext()
    asyncio.run(agent.run("Authored public instruction", Environment(), context))
    assert len(calls) == 1
    assert context.metadata["provider_stop"] == provider_stop_observation(report)
    assert context.metadata["provider_stop"]["state"] == "rate_limited"
    assert context.metadata["driver_returncode"] == 1
    assert context.metadata["task_completed"] is False
    assert context.metadata["usage"]["provider_calls"] == 1
    assert context.metadata["usage"]["usage_accounting_complete"] is False
    assert context.n_input_tokens is context.n_cache_tokens is context.n_output_tokens is None
    assert "report_unavailable" not in context.metadata
