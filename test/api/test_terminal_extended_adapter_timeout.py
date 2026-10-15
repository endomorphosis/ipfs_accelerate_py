"""Extended run budgets preserve cleanup, receipt binding and old profiles."""

import asyncio
import json

import pytest
from benchmarks.agent_supervisor.container_coding import full_supervisor_harbor_agent as adapter
from benchmarks.agent_supervisor.container_coding.timeout_policy import (
    DEFAULT_AGENT_TIMEOUT_SECONDS,
    MAX_AGENT_TIMEOUT_SECONDS,
)
from harbor.models.agent.context import AgentContext

from test.api.test_terminal_adapter_deadline import Clock, Environment, _install


@pytest.mark.parametrize("seconds,expected_driver", [(300, 265), (450, 415), (600, 565)])
def test_actual_adapter_receives_selected_budget_and_retains_all_reserves(
    tmp_path, monkeypatch, seconds, expected_driver,
):
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "manifest.json").write_text(json.dumps({
        "archive_sha256": "authored-extended-timeout", "learned_requirements": [],
    }))
    clock = Clock()
    environment = Environment(clock)
    _install(monkeypatch, clock, environment)
    context = AgentContext()
    agent = adapter.FullSupervisorAgent(
        logs_dir=tmp_path / "logs", model_name="gpt-5.6-sol", runtime_archive=str(archive),
        arm="no-index", agent_timeout_seconds=seconds,
    )
    asyncio.run(agent.run("Literal public task", environment, context))
    command = environment.commands[0]
    argv = command["argv"]
    assert int(argv[argv.index("--timeout-seconds") + 1]) == expected_driver
    assert command["timeout_sec"] == expected_driver + 10
    assert context.metadata["adapter_deadline"]["budget_seconds"] == seconds
    assert context.metadata["adapter_deadline"]["finalization_reserve_seconds"] == 25
    assert environment.events[-3:] == ["report_export", "admission_export", "public_output_export"]
    assert context.metadata["task_completed"] is False


def test_default_adapter_budget_is_ten_minutes_and_old_deadline_still_expands(monkeypatch):
    monkeypatch.setattr(adapter, "monotonic", Clock())
    default = adapter._AdapterRunDeadline({})
    assert DEFAULT_AGENT_TIMEOUT_SECONDS == MAX_AGENT_TIMEOUT_SECONDS == 600
    assert default.receipt["budget_seconds"] == 600
    assert default.driver_seconds() == 565
    legacy = adapter._AdapterRunDeadline({}, agent_timeout_seconds=300)
    assert legacy.driver_seconds() == 265


def test_extended_budget_charges_elapsed_transport_without_restarting_clock(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(adapter, "monotonic", clock)
    deadline = adapter._AdapterRunDeadline({})
    clock.advance(107)
    assert deadline.driver_seconds() == 458
    clock.advance(313.5)
    assert deadline.driver_seconds() == 144
    clock.advance(60)
    with pytest.raises(adapter._AdapterBudgetExhausted):
        deadline.driver_seconds()


def test_smallest_accepted_budget_refuses_dispatch_when_transport_consumes_reserves(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(adapter, "monotonic", clock)
    deadline = adapter._AdapterRunDeadline({}, agent_timeout_seconds=120)
    clock.advance(0.1)
    with pytest.raises(adapter._AdapterBudgetExhausted):
        deadline.driver_seconds()
    assert deadline.receipt["stages"]["driver"] == {
        "status": "not_started_budget_exhausted", "available_driver_seconds": 84,
    }
    assert deadline.receipt["exhaustion_reason"] == "insufficient_driver_budget"


@pytest.mark.parametrize("seconds", [True, False, 119, 601, 600.0, "600", None])
def test_invalid_extended_budget_refuses_before_framework_or_environment(monkeypatch, seconds):
    monkeypatch.setattr(adapter.BaseAgent, "__init__", lambda *_a, **_k: pytest.fail(
        "invalid budget reached framework initialization"))
    with pytest.raises(ValueError, match="agent timeout"):
        adapter.FullSupervisorAgent(runtime_archive="/unread", agent_timeout_seconds=seconds)
    with pytest.raises(ValueError, match="agent timeout"):
        adapter._AdapterRunDeadline({}, agent_timeout_seconds=seconds)
