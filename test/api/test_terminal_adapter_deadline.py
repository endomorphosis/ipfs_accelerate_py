"""One Harbor run budget includes transport, execution and retained evidence.

Authored clocks and cancellable environment doubles make no provider, Docker,
database or verifier calls. Counter records are fixtures, not benchmark usage.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest
from benchmarks.agent_supervisor.container_coding import (
    full_supervisor_harbor_agent as adapter,
)
from harbor.models.agent.context import AgentContext


class Clock:
    value = 0.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


def _agent(tmp_path, *, arm="no-index", contract=None):
    archive = tmp_path / "archive"
    archive.mkdir()
    from benchmarks.agent_supervisor.container_coding import terminal_worker_capability as capability
    # Authored transport fixture; the real policy checks current source pins.
    # This is not a deployed or provider-qualified runtime archive.
    held = capability._current_capability()
    (archive / "manifest.json").write_text(json.dumps({
        "archive_sha256": "authored-runtime", "learned_requirements": [],
        capability.KEY: held,
        "files": [{"path": name, "sha256": digest}
                  for name, digest in held["source_sha256"].items()],
    }))
    return adapter.FullSupervisorAgent(
        logs_dir=tmp_path / "logs", model_name=adapter.MODEL,
        runtime_archive=str(archive), arm=arm, intent_requirement_contract=contract,
        agent_timeout_seconds=300,
    )


def _report(*, usage=False):
    report = {"task_completed": False, "seconds": 56, "phases": {},
              "remaining_processes": 0, "provider_invocations": []}
    if usage:
        report["provider_invocations"] = [{
            "schema": "router-implementation-invocation@1", "invocation_id": "fixture-coding",
            "provider": "codex_cli", "model": "gpt-5.6-sol", "workspace": "/isolated/fixture",
            "status": "failed", "phase": "coding",
            "usage": {"thread_id": "authored-session-123456789", "model_id": "gpt-5.6-sol"},
            "native_rollout_usage": {
                "schema": "native-codex-cumulative-usage@1", "thread_id": "authored-session-123456789",
                "rollout_sha256": hashlib.sha256(b"authored interrupted usage").hexdigest(),
                "observed_token_count_records": 1, "usage_available": True,
                "task_complete_observed": False, "cache_included_in_input": True,
                "usage": {"input_tokens": 100, "cached_input_tokens": 80,
                          "output_tokens": 20, "total_tokens": 120},
            },
        }]
    return report


class Environment:
    def __init__(self, clock, *, costs=None, failures=None, report=None):
        self.clock = clock
        self.costs = costs or {}
        self.failures = failures or {}
        self.report = report or _report()
        self.events = []
        self.commands = []
        self.downloads = []

    async def perform(self, name):
        self.events.append(name)
        self.clock.advance(self.costs.get(name, 0))
        await asyncio.sleep(0)
        if name in self.failures:
            raise self.failures[name]

    async def upload_file(self, _source, target):
        name = "requirement_upload" if target.endswith("intent-requirements.json") else "instruction_upload"
        await self.perform(name)

    async def exec(self, **kwargs):
        self.commands.append({**kwargs, "argv": shlex.split(kwargs["command"])})
        await self.perform("driver")
        return SimpleNamespace(return_code=1, stdout="fixture stdout", stderr="fixture stderr")

    async def download_file(self, source, target):
        self.downloads.append(source)
        report = source.endswith("-result.json")
        await self.perform("report_export" if report else "admission_export")
        Path(target).write_text(json.dumps(self.report if report else {"public": "authored admission"}))


def _install(monkeypatch, clock, environment):
    monkeypatch.setattr(adapter, "monotonic", clock)

    async def capture(*_):
        await environment.perform("public_input_capture")
        return {"status": "authored baseline"}

    async def export(*_):
        await environment.perform("public_output_export")
        return {"status": "authored public output"}

    monkeypatch.setattr(adapter, "capture_public_inputs", capture)
    monkeypatch.setattr(adapter, "export_public_outputs", export)


@pytest.mark.parametrize("arm", ["full", "no-index"])
def test_uploads_capture_driver_and_exports_share_original_budget(tmp_path, monkeypatch, arm):
    clock = Clock()
    costs = {"instruction_upload": 7, "requirement_upload": 11, "public_input_capture": 3,
             "driver": 150, "report_export": 4, "admission_export": 3, "public_output_export": 2}
    environment = Environment(clock, costs=costs)
    _install(monkeypatch, clock, environment)
    agent = _agent(tmp_path, arm=arm, contract={"schema": "authored transport fixture"})
    context = AgentContext()
    asyncio.run(agent.run("Literal public instruction", environment, context))

    assert environment.events == list(costs)
    command = environment.commands[0]
    driver_seconds = int(command["argv"][command["argv"].index("--timeout-seconds") + 1])
    assert driver_seconds == 244  # 300 - 21 spent - 20 exports - 5 receipt - 10 exec grace.
    assert command["timeout_sec"] == driver_seconds + 10
    state = command["argv"][command["argv"].index("--state") + 1]
    assert state == adapter.ROOT + "/state/benchmark-" + context.metadata["adapter_run_id"]
    assert environment.downloads == [state + "-result.json", state + "/admission.json"]
    assert driver_seconds >= 90 and driver_seconds <= 285
    assert context.metadata["adapter_deadline"]["budget_seconds"] == 300
    assert context.metadata["adapter_deadline"]["seconds"] == sum(costs.values())
    assert context.metadata["adapter_deadline"]["remaining_seconds"] == 120
    assert context.metadata["adapter_deadline"]["stages"]["driver"]["timeout_seconds"] == 254
    assert context.metadata["task_completed"] is False
    assert context.metadata["signed_admission_exported"] is True
    assert context.metadata["public_output_evidence"]["status"] == "authored public output"
    assert (agent.logs_dir / "supervisor.stdout").read_text() == "fixture stdout"
    assert (agent.logs_dir / "instruction.md").read_text() == "Literal public instruction"


@pytest.mark.parametrize("spent, dispatch", [(175, True), (175.5, False)])
def test_minimum_driver_budget_is_not_clamped_or_reset(tmp_path, monkeypatch, spent, dispatch):
    clock = Clock()
    environment = Environment(clock, costs={"instruction_upload": spent})
    _install(monkeypatch, clock, environment)
    agent, context = _agent(tmp_path), AgentContext()
    if dispatch:
        asyncio.run(agent.run("Public instruction", environment, context))
        argv = environment.commands[0]["argv"]
        assert argv[argv.index("--timeout-seconds") + 1] == "90"
        assert environment.commands[0]["timeout_sec"] == 100
    else:
        with pytest.raises(adapter._AdapterBudgetExhausted, match="before driver"):
            asyncio.run(agent.run("Public instruction", environment, context))
        assert environment.commands == []
        assert context.metadata["adapter_deadline"]["exhaustion_reason"] == "insufficient_driver_budget"
        assert context.metadata["adapter_deadline"]["exhausted_stage"] == "driver"
        assert context.metadata["public_output_evidence"]["status"] == "authored public output"


def test_timeout_keeps_partial_usage_and_public_outputs_in_reserved_time(tmp_path, monkeypatch):
    clock = Clock()
    environment = Environment(clock, costs={"instruction_upload": 1, "public_input_capture": 2,
        "driver": 272, "report_export": 4, "admission_export": 4, "public_output_export": 4},
        failures={"driver": TimeoutError("authored interrupted driver")}, report=_report(usage=True))
    _install(monkeypatch, clock, environment)
    agent, context = _agent(tmp_path), AgentContext()
    with pytest.raises(TimeoutError, match="interrupted driver"):
        asyncio.run(agent.run("Public instruction", environment, context))
    assert environment.commands[0]["timeout_sec"] == 272
    assert environment.events[-3:] == ["report_export", "admission_export", "public_output_export"]
    assert context.metadata["adapter_deadline"]["seconds"] == 287
    assert context.metadata["driver_error_type"] == "TimeoutError"
    assert context.metadata["usage"]["input_tokens"] == 100
    assert context.metadata["usage"]["total_tokens"] == 120
    assert context.metadata["usage"]["native_scoped_usage_complete_observed"] is False
    assert context.metadata["usage"]["harbor_usage_exportable"] is False
    assert context.metadata["usage"]["usage_accounting_complete"] is False
    assert context.n_input_tokens is context.n_cache_tokens is context.n_output_tokens is None
    assert context.metadata["public_output_evidence"]["status"] == "authored public output"


@pytest.mark.parametrize("stage", ["instruction_upload", "requirement_upload"])
def test_failed_upload_still_finalizes_without_starting_driver(tmp_path, monkeypatch, stage):
    clock = Clock()
    environment = Environment(clock, costs={stage: 3}, failures={stage: RuntimeError("authored upload failure")})
    _install(monkeypatch, clock, environment)
    agent = _agent(tmp_path, contract={"schema": "authored transport fixture"})
    context = AgentContext()
    with pytest.raises(RuntimeError, match="upload failure"):
        asyncio.run(agent.run("Public instruction", environment, context))
    assert environment.commands == []
    assert "public_input_capture" not in environment.events
    assert environment.events[-3:] == ["report_export", "admission_export", "public_output_export"]
    assert context.metadata["adapter_deadline"]["stages"][stage]["status"] == "failed"
    assert "driver_error_type" not in context.metadata


@pytest.mark.parametrize("stage", ["instruction_upload", "driver"])
def test_cancellation_cancels_environment_then_exports_and_reraises(tmp_path, monkeypatch, stage):
    clock = Clock()
    entered = asyncio.Event()
    stopped = []

    class CancellableEnvironment(Environment):
        async def perform(self, name):
            await super().perform(name)
            if name == stage:
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    stopped.append(name)

    environment = CancellableEnvironment(clock)
    _install(monkeypatch, clock, environment)
    agent, context = _agent(tmp_path), AgentContext()

    async def scenario():
        run = asyncio.create_task(agent.run("Public instruction", environment, context))
        await entered.wait()
        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run
        assert not [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]

    asyncio.run(scenario())
    assert stopped == [stage]
    assert environment.events[-3:] == ["report_export", "admission_export", "public_output_export"]
    assert context.metadata["adapter_deadline"]["stages"][stage]["status"] == "cancelled"
    assert context.metadata["public_output_evidence"]["status"] == "authored public output"


def test_spent_deadline_starts_no_further_export_attempts(tmp_path, monkeypatch):
    clock = Clock()
    # A late executor failure cannot grant fresh budgets to subsequent exports.
    environment = Environment(clock, costs={"driver": 300}, failures={"driver": TimeoutError("late executor")})
    _install(monkeypatch, clock, environment)
    agent, context = _agent(tmp_path), AgentContext()
    with pytest.raises(TimeoutError, match="late executor"):
        asyncio.run(agent.run("Public instruction", environment, context))
    assert environment.events == ["instruction_upload", "public_input_capture", "driver"]
    receipt = context.metadata["adapter_deadline"]
    for stage in ("report_export", "admission_export", "public_output_export"):
        assert receipt["stages"][stage]["status"] == "not_started_budget_exhausted"
    assert receipt["budget_exhausted"] is True
    assert receipt["exhausted_stage"] == "report_export"
    assert receipt["deadline_exhausted"] is True
    assert context.metadata["signed_admission_exported"] is False


def test_failed_early_run_never_reads_stale_previous_report(tmp_path, monkeypatch):
    clock = Clock()
    old_report = _report(usage=True)
    old_report["task_completed"] = True
    requested = []

    class StaleEnvironment(Environment):
        async def download_file(self, source, target):
            requested.append(source)
            if source == adapter.ROOT + "/state/benchmark-result.json":
                Path(target).write_text(json.dumps(old_report))
            else:
                raise FileNotFoundError("current run produced no public report")

    environment = StaleEnvironment(clock, failures={"instruction_upload": RuntimeError("upload failed")})
    _install(monkeypatch, clock, environment)
    agent, context = _agent(tmp_path), AgentContext()
    context.n_input_tokens, context.n_cache_tokens, context.n_output_tokens = 100, 80, 20
    with pytest.raises(RuntimeError, match="upload failed"):
        asyncio.run(agent.run("Public instruction", environment, context))
    run_id = context.metadata["adapter_run_id"]
    assert len(run_id) == 32 and int(run_id, 16) >= 0
    state = adapter.ROOT + "/state/benchmark-" + run_id
    assert requested == [state + "-result.json", state + "/admission.json"]
    assert environment.commands == []
    assert context.metadata["task_completed"] is False
    assert context.metadata["report_unavailable"] == "FileNotFoundError"
    assert "usage" not in context.metadata
    assert context.n_input_tokens is context.n_cache_tokens is context.n_output_tokens is None


def test_finalization_waits_shrink_and_do_not_create_exhausted_coroutines(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(adapter, "monotonic", clock)
    budget = adapter._AdapterRunDeadline({}, agent_timeout_seconds=300)
    clock.advance(293)
    calls = []

    async def operation():
        calls.append("called")
        clock.advance(1)

    async def scenario():
        await budget.stage("first", operation, maximum=5)
        await budget.stage("second", operation, maximum=5)
        with pytest.raises(adapter._AdapterBudgetExhausted):
            await budget.stage("third", operation, maximum=5)

    asyncio.run(scenario())
    assert calls == ["called", "called"]
    assert budget.receipt["stages"]["first"]["timeout_seconds"] == 2
    assert budget.receipt["stages"]["second"]["timeout_seconds"] == 1
    assert budget.receipt["stages"]["third"]["status"] == "not_started_budget_exhausted"


def test_wait_timeout_cancels_stage_operation_before_returning(monkeypatch):
    monkeypatch.setattr(adapter, "monotonic", Clock())
    budget = adapter._AdapterRunDeadline({}, agent_timeout_seconds=300)
    stopped = []

    async def operation():
        try:
            await asyncio.Event().wait()
        finally:
            stopped.append(True)

    async def scenario():
        with pytest.raises(TimeoutError):
            await budget.stage("bounded", operation, maximum=0.001)
        assert not [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]

    asyncio.run(scenario())
    assert stopped == [True]
    assert budget.receipt["stages"]["bounded"]["status"] == "timed_out"


def test_invalid_semantic_options_still_fail_before_environment_access(tmp_path, monkeypatch):
    clock = Clock()
    environment = Environment(clock)
    _install(monkeypatch, clock, environment)
    agent = _agent(tmp_path)
    agent.semantic_transport_schema = "supervisor-semantic-router-input@2"
    with pytest.raises(ValueError, match="full indexed arm"):
        asyncio.run(agent.run("Public instruction", environment, AgentContext()))
    assert environment.events == []
