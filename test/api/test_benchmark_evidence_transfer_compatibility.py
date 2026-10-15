"""Authored transport/profile fixtures; no provider, Docker or native admission.

The archive tests use the real closed allowlist/hash verifier. Environment and
clock doubles only exercise adapter transport, cancellation and physical time.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import shlex
from types import SimpleNamespace

import pytest
from harbor.models.agent.context import AgentContext

from benchmarks.agent_supervisor.container_coding import full_supervisor_harbor_agent as adapter
from benchmarks.agent_supervisor.container_coding import benchmark_resource_profile as profiles
from benchmarks.agent_supervisor.container_coding import terminal_context_evidence as evidence
from benchmarks.agent_supervisor.container_coding import terminal_worker_capability as capability
from test.api.test_terminal_context_evidence_delivery import _bundle


class Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value


@pytest.mark.parametrize("profile", [None, *profiles.PROFILES])
def test_profile_physical_deadline_preserves_driver_and_exec_caps(monkeypatch, profile):
    clock = Clock()
    monkeypatch.setattr(adapter, "monotonic", clock)
    limits = profiles.execution_budget(profile)
    metadata = {}
    deadline = adapter._AdapterRunDeadline(metadata, resource_profile=profile)
    grace = limits["exec_seconds"] - limits["driver_seconds"]
    expected = min(limits["driver_seconds"], limits["harbor_seconds"] - 25 - grace)
    assert deadline.driver_seconds() == expected
    assert deadline.driver_seconds() + grace <= limits["exec_seconds"]
    assert deadline.receipt["budget_seconds"] == limits["harbor_seconds"]
    assert deadline.receipt["finalization_reserve_seconds"] == 25
    assert deadline.receipt["driver_exec_grace_seconds"] == grace
    clock.value = limits["harbor_seconds"] - 25 - grace - 90
    assert deadline.driver_seconds() == 90
    clock.value += .125
    with pytest.raises(adapter._AdapterBudgetExhausted):
        deadline.driver_seconds()
    assert metadata["adapter_deadline"]["exhaustion_reason"] == "insufficient_driver_budget"


def test_elapsed_preparation_is_charged_without_resetting_profile(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(adapter, "monotonic", clock)
    clock.value = 21
    deadline = adapter._AdapterRunDeadline({}, started=0)
    assert deadline.driver_seconds() == 244
    assert deadline.receipt["budget_seconds"] == 300
    with pytest.raises(ValueError, match="selected resource profile"):
        adapter._AdapterRunDeadline({}, agent_timeout_seconds=600)


def test_exhausted_export_never_constructs_an_environment_operation(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(adapter, "monotonic", clock)
    deadline = adapter._AdapterRunDeadline({})
    clock.value = 285
    calls = []

    def operation():
        calls.append("constructed")
        raise AssertionError("reserved export must not be constructed")

    with pytest.raises(adapter._AdapterBudgetExhausted):
        asyncio.run(deadline.stage("context_replay_evidence_export", operation,
                                  maximum=5, reserve=15))
    assert calls == []
    assert deadline.receipt["stages"]["context_replay_evidence_export"]["status"] == "not_started_budget_exhausted"


class Transport:
    def __init__(self, raw, *, effect=None):
        self.raw, self.effect = raw, effect
        self.sources = []

    async def download_file(self, source, target):
        self.sources.append(source)
        if self.effect is not None:
            await self.effect(Path(target))
        else:
            Path(target).write_bytes(self.raw)


def _transfer(tmp_path, transport, receipt):
    return asyncio.run(adapter._download_context_replay_evidence(
        transport, state="/opt/ipfs-supervisor/state/authored", receipt=receipt, output=tmp_path))


def test_real_closed_verifier_retains_only_exact_attested_sibling(tmp_path):
    state = "/opt/ipfs-supervisor/state/authored"
    raw, receipt = _bundle(state)
    transport = Transport(raw)
    result = _transfer(tmp_path, transport, receipt)
    assert transport.sources == [state + "-context-evidence.tar"]
    assert result["status"] == "delivered" and result["verified"] is True
    target = Path(result["artifact"])
    assert target.read_bytes() == raw and target.stat().st_mode & 0o777 == 0o444
    assert target.stat().st_nlink == 1
    assert json.loads((tmp_path / "context-replay-evidence-receipt.json").read_text()) == result
    assert not list(tmp_path.glob(".context-replay-evidence-*.tar"))


@pytest.mark.parametrize("changes", [
    {"artifact": "/opt/ipfs-supervisor/state/foreign-context-evidence.tar"},
    {"state_origin": "/opt/ipfs-supervisor/state/foreign"},
    {"execution_authority": True}, {"private_field": "AUTHORED_PRIVATE_MARKER"},
])
def test_owner_receipt_refuses_foreign_or_expanded_scope_before_download(tmp_path, changes):
    raw, receipt = _bundle("/opt/ipfs-supervisor/state/authored")
    receipt.update(changes)
    transport = Transport(raw)
    with pytest.raises(ValueError):
        _transfer(tmp_path, transport, receipt)
    assert transport.sources == [] and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("change", ["hash", "private_member", "symlink"])
def test_invalid_download_cannot_publish_or_leave_temporary_files(tmp_path, change):
    raw, receipt = _bundle("/opt/ipfs-supervisor/state/authored", private_member=change == "private_member")
    if change == "hash":
        receipt["sha256"] = "0" * 64
    protected = tmp_path / "authored-protected"
    protected.write_bytes(b"unchanged")

    async def symlink(target):
        target.unlink()
        target.symlink_to(protected)

    transport = Transport(raw, effect=symlink if change == "symlink" else None)
    with pytest.raises(ValueError):
        _transfer(tmp_path, transport, receipt)
    assert protected.read_bytes() == b"unchanged"
    assert not (tmp_path / "context-replay-evidence.tar").exists()
    assert not (tmp_path / "context-replay-evidence-receipt.json").exists()
    assert not list(tmp_path.glob(".context-replay-evidence-*.tar"))


@pytest.mark.parametrize("racing_name", ["context-replay-evidence.tar", "context-replay-evidence-receipt.json"])
def test_exclusive_publication_preserves_racing_artifact_and_rolls_back_owned_archive(tmp_path, racing_name):
    raw, receipt = _bundle("/opt/ipfs-supervisor/state/authored")
    racing = tmp_path / racing_name

    async def effect(target):
        target.write_bytes(raw)
        racing.write_bytes(b"authored racing artifact")

    with pytest.raises(FileExistsError):
        _transfer(tmp_path, Transport(raw, effect=effect), receipt)
    assert racing.read_bytes() == b"authored racing artifact"
    other = tmp_path / ("context-replay-evidence-receipt.json" if racing_name.endswith(".tar")
                        else "context-replay-evidence.tar")
    assert not other.exists()
    assert not list(tmp_path.glob(".context-replay-evidence-*.tar"))


def test_cancelled_transfer_stops_operation_and_removes_partial_owned_file(tmp_path):
    raw, receipt = _bundle("/opt/ipfs-supervisor/state/authored")
    entered = asyncio.Event()
    stopped = []

    async def effect(target):
        target.write_bytes(raw[:7])
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.append(True)

    async def scenario():
        task = asyncio.create_task(adapter._download_context_replay_evidence(
            Transport(raw, effect=effect), state=receipt["state_origin"], receipt=receipt, output=tmp_path))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not [item for item in asyncio.all_tasks() if item is not asyncio.current_task()]

    asyncio.run(scenario())
    assert stopped == [True] and list(tmp_path.iterdir()) == []


def _agent(tmp_path, profile):
    # A declared test manifest with actual current capability source pins is
    # checked by the real archive policy. It is not a deployed runtime archive.
    archive = tmp_path / "archive"
    archive.mkdir()
    held = capability._current_capability()
    manifest = {"archive_sha256": "a" * 64, "learned_requirements": [],
                capability.KEY: held,
                "files": [{"path": name, "sha256": digest}
                          for name, digest in held["source_sha256"].items()]}
    (archive / "manifest.json").write_text(json.dumps(manifest))
    return adapter.FullSupervisorAgent(logs_dir=tmp_path / "logs", model_name=adapter.MODEL,
        runtime_archive=str(archive), arm="no-index", resource_profile=profile)


class RunEnvironment:
    def __init__(self, clock, *, transfer_error=False, driver_error=None):
        self.clock, self.transfer_error, self.driver_error = clock, transfer_error, driver_error
        self.events, self.commands, self.downloads = [], [], []
        self.report = {"task_completed": False, "seconds": 10, "phases": {},
                       "remaining_processes": 0, "provider_invocations": []}

    async def upload_file(self, source, target):
        self.events.append("instruction_upload")
        self.clock.value += 2

    async def exec(self, **kwargs):
        self.commands.append(kwargs)
        self.events.append("driver")
        self.clock.value += 10
        argv = shlex.split(kwargs["command"])
        self.state = argv[argv.index("--state") + 1]
        self.raw, receipt = _bundle(self.state)
        self.report["context_replay_evidence"] = receipt
        if self.driver_error is not None:
            raise self.driver_error
        return SimpleNamespace(return_code=1, stdout="authored", stderr="authored")

    async def download_file(self, source, target):
        self.downloads.append(source)
        if source.endswith("-context-evidence.tar"):
            self.events.append("context_replay_evidence_export")
            if self.transfer_error:
                Path(target).write_bytes(b"authored partial transfer")
                raise OSError("AUTHORED_PRIVATE_ERROR_MARKER")
            Path(target).write_bytes(self.raw)
        elif source.endswith("-result.json"):
            self.events.append("report_export")
            Path(target).write_text(json.dumps(self.report))
        else:
            assert source == self.state + "/admission.json"
            self.events.append("admission_export")
            Path(target).write_text('{"authored_public_envelope":true}')
        self.clock.value += 1


def _install(monkeypatch, clock, environment):
    monkeypatch.setattr(adapter, "monotonic", clock)

    async def capture(*_):
        environment.events.append("public_input_capture")
        clock.value += 3
        return {"status": "authored"}

    async def export(*_):
        environment.events.append("public_output_export")
        clock.value += 1
        return {"status": "authored"}

    monkeypatch.setattr(adapter, "capture_public_inputs", capture)
    monkeypatch.setattr(adapter, "export_public_outputs", export)


@pytest.mark.parametrize("profile", [None, profiles.EXTENDED_SOURCE384_PROFILE])
def test_actual_adapter_profile_transport_and_four_exports_share_one_deadline(tmp_path, monkeypatch, profile):
    clock = Clock()
    environment = RunEnvironment(clock)
    _install(monkeypatch, clock, environment)
    agent, context = _agent(tmp_path, profile), AgentContext()
    asyncio.run(agent.run("Authored public instruction", environment, context))
    limits = profiles.execution_budget(profile)
    grace = limits["exec_seconds"] - limits["driver_seconds"]
    expected = min(limits["driver_seconds"], limits["harbor_seconds"] - 5 - 25 - grace)
    command = environment.commands[0]
    argv = shlex.split(command["command"])
    assert int(argv[argv.index("--timeout-seconds") + 1]) == expected
    assert command["timeout_sec"] == expected + grace
    assert context.metadata["adapter_deadline"]["seconds"] == 19
    assert context.metadata["adapter_deadline"]["budget_seconds"] == limits["harbor_seconds"]
    assert context.metadata["context_replay_evidence"]["verified"] is True
    assert environment.events[-4:] == ["report_export", "context_replay_evidence_export", "admission_export", "public_output_export"]
    assert environment.downloads == [environment.state + "-result.json",
        environment.state + "-context-evidence.tar", environment.state + "/admission.json"]
    assert environment.state.endswith("benchmark-" + context.metadata["adapter_run_id"])
    assert not list(agent.logs_dir.glob(".context-replay-evidence-*.tar"))


def test_delivery_error_keeps_report_and_remaining_exports_without_error_text(tmp_path, monkeypatch):
    clock = Clock()
    environment = RunEnvironment(clock, transfer_error=True)
    _install(monkeypatch, clock, environment)
    agent, context = _agent(tmp_path, None), AgentContext()
    asyncio.run(agent.run("Authored public instruction", environment, context))
    assert context.metadata["context_replay_evidence"] == {
        "status": "unavailable", "reason": "delivery_unavailable", "verified": False, "error_type": "OSError"}
    assert "AUTHORED_PRIVATE_ERROR_MARKER" not in json.dumps(context.metadata)
    assert context.metadata["signed_admission_exported"] is True
    assert context.metadata["public_output_evidence"]["status"] == "authored"
    assert json.loads((agent.logs_dir / "supervisor-result.json").read_text()) == environment.report
    assert not (agent.logs_dir / "context-replay-evidence.tar").exists()
    assert not list(agent.logs_dir.glob(".context-replay-evidence-*.tar"))


def test_original_driver_error_is_reraised_after_evidence_and_public_exports(tmp_path, monkeypatch):
    clock = Clock()
    original = TimeoutError("authored driver timeout")
    environment = RunEnvironment(clock, driver_error=original)
    _install(monkeypatch, clock, environment)
    agent, context = _agent(tmp_path, None), AgentContext()
    with pytest.raises(TimeoutError) as raised:
        asyncio.run(agent.run("Authored public instruction", environment, context))
    assert raised.value is original
    assert context.metadata["context_replay_evidence"]["verified"] is True
    assert context.metadata["signed_admission_exported"] is True
    assert context.metadata["public_output_evidence"]["status"] == "authored"


def test_unselected_usage_still_uses_unchanged_native_projection():
    report = {"provider_invocations": [], "task_completed": False}
    assert adapter.measured_usage(report) == adapter._measured_native_usage(report)
