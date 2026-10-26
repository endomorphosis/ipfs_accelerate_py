"""Authored Harbor transport checks; no provider or remote publisher runs.

The source/snapshot helpers and adapter deadline execute locally. Environment
calls return authored receipts, while separate tests exercise publisher code.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
import shlex
import stat
from types import SimpleNamespace

import pytest
from benchmarks.agent_supervisor.container_coding.test_terminal_deployment import _supervisor_manifest
from harbor.models.agent.context import AgentContext

from benchmarks.agent_supervisor.container_coding import full_supervisor_harbor_agent as adapter
from benchmarks.agent_supervisor.container_coding import semantic_decision_nomination_delivery as delivery
from test.api.test_current_decision_configuration import _archive as _current_decision_archive


def _source(tmp_path):
    value = {
        "schema": delivery.NOMINATION_SCHEMA,
        "policy": {"required_capsule_ids": ["required-λ"],
                   "optional_capsule_ids": ["optional-body"],
                   "selected_optional_capsule_ids": []},
    }
    raw = (json.dumps(value, ensure_ascii=False, indent=2).replace("\n", "\r\n") + "\r\n").encode()
    source = tmp_path / "nomination café.json"
    source.write_bytes(raw)
    return source, raw, hashlib.sha256(raw).hexdigest()


class _Environment:
    def __init__(self, *, source, raw, digest, logs, failure=None, clock=None, exhaust=None,
                 change_source=False):
        self.source, self.raw, self.digest, self.logs = source, raw, digest, logs
        self.failure, self.clock, self.exhaust = failure, clock, exhaust
        self.change_source = change_source
        self.events, self.commands, self.uploads, self.downloads = [], [], [], []
        self.snapshot = None

    async def upload_file(self, source, target):
        stage = "nomination_upload" if target.endswith("/.upload.json") else "instruction_upload"
        self.events.append(stage)
        self.uploads.append((Path(source), target))
        if stage == "nomination_upload":
            self.snapshot = Path(source)
            assert self.snapshot.is_absolute() and self.snapshot.read_bytes() == self.raw
            assert stat.S_IMODE(self.snapshot.stat().st_mode) == 0o444
            if self.failure == "upload_failure":
                raise RuntimeError("authored upload failure")

    async def exec(self, **kwargs):
        arguments = shlex.split(kwargs["command"])
        self.commands.append((arguments, kwargs))
        if kwargs["user"] != "root":
            self.events.append("driver")
            return SimpleNamespace(stdout="", stderr="", return_code=0)
        assert arguments[:3] == [adapter.PYTHON, "-P", "-c"]
        assert arguments[3] == delivery.NOMINATION_DELIVERY_CODE
        operation, root, run_id, digest, count = arguments[4:]
        self.events.append("nomination_" + operation)
        assert root == adapter.ROOT and digest == self.digest and int(count) == len(self.raw)
        assert kwargs["cwd"] == adapter.ROOT and kwargs["timeout_sec"] == 15
        assert kwargs["env"] == adapter.runtime_environment()
        locations = delivery.nomination_delivery_arguments(root=root, run_id=run_id,
            expected_sha256=digest, expected_bytes=int(count))
        receipt = {"schema": delivery.DELIVERY_SCHEMA, "operation": operation,
                   "published": operation == "publish", "artifact": locations["artifact"],
                   "sha256": digest, "bytes": int(count),
                   "provider_called": False, "completion_authority": False}
        if operation == "prepare":
            if self.change_source:
                self.source.write_bytes(b"source changed after capture")
            if self.failure == "snapshot_changed":
                snapshot, = self.logs.absolute().glob("decision-nomination-*/*.json")
                snapshot.chmod(0o644)
                snapshot.write_bytes(self.raw + b" ")
            if self.exhaust == "before_upload":
                self.clock[0] = adapter.execution_budget()["harbor_seconds"] - adapter.ADAPTER_FINALIZATION_SECONDS + 1
        if operation == "publish" and self.exhaust == "before_driver":
            self.clock[0] = adapter.execution_budget()["exec_seconds"] - adapter.ADAPTER_FINALIZATION_SECONDS - 10
        selected = (self.failure or "").removeprefix(operation + "_")
        if self.failure and self.failure.startswith(operation + "_"):
            if selected == "failure":
                raise RuntimeError("authored command failure")
            if selected == "nonzero":
                return SimpleNamespace(stdout=json.dumps(receipt), stderr="authored", return_code=1)
            if selected == "invalid_json":
                return SimpleNamespace(stdout="{", stderr="", return_code=0)
            if selected == "wrong_operation":
                receipt["operation"] = "publish" if operation == "prepare" else "prepare"
            elif selected == "wrong_published":
                receipt["published"] = operation != "publish"
            elif selected == "wrong_digest":
                receipt["sha256"] = "f" * 64
            elif selected == "extra":
                receipt["untrusted"] = "extra"
            elif selected == "boolean_bytes":
                receipt["bytes"] = True
            elif selected == "duplicate":
                raw_receipt = json.dumps(receipt)[:-1] + ', "published": true}'
                return SimpleNamespace(stdout=raw_receipt, stderr="", return_code=0)
            elif selected == "oversize":
                return SimpleNamespace(stdout=" " * 4097, stderr="", return_code=0)
        return SimpleNamespace(stdout=json.dumps(receipt), stderr="", return_code=0)

    async def download_file(self, source, target):
        self.downloads.append(source)
        self.events.append("final_download")
        raise FileNotFoundError(source)


def _fixture(tmp_path, monkeypatch, *, limits=None, failure=None, clock=None,
             exhaust=None, change_source=False, relative_logs=False):
    source, raw, digest = _source(tmp_path)
    logs = Path("logs") if relative_logs else tmp_path / "logs"
    if relative_logs:
        monkeypatch.chdir(tmp_path)
    options = dict(semantic_decision_nomination_source=source,
                   semantic_decision_nomination_sha256=digest, **(limits or {}))
    archive, _ = _current_decision_archive(tmp_path, options)
    agent = adapter.FullSupervisorAgent(logs_dir=logs, model_name=adapter.resolve_provider_profile()["model"],
        runtime_archive=str(archive), arm="full", **options)
    environment = _Environment(source=source, raw=raw, digest=digest, logs=agent.logs_dir,
        failure=failure, clock=clock, exhaust=exhaust, change_source=change_source)

    async def capture(*args, **kwargs):
        environment.events.append("public_input_capture")
        return {"status": "authored"}

    async def export(*args, **kwargs):
        environment.events.append("public_output_export")
        return {"status": "authored"}

    monkeypatch.setattr(adapter, "capture_public_inputs", capture)
    monkeypatch.setattr(adapter, "export_public_outputs", export)
    return agent, environment, AgentContext()


def _assert_finalized(context, environment):
    assert len(environment.downloads) == 2
    assert environment.downloads[0].endswith("-result.json")
    assert environment.downloads[1].endswith("/admission.json")
    assert environment.events[-1] == "public_output_export"
    assert context.metadata["signed_admission_exported"] is False
    assert context.metadata["task_completed"] is False
    assert context.n_input_tokens is context.n_cache_tokens is context.n_output_tokens is None
    assert "remaining_seconds" in context.metadata["adapter_deadline"]


@pytest.mark.parametrize("limits,change_source,relative_logs", [
    ({}, False, False),
    ({"semantic_decision_max_calls": 2, "semantic_decision_max_total_input_tokens": 30000}, True, False),
    ({}, False, True),
], ids=["default", "custom_source_changed", "relative_logs"])
def test_source_delivery_uses_exact_snapshot_and_only_derived_child_flags(
        tmp_path, monkeypatch, limits, change_source, relative_logs):
    agent, environment, context = _fixture(tmp_path, monkeypatch, limits=limits,
        change_source=change_source, relative_logs=relative_logs)
    observed_stages = {}
    original_stage = adapter._AdapterRunDeadline.stage

    async def record_stage(self, name, operation, *, maximum=None,
                           reserve=adapter.ADAPTER_RECEIPT_MARGIN_SECONDS):
        observed_stages[name] = (maximum, reserve)
        return await original_stage(self, name, operation, maximum=maximum, reserve=reserve)

    monkeypatch.setattr(adapter._AdapterRunDeadline, "stage", record_stage)
    asyncio.run(agent.run("Authored public instruction.", environment, context))
    assert environment.events[:4] == ["nomination_prepare", "nomination_upload",
                                      "nomination_publish", "instruction_upload"]
    assert environment.snapshot.read_bytes() == environment.raw
    transfer = context.metadata["semantic_decision_nomination_delivery"]
    assert transfer["status"] == "published" and transfer["operation"] == "publish"
    assert transfer["published"] is True
    assert transfer["sha256"] == environment.digest and transfer["bytes"] == len(environment.raw)
    assert transfer["provider_called"] is False and transfer["completion_authority"] is False
    diagnostics = sorted(environment.snapshot.parent.glob("publisher-*.json"))
    assert [path.name for path in diagnostics] == ["publisher-prepare.json", "publisher-publish.json"]
    for path in diagnostics:
        assert stat.S_IMODE(path.stat().st_mode) == 0o600 and path.stat().st_nlink == 1
        diagnostic = json.loads(path.read_text())
        assert diagnostic["schema"] == "terminal-semantic-decision-nomination-command@1"
        assert diagnostic["return_code"] == 0 and diagnostic["stderr"] == ""
        assert diagnostic["provider_called"] is False and diagnostic["completion_authority"] is False
        assert json.loads(diagnostic["stdout"])["sha256"] == environment.digest
    assert not any(value == environment.raw or value == environment.raw.decode() for value in transfer.values())
    for stage in ("nomination_prepare", "nomination_upload", "nomination_publish"):
        assert observed_stages[stage] == (15, adapter.ADAPTER_FINALIZATION_SECONDS)
        measured = context.metadata["adapter_deadline"]["stages"][stage]
        assert measured["status"] == "completed" and 0 < measured["timeout_seconds"] <= 15
    driver, kwargs = environment.commands[-1]
    assert kwargs["user"] == "supervisor" and kwargs["cwd"] == "/app"
    flag = "--semantic-decision-nomination-artifact"
    assert driver.count(flag) == 1 and driver[driver.index(flag) + 1] == transfer["artifact"]
    assert driver[driver.index("--semantic-decision-nomination-sha256") + 1] == environment.digest
    assert driver[driver.index("--semantic-decision-max-calls") + 1] == str(limits.get("semantic_decision_max_calls", 3))
    if limits:
        assert driver[driver.index("--semantic-decision-max-total-input-tokens") + 1] == "30000"
    else:
        assert "--semantic-decision-max-total-input-tokens" not in driver
    assert str(environment.source) not in driver
    assert not any(argument.startswith(("--semantic-decision-nomination-source", "--semantic-decision-policy-",
                                        "--semantic-decision-task-", "--task-id", "--task-key")) for argument in driver)
    _assert_finalized(context, environment)


@pytest.mark.parametrize("failure", [
    "prepare_nonzero", "prepare_invalid_json", "prepare_wrong_operation", "prepare_wrong_published",
    "upload_failure", "publish_nonzero", "publish_failure", "publish_wrong_digest",
    "publish_wrong_published", "publish_extra", "publish_duplicate", "publish_boolean_bytes", "publish_oversize",
])
def test_delivery_failure_blocks_driver_and_preserves_bounded_failure_and_finalization(tmp_path, monkeypatch, failure):
    agent, environment, context = _fixture(tmp_path, monkeypatch, failure=failure)
    with pytest.raises((ValueError, RuntimeError)):
        asyncio.run(agent.run("Authored public instruction.", environment, context))
    assert "driver" not in environment.events and "instruction_upload" not in environment.events
    transfer = context.metadata["semantic_decision_nomination_delivery"]
    assert transfer["status"] == "failed" and transfer["published"] is False
    assert transfer["stage"] == "nomination_" + failure.split("_", 1)[0]
    assert transfer["error_type"] in {"ValueError", "RuntimeError", "JSONDecodeError"}
    assert transfer["provider_called"] is False and transfer["completion_authority"] is False
    assert len(json.dumps(transfer).encode()) < 2048
    if failure.startswith("prepare_"):
        assert not environment.uploads and "nomination_publish" not in environment.events
    _assert_finalized(context, environment)


def test_mutated_source_pin_refuses_before_environment_or_metadata_initialization(tmp_path, monkeypatch):
    agent, environment, context = _fixture(tmp_path, monkeypatch)
    previous_metadata = context.metadata
    agent.semantic_decision_nomination_sha256 = "f" * 64
    with pytest.raises(ValueError, match="prepared semantic decision selection changed"):
        asyncio.run(agent.run("Authored public instruction.", environment, context))
    assert environment.commands == environment.uploads == environment.downloads == environment.events == []
    assert context.metadata is previous_metadata
    assert context.n_input_tokens is context.n_cache_tokens is context.n_output_tokens is None


def test_failed_publish_retains_bounded_private_stderr_before_receipt_refusal(tmp_path, monkeypatch):
    agent, environment, context = _fixture(tmp_path, monkeypatch)
    execute = environment.exec
    marker = "benign nomination publisher diagnostic marker"
    output = "y" * 6000 + "benign publisher stdout"
    error = "x" * 6000 + "\n" + marker

    async def fail_publish(**kwargs):
        result = await execute(**kwargs)
        arguments = shlex.split(kwargs["command"])
        if kwargs["user"] == "root" and arguments[4] == "publish":
            return SimpleNamespace(stdout=output, stderr=error, return_code=7)
        return result

    monkeypatch.setattr(environment, "exec", fail_publish)
    with pytest.raises(ValueError, match="nomination delivery command failed"):
        asyncio.run(agent.run("Authored public instruction.", environment, context))
    diagnostic_path = environment.snapshot.parent / "publisher-publish.json"
    diagnostic = json.loads(diagnostic_path.read_text())
    assert stat.S_IMODE(diagnostic_path.stat().st_mode) == 0o600
    assert diagnostic_path.stat().st_size < 65_536
    assert diagnostic["operation"] == "publish" and diagnostic["return_code"] == 7
    assert diagnostic["stderr"].endswith(marker) and diagnostic["stdout"].endswith("benign publisher stdout")
    assert len(diagnostic["stderr"].encode()) <= 4096 and len(diagnostic["stdout"].encode()) <= 4096
    assert diagnostic["stderr_bytes"] == len(error.encode()) and diagnostic["stdout_bytes"] == len(output.encode())
    assert diagnostic["stderr_truncated"] is diagnostic["stdout_truncated"] is True
    assert diagnostic["provider_called"] is False and diagnostic["completion_authority"] is False
    assert "driver" not in environment.events and "instruction_upload" not in environment.events
    transfer = context.metadata["semantic_decision_nomination_delivery"]
    assert transfer["stage"] == "nomination_publish" and transfer["status"] == "failed"
    assert transfer["provider_called"] is False and transfer["published"] is False
    assert marker not in json.dumps(transfer)
    _assert_finalized(context, environment)


def test_changed_host_snapshot_refuses_before_upload(tmp_path, monkeypatch):
    agent, environment, context = _fixture(tmp_path, monkeypatch, failure="snapshot_changed")
    with pytest.raises(ValueError):
        asyncio.run(agent.run("Authored public instruction.", environment, context))
    assert not environment.uploads
    assert [event for event in environment.events if event.startswith("nomination_")] == ["nomination_prepare"]
    transfer = context.metadata["semantic_decision_nomination_delivery"]
    assert transfer["stage"] == "nomination_upload" and transfer["status"] == "failed"
    assert transfer["published"] is False
    _assert_finalized(context, environment)


@pytest.mark.parametrize("exhaust", ["before_upload", "before_driver"])
def test_original_adapter_budget_prevents_dependent_upload_or_driver(tmp_path, monkeypatch, exhaust):
    clock = [0.0]
    monkeypatch.setattr(adapter, "monotonic", lambda: clock[0])
    agent, environment, context = _fixture(tmp_path, monkeypatch, clock=clock, exhaust=exhaust)
    with pytest.raises(adapter._AdapterBudgetExhausted):
        asyncio.run(agent.run("Authored public instruction.", environment, context))
    assert "driver" not in environment.events
    deadline = context.metadata["adapter_deadline"]
    assert deadline["budget_exhausted"] is True
    transfer = context.metadata["semantic_decision_nomination_delivery"]
    if exhaust == "before_upload":
        assert not environment.uploads
        assert transfer["status"] == "failed" and transfer["stage"] == "nomination_upload"
        assert deadline["stages"]["nomination_upload"]["status"] == "not_started_budget_exhausted"
    else:
        assert "nomination_publish" in environment.events and "instruction_upload" in environment.events
        assert transfer["status"] == "published" and transfer["published"] is True
        assert deadline["stages"]["driver"]["status"] == "not_started_budget_exhausted"
    _assert_finalized(context, environment)
