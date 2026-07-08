#!/usr/bin/env python3
"""Contract tests for Docker execution persistence/provenance hooks."""

import subprocess

from ipfs_accelerate_py.docker_executor import DockerExecutor, DockerExecutionConfig


class _FakeStorage:
    def __init__(self):
        self.records = []
        self.retrieve_calls = []

    def store(self, data, filename=None, pin=False):
        self.records.append({"data": data, "filename": filename, "pin": pin})
        return f"cid-{len(self.records)}"

    def retrieve(self, cid):
        self.retrieve_calls.append(cid)
        if cid == "cid-model":
            return b"model-bytes"
        return None


class _FakeDatasetsManager:
    def __init__(self):
        self.events = []
        self.provenance = []

    def log_event(self, event_type, data, level="INFO", category="GENERAL"):
        self.events.append({
            "event_type": event_type,
            "data": data,
            "level": level,
            "category": category,
        })
        return True

    def track_provenance(self, operation, data, record_type="TRANSFORMATION"):
        self.provenance.append({
            "operation": operation,
            "data": data,
            "record_type": record_type,
        })
        return "cid-prov"


class _FakeProvenanceLogger:
    def __init__(self):
        self.records = []

    def log_transformation(self, operation, data, input_cid=None, output_cid=None):
        self.records.append({
            "operation": operation,
            "data": data,
            "input_cid": input_cid,
            "output_cid": output_cid,
        })
        return "cid-prov"


def test_docker_executor_persists_artifacts_and_provenance(monkeypatch):
    monkeypatch.setattr(DockerExecutor, "_verify_docker_available", lambda self: None)

    storage = _FakeStorage()
    datasets = _FakeDatasetsManager()
    provenance = _FakeProvenanceLogger()
    executor = DockerExecutor(
        storage=storage,
        datasets_manager=datasets,
        provenance_logger=provenance,
    )

    monkeypatch.setattr(
        executor,
        "_build_docker_command",
        lambda config: ["docker", "run", config.image, "echo", "hello"],
    )
    monkeypatch.setattr(
        executor,
        "_execute_capture",
        lambda cmd, timeout: subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="hello from docker\n",
            stderr="",
        ),
    )

    result = executor.execute_container(
        DockerExecutionConfig(
            image="python:3.12-slim",
            command=["echo", "hello"],
            environment={"MODEL_ID": "demo"},
        )
    )

    assert result.success is True
    assert result.exit_code == 0
    assert result.output_cid == "cid-1"
    assert result.provenance_cid == "cid-prov"
    assert result.metadata["image"] == "python:3.12-slim"
    assert result.metadata["output_cid"] == "cid-1"
    assert storage.records[0]["filename"].endswith(".json")
    assert "hello from docker" in storage.records[0]["data"]
    assert datasets.events[0]["event_type"] == "container_execution_completed"
    assert datasets.events[0]["data"]["output_cid"] == "cid-1"
    assert provenance.records[0]["operation"] == "docker_execution"
    assert provenance.records[0]["output_cid"] == "cid-1"


def test_docker_executor_materializes_model_artifact_by_cid(monkeypatch):
    monkeypatch.setattr(DockerExecutor, "_verify_docker_available", lambda self: None)

    storage = _FakeStorage()
    datasets = _FakeDatasetsManager()
    provenance = _FakeProvenanceLogger()
    executor = DockerExecutor(
        storage=storage,
        datasets_manager=datasets,
        provenance_logger=provenance,
    )

    seen = {}

    def _capture_command(config):
        seen["config"] = config
        return ["docker", "run", config.image, "echo", "hello"]

    monkeypatch.setattr(executor, "_build_docker_command", _capture_command)
    monkeypatch.setattr(
        executor,
        "_execute_capture",
        lambda cmd, timeout: subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="ok\n",
            stderr="",
        ),
    )

    result = executor.execute_container(
        DockerExecutionConfig(
            image="python:3.12-slim",
            command=["echo", "hello"],
            model_artifact_cid="cid-model",
            model_artifact_mount_path="/workspace/model.bin",
        )
    )

    assert result.success is True
    assert storage.retrieve_calls == ["cid-model"]
    used_config = seen["config"]
    assert used_config.volumes
    assert "/workspace/model.bin" in used_config.volumes.values()
    model_meta = (result.metadata or {}).get("model_artifact") or {}
    assert model_meta.get("retrieved") is True
    assert model_meta.get("cid") == "cid-model"


def test_docker_executor_falls_back_to_datasets_provenance_when_logger_missing(monkeypatch):
    monkeypatch.setattr(DockerExecutor, "_verify_docker_available", lambda self: None)

    storage = _FakeStorage()
    datasets = _FakeDatasetsManager()
    executor = DockerExecutor(
        storage=storage,
        datasets_manager=datasets,
        provenance_logger=None,
    )

    monkeypatch.setattr(
        executor,
        "_build_docker_command",
        lambda config: ["docker", "run", config.image, "echo", "hello"],
    )
    monkeypatch.setattr(
        executor,
        "_execute_capture",
        lambda cmd, timeout: subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="hello from docker\n",
            stderr="",
        ),
    )

    result = executor.execute_container(
        DockerExecutionConfig(
            image="python:3.12-slim",
            command=["echo", "hello"],
        )
    )

    assert result.success is True
    assert result.provenance_cid == "cid-prov"
    assert datasets.provenance[0]["operation"] == "docker_execution"


def test_docker_executor_required_model_artifact_policy_fails_fast(monkeypatch):
    monkeypatch.setattr(DockerExecutor, "_verify_docker_available", lambda self: None)

    storage = _FakeStorage()
    datasets = _FakeDatasetsManager()
    executor = DockerExecutor(
        storage=storage,
        datasets_manager=datasets,
        provenance_logger=None,
    )

    called = {"build": False}

    def _build(_config):
        called["build"] = True
        return ["docker", "run", "python:3.12-slim", "echo", "hello"]

    monkeypatch.setattr(executor, "_build_docker_command", _build)

    result = executor.execute_container(
        DockerExecutionConfig(
            image="python:3.12-slim",
            command=["echo", "hello"],
            model_artifact_cid="cid-missing",
            model_artifact_policy="required",
        )
    )

    assert result.success is False
    assert "model_artifact_required" in str(result.error_message or "")
    assert storage.retrieve_calls == ["cid-missing"]
    assert called["build"] is False
    assert any(event["event_type"] == "container_execution_failed" for event in datasets.events)
    assert any(event["event_type"] == "model_artifact_materialization_failed" for event in datasets.events)
    assert any(item["operation"] == "model_artifact_materialization_failed" for item in datasets.provenance)


def test_docker_executor_optional_model_artifact_policy_emits_degraded_event(monkeypatch):
    monkeypatch.setattr(DockerExecutor, "_verify_docker_available", lambda self: None)

    storage = _FakeStorage()
    datasets = _FakeDatasetsManager()
    executor = DockerExecutor(
        storage=storage,
        datasets_manager=datasets,
        provenance_logger=None,
    )

    monkeypatch.setattr(
        executor,
        "_build_docker_command",
        lambda config: ["docker", "run", config.image, "echo", "hello"],
    )
    monkeypatch.setattr(
        executor,
        "_execute_capture",
        lambda cmd, timeout: subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="ok\n",
            stderr="",
        ),
    )

    result = executor.execute_container(
        DockerExecutionConfig(
            image="python:3.12-slim",
            command=["echo", "hello"],
            model_artifact_cid="cid-missing",
            model_artifact_policy="optional",
        )
    )

    assert result.success is True
    degraded_events = [e for e in datasets.events if e["event_type"] == "model_artifact_materialization_degraded"]
    assert degraded_events
    assert degraded_events[0]["level"] == "WARNING"
    assert any(item["operation"] == "model_artifact_materialization_degraded" for item in datasets.provenance)
