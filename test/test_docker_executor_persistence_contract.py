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

    def log_event(self, event_type, data, level="INFO", category="GENERAL"):
        self.events.append({
            "event_type": event_type,
            "data": data,
            "level": level,
            "category": category,
        })
        return True


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
