#!/usr/bin/env python3
"""Contract tests for the minimal Kubernetes backend."""

from pathlib import Path

from ipfs_accelerate_py.container_backends.kubernetes.kubernetes import (
    KubernetesBackend,
    KubernetesExecutionConfig,
    KubernetesJobStatus,
)


class _FakeStorage:
    def __init__(self):
        self.retrieve_calls = []

    def retrieve(self, cid):
        self.retrieve_calls.append(cid)
        if cid == "cid-model":
            return b"model-bytes"
        return None


def test_kubernetes_backend_builds_job_spec_and_collects_result(tmp_path):
    backend = KubernetesBackend(namespace="test-namespace")
    config = KubernetesExecutionConfig(
        image="python:3.12-slim",
        command=["python", "-c"],
        args=["print('hello from kubernetes')"],
        namespace="test-namespace",
        job_name="ipfs-accel-test-job",
        environment={"MODEL_ID": "demo-model"},
        volumes={str(tmp_path): "/workspace"},
        labels={"suite": "contract"},
        annotations={"owner": "tests"},
    )

    job_spec = backend.build_job_spec(config)
    assert job_spec["metadata"]["name"] == "ipfs-accel-test-job"
    assert job_spec["metadata"]["namespace"] == "test-namespace"
    assert job_spec["spec"]["template"]["spec"]["containers"][0]["image"] == "python:3.12-slim"
    assert job_spec["spec"]["template"]["spec"]["containers"][0]["env"][0] == {
        "name": "MODEL_ID",
        "value": "demo-model",
    }
    assert job_spec["spec"]["template"]["spec"]["containers"][0]["volumeMounts"][0]["mountPath"] == "/workspace"

    job_id = backend.submit_job(config)
    assert job_id == "ipfs-accel-test-job"
    assert backend.get_job_status(job_id) in {KubernetesJobStatus.RUNNING, KubernetesJobStatus.FAILED}

    backend.record_job_artifacts(
        job_id,
        stdout="job complete",
        stderr="",
        exit_code=0,
        output_cid="cid-output",
        provenance_cid="cid-prov",
    )
    result = backend.collect_result(job_id)

    assert result.success is True
    assert result.exit_code == 0
    assert result.stdout == "job complete"
    assert result.container_id == "ipfs-accel-test-job-pod"
    assert result.output_cid == "cid-output"
    assert result.provenance_cid == "cid-prov"
    assert result.metadata["job_id"] == "ipfs-accel-test-job"
    assert result.metadata["output_cid"] == "cid-output"
    assert backend.list_jobs()[0]["job_id"] == "ipfs-accel-test-job"
    assert backend.list_jobs()[0]["status"] == KubernetesJobStatus.SUCCEEDED.value
    assert "ipfs-accel-test-job" in backend.to_json()


def test_kubernetes_backend_materializes_model_artifact_by_cid(tmp_path):
    storage = _FakeStorage()
    backend = KubernetesBackend(namespace="test-namespace", storage=storage)
    config = KubernetesExecutionConfig(
        image="python:3.12-slim",
        namespace="test-namespace",
        job_name="ipfs-accel-model-cid-job",
        model_artifact_cid="cid-model",
        model_artifact_mount_path="/workspace/model.bin",
    )

    job_id = backend.submit_job(config)
    assert job_id == "ipfs-accel-model-cid-job"
    assert storage.retrieve_calls == ["cid-model"]

    jobs = backend.list_jobs()
    assert jobs
    model_meta = (jobs[0].get("metadata") or {}).get("model_artifact") or {}
    assert model_meta.get("retrieved") is True
    assert model_meta.get("cid") == "cid-model"

    spec = jobs[0]["job_spec"]
    mounts = spec["spec"]["template"]["spec"]["containers"][0].get("volumeMounts") or []
    assert any(m.get("mountPath") == "/workspace/model.bin" for m in mounts)
