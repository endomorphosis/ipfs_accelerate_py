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
        self.store_calls = []

    def retrieve(self, cid):
        self.retrieve_calls.append(cid)
        if cid == "cid-model":
            return b"model-bytes"
        return None

    def store(self, data, filename=None, pin=False):
        self.store_calls.append({"data": data, "filename": filename, "pin": pin})
        return f"cid-store-{len(self.store_calls)}"


class _FakeDatasetsManager:
    def __init__(self):
        self.events = []
        self.provenance = []

    def log_event(self, event_type, data, level="INFO", category="GENERAL"):
        self.events.append({"event_type": event_type, "data": data, "level": level, "category": category})
        return True

    def track_provenance(self, operation, data, record_type="TRANSFORMATION"):
        self.provenance.append({"operation": operation, "data": data, "record_type": record_type})
        return "cid-prov"


class _FakeProvenanceLogger:
    def __init__(self):
        self.calls = []

    def log_transformation(self, operation, data, input_cid=None, output_cid=None):
        self.calls.append({"operation": operation, "data": data, "input_cid": input_cid, "output_cid": output_cid})
        return "cid-prov"


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
    for key in ["image", "command", "container_id", "execution_time", "exit_code", "success", "stdout", "stderr", "error_message", "output_cid", "provenance_cid"]:
        assert key in result.metadata


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


def test_kubernetes_backend_persists_artifacts_and_provenance_when_missing(tmp_path):
    storage = _FakeStorage()
    datasets = _FakeDatasetsManager()
    provenance = _FakeProvenanceLogger()
    backend = KubernetesBackend(
        namespace="test-namespace",
        storage=storage,
        datasets_manager=datasets,
        provenance_logger=provenance,
    )
    config = KubernetesExecutionConfig(
        image="python:3.12-slim",
        command=["python", "-c"],
        args=["print('ok')"],
        namespace="test-namespace",
        job_name="ipfs-accel-persist-job",
    )

    job_id = backend.submit_job(config)
    backend.record_job_artifacts(job_id, stdout="ok", stderr="", exit_code=0)
    result = backend.collect_result(job_id)

    assert result.success is True
    assert result.output_cid == "cid-store-1"
    assert result.provenance_cid == "cid-prov"
    assert storage.store_calls
    assert datasets.events[0]["event_type"] == "container_execution_completed"
    assert datasets.events[0]["data"]["job_id"] == "ipfs-accel-persist-job"
    assert provenance.calls[0]["operation"] == "kubernetes_execution"


def test_kubernetes_backend_falls_back_to_datasets_provenance_when_logger_missing(tmp_path):
    storage = _FakeStorage()
    datasets = _FakeDatasetsManager()
    backend = KubernetesBackend(
        namespace="test-namespace",
        storage=storage,
        datasets_manager=datasets,
        provenance_logger=None,
    )
    config = KubernetesExecutionConfig(
        image="python:3.12-slim",
        command=["python", "-c"],
        args=["print('ok')"],
        namespace="test-namespace",
        job_name="ipfs-accel-fallback-job",
    )

    job_id = backend.submit_job(config)
    backend.record_job_artifacts(job_id, stdout="ok", stderr="", exit_code=0)
    result = backend.collect_result(job_id)

    assert result.success is True
    assert result.provenance_cid == "cid-prov"
    assert datasets.provenance[0]["operation"] == "kubernetes_execution"


def test_kubernetes_backend_records_failure_diagnostics_for_nonzero_exit(tmp_path):
    backend = KubernetesBackend(namespace="test-namespace")
    config = KubernetesExecutionConfig(
        image="python:3.12-slim",
        namespace="test-namespace",
        job_name="ipfs-accel-fail-job",
    )

    job_id = backend.submit_job(config)
    backend.record_job_artifacts(job_id, stdout="", stderr="process crashed", exit_code=137)
    result = backend.collect_result(job_id)

    assert result.success is False
    assert result.exit_code == 137
    assert result.error_message == "process crashed"
    assert result.metadata["failure_reason"] == "non_zero_exit_code"
    assert result.metadata["failure_message"] == "process crashed"
    assert result.metadata["failure_phase"] == "failed"
    assert result.metadata["failure_retryable"] is False


def test_kubernetes_backend_records_conditions_and_events_for_failed_status(tmp_path):
    class _Obj:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class _BatchApi:
        def create_namespaced_job(self, namespace, body):
            return None

        def read_namespaced_job(self, name, namespace):
            failed_cond = _Obj(
                type="Failed",
                status="True",
                reason="BackoffLimitExceeded",
                message="Job has reached backoff limit",
                last_transition_time="2026-06-30T12:00:00Z",
            )
            return _Obj(status=_Obj(failed=1, succeeded=0, active=0, reason="", message="", conditions=[failed_cond]))

    class _CoreApi:
        def read_namespaced_pod(self, name, namespace):
            pod_cond = _Obj(
                type="Ready",
                status="False",
                reason="ContainersNotReady",
                message="containers with unready status",
                last_transition_time="2026-06-30T12:00:01Z",
            )
            return _Obj(status=_Obj(phase="Failed", reason="Error", message="Pod failed", conditions=[pod_cond]))

        def list_namespaced_event(self, namespace, field_selector):
            ev = _Obj(
                reason="Failed",
                message="Back-off restarting failed container",
                type="Warning",
                count=3,
                last_timestamp="2026-06-30T12:00:02Z",
            )
            return _Obj(items=[ev])

    backend = KubernetesBackend(namespace="test-namespace")
    backend._client_available = True
    backend._batch_v1 = _BatchApi()
    backend._core_v1 = _CoreApi()

    config = KubernetesExecutionConfig(
        image="python:3.12-slim",
        namespace="test-namespace",
        job_name="ipfs-accel-cond-job",
    )
    job_id = backend.submit_job(config)
    status = backend.get_job_status(job_id)
    result = backend.collect_result(job_id)

    assert status == KubernetesJobStatus.FAILED
    assert result.success is False
    assert result.metadata["failure_reason"] == "BackoffLimitExceeded"
    assert "backoff limit" in result.metadata["failure_message"].lower()
    assert result.metadata["failure_phase"] == "Failed"
    assert isinstance(result.metadata.get("failure_conditions"), list)
    assert isinstance(result.metadata.get("failure_events"), list)
    assert result.metadata["failure_events"][0]["reason"] == "Failed"


def test_kubernetes_backend_required_model_artifact_policy_fails_fast(tmp_path):
    storage = _FakeStorage()
    datasets = _FakeDatasetsManager()
    backend = KubernetesBackend(
        namespace="test-namespace",
        storage=storage,
        datasets_manager=datasets,
        provenance_logger=None,
    )

    config = KubernetesExecutionConfig(
        image="python:3.12-slim",
        namespace="test-namespace",
        job_name="ipfs-accel-k8s-required-artifact-job",
        model_artifact_cid="cid-missing",
        model_artifact_policy="required",
    )

    job_id = backend.submit_job(config)
    result = backend.collect_result(job_id)

    assert job_id == "ipfs-accel-k8s-required-artifact-job"
    assert result.success is False
    assert result.metadata["failure_reason"] == "model_artifact_materialization_required"
    assert storage.retrieve_calls == ["cid-missing"]
    assert any(event["event_type"] == "container_execution_failed" for event in datasets.events)
    assert any(event["event_type"] == "model_artifact_materialization_failed" for event in datasets.events)
    assert any(item["operation"] == "model_artifact_materialization_failed" for item in datasets.provenance)


def test_kubernetes_backend_optional_model_artifact_policy_emits_degraded_event(tmp_path):
    storage = _FakeStorage()
    datasets = _FakeDatasetsManager()
    backend = KubernetesBackend(
        namespace="test-namespace",
        storage=storage,
        datasets_manager=datasets,
        provenance_logger=None,
    )

    config = KubernetesExecutionConfig(
        image="python:3.12-slim",
        namespace="test-namespace",
        job_name="ipfs-accel-k8s-optional-artifact-job",
        model_artifact_cid="cid-missing",
        model_artifact_policy="optional",
    )

    job_id = backend.submit_job(config)
    result = backend.collect_result(job_id)

    assert job_id == "ipfs-accel-k8s-optional-artifact-job"
    assert result.success is True
    degraded_events = [e for e in datasets.events if e["event_type"] == "model_artifact_materialization_degraded"]
    assert degraded_events
    assert degraded_events[0]["level"] == "WARNING"
    assert any(item["operation"] == "model_artifact_materialization_degraded" for item in datasets.provenance)
