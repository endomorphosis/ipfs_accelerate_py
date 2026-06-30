import asyncio
import subprocess
from fastapi.testclient import TestClient

from ipfs_accelerate_py.docker_executor import DockerExecutionConfig, DockerExecutor
from ipfs_accelerate_py.inference_backend_manager import BackendCapabilities, BackendType, InferenceBackendManager
from ipfs_accelerate_py.model_manager import DataType, IOSpec, ModelManager, ModelMetadata, ModelType
from ipfs_accelerate_py.container_backends.kubernetes.kubernetes import KubernetesBackend, KubernetesExecutionConfig
from ipfs_accelerate_py.p2p_tasks.task_queue import TaskQueue
from ipfs_accelerate_py.p2p_tasks.worker import run_worker
from ipfs_accelerate_py.unified_inference_service import UnifiedInferenceService


class _CallTrackingStorage:
    def __init__(self):
        self.store_calls = []
        self.retrieve_calls = []

    def store(self, data, filename=None, pin=False):
        self.store_calls.append({"data": data, "filename": filename, "pin": pin})
        return f"cid-{len(self.store_calls)}"

    def retrieve(self, cid):
        self.retrieve_calls.append(cid)
        if cid == "cid-model":
            return b"model-bytes"
        return None


class _CallTrackingDatasets:
    def __init__(self):
        self.event_calls = []
        self.provenance_calls = []

    def log_event(self, event_type, data, level="INFO", category="GENERAL"):
        self.event_calls.append(
            {
                "event_type": event_type,
                "data": dict(data or {}),
                "level": level,
                "category": category,
            }
        )
        return True

    def track_provenance(self, operation, data, record_type="TRANSFORMATION"):
        self.provenance_calls.append(
            {
                "operation": operation,
                "data": dict(data or {}),
                "record_type": record_type,
            }
        )
        return "cid-prov"


class _CallTrackingProvenance:
    def __init__(self):
        self.calls = []

    def log_transformation(self, operation, data, input_cid=None, output_cid=None):
        self.calls.append(
            {
                "operation": operation,
                "data": dict(data or {}),
                "input_cid": input_cid,
                "output_cid": output_cid,
            }
        )
        return "cid-prov"


class _ReadyBackend:
    async def run_inference(self, **kwargs):
        return {"text": "ok", "kwargs": kwargs}


class _FakeBackendManager:
    def __init__(self):
        self.calls = []

    async def execute_task(self, *, task, model, inputs, preferred_types=None, required_protocols=None, parameters=None):
        self.calls.append(
            {
                "task": task,
                "model": model,
                "inputs": list(inputs),
                "parameters": dict(parameters or {}),
            }
        )
        return {
            "backend_id": "backend-1",
            "backend_type": "api",
            "output_cid": "cid-output",
            "provenance_cid": "cid-provenance",
            "result": {"text": "ok"},
        }


def test_call_matrix_unified_service_persists_and_indexes(monkeypatch):
    storage = _CallTrackingStorage()
    datasets = _CallTrackingDatasets()

    service = UnifiedInferenceService()
    monkeypatch.setattr(service, "_get_storage_client", lambda: storage)
    monkeypatch.setattr(service, "_get_datasets_manager", lambda: datasets)

    result = service.record_inference_result(
        model="model-a",
        inputs=["hello"],
        result={"text": "ok", "processing_time": 0.01},
        backend_id="backend-1",
        backend_type="api",
        endpoint="http://example.invalid",
        device="cpu",
    )

    assert len(storage.store_calls) == 2
    assert result.get("input_cid") is not None
    assert result.get("output_cid") is not None
    assert len(datasets.event_calls) == 1
    assert datasets.event_calls[0]["event_type"] == "inference_completed"
    assert len(datasets.provenance_calls) == 1
    assert datasets.provenance_calls[0]["operation"] == "inference"


def test_call_matrix_backend_manager_execute_task_finalizes_result(tmp_path):
    recorder_calls = []
    manager = InferenceBackendManager(
        {
            "registry_state_path": str(tmp_path / "backend_registry.json"),
            "persist_registry": False,
            "result_recorder": lambda **kwargs: recorder_calls.append(kwargs) or kwargs["result"],
        }
    )
    assert manager.register_backend(
        backend_id="backend-1",
        backend_type=BackendType.API,
        name="backend-1",
        instance=_ReadyBackend(),
        capabilities=BackendCapabilities(supported_tasks={"text-generation"}, protocols={"http"}),
        endpoint="http://example.invalid",
    )

    result = asyncio.run(manager.execute_task(task="text-generation", model="model-a", inputs=["hello"]))
    assert result["backend_id"] == "backend-1"
    assert result["task"] == "text-generation"
    assert recorder_calls and recorder_calls[0]["backend_id"] == "backend-1"


def test_call_matrix_worker_routes_inference_via_backend_manager(monkeypatch, tmp_path):
    fake_manager = _FakeBackendManager()
    import ipfs_accelerate_py.inference_backend_manager as ibm

    monkeypatch.setattr(ibm, "get_backend_manager", lambda config=None: fake_manager)

    queue_path = str(tmp_path / "task_queue.duckdb")
    queue = TaskQueue(queue_path)
    task_id = queue.submit(
        task_type="text-generation",
        model_name="model-x",
        payload={"prompt": "hello world", "workflow_id": "wf-99", "persistence_policy": "required", "provenance_policy": "strict"},
    )

    rc = run_worker(
        queue_path=queue_path,
        worker_id="w1",
        poll_interval_s=0.05,
        once=True,
        p2p_service=False,
        supported_task_types=["text-generation"],
    )
    assert rc == 0
    assert len(fake_manager.calls) == 1

    out = queue.get(task_id)
    assert out is not None
    result = out.get("result") or {}
    assert result.get("backend_id") == "backend-1"
    assert result.get("output_cid") == "cid-output"


def test_call_matrix_docker_persists_outputs_and_provenance(monkeypatch):
    monkeypatch.setattr(DockerExecutor, "_verify_docker_available", lambda self: None)

    storage = _CallTrackingStorage()
    datasets = _CallTrackingDatasets()
    provenance = _CallTrackingProvenance()

    executor = DockerExecutor(
        storage=storage,
        datasets_manager=datasets,
        provenance_logger=provenance,
    )

    monkeypatch.setattr(executor, "_build_docker_command", lambda config: ["docker", "run", config.image, "echo", "ok"])
    monkeypatch.setattr(
        executor,
        "_execute_capture",
        lambda cmd, timeout: subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok\n", stderr=""),
    )

    result = executor.execute_container(DockerExecutionConfig(image="python:3.12-slim", command=["echo", "ok"]))

    assert result.success is True
    assert result.output_cid is not None
    assert result.provenance_cid == "cid-prov"
    assert len(storage.store_calls) == 1
    assert datasets.event_calls and datasets.event_calls[0]["event_type"] == "container_execution_completed"
    assert provenance.calls and provenance.calls[0]["operation"] == "docker_execution"


def test_call_matrix_model_manager_stores_artifacts_and_updates_datasets(tmp_path):
    storage = _CallTrackingStorage()
    datasets = _CallTrackingDatasets()

    model_path = tmp_path / "model.bin"
    config_path = tmp_path / "config.json"
    tokenizer_path = tmp_path / "tokenizer.json"
    model_path.write_bytes(b"weights")
    config_path.write_text("{}", encoding="utf-8")
    tokenizer_path.write_text("{}", encoding="utf-8")

    manager = ModelManager(storage_path=str(tmp_path / "models.duckdb"), enable_ipfs=True)
    manager._artifact_storage = storage
    manager._datasets_manager = datasets

    metadata = ModelMetadata(
        model_id="model-a",
        model_name="model-a",
        model_type=ModelType.LANGUAGE_MODEL,
        architecture="decoder-only",
        inputs=[IOSpec(name="prompt", data_type=DataType.TEXT, description="Prompt text")],
        outputs=[IOSpec(name="text", data_type=DataType.TEXT, description="Generated text")],
    )

    registered = manager.add_model_with_ipfs_storage(
        metadata=metadata,
        model_path=str(model_path),
        config_path=str(config_path),
        tokenizer_path=str(tokenizer_path),
    )

    assert registered is not None
    success, artifact_cid = registered
    assert success is True
    assert artifact_cid is not None
    assert len(storage.store_calls) >= 3
    assert datasets.event_calls
    assert datasets.event_calls[0]["event_type"] == "model_registered"
    assert datasets.provenance_calls
    assert datasets.provenance_calls[0]["operation"] == "model_registration"


def test_call_matrix_kubernetes_materializes_model_artifact_by_cid():
    storage = _CallTrackingStorage()
    backend = KubernetesBackend(namespace="default", storage=storage)

    job_id = backend.submit_job(
        KubernetesExecutionConfig(
            image="python:3.12-slim",
            job_name="readiness-k8s-model-cid",
            model_artifact_cid="cid-model",
            model_artifact_mount_path="/workspace/model.bin",
        )
    )

    assert job_id == "readiness-k8s-model-cid"
    assert storage.retrieve_calls == ["cid-model"]
    jobs = backend.list_jobs()
    assert jobs
    model_meta = (jobs[0].get("metadata") or {}).get("model_artifact") or {}
    assert model_meta.get("retrieved") is True


def test_call_matrix_hf_model_server_failure_emits_datasets_events(monkeypatch):
    from ipfs_accelerate_py.hf_model_server.config import ServerConfig
    from ipfs_accelerate_py.hf_model_server import server as server_module

    class _FailingBackendManager:
        async def execute_task(self, **kwargs):
            raise RuntimeError("backend unavailable")

    datasets = _CallTrackingDatasets()

    monkeypatch.setattr(server_module, "get_backend_manager", lambda config=None: _FailingBackendManager())
    monkeypatch.setattr(server_module, "HAVE_BACKEND_MANAGER", True)
    monkeypatch.setattr(server_module, "DatasetsManager", lambda *args, **kwargs: datasets)
    monkeypatch.setattr(server_module, "HAVE_DATASETS_MANAGER", True)

    server = server_module.HFModelServer(
        ServerConfig(
            enable_auth=False,
            enable_metrics=False,
            auto_discover=False,
            enable_hardware_detection=False,
        )
    )

    with TestClient(server.app) as client:
        response = client.post(
            "/v1/completions",
            json={
                "model": "demo-model",
                "prompt": "hello world",
            },
        )

    assert response.status_code == 503
    assert datasets.event_calls
    assert datasets.event_calls[0]["event_type"] == "inference_failed"
    assert datasets.provenance_calls
    assert datasets.provenance_calls[0]["operation"] == "inference_failed"


def test_call_matrix_hf_model_server_model_load_failure_emits_datasets_events(monkeypatch):
    from ipfs_accelerate_py.hf_model_server.config import ServerConfig
    from ipfs_accelerate_py.hf_model_server import server as server_module

    class _FailingModelManager:
        def add_model_with_ipfs_storage(self, metadata, model_path=None, config_path=None, tokenizer_path=None, store_to_ipfs=True):
            return False, None

    datasets = _CallTrackingDatasets()

    monkeypatch.setattr(server_module, "ModelManager", lambda *args, **kwargs: _FailingModelManager())
    monkeypatch.setattr(server_module, "HAVE_MODEL_MANAGER", True)
    monkeypatch.setattr(server_module, "DatasetsManager", lambda *args, **kwargs: datasets)
    monkeypatch.setattr(server_module, "HAVE_DATASETS_MANAGER", True)

    server = server_module.HFModelServer(
        ServerConfig(
            enable_auth=False,
            enable_metrics=False,
            auto_discover=False,
            enable_hardware_detection=False,
        )
    )

    with TestClient(server.app) as client:
        response = client.post(
            "/models/load",
            json={
                "model_id": "broken-model",
                "hardware": "cpu",
                "options": {
                    "model_path": "/tmp/broken-model.bin",
                    "store_to_ipfs": True,
                },
            },
        )

    assert response.status_code == 500
    assert datasets.event_calls
    assert datasets.event_calls[0]["event_type"] == "model_load_failed"
    assert datasets.provenance_calls
    assert datasets.provenance_calls[0]["operation"] == "model_load_failed"
