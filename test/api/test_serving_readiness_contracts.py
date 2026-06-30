from ipfs_accelerate_py.docker_executor import DockerExecutionConfig, DockerExecutor
from ipfs_accelerate_py.inference_backend_manager import BackendCapabilities, BackendStatus, BackendType, InferenceBackendManager
from ipfs_accelerate_py.model_manager import DataType, IOSpec, ModelManager, ModelMetadata, ModelType
from ipfs_accelerate_py.p2p_tasks.capability_registry import PeerCapabilityRegistry
from ipfs_accelerate_py.p2p_tasks.orchestrator import OrchestratorConfig, TaskOrchestrator
from ipfs_accelerate_py.p2p_tasks.task_queue import TaskQueue
from ipfs_accelerate_py.p2p_tasks.worker import run_worker
from ipfs_accelerate_py.container_backends.kubernetes.kubernetes import KubernetesBackend, KubernetesExecutionConfig, KubernetesJobStatus


class _FakeStorage:
    def __init__(self):
        self.records = []

    def store(self, data, filename=None, pin=False):
        self.records.append({"data": data, "filename": filename, "pin": pin})
        return f"cid-{len(self.records)}"

    def retrieve(self, cid):
        if cid == "cid-1":
            return b"model-bytes"
        return None


class _FakeDatasetsManager:
    def __init__(self):
        self.events = []
        self.provenance = []

    def log_event(self, event_type, data, level="INFO", category="GENERAL"):
        self.events.append((event_type, data, level, category))
        return True

    def track_provenance(self, operation, data, record_type="TRANSFORMATION"):
        self.provenance.append((operation, data, record_type))
        return "cid-prov"


class _FakeProvenanceLogger:
    def log_transformation(self, operation, data, input_cid=None, output_cid=None):
        return "cid-prov"


class _ReadyBackend:
    async def run_inference(self, **kwargs):
        return {"text": "ok", "kwargs": kwargs}


class _Remote:
    def __init__(self, peer_id: str, multiaddr: str):
        self.peer_id = peer_id
        self.multiaddr = multiaddr


def test_readiness_backend_manager_result_recording(tmp_path):
    recorder_calls = []
    manager = InferenceBackendManager(
        {
            "registry_state_path": str(tmp_path / "backend_registry.json"),
            "persist_registry": True,
            "result_recorder": lambda **kwargs: recorder_calls.append(kwargs) or kwargs["result"],
        }
    )
    backend = _ReadyBackend()
    assert manager.register_backend(
        backend_id="backend-1",
        backend_type=BackendType.API,
        name="backend-1",
        instance=backend,
        capabilities=BackendCapabilities(supported_tasks={"text-generation"}, protocols={"http"}),
        endpoint="http://example.invalid",
    )

    import asyncio

    result = asyncio.run(manager.execute_task(task="text-generation", model="model-a", inputs=["hello"]))
    assert result["backend_id"] == "backend-1"
    assert result["text"] == "ok"
    assert recorder_calls and recorder_calls[0]["backend_id"] == "backend-1"


def test_readiness_model_registry_roundtrip(tmp_path):
    storage = _FakeStorage()
    datasets = _FakeDatasetsManager()
    model_path = tmp_path / "model.bin"
    config_path = tmp_path / "config.json"
    tokenizer_path = tmp_path / "tokenizer.json"
    model_path.write_bytes(b"weights")
    config_path.write_text("{}", encoding="utf-8")
    tokenizer_path.write_text("{}", encoding="utf-8")

    manager = ModelManager(storage_path=str(tmp_path / "models.duckdb"), enable_ipfs=True)
    manager._artifact_storage = storage
    manager._datasets_manager = datasets
    manager._provenance_logger = _FakeProvenanceLogger()
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

    stored = manager.models.get("model-a")
    assert stored is not None
    assert stored.model_cid is not None
    assert stored.config_cid is not None
    assert stored.tokenizer_cid is not None


def test_readiness_peer_capability_registry_roundtrip(tmp_path):
    registry = PeerCapabilityRegistry(path=str(tmp_path / "peer_capability_registry.json"))
    record = registry.upsert_from_status(
        peer_id="peer-1",
        multiaddr="/ip4/127.0.0.1/tcp/4001/p2p/peer-1",
        status={
            "ok": True,
            "queued": 0,
            "running": 0,
            "queued_by_type": {"text-generation": 1},
            "capabilities": {"supported_task_types": ["text-generation"], "loaded_models": ["model-a"]},
            "detail": {"runtime": {"cuda_available": True}},
        },
    )
    assert record is not None
    assert registry.score_peer_for_task(peer_id="peer-1", task_type="text-generation") > 0
    assert registry.get_record("peer-1") is not None


def test_readiness_container_envelopes(tmp_path, monkeypatch):
    monkeypatch.setattr(DockerExecutor, "_verify_docker_available", lambda self: None)
    docker = DockerExecutor(
        storage=_FakeStorage(),
        datasets_manager=_FakeDatasetsManager(),
        provenance_logger=_FakeProvenanceLogger(),
    )
    monkeypatch.setattr(
        docker,
        "_build_docker_command",
        lambda config: ["docker", "run", config.image, "echo", "ok"],
    )
    monkeypatch.setattr(
        docker,
        "_execute_capture",
        lambda cmd, timeout: __import__("subprocess").CompletedProcess(args=cmd, returncode=0, stdout="ok\n", stderr=""),
    )
    docker_result = docker.execute_container(DockerExecutionConfig(image="python:3.12-slim", command=["echo", "ok"]))
    assert docker_result.output_cid is not None
    assert docker_result.provenance_cid is not None

    kubernetes = KubernetesBackend(namespace="default")
    job_id = kubernetes.submit_job(KubernetesExecutionConfig(image="python:3.12-slim", job_name="readiness-job"))
    kubernetes.record_job_artifacts(job_id, stdout="ok", stderr="", output_cid="cid-output", provenance_cid="cid-prov", exit_code=0)
    kube_result = kubernetes.collect_result(job_id)
    assert kube_result.output_cid == "cid-output"
    assert kube_result.provenance_cid == "cid-prov"
    assert kube_result.success is True
    assert kubernetes.get_job_status(job_id) == KubernetesJobStatus.SUCCEEDED


def test_readiness_worker_orchestrator_lineage_and_routing(monkeypatch, tmp_path):
    queue_path = str(tmp_path / "tasks.duckdb")
    orchestrator = TaskOrchestrator(
        config=OrchestratorConfig(queue_path=queue_path, orchestrator_id="orch-test", base_worker_id="worker-test", min_workers=0, max_workers=0),
        supported_task_types=["text-generation"],
    )
    remote = _Remote("peer-a", "/ip4/127.0.0.1/tcp/4001/p2p/peer-a")
    registry = PeerCapabilityRegistry(path=str(tmp_path / "peer_capability_registry.json"))
    registry.upsert_from_status(
        peer_id="peer-a",
        multiaddr=remote.multiaddr,
        status={"ok": True, "queued": 0, "running": 0, "capabilities": {"supported_task_types": ["text-generation"]}},
    )
    monkeypatch.setattr(orchestrator, "_get_capability_registry", lambda: registry)

    import ipfs_accelerate_py.p2p_tasks.client as client

    monkeypatch.setattr(client, "claim_many_sync", lambda **kwargs: [{"task_id": "remote-task-1", "task_type": "text-generation", "model_name": "model-a", "payload": {"prompt": "hello"}}] if getattr(kwargs.get("remote"), "peer_id", "") == "peer-a" else [])
    claimed = orchestrator._claim_from_peers(peers=[remote], max_tasks=1)
    assert claimed and claimed[0][0].peer_id == "peer-a"

    queue = TaskQueue(queue_path)
    task_id = queue.submit(task_type="text-generation", model_name="model-x", payload={"prompt": "hello", "workflow_id": "wf-1", "persistence_policy": "required", "provenance_policy": "strict"})

    class _BackendManager:
        async def execute_task(self, **kwargs):
            return {"backend_id": "backend-1", "output_cid": "cid-output", "provenance_cid": "cid-prov", "result": {"text": "ok"}}

    import ipfs_accelerate_py.inference_backend_manager as ibm
    monkeypatch.setattr(ibm, "get_backend_manager", lambda config=None: _BackendManager())

    rc = run_worker(queue_path=queue_path, worker_id="worker-1", poll_interval_s=0.05, once=True, p2p_service=False, supported_task_types=["text-generation"])
    assert rc == 0
    out = queue.get(task_id)
    assert out is not None
    assert out.get("status") == "completed"
    assert (out.get("result") or {}).get("lineage", {}).get("workflow_id") == "wf-1"
