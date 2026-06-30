import asyncio
import time

from ipfs_accelerate_py.docker_executor import DockerExecutionConfig, DockerExecutor
from ipfs_accelerate_py.inference_backend_manager import BackendCapabilities, BackendType, InferenceBackendManager
from ipfs_accelerate_py.model_manager import DataType, IOSpec, ModelManager, ModelMetadata, ModelType
from ipfs_accelerate_py.p2p_tasks.task_queue import TaskQueue


class _BenchmarkStorage:
    def __init__(self):
        self.records = []

    def store(self, data, filename=None, pin=False):
        self.records.append((data, filename, pin))
        return f"cid-{len(self.records)}"


class _BenchmarkDatasetsManager:
    def __init__(self):
        self.events = []

    def log_event(self, event_type, data, level="INFO", category="GENERAL"):
        self.events.append((event_type, data, level, category))
        return True

    def track_provenance(self, operation, data, record_type="TRANSFORMATION"):
        return "cid-prov"


class _BenchmarkProvenanceLogger:
    def log_transformation(self, operation, data, input_cid=None, output_cid=None):
        return "cid-prov"


class _ReadyBackend:
    async def run_inference(self, **kwargs):
        return {"text": "ok", "kwargs": kwargs}


def test_benchmark_backend_throughput(tmp_path):
    manager = InferenceBackendManager(
        {
            "registry_state_path": str(tmp_path / "backend_registry.json"),
            "persist_registry": False,
            "result_recorder": lambda **kwargs: kwargs["result"],
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

    started = time.perf_counter()
    completed = 0
    for _ in range(50):
        result = asyncio.run(manager.execute_task(task="text-generation", model="model-a", inputs=["hello"]))
        assert result["backend_id"] == "backend-1"
        completed += 1
    elapsed = time.perf_counter() - started

    assert completed == 50
    assert elapsed < 2.0


def test_benchmark_model_persistence_latency(tmp_path):
    storage = _BenchmarkStorage()
    datasets = _BenchmarkDatasetsManager()
    model_root = tmp_path / "model"
    model_root.mkdir()
    model_path = model_root / "model.bin"
    config_path = model_root / "config.json"
    tokenizer_path = model_root / "tokenizer.json"
    model_path.write_bytes(b"weights")
    config_path.write_text("{}", encoding="utf-8")
    tokenizer_path.write_text("{}", encoding="utf-8")

    manager = ModelManager(storage_path=str(tmp_path / "models.duckdb"), enable_ipfs=True)
    manager._artifact_storage = storage
    manager._datasets_manager = datasets
    manager._provenance_logger = _BenchmarkProvenanceLogger()

    started = time.perf_counter()
    for index in range(3):
        metadata = ModelMetadata(
            model_id=f"model-{index}",
            model_name=f"model-{index}",
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

    elapsed = time.perf_counter() - started

    assert len(storage.records) >= 3
    assert len(datasets.events) >= 3
    assert elapsed < 5.0


def test_benchmark_failover_reselection_latency(tmp_path):
    manager = InferenceBackendManager(
        {
            "registry_state_path": str(tmp_path / "backend_registry.json"),
            "persist_registry": False,
        }
    )
    healthy = _ReadyBackend()
    assert manager.register_backend(
        backend_id="backend-unhealthy",
        backend_type=BackendType.API,
        name="backend-unhealthy",
        instance=healthy,
        capabilities=BackendCapabilities(supported_tasks={"text-generation"}, protocols={"http"}),
        endpoint="http://example.invalid/1",
    )
    assert manager.register_backend(
        backend_id="backend-healthy",
        backend_type=BackendType.API,
        name="backend-healthy",
        instance=healthy,
        capabilities=BackendCapabilities(supported_tasks={"text-generation"}, protocols={"http"}),
        endpoint="http://example.invalid/2",
    )
    manager.backends["backend-unhealthy"].status = manager.backends["backend-unhealthy"].status.UNHEALTHY

    started = time.perf_counter()
    chosen = None
    for _ in range(200):
        chosen = manager.select_backend_for_task(task="text-generation", model="model-a")
        assert chosen is not None
        assert chosen.backend_id == "backend-healthy"
    elapsed = time.perf_counter() - started

    assert chosen is not None
    assert elapsed < 2.0


def test_benchmark_backlog_drain_latency(tmp_path):
    queue = TaskQueue(str(tmp_path / "tasks.duckdb"))
    for index in range(20):
        queue.submit(
            task_type="text-generation",
            model_name=f"model-{index}",
            payload={"prompt": f"prompt-{index}"},
        )

    started = time.perf_counter()
    claimed = queue.claim_next_many(
        worker_id="worker-1",
        supported_task_types=["text-generation"],
        max_tasks=20,
        same_task_type=True,
    )
    assert len(claimed) == 20

    for task in claimed:
        assert queue.complete(task_id=task.task_id, status="completed", result={"text": "ok"})

    elapsed = time.perf_counter() - started

    assert queue.count(status="queued") == 0
    assert queue.count(status="completed") == 20
    assert elapsed < 5.0