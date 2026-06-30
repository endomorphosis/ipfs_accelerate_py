from fastapi.testclient import TestClient


class _FakeBackendManager:
    def __init__(self):
        self.calls = []
        self._result_recorder = None

    async def execute_task(self, *, task, model, inputs, preferred_types=None, required_protocols=None, parameters=None):
        self.calls.append(
            {
                "task": task,
                "model": model,
                "inputs": list(inputs),
                "parameters": dict(parameters or {}),
            }
        )
        if task == "text-embedding":
            return {
                "embeddings": [[0.1, 0.2], [0.3, 0.4]],
                "processing_time": 0.01,
                "device": "cpu",
            }
        return {
            "outputs": ["generated response"],
            "processing_time": 0.01,
            "device": "cpu",
        }


class _FakeModelManager:
    def __init__(self):
        self.add_calls = []
        self.remove_calls = []
        self.known_models = set()

    def add_model_with_ipfs_storage(self, metadata, model_path=None, config_path=None, tokenizer_path=None, store_to_ipfs=True):
        self.add_calls.append(
            {
                "metadata": metadata,
                "model_path": model_path,
                "config_path": config_path,
                "tokenizer_path": tokenizer_path,
                "store_to_ipfs": store_to_ipfs,
            }
        )
        self.known_models.add(metadata.model_id)
        return True, "cid-artifact-1"

    def remove_model(self, model_id):
        self.remove_calls.append(model_id)
        if model_id in self.known_models:
            self.known_models.remove(model_id)
            return True
        return False


class _FakeDatasetsManager:
    def __init__(self):
        self.events = []
        self.provenance = []

    def log_event(self, event_type, data, level="INFO", category="GENERAL"):
        self.events.append(
            {
                "event_type": event_type,
                "data": dict(data or {}),
                "level": level,
                "category": category,
            }
        )
        return True

    def track_provenance(self, operation, data, record_type="TRANSFORMATION"):
        self.provenance.append(
            {
                "operation": operation,
                "data": dict(data or {}),
                "record_type": record_type,
            }
        )
        return "cid-prov"


class _FailingBackendManager:
    async def execute_task(self, **kwargs):
        raise RuntimeError("backend failure")


class _FailingModelManager(_FakeModelManager):
    def add_model_with_ipfs_storage(self, metadata, model_path=None, config_path=None, tokenizer_path=None, store_to_ipfs=True):
        self.add_calls.append(
            {
                "metadata": metadata,
                "model_path": model_path,
                "config_path": config_path,
                "tokenizer_path": tokenizer_path,
                "store_to_ipfs": store_to_ipfs,
            }
        )
        return False, None


def _build_test_server(monkeypatch, *, backend_manager=None):
    from ipfs_accelerate_py.hf_model_server.config import ServerConfig
    from ipfs_accelerate_py.hf_model_server import server as server_module

    fake_manager = backend_manager or _FakeBackendManager()
    fake_model_manager = _FakeModelManager()
    fake_datasets = _FakeDatasetsManager()
    monkeypatch.setattr(server_module, "get_backend_manager", lambda config=None: fake_manager)
    monkeypatch.setattr(server_module, "HAVE_BACKEND_MANAGER", True)
    monkeypatch.setattr(server_module, "ModelManager", lambda *args, **kwargs: fake_model_manager)
    monkeypatch.setattr(server_module, "HAVE_MODEL_MANAGER", True)
    monkeypatch.setattr(server_module, "DatasetsManager", lambda *args, **kwargs: fake_datasets)
    monkeypatch.setattr(server_module, "HAVE_DATASETS_MANAGER", True)

    config = ServerConfig(
        enable_auth=False,
        enable_metrics=False,
        auto_discover=False,
        enable_hardware_detection=False,
    )
    return server_module.HFModelServer(config), fake_manager, fake_model_manager, fake_datasets


def test_completions_endpoint_routes_via_backend_manager(monkeypatch):
    server, fake_manager, _, _ = _build_test_server(monkeypatch)

    with TestClient(server.app) as client:
        response = client.post(
            "/v1/completions",
            json={
                "model": "demo-model",
                "prompt": "hello world",
                "max_tokens": 16,
                "temperature": 0.2,
                "top_p": 0.9,
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["text"] == "generated response"

    assert len(fake_manager.calls) == 1
    call = fake_manager.calls[0]
    assert call["task"] == "text-generation"
    assert call["model"] == "demo-model"
    assert call["inputs"] == ["hello world"]


def test_embeddings_endpoint_routes_via_backend_manager(monkeypatch):
    server, fake_manager, _, _ = _build_test_server(monkeypatch)

    with TestClient(server.app) as client:
        response = client.post(
            "/v1/embeddings",
            json={
                "model": "embed-model",
                "input": ["hello", "world"],
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert len(body["data"]) == 2
    assert body["data"][0]["embedding"] == [0.1, 0.2]

    assert len(fake_manager.calls) == 1
    call = fake_manager.calls[0]
    assert call["task"] == "text-embedding"
    assert call["inputs"] == ["hello", "world"]


def test_model_load_endpoint_registers_with_model_manager(monkeypatch):
    server, _, fake_model_manager, _ = _build_test_server(monkeypatch)

    with TestClient(server.app) as client:
        response = client.post(
            "/models/load",
            json={
                "model_id": "demo-model",
                "hardware": "cpu",
                "options": {
                    "model_path": "/tmp/demo-model.bin",
                    "config_path": "/tmp/demo-model.json",
                    "tokenizer_path": "/tmp/demo-tokenizer.json",
                    "store_to_ipfs": True,
                    "model_type": "language_model",
                },
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "loaded"
    assert "artifact_cid=cid-artifact-1" in body["message"]

    assert len(fake_model_manager.add_calls) == 1
    call = fake_model_manager.add_calls[0]
    assert call["metadata"].model_id == "demo-model"
    assert call["model_path"] == "/tmp/demo-model.bin"
    assert call["config_path"] == "/tmp/demo-model.json"
    assert call["tokenizer_path"] == "/tmp/demo-tokenizer.json"


def test_model_unload_endpoint_removes_from_model_manager(monkeypatch):
    server, _, fake_model_manager, _ = _build_test_server(monkeypatch)
    fake_model_manager.known_models.add("demo-model")

    with TestClient(server.app) as client:
        response = client.post(
            "/models/unload",
            json={
                "model_id": "demo-model",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "unloaded"
    assert fake_model_manager.remove_calls == ["demo-model"]


def test_inference_failure_emits_datasets_failure_event(monkeypatch):
    server, _, _, fake_datasets = _build_test_server(monkeypatch, backend_manager=_FailingBackendManager())

    with TestClient(server.app) as client:
        response = client.post(
            "/v1/completions",
            json={
                "model": "demo-model",
                "prompt": "hello world",
            },
        )

    assert response.status_code == 503
    assert fake_datasets.events
    assert fake_datasets.events[0]["event_type"] == "inference_failed"
    assert fake_datasets.provenance
    assert fake_datasets.provenance[0]["operation"] == "inference_failed"


def test_model_unload_missing_emits_failure_event(monkeypatch):
    server, _, fake_model_manager, fake_datasets = _build_test_server(monkeypatch)
    fake_model_manager.known_models.discard("missing-model")

    with TestClient(server.app) as client:
        response = client.post(
            "/models/unload",
            json={
                "model_id": "missing-model",
            },
        )

    assert response.status_code == 404
    assert fake_datasets.events
    assert fake_datasets.events[0]["event_type"] == "model_unload_failed"
    assert fake_datasets.provenance
    assert fake_datasets.provenance[0]["operation"] == "model_unload_failed"


def test_model_load_failure_emits_failure_event(monkeypatch):
    server, _, _, fake_datasets = _build_test_server(monkeypatch)
    server._model_manager = _FailingModelManager()

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
    assert fake_datasets.events
    assert fake_datasets.events[0]["event_type"] == "model_load_failed"
    assert fake_datasets.provenance
    assert fake_datasets.provenance[0]["operation"] == "model_load_failed"
