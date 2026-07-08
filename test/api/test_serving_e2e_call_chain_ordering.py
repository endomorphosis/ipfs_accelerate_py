"""
Phase D — End-to-end call-chain ordering tests.

These tests assert the *order* in which the serving stack calls its collaborators
so that CI catches ordering regressions before merge.  All dependencies are
deterministic fakes / stubs; no network or real service is required.

Canonical success order:
    backend.run_inference
    → storage.store (input)
    → storage.store (output)
    → datasets.log_event ("inference_completed")
    → datasets.track_provenance ("inference")
    → model_manager.mark_model_used

Canonical inference-failure order:
    backend.run_inference  (raises)
    → storage.store (input)
    → storage.store (failure)
    → datasets.log_event ("inference_failed")
    → datasets.track_provenance ("inference_failed")
    [mark_model_used is NOT called]

Canonical model-load-failure order:
    model_manager.add_model_with_ipfs_storage  (returns False)
    → datasets.log_event ("model_load_failed")
    → datasets.track_provenance ("model_load_failed")

Canonical model-load-success order:
    model_manager.add_model_with_ipfs_storage  (returns True)
    → datasets.log_event ("model_loaded")
    → datasets.track_provenance ("model_load")
"""

import pytest
from fastapi.testclient import TestClient

from ipfs_accelerate_py.inference_backend_manager import (
    BackendCapabilities,
    BackendType,
    InferenceBackendManager,
)
from ipfs_accelerate_py.hf_model_server.config import ServerConfig
from ipfs_accelerate_py.hf_model_server import server as server_module

# ---------------------------------------------------------------------------
# Shared call-log helpers
# ---------------------------------------------------------------------------


class _CallLog:
    """Append-only log of (operation, detail) tuples produced by all fakes."""

    def __init__(self):
        self.ops: list[dict] = []

    def record(self, op: str, **extra):
        self.ops.append({"op": op, **extra})

    def op_names(self) -> list[str]:
        return [item["op"] for item in self.ops]

    def index(self, op: str) -> int:
        return self.op_names().index(op)

    def contains(self, op: str) -> bool:
        return op in self.op_names()


# ---------------------------------------------------------------------------
# Fake collaborators
# ---------------------------------------------------------------------------


class _RecordingStorage:
    """Fake storage that records store/retrieve calls in the shared log."""

    def __init__(self, log: _CallLog):
        self._log = log
        self._counter = 0

    def store(self, data, filename=None, pin=False):
        self._counter += 1
        cid = f"cid-store-{self._counter}"
        # Distinguish input vs output / failure stores by filename suffix
        kind = "input" if str(filename or "").endswith("_input.json") else "output"
        self._log.record(f"storage.store.{kind}", filename=filename, cid=cid)
        return cid

    def retrieve(self, cid):
        return None


class _RecordingDatasets:
    """Fake datasets manager that records log_event / track_provenance calls."""

    def __init__(self, log: _CallLog):
        self._log = log

    def log_event(self, event_type, data, level="INFO", category="GENERAL"):
        self._log.record(f"datasets.log_event.{event_type}", event_type=event_type)
        return True

    def track_provenance(self, operation, data, record_type="TRANSFORMATION"):
        self._log.record(f"datasets.track_provenance.{operation}", operation=operation)
        return "cid-prov"


class _RecordingModelManager:
    """Fake model manager that records mark_model_used calls."""

    def __init__(self, log: _CallLog, load_success: bool = True):
        self._log = log
        self._load_success = load_success

    def add_model_with_ipfs_storage(
        self, metadata, model_path=None, config_path=None, tokenizer_path=None, store_to_ipfs=True
    ):
        self._log.record("model_manager.add_model_with_ipfs_storage", success=self._load_success)
        return (self._load_success, "cid-artifact" if self._load_success else None)

    def remove_model(self, model_id):
        self._log.record("model_manager.remove_model", model_id=model_id)
        return True

    def get_model(self, model_id):
        return {"model_id": model_id}

    def mark_model_used(self, model_id, inference_cid=None, run_id=None):
        self._log.record("model_manager.mark_model_used", model_id=model_id, inference_cid=inference_cid)
        return True


class _SuccessBackend:
    """A recording backend that always succeeds."""

    def __init__(self, log: _CallLog):
        self._log = log

    async def run_inference(self, **kwargs):
        self._log.record("backend.run_inference", model=kwargs.get("model"))
        return {"text": "hello from backend", "processing_time": 0.001}


class _FailingBackend:
    """A recording backend that always raises."""

    def __init__(self, log: _CallLog):
        self._log = log

    async def run_inference(self, **kwargs):
        self._log.record("backend.run_inference.raised")
        raise RuntimeError("backend unavailable")


# ---------------------------------------------------------------------------
# Server factory helpers
# ---------------------------------------------------------------------------


def _make_server(monkeypatch, backend_instance, log: _CallLog, *, tmp_path):
    """
    Build a minimal HFModelServer with a pre-registered recording backend and
    fake storage / datasets / model_manager injected directly on the instance.
    """
    registry_path = str(tmp_path / "backend_registry.json")
    manager = InferenceBackendManager(
        {"registry_state_path": registry_path, "persist_registry": False}
    )
    manager.register_backend(
        backend_id="ordering-backend",
        backend_type=BackendType.API,
        name="ordering-backend",
        instance=backend_instance,
        capabilities=BackendCapabilities(
            supported_tasks={"text-generation"}, protocols={"http"}
        ),
        endpoint="http://ordering.test.invalid",
    )

    monkeypatch.setattr(server_module, "get_backend_manager", lambda config=None: manager)
    monkeypatch.setattr(server_module, "HAVE_BACKEND_MANAGER", True)

    server = server_module.HFModelServer(
        ServerConfig(
            enable_auth=False,
            enable_metrics=False,
            auto_discover=False,
            enable_hardware_detection=False,
        )
    )

    # Pre-inject fakes so the lazy-init checks find them immediately.
    server._storage_client = _RecordingStorage(log)
    server._datasets_manager = _RecordingDatasets(log)
    server._model_manager = _RecordingModelManager(log)

    return server


# ---------------------------------------------------------------------------
# Phase D Task 2 — Inference success path call-chain ordering
# ---------------------------------------------------------------------------


def test_e2e_inference_success_call_chain_ordering(monkeypatch, tmp_path):
    """
    Gate ordering for the core inference success path:
        backend.run_inference
        → storage.store (input)
        → storage.store (output)
        → datasets.log_event (inference_completed)
        → datasets.track_provenance (inference)
        → model_manager.mark_model_used
    """
    log = _CallLog()
    server = _make_server(monkeypatch, _SuccessBackend(log), log, tmp_path=tmp_path)

    with TestClient(server.app) as client:
        response = client.post(
            "/v1/completions",
            json={"model": "ordering-model", "prompt": "hello"},
        )

    assert response.status_code == 200, f"Unexpected status: {response.status_code} — {response.text}"

    names = log.op_names()

    # Every expected step must be present
    assert "backend.run_inference" in names, "backend.run_inference not recorded"
    assert "storage.store.input" in names, "input storage.store not recorded"
    assert "storage.store.output" in names, "output storage.store not recorded"
    assert "datasets.log_event.inference_completed" in names, "audit log_event not recorded"
    assert "datasets.track_provenance.inference" in names, "provenance track not recorded"
    assert "model_manager.mark_model_used" in names, "mark_model_used not recorded"

    # Order assertions: each step must come strictly before the next
    assert log.index("backend.run_inference") < log.index("storage.store.input"), (
        "backend.run_inference must precede storage.store (input)"
    )
    assert log.index("storage.store.input") < log.index("storage.store.output"), (
        "input store must precede output store"
    )
    assert log.index("storage.store.output") < log.index("datasets.log_event.inference_completed"), (
        "output store must precede audit log"
    )
    assert log.index("datasets.log_event.inference_completed") < log.index(
        "datasets.track_provenance.inference"
    ), "audit log must precede provenance"
    assert log.index("datasets.track_provenance.inference") < log.index(
        "model_manager.mark_model_used"
    ), "provenance must precede mark_model_used"


# ---------------------------------------------------------------------------
# Phase D Task 3 — Inference failure path call-chain ordering
# ---------------------------------------------------------------------------


def test_e2e_inference_failure_call_chain_ordering(monkeypatch, tmp_path):
    """
    Failure path ordering:
        (backend raises)
        → storage.store (input)
        → storage.store (failure / output)
        → datasets.log_event (inference_failed)
        → datasets.track_provenance (inference_failed)
        [model_manager.mark_model_used must NOT appear]
    """
    log = _CallLog()
    server = _make_server(monkeypatch, _FailingBackend(log), log, tmp_path=tmp_path)

    with TestClient(server.app) as client:
        response = client.post(
            "/v1/completions",
            json={"model": "ordering-model", "prompt": "hello"},
        )

    assert response.status_code == 503, f"Expected 503, got {response.status_code}"

    names = log.op_names()

    # Backend recorded its call (even though it raised)
    assert "backend.run_inference.raised" in names, "backend raise not recorded"

    # Failure-path storage and audit steps must have fired
    assert "storage.store.input" in names, "input store not recorded on failure path"
    assert "datasets.log_event.inference_failed" in names, "inference_failed log_event not recorded"
    assert "datasets.track_provenance.inference_failed" in names, "inference_failed provenance not recorded"

    # mark_model_used must NOT be called on the failure path
    assert "model_manager.mark_model_used" not in names, (
        "mark_model_used must not be called when inference fails"
    )

    # Ordering: audit event precedes provenance
    assert log.index("datasets.log_event.inference_failed") < log.index(
        "datasets.track_provenance.inference_failed"
    ), "audit must precede provenance on failure path"


# ---------------------------------------------------------------------------
# Phase D Task 3 — Model lifecycle failure call-chain ordering
# ---------------------------------------------------------------------------


def _make_server_with_failing_model_manager(monkeypatch, log: _CallLog, *, tmp_path):
    """
    Server with a backend manager (for startup) but a model manager that
    reports load failure.
    """
    registry_path = str(tmp_path / "backend_registry.json")
    manager = InferenceBackendManager(
        {"registry_state_path": registry_path, "persist_registry": False}
    )
    # No backend registered — this server is only exercised via /models/load.
    monkeypatch.setattr(server_module, "get_backend_manager", lambda config=None: manager)
    monkeypatch.setattr(server_module, "HAVE_BACKEND_MANAGER", True)

    server = server_module.HFModelServer(
        ServerConfig(
            enable_auth=False,
            enable_metrics=False,
            auto_discover=False,
            enable_hardware_detection=False,
        )
    )
    server._storage_client = _RecordingStorage(log)
    server._datasets_manager = _RecordingDatasets(log)
    # Inject a failing model manager
    server._model_manager = _RecordingModelManager(log, load_success=False)
    return server


def test_e2e_model_load_failure_call_chain_ordering(monkeypatch, tmp_path):
    """
    Model load failure ordering:
        model_manager.add_model_with_ipfs_storage  (returns False)
        → datasets.log_event (model_load_failed)
        → datasets.track_provenance (model_load_failed)
    """
    log = _CallLog()
    server = _make_server_with_failing_model_manager(monkeypatch, log, tmp_path=tmp_path)

    with TestClient(server.app) as client:
        response = client.post(
            "/models/load",
            json={"model_id": "test-model", "hardware": "cpu"},
        )

    assert response.status_code == 500, f"Expected 500, got {response.status_code}"

    names = log.op_names()

    assert "model_manager.add_model_with_ipfs_storage" in names, "add_model not called"
    assert "datasets.log_event.model_load_failed" in names, "model_load_failed log_event not recorded"
    assert "datasets.track_provenance.model_load_failed" in names, "model_load_failed provenance not recorded"

    assert log.index("model_manager.add_model_with_ipfs_storage") < log.index(
        "datasets.log_event.model_load_failed"
    ), "model_manager call must precede audit on load failure"
    assert log.index("datasets.log_event.model_load_failed") < log.index(
        "datasets.track_provenance.model_load_failed"
    ), "audit must precede provenance on model load failure"


def _make_server_with_successful_model_manager(monkeypatch, log: _CallLog, *, tmp_path):
    registry_path = str(tmp_path / "backend_registry.json")
    manager = InferenceBackendManager(
        {"registry_state_path": registry_path, "persist_registry": False}
    )
    monkeypatch.setattr(server_module, "get_backend_manager", lambda config=None: manager)
    monkeypatch.setattr(server_module, "HAVE_BACKEND_MANAGER", True)

    server = server_module.HFModelServer(
        ServerConfig(
            enable_auth=False,
            enable_metrics=False,
            auto_discover=False,
            enable_hardware_detection=False,
        )
    )
    server._storage_client = _RecordingStorage(log)
    server._datasets_manager = _RecordingDatasets(log)
    server._model_manager = _RecordingModelManager(log, load_success=True)
    return server


def test_e2e_model_load_success_call_chain_ordering(monkeypatch, tmp_path):
    """
    Model load success ordering:
        model_manager.add_model_with_ipfs_storage  (returns True)
        → datasets.log_event (model_loaded)
        → datasets.track_provenance (model_load)
    """
    log = _CallLog()
    server = _make_server_with_successful_model_manager(monkeypatch, log, tmp_path=tmp_path)

    with TestClient(server.app) as client:
        response = client.post(
            "/models/load",
            json={"model_id": "test-model", "hardware": "cpu"},
        )

    assert response.status_code == 200, f"Expected 200, got {response.status_code} — {response.text}"

    names = log.op_names()

    assert "model_manager.add_model_with_ipfs_storage" in names, "add_model not called"
    assert "datasets.log_event.model_loaded" in names, "model_loaded log_event not recorded"
    assert "datasets.track_provenance.model_load" in names, "model_load provenance not recorded"

    assert log.index("model_manager.add_model_with_ipfs_storage") < log.index(
        "datasets.log_event.model_loaded"
    ), "model_manager call must precede audit on load success"
    assert log.index("datasets.log_event.model_loaded") < log.index(
        "datasets.track_provenance.model_load"
    ), "audit must precede provenance on model load success"
