from ipfs_accelerate_py.p2p_tasks.task_queue import TaskQueue
from ipfs_accelerate_py.p2p_tasks.worker import run_worker


class _FailingBackendManager:
    async def execute_task(self, **kwargs):
        raise RuntimeError("backend unavailable")


class _FakeDatasetsManager:
    def __init__(self):
        self.events = []
        self.provenance = []

    def log_event(self, event_type, data, level="INFO", category="GENERAL"):
        self.events.append({"event_type": event_type, "data": dict(data or {}), "level": level, "category": category})
        return True

    def track_provenance(self, operation, data, record_type="TRANSFORMATION"):
        self.provenance.append({"operation": operation, "data": dict(data or {}), "record_type": record_type})
        return "cid-prov"


def test_worker_fails_closed_when_backend_manager_required(monkeypatch, tmp_path):
    import ipfs_accelerate_py.inference_backend_manager as ibm

    monkeypatch.setattr(ibm, "get_backend_manager", lambda config=None: _FailingBackendManager())

    queue_path = str(tmp_path / "task_queue.duckdb")
    queue = TaskQueue(queue_path)

    task_id = queue.submit(
        task_type="text-generation",
        model_name="model-x",
        payload={
            "prompt": "must use backend manager",
            "workflow_id": "wf-strict",
            "persistence_policy": "required",
            "provenance_policy": "strict",
            "dispatch_via_backend_manager": "required",
        },
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

    out = queue.get(task_id)
    assert out is not None
    assert out.get("status") == "failed"
    assert "backend_manager_routing_required" in str(out.get("error") or "")

    result = out.get("result") or {}
    assert result.get("success") is False
    assert result.get("error_message") == "backend_manager_routing_required"

    lineage = result.get("lineage") or {}
    assert lineage.get("workflow_id") == "wf-strict"
    assert lineage.get("task_id") == task_id
    assert lineage.get("model_id") == "model-x"
    assert lineage.get("persistence_policy") == "required"
    assert lineage.get("provenance_policy") == "strict"


def test_worker_emits_backend_routing_failure_event_when_backend_manager_raises(monkeypatch, tmp_path):
    import ipfs_accelerate_py.inference_backend_manager as ibm
    import ipfs_accelerate_py.datasets_integration as datasets_integration

    fake_datasets = _FakeDatasetsManager()
    monkeypatch.setattr(ibm, "get_backend_manager", lambda config=None: _FailingBackendManager())
    monkeypatch.setattr(datasets_integration, "DatasetsManager", lambda *args, **kwargs: fake_datasets)

    queue_path = str(tmp_path / "task_queue.duckdb")
    queue = TaskQueue(queue_path)

    task_id = queue.submit(
        task_type="text-generation",
        model_name="model-x",
        payload={
            "prompt": "must use backend manager",
            "workflow_id": "wf-strict",
            "persistence_policy": "required",
            "provenance_policy": "strict",
            "dispatch_via_backend_manager": "required",
        },
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

    out = queue.get(task_id)
    assert out is not None
    assert out.get("status") == "failed"
    assert fake_datasets.events
    assert any(event["event_type"] == "workflow_task_failed_backend_routing" for event in fake_datasets.events)
    assert fake_datasets.provenance
    assert any(item["operation"] == "workflow_task_failed_backend_routing" for item in fake_datasets.provenance)


def test_worker_emits_degraded_event_when_backend_manager_optional_and_falls_back(monkeypatch, tmp_path):
    import ipfs_accelerate_py.inference_backend_manager as ibm
    import ipfs_accelerate_py.datasets_integration as datasets_integration
    import ipfs_accelerate_py.p2p_tasks.worker as worker_module

    fake_datasets = _FakeDatasetsManager()
    monkeypatch.setattr(ibm, "get_backend_manager", lambda config=None: _FailingBackendManager())
    monkeypatch.setattr(datasets_integration, "DatasetsManager", lambda *args, **kwargs: fake_datasets)
    monkeypatch.setattr(worker_module, "_run_text_generation", lambda task, accelerate_instance=None: {"text": "fallback"})

    queue_path = str(tmp_path / "task_queue.duckdb")
    queue = TaskQueue(queue_path)

    task_id = queue.submit(
        task_type="text-generation",
        model_name="model-x",
        payload={
            "prompt": "backend manager optional",
            "workflow_id": "wf-optional",
            "dispatch_via_backend_manager": "1",
        },
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

    out = queue.get(task_id)
    assert out is not None
    assert out.get("status") == "completed"
    assert (out.get("result") or {}).get("text") == "fallback"

    degraded_events = [e for e in fake_datasets.events if e["event_type"] == "workflow_backend_routing_degraded"]
    assert degraded_events
    assert degraded_events[0]["level"] == "WARNING"
    assert any(item["operation"] == "workflow_backend_routing_degraded" for item in fake_datasets.provenance)
