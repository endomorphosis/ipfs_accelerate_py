from ipfs_accelerate_py.p2p_tasks.orchestrator import OrchestratorConfig, TaskOrchestrator
from ipfs_accelerate_py.p2p_tasks.task_queue import TaskQueue


class _Remote:
    def __init__(self, peer_id: str, multiaddr: str):
        self.peer_id = peer_id
        self.multiaddr = multiaddr


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


def test_orchestrator_proxy_submission_carries_lineage(tmp_path):
    queue_path = str(tmp_path / "tasks.duckdb")
    orchestrator = TaskOrchestrator(
        config=OrchestratorConfig(
            queue_path=queue_path,
            orchestrator_id="orch-test",
            base_worker_id="worker-test",
            min_workers=0,
            max_workers=0,
        )
    )

    remote = _Remote("peer-1", "/ip4/127.0.0.1/tcp/4001/p2p/peer-1")
    claimed = [
        (
            remote,
            {
                "task_id": "remote-task-1",
                "task_type": "text-generation",
                "model_name": "model-a",
                "workflow_id": "wf-123",
                "payload": {
                    "prompt": "hello",
                    "persistence_policy": "required",
                    "provenance_policy": "strict",
                },
            },
        )
    ]

    submitted = orchestrator._submit_proxy_tasks(claimed=claimed)
    assert submitted == 1

    queue = TaskQueue(queue_path)
    rows = queue.list(status="queued", limit=5)
    assert len(rows) == 1
    payload = rows[0]["payload"]

    assert payload["_p2p_proxy"]["task_id"] == "remote-task-1"
    assert payload["_lineage"]["workflow_id"] == "wf-123"
    assert payload["_lineage"]["task_id"] == "remote-task-1"
    assert payload["_lineage"]["model_id"] == "model-a"
    assert payload["_lineage"]["persistence_policy"] == "required"
    assert payload["_lineage"]["provenance_policy"] == "strict"


def test_orchestrator_workflow_events_emit_provenance(monkeypatch, tmp_path):
    queue_path = str(tmp_path / "tasks.duckdb")
    orchestrator = TaskOrchestrator(
        config=OrchestratorConfig(
            queue_path=queue_path,
            orchestrator_id="orch-test",
            base_worker_id="worker-test",
            min_workers=0,
            max_workers=0,
        )
    )
    fake_datasets = _FakeDatasetsManager()
    monkeypatch.setattr(orchestrator, "_get_datasets_manager", lambda: fake_datasets)

    orchestrator._log_workflow_event(
        "workflow_dispatched",
        {
            "workflow_id": "wf-123",
            "task_id": "task-1",
            "model_id": "model-a",
            "orchestrator_id": "orch-test",
            "peer_id": "peer-1",
        },
    )

    assert len(fake_datasets.events) == 1
    assert fake_datasets.events[0]["event_type"] == "workflow_dispatched"
    assert len(fake_datasets.provenance) == 1
    assert fake_datasets.provenance[0]["operation"] == "workflow_dispatched"
