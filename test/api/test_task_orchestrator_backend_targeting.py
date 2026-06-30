import pytest

from ipfs_accelerate_py.p2p_tasks.orchestrator import OrchestratorConfig, TaskOrchestrator
from ipfs_accelerate_py.p2p_tasks.task_queue import TaskQueue
from ipfs_accelerate_py.p2p_tasks.worker import run_worker


class _Remote:
    def __init__(self, peer_id: str, multiaddr: str):
        self.peer_id = peer_id
        self.multiaddr = multiaddr


class _FakeBackendManager:
    def __init__(self):
        self.calls = []

    async def execute_task(self, *, task, model, inputs, preferred_types=None, required_protocols=None, parameters=None):
        self.calls.append(
            {
                "task": task,
                "model": model,
                "inputs": list(inputs),
                "preferred_types": list(preferred_types or []),
                "required_protocols": list(required_protocols or []),
                "parameters": dict(parameters or {}),
            }
        )
        return {
            "backend_id": "backend-selected",
            "backend_type": "hybrid",
            "protocol": (required_protocols or [None])[0],
            "protocols": list(required_protocols or []),
            "output_cid": "cid-out",
            "provenance_cid": "cid-prov",
            "result": {"text": "ok"},
        }


@pytest.mark.parametrize("target_backend", ["docker", "kubernetes"])
def test_orchestrator_proxy_task_routes_backend_target_via_backend_manager(monkeypatch, tmp_path, target_backend):
    fake_manager = _FakeBackendManager()

    import ipfs_accelerate_py.inference_backend_manager as ibm

    monkeypatch.setattr(ibm, "get_backend_manager", lambda config=None: fake_manager)

    queue_path = str(tmp_path / "task_queue.duckdb")
    orchestrator = TaskOrchestrator(
        config=OrchestratorConfig(
            queue_path=queue_path,
            orchestrator_id="orch-test",
            base_worker_id="worker-test",
            min_workers=0,
            max_workers=0,
        ),
        supported_task_types=["text-generation"],
    )

    remote = _Remote("peer-a", "/ip4/127.0.0.1/tcp/4001/p2p/peer-a")
    submitted = orchestrator._submit_proxy_tasks(
        claimed=[
            (
                remote,
                {
                    "task_id": f"remote-{target_backend}",
                    "task_type": "text-generation",
                    "model_name": "model-z",
                    "workflow_id": "wf-backend-target",
                    "payload": {
                        "prompt": "route me",
                        "execution_backend": target_backend,
                        "dispatch_via_backend_manager": True,
                        "persistence_policy": "required",
                        "provenance_policy": "strict",
                    },
                },
            )
        ]
    )
    assert submitted == 1

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
    call = fake_manager.calls[0]
    assert call["task"] == "text-generation"
    assert call["model"] == "model-z"
    assert call["inputs"] == ["route me"]
    assert call["required_protocols"] == [target_backend]

    queue = TaskQueue(queue_path)
    rows = queue.list(status="completed", limit=5)
    assert len(rows) == 1
    result = rows[0]["result"]
    lineage = result["lineage"]

    assert result["backend_id"] == "backend-selected"
    assert result["protocol"] == target_backend
    assert result["output_cid"] == "cid-out"
    assert result["provenance_cid"] == "cid-prov"
    assert lineage["workflow_id"] == "wf-backend-target"
    assert lineage["model_id"] == "model-z"
