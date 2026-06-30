from ipfs_accelerate_py.p2p_tasks.task_queue import TaskQueue
from ipfs_accelerate_py.p2p_tasks.worker import run_worker


class _FailingBackendManager:
    async def execute_task(self, **kwargs):
        raise RuntimeError("backend unavailable")


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
