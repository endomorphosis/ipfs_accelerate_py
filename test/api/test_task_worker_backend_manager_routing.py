from ipfs_accelerate_py.p2p_tasks.task_queue import TaskQueue
from ipfs_accelerate_py.p2p_tasks.worker import run_worker


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


def test_worker_routes_text_generation_via_backend_manager(monkeypatch, tmp_path):
    fake_manager = _FakeBackendManager()

    import ipfs_accelerate_py.inference_backend_manager as ibm

    monkeypatch.setattr(ibm, "get_backend_manager", lambda config=None: fake_manager)

    queue_path = str(tmp_path / "task_queue.duckdb")
    queue = TaskQueue(queue_path)

    task_id = queue.submit(
        task_type="text-generation",
        model_name="model-x",
        payload={
            "prompt": "hello world",
            "workflow_id": "wf-99",
            "persistence_policy": "required",
            "provenance_policy": "strict",
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

    assert len(fake_manager.calls) == 1
    call = fake_manager.calls[0]
    assert call["task"] == "text-generation"
    assert call["model"] == "model-x"
    assert call["inputs"] == ["hello world"]

    out = queue.get(task_id)
    assert out is not None
    assert out.get("status") == "completed"
    result = out.get("result") or {}

    assert result.get("backend_id") == "backend-1"
    assert result.get("output_cid") == "cid-output"
    assert result.get("provenance_cid") == "cid-provenance"

    lineage = result.get("lineage") or {}
    assert lineage.get("workflow_id") == "wf-99"
    assert lineage.get("task_id") == task_id
    assert lineage.get("model_id") == "model-x"
    assert lineage.get("persistence_policy") == "required"
    assert lineage.get("provenance_policy") == "strict"
