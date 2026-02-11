import threading
import time


def test_worker_publishes_heartbeat_for_text_generation(monkeypatch, tmp_path):
    from ipfs_accelerate_py.p2p_tasks.task_queue import TaskQueue
    from ipfs_accelerate_py.p2p_tasks.worker import run_worker

    # Make generation slow enough that we can observe heartbeats.
    import ipfs_accelerate_py.llm_router as llm_router

    def slow_generate_text(*args, **kwargs):
        time.sleep(0.8)
        return "ok"

    monkeypatch.setattr(llm_router, "generate_text", slow_generate_text)

    queue_path = str(tmp_path / "q.duckdb")
    queue = TaskQueue(queue_path)

    tid = queue.submit(
        task_type="text-generation",
        model_name="any",
        payload={"prompt": "hi", "max_new_tokens": 4},
    )

    t = threading.Thread(
        target=lambda: run_worker(
            queue_path=queue_path,
            worker_id="w1",
            once=True,
            supported_task_types=["text-generation"],
        ),
        daemon=True,
    )
    t.start()

    # Poll until running and heartbeat appears.
    hb1 = None
    deadline = time.time() + 2.0
    while time.time() < deadline:
        task = queue.get(tid)
        assert task is not None
        if task["status"] == "running":
            prog = (task.get("result") or {}).get("progress") if isinstance(task.get("result"), dict) else None
            if isinstance(prog, dict) and prog.get("heartbeat_ts"):
                hb1 = float(prog["heartbeat_ts"])
                break
        time.sleep(0.05)
    assert hb1 is not None

    # Ensure the heartbeat advances while still running.
    hb2 = hb1
    deadline = time.time() + 2.0
    while time.time() < deadline:
        task = queue.get(tid)
        assert task is not None
        if task["status"] == "running":
            prog = (task.get("result") or {}).get("progress") if isinstance(task.get("result"), dict) else None
            if isinstance(prog, dict) and prog.get("heartbeat_ts"):
                hb2 = float(prog["heartbeat_ts"])
                if hb2 > hb1:
                    break
        time.sleep(0.1)
    assert hb2 > hb1

    t.join(timeout=5)
    final = queue.get(tid)
    assert final is not None
    assert final["status"] == "completed"
    assert final["result"]["text"] == "ok"
