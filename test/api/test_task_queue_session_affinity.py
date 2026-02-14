import threading
import time


def test_taskqueue_claim_next_respects_session_id(tmp_path):
    from ipfs_accelerate_py.p2p_tasks.task_queue import TaskQueue

    queue_path = str(tmp_path / "q.duckdb")
    q = TaskQueue(queue_path)

    t1 = q.submit(task_type="llm.generate", model_name="", payload={"prompt": "a", "session_id": "S1"})
    t2 = q.submit(task_type="llm.generate", model_name="", payload={"prompt": "b", "session_id": "S2"})
    t3 = q.submit(task_type="llm.generate", model_name="", payload={"prompt": "c"})

    claimed = q.claim_next(worker_id="w", supported_task_types=["llm.generate"], session_id="S1")
    assert claimed is not None
    assert claimed.task_id in {t1, t3}
    if claimed.task_id == t3:
        # If the no-session task was picked first, the session-bound S1 task should still be claimable.
        claimed2 = q.claim_next(worker_id="w", supported_task_types=["llm.generate"], session_id="S1")
        assert claimed2 is not None
        assert claimed2.task_id == t1

    # S2 task must not be claimed by an S1 worker.
    t2_row = q.get(t2)
    assert t2_row is not None
    assert t2_row["status"] == "queued"


def test_worker_skips_mismatched_session_tasks(monkeypatch, tmp_path):
    from ipfs_accelerate_py.p2p_tasks.task_queue import TaskQueue
    from ipfs_accelerate_py.p2p_tasks.worker import run_worker

    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_COPILOT_CLI", "1")
    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_P2P_SESSION", "S1")

    import ipfs_accelerate_py.llm_router as llm_router

    def fake_generate_text(*args, **kwargs):
        return "ok"

    monkeypatch.setattr(llm_router, "generate_text", fake_generate_text)

    queue_path = str(tmp_path / "q.duckdb")
    q = TaskQueue(queue_path)

    tid_wrong = q.submit(
        task_type="llm.generate",
        model_name="",
        payload={"prompt": "nope", "provider": "copilot_cli", "session_id": "S2"},
    )
    tid_right = q.submit(
        task_type="llm.generate",
        model_name="",
        payload={"prompt": "yep", "provider": "copilot_cli", "session_id": "S1"},
    )

    t = threading.Thread(
        target=lambda: run_worker(
            queue_path=queue_path,
            worker_id="w1",
            once=True,
            supported_task_types=["llm.generate"],
        ),
        daemon=True,
    )
    t.start()
    t.join(timeout=10)

    wrong = q.get(tid_wrong)
    assert wrong is not None
    assert wrong["status"] == "queued"

    right = q.get(tid_right)
    assert right is not None
    assert right["status"] == "completed"
    assert (right.get("result") or {}).get("session_id") == "S1"
    assert (right.get("result") or {}).get("provider") == "copilot_cli"
    assert (right.get("result") or {}).get("text") == "ok"
