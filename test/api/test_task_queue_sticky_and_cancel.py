from __future__ import annotations

import os
import tempfile


def _new_queue_path() -> str:
    td = tempfile.TemporaryDirectory(prefix="ipfs-accel-taskq-")
    # Keep directory alive by attaching to function attribute.
    if not hasattr(_new_queue_path, "_tds"):
        _new_queue_path._tds = []  # type: ignore[attr-defined]
    _new_queue_path._tds.append(td)  # type: ignore[attr-defined]
    return os.path.join(td.name, "queue.duckdb")


def test_sticky_worker_id_only_claimable_by_target_worker() -> None:
    from ipfs_accelerate_py.p2p_tasks.task_queue import TaskQueue

    q = TaskQueue(_new_queue_path())
    try:
        tid = q.submit(
            task_type="llm.generate",
            model_name="",
            payload={"prompt": "hi", "sticky_worker_id": "worker-A"},
        )

        # Wrong worker cannot claim.
        t0 = q.claim_next(worker_id="worker-B", supported_task_types=["llm.generate"], session_id=None)
        assert t0 is None

        # Correct worker can claim.
        t1 = q.claim_next(worker_id="worker-A", supported_task_types=["llm.generate"], session_id=None)
        assert t1 is not None
        assert t1.task_id == tid
        assert t1.assigned_worker == "worker-A"
    finally:
        q.close()


def test_cancel_only_affects_queued_tasks() -> None:
    from ipfs_accelerate_py.p2p_tasks.task_queue import TaskQueue

    q = TaskQueue(_new_queue_path())
    try:
        tid = q.submit(task_type="llm.generate", model_name="", payload={"prompt": "x"})
        assert q.cancel(task_id=tid, reason="test") is True

        rec = q.get(tid)
        assert rec is not None
        assert rec.get("status") == "cancelled"

        # Cannot cancel a non-queued task.
        assert q.cancel(task_id=tid, reason="again") is False
    finally:
        q.close()
