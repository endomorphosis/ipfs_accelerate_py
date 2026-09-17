#!/usr/bin/env python3
"""Smoke test: producer -> shared task queue -> hookable worker -> results.

Demonstrates the full loop Benjamin asked for:

1. A "muse" model is registered in the ModelManager.
2. A producer submits tasks to the shared task queue.
3. A consumer (the P2P worker) claims them via a hooked task handler that
   resolves the "muse" model from the model manager to fulfill each task.
4. Tasks drain and every output is verified correct.

Run directly::

    python test/test_smoke_p2p_hooks.py

or via pytest::

    python -m pytest test/test_smoke_p2p_hooks.py -q

The worker's claim/execute/complete path exercised here is the same one the
mesh worker uses to drain peers' queues over libp2p (see
``p2p_tasks/worker.py::_maybe_claim_from_mesh``); only the transport differs.
"""

import os
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TASK_TYPE = "agent.muse"
MODEL_ID = "muse"
NUM_TASKS = 10
DRAIN_TIMEOUT_S = 120


def _quiet_worker_env():
    # Keep the worker light: no HF/multimodal probing, no mesh discovery.
    os.environ.setdefault("IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_HF", "0")
    os.environ.setdefault("IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_MULTIMODAL", "0")


def register_muse_model(storage_path):
    """Register the muse model in the model manager; return the manager."""
    from ipfs_accelerate_py.model_manager import (
        DataType,
        IOSpec,
        ModelManager,
        ModelMetadata,
        ModelType,
    )

    manager = ModelManager(storage_path=storage_path, use_database=False)
    metadata = ModelMetadata(
        model_id=MODEL_ID,
        model_name="muse",
        model_type=ModelType.LANGUAGE_MODEL,
        architecture="transformer",
        inputs=[IOSpec(name="prompt", data_type=DataType.TEXT)],
        outputs=[IOSpec(name="reply", data_type=DataType.TEXT)],
        description="Muse model served through the P2P task-worker hook.",
        tags=["p2p", "smoke"],
    )
    assert manager.add_model(metadata), "failed to register muse model"
    registered = manager.get_model(MODEL_ID)
    assert registered is not None, "muse model not found after registration"
    print(f"[producer] registered model '{registered.model_id}' "
          f"({registered.model_type.value}) in the model manager")
    return manager


def make_handler(manager):
    """Build the hooked task handler the consumer worker runs."""

    def muse_handler(task):
        payload = task.get("payload") or {}
        # The hook fulfills the task through the model registered in the
        # model manager -- this is the "hook into the worker" seam.
        model = manager.get_model(MODEL_ID)
        if model is None:
            raise RuntimeError(f"model '{MODEL_ID}' is not registered")
        prompt = payload.get("prompt", "")
        # Deterministic stand-in for a model call: the smoke verifies the
        # exact transform below, proving the task drained through the hook.
        return {
            "reply": f"[{model.model_id}] {prompt.upper()}",
            "model": model.model_id,
            "index": payload.get("index"),
        }

    return muse_handler


def produce(queue_path, num_tasks):
    """Submit tasks; return the list of task ids."""
    from ipfs_accelerate_py.p2p_tasks.task_queue import TaskQueue

    queue = TaskQueue(queue_path)
    try:
        task_ids = []
        for i in range(num_tasks):
            task_id = queue.submit(
                task_type=TASK_TYPE,
                model_name=MODEL_ID,
                payload={"index": i, "prompt": f"smoke-prompt-{i}"},
            )
            task_ids.append(task_id)
        print(f"[producer] submitted {len(task_ids)} tasks of type '{TASK_TYPE}'")
        return task_ids
    finally:
        queue.close()


def consume(queue_path, handler, stop_event):
    """Run the worker until stop_event is set. Runs in its own thread."""
    from ipfs_accelerate_py.p2p_tasks.worker import run_worker

    rc = run_worker(
        queue_path=queue_path,
        worker_id="smoke-consumer",
        poll_interval_s=0.1,
        mesh=False,
        p2p_service=False,
        stop_event=stop_event,
        extra_handlers={TASK_TYPE: handler},
    )
    print(f"[consumer] worker exited rc={rc}")


def await_drain(queue_path, task_ids, timeout_s):
    """Poll until every task is completed; return {task_id: task}."""
    from ipfs_accelerate_py.p2p_tasks.task_queue import TaskQueue

    deadline = time.time() + timeout_s
    done = {}
    queue = TaskQueue(queue_path)
    try:
        while time.time() < deadline:
            for task_id in task_ids:
                if task_id in done:
                    continue
                task = queue.get(task_id)
                if task and task.get("status") == "completed":
                    done[task_id] = task
            if len(done) == len(task_ids):
                break
            time.sleep(0.25)
    finally:
        queue.close()
    return done


def verify(task_ids, done):
    """Check every output is exactly what the hook should have produced."""
    assert len(done) == len(task_ids), (
        f"only {len(done)}/{len(task_ids)} tasks drained"
    )
    for i, task_id in enumerate(task_ids):
        task = done[task_id]
        result = task.get("result") or {}
        expected_reply = f"[{MODEL_ID}] SMOKE-PROMPT-{i}"
        assert result.get("reply") == expected_reply, (
            f"task {task_id}: reply {result.get('reply')!r} != {expected_reply!r}"
        )
        assert result.get("model") == MODEL_ID, f"task {task_id}: wrong model"
        assert result.get("index") == i, f"task {task_id}: wrong index"
    print(f"[verify] all {len(task_ids)} tasks drained with correct outputs")


def main():
    _quiet_worker_env()
    tmp = tempfile.mkdtemp(prefix="p2p_hook_smoke_")
    queue_path = os.path.join(tmp, "queue.duckdb")
    manager_path = os.path.join(tmp, "model_manager.json")

    manager = register_muse_model(manager_path)
    task_ids = produce(queue_path, NUM_TASKS)

    stop_event = threading.Event()
    worker_thread = threading.Thread(
        target=consume,
        args=(queue_path, make_handler(manager), stop_event),
        daemon=True,
    )
    print("[consumer] starting worker with hooked 'agent.muse' handler")
    worker_thread.start()
    try:
        done = await_drain(queue_path, task_ids, DRAIN_TIMEOUT_S)
    finally:
        stop_event.set()
        worker_thread.join(timeout=30)

    verify(task_ids, done)
    print("SMOKE OK: producer -> queue -> hooked worker (muse model) -> "
          f"{len(task_ids)}/{len(task_ids)} correct results")


def test_smoke_producer_consumer_muse_model():
    main()


if __name__ == "__main__":
    main()
