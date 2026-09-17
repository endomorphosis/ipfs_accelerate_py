"""Smoke: hooked P2P task worker with all traffic over MCP++ on REAL libp2p.

Producer peer and consumer peer are both real py-libp2p hosts (localhost TCP).
The producer dials the consumer over libp2p and speaks MCP++ (/mcp+p2p/1.0.0):
  initialize -> tools/call taskqueue.submit x10 -> tools/call taskqueue.get_status
The consumer's hooked worker claims the tasks via the shared queue and fulfills
them by resolving the registered "muse" model from the ModelManager.

Requires the ``libp2p`` package. Run from the repo root::

    python test/test_smoke_p2p_hooks_mcp_libp2p.py

Exit code 0 on success; raises/prints on failure.
"""

import os
import sys
import tempfile
import threading

import trio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ipfs_accelerate_py.p2p_tasks.mcp_p2p import (  # noqa: E402
    PROTOCOL_MCP_P2P_V1,
    handle_mcp_p2p_stream,
)
from ipfs_accelerate_py.p2p_tasks.mcp_p2p_client import (  # noqa: E402
    MCPP2PClient,
    open_libp2p_stream_by_multiaddr,
    trio_libp2p_host_listen,
)
from ipfs_accelerate_py.p2p_tasks.task_queue import TaskQueue  # noqa: E402
from ipfs_accelerate_py.p2p_tasks.worker import run_worker  # noqa: E402

TASK_TYPE = "agent.muse"
MODEL_ID = "muse"
NUM_TASKS = 10
POLL_TIMEOUT_S = 120.0


def _quiet_worker_env():
    os.environ.setdefault("IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_HF", "0")
    os.environ.setdefault("IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_MULTIMODAL", "0")


def register_muse_model(storage_path):
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
        tags=["p2p", "smoke", "mcp++", "libp2p"],
    )
    assert manager.add_model(metadata), "failed to register muse model"
    registered = manager.get_model(MODEL_ID)
    assert registered is not None, "muse model not found after registration"
    print(
        f"[producer] registered model '{registered.model_id}' "
        f"({registered.model_type.value}) in the model manager",
        flush=True,
    )
    return manager


def make_handler(manager):
    def muse_handler(task):
        payload = task.get("payload") or {}
        model = manager.get_model(MODEL_ID)
        if model is None:
            raise RuntimeError(f"model '{MODEL_ID}' is not registered")
        prompt = payload.get("prompt", "")
        return {
            "reply": f"[{model.model_id}] {prompt.upper()}",
            "model": model.model_id,
            "index": payload.get("index"),
        }

    return muse_handler


class ToolRegistry:
    def __init__(self, tools):
        self.tools = tools


def build_registry(queue_path):
    queue = TaskQueue(queue_path)

    def submit(task_type, model_name="default", payload=None):
        task_id = queue.submit(
            task_type=str(task_type),
            model_name=str(model_name),
            payload=dict(payload or {}),
        )
        return {"task_id": task_id}

    def get_status(task_id):
        task = queue.get(str(task_id))
        if task is None:
            return {"task_id": str(task_id), "status": "unknown"}
        return {
            "task_id": task.get("task_id"),
            "status": task.get("status"),
            "result": task.get("result"),
            "error": task.get("error"),
        }

    return ToolRegistry(
        {
            "taskqueue.submit": {
                "description": "Submit a task to the shared queue.",
                "input_schema": {"type": "object"},
                "function": submit,
            },
            "taskqueue.get_status": {
                "description": "Get task status/result from the shared queue.",
                "input_schema": {"type": "object"},
                "function": get_status,
            },
        }
    )


def consume(queue_path, handler, stop_event):
    rc = run_worker(
        queue_path=queue_path,
        worker_id="smoke-consumer-mcp-libp2p",
        poll_interval_s=0.1,
        mesh=False,
        p2p_service=False,
        stop_event=stop_event,
        extra_handlers={TASK_TYPE: handler},
    )
    print(f"[consumer] worker exited rc={rc}", flush=True)


async def producer_flow(client: MCPP2PClient):
    init = await client.request(
        "initialize", {"protocolVersion": "2024-11-05", "capabilities": {}}
    )
    transport = init["result"].get("transport")
    server = init["result"].get("server")
    assert transport == PROTOCOL_MCP_P2P_V1, f"unexpected transport {transport!r}"
    print(
        f"[producer] MCP++ session initialized over libp2p "
        f"(transport={transport}, server={server})",
        flush=True,
    )

    print(f"[producer] submitting {NUM_TASKS} tasks over MCP++ ...", flush=True)
    task_ids = []
    for i in range(NUM_TASKS):
        resp = await client.request(
            "tools/call",
            {
                "name": "taskqueue.submit",
                "arguments": {
                    "task_type": TASK_TYPE,
                    "model_name": MODEL_ID,
                    "payload": {"index": i, "prompt": f"smoke-prompt-{i}"},
                },
            },
        )
        task_ids.append(resp["result"]["content"]["task_id"])
    print(f"[producer] submitted {len(task_ids)} tasks of type '{TASK_TYPE}'", flush=True)

    deadline = trio.current_time() + POLL_TIMEOUT_S
    results = {}
    while len(results) < NUM_TASKS:
        if trio.current_time() > deadline:
            raise TimeoutError(
                f"timed out waiting for tasks; got {len(results)}/{NUM_TASKS}"
            )
        for task_id in task_ids:
            if task_id in results:
                continue
            resp = await client.request(
                "tools/call",
                {"name": "taskqueue.get_status", "arguments": {"task_id": task_id}},
            )
            content = resp["result"]["content"]
            if content.get("status") == "completed":
                results[task_id] = content.get("result")
        if len(results) < NUM_TASKS:
            await trio.sleep(0.25)

    for i, task_id in enumerate(task_ids):
        res = results[task_id]
        expected_reply = f"[{MODEL_ID}] SMOKE-PROMPT-{i}"
        assert res.get("reply") == expected_reply, (
            f"task {task_id}: reply {res.get('reply')!r} != {expected_reply!r}"
        )
        assert res.get("model") == MODEL_ID, f"task {task_id}: wrong model"
        assert res.get("index") == i, f"task {task_id}: wrong index"
    print(
        f"[verify] all {NUM_TASKS} tasks drained over MCP++/libp2p with correct outputs",
        flush=True,
    )


async def amain():
    _quiet_worker_env()
    tmp = tempfile.mkdtemp(prefix="p2p_hook_smoke_mcp_libp2p_")
    queue_path = os.path.join(tmp, "queue.duckdb")
    manager_path = os.path.join(tmp, "model_manager.json")

    manager = register_muse_model(manager_path)
    registry = build_registry(queue_path)

    async with trio_libp2p_host_listen() as consumer:
        async def on_stream(stream):
            await handle_mcp_p2p_stream(
                stream, local_peer_id="smoke-consumer", registry=registry
            )

        consumer.set_stream_handler(PROTOCOL_MCP_P2P_V1, on_stream)
        port = None
        for addr in (str(a) for a in consumer.get_addrs()):
            parts = addr.split("/")
            if "tcp" in parts:
                port = parts[parts.index("tcp") + 1]
                break
        assert port, "no tcp listen addr found"
        consumer_id = consumer.get_id()
        peer_id = (
            consumer_id.to_base58()
            if hasattr(consumer_id, "to_base58")
            else str(consumer_id)
        )
        print(
            f"[consumer] libp2p peer {peer_id} listening on "
            f"{[str(a) for a in consumer.get_addrs()]}",
            flush=True,
        )

        stop_event = threading.Event()
        worker_thread = threading.Thread(
            target=consume,
            args=(queue_path, make_handler(manager), stop_event),
            name="smoke-mcp-libp2p-worker",
            daemon=True,
        )

        async with trio_libp2p_host_listen() as producer:
            stream = await open_libp2p_stream_by_multiaddr(
                producer,
                peer_multiaddr=f"/ip4/127.0.0.1/tcp/{port}/p2p/{peer_id}",
                protocols=[PROTOCOL_MCP_P2P_V1],
            )
            print("[producer] libp2p stream opened to consumer", flush=True)
            client = MCPP2PClient(stream)
            try:
                print("[consumer] starting worker with hooked 'agent.muse' handler", flush=True)
                worker_thread.start()
                try:
                    await producer_flow(client)
                finally:
                    stop_event.set()
                    worker_thread.join(timeout=30)
                    if worker_thread.is_alive():
                        print("[consumer] worker did not exit in time", flush=True)
            finally:
                await stream.close()

    print(
        "SMOKE OK (mcp++/libp2p): producer -> real libp2p -> /mcp+p2p/1.0.0 "
        f"-> queue -> hooked worker (muse model) -> {NUM_TASKS}/{NUM_TASKS} correct results",
        flush=True,
    )


def main() -> int:
    trio.run(amain)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
