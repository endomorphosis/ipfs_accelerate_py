#!/usr/bin/env python3
"""Smoke test: producer -> MCP++ protocol -> shared task queue -> worker.

Same loop as ``test_smoke_p2p_hooks.py``, but every producer interaction
flows over the MCP++ protocol (``/mcp+p2p/1.0.0``): u32-framed JSON-RPC
messages (``initialize`` -> ``tools/call``) carried on a libp2p-style stream.

The consumer side runs the real ``handle_mcp_p2p_stream`` protocol handler
with a tool registry exposing ``taskqueue.submit`` / ``taskqueue.get_status``
backed by the real ``TaskQueue``. The stream is an in-memory duplex pair
standing in for a libp2p stream (the ``libp2p`` package is not installed in
this environment, so no real sockets are opened); the framing, JSON-RPC
dispatch, and tool calls are the production code paths.

1. A "muse" model is registered in the ModelManager.
2. The producer speaks MCP++ over the stream: initialize, then
   ``tools/call taskqueue.submit`` x N.
3. The consumer worker claims tasks via the hooked handler, which resolves
   the "muse" model from the model manager to fulfill each task.
4. The producer polls ``tools/call taskqueue.get_status`` over the same
   protocol until every task drains; outputs are verified correct.

Run directly::

    python test/test_smoke_p2p_hooks_mcp.py
"""

import asyncio
import os
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ipfs_accelerate_py.p2p_tasks.mcp_p2p import (  # noqa: E402
    PROTOCOL_MCP_P2P_V1,
    handle_mcp_p2p_stream,
    read_u32_framed_json,
    write_u32_framed_json,
)

TASK_TYPE = "agent.muse"
MODEL_ID = "muse"
NUM_TASKS = 10
DRAIN_TIMEOUT_S = 120


def _quiet_worker_env():
    os.environ.setdefault("IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_HF", "0")
    os.environ.setdefault("IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_MULTIMODAL", "0")


# ---------------------------------------------------------------------------
# In-memory duplex stream pair (stands in for a libp2p stream).
# ---------------------------------------------------------------------------
class MemoryStream:
    """Async byte stream with libp2p-stream-like read/write/close."""

    def __init__(self):
        self._incoming = asyncio.Queue()
        self._peer = None
        self._buf = bytearray()
        self._eof = False
        self._closed = False

    def _link(self, peer):
        self._peer = peer

    async def read(self, n):
        n = max(1, int(n))
        while len(self._buf) < n and not self._eof:
            chunk = await self._incoming.get()
            if chunk is None:
                self._eof = True
                break
            self._buf.extend(chunk)
        out = bytes(self._buf[:n])
        del self._buf[:n]
        return out

    async def write(self, data):
        await self._peer._incoming.put(bytes(data))

    async def close(self):
        if not self._closed:
            self._closed = True
            await self._peer._incoming.put(None)


def make_stream_pair():
    a, b = MemoryStream(), MemoryStream()
    a._link(b)
    b._link(a)
    return a, b


# ---------------------------------------------------------------------------
# Model registration + hooked handler (same as the direct-queue smoke).
# ---------------------------------------------------------------------------
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
        tags=["p2p", "smoke", "mcp++"],
    )
    assert manager.add_model(metadata), "failed to register muse model"
    registered = manager.get_model(MODEL_ID)
    assert registered is not None, "muse model not found after registration"
    print(f"[producer] registered model '{registered.model_id}' "
          f"({registered.model_type.value}) in the model manager")
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


# ---------------------------------------------------------------------------
# Consumer-side MCP++ tool registry backed by the real task queue.
# ---------------------------------------------------------------------------
class ToolRegistry:
    def __init__(self, tools):
        self.tools = tools


def build_registry(queue_path):
    from ipfs_accelerate_py.p2p_tasks.task_queue import TaskQueue

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


# ---------------------------------------------------------------------------
# Producer: speaks MCP++ (JSON-RPC over framed stream).
# ---------------------------------------------------------------------------
class MCPProducer:
    def __init__(self, stream):
        self._stream = stream
        self._next_id = 0

    async def _request(self, method, params):
        self._next_id += 1
        msg_id = self._next_id
        await write_u32_framed_json(
            self._stream,
            {"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params},
        )
        resp, err = await read_u32_framed_json(self._stream)
        assert err is None, f"protocol read error: {err}"
        assert resp.get("id") == msg_id, f"id mismatch: {resp}"
        if "error" in resp:
            raise RuntimeError(f"MCP++ error: {resp['error']}")
        return resp.get("result")

    async def initialize(self):
        result = await self._request(
            "initialize",
            {"protocolVersion": "2024-11-05", "capabilities": {}},
        )
        transport = result.get("transport")
        assert transport == PROTOCOL_MCP_P2P_V1, f"unexpected transport {transport}"
        print(f"[producer] MCP++ session initialized "
              f"(transport={transport}, server={result.get('server')})")

    async def call_tool(self, name, arguments):
        result = await self._request(
            "tools/call", {"name": name, "arguments": arguments}
        )
        return result.get("content")

    async def submit_task(self, task_type, model_name, payload):
        content = await self.call_tool(
            "taskqueue.submit",
            {"task_type": task_type, "model_name": model_name,
             "payload": payload},
        )
        return content["task_id"]

    async def get_status(self, task_id):
        return await self.call_tool("taskqueue.get_status", {"task_id": task_id})

    async def close(self):
        await self._stream.close()


# ---------------------------------------------------------------------------
# Consumer worker thread (same hook path as the direct-queue smoke).
# ---------------------------------------------------------------------------
def consume(queue_path, handler, stop_event):
    from ipfs_accelerate_py.p2p_tasks.worker import run_worker

    rc = run_worker(
        queue_path=queue_path,
        worker_id="smoke-consumer-mcp",
        poll_interval_s=0.1,
        mesh=False,
        p2p_service=False,
        stop_event=stop_event,
        extra_handlers={TASK_TYPE: handler},
    )
    print(f"[consumer] worker exited rc={rc}")


async def amain():
    _quiet_worker_env()
    tmp = tempfile.mkdtemp(prefix="p2p_hook_smoke_mcp_")
    queue_path = os.path.join(tmp, "queue.duckdb")
    manager_path = os.path.join(tmp, "model_manager.json")

    manager = register_muse_model(manager_path)

    # Consumer side of the MCP++ stream: real protocol handler + tool registry.
    producer_stream, consumer_stream = make_stream_pair()
    registry = build_registry(queue_path)
    handler_task = asyncio.create_task(
        handle_mcp_p2p_stream(
            consumer_stream, local_peer_id="smoke-consumer", registry=registry
        )
    )

    # Worker thread drains the queue through the hooked handler.
    stop_event = threading.Event()
    worker_thread = threading.Thread(
        target=consume,
        args=(queue_path, make_handler(manager), stop_event),
        daemon=True,
    )

    producer = MCPProducer(producer_stream)
    try:
        await producer.initialize()

        print(f"[producer] submitting {NUM_TASKS} tasks over MCP++ ...")
        task_ids = []
        for i in range(NUM_TASKS):
            task_id = await producer.submit_task(
                TASK_TYPE, MODEL_ID, {"index": i, "prompt": f"smoke-prompt-{i}"}
            )
            task_ids.append(task_id)
        print(f"[producer] submitted {len(task_ids)} tasks of type '{TASK_TYPE}'")

        print("[consumer] starting worker with hooked 'agent.muse' handler")
        worker_thread.start()

        deadline = time.time() + DRAIN_TIMEOUT_S
        done = {}
        while time.time() < deadline and len(done) < len(task_ids):
            for task_id in task_ids:
                if task_id in done:
                    continue
                status = await producer.get_status(task_id)
                if status.get("status") == "completed":
                    done[task_id] = status
            if len(done) < len(task_ids):
                await asyncio.sleep(0.25)
        assert len(done) == len(task_ids), (
            f"only {len(done)}/{len(task_ids)} tasks drained over MCP++"
        )

        for i, task_id in enumerate(task_ids):
            result = done[task_id].get("result") or {}
            expected_reply = f"[{MODEL_ID}] SMOKE-PROMPT-{i}"
            assert result.get("reply") == expected_reply, (
                f"task {task_id}: reply {result.get('reply')!r} != {expected_reply!r}"
            )
            assert result.get("model") == MODEL_ID, f"task {task_id}: wrong model"
            assert result.get("index") == i, f"task {task_id}: wrong index"
        print(f"[verify] all {len(task_ids)} tasks drained over MCP++ "
              f"with correct outputs")
    finally:
        await producer.close()
        stop_event.set()
        worker_thread.join(timeout=30)
        await asyncio.wait_for(handler_task, timeout=15)

    print("SMOKE OK (mcp++): producer -> /mcp+p2p/1.0.0 -> queue -> "
          f"hooked worker (muse model) -> {len(task_ids)}/{len(task_ids)} "
          "correct results")


def main():
    asyncio.run(amain())


def test_smoke_producer_consumer_mcp_plus_plus():
    main()


if __name__ == "__main__":
    main()
