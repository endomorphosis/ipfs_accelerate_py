"""Worker loop for accelerate task delegation.

This is intentionally minimal and only supports `text-generation` tasks for now.

It is designed to be run inside long-lived services (e.g. systemd) and can
optionally start the libp2p TaskQueue RPC service in-process.
"""

from __future__ import annotations

import argparse
import threading
import time
from typing import Any, Dict, Optional

import importlib
from .task_queue import TaskQueue


def _extract_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("text", "generated_text", "output"):
            v = value.get(key)
            if isinstance(v, str) and v:
                return v
        if "infer" in value:
            return _extract_text(value.get("infer"))
    return str(value)


def _run_text_generation(task: Dict[str, Any], *, accelerate_instance: object | None = None) -> Dict[str, Any]:
    model_name = str(task.get("model_name") or "")
    payload = task.get("payload") or {}
    prompt = payload.get("prompt") if isinstance(payload, dict) else payload

    # Prefer dispatching through the main ipfs_accelerate_py instance if provided.
    # This allows the worker to reuse the same endpoint handlers / queues / model
    # management configuration as the MCP service.
    if accelerate_instance is not None and isinstance(payload, dict):
        try:
            import anyio
            import inspect

            infer = getattr(accelerate_instance, "infer", None)
            if callable(infer):
                endpoint_hint = payload.get("endpoint")
                endpoint_type_hint = payload.get("endpoint_type")

                data = {
                    "prompt": str(prompt or ""),
                    "max_new_tokens": int(payload.get("max_new_tokens") or payload.get("max_tokens") or 128),
                    "temperature": float(payload.get("temperature") or 0.2),
                }

                async def _do_infer() -> Any:
                    result = infer(model_name or None, data, endpoint=endpoint_hint, endpoint_type=endpoint_type_hint)
                    if inspect.isawaitable(result):
                        return await result
                    return result

                accel_result = anyio.run(_do_infer, backend="trio")
                return {"text": _extract_text(accel_result)}
        except Exception:
            # Fall back to router-based generation.
            pass

    # Fallback path: use the accelerate llm_router, which can still integrate
    # with InferenceBackendManager multiplexing when enabled via env vars.
    from ipfs_accelerate_py import llm_router

    text = llm_router.generate_text(
        str(prompt or ""),
        provider=None,
        model_name=model_name or None,
        max_new_tokens=int(payload.get("max_new_tokens") or payload.get("max_tokens") or 128),
        temperature=float(payload.get("temperature") or 0.2),
    )
    return {"text": str(text)}


def run_worker(
    *,
    queue_path: str,
    worker_id: str,
    poll_interval_s: float = 0.5,
    once: bool = False,
    p2p_service: bool = False,
    p2p_listen_port: Optional[int] = None,
    accelerate_instance: object | None = None,
) -> int:
    if p2p_service:
        # Run the libp2p service in a background thread so the worker loop can
        # remain simple and blocking.
        def _run_service() -> None:
            try:
                import anyio

                service_module = importlib.import_module("ipfs_accelerate_py.p2p_tasks.service")
                a_serve_task_queue = getattr(service_module, "serve_task_queue")

                async def _main() -> None:
                    await a_serve_task_queue(
                        queue_path=queue_path,
                        listen_port=p2p_listen_port,
                        accelerate_instance=accelerate_instance,
                    )

                anyio.run(_main, backend="trio")
            except Exception as exc:
                import sys
                import traceback

                print(f"ipfs_accelerate_py worker: failed to start p2p task service: {exc}", file=sys.stderr)
                traceback.print_exc()

        t = threading.Thread(
            target=_run_service,
            name=f"ipfs_accelerate_p2p_task_service[{worker_id}]",
            daemon=True,
        )
        t.start()

    queue = TaskQueue(queue_path)

    while True:
        task = queue.claim_next(worker_id=worker_id, supported_task_types=["text-generation"])
        if task is None:
            if once:
                return 0
            time.sleep(max(0.05, float(poll_interval_s)))
            continue

        try:
            if task.task_type in {"text-generation", "text_generation", "generation"}:
                result = _run_text_generation(
                    {
                        "task_id": task.task_id,
                        "task_type": task.task_type,
                        "model_name": task.model_name,
                        "payload": task.payload,
                    },
                    accelerate_instance=accelerate_instance,
                )
                queue.complete(task_id=task.task_id, status="completed", result=result)
            else:
                queue.complete(task_id=task.task_id, status="failed", error=f"Unsupported task_type: {task.task_type}")
        except Exception as exc:
            queue.complete(task_id=task.task_id, status="failed", error=str(exc))

        if once:
            return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="ipfs_accelerate_py task worker")
    parser.add_argument("--queue", dest="queue_path", required=True, help="Path to task queue DuckDB file")
    parser.add_argument("--worker-id", dest="worker_id", required=True, help="Worker identifier")
    parser.add_argument("--poll-interval-s", dest="poll_interval_s", type=float, default=0.5)
    parser.add_argument("--once", action="store_true", help="Process at most one task")
    parser.add_argument("--p2p-service", action="store_true", help="Also start a local libp2p TaskQueue RPC service")
    parser.add_argument(
        "--p2p-listen-port",
        type=int,
        default=None,
        help="TCP port for libp2p service (default: env or 9710)",
    )

    args = parser.parse_args(argv)
    return run_worker(
        queue_path=args.queue_path,
        worker_id=args.worker_id,
        poll_interval_s=float(args.poll_interval_s),
        once=bool(args.once),
        p2p_service=bool(args.p2p_service),
        p2p_listen_port=args.p2p_listen_port,
    )


if __name__ == "__main__":
    raise SystemExit(main())
