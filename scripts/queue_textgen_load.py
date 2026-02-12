#!/usr/bin/env python3
"""Submit lots of text-generation tasks to one or more TaskQueue P2P peers.

Default task_type is `text-generation` and default model is `gpt2`.

Examples:
  # Submit 200 GPT-2 generations to a single peer (do not wait)
  python scripts/queue_textgen_load.py --announce-file /var/cache/ipfs-accelerate/task_p2p_announce.json --count 200

  # Submit and wait for completion (bounded concurrency)
  python scripts/queue_textgen_load.py --announce-file /var/cache/ipfs-accelerate/task_p2p_announce.json --count 50 --wait

  # Round-robin across multiple peers
  python scripts/queue_textgen_load.py --announce-file a.json --announce-file b.json --count 200 --concurrency 20 --wait

Output is a single JSON object on stdout.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple


def _peer_id_from_multiaddr(multiaddr: str) -> str:
    try:
        return str(multiaddr).rsplit("/p2p/", 1)[-1].strip()
    except Exception:
        return ""


def _load_announce(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.loads(handle.read())
    return data if isinstance(data, dict) else {}


def _print_json(obj: Any) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False))
    sys.stdout.write("\n")
    try:
        sys.stdout.flush()
    except Exception:
        pass


def _write_json_file(path: str, obj: Any) -> None:
    # Atomic write so callers never observe a partial/empty file.
    # (Especially useful when runs are interrupted.)
    path = str(path)
    if not path or path == "-":
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    tmp_path = f"{path}.tmp"
    data = json.dumps(obj, ensure_ascii=False)
    with open(tmp_path, "w", encoding="utf-8") as handle:
        handle.write(data)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def _build_remote_targets(args: argparse.Namespace) -> List[Tuple[str, str]]:
    targets: list[Tuple[str, str]] = []
    for ap in args.announce_file or []:
        info = _load_announce(str(ap))
        ma = str(info.get("multiaddr") or "").strip()
        pid = str(info.get("peer_id") or "").strip()
        if not ma and not pid:
            raise SystemExit(f"announce file missing peer_id/multiaddr: {ap}")
        targets.append((pid, ma))

    for multiaddr in args.multiaddr or []:
        peer_ids = getattr(args, "peer_id", [])
        if isinstance(peer_ids, str):
            peer_ids = [peer_ids]
        peer_ids = [str(x).strip() for x in (peer_ids or []) if str(x).strip()]
        if len(peer_ids) > 1:
            raise SystemExit("--multiaddr supports at most one --peer-id hint")
        pid_hint = str(peer_ids[0]).strip() if peer_ids else ""

        ma_text = str(multiaddr).strip()
        pid_from_ma = _peer_id_from_multiaddr(ma_text)
        if pid_hint and pid_from_ma and pid_hint != pid_from_ma:
            raise SystemExit("--peer-id hint does not match --multiaddr /p2p/<peerid>")
        targets.append(((pid_from_ma or pid_hint), ma_text))

    if not targets:
        # Allow peer-id-only discovery (e.g., via mDNS/DHT/rendezvous) when
        # caller sets shared discovery env.
        peer_ids = getattr(args, "peer_id", [])
        if isinstance(peer_ids, str):
            peer_ids = [peer_ids]
        peer_ids = [str(x).strip() for x in (peer_ids or []) if str(x).strip()]
        if not peer_ids:
            raise SystemExit("provide at least one --announce-file or --multiaddr (or set --peer-id for discovery)")
        for pid in peer_ids:
            targets.append((pid, ""))
    return targets


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
        if "result" in value:
            return _extract_text(value.get("result"))
    return str(value)


def _exc_to_dict(exc: BaseException, *, max_depth: int = 3, max_children: int = 6) -> Dict[str, Any]:
    def _one(e: BaseException, depth: int) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "type": type(e).__name__,
            "message": str(e),
        }

        # Keep this short; we mainly want the leaf error plus a hint
        # whether it was a group.
        try:
            d["repr"] = repr(e)
        except Exception:
            pass

        if depth <= 0:
            return d

        if isinstance(e, BaseExceptionGroup):
            children: list[Dict[str, Any]] = []
            for child in list(getattr(e, "exceptions", []) or [])[:max_children]:
                if isinstance(child, BaseException):
                    children.append(_one(child, depth - 1))
            d["children"] = children

        cause = getattr(e, "__cause__", None)
        if isinstance(cause, BaseException):
            d["cause"] = _one(cause, depth - 1)

        ctx = getattr(e, "__context__", None)
        if isinstance(ctx, BaseException) and ctx is not cause:
            d["context"] = _one(ctx, depth - 1)

        return d

    try:
        return _one(exc, max_depth)
    except Exception:
        return {"type": type(exc).__name__, "message": str(exc)}


def _exc_one_line(exc: BaseException) -> str:
    # Prefer a stable, compact message.
    try:
        return f"{type(exc).__name__}: {exc}"
    except Exception:
        return type(exc).__name__


async def _main_async(args: argparse.Namespace) -> Dict[str, Any]:
    if args.count <= 0:
        raise SystemExit("--count must be > 0")
    concurrency = max(1, int(args.concurrency or 1))

    # Keep stdout JSON-only.
    json_stdout = sys.stdout
    sys.stdout = sys.stderr

    try:
        import anyio

        with contextlib.redirect_stdout(sys.stderr):
            from ipfs_accelerate_py.p2p_tasks.client import RemoteQueue, submit_task, wait_task

        targets = _build_remote_targets(args)
        remotes = [RemoteQueue(peer_id=pid, multiaddr=ma) for (pid, ma) in targets]

        prompt = str(args.prompt or "Hello")
        task_type = str(args.task_type or "text-generation")
        model_name = str(args.model or "gpt2")

        payload_base: Dict[str, Any] = {
            "prompt": prompt,
            "max_new_tokens": int(args.max_new_tokens),
            "temperature": float(args.temperature),
        }
        if args.endpoint:
            payload_base["endpoint"] = str(args.endpoint)
        if args.endpoint_type:
            payload_base["endpoint_type"] = str(args.endpoint_type)

        submitted: list[Dict[str, Any]] = []
        submit_failures: list[Dict[str, Any]] = []
        submit_lock = anyio.Lock()
        start = time.time()

        sem = anyio.Semaphore(concurrency)

        async def _submit_one(i: int) -> None:
            target_index = i % len(remotes)
            remote = remotes[target_index]
            payload = dict(payload_base)
            if args.suffix_index:
                payload["prompt"] = f"{prompt} [{i}]"
            async with sem:
                tid: Optional[str] = None
                submit_error: Optional[BaseException] = None

                retries = max(0, int(getattr(args, "submit_retries", 0) or 0))
                retry_sleep_s = float(getattr(args, "submit_retry_sleep_s", 0.35) or 0.35)

                for attempt in range(retries + 1):
                    try:
                        tid = await submit_task(
                            remote=remote,
                            task_type=task_type,
                            model_name=model_name,
                            payload=payload,
                        )
                        submit_error = None
                        break
                    except BaseException as e:
                        if isinstance(e, (KeyboardInterrupt, SystemExit)):
                            raise
                        submit_error = e
                        if attempt < retries:
                            await anyio.sleep(max(0.0, retry_sleep_s * (1.5**attempt)))

            async with submit_lock:
                base_item = {
                    "task_id": str(tid or ""),
                    "peer_id": str(getattr(remote, "peer_id", "") or ""),
                    "multiaddr": str(getattr(remote, "multiaddr", "") or ""),
                    "i": int(i),
                    "target_index": int(target_index),
                }
                if tid:
                    base_item["submit_ok"] = True
                    submitted.append(base_item)
                else:
                    base_item["submit_ok"] = False
                    if submit_error is not None:
                        base_item["submit_error"] = _exc_to_dict(submit_error)
                        base_item["submit_error_one_line"] = _exc_one_line(submit_error)
                    submit_failures.append(base_item)

        async with anyio.create_task_group() as tg:
            for i in range(int(args.count)):
                tg.start_soon(_submit_one, i)

        submit_elapsed_s = float(time.time() - start)

        out: Dict[str, Any] = {
            "ok": True,
            "task_type": task_type,
            "model": model_name,
            "count": int(args.count),
            "concurrency": int(concurrency),
            "targets": [{"peer_id": pid, "multiaddr": ma} for (pid, ma) in targets],
            "submitted": submitted,
            "submit_failed": submit_failures,
            "submit_ok_count": int(len(submitted)),
            "submit_failed_count": int(len(submit_failures)),
            "submit_elapsed_s": submit_elapsed_s,
            "wait": bool(args.wait),
        }

        if not args.wait:
            return out

        # Wait for completion (best-effort) with bounded concurrency.
        wait_sem = anyio.Semaphore(concurrency)
        results_lock = anyio.Lock()
        completed = 0
        failed = 0
        timed_out = 0
        latencies_s: list[float] = []
        outputs: list[Dict[str, Any]] = []

        async def _wait_one(item: Dict[str, Any]) -> None:
            nonlocal completed, failed, timed_out
            peer_id = str(item.get("peer_id") or "")
            task_id = str(item.get("task_id") or "")
            target_index = int(item.get("target_index") or 0)
            if target_index < 0 or target_index >= len(remotes):
                target_index = 0
            remote = remotes[target_index]
            t0 = time.time()
            async with wait_sem:
                try:
                    task = await wait_task(remote=remote, task_id=task_id, timeout_s=float(args.timeout_s))
                except Exception as e:
                    task = {
                        "status": "failed",
                        "error": f"wait_exception: {type(e).__name__}: {e}",
                    }
            dt = float(time.time() - t0)
            async with results_lock:
                if task is None:
                    timed_out += 1
                else:
                    st = str(task.get("status") or "")
                    if st == "completed":
                        completed += 1
                    else:
                        failed += 1
                if bool(args.collect_results):
                    if task is None:
                        outputs.append(
                            {
                                "task_id": task_id,
                                "peer_id": peer_id,
                                "multiaddr": str(getattr(remote, "multiaddr", "") or ""),
                                "status": "timed_out",
                                "text": None,
                            }
                        )
                    else:
                        result = task.get("result")
                        outputs.append(
                            {
                                "task_id": task_id,
                                "peer_id": peer_id,
                                "multiaddr": str(getattr(remote, "multiaddr", "") or ""),
                                "status": str(task.get("status") or ""),
                                "text": _extract_text(result),
                                "result": result,
                                "error": task.get("error"),
                            }
                        )
                latencies_s.append(dt)

        wait_start = time.time()
        async with anyio.create_task_group() as tg:
            for item in list(submitted):
                tg.start_soon(_wait_one, item)
        wait_elapsed_s = float(time.time() - wait_start)

        lat_sorted = sorted([x for x in latencies_s if isinstance(x, (int, float))])
        def _p(pct: float) -> Optional[float]:
            if not lat_sorted:
                return None
            k = int(round((pct / 100.0) * (len(lat_sorted) - 1)))
            k = max(0, min(k, len(lat_sorted) - 1))
            return float(lat_sorted[k])

        out.update(
            {
                "wait_elapsed_s": wait_elapsed_s,
                "completed": int(completed),
                "failed": int(failed),
                "timed_out": int(timed_out),
                "latency_s": {
                    "min": float(lat_sorted[0]) if lat_sorted else None,
                    "p50": _p(50.0),
                    "p90": _p(90.0),
                    "p99": _p(99.0),
                    "max": float(lat_sorted[-1]) if lat_sorted else None,
                },
            }
        )
        if bool(args.collect_results):
            out["outputs"] = outputs
        return out
    finally:
        sys.stdout = json_stdout


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Queue lots of text-generation tasks on P2P peers")
    parser.add_argument(
        "--announce-file",
        action="append",
        default=[],
        help="Path to peer announce JSON {peer_id, multiaddr}. Repeatable.",
    )
    parser.add_argument(
        "--multiaddr",
        action="append",
        default=[],
        help="Remote multiaddr (/ip4/.../tcp/.../p2p/...). Repeatable.",
    )
    parser.add_argument(
        "--peer-id",
        action="append",
        default=[],
        help="Peer-id hint/target. Repeatable. If provided without --announce-file/--multiaddr, peers are dialed via discovery (e.g., mDNS).",
    )

    parser.add_argument("--task-type", default="text-generation")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--suffix-index", action="store_true", help="Append [i] to prompt")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--endpoint", default="")
    parser.add_argument("--endpoint-type", default="")

    parser.add_argument("--count", type=int, required=True, help="Total number of tasks to submit")
    parser.add_argument("--concurrency", type=int, default=10, help="Submit/wait concurrency")
    parser.add_argument(
        "--submit-retries",
        type=int,
        default=0,
        help="Retries per submit on transient libp2p failures (default: 0).",
    )
    parser.add_argument(
        "--submit-retry-sleep-s",
        type=float,
        default=0.35,
        help="Base sleep between submit retries (default: 0.35s).",
    )
    parser.add_argument("--wait", action="store_true", help="Wait for completion")
    parser.add_argument("--timeout-s", type=float, default=120.0, help="Per-task wait timeout")
    parser.add_argument(
        "--collect-results",
        action="store_true",
        help="Include per-task outputs in the final JSON (can be large).",
    )
    parser.add_argument(
        "--output",
        default="-",
        help="Write the final JSON report to this file (atomic). Use '-' to disable file output.",
    )

    args = parser.parse_args(argv)

    # Dependency check.
    try:
        import importlib.util

        if importlib.util.find_spec("libp2p") is None:
            _print_json(
                {
                    "ok": False,
                    "error": "optional dependency 'libp2p' is not installed",
                    "hint": "pip install -e '.[libp2p]'",
                }
            )
            return 2
    except Exception:
        pass

    try:
        import anyio

        result = anyio.run(_main_async, args, backend="trio")
        _write_json_file(str(getattr(args, "output", "-") or "-"), result)
        _print_json(result)
        return 0
    except SystemExit:
        raise
    except Exception as exc:
        err_obj = {"ok": False, "error": str(exc)}
        _write_json_file(str(getattr(args, "output", "-") or "-"), err_obj)
        _print_json(err_obj)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
