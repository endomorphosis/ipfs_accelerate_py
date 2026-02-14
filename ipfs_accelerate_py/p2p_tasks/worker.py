"""Worker loop for accelerate task delegation.

Designed to be run inside long-lived services (e.g. systemd). It can optionally
start the libp2p TaskQueue RPC service in-process.

Task handlers are intentionally minimal and gated:
- Always enabled: ``text-generation``
- Enabled when an accelerate instance provides ``call_tool``: ``tool.call``
- Opt-in via env: ``shell`` (disabled by default)
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple

import importlib
from .task_queue import TaskQueue


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


_HF_TEXTGEN_LOCK = threading.RLock()
_HF_TEXTGEN_PIPELINE: object | None = None
_HF_TEXTGEN_MODEL_ID: str | None = None


def _hf_textgen(prompt: str, *, model_name: str | None, max_new_tokens: int, temperature: float) -> str:
    """Minimal local text-generation without importing ipfs_accelerate_py core."""

    global _HF_TEXTGEN_PIPELINE, _HF_TEXTGEN_MODEL_ID

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"transformers is required for minimal text-generation: {exc}")

    requested_model = str(model_name or os.environ.get("IPFS_ACCELERATE_PY_LLM_MODEL") or "gpt2").strip() or "gpt2"
    safe_max_new = max(1, min(int(max_new_tokens or 128), 1024))
    temp = float(temperature) if temperature is not None else 0.2

    with _HF_TEXTGEN_LOCK:
        if _HF_TEXTGEN_PIPELINE is None or _HF_TEXTGEN_MODEL_ID != requested_model:
            tokenizer = AutoTokenizer.from_pretrained(requested_model)
            model = AutoModelForCausalLM.from_pretrained(requested_model)
            if (
                getattr(tokenizer, "pad_token_id", None) is None
                and getattr(tokenizer, "eos_token_id", None) is not None
            ):
                tokenizer.pad_token_id = tokenizer.eos_token_id

            _HF_TEXTGEN_PIPELINE = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=-1,
            )
            _HF_TEXTGEN_MODEL_ID = requested_model

        gen = _HF_TEXTGEN_PIPELINE

    # Pipeline calls are not guaranteed thread-safe; guard the call.
    with _HF_TEXTGEN_LOCK:
        out = gen(
            str(prompt or ""),
            max_new_tokens=safe_max_new,
            do_sample=temp > 0,
            temperature=max(temp, 1e-6),
            pad_token_id=getattr(getattr(gen, "tokenizer", None), "pad_token_id", None),
        )

    if isinstance(out, list) and out and isinstance(out[0], dict):
        text = out[0].get("generated_text")
        if isinstance(text, str):
            return text
    return str(out)


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
                # Some implementations return an Exception object instead of
                # raising it. Treat that as an error so we fall back to router-
                # based generation below.
                if isinstance(accel_result, BaseException):
                    raise accel_result
                return {"text": _extract_text(accel_result)}
        except Exception:
            # Fall back to router-based generation.
            pass

    max_new_tokens = int(payload.get("max_new_tokens") or payload.get("max_tokens") or 128)
    temperature = float(payload.get("temperature") or 0.2)

    # Minimal path for deterministic/local-first testing: avoid importing the
    # full ipfs_accelerate_py package (which can enable optional integrations).
    if _truthy(os.environ.get("IPFS_ACCEL_SKIP_CORE")) or _truthy(
        os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_MINIMAL_LLM")
    ):
        text = _hf_textgen(
            str(prompt or ""),
            model_name=model_name or None,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        return {"text": str(text)}

    # Default fallback: use the accelerate llm_router, which can still integrate
    # with InferenceBackendManager multiplexing when enabled via env vars.
    from ipfs_accelerate_py import llm_router

    try:
        text = llm_router.generate_text(
            str(prompt or ""),
            provider=None,
            model_name=model_name or None,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        return {"text": str(text)}
    except Exception:
        # If optional providers (e.g. Codex/Copilot CLIs) are importable but not
        # actually functional on this machine, fall back to a minimal local HF
        # path so LAN workers can still run text-generation workloads.
        text = _hf_textgen(
            str(prompt or ""),
            model_name=model_name or None,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        return {"text": str(text)}


def _run_tool_call(task: Dict[str, Any], *, accelerate_instance: object | None = None) -> Dict[str, Any]:
    payload = task.get("payload") or {}
    if not isinstance(payload, dict):
        raise ValueError("tool.call payload must be a dict")

    tool_name = str(payload.get("tool") or payload.get("tool_name") or payload.get("name") or "").strip()
    if not tool_name:
        raise ValueError("tool.call missing tool name")
    args = payload.get("args") or payload.get("arguments") or payload.get("params") or {}
    if not isinstance(args, dict):
        args = {"value": args}

    if accelerate_instance is None:
        raise RuntimeError("tool.call requires accelerate_instance")
    fn = getattr(accelerate_instance, "call_tool", None)
    if not callable(fn):
        raise RuntimeError("accelerate_instance does not implement call_tool")

    try:
        import anyio
        import inspect

        async def _do() -> Any:
            result = fn(tool_name, args)
            if inspect.isawaitable(result):
                return await result
            return result

        out = anyio.run(_do, backend="trio")
    except Exception as exc:
        raise RuntimeError(f"tool.call failed: {exc}")
    return {"tool": tool_name, "result": out}


def _run_shell(task: Dict[str, Any]) -> Dict[str, Any]:
    allow = str(os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_SHELL") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not allow:
        raise RuntimeError("shell task_type disabled (set IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_SHELL=1)")

    if not _docker_tasks_enabled():
        raise RuntimeError("shell requires docker (set IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_DOCKER=1)")

    payload = task.get("payload") or {}
    if not isinstance(payload, dict):
        raise ValueError("shell payload must be a dict")

    argv = payload.get("argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(x, str) and x for x in argv):
        raise ValueError("shell payload.argv must be a non-empty list[str]")

    timeout_s = payload.get("timeout_s")
    try:
        timeout_v = float(timeout_s) if timeout_s is not None else None
    except Exception:
        timeout_v = None

    image = (
        str(os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_SHELL_IMAGE") or "ubuntu:22.04").strip()
        or "ubuntu:22.04"
    )
    cmd_str = shlex.join([str(x) for x in argv])
    command = ["/bin/sh", "-lc", cmd_str]

    from ipfs_accelerate_py.docker_executor import execute_docker_hub_container

    res = execute_docker_hub_container(
        image=image,
        command=command,
        timeout=int(timeout_v) if timeout_v is not None else 300,
        network_mode="none",
        no_new_privileges=True,
        stream_output=bool(payload.get("stream_output")),
    )

    return {
        "returncode": int(getattr(res, "exit_code", -1)),
        "stdout": str(getattr(res, "stdout", "") or ""),
        "stderr": str(getattr(res, "stderr", "") or ""),
        "success": bool(getattr(res, "success", False)),
        "exit_code": int(getattr(res, "exit_code", -1)),
    }


def _docker_tasks_enabled() -> bool:
    raw = os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_DOCKER")
    if raw is not None:
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}

    # Auto-enable if daemon is accessible.
    return _docker_daemon_available()


def _docker_daemon_available() -> bool:
    try:
        from ipfs_accelerate_py.docker_executor import docker_daemon_available

        return bool(docker_daemon_available())
    except Exception:
        return False


def _maybe_split_argv(value: Any) -> Optional[list[str]]:
    if value is None:
        return None
    if isinstance(value, list) and all(isinstance(x, str) for x in value):
        return [x for x in value if x]
    if isinstance(value, str):
        parts = [p for p in value.split() if p]
        return parts if parts else None
    return None


def _build_docker_run_cmd(*, payload: Dict[str, Any]) -> Tuple[list[str], float | None]:
    image = str(payload.get("image") or "").strip()
    if not image:
        raise ValueError("docker task missing payload.image")

    cmd: list[str] = ["docker", "run", "--rm"]

    # GPU support (optional): docker run --gpus <value>
    # Common values: "all", "device=0".
    gpus = payload.get("gpus")
    if gpus is None:
        gpus = payload.get("gpu")
    if gpus is True:
        gpus = "all"
    if isinstance(gpus, str) and gpus.strip():
        cmd.extend(["--gpus", str(gpus).strip()])

    memory_limit = payload.get("memory_limit")
    if memory_limit is not None:
        cmd.extend(["--memory", str(memory_limit)])
    cpu_limit = payload.get("cpu_limit")
    if cpu_limit is not None:
        try:
            cmd.extend(["--cpus", str(float(cpu_limit))])
        except Exception:
            pass

    if bool(payload.get("read_only")):
        cmd.append("--read-only")
    if payload.get("no_new_privileges") is None or bool(payload.get("no_new_privileges")):
        cmd.append("--security-opt=no-new-privileges")
    user = payload.get("user")
    if user is not None and str(user).strip():
        cmd.extend(["--user", str(user).strip()])

    network_mode = payload.get("network_mode")
    if network_mode is not None and str(network_mode).strip():
        cmd.extend(["--network", str(network_mode).strip()])
    else:
        cmd.extend(["--network", "none"])

    env = payload.get("environment") or payload.get("env") or {}
    if isinstance(env, dict):
        for k, v in env.items():
            cmd.extend(["-e", f"{str(k)}={str(v)}"])

    vols = payload.get("volumes") or {}
    if isinstance(vols, dict):
        for host_path, container_path in vols.items():
            cmd.extend(["-v", f"{str(host_path)}:{str(container_path)}"])

    working_dir = payload.get("working_dir")
    if working_dir is not None and str(working_dir).strip():
        cmd.extend(["-w", str(working_dir).strip()])

    entrypoint = _maybe_split_argv(payload.get("entrypoint"))
    command = _maybe_split_argv(payload.get("command") or payload.get("cmd"))
    if entrypoint:
        cmd.extend(["--entrypoint", entrypoint[0]])

    cmd.append(image)

    if entrypoint and len(entrypoint) > 1:
        cmd.extend(entrypoint[1:])
        if command:
            cmd.extend(command)
    elif command:
        cmd.extend(command)

    timeout_s = payload.get("timeout")
    timeout_v: float | None
    if timeout_s is None:
        timeout_v = None
    else:
        try:
            timeout_v = float(timeout_s)
        except Exception:
            timeout_v = None
    return cmd, timeout_v


def _stream_subprocess(
    *,
    argv: list[str],
    task_id: str,
    queue: TaskQueue,
    timeout_s: float | None,
    heartbeat_interval_s: float = 0.5,
    flush_interval_s: float = 0.25,
) -> Dict[str, Any]:
    start = time.time()
    deadline = (start + timeout_s) if timeout_s and timeout_s > 0 else None

    proc = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    stdout_buf: list[str] = []
    stderr_buf: list[str] = []

    last_flush = 0.0
    last_heartbeat = 0.0

    def _safe_update(**kwargs: Any) -> None:
        try:
            queue.update(**kwargs)
        except Exception:
            pass

    def _pump(stream, *, stream_name: str) -> None:
        if stream is None:
            return
        for line in iter(stream.readline, ""):
            if stream_name == "stdout":
                stdout_buf.append(line)
            else:
                stderr_buf.append(line)
        try:
            stream.close()
        except Exception:
            pass

    t_out = threading.Thread(target=_pump, args=(proc.stdout,), kwargs={"stream_name": "stdout"}, daemon=True)
    t_err = threading.Thread(target=_pump, args=(proc.stderr,), kwargs={"stream_name": "stderr"}, daemon=True)
    t_out.start()
    t_err.start()

    try:
        while True:
            now = time.time()

            if deadline is not None and now >= deadline:
                try:
                    proc.kill()
                except Exception:
                    pass
                _safe_update(
                    task_id=task_id,
                    status="running",
                    result_patch={"progress": {"phase": "timeout", "ts": now}},
                    append_log=f"[worker] timeout after {timeout_s}s; killed process",
                    log_stream="stderr",
                )
                break

            rc = proc.poll()

            if now - last_heartbeat >= heartbeat_interval_s:
                _safe_update(
                    task_id=task_id,
                    status="running",
                    result_patch={"progress": {"phase": "running", "heartbeat_ts": now}},
                )
                last_heartbeat = now

            if now - last_flush >= flush_interval_s:
                # Flush buffered stdout/stderr lines to the queue.
                while stdout_buf:
                    line = stdout_buf.pop(0)
                    _safe_update(
                        task_id=task_id,
                        status="running",
                        append_log=line.rstrip("\n"),
                        log_stream="stdout",
                    )
                while stderr_buf:
                    line = stderr_buf.pop(0)
                    _safe_update(
                        task_id=task_id,
                        status="running",
                        append_log=line.rstrip("\n"),
                        log_stream="stderr",
                    )
                last_flush = now

            if rc is not None:
                break

            time.sleep(0.05)

        # Final flush
        while stdout_buf:
            line = stdout_buf.pop(0)
            _safe_update(
                task_id=task_id,
                status="running",
                append_log=line.rstrip("\n"),
                log_stream="stdout",
            )
        while stderr_buf:
            line = stderr_buf.pop(0)
            _safe_update(
                task_id=task_id,
                status="running",
                append_log=line.rstrip("\n"),
                log_stream="stderr",
            )
    finally:
        try:
            t_out.join(timeout=0.5)
            t_err.join(timeout=0.5)
        except Exception:
            pass

    end = time.time()
    rc_final = proc.returncode if proc.returncode is not None else -1
    return {
        "success": bool(rc_final == 0),
        "exit_code": int(rc_final),
        "execution_time": float(end - start),
    }


def _run_docker_hub(task: Dict[str, Any]) -> Dict[str, Any]:
    raise RuntimeError("internal: _run_docker_hub is replaced by a queue-aware handler")


def _run_docker_github(task: Dict[str, Any]) -> Dict[str, Any]:
    raise RuntimeError("internal: _run_docker_github is replaced by a queue-aware handler")


def _supported_task_types_from_env(default: list[str]) -> list[str]:
    raw = (
        os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_TASK_TYPES")
        or os.environ.get("IPFS_DATASETS_PY_TASK_WORKER_TASK_TYPES")
        or ""
    )
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    if parts:
        return parts

    base = list(default)
    if _docker_tasks_enabled():
        base.extend(
            [
                "docker.execute",
                "docker.run",
                "docker.hub",
                "docker.execute_docker_container",
                "docker.github",
                "docker.github_repo",
                "docker.build_and_execute_github_repo",
            ]
        )
    # Shell tasks are docker-backed; only enable when explicitly allowed.
    enable_shell_raw = os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_SHELL") or ""
    enable_shell = str(enable_shell_raw).strip().lower() in {"1", "true", "yes", "on"}
    if enable_shell and _docker_tasks_enabled():
        base.append("shell")
    return base


def _task_types_overridden_via_env() -> bool:
    raw = (
        os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_TASK_TYPES")
        or os.environ.get("IPFS_DATASETS_PY_TASK_WORKER_TASK_TYPES")
        or ""
    )
    return bool([p.strip() for p in str(raw).split(",") if p.strip()])


def _accelerate_supports_tool_call(accelerate_instance: object | None) -> bool:
    fn = getattr(accelerate_instance, "call_tool", None)
    return bool(callable(fn))


def _compute_supported_task_types(
    *,
    supported_task_types: Optional[list[str]],
    accelerate_instance: object | None,
) -> list[str]:
    # If explicitly provided, respect it.
    if isinstance(supported_task_types, list) and supported_task_types:
        out = [str(x).strip() for x in supported_task_types if str(x).strip()]
        return out

    # Default: include all handler aliases we can run locally.
    base_defaults = ["text-generation", "text_generation", "generation"]
    out = _supported_task_types_from_env(base_defaults)

    # Add tool.call only when we can actually execute it, and only when the
    # task types weren't explicitly overridden via env (where the user likely
    # wants an exact allowlist).
    if (not _task_types_overridden_via_env()) and _accelerate_supports_tool_call(accelerate_instance):
        out.extend(["tool.call", "tool"])

    # Deduplicate while keeping order.
    seen: set[str] = set()
    deduped: list[str] = []
    for t in out:
        tt = str(t or "").strip()
        if not tt or tt in seen:
            continue
        seen.add(tt)
        deduped.append(tt)
    return deduped


def _worker_mesh_enabled() -> bool:
    raw = os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_MESH") or os.environ.get("IPFS_DATASETS_PY_TASK_WORKER_MESH")
    if raw is None:
        return False
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _worker_mesh_refresh_s() -> float:
    raw = os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_MESH_REFRESH_S")
    try:
        return max(0.5, float(raw)) if raw is not None else 5.0
    except Exception:
        return 5.0


def _worker_mesh_claim_interval_s() -> float:
    raw = os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_MESH_CLAIM_INTERVAL_S")
    try:
        return max(0.1, float(raw)) if raw is not None else 0.5
    except Exception:
        return 0.5


def _worker_mesh_max_peers() -> int:
    raw = os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_MESH_MAX_PEERS")
    try:
        return max(1, min(100, int(raw))) if raw is not None else 10
    except Exception:
        return 10


def _expected_session_tag() -> str:
    return str(os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_SESSION") or "").strip()


def _read_local_announce_peer_id() -> str:
    """Best-effort local peer id from announce JSON.

    Used only for mesh bookkeeping/logging; worker mesh scheduling uses a stable
    `peer_id` hint derived from worker_id.
    """

    path = str(os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_ANNOUNCE_FILE") or "").strip()
    if not path or path.lower() in {"0", "false", "no", "off"}:
        return ""
    try:
        if not os.path.exists(path):
            return ""
        data = json.loads(open(path, "r", encoding="utf-8").read() or "{}")
        if isinstance(data, dict):
            return str(data.get("peer_id") or "").strip()
    except Exception:
        return ""
    return ""


def _mesh_safe_task_types(supported: list[str]) -> list[str]:
    """Return supported task types for mesh claiming.

    Historically, mesh mode used an allowlist. We now allow whatever task types
    this worker is configured to support.
    """

    out: list[str] = []
    for t in list(supported or []):
        tt = str(t or "").strip()
        if tt:
            out.append(tt)

    if not out:
        out = ["text-generation"]
    return out


def _start_mesh_discovery_thread(
    *,
    stop: threading.Event,
    worker_id: str,
    peers_out: dict[str, object],
    peers_lock: threading.RLock,
    max_peers: int,
    refresh_s: float,
) -> threading.Thread:
    """Background mDNS discovery thread.

    Writes a list[RemoteQueue] into peers_out["peers"].
    """

    expected_session = _expected_session_tag()

    def _loop() -> None:
        # Import lazily: libp2p is optional in some environments.
        try:
            from ipfs_accelerate_py.p2p_tasks.client import (
                RemoteQueue,
                discover_peers_via_mdns_sync,
                request_status_sync,
            )
        except Exception:
            return

        # If our own peer id is known, help avoid selecting ourselves in
        # environments where exclude_self can't resolve it.
        local_peer_id = _read_local_announce_peer_id()

        while not stop.is_set():
            try:
                discovered = discover_peers_via_mdns_sync(timeout_s=1.0, limit=int(max_peers), exclude_self=True)
            except Exception:
                discovered = []

            keep: list[RemoteQueue] = []
            for rq in list(discovered or []):
                try:
                    pid = str(getattr(rq, "peer_id", "") or "").strip()
                    ma = str(getattr(rq, "multiaddr", "") or "").strip()
                except Exception:
                    continue
                if not pid or not ma:
                    continue
                if local_peer_id and pid == local_peer_id:
                    continue

                if expected_session:
                    try:
                        resp = request_status_sync(remote=rq, timeout_s=3.0, detail=False)
                        if not (isinstance(resp, dict) and resp.get("ok")):
                            continue
                        if str(resp.get("session") or "").strip() != expected_session:
                            continue
                    except Exception:
                        continue

                keep.append(rq)

            with peers_lock:
                peers_out["peers"] = keep

            stop.wait(float(refresh_s))

    t = threading.Thread(target=_loop, name=f"task_worker_mesh_mdns[{worker_id}]", daemon=True)
    t.start()
    return t


def _worker_mesh_static_peers() -> list[object]:
    """Return RemoteQueue targets to use for mesh claims, if configured.

    Env:
      IPFS_ACCELERATE_PY_TASK_WORKER_MESH_PEERS

    Format:
      - Comma-separated list of multiaddrs including /p2p/<peer_id>
        Example: /ip4/192.168.0.54/tcp/9101/p2p/12D3KooW...

    When provided and valid, the worker will skip mDNS discovery and only
    claim tasks from these peers.
    """

    raw = str(os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_MESH_PEERS") or "").strip()
    if not raw:
        return []

    # Lazy import: this is only used in mesh mode.
    try:
        from ipfs_accelerate_py.p2p_tasks.client import RemoteQueue
    except Exception:
        return []

    out: list[object] = []
    for part in [p.strip() for p in raw.split(",")]:
        if not part:
            continue

        multiaddr = str(part).strip()
        peer_id = ""
        if "/p2p/" in multiaddr:
            try:
                peer_id = multiaddr.split("/p2p/", 1)[1].split("/", 1)[0].strip()
            except Exception:
                peer_id = ""
        if not peer_id:
            # Require peer id to avoid ambiguous dialing.
            continue

        try:
            out.append(RemoteQueue(peer_id=str(peer_id), multiaddr=str(multiaddr)))
        except Exception:
            continue

    return out


def run_worker(
    *,
    queue_path: str,
    worker_id: str,
    poll_interval_s: float = 0.5,
    once: bool = False,
    p2p_service: bool = False,
    p2p_listen_port: Optional[int] = None,
    accelerate_instance: object | None = None,
    supported_task_types: Optional[list[str]] = None,
    mesh: Optional[bool] = None,
    mesh_refresh_s: Optional[float] = None,
    mesh_claim_interval_s: Optional[float] = None,
    mesh_max_peers: Optional[int] = None,
    stop_event: threading.Event | None = None,
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

    handlers: dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}

    def _wrap(fn: Callable[..., Dict[str, Any]]) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        def _runner(task_dict: Dict[str, Any]) -> Dict[str, Any]:
            return fn(task_dict, accelerate_instance=accelerate_instance)  # type: ignore[misc]

        return _runner

    handlers["text-generation"] = _wrap(_run_text_generation)
    handlers["text_generation"] = _wrap(_run_text_generation)
    handlers["generation"] = _wrap(_run_text_generation)
    handlers["tool.call"] = _wrap(_run_tool_call)
    handlers["tool"] = _wrap(_run_tool_call)

    def _shell_handler(task_dict: Dict[str, Any]) -> Dict[str, Any]:
        payload = task_dict.get("payload") or {}
        if not isinstance(payload, dict):
            raise ValueError("shell payload must be a dict")

        enable_shell_raw = os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_SHELL") or ""
        allow = str(enable_shell_raw).strip().lower() in {"1", "true", "yes", "on"}
        if not allow:
            raise RuntimeError("shell task_type disabled (set IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_SHELL=1)")

        if not _docker_tasks_enabled():
            raise RuntimeError("shell requires docker (set IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_DOCKER=1)")

        task_id = str(task_dict.get("task_id") or "")
        argv = payload.get("argv")
        if not isinstance(argv, list) or not argv or not all(isinstance(x, str) and x for x in argv):
            raise ValueError("shell payload.argv must be a non-empty list[str]")

        timeout_s = payload.get("timeout_s")
        try:
            timeout_v = float(timeout_s) if timeout_s is not None else None
        except Exception:
            timeout_v = None

        image = (
            str(os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_SHELL_IMAGE") or "ubuntu:22.04").strip()
            or "ubuntu:22.04"
        )
        cmd_str = shlex.join([str(x) for x in argv])
        command = ["/bin/sh", "-lc", cmd_str]

        try:
            queue.update(
                task_id=task_id,
                status="running",
                result_patch={"progress": {"phase": "starting", "ts": time.time(), "image": image}},
            )
        except Exception:
            pass

        from ipfs_accelerate_py.docker_executor import execute_docker_hub_container

        res = execute_docker_hub_container(
            image=image,
            command=command,
            timeout=int(timeout_v) if timeout_v is not None else 300,
            network_mode="none",
            no_new_privileges=True,
            stream_output=bool(payload.get("stream_output")),
        )

        stdout = str(getattr(res, "stdout", "") or "")
        stderr = str(getattr(res, "stderr", "") or "")
        for line in stdout.splitlines():
            try:
                queue.update(task_id=task_id, status="running", append_log=line, log_stream="stdout")
            except Exception:
                pass
        for line in stderr.splitlines():
            try:
                queue.update(task_id=task_id, status="running", append_log=line, log_stream="stderr")
            except Exception:
                pass

        return {
            "returncode": int(getattr(res, "exit_code", -1)),
            "stdout": stdout,
            "stderr": stderr,
            "success": bool(getattr(res, "success", False)),
            "exit_code": int(getattr(res, "exit_code", -1)),
        }

    handlers["shell"] = _shell_handler

    def _docker_hub_handler(task_dict: Dict[str, Any]) -> Dict[str, Any]:
        if not _docker_tasks_enabled():
            raise RuntimeError(
                "docker task_type disabled (set IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_DOCKER=1)"
            )

        payload = task_dict.get("payload") or {}
        if not isinstance(payload, dict):
            raise ValueError("docker payload must be a dict")
        task_id = str(task_dict.get("task_id") or "")
        try:
            queue.update(
                task_id=task_id,
                status="running",
                result_patch={"progress": {"phase": "starting", "ts": time.time()}},
            )
        except Exception:
            pass

        image = str(payload.get("image") or "").strip()
        if not image:
            raise ValueError("docker task missing payload.image")

        # Optional true streaming mode (uses docker CLI directly). Keep disabled
        # by default to preserve testability (tests monkeypatch docker_executor).
        stream_mode = bool(payload.get("stream_output")) or str(
            os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_DOCKER_STREAM") or ""
        ).strip().lower() in {"1", "true", "yes", "on"}

        if stream_mode:
            argv, timeout_v = _build_docker_run_cmd(payload=payload)
            try:
                queue.update(
                    task_id=task_id,
                    status="running",
                    append_log=f"[worker] exec: {' '.join(argv)}",
                    log_stream="stderr",
                )
            except Exception:
                pass
            result = _stream_subprocess(argv=argv, task_id=task_id, queue=queue, timeout_s=timeout_v)
            try:
                queue.update(
                    task_id=task_id,
                    status="running",
                    result_patch={"progress": {"phase": "exited", "ts": time.time()}},
                )
            except Exception:
                pass
            return result

        from ipfs_accelerate_py.docker_executor import execute_docker_hub_container

        command = _maybe_split_argv(payload.get("command") or payload.get("cmd"))
        entrypoint = _maybe_split_argv(payload.get("entrypoint"))

        environment = payload.get("environment") or payload.get("env") or {}
        if not isinstance(environment, dict):
            environment = {}
        environment = {str(k): str(v) for k, v in environment.items()}

        volumes = payload.get("volumes") or {}
        if not isinstance(volumes, dict):
            volumes = {}
        volumes = {str(k): str(v) for k, v in volumes.items()}

        kwargs: Dict[str, Any] = {}
        # GPU support (optional): passed through to docker_executor.
        if payload.get("gpus") is not None:
            kwargs["gpus"] = "all" if payload.get("gpus") is True else str(payload.get("gpus"))
        elif payload.get("gpu") is not None:
            kwargs["gpus"] = "all" if payload.get("gpu") is True else str(payload.get("gpu"))
        if payload.get("memory_limit") is not None:
            kwargs["memory_limit"] = str(payload.get("memory_limit"))
        if payload.get("cpu_limit") is not None:
            try:
                kwargs["cpu_limit"] = float(payload.get("cpu_limit"))
            except Exception:
                pass
        if payload.get("timeout") is not None:
            try:
                kwargs["timeout"] = int(payload.get("timeout"))
            except Exception:
                pass
        if payload.get("network_mode") is not None:
            kwargs["network_mode"] = str(payload.get("network_mode"))
        if payload.get("working_dir") is not None:
            kwargs["working_dir"] = str(payload.get("working_dir"))
        if payload.get("read_only") is not None:
            kwargs["read_only"] = bool(payload.get("read_only"))
        if payload.get("no_new_privileges") is not None:
            kwargs["no_new_privileges"] = bool(payload.get("no_new_privileges"))
        if payload.get("user") is not None:
            kwargs["user"] = str(payload.get("user"))

        stop = threading.Event()

        def _hb() -> None:
            while not stop.is_set():
                try:
                    queue.update(
                        task_id=task_id,
                        status="running",
                        result_patch={"progress": {"phase": "running", "heartbeat_ts": time.time(), "image": image}},
                    )
                except Exception:
                    pass
                stop.wait(0.5)

        t = threading.Thread(target=_hb, name=f"docker_hb[{task_id}]", daemon=True)
        t.start()
        try:
            res = execute_docker_hub_container(
                image=image,
                command=command,
                entrypoint=entrypoint,
                environment=environment,
                volumes=volumes,
                **kwargs,
            )
        finally:
            stop.set()
            try:
                t.join(timeout=0.2)
            except Exception:
                pass

        stdout = str(getattr(res, "stdout", "") or "")
        stderr = str(getattr(res, "stderr", "") or "")
        for line in stdout.splitlines():
            try:
                queue.update(task_id=task_id, status="running", append_log=line, log_stream="stdout")
            except Exception:
                pass
        for line in stderr.splitlines():
            try:
                queue.update(task_id=task_id, status="running", append_log=line, log_stream="stderr")
            except Exception:
                pass
        if getattr(res, "error_message", None):
            try:
                queue.update(
                    task_id=task_id,
                    status="running",
                    append_log=str(getattr(res, "error_message")),
                    log_stream="stderr",
                )
            except Exception:
                pass
        try:
            queue.update(
                task_id=task_id,
                status="running",
                result_patch={"progress": {"phase": "exited", "ts": time.time(), "image": image}},
            )
        except Exception:
            pass

        return {
            "success": bool(getattr(res, "success", False)),
            "exit_code": int(getattr(res, "exit_code", -1)),
            "stdout": stdout,
            "stderr": stderr,
            "execution_time": float(getattr(res, "execution_time", 0.0) or 0.0),
            "error_message": getattr(res, "error_message", None),
        }

    def _docker_github_handler(task_dict: Dict[str, Any]) -> Dict[str, Any]:
        if not _docker_tasks_enabled():
            raise RuntimeError(
                "docker task_type disabled (set IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_DOCKER=1)"
            )

        payload = task_dict.get("payload") or {}
        if not isinstance(payload, dict):
            raise ValueError("docker.github payload must be a dict")
        task_id = str(task_dict.get("task_id") or "")

        repo_url = str(payload.get("repo_url") or payload.get("repo") or "").strip()
        if not repo_url:
            raise ValueError("docker.github task missing payload.repo_url")

        branch = str(payload.get("branch") or "main")
        dockerfile_path = str(payload.get("dockerfile_path") or payload.get("dockerfile") or "Dockerfile")
        context_path = str(payload.get("context_path") or payload.get("context") or ".")

        command = _maybe_split_argv(payload.get("command") or payload.get("cmd"))
        entrypoint = _maybe_split_argv(payload.get("entrypoint"))

        environment = payload.get("environment") or payload.get("env") or {}
        if not isinstance(environment, dict):
            environment = {}
        environment = {str(k): str(v) for k, v in environment.items()}

        build_args = payload.get("build_args") or {}
        if not isinstance(build_args, dict):
            build_args = {}
        build_args = {str(k): str(v) for k, v in build_args.items()}

        kwargs: Dict[str, Any] = {}
        if payload.get("memory_limit") is not None:
            kwargs["memory_limit"] = str(payload.get("memory_limit"))
        if payload.get("cpu_limit") is not None:
            try:
                kwargs["cpu_limit"] = float(payload.get("cpu_limit"))
            except Exception:
                pass
        if payload.get("timeout") is not None:
            try:
                kwargs["timeout"] = int(payload.get("timeout"))
            except Exception:
                pass
        if payload.get("network_mode") is not None:
            kwargs["network_mode"] = str(payload.get("network_mode"))
        if payload.get("working_dir") is not None:
            kwargs["working_dir"] = str(payload.get("working_dir"))
        if payload.get("volumes") is not None and isinstance(payload.get("volumes"), dict):
            kwargs["volumes"] = {str(k): str(v) for k, v in payload.get("volumes").items()}
        if payload.get("read_only") is not None:
            kwargs["read_only"] = bool(payload.get("read_only"))
        if payload.get("no_new_privileges") is not None:
            kwargs["no_new_privileges"] = bool(payload.get("no_new_privileges"))
        if payload.get("user") is not None:
            kwargs["user"] = str(payload.get("user"))

        try:
            queue.update(
                task_id=task_id,
                status="running",
                result_patch={"progress": {"phase": "building", "ts": time.time(), "repo_url": repo_url}},
            )
        except Exception:
            pass

        from ipfs_accelerate_py.docker_executor import build_and_execute_from_github

        stop = threading.Event()

        def _hb() -> None:
            while not stop.is_set():
                try:
                    queue.update(
                        task_id=task_id,
                        status="running",
                        result_patch={
                            "progress": {
                                "phase": "building_and_running",
                                "heartbeat_ts": time.time(),
                                "repo_url": repo_url,
                                "branch": branch,
                            }
                        },
                    )
                except Exception:
                    pass
                stop.wait(0.5)

        t = threading.Thread(target=_hb, name=f"docker_github_hb[{task_id}]", daemon=True)
        t.start()
        try:
            res = build_and_execute_from_github(
                repo_url=repo_url,
                branch=branch,
                dockerfile_path=dockerfile_path,
                context_path=context_path,
                command=command,
                entrypoint=entrypoint,
                environment=environment,
                build_args=build_args,
                **kwargs,
            )
        finally:
            stop.set()
            try:
                t.join(timeout=0.2)
            except Exception:
                pass

        stdout = str(getattr(res, "stdout", "") or "")
        stderr = str(getattr(res, "stderr", "") or "")
        for line in stdout.splitlines():
            try:
                queue.update(task_id=task_id, status="running", append_log=line, log_stream="stdout")
            except Exception:
                pass
        for line in stderr.splitlines():
            try:
                queue.update(task_id=task_id, status="running", append_log=line, log_stream="stderr")
            except Exception:
                pass
        if getattr(res, "error_message", None):
            try:
                queue.update(
                    task_id=task_id,
                    status="running",
                    append_log=str(getattr(res, "error_message")),
                    log_stream="stderr",
                )
            except Exception:
                pass
        try:
            queue.update(
                task_id=task_id,
                status="running",
                result_patch={"progress": {"phase": "exited", "ts": time.time(), "repo_url": repo_url}},
            )
        except Exception:
            pass

        return {
            "success": bool(getattr(res, "success", False)),
            "exit_code": int(getattr(res, "exit_code", -1)),
            "stdout": stdout,
            "stderr": stderr,
            "execution_time": float(getattr(res, "execution_time", 0.0) or 0.0),
            "error_message": getattr(res, "error_message", None),
        }

    handlers["docker.execute"] = _docker_hub_handler
    handlers["docker.run"] = _docker_hub_handler
    handlers["docker.execute_docker_container"] = _docker_hub_handler
    handlers["docker.hub"] = _docker_hub_handler
    handlers["docker.github"] = _docker_github_handler
    handlers["docker.github_repo"] = _docker_github_handler
    handlers["docker.build_and_execute_github_repo"] = _docker_github_handler

    supported = _compute_supported_task_types(
        supported_task_types=supported_task_types,
        accelerate_instance=accelerate_instance,
    )

    mesh_enabled = bool(_worker_mesh_enabled()) if mesh is None else bool(mesh)
    mesh_refresh = float(_worker_mesh_refresh_s()) if mesh_refresh_s is None else float(mesh_refresh_s)
    mesh_claim_interval = (
        float(_worker_mesh_claim_interval_s())
        if mesh_claim_interval_s is None
        else float(mesh_claim_interval_s)
    )
    mesh_peers_limit = int(_worker_mesh_max_peers()) if mesh_max_peers is None else int(mesh_max_peers)

    # Mesh discovery state.
    mesh_stop = threading.Event()
    mesh_peers_lock = threading.RLock()
    mesh_peers_state: dict[str, object] = {"peers": []}
    mesh_rr = 0
    last_mesh_claim = 0.0

    mesh_thread: threading.Thread | None = None
    if mesh_enabled:
        static_peers = _worker_mesh_static_peers()
        if static_peers:
            with mesh_peers_lock:
                mesh_peers_state["peers"] = list(static_peers)
        else:
            mesh_thread = _start_mesh_discovery_thread(
                stop=mesh_stop,
                worker_id=str(worker_id),
                peers_out=mesh_peers_state,
                peers_lock=mesh_peers_lock,
                max_peers=int(mesh_peers_limit),
                refresh_s=float(mesh_refresh),
            )

    def _snapshot_mesh_peers() -> list[object]:
        with mesh_peers_lock:
            peers = mesh_peers_state.get("peers")
            return list(peers) if isinstance(peers, list) else []

    def _maybe_claim_from_mesh() -> Optional[tuple[object, Dict[str, Any]]]:
        nonlocal mesh_rr, last_mesh_claim
        if not mesh_enabled:
            return None
        now = time.time()
        if (now - last_mesh_claim) < float(mesh_claim_interval):
            return None

        peers = _snapshot_mesh_peers()
        if not peers:
            last_mesh_claim = now
            return None

        # Avoid tight loops when there are peers but all are empty/unreachable.
        last_mesh_claim = now

        # Lazy import to avoid requiring libp2p in non-mesh deployments.
        try:
            from ipfs_accelerate_py.p2p_tasks.client import claim_next_sync
        except Exception:
            return None

        # Restrict mesh to safe task types.
        mesh_supported = _mesh_safe_task_types(list(supported or []))
        if not mesh_supported:
            mesh_supported = ["text-generation"]

        # Try one peer per poll in round-robin order.
        idx = int(mesh_rr) % max(1, len(peers))
        mesh_rr = (idx + 1) % max(1, len(peers))
        remote = peers[idx]

        try:
            task = claim_next_sync(
                remote=remote,  # RemoteQueue
                worker_id=str(worker_id),
                supported_task_types=list(mesh_supported),
                peer_id=str(worker_id),
                clock=None,
            )
            if task is None:
                return None
            if isinstance(task, dict):
                return (remote, task)
            return None
        except Exception:
            return None

    def _complete_mesh_task(
        *,
        remote: object,
        task_id: str,
        ok: bool,
        result: Dict[str, Any] | None,
        error: str | None,
    ) -> None:
        try:
            from ipfs_accelerate_py.p2p_tasks.client import complete_task_sync

            complete_task_sync(
                remote=remote,  # RemoteQueue
                task_id=str(task_id),
                status="completed" if ok else "failed",
                result=(result if ok else None),
                error=(None if ok else str(error or "unknown error")),
            )
        except Exception:
            return

    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                return 0
            task = queue.claim_next(worker_id=worker_id, supported_task_types=supported)
            if task is None:
                # No local work; optionally help the mesh by claiming from peers.
                mesh_claim = _maybe_claim_from_mesh()
                if mesh_claim is not None:
                    remote, remote_task = mesh_claim
                    task_id = str(remote_task.get("task_id") or "").strip()
                    if not task_id:
                        # Nothing we can do.
                        continue

                    result: Dict[str, Any] | None = None
                    error: str | None = None
                    ok = False
                    try:
                        ttype = str(remote_task.get("task_type") or "").strip().lower()
                        handler = handlers.get(ttype)
                        if handler is None:
                            raise RuntimeError(f"Unsupported task_type: {remote_task.get('task_type')}")
                        result = handler(
                            {
                                "task_id": task_id,
                                "task_type": remote_task.get("task_type"),
                                "model_name": remote_task.get("model_name"),
                                "payload": remote_task.get("payload"),
                            }
                        )
                        if isinstance(result, dict):
                            # Ensure mesh executions are attributable.
                            result = dict(result)
                            progress = result.get("progress")
                            if not isinstance(progress, dict):
                                progress = {}
                            if not progress.get("worker_id"):
                                progress = dict(progress)
                                progress["worker_id"] = str(worker_id)
                                progress["task_type"] = str(remote_task.get("task_type") or "")
                                progress["mesh"] = True
                            result["progress"] = progress
                        ok = True
                    except Exception as exc:
                        ok = False
                        error = str(exc)
                    _complete_mesh_task(remote=remote, task_id=task_id, ok=ok, result=result, error=error)
                    if once:
                        return 0
                    continue

                if once:
                    return 0
                sleep_s = max(0.05, float(poll_interval_s))
                if stop_event is None:
                    time.sleep(sleep_s)
                else:
                    stop_event.wait(sleep_s)
                continue

            # Generic heartbeat/progress so peers can observe long-running tasks via
            # RPC get/wait polling even for non-docker handlers.
            stop_hb = threading.Event()

            def _task_hb() -> None:
                while not stop_hb.is_set():
                    try:
                        queue.update(
                            task_id=task.task_id,
                            status="running",
                            result_patch={
                                "progress": {
                                    "heartbeat_ts": time.time(),
                                    "worker_id": str(worker_id),
                                    "task_type": str(task.task_type),
                                }
                            },
                        )
                    except Exception:
                        pass
                    stop_hb.wait(0.5)

            # Also mark initial start.
            try:
                queue.update(
                    task_id=task.task_id,
                    status="running",
                    result_patch={
                        "progress": {
                            "phase": "starting",
                            "ts": time.time(),
                            "worker_id": str(worker_id),
                            "task_type": str(task.task_type),
                        }
                    },
                )
            except Exception:
                pass

            hb_thread = threading.Thread(target=_task_hb, name=f"task_hb[{task.task_id}]", daemon=True)
            hb_thread.start()

            result: Dict[str, Any] | None = None
            error: str | None = None
            status: str = "failed"
            try:
                ttype = str(task.task_type or "").strip().lower()
                handler = handlers.get(ttype)
                if handler is None:
                    status = "failed"
                    error = f"Unsupported task_type: {task.task_type}"
                else:
                    result = handler(
                        {
                            "task_id": task.task_id,
                            "task_type": task.task_type,
                            "model_name": task.model_name,
                            "payload": task.payload,
                        }
                    )
                    status = "completed"
            except Exception as exc:
                status = "failed"
                error = str(exc)
            finally:
                # Stop heartbeat before completing to avoid DuckDB write conflicts.
                stop_hb.set()
                try:
                    hb_thread.join(timeout=0.5)
                except Exception:
                    pass

            if status == "completed":
                queue.complete(task_id=task.task_id, status="completed", result=result or {})
            else:
                queue.complete(task_id=task.task_id, status="failed", error=error or "unknown error")

            if once:
                return 0
    finally:
        try:
            queue.close()
        except Exception:
            pass
        if mesh_enabled:
            mesh_stop.set()
            if mesh_thread is not None:
                try:
                    mesh_thread.join(timeout=0.5)
                except Exception:
                    pass


def run_autoscaled_workers(
    *,
    queue_path: str,
    base_worker_id: str,
    min_workers: int = 1,
    max_workers: int = 4,
    scale_poll_s: float = 2.0,
    scale_down_idle_s: float = 30.0,
    poll_interval_s: float = 0.25,
    once: bool = False,
    p2p_service: bool = False,
    p2p_listen_port: Optional[int] = None,
    accelerate_instance: object | None = None,
    supported_task_types: Optional[list[str]] = None,
    mesh: Optional[bool] = None,
    mesh_children: Optional[bool] = False,
    autoscale_remote: bool = False,
    remote_refresh_s: float = 5.0,
    remote_max_peers: int = 10,
    stop_event: threading.Event | None = None,
) -> int:
    """Autoscale worker threads based on local and (optionally) remote backlog.

    This manager starts between `min_workers` and `max_workers` worker threads.
    Each worker runs `run_worker(...)` with a unique worker_id.

        Notes:
        - By default, scale decisions are based on the *local* DuckDB queue backlog.
        - When `autoscale_remote=True`, the manager also polls discovered peers
            (via TaskQueue status(detail=True)) and scales up when remote queues have
            queued tasks of types this node supports.
    - Workers stop cooperatively after they finish their current task.
    """

    import uuid

    min_w = max(0, int(min_workers))
    max_w = max(min_w, int(max_workers))
    poll_s = max(0.2, float(scale_poll_s))
    idle_s = max(0.0, float(scale_down_idle_s))

    if once:
        # Autoscale manager is intended for long-running services.
        # For one-shot runs, just run a single worker.
        return run_worker(
            queue_path=queue_path,
            worker_id=str(base_worker_id),
            poll_interval_s=float(poll_interval_s),
            once=True,
            p2p_service=bool(p2p_service),
            p2p_listen_port=p2p_listen_port,
            accelerate_instance=accelerate_instance,
            supported_task_types=supported_task_types,
            mesh=mesh,
            stop_event=stop_event,
        )

    # Determine supported task types for counting and for child workers.
    # If not explicitly provided, mirror run_worker() defaults.
    supported_for_workers = _compute_supported_task_types(
        supported_task_types=supported_task_types,
        accelerate_instance=accelerate_instance,
    )

    # Lightweight queue reader used only for counts.
    q = TaskQueue(queue_path)

    workers_lock = threading.RLock()
    workers: list[tuple[str, threading.Thread, threading.Event]] = []
    last_nonzero_ts = 0.0

    # Remote backlog aggregation state.
    remote_lock = threading.RLock()
    remote_state: dict[str, object] = {
        "queued": 0,
        "queued_by_type": {},
        "peers": [],
        "ts": 0.0,
    }
    remote_stop = threading.Event()
    remote_thread: threading.Thread | None = None

    def _make_id(idx: int) -> str:
        return f"{str(base_worker_id)}-w{int(idx)}-{uuid.uuid4().hex[:6]}"

    def _start_one(*, idx: int, start_service: bool) -> None:
        wid = _make_id(idx)
        ev = threading.Event()

        def _run() -> None:
            run_worker(
                queue_path=queue_path,
                worker_id=wid,
                poll_interval_s=float(poll_interval_s),
                once=False,
                p2p_service=bool(start_service),
                p2p_listen_port=p2p_listen_port,
                accelerate_instance=accelerate_instance,
                supported_task_types=list(supported_for_workers or []),
                mesh=mesh if start_service else (mesh_children if mesh_children is not None else mesh),
                stop_event=ev,
            )

        t = threading.Thread(target=_run, name=f"task_worker[{wid}]", daemon=True)
        t.start()
        with workers_lock:
            workers.append((wid, t, ev))

    def _stop_extra(desired: int) -> None:
        with workers_lock:
            while len(workers) > desired:
                wid, t, ev = workers.pop()
                try:
                    ev.set()
                except Exception:
                    pass
                try:
                    t.join(timeout=0.2)
                except Exception:
                    pass

    # Start initial pool.
    initial = min_w if min_w > 0 else 1
    initial = min(initial, max_w)
    for i in range(initial):
        _start_one(idx=i, start_service=(bool(p2p_service) and i == 0))

    # Optional: remote backlog polling thread.
    if bool(autoscale_remote):
        try:
            from ipfs_accelerate_py.p2p_tasks.client import (
                RemoteQueue,
                discover_peers_via_mdns_sync,
                request_status_sync,
            )

            expected_session = _expected_session_tag()

            def _remote_loop() -> None:
                refresh_s = max(0.5, float(remote_refresh_s))
                limit = max(1, min(100, int(remote_max_peers)))

                while not remote_stop.is_set() and (stop_event is None or not stop_event.is_set()):
                    peers: list[RemoteQueue] = []
                    try:
                        peers = discover_peers_via_mdns_sync(timeout_s=1.0, limit=limit, exclude_self=True)
                    except Exception:
                        peers = []

                    queued_total = 0
                    queued_by_type: dict[str, int] = {}
                    keep: list[RemoteQueue] = []
                    for rq in list(peers or []):
                        try:
                            resp = request_status_sync(remote=rq, timeout_s=3.0, detail=True)
                        except Exception:
                            continue
                        if not (isinstance(resp, dict) and resp.get("ok")):
                            continue
                        if expected_session and str(resp.get("session") or "").strip() != expected_session:
                            continue

                        queue_info = resp.get("queue")
                        if not isinstance(queue_info, dict):
                            continue
                        qb = queue_info.get("queued_by_type")
                        if not isinstance(qb, dict):
                            qb = {}

                        keep.append(rq)
                        for ttype, count in qb.items():
                            tt = str(ttype or "").strip()
                            if not tt:
                                continue
                            if supported_for_workers and tt not in supported_for_workers:
                                continue
                            try:
                                n = int(count)
                            except Exception:
                                n = 0
                            if n <= 0:
                                continue
                            queued_total += n
                            queued_by_type[tt] = int(queued_by_type.get(tt, 0) + n)

                    with remote_lock:
                        remote_state["queued"] = int(queued_total)
                        remote_state["queued_by_type"] = dict(queued_by_type)
                        remote_state["peers"] = list(keep)
                        remote_state["ts"] = float(time.time())

                    remote_stop.wait(refresh_s)

            remote_thread = threading.Thread(
                target=_remote_loop,
                name=f"task_worker_autoscale_remote[{base_worker_id}]",
                daemon=True,
            )
            remote_thread.start()
        except Exception:
            remote_thread = None

    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                break

            try:
                pending_local = int(q.count(status="queued", task_types=list(supported_for_workers or [])))
            except Exception:
                pending_local = 0

            pending_remote = 0
            if bool(autoscale_remote):
                with remote_lock:
                    try:
                        pending_remote = int(remote_state.get("queued") or 0)
                    except Exception:
                        pending_remote = 0

            pending = max(0, int(pending_local + pending_remote))

            now = time.time()
            if pending > 0:
                last_nonzero_ts = now

            # Simple heuristic: scale workers up to min(max_w, max(min_w, pending)).
            desired = min_w
            if pending > 0:
                desired = max(min_w, min(max_w, pending))

            # Scale down only after being idle for a while.
            with workers_lock:
                current = len(workers)
            if desired < current:
                if idle_s <= 0.0 or (last_nonzero_ts and (now - last_nonzero_ts) >= idle_s) or pending == 0:
                    _stop_extra(desired)
            elif desired > current:
                for i in range(current, desired):
                    _start_one(idx=i, start_service=False)

            if stop_event is None:
                time.sleep(poll_s)
            else:
                stop_event.wait(poll_s)

    finally:
        remote_stop.set()
        if remote_thread is not None:
            try:
                remote_thread.join(timeout=0.5)
            except Exception:
                pass
        # Stop all child workers.
        with workers_lock:
            for _wid, _t, _ev in workers:
                try:
                    _ev.set()
                except Exception:
                    pass
            for _wid, _t, _ev in workers:
                try:
                    _t.join(timeout=0.5)
                except Exception:
                    pass
            workers.clear()
        try:
            q.close()
        except Exception:
            pass

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
    parser.add_argument(
        "--mesh",
        action="store_true",
        help="Enable mDNS mesh mode: discover peers and claim tasks from them (opt-in).",
    )
    parser.add_argument(
        "--mesh-refresh-s",
        type=float,
        default=None,
        help="How often to refresh mDNS peer list (default: env or 5s)",
    )
    parser.add_argument(
        "--mesh-claim-interval-s",
        type=float,
        default=None,
        help="Minimum seconds between mesh claim attempts when idle (default: env or 0.5s)",
    )
    parser.add_argument(
        "--mesh-max-peers",
        type=int,
        default=None,
        help="Max peers to keep from mDNS discovery (default: env or 10)",
    )

    args = parser.parse_args(argv)
    return run_worker(
        queue_path=args.queue_path,
        worker_id=args.worker_id,
        poll_interval_s=float(args.poll_interval_s),
        once=bool(args.once),
        p2p_service=bool(args.p2p_service),
        p2p_listen_port=args.p2p_listen_port,
        mesh=bool(args.mesh) if bool(args.mesh) else None,
        mesh_refresh_s=args.mesh_refresh_s,
        mesh_claim_interval_s=args.mesh_claim_interval_s,
        mesh_max_peers=args.mesh_max_peers,
    )


if __name__ == "__main__":
    raise SystemExit(main())
