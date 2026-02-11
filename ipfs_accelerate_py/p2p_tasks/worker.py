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
import os
import subprocess
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple

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

    proc = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=timeout_v,
        check=False,
    )
    return {"returncode": int(proc.returncode), "stdout": proc.stdout, "stderr": proc.stderr}


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
                queue.update(
                    task_id=task_id,
                    status="running",
                    result_patch={"progress": {"phase": "timeout", "ts": now}},
                    append_log=f"[worker] timeout after {timeout_s}s; killed process",
                    log_stream="stderr",
                )
                break

            rc = proc.poll()

            if now - last_heartbeat >= heartbeat_interval_s:
                queue.update(
                    task_id=task_id,
                    status="running",
                    result_patch={"progress": {"phase": "running", "heartbeat_ts": now}},
                )
                last_heartbeat = now

            if now - last_flush >= flush_interval_s:
                # Flush buffered stdout/stderr lines to the queue.
                while stdout_buf:
                    line = stdout_buf.pop(0)
                    queue.update(task_id=task_id, status="running", append_log=line.rstrip("\n"), log_stream="stdout")
                while stderr_buf:
                    line = stderr_buf.pop(0)
                    queue.update(task_id=task_id, status="running", append_log=line.rstrip("\n"), log_stream="stderr")
                last_flush = now

            if rc is not None:
                break

            time.sleep(0.05)

        # Final flush
        while stdout_buf:
            line = stdout_buf.pop(0)
            queue.update(task_id=task_id, status="running", append_log=line.rstrip("\n"), log_stream="stdout")
        while stderr_buf:
            line = stderr_buf.pop(0)
            queue.update(task_id=task_id, status="running", append_log=line.rstrip("\n"), log_stream="stderr")
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
    return base


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

        allow = str(os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_SHELL") or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if not allow:
            raise RuntimeError("shell task_type disabled (set IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_SHELL=1)")

        task_id = str(task_dict.get("task_id") or "")
        argv = payload.get("argv")
        if not isinstance(argv, list) or not argv or not all(isinstance(x, str) and x for x in argv):
            raise ValueError("shell payload.argv must be a non-empty list[str]")

        timeout_s = payload.get("timeout_s")
        try:
            timeout_v = float(timeout_s) if timeout_s is not None else None
        except Exception:
            timeout_v = None

        stream_mode = bool(payload.get("stream_output"))
        if stream_mode:
            queue.update(task_id=task_id, status="running", append_log=f"[worker] exec: {' '.join(argv)}", log_stream="stderr")
            return _stream_subprocess(argv=argv, task_id=task_id, queue=queue, timeout_s=timeout_v)

        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout_v,
            check=False,
        )
        for line in (proc.stdout or "").splitlines():
            queue.update(task_id=task_id, status="running", append_log=line, log_stream="stdout")
        for line in (proc.stderr or "").splitlines():
            queue.update(task_id=task_id, status="running", append_log=line, log_stream="stderr")
        return {"returncode": int(proc.returncode), "stdout": proc.stdout, "stderr": proc.stderr}

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
        queue.update(
            task_id=task_id,
            status="running",
            result_patch={"progress": {"phase": "starting", "ts": time.time()}},
        )

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
            queue.update(
                task_id=task_id,
                status="running",
                append_log=f"[worker] exec: {' '.join(argv)}",
                log_stream="stderr",
            )
            result = _stream_subprocess(argv=argv, task_id=task_id, queue=queue, timeout_s=timeout_v)
            queue.update(
                task_id=task_id,
                status="running",
                result_patch={"progress": {"phase": "exited", "ts": time.time()}},
            )
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
                queue.update(
                    task_id=task_id,
                    status="running",
                    result_patch={"progress": {"phase": "running", "heartbeat_ts": time.time(), "image": image}},
                )
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
            queue.update(task_id=task_id, status="running", append_log=line, log_stream="stdout")
        for line in stderr.splitlines():
            queue.update(task_id=task_id, status="running", append_log=line, log_stream="stderr")
        if getattr(res, "error_message", None):
            queue.update(
                task_id=task_id,
                status="running",
                append_log=str(getattr(res, "error_message")),
                log_stream="stderr",
            )
        queue.update(
            task_id=task_id,
            status="running",
            result_patch={"progress": {"phase": "exited", "ts": time.time(), "image": image}},
        )

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

        queue.update(
            task_id=task_id,
            status="running",
            result_patch={"progress": {"phase": "building", "ts": time.time(), "repo_url": repo_url}},
        )

        from ipfs_accelerate_py.docker_executor import build_and_execute_from_github

        stop = threading.Event()

        def _hb() -> None:
            while not stop.is_set():
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
            queue.update(task_id=task_id, status="running", append_log=line, log_stream="stdout")
        for line in stderr.splitlines():
            queue.update(task_id=task_id, status="running", append_log=line, log_stream="stderr")
        if getattr(res, "error_message", None):
            queue.update(
                task_id=task_id,
                status="running",
                append_log=str(getattr(res, "error_message")),
                log_stream="stderr",
            )
        queue.update(
            task_id=task_id,
            status="running",
            result_patch={"progress": {"phase": "exited", "ts": time.time(), "repo_url": repo_url}},
        )

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

    supported = supported_task_types if isinstance(supported_task_types, list) and supported_task_types else None
    if supported is None:
        supported = _supported_task_types_from_env(["text-generation"])

    while True:
        task = queue.claim_next(worker_id=worker_id, supported_task_types=supported)
        if task is None:
            if once:
                return 0
            time.sleep(max(0.05, float(poll_interval_s)))
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
