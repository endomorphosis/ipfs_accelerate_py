"""Supervisor hello-world check for Muse Code.

Installs the official ``muse`` CLI when missing, then runs a bounded
``muse exec`` prompt that must reply ``hello world``. Implementation work
still uses :mod:`cli_implement_runner`; this helper is the smoke path.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional

HELLO_WORLD_PROMPT = (
    "Reply with exactly the two words: hello world\n"
    "Do not edit files. Do not run commands."
)
HELLO_WORLD_MARKER = "hello world"


def run_muse_hello_world_check(
    *,
    workspace: Path,
    auto_install: bool = True,
    timeout_seconds: float = 180.0,
    max_model_steps: int = 8,
    executable: Optional[str] = None,
) -> dict[str, Any]:
    """Install Muse Code if needed and run a hello-world ``muse exec``.

    Never logs credentials. Returns a bounded receipt with text, exit code,
    and the resolved executable.
    """

    from ipfs_accelerate_py.agent_supervisor.todo_daemon.cli_provider_balance import (
        ensure_muse_cli_binary,
        muse_code_auth_available,
        resolve_muse_cli_binary,
    )
    from ipfs_accelerate_py.cli_runtime.providers.muse import (
        build_muse_exec_argv,
        parse_muse_jsonl,
    )

    workspace = Path(workspace).expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    installed = False
    muse = str(executable or "").strip()
    if not muse:
        prior = resolve_muse_cli_binary()
        if auto_install:
            muse = str(ensure_muse_cli_binary() or "")
            installed = bool(muse) and muse != str(prior or "")
        else:
            muse = str(prior or "")
    if not muse:
        return {
            "ok": False,
            "text": "",
            "executable": "",
            "exit_code": 127,
            "installed": False,
            "reason": "muse_not_installed",
        }

    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix="muse-hello-world-",
        suffix=".txt",
        delete=False,
    )
    prompt_file = handle.name
    try:
        handle.write(HELLO_WORLD_PROMPT)
        handle.flush()
        handle.close()
        # Do not inject secrets-manager META_API_KEY. Muse docs: that env
        # always overrides `muse login`. Headless CI can still set the env.
        env = os.environ.copy()
        runtime_tmp = workspace.parent / ".muse-hello-tmp"
        runtime_tmp.mkdir(parents=True, exist_ok=True)
        env["TMPDIR"] = str(runtime_tmp)

        def _run(extra: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
            argv = build_muse_exec_argv(
                executable=muse,
                prompt_file=prompt_file,
                json_events=True,
                disable_approval=True,
                yolo=False,
                workspace=str(workspace),
                max_model_steps=max(1, int(max_model_steps)),
                extra_args=extra,
            )
            return subprocess.run(
                argv,
                cwd=str(workspace),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )

        # Default Meta provider (not --provider echo). Echo is a local stub
        # inside the CLI and is not Muse Spark output.
        completed = _run(("--no-session-log",))
        combined = (
            str(getattr(completed, "stdout", "") or "")
            + "\n"
            + str(getattr(completed, "stderr", "") or "")
        ).lower()
        if int(getattr(completed, "returncode", 1) or 0) == 2 and "no-session-log" in combined:
            completed = _run(())
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "text": "",
            "executable": muse,
            "exit_code": 124,
            "installed": installed,
            "reason": "timeout",
        }
    finally:
        try:
            os.unlink(prompt_file)
        except OSError:
            pass

    stdout = str(getattr(completed, "stdout", "") or "")
    stderr = str(getattr(completed, "stderr", "") or "")
    parsed = parse_muse_jsonl(stdout)
    model_id, error = _model_and_error_from_jsonl(stdout)
    text = str(parsed.text or "").strip()
    marker = HELLO_WORLD_MARKER in text.lower()
    exit_code = int(getattr(completed, "returncode", 1) or 0)
    ok = bool(marker)
    reason = "hello_world_ok" if ok else "hello_world_mismatch"
    if error and not marker:
        reason = "muse_api_error"
    elif not marker and exit_code not in {0, None}:
        reason = "muse_exec_failed"
    return {
        "ok": ok,
        "text": (text or error or stderr.strip())[:2048],
        "executable": muse,
        "exit_code": exit_code,
        "installed": installed,
        "authenticated": muse_code_auth_available(),
        "reason": reason,
        "model_id": model_id,
        "error": error[:512],
        "provider": "meta",
    }


def _model_and_error_from_jsonl(stdout: str) -> tuple[str, str]:
    import json

    model_id = ""
    error = ""
    for line in str(stdout or "").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        payload_type = str(event.get("payload_type") or "")
        if payload.get("model_id") and not model_id:
            model_id = str(payload.get("model_id") or "")[:128]
        nested = payload.get("event") if isinstance(payload.get("event"), dict) else {}
        for candidate in (
            payload.get("reason"),
            payload.get("error"),
            payload.get("message"),
            nested.get("reason"),
            nested.get("error"),
        ):
            if isinstance(candidate, str) and candidate.strip() and (
                "error" in candidate.lower() or "fail" in candidate.lower()
            ):
                error = candidate.strip()[:512]
        if payload_type.endswith("failed") and not error:
            error = str(payload.get("reason") or payload_type)[:512]
    return model_id, error


def live_muse_hello_world_enabled(environ: Optional[Mapping[str, str]] = None) -> bool:
    env = os.environ if environ is None else environ
    raw = str(env.get("IPFS_ACCELERATE_MUSE_LIVE") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


__all__ = [
    "HELLO_WORLD_MARKER",
    "HELLO_WORLD_PROMPT",
    "live_muse_hello_world_enabled",
    "run_muse_hello_world_check",
]
