"""Content-free admission checks for an ordinary Grok turn-limit candidate.

These observations authorize declared validation only. They never authorize
completion, another provider invocation, a refund, or release of a held task.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import sys
import time
from typing import Any, Sequence


MAX_LOG_BYTES = 16_000_000
MAX_FRAME_BYTES = 2_000_000
SCHEMA = "ipfs_accelerate_py/agent-supervisor/grok-turn-limit-observation@1"
_MODULE = "ipfs_accelerate_py.agent_supervisor.runtime.grok_cli_runner"
_FRAME_TYPES = {"available_commands", "thought", "text", "usage", "tool_call",
                "tool_call_update", "max_turns_reached", "end"}
_OPTIONS = {"--workspace", "--model", "--max-turns", "--mode", "--grok-bin",
            "--codex-fallback-reasoning-effort", "--permission-mode"}
_WARNING = re.compile(r'warning: --disallowed-tools (?:"[A-Za-z_][A-Za-z0-9_]*"(?:, )?)+ name no tool, so they remove nothing')


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def _object(raw: str) -> dict[str, Any]:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, "duplicate_stream_key")
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=unique,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite_stream_value")))
    _require(isinstance(value, dict), "nonobject_stream_frame")
    return value


def ordinary_grok_arguments(command: Sequence[str], workspace: Path) -> dict[str, str]:
    """Accept the exact ordinary no-fallback module entry, never a scoped route."""
    _require(list(command[:3]) == [sys.executable, "-m", _MODULE], "not_ordinary_grok_command")
    tail = list(command[3:])
    _require(len(tail) % 2 == 0, "invalid_grok_arguments")
    values: dict[str, str] = {}
    for key, value in zip(tail[::2], tail[1::2]):
        _require(key in _OPTIONS and key not in values and isinstance(value, str)
                 and value and not any(c in value for c in "\0\r\n"), "invalid_grok_arguments")
        values[key] = value
    _require(values.get("--workspace") == str(workspace.resolve())
             and values.get("--model") == "grok-4.7"
             and values.get("--mode") == "agent"
             and re.fullmatch(r"[1-9][0-9]{0,9}", values.get("--max-turns", "")) is not None
             and int(values["--max-turns"]) <= 4_294_967_295, "grok_invocation_binding_mismatch")
    return values


def observe_terminal(log_path: Path, *, command: Sequence[str], workspace: Path,
                     task_id: str, title: str, started_at: str, branch: str,
                     baseline_ref: str) -> dict[str, Any]:
    """Verify the complete native invocation log, retaining no prompt/tool bodies."""
    values = ordinary_grok_arguments(command, workspace)
    descriptor = os.open(log_path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        _require(stat.S_ISREG(before.st_mode) and before.st_uid == os.getuid()
                 and before.st_nlink == 1 and not before.st_mode & 0o077
                 and 0 < before.st_size <= MAX_LOG_BYTES, "unowned_or_unbounded_grok_log")
        with os.fdopen(os.dup(descriptor), "rb") as handle:
            raw = handle.read(MAX_LOG_BYTES + 1)
        after = log_path.lstat()
        _require(len(raw) == before.st_size and all(getattr(before, key) == getattr(after, key)
                 for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")), "grok_log_changed")
    finally:
        os.close(descriptor)
    _require(raw.endswith(b"\n"), "incomplete_grok_stream")
    prefix = (f"Task: {task_id} {title}\nStarted: {started_at}\nWorkspace: {workspace}\n"
              f"Branch: {branch}\nBaseline: {baseline_ref}\nCommand: {shlex.join(command)}\n\n").encode()
    _require(raw.startswith(prefix), "grok_log_invocation_mismatch")
    maximum = int(values["--max-turns"])
    turn_frame = None
    ending = None
    stderr_terminal = 0
    frame_count = 0
    for raw_line in raw[len(prefix):].splitlines(keepends=True):
        _require(raw_line.endswith(b"\n") and len(raw_line) <= MAX_FRAME_BYTES
                 and b"\r" not in raw_line, "incomplete_or_oversized_grok_frame")
        line = raw_line.decode("utf-8", errors="strict").strip()
        if not line:
            continue
        if line == "Error: max turns reached":
            stderr_terminal += 1
            _require(stderr_terminal == 1, "duplicate_grok_stderr_terminal")
            continue
        if re.fullmatch(r"provider command available: [A-Za-z0-9_.+\-]+", line) or _WARNING.fullmatch(line):
            continue  # Native runner's bounded stderr may drain after stdout.
        event = _object(line)
        kind = event.get("type")
        _require(kind in _FRAME_TYPES and ending is None, "error_unknown_or_postterminal_grok_frame")
        _require(turn_frame is None or kind == "end", "nonterminal_frame_after_turn_limit")
        frame_count += 1
        if kind == "max_turns_reached":
            _require(turn_frame is None and ("num_turns" not in event or
                     type(event["num_turns"]) is int and event["num_turns"] == maximum), "invalid_turn_limit_frame")
            turn_frame = event
        elif kind == "end":
            _require(turn_frame is not None and event.get("stopReason") == "max_turn_requests"
                     and type(event.get("num_turns")) is int and event["num_turns"] == maximum
                     and event.get("cwd") == str(workspace.resolve()), "unmatched_grok_terminal")
            ending = event
    _require(turn_frame is not None and ending is not None and stderr_terminal == 1, "missing_grok_turn_terminal")
    return {"schema": SCHEMA, "provider": "grok_cli", "model": values["--model"],
            "provider_returncode": 1, "terminal_reason": "max_turn_requests", "num_turns": maximum,
            "log_sha256": hashlib.sha256(raw).hexdigest(), "log_size_bytes": len(raw),
            "command_sha256": hashlib.sha256(json.dumps(list(command), separators=(",", ":")).encode()).hexdigest(),
            "frame_count": frame_count, "completion_authority": False, "attempt_refunded": False,
            "new_provider_authority": False}


def observe_quiescence(process: subprocess.Popen[Any], *, command: Sequence[str],
                       birth: tuple[int, int, int, int] | None, workspace: Path) -> dict[str, Any]:
    """Observe the waited owned group and trusted local Docker; perform no cleanup."""
    _require(type(process) is subprocess.Popen and list(process.args) == list(command)
             and process.returncode == process.poll() == 1
             and birth is not None and birth[0] > 0 and birth[1] == os.getpid()
             and birth[2] == birth[3] == process.pid, "provider_process_custody_unverified")
    try:
        os.killpg(process.pid, 0)
    except ProcessLookupError:
        pass
    else:
        raise ValueError("provider_process_group_present")
    from ..runtime import grok_cli_runner as runtime
    binary = runtime._docker_isolation_binary()
    _require(bool(binary), "docker_quiescence_unavailable")
    argv = [binary, "--host=unix:///var/run/docker.sock"]
    deadline = time.monotonic() + 20
    def inspect(arguments):
        remaining = deadline - time.monotonic()
        _require(remaining > 0, "docker_quiescence_deadline_expired")
        result = subprocess.run(argv + arguments, stdin=subprocess.DEVNULL,
            capture_output=True, text=True, timeout=min(5, remaining), check=True, env=runtime._docker_control_env())
        _require(len(result.stdout.encode()) <= 256_000 and len(result.stderr.encode()) <= 256_000,
                 "docker_quiescence_exceeds_bound")
        return result.stdout
    identifiers = inspect(["ps", "-a", "--no-trunc", "--filter", "label=ipfs_accelerate.grok_isolation=true",
                           "--format", "{{.ID}}"]).splitlines()
    _require(len(identifiers) <= 32 and all(re.fullmatch(r"[0-9a-f]{64}", value) for value in identifiers),
             "docker_quiescence_inventory_invalid")
    for identifier in identifiers:
        mounts = json.loads(inspect(["inspect", "--format", "{{json .Mounts}}", identifier]))
        _require(isinstance(mounts, list) and len(mounts) <= 256, "docker_mounts_invalid")
        for mount in mounts:
            _require(isinstance(mount, dict), "docker_mount_invalid")
            if mount.get("Type") != "bind":
                continue
            source = Path(str(mount.get("Source", "")))
            _require(source.is_absolute(), "docker_mount_source_invalid")
            _require(not (source == workspace or source.is_relative_to(workspace) or workspace.is_relative_to(source)),
                     "provider_workspace_container_present")
    return {"provider_pid": process.pid, "provider_returncode": 1, "process_reaped": True,
            "process_group_absent": True, "workspace_containers_absent": True,
            "inspected_container_count": len(identifiers)}
