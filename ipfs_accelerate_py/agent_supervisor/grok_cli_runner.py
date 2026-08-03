#!/usr/bin/env python3
"""Thin process entry: Grok Build CLI agent for implementation worktrees.

Reads the implementation prompt from stdin (daemon contract), writes it to a
temp prompt file, then execs the official ``grok`` binary with agent-capable
flags. Command policy lives next to other CLI peers in :mod:`llm_router`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))


DEFAULT_GROK_MODEL = "grok-4.5"
# Grok CLI validates --max-turns as 1..=4294967295 (u32::MAX).
DEFAULT_GROK_MAX_TURNS = 4_294_967_295
MAX_CODEX_FALLBACK_ARGUMENTS = 64
MAX_CODEX_FALLBACK_ARGUMENT_BYTES = 4_096
MAX_GROK_STREAM_EVENT_BYTES = 64 * 1024
GROK_QUOTA_ERROR_TYPES = frozenset({"usage_pool_exhausted", "usage_limit_reached"})
CODEX_QUOTA_FALLBACK_MODEL = "gpt-5.6-terra"
CODEX_QUOTA_FALLBACK_REASONING = 'model_reasoning_effort="medium"'
_LEGACY_GROK_BALANCE_EXHAUSTED_MESSAGE = (
    "API error (status 402 Payment Required): Grok Build usage balance exhausted"
)
_CODEX_FALLBACK_CONFIG_KEYS = frozenset(
    {
        "agents.max_depth",
        "agents.max_threads",
        "model_context_window",
        "model_reasoning_effort",
    }
)


def _resolve_grok_bin(configured: str = "") -> str:
    if configured.strip():
        path = Path(configured).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    try:
        from ipfs_accelerate_py.llm_router import _grok_cli_command

        candidate = str(_grok_cli_command() or "").strip()
        if candidate:
            found = shutil.which(candidate) or (
                candidate if Path(candidate).is_file() else ""
            )
            if found:
                return found
    except Exception:
        pass
    return shutil.which("grok") or ""


def build_grok_agent_command(
    *,
    workspace: Path,
    prompt_file: Path,
    model: str,
    max_turns: int,
    permission_mode: str,
    grok_bin: str,
) -> list[str]:
    """Build an agent-mode ``grok`` argv for an implementation worktree."""

    cmd = [
        grok_bin,
        "--cwd",
        str(workspace),
        "--model",
        model,
        "--permission-mode",
        permission_mode,
        "--always-approve",
        "--max-turns",
        str(max_turns),
        "--output-format",
        "plain",
        "--prompt-file",
        str(prompt_file),
    ]
    return cmd


def _parse_codex_fallback_command(raw: str) -> list[str]:
    """Decode the daemon-authored Codex fallback without invoking a shell."""

    if not raw.strip():
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Codex fallback command is not valid JSON") from exc
    if (
        not isinstance(payload, list)
        or not 2 <= len(payload) <= MAX_CODEX_FALLBACK_ARGUMENTS
    ):
        raise ValueError("Codex fallback command must be a bounded argv array")
    command: list[str] = []
    for item in payload:
        if (
            not isinstance(item, str)
            or not item
            or len(item.encode("utf-8")) > MAX_CODEX_FALLBACK_ARGUMENT_BYTES
        ):
            raise ValueError("Codex fallback command contains an invalid argument")
        command.append(item)
    if Path(command[0]).name.lower() not in {"codex", "codex.exe"}:
        raise ValueError("Codex fallback executable must be codex")
    if command[1] != "exec" or command[-1] != "-":
        raise ValueError("Codex fallback command must use `codex exec ... -`")
    _validate_codex_quota_fallback_command(command)
    return command


def _validate_codex_quota_fallback_command(
    command: Sequence[str],
    *,
    workspace: Path | None = None,
) -> None:
    """Require the exact daemon-owned Terra/medium fallback shape."""

    if len(command) < 8 or Path(command[0]).name.lower() not in {
        "codex",
        "codex.exe",
    }:
        raise ValueError("Codex quota fallback executable must be codex")
    if command[1] != "exec" or command[-1] != "-":
        raise ValueError("Codex quota fallback must use `codex exec ... -`")

    bypass_count = 0
    option_values: dict[str, list[str]] = {"-C": [], "-m": [], "-c": []}
    index = 2
    while index < len(command) - 1:
        item = command[index]
        if item == "--dangerously-bypass-approvals-and-sandbox":
            bypass_count += 1
            index += 1
            continue
        if item not in option_values or index + 1 >= len(command) - 1:
            raise ValueError(
                "Codex quota fallback contains an unauthorized route option"
            )
        option_values[item].append(command[index + 1])
        index += 2

    if bypass_count != 1:
        raise ValueError(
            "Codex quota fallback must contain exactly one implementation "
            "sandbox policy"
        )
    if option_values["-m"] != [CODEX_QUOTA_FALLBACK_MODEL]:
        raise ValueError("Codex quota fallback model is not exactly gpt-5.6-terra")
    if len(option_values["-C"]) != 1:
        raise ValueError("Codex quota fallback must contain exactly one workspace")
    fallback_workspace = Path(option_values["-C"][0]).resolve()
    if workspace is not None and fallback_workspace != workspace:
        raise ValueError("Codex quota fallback workspace does not match Grok workspace")

    configs: dict[str, str] = {}
    for config in option_values["-c"]:
        key, separator, value = config.partition("=")
        if (
            not separator
            or key not in _CODEX_FALLBACK_CONFIG_KEYS
            or key in configs
        ):
            raise ValueError(
                "Codex quota fallback contains an unauthorized or duplicate config"
            )
        configs[key] = value
    if configs.get("model_reasoning_effort") != '"medium"':
        raise ValueError("Codex quota fallback reasoning is not exactly medium")
    for key in ("agents.max_depth", "agents.max_threads", "model_context_window"):
        value = configs.get(key)
        if value is not None and re.fullmatch(r"[1-9][0-9]*", value) is None:
            raise ValueError(f"Codex quota fallback {key} must be a positive integer")


def _grok_failure_type_from_stream_event(line: str) -> str:
    """Project one CLI-owned native failure event, never model-authored text."""

    if (
        not line
        or len(line.encode("utf-8", errors="replace"))
        > MAX_GROK_STREAM_EVENT_BYTES
    ):
        return ""
    try:
        payload = json.loads(line)
    except (json.JSONDecodeError, TypeError):
        return ""
    if not isinstance(payload, dict) or payload.get("method") not in {
        "_x.ai/session/update",
        "session/update",
    }:
        return ""
    params = payload.get("params")
    update = params.get("update") if isinstance(params, dict) else None
    if (
        not isinstance(update, dict)
        or update.get("sessionUpdate") != "retry_state"
        or update.get("type") != "failed"
    ):
        return ""
    error_type = str(update.get("error_type") or "").strip().casefold()
    if error_type in GROK_QUOTA_ERROR_TYPES:
        return error_type
    if (
        error_type == "api"
        and str(update.get("message") or "").strip()
        == _LEGACY_GROK_BALANCE_EXHAUSTED_MESSAGE
    ):
        return "usage_pool_exhausted"
    return error_type or "unknown"


def _stream_pipe(
    source: TextIO,
    destination: TextIO,
    *,
    failure_types: set[str] | None = None,
) -> None:
    """Tee a child stream while inspecting only bounded complete JSON lines."""

    pending = ""
    discarding_oversized_line = False
    while True:
        chunk = source.read(16 * 1024)
        if not chunk:
            break
        destination.write(chunk)
        destination.flush()
        if failure_types is None:
            continue
        pending += chunk
        while "\n" in pending:
            line, pending = pending.split("\n", 1)
            if not discarding_oversized_line:
                failure_type = _grok_failure_type_from_stream_event(line)
                if failure_type:
                    failure_types.add(failure_type)
            discarding_oversized_line = False
        if (
            len(pending.encode("utf-8", errors="replace"))
            > MAX_GROK_STREAM_EVENT_BYTES
        ):
            pending = ""
            discarding_oversized_line = True
    if pending and not discarding_oversized_line and failure_types is not None:
        failure_type = _grok_failure_type_from_stream_event(pending)
        if failure_type:
            failure_types.add(failure_type)


def _run_grok_with_typed_failure_capture(
    command: Sequence[str],
    *,
    env: dict[str, str],
) -> tuple[int, set[str]]:
    """Run Grok with live output and return native structured failure types."""

    process = subprocess.Popen(
        list(command),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    failure_types: set[str] = set()
    stdout_thread = threading.Thread(
        target=_stream_pipe,
        args=(process.stdout, sys.stdout),
        kwargs={"failure_types": failure_types},
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_stream_pipe,
        args=(process.stderr, sys.stderr),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()
    returncode = int(process.wait())
    stdout_thread.join()
    stderr_thread.join()
    return returncode, failure_types


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Authorized Grok CLI agent entry (llm_router.grok_cli)."
    )
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--grok-bin", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--max-turns", default="")
    parser.add_argument(
        "--permission-mode",
        default="",
        help="Grok permission mode (default: bypassPermissions in agent mode).",
    )
    parser.add_argument(
        "--mode",
        default="agent",
        choices=("agent", "chat"),
        help="agent enables tool approvals for implementation work",
    )
    parser.add_argument(
        "--codex-fallback-command-json",
        default="",
        help=(
            "Internal default-route Codex argv. It is run only after Grok "
            "emits a verified native quota-exhaustion event; forced-Grok "
            "routes omit this option."
        ),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        codex_fallback_command = _parse_codex_fallback_command(
            str(args.codex_fallback_command_json)
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    from ipfs_accelerate_py.llm_router import (
        LLMRouterError,
        build_grok_cli_command,
        build_grok_cli_env,
        find_grok_cli,
    )

    workspace = args.workspace.expanduser().resolve()
    if not workspace.is_dir():
        print(f"workspace is not a directory: {workspace}", file=sys.stderr)
        return 2
    if codex_fallback_command:
        try:
            _validate_codex_quota_fallback_command(
                codex_fallback_command,
                workspace=workspace,
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2

    grok_bin = str(args.grok_bin).strip() or find_grok_cli() or ""
    if not grok_bin:
        print("grok CLI not found on PATH", file=sys.stderr)
        return 127

    model = (
        str(args.model).strip()
        or os.environ.get("IPFS_ACCELERATE_AGENT_GROK_MODEL", "").strip()
        or os.environ.get("ipfs_accelerate_py_GROK_CLI_MODEL", "").strip()
        or os.environ.get("GROK_CLI_MODEL", "").strip()
        or DEFAULT_GROK_MODEL
    )
    if codex_fallback_command and model != DEFAULT_GROK_MODEL:
        print(
            "Default Grok/Codex route requires primary model grok-4.5",
            file=sys.stderr,
        )
        return 2
    max_turns_raw = (
        str(args.max_turns).strip()
        or os.environ.get("IPFS_ACCELERATE_AGENT_GROK_MAX_TURNS", "").strip()
        or os.environ.get("ipfs_accelerate_py_GROK_CLI_MAX_TURNS", "").strip()
        or str(DEFAULT_GROK_MAX_TURNS)
    )
    try:
        max_turns = max(1, min(DEFAULT_GROK_MAX_TURNS, int(max_turns_raw)))
    except ValueError:
        max_turns = DEFAULT_GROK_MAX_TURNS
    permission_mode = (
        str(args.permission_mode).strip()
        or os.environ.get("IPFS_ACCELERATE_AGENT_GROK_PERMISSION_MODE", "").strip()
        or os.environ.get("ipfs_accelerate_py_GROK_CLI_PERMISSION_MODE", "").strip()
        or "bypassPermissions"
    )

    prompt = sys.stdin.read()
    if not prompt.strip():
        print("empty implementation prompt on stdin", file=sys.stderr)
        return 2

    prompt_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix="asref-grok-prompt-",
            suffix=".txt",
            delete=False,
        ) as handle:
            handle.write(prompt)
            prompt_path = handle.name

        try:
            cmd = build_grok_cli_command(
                mode=str(args.mode),
                workspace=workspace,
                model_name=model,
                max_turns=max_turns,
                grok_bin=grok_bin,
                prompt_file=prompt_path,
                permission_mode=permission_mode,
            )
            env = build_grok_cli_env()
        except LLMRouterError as exc:
            print(str(exc), file=sys.stderr)
            return 2

        os.chdir(workspace)
        failure_types: set[str] = set()
        if codex_fallback_command:
            output_index = cmd.index("--output-format") + 1
            cmd[output_index] = "streaming-json"
            try:
                primary_returncode, failure_types = (
                    _run_grok_with_typed_failure_capture(cmd, env=env)
                )
            except OSError as exc:
                print(f"unable to launch Grok CLI: {exc}", file=sys.stderr)
                return 127
        else:
            completed = subprocess.run(cmd, env=env, check=False)
            primary_returncode = int(completed.returncode)
        if primary_returncode == 0 or not codex_fallback_command:
            return primary_returncode

        quota_types = failure_types & GROK_QUOTA_ERROR_TYPES
        if not quota_types or failure_types - GROK_QUOTA_ERROR_TYPES:
            print(
                "Grok CLI failed without a verified quota-exhaustion event; "
                "Codex fallback is forbidden",
                file=sys.stderr,
            )
            return primary_returncode

        print(
            "Grok quota exhausted; invoking the pinned Terra/medium fallback",
            file=sys.stderr,
        )
        try:
            fallback = subprocess.run(
                codex_fallback_command,
                cwd=workspace,
                env=os.environ.copy(),
                input=prompt,
                text=True,
                check=False,
            )
        except OSError as exc:
            print(f"unable to launch Codex fallback: {exc}", file=sys.stderr)
            return 127
        return int(fallback.returncode)
    finally:
        if prompt_path:
            try:
                os.unlink(prompt_path)
            except OSError:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
