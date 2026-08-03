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
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from ipfs_accelerate_py.agent_supervisor.todo_daemon.llm_defaults import (  # noqa: E402
    DEFAULT_CODEX_FALLBACK_MODEL,
    DEFAULT_CODEX_FALLBACK_REASONING_EFFORT,
)

DEFAULT_GROK_MODEL = "grok-4.5"
# Grok CLI validates --max-turns as 1..=4294967295 (u32::MAX).
DEFAULT_GROK_MAX_TURNS = 4_294_967_295
MAX_CODEX_FALLBACK_ARGUMENTS = 64
MAX_CODEX_FALLBACK_ARGUMENT_BYTES = 4_096
MAX_GROK_QUOTA_STDERR_BYTES = 64 * 1024
_CODEX_FALLBACK_MODEL = DEFAULT_CODEX_FALLBACK_MODEL
_CODEX_FALLBACK_REASONING = DEFAULT_CODEX_FALLBACK_REASONING_EFFORT
_GROK_QUOTA_REASONS = frozenset(
    {"usage_pool_exhausted", "usage_limit_reached"}
)
_GROK_BUILD_BALANCE_MESSAGE = (
    "API error (status 402 Payment Required): "
    "Grok Build usage balance exhausted"
)
_GROK_INTERNAL_ERROR_PREFIX = "Internal error: "
_GROK_MODEL_ENVIRONMENT_NAMES = (
    "IPFS_ACCELERATE_AGENT_GROK_MODEL",
    "ipfs_accelerate_py_GROK_CLI_MODEL",
    "IPFS_ACCELERATE_PY_GROK_CLI_MODEL",
    "IPFS_DATASETS_PY_GROK_CLI_MODEL",
    "GROK_CLI_MODEL",
    "GROK_MODEL",
    "ipfs_accelerate_py_XAI_MODEL",
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
    if not isinstance(payload, list) or not 2 <= len(payload) <= MAX_CODEX_FALLBACK_ARGUMENTS:
        raise ValueError("Codex fallback command must be a bounded argv array")
    command: list[str] = []
    for item in payload:
        if (
            not isinstance(item, str)
            or not item
            or len(item.encode("utf-8")) > MAX_CODEX_FALLBACK_ARGUMENT_BYTES
        ):
            raise ValueError(
                "Codex fallback command contains an invalid argument"
            )
        command.append(item)
    if Path(command[0]).name.lower() not in {"codex", "codex.exe"}:
        raise ValueError("Codex fallback executable must be codex")
    if command[1] != "exec" or command[-1] != "-":
        raise ValueError("Codex fallback command must use `codex exec ... -`")
    model_positions = [
        index for index, argument in enumerate(command) if argument == "-m"
    ]
    if (
        len(model_positions) != 1
        or model_positions[0] + 1 >= len(command)
        or command[model_positions[0] + 1] != _CODEX_FALLBACK_MODEL
    ):
        raise ValueError(
            f"Codex fallback model must be {_CODEX_FALLBACK_MODEL}"
        )
    if any(
        argument == "--model" or argument.startswith("--model=")
        for argument in command
    ):
        raise ValueError("Codex fallback contains a conflicting model option")
    reasoning_overrides = [
        command[index + 1]
        for index, argument in enumerate(command[:-1])
        if argument == "-c"
        and command[index + 1].split("=", 1)[0].strip()
        == "model_reasoning_effort"
    ]
    expected_reasoning = f'model_reasoning_effort="{_CODEX_FALLBACK_REASONING}"'
    if reasoning_overrides != [expected_reasoning]:
        raise ValueError(
            "Codex fallback reasoning effort must be "
            f"{_CODEX_FALLBACK_REASONING}"
        )
    conflicting_model_overrides = [
        command[index + 1]
        for index, argument in enumerate(command[:-1])
        if argument == "-c"
        and command[index + 1].split("=", 1)[0].strip() == "model"
    ]
    if conflicting_model_overrides:
        raise ValueError("Codex fallback contains a conflicting model override")
    return command


def _structured_grok_quota_payload(raw_stderr: bytes) -> dict[str, object] | None:
    """Decode one bounded, wholly structured Grok error payload.

    Arbitrary prose and JSON fragments embedded in logs are intentionally not
    accepted.  Grok Build's current ``Internal error: {...}`` envelope is the
    sole non-JSON prefix supported, and only for its exact 402 balance error.
    """

    try:
        text = raw_stderr.decode("utf-8", errors="strict").strip()
    except UnicodeDecodeError:
        return None
    if not text:
        return None
    json_text = text
    if text.startswith(_GROK_INTERNAL_ERROR_PREFIX):
        json_text = text[len(_GROK_INTERNAL_ERROR_PREFIX) :].strip()
    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _payload_has_exact_quota_reason(payload: dict[str, object]) -> bool:
    candidates: list[dict[str, object]] = [payload]
    nested_error = payload.get("error")
    if isinstance(nested_error, dict):
        candidates.append(nested_error)
    return any(
        candidate.get(field) in _GROK_QUOTA_REASONS
        for candidate in candidates
        for field in ("code", "kind", "reason", "reason_code", "type")
    )


def _payload_is_exact_grok_build_402(payload: dict[str, object]) -> bool:
    candidates: list[dict[str, object]] = [payload]
    nested_error = payload.get("error")
    if isinstance(nested_error, dict):
        candidates.append(nested_error)
    return any(
        candidate.get("http_status") == 402
        and candidate.get("message") == _GROK_BUILD_BALANCE_MESSAGE
        for candidate in candidates
    )


def _stderr_allows_codex_fallback(
    raw_stderr: bytes,
    *,
    overflow: bool,
) -> bool:
    """Return true only for the reviewed Grok quota-exhaustion schemas."""

    if overflow or len(raw_stderr) > MAX_GROK_QUOTA_STDERR_BYTES:
        return False
    payload = _structured_grok_quota_payload(raw_stderr)
    if payload is None:
        return False
    if _payload_has_exact_quota_reason(payload):
        # The Grok prose envelope is allowed only for the exact Build 402
        # schema below, never as a carrier for an otherwise injectable reason.
        try:
            text = raw_stderr.decode("utf-8", errors="strict").lstrip()
        except UnicodeDecodeError:
            return False
        return not text.startswith(_GROK_INTERNAL_ERROR_PREFIX)
    return _payload_is_exact_grok_build_402(payload)


def _emit_grok_stderr(raw_stderr: bytes, *, overflow: bool) -> None:
    if raw_stderr:
        text = raw_stderr.decode("utf-8", errors="replace")
        sys.stderr.write(text)
        if not text.endswith("\n"):
            sys.stderr.write("\n")
    if overflow:
        print(
            "grok CLI stderr exceeded the quota-classification bound; "
            "Codex fallback is disabled",
            file=sys.stderr,
        )


def _run_grok_with_bounded_stderr(
    cmd: Sequence[str],
    *,
    env: dict[str, str],
) -> tuple[subprocess.CompletedProcess[bytes], bytes, bool]:
    """Run Grok while retaining at most one bounded stderr classification."""

    with tempfile.TemporaryFile(mode="w+b") as stderr_file:
        completed = subprocess.run(
            list(cmd),
            env=env,
            check=False,
            stderr=stderr_file,
        )
        stderr_size = stderr_file.tell()
        stderr_file.seek(0)
        raw_stderr = stderr_file.read(MAX_GROK_QUOTA_STDERR_BYTES + 1)
    overflow = stderr_size > MAX_GROK_QUOTA_STDERR_BYTES
    return completed, raw_stderr[:MAX_GROK_QUOTA_STDERR_BYTES], overflow


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
            "returns a verified quota-exhaustion reason; forced-Grok routes "
            "omit this option."
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

    grok_bin = str(args.grok_bin).strip() or find_grok_cli() or ""
    if not grok_bin:
        print("grok CLI not found on PATH", file=sys.stderr)
        return 127

    # The supervisor's reviewed primary is immutable.  ``--model`` remains a
    # compatibility argument for older launchers, but neither it nor ambient
    # environment variables may silently change the provider contract.
    model = DEFAULT_GROK_MODEL
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
        or os.environ.get(
            "ipfs_accelerate_py_GROK_CLI_PERMISSION_MODE", ""
        ).strip()
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
            for environment_name in _GROK_MODEL_ENVIRONMENT_NAMES:
                env.pop(environment_name, None)
        except LLMRouterError as exc:
            print(str(exc), file=sys.stderr)
            return 2

        os.chdir(workspace)
        try:
            completed, grok_stderr, stderr_overflow = (
                _run_grok_with_bounded_stderr(cmd, env=env)
            )
        except OSError as exc:
            print(f"unable to launch Grok CLI: {exc}", file=sys.stderr)
            return 127
        _emit_grok_stderr(grok_stderr, overflow=stderr_overflow)
        primary_returncode = int(completed.returncode)
        if primary_returncode == 0 or not codex_fallback_command:
            return primary_returncode
        if primary_returncode <= 0 or primary_returncode in {124, 137, 143}:
            return primary_returncode
        if not _stderr_allows_codex_fallback(
            grok_stderr,
            overflow=stderr_overflow,
        ):
            return primary_returncode

        print(
            "grok CLI reported verified quota exhaustion with exit "
            f"{primary_returncode}; falling back to codex",
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
