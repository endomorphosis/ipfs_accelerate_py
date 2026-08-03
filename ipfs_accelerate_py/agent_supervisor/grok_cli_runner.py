#!/usr/bin/env python3
"""Thin process entry: Grok Build CLI agent for implementation worktrees.

Reads the implementation prompt from stdin (daemon contract), writes it to a
temp prompt file, then execs the official ``grok`` binary with agent-capable
flags. Command policy lives next to other CLI peers in :mod:`llm_router`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Sequence

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))


DEFAULT_GROK_MODEL = "grok-4.5"
# Grok CLI validates --max-turns as 1..=4294967295 (u32::MAX).
DEFAULT_GROK_MAX_TURNS = 4_294_967_295
GROK_QUOTA_EXHAUSTED_EXIT_CODE = 86
GROK_QUOTA_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/grok-quota-error@1"
)
MAX_GROK_ERROR_BYTES = 128 * 1024
_GROK_USAGE_LIMIT_PATTERN = re.compile(
    r"\A\s*(?:error:\s*)?you(?:'|\u2019)?ve\s+hit\s+your\s+usage\s+limit\.?"
    r"(?:\s*\n\s*try\s+again\s+at\s+[^\n]+\.?)?\s*\Z",
    re.IGNORECASE,
)
_GROK_BALANCE_MESSAGE = (
    "API error (status 402 Payment Required): "
    "Grok Build usage balance exhausted"
)


def parse_grok_quota_error(text: str) -> dict[str, object]:
    """Parse only complete, known Grok quota error envelopes."""

    stripped = text.strip()
    if _GROK_USAGE_LIMIT_PATTERN.fullmatch(stripped):
        return {"kind": "usage_limit", "http_status": None}
    lowered = stripped.lower()
    prefixes = ("internal error:", "error:")
    prefix = next((item for item in prefixes if lowered.startswith(item)), "")
    if not prefix:
        return {}
    payload_text = stripped[len(prefix) :].strip()
    try:
        payload = json.loads(payload_text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}
    if not isinstance(payload, dict) or set(payload) != {"message", "http_status"}:
        return {}
    status = payload.get("http_status")
    message = payload.get("message")
    if (
        isinstance(status, bool)
        or not isinstance(status, int)
        or status != 402
        or not isinstance(message, str)
        or " ".join(message.split()) != _GROK_BALANCE_MESSAGE
    ):
        return {}
    return {"kind": "usage_balance_exhausted", "http_status": 402}


def _run_grok_with_bounded_stderr(
    command: Sequence[str],
    *,
    env: dict[str, str],
) -> tuple[int, bytes, int, bool]:
    """Drain child stderr without unbounded memory or disk growth."""

    process = subprocess.Popen(
        list(command),
        env=env,
        stderr=subprocess.PIPE,
    )
    assert process.stderr is not None
    retained = bytearray()
    total = 0
    while True:
        chunk = process.stderr.read(64 * 1024)
        if not chunk:
            break
        total += len(chunk)
        remaining = MAX_GROK_ERROR_BYTES - len(retained)
        if remaining > 0:
            retained.extend(chunk[:remaining])
    process.stderr.close()
    returncode = int(process.wait())
    return returncode, bytes(retained), total, total > MAX_GROK_ERROR_BYTES


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


def main(argv: Optional[Sequence[str]] = None) -> int:
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
    args = parser.parse_args(list(argv) if argv is not None else None)

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

    model = (
        str(args.model).strip()
        or os.environ.get("IPFS_ACCELERATE_AGENT_GROK_MODEL", "").strip()
        or os.environ.get("ipfs_accelerate_py_GROK_CLI_MODEL", "").strip()
        or os.environ.get("GROK_CLI_MODEL", "").strip()
        or DEFAULT_GROK_MODEL
    )
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
        except LLMRouterError as exc:
            print(str(exc), file=sys.stderr)
            return 2

        os.chdir(workspace)
        child_returncode, error_bytes, error_size, error_overflow = (
            _run_grok_with_bounded_stderr(cmd, env=env)
        )
        if error_bytes:
            sys.stderr.buffer.write(error_bytes)
            if not error_bytes.endswith(b"\n"):
                sys.stderr.buffer.write(b"\n")
            sys.stderr.buffer.flush()
        if error_overflow:
            print(
                "grok stderr exceeded the trusted quota-envelope limit "
                f"({error_size} > {MAX_GROK_ERROR_BYTES} bytes); "
                "quota fallback forbidden",
                file=sys.stderr,
            )
            return (
                1
                if child_returncode == GROK_QUOTA_EXHAUSTED_EXIT_CODE
                else child_returncode
            )
        quota_error = parse_grok_quota_error(
            error_bytes.decode("utf-8", errors="replace")
        )
        if child_returncode != 0 and quota_error:
            receipt = {
                "schema": GROK_QUOTA_RECEIPT_SCHEMA,
                "provider": "grok_cli",
                "model": model,
                "failure_kind": "quota_or_balance_exhausted",
                "message": "Grok Build usage balance exhausted",
                "raw_error_sha256": hashlib.sha256(error_bytes).hexdigest(),
                "raw_error_size": len(error_bytes),
                **quota_error,
            }
            print(
                json.dumps(receipt, sort_keys=True, separators=(",", ":")),
                file=sys.stderr,
            )
            return GROK_QUOTA_EXHAUSTED_EXIT_CODE
        return (
            1
            if child_returncode == GROK_QUOTA_EXHAUSTED_EXIT_CODE
            else child_returncode
        )
    finally:
        if prompt_path:
            try:
                os.unlink(prompt_path)
            except OSError:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
