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
CAPACITY_RESULT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/grok-cli-capacity-result@1"
)
CAPACITY_RESULT_REASON_CODES = ("capacity_unavailable", "quota_exhausted")
# The wrapper, not the provider subprocess, owns this status.  A non-quota
# child collision is remapped below, so a sidecar file alone cannot authorize
# the daemon's quota route.
HARD_QUOTA_EXIT_CODE = 75
_CAPACITY_DIAGNOSTIC_TAIL_BYTES = 256 * 1024
_HARD_QUOTA_LINE_PATTERNS = (
    re.compile(
        r"^\s*(?:error|fatal)(?:\s*[:\]]|\s+).*"
        r"\binsufficient_quota\b",
        re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(
        r"^\s*(?:error|fatal)\s*:\s*you(?:'|\u2019)?ve hit your usage limit\s*[.!]?\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(
        r"^\s*(?:error|fatal)\s*:\s*.*\b(?:quota (?:has been )?exceeded|"
        r"quota exhausted|(?:usage )?balance exhausted)\b",
        re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(
        r"^\s*(?:error\s*:\s*)?(?:http\s*)?402\s+payment required\b",
        re.IGNORECASE | re.MULTILINE,
    ),
)


def _is_hard_quota_exhaustion(stderr_text: str) -> bool:
    """Return true only for durable quota/balance exhaustion from Grok stderr."""

    lowered = stderr_text.lower()
    transient_markers = (
        "429",
        "rate limit",
        "rate_limit",
        "too many requests",
        "resource exhausted",
        "resource_exhausted",
    )
    if any(marker in lowered for marker in transient_markers):
        return False
    return any(pattern.search(stderr_text) for pattern in _HARD_QUOTA_LINE_PATTERNS)


def _replace_capacity_result(path: Path, payload: dict[str, object]) -> None:
    """Atomically publish a bounded, prompt-free runner result."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=str(path.parent),
            delete=False,
        ) as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            temporary_path = handle.name
        os.replace(temporary_path, path)
        temporary_path = ""
    finally:
        if temporary_path:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass


def _remove_capacity_result(path: Optional[Path]) -> None:
    if path is None:
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass


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
        "--capacity-result-path",
        type=Path,
        default=None,
        help="Daemon-owned path for a typed hard-quota result (never prompt data).",
    )
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
    capacity_result_path = (
        Path(os.path.abspath(os.path.expanduser(str(args.capacity_result_path))))
        if args.capacity_result_path is not None
        else None
    )
    _remove_capacity_result(capacity_result_path)

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
        diagnostic = hashlib.sha256()
        diagnostic_tail = b""
        if capacity_result_path is None:
            completed = subprocess.run(cmd, env=env, check=False)
            return int(completed.returncode)

        process = subprocess.Popen(cmd, env=env, stderr=subprocess.PIPE)
        assert process.stderr is not None
        try:
            while True:
                chunk = process.stderr.read1(64 * 1024)
                if not chunk:
                    break
                diagnostic.update(chunk)
                diagnostic_tail = (diagnostic_tail + chunk)[
                    -_CAPACITY_DIAGNOSTIC_TAIL_BYTES:
                ]
                sys.stderr.write(chunk.decode("utf-8", errors="replace"))
                sys.stderr.flush()
        finally:
            process.stderr.close()
        provider_returncode = int(process.wait())
        hard_quota = provider_returncode != 0 and _is_hard_quota_exhaustion(
            diagnostic_tail.decode("utf-8", errors="replace")
        )
        # A child process cannot authorize fallback by writing this path: the
        # runner removes/replaces it only after the child has terminated.
        _remove_capacity_result(capacity_result_path)
        if hard_quota and capacity_result_path is not None:
            _replace_capacity_result(
                capacity_result_path,
                {
                    "diagnostic_sha256": diagnostic.hexdigest(),
                    "model": model,
                    "provider": "grok_cli",
                    "provider_returncode": provider_returncode,
                    "reason": "provider_quota_exhausted",
                    "reason_codes": list(CAPACITY_RESULT_REASON_CODES),
                    "returncode": HARD_QUOTA_EXIT_CODE,
                    "schema": CAPACITY_RESULT_SCHEMA,
                },
            )
            return HARD_QUOTA_EXIT_CODE
        # Reserve HARD_QUOTA_EXIT_CODE for a runner-verified hard-quota result.
        # Ordinary provider failures retain their status except for this one
        # collision, which remains an ordinary consuming failure.
        if provider_returncode == HARD_QUOTA_EXIT_CODE:
            return 1
        return provider_returncode
    finally:
        if prompt_path:
            try:
                os.unlink(prompt_path)
            except OSError:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
