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
from collections.abc import Sequence
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from ipfs_accelerate_py.agent_supervisor.provider_failure_policy import (
    GROK_FAILURE_RECEIPT_PREFIX,
    GROK_QUOTA_PROBE_PROMPT,
    GROK_QUOTA_PROBE_TIMEOUT_SECONDS,
    MAX_GROK_FAILURE_EVIDENCE_BYTES,
    build_grok_failure_receipt,
    render_grok_failure_receipt,
)

DEFAULT_GROK_MODEL = "grok-4.5"
# Grok CLI validates --max-turns as 1..=4294967295 (u32::MAX).
DEFAULT_GROK_MAX_TURNS = 4_294_967_295
MAX_CODEX_FALLBACK_ARGUMENTS = 64
MAX_CODEX_FALLBACK_ARGUMENT_BYTES = 4_096


def _resolve_grok_bin(configured: str = "") -> str:
    if configured.strip():
        path = Path(configured).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    try:
        from ipfs_accelerate_py.llm_router import _grok_cli_command

        candidate = str(_grok_cli_command() or "").strip()
        if candidate:
            found = shutil.which(candidate) or (candidate if Path(candidate).is_file() else "")
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
            raise ValueError("Codex fallback command contains an invalid argument")
        command.append(item)
    if Path(command[0]).name.lower() not in {"codex", "codex.exe"}:
        raise ValueError("Codex fallback executable must be codex")
    if command[1] != "exec" or command[-1] != "-":
        raise ValueError("Codex fallback command must use `codex exec ... -`")
    return command


def _run_grok_with_stderr_probe(
    command: Sequence[str],
    *,
    env: dict[str, str],
) -> tuple[int, str]:
    """Run task Grok while escaping receipt-like child output.

    The runner's own receipt line is a control-plane record. Child stdout and
    stderr share this filtered data path so neither can imitate that prefix.
    """

    process = subprocess.Popen(
        list(command),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    tail = bytearray()
    assert process.stdout is not None
    receipt_prefix = GROK_FAILURE_RECEIPT_PREFIX.encode("utf-8")
    at_line_start = True
    while True:
        chunk = process.stdout.readline(4096)
        if not chunk:
            break
        if at_line_start and chunk.startswith(receipt_prefix):
            chunk = b"[grok-child-output-escaped] " + chunk
        at_line_start = chunk.endswith(b"\n")
        sink = getattr(sys.stdout, "buffer", None)
        if sink is not None:
            sink.write(chunk)
            sink.flush()
        else:
            sys.stdout.write(chunk.decode("utf-8", errors="replace"))
            sys.stdout.flush()
        tail.extend(chunk)
        if len(tail) > MAX_GROK_FAILURE_EVIDENCE_BYTES:
            del tail[:-MAX_GROK_FAILURE_EVIDENCE_BYTES]
    return int(process.wait()), tail.decode("utf-8", errors="replace")


def _run_isolated_grok_quota_probe(
    command: Sequence[str],
    *,
    env: dict[str, str],
) -> tuple[int, str]:
    """Run the fixed no-tools quota probe without exposing task context."""

    try:
        completed = subprocess.run(
            list(command),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=False,
            timeout=GROK_QUOTA_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return 124, "isolated Grok quota probe timeout"
    stderr = bytes(completed.stderr or b"")
    return (
        int(completed.returncode),
        stderr[-MAX_GROK_FAILURE_EVIDENCE_BYTES:].decode(
            "utf-8",
            errors="replace",
        ),
    )


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
            "Deprecated internal Codex argv accepted for compatibility. "
            "Inline cross-provider fallback is forbidden; the supervisor "
            "may route a later attempt only after persisting a Grok quota "
            "exhaustion latch."
        ),
    )
    parser.add_argument(
        "--grok-failure-receipt-nonce",
        default="",
        help="Internal 256-bit nonce binding a runner-owned failure receipt.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        codex_fallback_command = _parse_codex_fallback_command(
            str(args.codex_fallback_command_json)
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    failure_receipt_nonce = str(args.grok_failure_receipt_nonce).strip()
    if failure_receipt_nonce and not re.fullmatch(
        r"[0-9a-f]{64}",
        failure_receipt_nonce,
    ):
        print("Grok failure receipt nonce must be 64 lowercase hex digits", file=sys.stderr)
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
        if failure_receipt_nonce:
            with tempfile.TemporaryDirectory(
                prefix="ipfs-accelerate-grok-quota-probe-"
            ) as probe_directory:
                probe_root = Path(probe_directory)
                probe_prompt_path = probe_root / "prompt.txt"
                probe_prompt_path.write_text(
                    GROK_QUOTA_PROBE_PROMPT,
                    encoding="utf-8",
                )
                try:
                    probe_command = build_grok_cli_command(
                        mode="chat",
                        workspace=probe_root,
                        model_name=model,
                        max_turns=1,
                        grok_bin=grok_bin,
                        prompt_file=probe_prompt_path,
                        permission_mode="dontAsk",
                        tools="",
                    )
                except LLMRouterError as exc:
                    print(str(exc), file=sys.stderr)
                    return 2
                probe_returncode, probe_stderr = _run_isolated_grok_quota_probe(
                    probe_command,
                    env=env,
                )
            if probe_returncode != 0:
                receipt = build_grok_failure_receipt(
                    probe_stderr_text=probe_stderr,
                    nonce=failure_receipt_nonce,
                    model=model,
                    probe_returncode=probe_returncode,
                    primary_dispatched=False,
                )
                print(render_grok_failure_receipt(receipt), file=sys.stderr)
                return probe_returncode
            primary_returncode, _stderr_tail = _run_grok_with_stderr_probe(
                cmd,
                env=env,
            )
        else:
            completed = subprocess.run(cmd, env=env, check=False)
            primary_returncode = int(completed.returncode)
        if primary_returncode != 0 and codex_fallback_command:
            print(
                "inline Codex fallback suppressed; the supervisor requires "
                "a durable Grok quota-exhaustion latch before routing "
                "gpt-5.6-terra",
                file=sys.stderr,
            )
        return primary_returncode
    finally:
        if prompt_path:
            try:
                os.unlink(prompt_path)
            except OSError:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
