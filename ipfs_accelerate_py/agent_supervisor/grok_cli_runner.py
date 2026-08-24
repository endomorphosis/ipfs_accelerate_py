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
import threading
from collections.abc import Sequence
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))


DEFAULT_GROK_MODEL = "grok-4.5"
# Grok CLI validates --max-turns as 1..=4294967295 (u32::MAX).
DEFAULT_GROK_MAX_TURNS = 4_294_967_295
TRUSTED_FAILURE_RECEIPT_FD_ENV = "IPFS_ACCELERATE_AGENT_TRUSTED_FAILURE_RECEIPT_FD"
GROK_STREAM_FRAME_MAX_BYTES = 256 * 1024
GROK_ACCOUNT_QUOTA_CODES = frozenset({"usage_limit_reached", "usage_pool_exhausted"})


class _BoundedTerminalFrameParser:
    """Retain one exact final NDJSON frame without buffering provider output."""

    def __init__(self, max_frame_bytes: int = GROK_STREAM_FRAME_MAX_BYTES) -> None:
        self.max_frame_bytes = max_frame_bytes
        self.pending = bytearray()
        self.overlong = False
        self.tainted = False
        self.frame_count = 0
        self.last_event: dict[str, object] | None = None

    def _append(self, value: bytes) -> None:
        if self.overlong:
            return
        if len(self.pending) + len(value) > self.max_frame_bytes:
            self.pending.clear()
            self.overlong = True
            self.tainted = True
            return
        self.pending.extend(value)

    def _finish_line(self) -> None:
        if self.overlong:
            self.last_event = None
        else:
            raw = bytes(self.pending).strip()
            if raw:
                self.frame_count += 1
                try:
                    value = json.loads(raw.decode("utf-8", errors="strict"))
                except (UnicodeDecodeError, ValueError, RecursionError):
                    self.last_event = None
                    self.tainted = True
                else:
                    if isinstance(value, dict):
                        self.last_event = value
                    else:
                        self.last_event = None
                        self.tainted = True
        self.pending.clear()
        self.overlong = False

    def feed(self, chunk: str, *, final: bool = False) -> None:
        if "\ufffd" in chunk:
            self.tainted = True
        encoded = chunk.encode("utf-8")
        start = 0
        while True:
            newline = encoded.find(b"\n", start)
            if newline < 0:
                self._append(encoded[start:])
                break
            self._append(encoded[start:newline])
            self._finish_line()
            start = newline + 1
        if final and (self.pending or self.overlong):
            self._finish_line()


def _terminal_grok_quota_code(event: object) -> str:
    """Return only an exact machine quota code from a top-level error frame."""

    if not isinstance(event, dict) or event.get("type") != "error":
        return ""
    records = [event]
    nested = event.get("error")
    if isinstance(nested, dict):
        records.insert(0, nested)
    explicit_values: list[str] = []
    for record in records:
        for field in ("code", "errorCode", "error_code", "reason"):
            if field not in record:
                continue
            raw_value = record[field]
            if not isinstance(raw_value, str):
                return ""
            explicit_values.append(raw_value.strip().casefold())
    if explicit_values:
        if any(not value for value in explicit_values):
            return ""
        distinct = set(explicit_values)
        if len(distinct) != 1:
            return ""
        [selected] = distinct
        return selected if selected in GROK_ACCOUNT_QUOTA_CODES else ""

    message_codes: set[str] = set()
    for record in records:
        if "message" not in record:
            continue
        message = record["message"]
        if not isinstance(message, str):
            return ""
        normalized = message.strip().casefold()
        if normalized not in GROK_ACCOUNT_QUOTA_CODES:
            return ""
        message_codes.add(normalized)
    return next(iter(message_codes)) if len(message_codes) == 1 else ""


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
        "streaming-json",
        "--prompt-file",
        str(prompt_file),
    ]
    return cmd


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
        "--require-terminal-quota-frame",
        action="store_true",
        help=(
            "mint quota failure only from one exact terminal streaming-JSON "
            "error frame (internal ordered-route contract)"
        ),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    from ipfs_accelerate_py.llm_router import (
        AgentCLIActivityState,
        AgentCLIFailureClassification,
        AgentCLIProviderFailureKind,
        AgentCLIProviderResult,
        AgentCLIStderrSanitizer,
        LLMRouterError,
        build_grok_cli_command,
        build_grok_cli_env,
        classify_grok_agent_cli_failure,
        find_grok_cli,
        serialize_agent_cli_failure_receipt,
    )

    trusted_receipt_fd = -1
    raw_receipt_fd = os.environ.pop(TRUSTED_FAILURE_RECEIPT_FD_ENV, "").strip()
    if raw_receipt_fd:
        try:
            candidate_fd = int(raw_receipt_fd)
            if candidate_fd >= 3:
                os.fstat(candidate_fd)
                trusted_receipt_fd = candidate_fd
        except (OSError, ValueError):
            trusted_receipt_fd = -1

    def emit_failure_receipt(
        classification: AgentCLIFailureClassification,
        *,
        returncode: int | None,
        activity_state: AgentCLIActivityState,
    ) -> None:
        nonlocal trusted_receipt_fd
        if trusted_receipt_fd < 0:
            return
        receipt = serialize_agent_cli_failure_receipt(
            classification,
            returncode=returncode,
            activity_state=activity_state,
        ).encode("utf-8")
        try:
            os.write(trusted_receipt_fd, receipt)
        except OSError:
            pass
        finally:
            try:
                os.close(trusted_receipt_fd)
            except OSError:
                pass
            trusted_receipt_fd = -1

    workspace = args.workspace.expanduser().resolve()
    if not workspace.is_dir():
        print(f"workspace is not a directory: {workspace}", file=sys.stderr)
        return 2

    grok_bin = str(args.grok_bin).strip() or find_grok_cli() or ""
    if not grok_bin:
        print("grok CLI not found on PATH", file=sys.stderr)
        emit_failure_receipt(
            AgentCLIFailureClassification(
                AgentCLIProviderFailureKind.LAUNCH_FAILURE,
                "grok_cli_unavailable",
            ),
            returncode=127,
            activity_state=AgentCLIActivityState.PRE_DISPATCH,
        )
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
            emit_failure_receipt(
                AgentCLIFailureClassification(
                    AgentCLIProviderFailureKind.MALFORMED_OUTPUT,
                    "grok_agent_command_invalid",
                ),
                returncode=2,
                activity_state=AgentCLIActivityState.PRE_DISPATCH,
            )
            return 2

        os.chdir(workspace)
        sanitizer = AgentCLIStderrSanitizer()
        try:
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                close_fds=True,
            )
        except OSError as exc:
            sanitized = (
                sanitizer.feed(f"Grok provider could not launch: {exc}\n") + sanitizer.finish()
            )
            sys.stderr.write(sanitized)
            sys.stderr.flush()
            emit_failure_receipt(
                AgentCLIFailureClassification(
                    AgentCLIProviderFailureKind.LAUNCH_FAILURE,
                    "grok_process_did_not_launch",
                ),
                returncode=127,
                activity_state=AgentCLIActivityState.PRE_DISPATCH,
            )
            return 127
        assert process.stdout is not None
        assert process.stderr is not None
        stderr_tail = ""
        stdout_seen = False
        terminal_parser = _BoundedTerminalFrameParser()

        def replay_stdout() -> None:
            nonlocal stdout_seen
            while True:
                chunk = process.stdout.read(8192)
                if not chunk:
                    terminal_parser.feed("", final=True)
                    return
                stdout_seen = True
                terminal_parser.feed(chunk)
                sys.stdout.write(chunk)
                sys.stdout.flush()

        def replay_stderr() -> None:
            nonlocal stderr_tail
            while True:
                chunk = process.stderr.read(8192)
                if not chunk:
                    final = sanitizer.finish()
                    if final:
                        sys.stderr.write(final)
                        sys.stderr.flush()
                        stderr_tail = (stderr_tail + final)[-(256 * 1024) :]
                    return
                sanitized = sanitizer.feed(chunk)
                if sanitized:
                    sys.stderr.write(sanitized)
                    sys.stderr.flush()
                    stderr_tail = (stderr_tail + sanitized)[-(256 * 1024) :]

        stdout_thread = threading.Thread(target=replay_stdout, daemon=True)
        stderr_thread = threading.Thread(target=replay_stderr, daemon=True)
        stdout_thread.start()
        stderr_thread.start()
        returncode = int(process.wait())
        stdout_thread.join()
        stderr_thread.join()
        if returncode != 0:
            quota_code = ""
            if (
                args.require_terminal_quota_frame
                and not terminal_parser.tainted
                and terminal_parser.frame_count == 1
            ):
                quota_code = _terminal_grok_quota_code(terminal_parser.last_event)
            if quota_code:
                activity = AgentCLIActivityState.NO_ACTIVITY
                classification = AgentCLIFailureClassification(
                    AgentCLIProviderFailureKind.GROK_QUOTA_EXHAUSTED,
                    f"grok_provider_{quota_code}",
                )
            else:
                activity = (
                    AgentCLIActivityState.UNKNOWN
                    if stdout_seen
                    else AgentCLIActivityState.NO_ACTIVITY
                )
                classification = classify_grok_agent_cli_failure(
                    AgentCLIProviderResult(
                        returncode,
                        stderr=stderr_tail,
                        launched=True,
                        activity_state=activity,
                    )
                )
                if (
                    args.require_terminal_quota_frame
                    and classification.kind is AgentCLIProviderFailureKind.GROK_QUOTA_EXHAUSTED
                ):
                    classification = AgentCLIFailureClassification(
                        AgentCLIProviderFailureKind.GENERIC_NONZERO_EXIT,
                        "grok_terminal_quota_unverified",
                    )
            emit_failure_receipt(
                classification,
                returncode=returncode,
                activity_state=activity,
            )
        return returncode
    finally:
        if prompt_path:
            try:
                os.unlink(prompt_path)
            except OSError:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
