#!/usr/bin/env python3
"""Thin process entry: Grok Build CLI agent for implementation worktrees.

Reads the implementation prompt from stdin (daemon contract), writes it to a
temp prompt file, then execs the official ``grok`` binary with agent-capable
flags. Command policy lives next to other CLI peers in :mod:`llm_router`.
"""

from __future__ import annotations

import argparse
import codecs
import json
import os
import selectors
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))


DEFAULT_GROK_MODEL = "grok-4.5"
# Grok CLI validates --max-turns as 1..=4294967295 (u32::MAX).
DEFAULT_GROK_MAX_TURNS = 4_294_967_295
MAX_GROK_STREAM_EVENT_BYTES = 64 * 1024
GROK_STREAM_READ_BYTES = 16 * 1024
GROK_STREAM_PROCESS_GROUP_GRACE_SECONDS = 0.1
GROK_QUOTA_FAILURE_MARKER = "verified_balance_exhausted"
GROK_QUOTA_ERROR_TYPES = frozenset({GROK_QUOTA_FAILURE_MARKER})
_LEGACY_GROK_BALANCE_EXHAUSTED_MESSAGE = (
    "API error (status 402 Payment Required): " "Grok Build usage balance exhausted"
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


def _grok_failure_type_from_stream_event(line: str) -> str:
    """Project one CLI-owned structured failure event, never model text.

    Grok 0.2.118 documents ``streaming-json`` as top-level, type-tagged events
    derived from ACP session updates.  Its persisted native ACP event is also
    accepted for compatibility.  In both shapes only the exact, locally
    observed 402 balance-exhaustion message is classified as quota evidence;
    provider enum names found in unrelated tool-error schemas are deliberately
    insufficient.  This raw runner never dispatches a fallback provider.
    """

    if (
        not line
        or len(line.encode("utf-8", errors="replace")) > MAX_GROK_STREAM_EVENT_BYTES
    ):
        return ""
    try:
        payload = json.loads(line)
    except (json.JSONDecodeError, TypeError):
        return ""
    if not isinstance(payload, dict):
        return ""
    if payload.get("type") == "error":
        if str(payload.get("message") or "").strip() == (
            _LEGACY_GROK_BALANCE_EXHAUSTED_MESSAGE
        ):
            return GROK_QUOTA_FAILURE_MARKER
        return "structured_error"
    if payload.get("method") not in {
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
    if (
        error_type == "api"
        and str(update.get("message") or "").strip()
        == _LEGACY_GROK_BALANCE_EXHAUSTED_MESSAGE
    ):
        return GROK_QUOTA_FAILURE_MARKER
    return error_type or "unknown"


def _write_stream_text(destination: TextIO, value: str) -> None:
    """Best-effort live tee which never stops draining a provider pipe."""

    if not value:
        return
    try:
        destination.write(value)
        destination.flush()
    except (BrokenPipeError, OSError, ValueError):
        # The provider must not deadlock merely because the runner's caller
        # closed or rejected its own stdout/stderr destination.
        pass


def _inspect_grok_stream_text(
    value: str,
    failure_types: set[str],
    state: dict[str, object],
    *,
    final: bool = False,
) -> None:
    """Inspect bounded, complete stdout JSON lines across arbitrary byte chunks."""

    pending = str(state.get("pending") or "") + value
    discarding = bool(state.get("discarding_oversized_line", False))
    while "\n" in pending:
        line, pending = pending.split("\n", 1)
        if not discarding:
            failure_type = _grok_failure_type_from_stream_event(line)
            if failure_type:
                failure_types.add(failure_type)
        discarding = False
    if len(pending.encode("utf-8", errors="replace")) > MAX_GROK_STREAM_EVENT_BYTES:
        pending = ""
        discarding = True
    if final and pending and not discarding:
        failure_type = _grok_failure_type_from_stream_event(pending)
        if failure_type:
            failure_types.add(failure_type)
        pending = ""
    state["pending"] = pending
    state["discarding_oversized_line"] = discarding


def _terminate_grok_process_group(
    process: subprocess.Popen[bytes],
    *,
    graceful: bool = True,
) -> None:
    """Best-effort cleanup for provider descendants owned by this invocation."""

    if os.name != "posix":
        if process.poll() is None:
            try:
                process.terminate()
            except OSError:
                pass
        return
    process_group = int(process.pid)
    if not graceful:
        try:
            os.killpg(process_group, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
        return
    try:
        os.killpg(process_group, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        return
    deadline = time.monotonic() + GROK_STREAM_PROCESS_GROUP_GRACE_SECONDS
    while time.monotonic() < deadline:
        try:
            os.killpg(process_group, 0)
        except (ProcessLookupError, PermissionError, OSError):
            return
        time.sleep(0.01)
    try:
        os.killpg(process_group, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _run_grok_with_typed_failure_capture(
    command: Sequence[str],
    *,
    env: dict[str, str],
) -> tuple[int, set[str]]:
    """Run Grok with bounded live drain and native structured failure types.

    The child owns a fresh process group.  Output is drained with nonblocking
    selectors instead of reader threads, so a daemonized descendant retaining
    a pipe descriptor cannot make the runner wait forever after Grok exits.
    """

    process = subprocess.Popen(
        list(command),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
        start_new_session=(os.name == "posix"),
    )
    assert process.stdout is not None
    assert process.stderr is not None
    failure_types: set[str] = set()
    selector = selectors.DefaultSelector()
    streams: dict[int, dict[str, object]] = {}
    for source, destination, inspect_failures in (
        (process.stdout, sys.stdout, True),
        (process.stderr, sys.stderr, False),
    ):
        descriptor = source.fileno()
        os.set_blocking(descriptor, False)
        streams[descriptor] = {
            "source": source,
            "destination": destination,
            "inspect_failures": inspect_failures,
            "decoder": codecs.getincrementaldecoder("utf-8")(errors="replace"),
            "inspection": {
                "pending": "",
                "discarding_oversized_line": False,
            },
        }
        selector.register(descriptor, selectors.EVENT_READ)

    returncode: int | None = None
    def close_stream(descriptor: int, *, final: bool) -> None:
        stream = streams.pop(descriptor, None)
        if stream is None:
            return
        try:
            selector.unregister(descriptor)
        except (KeyError, OSError, ValueError):
            pass
        decoder = stream["decoder"]
        assert isinstance(decoder, codecs.IncrementalDecoder)
        tail = decoder.decode(b"", final=final)
        destination = stream["destination"]
        _write_stream_text(destination, tail)  # type: ignore[arg-type]
        if stream["inspect_failures"]:
            inspection = stream["inspection"]
            assert isinstance(inspection, dict)
            _inspect_grok_stream_text(
                tail,
                failure_types,
                inspection,
                final=final,
            )
        source = stream["source"]
        try:
            source.close()  # type: ignore[union-attr]
        except (OSError, ValueError):
            pass

    def consume_chunk(stream: dict[str, object], chunk: bytes) -> None:
        decoder = stream["decoder"]
        assert isinstance(decoder, codecs.IncrementalDecoder)
        decoded = decoder.decode(chunk, final=False)
        destination = stream["destination"]
        _write_stream_text(destination, decoded)  # type: ignore[arg-type]
        if stream["inspect_failures"]:
            inspection = stream["inspection"]
            assert isinstance(inspection, dict)
            _inspect_grok_stream_text(decoded, failure_types, inspection)

    def fence_and_drain_buffered_output() -> None:
        """Kill descendants, drain only bytes already buffered, then close."""

        _terminate_grok_process_group(process, graceful=False)
        for descriptor in tuple(streams):
            stream = streams.get(descriptor)
            if stream is None:
                continue
            while True:
                try:
                    chunk = os.read(descriptor, GROK_STREAM_READ_BYTES)
                except BlockingIOError:
                    break
                except OSError:
                    chunk = b""
                if not chunk:
                    break
                consume_chunk(stream, chunk)
            # A setsid-escaped descendant may still own the write end. Closing
            # now prevents post-parent output from being misattributed to the
            # direct provider in diagnostic evidence.
            close_stream(descriptor, final=True)

    try:
        while streams or returncode is None:
            if returncode is None:
                observed = process.poll()
                if observed is not None:
                    returncode = int(observed)
                    fence_and_drain_buffered_output()
                    continue

            for key, _mask in selector.select(timeout=0.05):
                descriptor = int(key.fd)
                stream = streams.get(descriptor)
                if stream is None:
                    continue
                try:
                    chunk = os.read(descriptor, GROK_STREAM_READ_BYTES)
                except BlockingIOError:
                    continue
                if not chunk:
                    close_stream(descriptor, final=True)
                    continue
                consume_chunk(stream, chunk)

        if returncode is None:
            returncode = int(process.wait())
        # A provider process may have left same-session descendants after
        # closing its output. Such descendants are outside the completed call.
        _terminate_grok_process_group(process)
        return returncode, failure_types
    except BaseException:
        _terminate_grok_process_group(process)
        if process.poll() is None:
            try:
                process.kill()
            except OSError:
                pass
        raise
    finally:
        for descriptor in tuple(streams):
            close_stream(descriptor, final=True)
        selector.close()


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
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if str(args.codex_fallback_command_json).strip():
        print(
            "raw provider fallback is disabled; use the typed production "
            "packet route",
            file=sys.stderr,
        )
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

    requested_model = str(args.model).strip() or DEFAULT_GROK_MODEL
    if requested_model != DEFAULT_GROK_MODEL:
        print(
            f"implementation routing requires {DEFAULT_GROK_MODEL}",
            file=sys.stderr,
        )
        return 2
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
        try:
            completed = subprocess.run(cmd, env=env, check=False)
        except OSError as exc:
            print(f"unable to launch Grok CLI: {exc}", file=sys.stderr)
            return 127
        return int(completed.returncode)
    finally:
        if prompt_path:
            try:
                os.unlink(prompt_path)
            except OSError:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
