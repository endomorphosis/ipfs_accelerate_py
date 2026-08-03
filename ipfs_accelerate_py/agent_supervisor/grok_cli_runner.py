#!/usr/bin/env python3
"""Supervised Grok Build CLI entry for implementation worktrees.

The runner keeps ordinary Grok output live while parsing only top-level
``streaming-json`` frames. A terminal, typed account-quota error is projected
as an untrusted candidate over a file descriptor not directly inherited by
Grok. Same-UID descendants can still inject into the Grok stdout pipe through
procfs, so exit 86 and this candidate are diagnostics, never fallback proof.
The daemon's independently signed quota verifier is the authority root.
"""

from __future__ import annotations

import argparse
import codecs
import fcntl
import hashlib
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import stat
from pathlib import Path
from typing import Optional, Sequence

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from ipfs_accelerate_py.agent_supervisor.provider_command_environment import (
    PROVIDER_COMMAND_ENV_DIGEST_ENV,
    PROVIDER_COMMAND_ENV_WRAPPER_ENV,
    PROVIDER_COMMAND_REQUIRED_COMMANDS_ENV,
    ProviderCommandEnvironmentError,
    sealed_provider_command_environment,
)
from ipfs_accelerate_py.agent_supervisor.validation.validation_runtime import (
    FORMAL_TOOLCHAIN_CONTRACT_SHA256_ENV,
    ValidationRuntimeError,
)


DEFAULT_GROK_MODEL = "grok-4.5"
# Grok CLI validates --max-turns as 1..=4294967295 (u32::MAX).
DEFAULT_GROK_MAX_TURNS = 4_294_967_295
GROK_QUOTA_EXHAUSTED_EXIT_CODE = 86
GROK_TERMINAL_QUOTA_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/grok-terminal-quota-receipt@1"
)
GROK_TERMINAL_QUOTA_RECEIPT_PREFIX = (
    "IPFS_ACCELERATE_GROK_TERMINAL_QUOTA_RECEIPT "
)
GROK_TERMINAL_RECEIPT_FD_ENV = (
    "IPFS_ACCELERATE_GROK_TERMINAL_RECEIPT_FD"
)
GROK_INVOCATION_BINDING_FLAG = "--invocation-binding-sha256"
GROK_INVOCATION_ID_FLAG = "--invocation-id"
GROK_STREAM_FRAME_MAX_BYTES = 256 * 1024
GROK_TERMINAL_RECEIPT_MAX_BYTES = 4096
GROK_ACCOUNT_QUOTA_CODES = frozenset(
    {"usage_limit_reached", "usage_pool_exhausted"}
)


def grok_command_sha256(command: Sequence[str]) -> str:
    """Return a stable digest of one exact argv vector."""

    payload = json.dumps(
        [str(item) for item in command],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def bind_grok_runner_command(command: Sequence[str]) -> list[str]:
    """Append a unique, self-verifying outer-runner invocation binding."""

    values = [str(item) for item in command]
    if (
        GROK_INVOCATION_BINDING_FLAG in values
        or GROK_INVOCATION_ID_FLAG in values
    ):
        raise ValueError("Grok runner command already has an invocation binding")
    values.extend((GROK_INVOCATION_ID_FLAG, secrets.token_hex(16)))
    return [
        *values,
        GROK_INVOCATION_BINDING_FLAG,
        grok_command_sha256(values),
    ]


def validate_grok_runner_command_binding(command: Sequence[str]) -> str:
    """Return the verified outer-runner binding or an empty string."""

    values = [str(item) for item in command]
    invocation_indexes = [
        index
        for index, item in enumerate(values)
        if item == GROK_INVOCATION_ID_FLAG
    ]
    binding_indexes = [
        index
        for index, item in enumerate(values)
        if item == GROK_INVOCATION_BINDING_FLAG
    ]
    if len(invocation_indexes) != 1 or len(binding_indexes) != 1:
        return ""
    invocation_index = invocation_indexes[0]
    binding_index = binding_indexes[0]
    if invocation_index + 1 >= len(values) or binding_index + 1 >= len(values):
        return ""
    if not re.fullmatch(r"[0-9a-f]{32}", values[invocation_index + 1]):
        return ""
    binding = values[binding_index + 1]
    if not re.fullmatch(r"[0-9a-f]{64}", binding):
        return ""
    unsigned = values[:binding_index] + values[binding_index + 2 :]
    return binding if binding == grok_command_sha256(unsigned) else ""


def grok_terminal_quota_code(event: object) -> str:
    """Return an exact account-quota code from a top-level error frame."""

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
        distinct = {value for value in explicit_values if value}
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
        # Grok CLI 0.2.x can surface the machine code in ``message`` instead
        # of a dedicated code field.  Accept only the whole stripped machine
        # value.  Token containment would let model text, negation, or an
        # incidental diagnostic manufacture even a quota candidate.
        normalized_message = message.strip().casefold()
        if normalized_message not in GROK_ACCOUNT_QUOTA_CODES:
            return ""
        message_codes.add(normalized_message)
    return next(iter(message_codes)) if len(message_codes) == 1 else ""


def build_grok_terminal_quota_receipt(
    *,
    command: Sequence[str],
    model: str,
    inner_returncode: int,
    terminal_event: dict[str, object],
) -> dict[str, object]:
    """Project a typed terminal quota frame into one bounded candidate."""

    if isinstance(inner_returncode, bool) or int(inner_returncode) == 0:
        raise ValueError("terminal quota receipt requires a nonzero returncode")
    quota_code = grok_terminal_quota_code(terminal_event)
    if not quota_code:
        raise ValueError("terminal event is not a typed account-quota error")
    binding = validate_grok_runner_command_binding(command)
    if not binding:
        raise ValueError("Grok runner command has no valid invocation binding")
    try:
        terminal_bytes = json.dumps(
            terminal_event,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("terminal event is not canonical JSON") from exc
    return {
        "schema": GROK_TERMINAL_QUOTA_RECEIPT_SCHEMA,
        "provider": "grok",
        "model": str(model).strip() or DEFAULT_GROK_MODEL,
        "error_kind": "quota_exhausted",
        "quota_code": quota_code,
        "inner_returncode": int(inner_returncode),
        "runner_returncode": GROK_QUOTA_EXHAUSTED_EXIT_CODE,
        "invocation_binding_sha256": binding,
        "terminal_event_sha256": hashlib.sha256(terminal_bytes).hexdigest(),
    }


def encode_grok_terminal_quota_receipt(receipt: dict[str, object]) -> str:
    """Encode a durable diagnostic copy of an untrusted quota candidate."""

    return GROK_TERMINAL_QUOTA_RECEIPT_PREFIX + json.dumps(
        receipt,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def parse_grok_terminal_quota_receipt(
    value: str | bytes,
    *,
    expected_runner_command: Sequence[str] = (),
) -> dict[str, object]:
    """Validate one typed quota candidate and its optional command binding."""

    if isinstance(value, bytes):
        try:
            text = value.decode("utf-8")
        except UnicodeDecodeError:
            return {}
    else:
        text = str(value or "")
    text = text.strip()
    if text.startswith(GROK_TERMINAL_QUOTA_RECEIPT_PREFIX):
        text = text[len(GROK_TERMINAL_QUOTA_RECEIPT_PREFIX) :]
    if not text or len(text.encode("utf-8")) > GROK_TERMINAL_RECEIPT_MAX_BYTES:
        return {}
    try:
        receipt = json.loads(text)
    except (TypeError, ValueError, RecursionError):
        return {}
    if not isinstance(receipt, dict):
        return {}
    if set(receipt) != {
        "schema",
        "provider",
        "model",
        "error_kind",
        "quota_code",
        "inner_returncode",
        "runner_returncode",
        "invocation_binding_sha256",
        "terminal_event_sha256",
    }:
        return {}
    quota_code = receipt.get("quota_code")
    if (
        receipt.get("schema") != GROK_TERMINAL_QUOTA_RECEIPT_SCHEMA
        or receipt.get("provider") != "grok"
        or receipt.get("error_kind") != "quota_exhausted"
        or not isinstance(quota_code, str)
        or quota_code not in GROK_ACCOUNT_QUOTA_CODES
        or receipt.get("runner_returncode")
        != GROK_QUOTA_EXHAUSTED_EXIT_CODE
        or not isinstance(receipt.get("model"), str)
        or not receipt["model"].strip()
    ):
        return {}
    inner_returncode = receipt.get("inner_returncode")
    if (
        not isinstance(inner_returncode, int)
        or isinstance(inner_returncode, bool)
        or inner_returncode == 0
    ):
        return {}
    for key in ("invocation_binding_sha256", "terminal_event_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(receipt.get(key) or "")):
            return {}
    if expected_runner_command:
        binding = validate_grok_runner_command_binding(expected_runner_command)
        if not binding or receipt["invocation_binding_sha256"] != binding:
            return {}
    return dict(receipt)


class _BoundedStreamingJsonParser:
    """Incrementally retain only the final bounded top-level NDJSON frame."""

    def __init__(self, max_frame_bytes: int = GROK_STREAM_FRAME_MAX_BYTES) -> None:
        self.max_frame_bytes = max_frame_bytes
        self.pending = bytearray()
        self.overlong = False
        self.tainted = False
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
                try:
                    value = json.loads(raw.decode("utf-8"))
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

    def feed(self, chunk: bytes, *, final: bool = False) -> None:
        start = 0
        while True:
            newline = chunk.find(b"\n", start)
            if newline < 0:
                self._append(chunk[start:])
                break
            self._append(chunk[start:newline])
            self._finish_line()
            start = newline + 1
        if final and (self.pending or self.overlong):
            self._finish_line()


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
    """Build the public, plain-output agent-mode Grok invocation."""

    return [
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


def _stream_grok_process(
    command: Sequence[str],
    *,
    env: dict[str, str],
) -> tuple[int, dict[str, object] | None, bool]:
    """Tee Grok stdout live while retaining only its final bounded frame."""

    process = subprocess.Popen(
        [str(item) for item in command],
        env=env,
        stdout=subprocess.PIPE,
        # Stderr remains inherited and live; the runner never parses it.
        # close_fds keeps the candidate FD out of Grok, but same-UID procfs
        # access means stdout still cannot establish quota authority.
        stderr=None,
        close_fds=True,
    )
    if process.stdout is None:
        raise RuntimeError("Grok streaming stdout pipe was not created")
    parser = _BoundedStreamingJsonParser()
    output_buffer = getattr(sys.stdout, "buffer", None)
    decoder = (
        None
        if output_buffer is not None
        else codecs.getincrementaldecoder("utf-8")(errors="replace")
    )
    try:
        while True:
            read1 = getattr(process.stdout, "read1", process.stdout.read)
            chunk = read1(64 * 1024)
            if not chunk:
                break
            parser.feed(chunk)
            if output_buffer is not None:
                output_buffer.write(chunk)
                output_buffer.flush()
            else:
                assert decoder is not None
                rendered = decoder.decode(chunk, final=False)
                if rendered:
                    sys.stdout.write(rendered)
                    sys.stdout.flush()
        if decoder is not None:
            rendered = decoder.decode(b"", final=True)
            if rendered:
                sys.stdout.write(rendered)
                sys.stdout.flush()
        parser.feed(b"", final=True)
        return int(process.wait()), parser.last_event, parser.tainted
    except BaseException:
        try:
            process.terminate()
        except (AttributeError, OSError):
            pass
        try:
            process.wait(timeout=5)
        except (AttributeError, OSError, subprocess.TimeoutExpired):
            try:
                process.kill()
            except (AttributeError, OSError):
                pass
            try:
                process.wait(timeout=5)
            except (AttributeError, OSError, subprocess.TimeoutExpired):
                pass
        raise
    finally:
        try:
            process.stdout.close()
        except (AttributeError, OSError):
            pass


def _receipt_fd_from_environment() -> int:
    raw = os.environ.pop(GROK_TERMINAL_RECEIPT_FD_ENV, "").strip()
    try:
        descriptor = int(raw)
    except ValueError:
        return -1
    if descriptor < 3:
        return -1
    try:
        metadata = os.fstat(descriptor)
        flags = fcntl.fcntl(descriptor, fcntl.F_GETFL)
    except OSError:
        return -1
    writable = (flags & os.O_ACCMODE) in {os.O_WRONLY, os.O_RDWR}
    return descriptor if stat.S_ISFIFO(metadata.st_mode) and writable else -1


def _write_private_receipt(descriptor: int, receipt: dict[str, object]) -> bool:
    if descriptor < 3:
        return False
    encoded = (
        json.dumps(
            receipt,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    if len(encoded) > GROK_TERMINAL_RECEIPT_MAX_BYTES:
        return False
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                return False
            view = view[written:]
    except OSError:
        return False
    return True


def _run(args: argparse.Namespace, receipt_fd: int) -> int:
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

        required_commands = [
            str(os.environ.get(PROVIDER_COMMAND_REQUIRED_COMMANDS_ENV) or ""),
            *(str(item) for item in args.require_command),
        ]
        try:
            with sealed_provider_command_environment(
                os.environ,
                required_commands=required_commands,
            ) as command_environment:
                command = build_grok_cli_command(
                    mode=str(args.mode),
                    workspace=workspace,
                    model_name=model,
                    max_turns=max_turns,
                    grok_bin=grok_bin,
                    prompt_file=prompt_path,
                    permission_mode=permission_mode,
                )
                supervised_binding = validate_grok_runner_command_binding(
                    args.outer_runner_command
                )
                supervised = receipt_fd >= 3 and bool(supervised_binding)
                if supervised:
                    try:
                        output_index = command.index("--output-format")
                        command[output_index + 1] = "streaming-json"
                    except (ValueError, IndexError) as exc:
                        raise LLMRouterError(
                            "Grok agent command has no output-format slot"
                        ) from exc
                env = build_grok_cli_env(base_env=os.environ)
                env[PROVIDER_COMMAND_ENV_WRAPPER_ENV] = (
                    command_environment.wrapper_path
                )
                env[PROVIDER_COMMAND_ENV_DIGEST_ENV] = (
                    command_environment.contract_sha256
                )
                env[FORMAL_TOOLCHAIN_CONTRACT_SHA256_ENV] = (
                    command_environment.formal_toolchain_contract_sha256
                )
                # The receipt descriptor was popped from os.environ before
                # this environment was built and close_fds keeps it out of
                # the inner Grok process.
                env.pop(GROK_TERMINAL_RECEIPT_FD_ENV, None)
                os.chdir(workspace)
                if not supervised:
                    completed = subprocess.run(
                        command,
                        env=env,
                        check=False,
                    )
                    return int(completed.returncode)
                (
                    inner_returncode,
                    terminal_event,
                    stream_tainted,
                ) = _stream_grok_process(command, env=env)
                quota_code = grok_terminal_quota_code(terminal_event)
                if (
                    supervised
                    and not stream_tainted
                    and inner_returncode != 0
                    and quota_code
                ):
                    receipt = build_grok_terminal_quota_receipt(
                        command=args.outer_runner_command,
                        model=model,
                        inner_returncode=inner_returncode,
                        terminal_event=terminal_event or {},
                    )
                    if _write_private_receipt(receipt_fd, receipt):
                        return GROK_QUOTA_EXHAUSTED_EXIT_CODE
                if (
                    supervised
                    and inner_returncode == GROK_QUOTA_EXHAUSTED_EXIT_CODE
                ):
                    # A child collision is ordinary failure; only this wrapper
                    # can mint the reserved control status with a private
                    # receipt.
                    return 1
                return inner_returncode
        except (
            LLMRouterError,
            ProviderCommandEnvironmentError,
            ValidationRuntimeError,
            OSError,
            RuntimeError,
        ) as exc:
            print(str(exc), file=sys.stderr)
            return 2
    finally:
        if prompt_path:
            try:
                os.unlink(prompt_path)
            except OSError:
                pass


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
    parser.add_argument(
        "--require-command",
        action="append",
        default=[],
        help=(
            "bare command that must be identity-bound on the declared task "
            "PATH before Grok starts (repeatable)"
        ),
    )
    parser.add_argument(GROK_INVOCATION_ID_FLAG, default="")
    parser.add_argument(GROK_INVOCATION_BINDING_FLAG, default="")
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(raw_argv)
    executable = str(Path(__file__).resolve())
    args.outer_runner_command = [sys.executable, executable, *raw_argv]
    receipt_fd = _receipt_fd_from_environment()
    try:
        return _run(args, receipt_fd)
    finally:
        if receipt_fd >= 3:
            try:
                os.close(receipt_fd)
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
