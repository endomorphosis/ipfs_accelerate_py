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
import signal
import shutil
import subprocess
import sys
import tempfile
import stat
import threading
import time
import uuid
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import TextIO

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
MAX_CODEX_FALLBACK_ARGUMENTS = 64
MAX_CODEX_FALLBACK_ARGUMENT_BYTES = 4_096
MAX_GROK_STREAM_EVENT_BYTES = 64 * 1024
MAX_GROK_SESSION_RECORD_BYTES = 16 * 1024 * 1024
# Only this exact durable native 402 record has been observed and is authorized
# to cross the provider boundary. Other native types remain diagnostic only.
GROK_QUOTA_ERROR_TYPES = frozenset({"usage_pool_exhausted"})
CODEX_QUOTA_FALLBACK_MODEL = "gpt-5.6-terra"
CODEX_QUOTA_FALLBACK_REASONING = 'model_reasoning_effort="medium"'
GROK_PRIMARY_SANDBOX_PROFILE = "ipfs-accelerate-provider-isolated"
GROK_ISOLATION_GROK_SANDBOX = "grok-sandbox"
GROK_ISOLATION_DOCKER = "docker"
DEFAULT_GROK_ISOLATION_IMAGE = "ubuntu:24.04"
_DOCKER_LOCAL_HOST = "unix:///var/run/docker.sock"
_DOCKER_CLEANUP_WATCHDOG_ARG = "--internal-docker-cleanup-watchdog"
_DOCKER_CONTAINER_NAME_RE = re.compile(
    r"ipfs-accelerate-grok-[0-9]+-[0-9a-f]{32}"
)
_DOCKER_CLEANUP_TIMEOUT_SECONDS = 8.0
_SEALED_GROK_TOOLS = "read_file,search_replace,grep,list_dir,todo_write"
_SEALED_GROK_DISALLOWED_TOOLS = (
    "run_terminal_cmd,run_terminal_command,web_search,web_fetch,search_tool,"
    "use_tool,call_mcp_tool,list_mcp_resources,list_mcp_resource_templates,"
    "read_mcp_resource,fetch_mcp_resource,task,Agent,memory,lsp,spawn_subagent"
)
_ALTERNATE_PROVIDER_EXECUTABLES = (
    "codex",
    "copilot",
    "gh",
    "goose",
    "openai",
    "gemini",
    "claude",
    "vibe",
    "mistral",
    "ollama",
    "llama",
    "llama-server",
)
_CONTAINER_RUNTIME_EXECUTABLES = (
    "docker",
    "podman",
    "nerdctl",
    "buildah",
    "ctr",
    "crictl",
)
_GROK_DENIED_EXECUTABLES = (
    *_ALTERNATE_PROVIDER_EXECUTABLES,
    *_CONTAINER_RUNTIME_EXECUTABLES,
    "grok",
)
GROK_ISOLATION_DENY_RULES = tuple(
    rule
    for executable in _GROK_DENIED_EXECUTABLES
    for rule in (
        f"Bash({executable})",
        f"Bash({executable} *)",
        f"Bash(/usr/bin/{executable})",
        f"Bash(/usr/bin/{executable} *)",
        f"Bash(/usr/local/bin/{executable})",
        f"Bash(/usr/local/bin/{executable} *)",
    )
)
GROK_ISOLATION_DENY_RULES += (
    "Bash(/opt/ipfs-accelerate/grok)",
    "Bash(/opt/ipfs-accelerate/grok *)",
)
_ALTERNATE_PROVIDER_STANDARD_PATHS = tuple(
    path
    for executable in _ALTERNATE_PROVIDER_EXECUTABLES
    for path in (
        f"/usr/bin/{executable}",
        f"/usr/local/bin/{executable}",
        f"/opt/homebrew/bin/{executable}",
    )
)
_CONTAINER_RUNTIME_STANDARD_PATHS = tuple(
    path
    for executable in _CONTAINER_RUNTIME_EXECUTABLES
    for path in (
        f"/usr/bin/{executable}",
        f"/usr/local/bin/{executable}",
        f"/opt/homebrew/bin/{executable}",
    )
)
_CONTAINER_RUNTIME_STANDARD_SOCKETS = (
    "/var/run/docker.sock",
    "/run/docker.sock",
    "/run/podman/podman.sock",
    "/run/containerd/containerd.sock",
    "/var/run/containerd/containerd.sock",
)
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


def _operating_system_account_home() -> Path:
    """Resolve the login account home independently of inherited HOME."""

    if os.name == "posix":
        import pwd

        return Path(pwd.getpwuid(os.getuid()).pw_dir).resolve(strict=True)
    return Path.home().resolve(strict=True)


def resolve_codex_quota_fallback_executable(
    *,
    workspace: str | Path,
    configured: str = "",
) -> str:
    """Resolve a pinned executable that the Grok workspace cannot replace."""

    workspace_path = Path(workspace).expanduser().resolve()
    codex_candidate = str(configured or shutil.which("codex") or "").strip()
    if not codex_candidate:
        return ""
    candidate_path = Path(codex_candidate).expanduser()
    if not candidate_path.is_absolute():
        resolved_from_path = shutil.which(codex_candidate)
        if not resolved_from_path:
            return ""
        candidate_path = Path(resolved_from_path)
    try:
        resolved_candidate = candidate_path.resolve(strict=True)
    except OSError:
        return ""
    candidate_entry = Path(os.path.abspath(candidate_path))
    system_entries = {
        Path("/usr/bin/codex"),
        Path("/usr/local/bin/codex"),
        Path("/usr/bin/codex.exe"),
        Path("/usr/local/bin/codex.exe"),
    }
    package_roots = (
        Path("/usr/lib/node_modules/@openai/codex"),
        Path("/usr/local/lib/node_modules/@openai/codex"),
    )
    matched_root = next(
        (
            root
            for root in package_roots
            if resolved_candidate == root
            or resolved_candidate.is_relative_to(root)
        ),
        resolved_candidate.parent
        if resolved_candidate.parent in {Path("/usr/bin"), Path("/usr/local/bin")}
        else None,
    )
    try:
        trust_chain = (
            [candidate_entry, candidate_entry.parent, resolved_candidate]
            + (
                list(resolved_candidate.parents)[
                    : list(resolved_candidate.parents).index(matched_root) + 1
                ]
                if matched_root is not None and resolved_candidate != matched_root
                else ([matched_root] if matched_root is not None else [])
            )
        )
        trusted_chain = all(
            path.lstat().st_uid == 0
            and (path.is_symlink() or not path.stat().st_mode & 0o022)
            for path in trust_chain
        )
    except (OSError, ValueError):
        trusted_chain = False
    if (
        candidate_entry not in system_entries
        or matched_root is None
        or not trusted_chain
        or not candidate_entry.is_file()
        or not os.access(candidate_entry, os.X_OK)
        or candidate_entry.is_relative_to(workspace_path)
        or resolved_candidate.is_relative_to(workspace_path)
        or candidate_entry.name.casefold() not in {"codex", "codex.exe"}
    ):
        return ""
    return str(candidate_entry)


def _resolve_trusted_grok_bin(*, configured: str, workspace: Path) -> str:
    """Pin Grok to a system install or its versioned standalone download."""

    candidate = Path(str(configured or "").strip()).expanduser()
    if not candidate.is_absolute():
        resolved_from_path = shutil.which(str(candidate))
        if not resolved_from_path:
            return ""
        candidate = Path(resolved_from_path)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError:
        return ""
    try:
        resolved_stat = resolved.stat()
    except OSError:
        return ""
    # GROK_HOME is intentionally not an executable trust anchor: an inherited
    # override could redirect both quota invocations to an attacker-owned
    # binary. The sealed route accepts only the account's standard download.
    download_root = (
        _operating_system_account_home() / ".grok" / "downloads"
    ).resolve(strict=False)
    system_install = resolved.parent in {
        Path("/usr/bin"),
        Path("/usr/local/bin"),
    }
    versioned_download = (
        resolved.parent == download_root
        and re.fullmatch(
            r"grok-[0-9]+(?:\.[0-9]+){2}-(?:linux|darwin)-"
            r"(?:aarch64|arm64|x86_64|amd64)",
            resolved.name,
        )
        is not None
    )
    trusted_owner = (
        resolved_stat.st_uid == 0
        if system_install
        else resolved_stat.st_uid == os.getuid()
    )
    if (
        candidate.name.casefold() not in {"grok", "grok.exe"}
        or not resolved.is_file()
        or not os.access(resolved, os.X_OK)
        or resolved_stat.st_mode & 0o022
        or not trusted_owner
        or not (system_install or versioned_download)
        or candidate.absolute().is_relative_to(workspace)
        or resolved.is_relative_to(workspace)
    ):
        return ""
    return str(resolved)


def build_grok_quota_routed_agent_command(
    *,
    workspace: str | Path = ".",
    python_executable: str = "",
    grok_bin: str = "",
    codex_bin: str = "",
    max_turns: int = 100_000,
) -> list[str]:
    """Build the canonical Grok-4.5 then typed-quota Terra/medium route.

    The returned parent runner owns the Codex argv.  Grok receives neither the
    executable/auth authority nor any way to invoke this fallback directly.
    """

    workspace_text = str(workspace)
    codex = resolve_codex_quota_fallback_executable(
        workspace=workspace,
        configured=codex_bin,
    )
    command = [
        str(python_executable or sys.executable),
        "-m",
        "ipfs_accelerate_py.agent_supervisor.grok_cli_runner",
        "--workspace",
        workspace_text,
        "--model",
        DEFAULT_GROK_MODEL,
        "--max-turns",
        str(max(1, int(max_turns))),
        "--mode",
        "agent",
    ]
    if codex:
        fallback = [
            codex,
            "exec",
            "--ignore-user-config",
            "--ignore-rules",
            "--ephemeral",
            "-s",
            "workspace-write",
            "-C",
            workspace_text,
            "-m",
            CODEX_QUOTA_FALLBACK_MODEL,
            "-c",
            CODEX_QUOTA_FALLBACK_REASONING,
            "-",
        ]
        command.extend(
            [
                "--codex-fallback-command-json",
                json.dumps(fallback, separators=(",", ":")),
            ]
        )
    if str(grok_bin).strip():
        command.extend(["--grok-bin", str(grok_bin).strip()])
    return command


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

    cmd = [
        grok_bin,
        "--cwd",
        str(workspace),
        "--model",
        model,
        "--permission-mode",
        permission_mode,
        "--always-approve",
        "--no-subagents",
        "--disable-web-search",
        "--no-memory",
        "--disallowed-tools",
        _SEALED_GROK_DISALLOWED_TOOLS,
        "--tools",
        _SEALED_GROK_TOOLS,
        "--sandbox",
        GROK_PRIMARY_SANDBOX_PROFILE,
        "--max-turns",
        str(max_turns),
        "--output-format",
        "plain",
        "--prompt-file",
        str(prompt_file),
    ]
    for rule in GROK_ISOLATION_DENY_RULES:
        cmd.extend(["--deny", rule])
    return cmd


def _existing_path(path: Path) -> Path | None:
    """Return an absolute existing path without collapsing its symlink name."""

    try:
        expanded = path.expanduser()
        if not expanded.is_absolute() or not expanded.exists():
            return None
        return expanded.absolute()
    except OSError:
        return None


def _provider_payload_root(path: Path) -> Path | None:
    """Return a known npm provider package root for a resolved entrypoint."""

    for candidate in (path, *path.parents):
        if (
            candidate.name.casefold() in _ALTERNATE_PROVIDER_EXECUTABLES
            and candidate.parent.name.casefold() in {"@openai", "@github"}
        ):
            return candidate
    return None


def _which_in_environment(executable: str, env: dict[str, str]) -> Path | None:
    """Resolve one executable from an explicit environment without globals."""

    suffixes = ("", ".exe") if os.name == "nt" else ("",)
    for directory in os.get_exec_path(env):
        for suffix in suffixes:
            candidate = Path(directory or os.curdir) / f"{executable}{suffix}"
            try:
                if candidate.is_file() and os.access(candidate, os.X_OK):
                    return candidate.absolute()
            except OSError:
                continue
    return None


def _unix_socket_path(raw_value: str) -> Path | None:
    """Project a local container-host URI to its Unix socket path."""

    value = str(raw_value or "").strip()
    if not value.startswith("unix://"):
        return None
    raw_path = value.removeprefix("unix://")
    return Path(raw_path) if raw_path.startswith("/") else None


def _dbus_socket_path(raw_value: str) -> Path | None:
    """Project a D-Bus Unix address to its filesystem socket, if present."""

    value = str(raw_value or "").strip()
    match = re.search(r"(?:^|;)unix:path=([^,;]+)", value)
    if match is None:
        return None
    path = Path(match.group(1))
    return path if path.is_absolute() else None


def _grok_isolation_deny_paths(
    *,
    base_env: dict[str, str],
    codex_fallback_command: Sequence[str],
    grok_home: Path,
    sentinel_path: Path,
    workspace: Path | None = None,
) -> tuple[Path, ...]:
    """Resolve installed peer executables, payloads, and credential stores."""

    candidates: set[Path] = {
        grok_home.absolute(),
        sentinel_path.absolute(),
        Path("/proc"),
        Path("/dev"),
    }
    if workspace is not None:
        candidates.add(workspace / ".git")
    for executable in _GROK_DENIED_EXECUTABLES:
        found = _which_in_environment(executable, base_env)
        if found:
            candidates.add(found)
    if codex_fallback_command:
        candidates.add(Path(codex_fallback_command[0]))
    candidates.update(Path(item) for item in _ALTERNATE_PROVIDER_STANDARD_PATHS)
    candidates.update(Path(item) for item in _CONTAINER_RUNTIME_STANDARD_PATHS)
    candidates.update(Path(item) for item in _CONTAINER_RUNTIME_STANDARD_SOCKETS)

    for variable in ("DOCKER_HOST", "CONTAINER_HOST"):
        socket_path = _unix_socket_path(base_env.get(variable, ""))
        if socket_path is not None:
            candidates.add(socket_path)

    configured_home = str(base_env.get("HOME") or "").strip()
    user_home = Path(configured_home).expanduser() if configured_home else Path.home()
    candidates.update(
        {
            user_home / ".codex",
            user_home / ".copilot",
            user_home / ".config" / "gh",
            user_home / ".config" / "github-copilot",
            user_home / ".config" / "goose",
            user_home / ".local" / "share" / "goose",
            user_home / ".local" / "state" / "goose",
            user_home / ".openai",
            user_home / ".config" / "openai",
            user_home / ".gemini",
            user_home / ".config" / "gemini",
            user_home / ".claude",
            user_home / ".config" / "claude",
            user_home / ".mistral",
            user_home / ".vibe",
            user_home / ".config" / "mistral",
            user_home / ".ollama",
            user_home / ".config" / "ollama",
            user_home / ".cache" / "huggingface",
            user_home / ".config" / "huggingface",
            user_home / ".docker",
            user_home / ".config" / "containers",
            user_home / ".kube",
        }
    )
    source_grok_home_raw = str(base_env.get("GROK_HOME") or "").strip()
    source_grok_home = (
        Path(source_grok_home_raw).expanduser()
        if source_grok_home_raw
        else user_home / ".grok"
    )
    candidates.add(source_grok_home / "auth.json")
    candidates.update(
        user_home / ".local" / "bin" / executable
        for executable in _GROK_DENIED_EXECUTABLES
    )
    for variable in (
        "CODEX_HOME",
        "COPILOT_CONFIG_DIR",
        "GH_CONFIG_DIR",
        "GOOSE_CONFIG_DIR",
        "OPENAI_CONFIG_DIR",
        "GEMINI_CONFIG_DIR",
        "CLAUDE_CONFIG_DIR",
        "MISTRAL_CONFIG_DIR",
        "VIBE_HOME",
        "OLLAMA_CONFIG_DIR",
    ):
        configured = str(base_env.get(variable) or "").strip()
        if configured:
            candidates.add(Path(configured))
    xdg_config = str(base_env.get("XDG_CONFIG_HOME") or "").strip()
    if xdg_config:
        candidates.add(Path(xdg_config) / "gh")
        candidates.add(Path(xdg_config) / "github-copilot")
        candidates.add(Path(xdg_config) / "containers")
        candidates.add(Path(xdg_config) / "goose")
    xdg_state = str(base_env.get("XDG_STATE_HOME") or "").strip()
    if xdg_state:
        candidates.add(Path(xdg_state) / "goose")
    xdg_runtime = str(base_env.get("XDG_RUNTIME_DIR") or "").strip()
    if xdg_runtime:
        candidates.add(Path(xdg_runtime) / "docker.sock")
        candidates.add(Path(xdg_runtime) / "podman" / "podman.sock")
    if hasattr(os, "getuid"):
        runtime_root = Path("/run/user") / str(os.getuid())
        candidates.add(runtime_root / "docker.sock")
        candidates.add(runtime_root / "podman" / "podman.sock")
        candidates.add(runtime_root / "bus")
        candidates.add(runtime_root / "keyring" / "control")
        candidates.add(runtime_root / "gnupg" / "S.gpg-agent")
        candidates.add(runtime_root / "gnupg" / "S.gpg-agent.extra")
    dbus_socket = _dbus_socket_path(
        base_env.get("DBUS_SESSION_BUS_ADDRESS", "")
    )
    if dbus_socket is not None:
        candidates.add(dbus_socket)
    for variable in ("SSH_AUTH_SOCK", "GNOME_KEYRING_CONTROL"):
        configured = str(base_env.get(variable) or "").strip()
        if configured:
            candidates.add(Path(configured))
    gpg_agent = str(base_env.get("GPG_AGENT_INFO") or "").partition(":")[0]
    if gpg_agent:
        candidates.add(Path(gpg_agent))

    denied: set[Path] = set()
    for candidate in candidates:
        existing = _existing_path(candidate)
        if existing is None:
            continue
        try:
            resolved = existing.resolve(strict=True)
        except OSError:
            denied.add(existing)
            continue
        payload_root = _provider_payload_root(resolved)
        if payload_root is not None:
            # The package-root bind covers its resolved executable and makes
            # any public symlink entrypoint dangle.  Mounting both a directory
            # and a nested file is rejected by OCI runtimes once the directory
            # has become read-only.
            denied.add(payload_root)
            continue
        denied.add(existing)
        denied.add(resolved)

    # This is the fixed in-container destination of the trusted primary binary.
    # It does not exist on the host, but direct file tools must still receive a
    # deny rule for the fixed in-container primary-binary destination.
    denied.add(Path("/opt/ipfs-accelerate/grok"))

    directory_denies = tuple(path for path in denied if path.is_dir())
    nonoverlapping = {
        path
        for path in denied
        if not any(
            path != directory and path.is_relative_to(directory)
            for directory in directory_denies
        )
    }
    return tuple(sorted(nonoverlapping, key=lambda item: str(item)))


def _isolated_grok_home(
    *,
    base_env: dict[str, str],
    child_env: dict[str, str],
    codex_fallback_command: Sequence[str],
    workspace: Path | None = None,
) -> tuple[tempfile.TemporaryDirectory[str], dict[str, str], Path, tuple[Path, ...]]:
    """Create a private Grok home with a machine-resolved custom sandbox.

    A unique global profile avoids project/user profile precedence conflicts.
    Its non-empty exact-path deny set forces Grok's Linux bubblewrap backend;
    the sentinel guarantees that even hosts without peer CLIs fail closed if
    the kernel sandbox cannot be installed.
    """

    temporary_home = tempfile.TemporaryDirectory(prefix="asref-grok-home-")
    grok_home = Path(temporary_home.name)
    try:
        grok_home.chmod(0o700)
        sentinel_path = grok_home / "alternate-provider-deny-sentinel"
        sentinel_path.write_text("provider isolation sentinel\n", encoding="utf-8")
        sentinel_path.chmod(0o600)

        denied_paths = _grok_isolation_deny_paths(
            base_env=base_env,
            codex_fallback_command=codex_fallback_command,
            grok_home=grok_home,
            sentinel_path=sentinel_path,
            workspace=workspace,
        )
        if grok_home not in denied_paths:
            raise ValueError("Grok sandbox state-directory deny was not resolved")

        policy_lines = [
            f"[profiles.{GROK_PRIMARY_SANDBOX_PROFILE}]",
            'extends = "workspace"',
            "restrict_network = true",
            "deny = [",
        ]
        policy_lines.extend(f"  {json.dumps(str(path))}," for path in denied_paths)
        policy_lines.append("]")
        policy_path = grok_home / "sandbox.toml"
        policy_path.write_text("\n".join(policy_lines) + "\n", encoding="utf-8")
        policy_path.chmod(0o600)

        # Prevent compatibility discovery from importing peer-agent skills,
        # hooks, MCPs, or session authority from the parent account.
        config_path = grok_home / "config.toml"
        config_path.write_text(
            "\n".join(
                (
                    "[compat.cursor]",
                    "skills = false",
                    "rules = false",
                    "agents = false",
                    "mcps = false",
                    "hooks = false",
                    "sessions = false",
                    "",
                    "[compat.claude]",
                    "skills = false",
                    "rules = false",
                    "agents = false",
                    "mcps = false",
                    "hooks = false",
                    "sessions = false",
                    "",
                    "[compat.codex]",
                    "sessions = false",
                    "",
                    "[cli]",
                    "use_leader = false",
                )
            )
            + "\n",
            encoding="utf-8",
        )
        config_path.chmod(0o600)

        source_home_raw = str(base_env.get("GROK_HOME") or "").strip()
        source_home = (
            Path(source_home_raw).expanduser()
            if source_home_raw
            else user_home_from_env(base_env) / ".grok"
        )
        source_auth = source_home / "auth.json"
        if source_auth.is_file():
            # Preserve Grok's own authority without copying a credential.  The
            # alternate-provider stores are independently kernel-denied.
            (grok_home / "auth.json").symlink_to(source_auth.resolve(strict=True))

        isolated_env = dict(child_env)
        isolated_env["GROK_HOME"] = str(grok_home)
        isolated_env["HOME"] = str(grok_home)
        isolated_env["XDG_CONFIG_HOME"] = str(grok_home / "xdg-config")
        isolated_env["XDG_DATA_HOME"] = str(grok_home / "xdg-data")
        isolated_env["XDG_STATE_HOME"] = str(grok_home / "xdg-state")
        return temporary_home, isolated_env, policy_path, denied_paths
    except Exception:
        temporary_home.cleanup()
        raise


def user_home_from_env(env: dict[str, str]) -> Path:
    """Resolve HOME for child policy preparation without mutating process state."""

    configured = str(env.get("HOME") or "").strip()
    return Path(configured).expanduser() if configured else Path.home()


def _grok_executable_extension_paths(workspace: Path) -> tuple[Path, ...]:
    """Find project-scoped Grok/MCP/hook sources on the config search path."""

    relative_candidates = (
        Path(".grok/config.toml"),
        Path(".grok/hooks"),
        Path(".grok/plugins"),
        Path(".grok/lsp.json"),
        Path(".mcp.json"),
        Path(".claude/settings.json"),
        Path(".claude/settings.local.json"),
        Path(".claude/plugins"),
        Path(".cursor/hooks.json"),
        Path(".cursor/mcp.json"),
        Path(".cursor/plugins"),
    )
    roots: list[Path] = []
    current = workspace
    while True:
        roots.append(current)
        if (current / ".git").exists() or current.parent == current:
            break
        current = current.parent
    found = {
        candidate.absolute()
        for root in roots
        for relative in relative_candidates
        if (candidate := root / relative).exists()
    }
    return tuple(sorted(found, key=lambda item: str(item)))


def _grok_filesystem_deny_rules(paths: Sequence[Path]) -> tuple[str, ...]:
    """Build direct-tool path fences for the sealed capability route."""

    rules: list[str] = []
    for path in paths:
        value = str(path)
        if any(character in value for character in "*?[]()"):
            raise ValueError("Grok denied path cannot be represented safely")
        for operation in ("Read", "Grep", "Edit", "Write"):
            rules.append(f"{operation}({value})")
            rules.append(f"{operation}({value}/**)")
    return tuple(rules)


def _workspace_symlinks_reach_denied_paths(
    *,
    workspace: Path,
    denied_paths: Sequence[Path],
) -> tuple[Path, ...]:
    """Detect direct-tool symlink aliases into provider/control authority."""

    sensitive = tuple(path.resolve(strict=False) for path in denied_paths)
    violations: list[Path] = []
    try:
        for root, directories, files in os.walk(
            workspace,
            topdown=True,
            followlinks=False,
        ):
            root_path = Path(root)
            for name in (*directories, *files):
                candidate = root_path / name
                if not candidate.is_symlink():
                    continue
                target = candidate.resolve(strict=False)
                if any(
                    target == denied or target.is_relative_to(denied)
                    for denied in sensitive
                ):
                    violations.append(candidate)
    except OSError as exc:
        raise ValueError("unable to audit workspace symlinks") from exc
    return tuple(sorted(violations, key=lambda item: str(item)))


def _workspace_regular_file_hardlinks(workspace: Path) -> tuple[Path, ...]:
    """Find writable workspace files that may alias authority outside it."""

    violations: list[Path] = []
    try:
        for root, _directories, files in os.walk(
            workspace,
            topdown=True,
            followlinks=False,
        ):
            root_path = Path(root)
            for name in files:
                candidate = root_path / name
                stat_result = candidate.lstat()
                if (
                    not candidate.is_symlink()
                    and candidate.is_file()
                    and stat_result.st_nlink > 1
                ):
                    violations.append(candidate)
    except OSError as exc:
        raise ValueError("unable to audit workspace hardlinks") from exc
    return tuple(sorted(violations, key=lambda item: str(item)))


def _decode_mountinfo_path(value: str) -> Path:
    """Decode Linux mountinfo's octal path escapes."""

    decoded = re.sub(
        r"\\([0-7]{3})",
        lambda match: chr(int(match.group(1), 8)),
        value,
    )
    path = Path(decoded)
    if not path.is_absolute():
        raise ValueError("mountinfo contains a non-absolute mount target")
    return path


def _workspace_descendant_mountpoints(
    workspace: Path,
    *,
    mountinfo_path: Path = Path("/proc/self/mountinfo"),
) -> tuple[Path, ...]:
    """Find mounts below a workspace that could project external authority."""

    if sys.platform != "linux":
        return ()
    try:
        lines = mountinfo_path.read_text(encoding="utf-8").splitlines()
        targets: list[Path] = []
        for line in lines:
            left, separator, _right = line.partition(" - ")
            fields = left.split()
            if not separator or len(fields) < 6:
                raise ValueError("malformed Linux mountinfo record")
            target = _decode_mountinfo_path(fields[4])
            if target != workspace and target.is_relative_to(workspace):
                targets.append(target)
    except (OSError, UnicodeError) as exc:
        raise ValueError("unable to audit workspace mountpoints") from exc
    return tuple(sorted(set(targets), key=lambda item: str(item)))


def _workspace_content_fingerprint(workspace: Path) -> str:
    """Hash every workspace path, file byte, mode, and symlink target."""

    digest = hashlib.sha256()
    try:
        for root, directories, files in os.walk(
            workspace,
            topdown=True,
            followlinks=False,
        ):
            directories.sort()
            files.sort()
            root_path = Path(root)
            for name in (*directories, *files):
                candidate = root_path / name
                relative = candidate.relative_to(workspace).as_posix()
                stat_result = candidate.lstat()
                digest.update(relative.encode("utf-8", errors="surrogateescape"))
                digest.update(b"\0")
                digest.update(str(stat_result.st_mode).encode("ascii"))
                digest.update(b"\0")
                if candidate.is_symlink():
                    digest.update(b"L")
                    digest.update(
                        os.readlink(candidate).encode(
                            "utf-8",
                            errors="surrogateescape",
                        )
                    )
                elif candidate.is_dir():
                    digest.update(b"D")
                elif candidate.is_file():
                    digest.update(b"F")
                    with candidate.open("rb") as handle:
                        while chunk := handle.read(1024 * 1024):
                            digest.update(chunk)
                else:
                    raise ValueError(
                        f"unsupported special file in Grok workspace: {candidate}"
                    )
                digest.update(b"\0")
    except (OSError, UnicodeError) as exc:
        raise ValueError("unable to fingerprint Grok workspace") from exc
    return digest.hexdigest()


def _grok_custom_sandbox_available() -> bool:
    """Return whether this host can execute Grok's native sandbox backend."""

    # Grok uses Seatbelt for custom deny profiles on macOS; bubblewrap is the
    # Linux implementation and is neither present nor required there.
    if sys.platform == "darwin":
        return True

    bwrap = shutil.which("bwrap")
    if not bwrap:
        return False
    try:
        completed = subprocess.run(
            [bwrap, "--ro-bind", "/", "/", "--", "/bin/true"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def _docker_isolation_binary() -> str:
    """Resolve a working Docker CLI with the pinned local isolation image."""

    docker_candidate = shutil.which("docker") or ""
    if not docker_candidate:
        return ""
    try:
        docker = Path(docker_candidate).resolve(strict=True)
        stat_result = docker.stat()
    except OSError:
        return ""
    if (
        docker not in {Path("/usr/bin/docker"), Path("/usr/local/bin/docker")}
        or not docker.is_file()
        or not os.access(docker, os.X_OK)
        or stat_result.st_uid != 0
        or stat_result.st_mode & 0o022
    ):
        return ""
    image = DEFAULT_GROK_ISOLATION_IMAGE
    try:
        with tempfile.TemporaryDirectory(
            prefix="asref-docker-config-probe-"
        ) as config_root:
            completed = subprocess.run(
                [
                    str(docker),
                    f"--host={_DOCKER_LOCAL_HOST}",
                    "--config",
                    config_root,
                    "image",
                    "inspect",
                    image,
                ],
                env=_docker_control_env(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=False,
            )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return str(docker) if completed.returncode == 0 else ""


def _docker_isolation_image_id(
    docker_bin: str,
    *,
    docker_config: Path,
    base_env: dict[str, str] | None = None,
) -> str:
    """Resolve the configured cached tag to an immutable local image ID."""

    # The quota route never accepts an environment-selected execution image.
    # Resolve the shipped local tag, then launch its exact immutable image ID.
    del base_env
    image = DEFAULT_GROK_ISOLATION_IMAGE
    try:
        completed = subprocess.run(
            [
                docker_bin,
                f"--host={_DOCKER_LOCAL_HOST}",
                "--config",
                str(docker_config),
                "image",
                "inspect",
                "--format",
                "{{.Id}}",
                image,
            ],
            env=_docker_control_env(),
            stdin=subprocess.DEVNULL,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    candidate = completed.stdout.strip()
    return (
        candidate
        if completed.returncode == 0
        and re.fullmatch(r"sha256:[0-9a-f]{64}", candidate)
        else ""
    )


def _docker_control_env(
    child_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Return a Docker CLI env without daemon/context/TLS redirection."""

    if child_env is not None:
        environment = {
            name: value
            for name, value in child_env.items()
            if not name.upper().startswith(
                ("DOCKER_", "CONTAINER_", "PODMAN_", "BUILDAH_")
            )
        }
    else:
        environment = {}
    environment.setdefault("PATH", "/usr/bin:/bin")
    environment.setdefault("HOME", "/nonexistent")
    return environment


def _select_grok_isolation_backend(*, require_container_boundary: bool = False) -> str:
    """Select an enforceable kernel boundary, never an unsandboxed route."""

    if require_container_boundary:
        if _docker_isolation_binary():
            return GROK_ISOLATION_DOCKER
        raise ValueError(
            "Default Grok quota route requires the pinned local Docker "
            "isolation image"
        )
    if _grok_custom_sandbox_available():
        return GROK_ISOLATION_GROK_SANDBOX
    if _docker_isolation_binary():
        return GROK_ISOLATION_DOCKER
    raise ValueError(
        "Grok provider isolation unavailable: bubblewrap cannot create its "
        "namespace and the pinned local Docker image is unavailable"
    )


def _git_metadata_roots(workspace: Path) -> tuple[Path, ...]:
    """Resolve linked-worktree Git metadata needed for read-only Git commands."""

    marker = workspace / ".git"
    if not marker.is_file():
        return ()
    try:
        prefix, separator, raw_git_dir = marker.read_text(
            encoding="utf-8"
        ).strip().partition(":")
    except (OSError, UnicodeError):
        return ()
    if prefix.casefold() != "gitdir" or not separator or not raw_git_dir.strip():
        return ()
    git_dir = Path(raw_git_dir.strip())
    if not git_dir.is_absolute():
        git_dir = marker.parent / git_dir
    try:
        git_dir = git_dir.resolve(strict=True)
    except OSError:
        return ()
    common_dir = git_dir
    common_marker = git_dir / "commondir"
    if common_marker.is_file():
        try:
            raw_common = common_marker.read_text(encoding="utf-8").strip()
            candidate = Path(raw_common)
            if not candidate.is_absolute():
                candidate = git_dir / candidate
            common_dir = candidate.resolve(strict=True)
        except (OSError, UnicodeError):
            common_dir = git_dir
    return tuple(dict.fromkeys((common_dir, git_dir)))


def _docker_mount(
    source: Path,
    *,
    destination: Path | None = None,
    read_only: bool,
) -> list[str]:
    """Return one Docker bind-mount argument without invoking a shell."""

    target = destination or source
    fields = [
        "type=bind",
        f"src={source}",
        f"dst={target}",
    ]
    if read_only:
        fields.append("readonly")
    return ["--mount", ",".join(fields)]


def _remove_exact_docker_container(
    *,
    docker_bin: str,
    docker_config: Path,
    container_name: str,
    settle_for_creation: bool,
) -> None:
    """Boundedly force-remove one runner-owned container by an exact name."""

    deadline = time.monotonic() + _DOCKER_CLEANUP_TIMEOUT_SECONDS
    while True:
        try:
            completed = subprocess.run(
                [
                    docker_bin,
                    f"--host={_DOCKER_LOCAL_HOST}",
                    "--config",
                    str(docker_config),
                    "rm",
                    "--force",
                    container_name,
                ],
                env=_docker_control_env(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2,
                check=False,
            )
            if completed.returncode == 0:
                return
        except (OSError, subprocess.TimeoutExpired):
            pass
        now = time.monotonic()
        if now >= deadline or not settle_for_creation:
            return
        time.sleep(0.1)


def _robust_remove_runner_temp_tree(path: Path) -> None:
    """Remove one runner-owned temp tree without following credential symlinks."""

    try:
        path.chmod(0o700, follow_symlinks=False)
    except (FileNotFoundError, NotImplementedError, OSError):
        pass
    try:
        for root, directories, files in os.walk(path, topdown=True, followlinks=False):
            root_path = Path(root)
            try:
                root_path.chmod(0o700, follow_symlinks=False)
            except (NotImplementedError, OSError):
                pass
            for name in directories:
                candidate = root_path / name
                if candidate.is_symlink():
                    continue
                try:
                    candidate.chmod(0o700, follow_symlinks=False)
                except (NotImplementedError, OSError):
                    pass
            for name in files:
                candidate = root_path / name
                if candidate.is_symlink():
                    continue
                try:
                    candidate.chmod(0o600, follow_symlinks=False)
                except (NotImplementedError, OSError):
                    pass
    except OSError:
        pass
    try:
        shutil.rmtree(path)
    except (FileNotFoundError, OSError):
        pass


def _docker_cleanup_watchdog_main(argv: Sequence[str]) -> int:
    """Remove a leaked container after the owning runner closes or dies."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--docker-bin", required=True)
    parser.add_argument("--container-name", required=True)
    parser.add_argument("--cidfile", type=Path, required=True)
    parser.add_argument("--lease-root", type=Path, required=True)
    parser.add_argument("--grok-home", type=Path, required=True)
    parser.add_argument("--prompt-path", type=Path, required=True)
    args = parser.parse_args(list(argv))

    try:
        docker_path = Path(args.docker_bin).resolve(strict=True)
        docker_stat = docker_path.stat()
    except OSError:
        return 2
    lease_root = args.lease_root.absolute()
    docker_config = lease_root / "docker-config"
    cidfile = args.cidfile.absolute()
    grok_home = args.grok_home.absolute()
    prompt_path = args.prompt_path.absolute()
    temporary_root = Path(tempfile.gettempdir()).resolve()
    if (
        docker_path not in {Path("/usr/bin/docker"), Path("/usr/local/bin/docker")}
        or docker_path.name not in {"docker", "docker.exe"}
        or docker_stat.st_uid != 0
        or docker_stat.st_mode & 0o022
        or _DOCKER_CONTAINER_NAME_RE.fullmatch(args.container_name) is None
        or lease_root.parent != temporary_root
        or not lease_root.name.startswith("asref-grok-container-")
        or cidfile.parent != lease_root
        or cidfile.name != "container.cid"
        or not docker_config.is_dir()
        or grok_home.parent != temporary_root
        or not grok_home.name.startswith("asref-grok-home-")
        or prompt_path.parent != temporary_root
        or not prompt_path.name.startswith("asref-grok-prompt-")
    ):
        return 2

    # A clean marker means docker-run returned, but the final rm remains a
    # defensive idempotent action.  Empty input means the runner was killed;
    # retry briefly to cover a daemon-side container-creation race.
    cleanup_started = False

    def cleanup(*, settle_for_creation: bool) -> None:
        nonlocal cleanup_started
        if cleanup_started:
            return
        cleanup_started = True
        _remove_exact_docker_container(
            docker_bin=str(docker_path),
            docker_config=docker_config,
            container_name=args.container_name,
            settle_for_creation=settle_for_creation,
        )

    def terminate_watchdog(signum: int, _frame: object) -> None:
        # The supervisor deliberately terminates separately owned descendant
        # process groups before the runner group.  Reap synchronously here so
        # that ordering cannot strand the runner-owned workspace mount.
        cleanup(settle_for_creation=True)
        raise SystemExit(128 + signum)

    try:
        signal.signal(signal.SIGTERM, terminate_watchdog)
        signal.signal(signal.SIGINT, terminate_watchdog)
        clean_exit = sys.stdin.buffer.read(1) == b"C"
        cleanup(settle_for_creation=not clean_exit)
    finally:
        try:
            prompt_path.unlink()
        except FileNotFoundError:
            pass
        mask_root = lease_root / "provider-masks"
        _restore_mask_permissions(mask_root)
        try:
            shutil.rmtree(mask_root)
        except FileNotFoundError:
            pass
        _robust_remove_runner_temp_tree(grok_home)
        try:
            cidfile.unlink()
        except FileNotFoundError:
            pass
        try:
            docker_config.rmdir()
        except (FileNotFoundError, OSError):
            pass
        try:
            lease_root.rmdir()
        except (FileNotFoundError, OSError):
            pass
    return 0


class _DockerContainerLease:
    """Own a Docker container and an out-of-process kill-safe reaper."""

    def __init__(
        self,
        *,
        docker_bin: str,
        container_name: str,
        lease_root: Path,
        docker_config: Path,
        cidfile: Path,
        write_fd: int,
        watchdog: subprocess.Popen[bytes],
    ) -> None:
        self.docker_bin = docker_bin
        self.container_name = container_name
        self.lease_root = lease_root
        self.docker_config = docker_config
        self.cidfile = cidfile
        self._write_fd = write_fd
        self._watchdog = watchdog
        self._closed = False

    @classmethod
    def create(
        cls,
        docker_bin: str,
        *,
        grok_home: Path,
        prompt_path: Path,
    ) -> "_DockerContainerLease":
        docker_path = Path(docker_bin).resolve(strict=True)
        docker_stat = docker_path.stat()
        if (
            docker_path not in {Path("/usr/bin/docker"), Path("/usr/local/bin/docker")}
            or docker_path.name not in {"docker", "docker.exe"}
            or docker_stat.st_uid != 0
            or docker_stat.st_mode & 0o022
        ):
            raise ValueError("Docker isolation executable is not docker")
        lease_root = Path(
            tempfile.mkdtemp(prefix="asref-grok-container-")
        ).resolve()
        cidfile = lease_root / "container.cid"
        docker_config = lease_root / "docker-config"
        docker_config.mkdir(mode=0o700)
        container_name = (
            f"ipfs-accelerate-grok-{os.getpid()}-{uuid.uuid4().hex}"
        )
        read_fd, write_fd = os.pipe()
        try:
            watchdog = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    _DOCKER_CLEANUP_WATCHDOG_ARG,
                    "--docker-bin",
                    str(docker_path),
                    "--container-name",
                    container_name,
                    "--cidfile",
                    str(cidfile),
                    "--lease-root",
                    str(lease_root),
                    "--grok-home",
                    str(grok_home),
                    "--prompt-path",
                    str(prompt_path),
                ],
                stdin=read_fd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                close_fds=True,
            )
        except Exception:
            os.close(read_fd)
            os.close(write_fd)
            docker_config.rmdir()
            lease_root.rmdir()
            raise
        os.close(read_fd)
        if watchdog.poll() is not None:
            os.close(write_fd)
            docker_config.rmdir()
            lease_root.rmdir()
            raise ValueError("Docker cleanup watchdog failed to start")
        return cls(
            docker_bin=str(docker_path),
            container_name=container_name,
            lease_root=lease_root,
            docker_config=docker_config,
            cidfile=cidfile,
            write_fd=write_fd,
            watchdog=watchdog,
        )

    def close(self, *, docker_run_finished: bool) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if docker_run_finished:
                os.write(self._write_fd, b"C")
        except OSError:
            pass
        finally:
            try:
                os.close(self._write_fd)
            except OSError:
                pass
        try:
            self._watchdog.wait(timeout=_DOCKER_CLEANUP_TIMEOUT_SECONDS + 2)
        except subprocess.TimeoutExpired:
            _remove_exact_docker_container(
                docker_bin=self.docker_bin,
                docker_config=self.docker_config,
                container_name=self.container_name,
                settle_for_creation=False,
            )
            self._watchdog.terminate()
            try:
                self._watchdog.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._watchdog.kill()
                self._watchdog.wait(timeout=2)
        try:
            self.cidfile.unlink()
        except FileNotFoundError:
            pass
        mask_root = self.lease_root / "provider-masks"
        _restore_mask_permissions(mask_root)
        try:
            shutil.rmtree(mask_root)
        except FileNotFoundError:
            pass
        try:
            self.docker_config.rmdir()
        except (FileNotFoundError, OSError):
            pass
        try:
            self.lease_root.rmdir()
        except (FileNotFoundError, OSError):
            pass


def _restore_mask_permissions(mask_root: Path) -> None:
    """Make runner-created 000 mask directories removable after Docker exits."""

    try:
        entries = tuple(os.scandir(mask_root))
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        return
    for entry in entries:
        path = Path(entry.path)
        try:
            if entry.is_symlink():
                path.unlink()
            elif entry.is_dir(follow_symlinks=False):
                path.chmod(0o700, follow_symlinks=False)
            else:
                path.chmod(0o600, follow_symlinks=False)
        except (FileNotFoundError, NotImplementedError, OSError):
            continue


def _docker_grok_command(
    *,
    grok_command: Sequence[str],
    grok_bin: Path,
    workspace: Path,
    prompt_path: Path,
    grok_home: Path,
    base_env: dict[str, str],
    child_env: dict[str, str],
    denied_paths: Sequence[Path],
    mask_root: Path,
    docker_config: Path,
    container_name: str,
    cidfile: Path,
    docker_bin: str = "",
    isolation_image: str = "",
) -> list[str]:
    """Wrap Grok in a peer-provider capability boundary without shell tools.

    Grok necessarily retains its own read-only auth and writable ephemeral
    session state.  This boundary withholds peer providers; it is not a
    confidentiality boundary against Grok's own in-process file tools.
    """

    docker = str(docker_bin or _docker_isolation_binary())
    if not docker:
        raise ValueError("Docker Grok isolation became unavailable before launch")
    image = str(isolation_image).strip()
    if re.fullmatch(r"sha256:[0-9a-f]{64}", image) is None:
        raise ValueError("Docker Grok isolation image is not an immutable image ID")
    container_grok = Path("/opt/ipfs-accelerate/grok")
    command = [
        docker,
        f"--host={_DOCKER_LOCAL_HOST}",
        "--config",
        str(docker_config),
        "run",
        "--pull=never",
        "--rm",
        "--interactive",
        "--read-only",
        "--tmpfs",
        (
            "/tmp:rw,nosuid,nodev,noexec,mode=0700,"
            f"uid={os.getuid()},gid={os.getgid()}"
        ),
        "--tmpfs",
        (
            "/var/tmp:rw,nosuid,nodev,noexec,mode=0700,"
            f"uid={os.getuid()},gid={os.getgid()}"
        ),
        "--name",
        container_name,
        "--cidfile",
        str(cidfile),
        "--label",
        "ipfs_accelerate.grok_isolation=true",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=1024",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--workdir",
        str(workspace),
    ]
    # Docker receives values through its already-sanitized process environment;
    # secrets are never serialized into argv or process listings.
    for name in sorted(child_env):
        command.extend(["--env", name])

    # Host tools and libraries are readable for validation, while only the
    # active implementation worktree and Grok's ephemeral state are writable.
    host_usr = _existing_path(Path("/usr"))
    if host_usr is not None:
        command.extend(_docker_mount(host_usr, read_only=True))
    for git_root in _git_metadata_roots(workspace):
        command.extend(_docker_mount(git_root, read_only=True))
    command.extend(_docker_mount(workspace, read_only=False))
    git_control_path = _existing_path(workspace / ".git")
    if git_control_path is not None:
        command.extend(_docker_mount(git_control_path, read_only=True))
    command.extend(_docker_mount(prompt_path, read_only=True))
    command.extend(_docker_mount(grok_home, read_only=False))
    command.extend(
        _docker_mount(grok_bin, destination=container_grok, read_only=True)
    )

    source_home_raw = str(base_env.get("GROK_HOME") or "").strip()
    source_home = (
        Path(source_home_raw).expanduser()
        if source_home_raw
        else user_home_from_env(base_env) / ".grok"
    )
    source_auth = _existing_path(source_home / "auth.json")
    if source_auth is not None:
        # The symlink in the ephemeral home resolves to this exact path.  No
        # alternate-provider home or credential directory is mounted, and the
        # isolated child cannot mutate the parent's Grok credential either.
        command.extend(_docker_mount(source_auth, read_only=True))

    mask_root.mkdir(mode=0o700)
    sentinel = grok_home / "alternate-provider-deny-sentinel"
    for index, denied in enumerate(denied_paths):
        if (
            denied in {
                sentinel,
                grok_home,
                Path("/proc"),
                Path("/dev"),
                container_grok,
                git_control_path,
                source_auth,
            }
            or denied.is_relative_to(grok_home)
        ):
            continue
        mask_path = mask_root / str(index)
        if denied.is_dir():
            mask_path.mkdir(mode=0o000)
        else:
            mask_path.write_bytes(b"")
            mask_path.chmod(0o000)
        command.extend(
            _docker_mount(mask_path, destination=denied, read_only=True)
        )

    inner = list(grok_command)
    inner[0] = str(container_grok)
    command.extend([image, *inner])
    return command


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
    executable = Path(command[0])
    if (
        not executable.is_absolute()
        or executable.name.lower() not in {"codex", "codex.exe"}
    ):
        raise ValueError("Codex fallback executable must be an absolute codex path")
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

    flag_counts = {
        "--ignore-user-config": 0,
        "--ignore-rules": 0,
        "--ephemeral": 0,
    }
    option_values: dict[str, list[str]] = {
        "-C": [],
        "-m": [],
        "-c": [],
        "-s": [],
    }
    index = 2
    while index < len(command) - 1:
        item = command[index]
        if item in flag_counts:
            flag_counts[item] += 1
            index += 1
            continue
        if item not in option_values or index + 1 >= len(command) - 1:
            raise ValueError(
                "Codex quota fallback contains an unauthorized route option"
            )
        option_values[item].append(command[index + 1])
        index += 2

    if flag_counts["--ignore-user-config"] != 1:
        raise ValueError(
            "Codex quota fallback must ignore user configuration exactly once"
        )
    if flag_counts["--ephemeral"] != 1:
        raise ValueError("Codex quota fallback must be ephemeral exactly once")
    if flag_counts["--ignore-rules"] != 1:
        raise ValueError("Codex quota fallback must ignore rules exactly once")
    if option_values["-s"] != ["workspace-write"]:
        raise ValueError("Codex quota fallback sandbox is not exactly workspace-write")
    if option_values["-m"] != [CODEX_QUOTA_FALLBACK_MODEL]:
        raise ValueError("Codex quota fallback model is not exactly gpt-5.6-terra")
    if len(option_values["-C"]) != 1:
        raise ValueError("Codex quota fallback must contain exactly one workspace")
    fallback_workspace = Path(option_values["-C"][0]).resolve()
    if workspace is not None and fallback_workspace != workspace:
        raise ValueError("Codex quota fallback workspace does not match Grok workspace")
    executable = Path(command[0])
    try:
        resolved_executable = executable.resolve(strict=True)
    except OSError as exc:
        raise ValueError("Codex quota fallback executable does not exist") from exc
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError("Codex quota fallback executable is not executable")
    if workspace is not None and (
        executable.is_relative_to(workspace)
        or resolved_executable.is_relative_to(workspace)
    ):
        raise ValueError("Codex quota fallback executable must be outside workspace")

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


def _codex_quota_fallback_env(
    *,
    workspace: Path,
    base_env: dict[str, str],
) -> dict[str, str]:
    """Build a minimal official-endpoint Codex environment with pinned auth."""

    configured_home = str(base_env.get("CODEX_HOME") or "").strip()
    home = Path(str(base_env.get("HOME") or Path.home())).expanduser()
    candidate = Path(configured_home).expanduser() if configured_home else home / ".codex"
    try:
        codex_home = candidate.resolve(strict=True)
        auth_path = (codex_home / "auth.json").resolve(strict=True)
    except OSError as exc:
        raise ValueError("Codex quota fallback requires a validated auth.json") from exc
    if (
        not codex_home.is_dir()
        or not auth_path.is_file()
        or codex_home.is_relative_to(workspace)
        or auth_path.is_relative_to(workspace)
    ):
        raise ValueError("Codex quota fallback auth must be outside the workspace")

    allowed_exact = {
        "LANG",
        "LOGNAME",
        "NO_COLOR",
        "TERM",
        "USER",
    }
    environment = {
        name: value
        for name, value in base_env.items()
        if name in allowed_exact or name.startswith("LC_")
    }
    environment["HOME"] = str(codex_home)
    environment["CODEX_HOME"] = str(codex_home)
    environment["PATH"] = "/usr/bin:/bin"
    return environment


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


def _terminal_grok_failure_type_from_isolated_home(
    grok_home: Path,
    *,
    expected_model: str = DEFAULT_GROK_MODEL,
    expected_session_id: str = "",
) -> str:
    """Read one native session's final, terminal-correlated failure verdict.

    The isolated home starts without sessions and the runner supplies the exact
    UUID.  This record is necessary but never sufficient authority: a second,
    fresh tool-free Grok invocation must independently confirm quota.  Projected
    stdout is display-only.
    """

    try:
        records = tuple((grok_home / "sessions").rglob("updates.jsonl"))
    except OSError:
        return ""
    if len(records) != 1:
        return ""
    record = records[0]
    try:
        if (
            record.is_symlink()
            or not record.is_file()
            or not record.resolve(strict=True).is_relative_to(
                grok_home.resolve(strict=True)
            )
            or not 0 < record.stat().st_size <= MAX_GROK_SESSION_RECORD_BYTES
        ):
            return ""
        uuid.UUID(record.parent.name)
    except (OSError, ValueError):
        return ""

    recorded_session_id = record.parent.name
    if expected_session_id and recorded_session_id != expected_session_id:
        return ""
    observed_models: set[str] = set()
    latest_failure: tuple[str, str] | None = None
    latest_relevant = ""
    terminal_verdict = ""
    final_update_type = ""
    retry_failure_count = 0
    user_message_count = 0
    allowed_update_types = {
        "retry_state",
        "user_message_chunk",
        "turn_completed",
    }
    try:
        with record.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                if len(raw_line.encode("utf-8", errors="replace")) > (
                    MAX_GROK_SESSION_RECORD_BYTES
                ):
                    return ""
                try:
                    payload = json.loads(raw_line)
                except (json.JSONDecodeError, TypeError):
                    return ""
                if not isinstance(payload, dict) or payload.get("method") not in {
                    "_x.ai/session/update",
                    "session/update",
                }:
                    return ""
                params = payload.get("params")
                if (
                    not isinstance(params, dict)
                    or params.get("sessionId") != recorded_session_id
                ):
                    return ""
                update = params.get("update")
                if not isinstance(update, dict):
                    return ""
                update_type = str(update.get("sessionUpdate") or "")
                if update_type not in allowed_update_types:
                    return ""
                final_update_type = update_type
                metadata = update.get("_meta")
                if isinstance(metadata, dict):
                    model_id = str(metadata.get("modelId") or "").strip()
                    if model_id:
                        observed_models.add(model_id)

                if update_type == "retry_state":
                    if update.get("type") != "failed":
                        latest_failure = None
                        latest_relevant = "retry_state"
                        terminal_verdict = ""
                        continue
                    failure_type = _grok_failure_type_from_stream_event(raw_line)
                    failure_message = str(update.get("message") or "").strip()
                    retry_failure_count += 1
                    latest_failure = (failure_type, failure_message)
                    latest_relevant = "retry_state"
                    terminal_verdict = ""
                elif update_type == "turn_completed":
                    terminal_verdict = ""
                    if (
                        str(update.get("stop_reason") or "").casefold() == "error"
                        and latest_relevant == "retry_state"
                        and latest_failure is not None
                        and latest_failure[0] in GROK_QUOTA_ERROR_TYPES
                        and latest_failure[1]
                        and str(update.get("agent_result") or "").strip()
                        == latest_failure[1]
                    ):
                        terminal_verdict = latest_failure[0]
                    latest_relevant = "turn_completed"
                elif update_type == "user_message_chunk":
                    user_message_count += 1
    except (OSError, UnicodeError):
        return ""

    summary_path = record.parent / "summary.json"
    try:
        if (
            summary_path.is_symlink()
            or not summary_path.is_file()
            or summary_path.stat().st_size > MAX_GROK_SESSION_RECORD_BYTES
        ):
            return ""
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary_info = summary.get("info") if isinstance(summary, dict) else None
        summary_home = Path(str(summary.get("grok_home") or "")).resolve(strict=True)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
        return ""

    if (
        final_update_type != "turn_completed"
        # Initial balance exhaustion is durably emitted as a two-update
        # retry/terminal record with no user chunk or update-level model ID.
        # Later attempts add one user chunk. In both shapes summary.json is
        # required to pin the exact session, model, and isolated home.
        or not observed_models.issubset({expected_model})
        or retry_failure_count != 1
        or user_message_count > 1
        or not isinstance(summary_info, dict)
        or summary_info.get("id") != recorded_session_id
        or summary.get("current_model_id") != expected_model
        or summary_home != grok_home.resolve()
        or latest_failure
        != ("usage_pool_exhausted", _LEGACY_GROK_BALANCE_EXHAUSTED_MESSAGE)
    ):
        return ""
    return terminal_verdict


def _stream_pipe(
    source: TextIO,
    destination: TextIO,
) -> None:
    """Tee a child stream for operator visibility only."""

    while True:
        chunk = source.read(16 * 1024)
        if not chunk:
            break
        destination.write(chunk)
        destination.flush()


def _run_grok_with_typed_failure_capture(
    command: Sequence[str],
    *,
    env: dict[str, str],
) -> int:
    """Run Grok with live output; stdout never grants fallback authority."""

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
    stdout_thread = threading.Thread(
        target=_stream_pipe,
        args=(process.stdout, sys.stdout),
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
    return returncode


def _independently_verify_grok_quota(
    *,
    grok_bin: str,
    base_env: dict[str, str],
) -> str:
    """Confirm quota with a fresh pinned, tool-free Grok-4.5 invocation."""

    from ipfs_accelerate_py.llm_router import build_grok_cli_command, build_grok_cli_env

    verifier_root = Path(tempfile.mkdtemp(prefix="asref-grok-quota-verifier-"))
    isolated_home: tempfile.TemporaryDirectory[str] | None = None
    try:
        verifier_workspace = verifier_root / "workspace"
        verifier_workspace.mkdir(mode=0o700)
        prompt_path = verifier_root / "prompt.txt"
        prompt_path.write_text(
            "Reply with exactly the single word OK.",
            encoding="utf-8",
        )
        child_env = build_grok_cli_env(
            base_env=base_env,
            isolate_alternate_providers=True,
        )
        isolated_home, verifier_env, _policy, _denied = _isolated_grok_home(
            base_env=base_env,
            child_env=child_env,
            codex_fallback_command=(),
            workspace=verifier_workspace,
        )
        verifier_home = Path(verifier_env["GROK_HOME"])
        verifier_env.update(
            {
                "HOME": str(verifier_home),
                "XDG_CONFIG_HOME": str(verifier_home / "xdg-config"),
                "XDG_DATA_HOME": str(verifier_home / "xdg-data"),
                "XDG_STATE_HOME": str(verifier_home / "xdg-state"),
            }
        )
        command = build_grok_cli_command(
            mode="chat",
            workspace=verifier_workspace,
            model_name=DEFAULT_GROK_MODEL,
            max_turns=1,
            grok_bin=grok_bin,
            prompt_file=prompt_path,
            permission_mode="dontAsk",
            tools="",
        )
        verifier_session_id = str(uuid.uuid4())
        command.extend(
            [
                "--session-id",
                verifier_session_id,
                "--disallowed-tools",
                _SEALED_GROK_DISALLOWED_TOOLS,
            ]
        )
        output_index = command.index("--output-format") + 1
        command[output_index] = "streaming-json"
        try:
            completed = subprocess.run(
                command,
                env=verifier_env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=90,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return ""
        if completed.returncode == 0:
            return ""
        return _terminal_grok_failure_type_from_isolated_home(
            verifier_home,
            expected_session_id=verifier_session_id,
        )
    finally:
        if isolated_home is not None:
            _robust_remove_runner_temp_tree(Path(isolated_home.name))
            isolated_home.cleanup()
        _robust_remove_runner_temp_tree(verifier_root)




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

    try:
        codex_fallback_command = _parse_codex_fallback_command(
            str(args.codex_fallback_command_json)
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if codex_fallback_command and validate_grok_runner_command_binding(
        args.outer_runner_command
    ):
        print(
            "command-bound Grok supervision forbids an in-runner Codex "
            "fallback; the daemon must authorize a fresh retry",
            file=sys.stderr,
        )
        return 2

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
            # The runner changes cwd before dispatch.  Store the already
            # validated absolute workspace so a relative -C cannot be
            # reinterpreted beneath itself or redirected through a new link.
            workspace_index = codex_fallback_command.index("-C") + 1
            codex_fallback_command[workspace_index] = str(workspace)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        executable_extensions = _grok_executable_extension_paths(workspace)
        if executable_extensions:
            print(
                "Default Grok route refuses project MCP, hook, plugin, or LSP "
                "configuration: "
                + ", ".join(str(path) for path in executable_extensions),
                file=sys.stderr,
            )
            return 2

    grok_bin = str(args.grok_bin).strip() or find_grok_cli() or ""
    if not grok_bin:
        print("grok CLI not found on PATH", file=sys.stderr)
        return 127
    if codex_fallback_command:
        grok_bin = _resolve_trusted_grok_bin(
            configured=grok_bin,
            workspace=workspace,
        )
        if not grok_bin:
            print(
                "quota-routed Grok executable must be a pinned executable "
                "outside the writable workspace",
                file=sys.stderr,
            )
            return 2

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
        "bypassPermissions"
        if codex_fallback_command
        else (
            str(args.permission_mode).strip()
            or os.environ.get(
                "IPFS_ACCELERATE_AGENT_GROK_PERMISSION_MODE", ""
            ).strip()
            or os.environ.get(
                "ipfs_accelerate_py_GROK_CLI_PERMISSION_MODE", ""
            ).strip()
            or "bypassPermissions"
        )
    )

    prompt = sys.stdin.read()
    if not prompt.strip():
        print("empty implementation prompt on stdin", file=sys.stderr)
        return 2

    prompt_path = ""
    isolated_home: tempfile.TemporaryDirectory[str] | None = None
    docker_lease: _DockerContainerLease | None = None
    docker_run_finished = False
    grok_launch_env: dict[str, str] = {}
    workspace_baseline = ""
    command_environment_stack = ExitStack()
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
            command_environment = command_environment_stack.enter_context(
                sealed_provider_command_environment(
                    os.environ,
                    required_commands=required_commands,
                )
            )
            supervised_binding = validate_grok_runner_command_binding(
                args.outer_runner_command
            )
            supervised = (
                not codex_fallback_command
                and receipt_fd >= 3
                and bool(supervised_binding)
            )
            if supervised:
                command = build_grok_cli_command(
                    mode=str(args.mode),
                    workspace=workspace,
                    model_name=model,
                    max_turns=max_turns,
                    grok_bin=grok_bin,
                    prompt_file=prompt_path,
                    permission_mode=permission_mode,
                )
                try:
                    output_index = command.index("--output-format")
                    command[output_index + 1] = "streaming-json"
                except (ValueError, IndexError) as exc:
                    raise LLMRouterError(
                        "Grok agent command has no output-format slot"
                    ) from exc
                supervised_env = build_grok_cli_env(base_env=os.environ)
                supervised_env[PROVIDER_COMMAND_ENV_WRAPPER_ENV] = (
                    command_environment.wrapper_path
                )
                supervised_env[PROVIDER_COMMAND_ENV_DIGEST_ENV] = (
                    command_environment.contract_sha256
                )
                supervised_env[FORMAL_TOOLCHAIN_CONTRACT_SHA256_ENV] = (
                    command_environment.formal_toolchain_contract_sha256
                )
                supervised_env.pop(GROK_TERMINAL_RECEIPT_FD_ENV, None)
                os.chdir(workspace)
                (
                    inner_returncode,
                    terminal_event,
                    stream_tainted,
                ) = _stream_grok_process(command, env=supervised_env)
                quota_code = grok_terminal_quota_code(terminal_event)
                if not stream_tainted and inner_returncode != 0 and quota_code:
                    receipt = build_grok_terminal_quota_receipt(
                        command=args.outer_runner_command,
                        model=model,
                        inner_returncode=inner_returncode,
                        terminal_event=terminal_event or {},
                    )
                    if _write_private_receipt(receipt_fd, receipt):
                        return GROK_QUOTA_EXHAUSTED_EXIT_CODE
                if inner_returncode == GROK_QUOTA_EXHAUSTED_EXIT_CODE:
                    # Only this wrapper may mint the reserved control status.
                    return 1
                return inner_returncode

            if args.receipt_fd_declared and not codex_fallback_command:
                # An invalid/read-only ambient descriptor does not establish
                # supervision. Preserve the direct runner contract without
                # minting a receipt or reserving the child's exit status.
                command = build_grok_cli_command(
                    mode=str(args.mode),
                    workspace=workspace,
                    model_name=model,
                    max_turns=max_turns,
                    grok_bin=grok_bin,
                    prompt_file=prompt_path,
                    permission_mode=permission_mode,
                )
                direct_env = build_grok_cli_env(base_env=os.environ)
                direct_env[PROVIDER_COMMAND_ENV_WRAPPER_ENV] = (
                    command_environment.wrapper_path
                )
                direct_env[PROVIDER_COMMAND_ENV_DIGEST_ENV] = (
                    command_environment.contract_sha256
                )
                direct_env[FORMAL_TOOLCHAIN_CONTRACT_SHA256_ENV] = (
                    command_environment.formal_toolchain_contract_sha256
                )
                direct_env.pop(GROK_TERMINAL_RECEIPT_FD_ENV, None)
                os.chdir(workspace)
                completed = subprocess.run(command, env=direct_env, check=False)
                return int(completed.returncode)

            base_env = os.environ.copy()
            isolation_backend = _select_grok_isolation_backend(
                require_container_boundary=bool(codex_fallback_command),
            )
            cmd = build_grok_cli_command(
                mode=str(args.mode),
                workspace=workspace,
                model_name=model,
                max_turns=max_turns,
                grok_bin=grok_bin,
                prompt_file=prompt_path,
                permission_mode=permission_mode,
                tools=_SEALED_GROK_TOOLS,
                sandbox_profile=(
                    GROK_PRIMARY_SANDBOX_PROFILE
                    if isolation_backend == GROK_ISOLATION_GROK_SANDBOX
                    else None
                ),
                deny_rules=GROK_ISOLATION_DENY_RULES,
            )
            primary_session_id = str(uuid.uuid4())
            cmd.extend(
                [
                    "--session-id",
                    primary_session_id,
                    "--no-subagents",
                    "--disable-web-search",
                    "--no-memory",
                    "--disallowed-tools",
                    _SEALED_GROK_DISALLOWED_TOOLS,
                ]
            )
            child_env = build_grok_cli_env(
                base_env=base_env,
                isolate_alternate_providers=True,
            )
            isolated_home, env, _policy_path, _denied_paths = _isolated_grok_home(
                base_env=base_env,
                child_env=child_env,
                codex_fallback_command=codex_fallback_command,
                workspace=workspace,
            )
            env[PROVIDER_COMMAND_ENV_WRAPPER_ENV] = (
                command_environment.wrapper_path
            )
            env[PROVIDER_COMMAND_ENV_DIGEST_ENV] = (
                command_environment.contract_sha256
            )
            env[FORMAL_TOOLCHAIN_CONTRACT_SHA256_ENV] = (
                command_environment.formal_toolchain_contract_sha256
            )
            env.pop(GROK_TERMINAL_RECEIPT_FD_ENV, None)
            for rule in _grok_filesystem_deny_rules(_denied_paths):
                cmd.extend(["--deny", rule])
            if codex_fallback_command:
                symlink_violations = _workspace_symlinks_reach_denied_paths(
                    workspace=workspace,
                    denied_paths=_denied_paths,
                )
                if symlink_violations:
                    raise ValueError(
                        "Default Grok route refuses workspace symlinks into "
                        "provider/control paths: "
                        + ", ".join(str(path) for path in symlink_violations)
                    )
                hardlink_violations = _workspace_regular_file_hardlinks(workspace)
                if hardlink_violations:
                    raise ValueError(
                        "Default Grok route refuses multiply linked regular "
                        "workspace files: "
                        + ", ".join(str(path) for path in hardlink_violations)
                    )
                descendant_mounts = _workspace_descendant_mountpoints(workspace)
                if descendant_mounts:
                    raise ValueError(
                        "Default Grok route refuses descendant workspace "
                        "mountpoints: "
                        + ", ".join(str(path) for path in descendant_mounts)
                    )
            grok_launch_env = env
            if isolation_backend == GROK_ISOLATION_DOCKER:
                docker_bin = _docker_isolation_binary()
                if not docker_bin:
                    raise ValueError(
                        "Docker Grok isolation became unavailable before launch"
                    )
                docker_lease = _DockerContainerLease.create(
                    docker_bin,
                    grok_home=_policy_path.parent,
                    prompt_path=Path(prompt_path).resolve(strict=True),
                )
                isolation_image = _docker_isolation_image_id(
                    docker_lease.docker_bin,
                    docker_config=docker_lease.docker_config,
                    base_env=base_env,
                )
                if not isolation_image:
                    raise ValueError(
                        "Docker Grok isolation image could not be pinned locally"
                    )
                cmd = _docker_grok_command(
                    grok_command=cmd,
                    grok_bin=Path(grok_bin).resolve(strict=True),
                    workspace=workspace,
                    prompt_path=Path(prompt_path).resolve(strict=True),
                    grok_home=_policy_path.parent,
                    base_env=base_env,
                    child_env=env,
                    denied_paths=_denied_paths,
                    mask_root=docker_lease.lease_root / "provider-masks",
                    docker_config=docker_lease.docker_config,
                    container_name=docker_lease.container_name,
                    cidfile=docker_lease.cidfile,
                    docker_bin=docker_lease.docker_bin,
                    isolation_image=isolation_image,
                )
                # Docker is pinned to the validated local socket and empty
                # runner-owned config. Only explicitly named sanitized
                # variables cross into Grok via ``--env NAME`` arguments.
                grok_launch_env = _docker_control_env(env)
        except (
            LLMRouterError,
            ProviderCommandEnvironmentError,
            ValidationRuntimeError,
            OSError,
            RuntimeError,
            ValueError,
        ) as exc:
            print(str(exc), file=sys.stderr)
            return 2

        os.chdir(workspace)
        if codex_fallback_command:
            try:
                workspace_baseline = _workspace_content_fingerprint(workspace)
            except ValueError as exc:
                print(str(exc), file=sys.stderr)
                return 2
        failure_type = ""
        if codex_fallback_command:
            output_index = cmd.index("--output-format") + 1
            cmd[output_index] = "streaming-json"
            try:
                primary_returncode = _run_grok_with_typed_failure_capture(
                    cmd,
                    env=grok_launch_env,
                )
                docker_run_finished = True
            except OSError as exc:
                print(f"unable to launch Grok CLI: {exc}", file=sys.stderr)
                return 127
        else:
            completed = subprocess.run(cmd, env=grok_launch_env, check=False)
            docker_run_finished = True
            primary_returncode = int(completed.returncode)
        if primary_returncode == 0 or not codex_fallback_command:
            return primary_returncode

        try:
            workspace_after_primary = _workspace_content_fingerprint(workspace)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return primary_returncode
        if workspace_after_primary != workspace_baseline:
            print(
                "Grok changed the workspace before failing; Codex fallback is "
                "forbidden until the supervisor restores a clean attempt",
                file=sys.stderr,
            )
            return primary_returncode

        failure_type = _terminal_grok_failure_type_from_isolated_home(
            Path(grok_launch_env["GROK_HOME"]),
            expected_session_id=primary_session_id,
        )
        if failure_type not in GROK_QUOTA_ERROR_TYPES:
            print(
                "Grok CLI failed without a terminal-correlated native quota "
                "record; Codex fallback is forbidden",
                file=sys.stderr,
            )
            return primary_returncode

        verifier_failure_type = _independently_verify_grok_quota(
            grok_bin=grok_bin,
            base_env=os.environ.copy(),
        )
        if verifier_failure_type not in GROK_QUOTA_ERROR_TYPES:
            print(
                "Independent pinned Grok-4.5 verifier did not confirm quota; "
                "Codex fallback is forbidden",
                file=sys.stderr,
            )
            return primary_returncode

        try:
            workspace_before_fallback = _workspace_content_fingerprint(workspace)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return primary_returncode
        if workspace_before_fallback != workspace_baseline:
            print(
                "The workspace changed while Grok quota was being verified; "
                "Codex fallback is forbidden",
                file=sys.stderr,
            )
            return primary_returncode

        print(
            "Grok quota exhausted; invoking the pinned Terra/medium fallback",
            file=sys.stderr,
        )
        try:
            fallback_env = _codex_quota_fallback_env(
                workspace=workspace,
                base_env=os.environ.copy(),
            )
            fallback = subprocess.run(
                codex_fallback_command,
                cwd=workspace,
                env=fallback_env,
                input=prompt,
                text=True,
                check=False,
            )
        except (OSError, ValueError) as exc:
            print(f"unable to launch Codex fallback: {exc}", file=sys.stderr)
            return 127
        return int(fallback.returncode)
    finally:
        command_environment_stack.close()
        if docker_lease is not None:
            docker_lease.close(docker_run_finished=docker_run_finished)
        if isolated_home is not None:
            _robust_remove_runner_temp_tree(Path(isolated_home.name))
            isolated_home.cleanup()
        if prompt_path:
            try:
                os.unlink(prompt_path)
            except OSError:
                pass


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
            "records a terminal-correlated quota failure and an independent "
            "tool-free Grok verifier confirms it; forced-Grok routes omit "
            "this option."
        ),
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
    args.receipt_fd_declared = bool(
        os.environ.get(GROK_TERMINAL_RECEIPT_FD_ENV, "").strip()
    )
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
    if len(sys.argv) > 1 and sys.argv[1] == _DOCKER_CLEANUP_WATCHDOG_ARG:
        raise SystemExit(_docker_cleanup_watchdog_main(sys.argv[2:]))
    raise SystemExit(main())
