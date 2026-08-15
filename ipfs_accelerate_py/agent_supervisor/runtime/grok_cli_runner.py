#!/usr/bin/env python3
"""Supervised Grok Build CLI entry for implementation worktrees.

The runner keeps ordinary Grok output live while parsing only top-level
``streaming-json`` frames. A terminal, typed account-quota error is projected
as an untrusted candidate over a file descriptor not directly inherited by
Grok. Same-UID descendants can still inject into the Grok stdout pipe through
procfs, so exit 86 and this candidate are diagnostics, never fallback proof.
Only an exact pre-effect authentication finding or independently confirmed
quota evidence may authorize the isolated fallback boundary.
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
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import TextIO

_PACKAGE_ROOT = Path(__file__).resolve().parents[3]
# This file is launched by absolute path for scoped routes.  Put its accepted
# package capsule first and remove the writable candidate cwd from import
# search before importing any project module.  The candidate remains available
# only through the explicit ``--workspace`` argument.
_STARTUP_CWD = Path.cwd().resolve(strict=False)
_accepted_root_text = str(_PACKAGE_ROOT)
sys.path[:] = [
    _accepted_root_text,
    *[
        entry
        for entry in sys.path
        if entry
        and Path(entry).resolve(strict=False) != _STARTUP_CWD
        and Path(entry).resolve(strict=False) != _PACKAGE_ROOT
    ],
]

from ipfs_accelerate_py.agent_supervisor.runtime.provider_command_binding import (
    ensure_provider_command_bindings,
    recover_provider_command_name_error,
    scan_source_for_provider_command_names,
)
from ipfs_accelerate_py.agent_supervisor.runtime.provider_command_environment import (
    FORMAL_TOOLCHAIN_CONTRACT_SHA256_ENV,
    PROVIDER_COMMAND_ENV_DIGEST_ENV,
    PROVIDER_COMMAND_ENV_WRAPPER_ENV,
    PROVIDER_COMMAND_REQUIRED_COMMANDS_ENV,
    ProviderCommandEnvironmentError,
    sealed_provider_command_environment,
)
from ipfs_accelerate_py.agent_supervisor.runtime.provider_failure_policy import (
    GROK_FAILURE_RECEIPT_PREFIX,
    GROK_QUOTA_PROBE_PROMPT,
    GROK_QUOTA_PROBE_TIMEOUT_SECONDS,
    GROK_ROUTE_OUTCOME_PREFIX,
    MAX_GROK_FAILURE_EVIDENCE_BYTES,
    build_grok_failure_receipt,
    build_grok_route_outcome,
    render_grok_failure_receipt,
    render_grok_route_outcome,
    valid_grok_failure_receipt,
)
from ipfs_accelerate_py.agent_supervisor.validation.validation_runtime import (
    ValidationRuntimeError,
)
from ipfs_accelerate_py.llm_router import (
    AGENT_IMPLEMENTATION_CODEX_IMAGE_ID,
    AGENT_IMPLEMENTATION_CODEX_IMAGE_LABEL,
    AGENT_IMPLEMENTATION_ROUTE_OUTCOME_PREFIX,
    AGENT_IMPLEMENTATION_QUOTA_VERIFIER_DISALLOWED_TOOLS,
)

# Self-heal: if a static import is incomplete on an older pin or partial merge,
# bind every provider-command symbol this module loads by name.
try:
    _SOURCE = Path(__file__).read_text(encoding="utf-8")
    _REQUIRED_PROVIDER_COMMAND_SYMBOLS = scan_source_for_provider_command_names(
        _SOURCE
    )
except OSError:
    _REQUIRED_PROVIDER_COMMAND_SYMBOLS = frozenset(
        {
            "FORMAL_TOOLCHAIN_CONTRACT_SHA256_ENV",
            "PROVIDER_COMMAND_ENV_DIGEST_ENV",
            "PROVIDER_COMMAND_ENV_WRAPPER_ENV",
            "PROVIDER_COMMAND_REQUIRED_COMMANDS_ENV",
            "ProviderCommandEnvironmentError",
            "sealed_provider_command_environment",
        }
    )
ensure_provider_command_bindings(
    globals(),
    required=_REQUIRED_PROVIDER_COMMAND_SYMBOLS,
    namespace_name=__name__,
    strict=False,
)

DEFAULT_GROK_MODEL = "grok-4.6"
# Grok CLI validates --max-turns as 1..=4294967295 (u32::MAX).
DEFAULT_GROK_MAX_TURNS = 4_294_967_295
GROK_QUOTA_EXHAUSTED_EXIT_CODE = 86
GROK_QUOTA_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/grok-quota-error@1"
)
MAX_GROK_ERROR_BYTES = 128 * 1024
_SCOPED_ROUTE_MAX_AGE_MS = 5 * 60 * 1000


def _agent_prompt_cid(prompt: str) -> str:
    return "sha256:" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()
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
    cwd: Path,
) -> tuple[int, str, int, bool]:
    """Run the fixed probe while retaining only a bounded stderr tail.

    ``subprocess.run(..., stderr=PIPE)`` buffers the complete provider output
    before returning.  Besides permitting unbounded memory growth, taking a
    trusted tail afterwards can erase an earlier conflicting 403/429 signal.
    Drain concurrently, count every byte, and surface overflow as explicit
    fail-closed metadata to the route decision.
    """

    process = subprocess.Popen(
        list(command),
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        close_fds=True,
    )
    if process.stderr is None:
        raise RuntimeError("isolated Grok quota probe stderr pipe was not created")
    retained = bytearray()
    byte_count = 0

    def drain_stderr() -> None:
        nonlocal byte_count
        while True:
            chunk = process.stderr.read(16 * 1024)
            if not chunk:
                return
            byte_count += len(chunk)
            retained.extend(chunk)
            if len(retained) > MAX_GROK_FAILURE_EVIDENCE_BYTES:
                del retained[:-MAX_GROK_FAILURE_EVIDENCE_BYTES]

    drain_thread = threading.Thread(
        target=drain_stderr,
        name="grok-quota-probe-stderr",
        daemon=True,
    )
    drain_thread.start()
    timed_out = False
    try:
        returncode = int(
            process.wait(timeout=GROK_QUOTA_PROBE_TIMEOUT_SECONDS)
        )
    except subprocess.TimeoutExpired:
        timed_out = True
        process.kill()
        returncode = 124
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    finally:
        drain_thread.join(timeout=5)
        try:
            process.stderr.close()
        except OSError:
            pass
    if drain_thread.is_alive():
        raise RuntimeError("isolated Grok quota probe stderr drain did not finish")
    if timed_out and not retained:
        retained.extend(b"isolated Grok quota probe timeout")
        byte_count = len(retained)
    return (
        returncode,
        retained.decode("utf-8", errors="replace"),
        byte_count,
        byte_count > MAX_GROK_FAILURE_EVIDENCE_BYTES,
    )


MAX_CODEX_FALLBACK_ARGUMENTS = 64
MAX_CODEX_FALLBACK_ARGUMENT_BYTES = 4_096
CODEX_QUOTA_FALLBACK_MODEL = "gpt-5.6-terra"
DEFAULT_CODEX_QUOTA_FALLBACK_REASONING_EFFORT = "medium"
CODEX_QUOTA_FALLBACK_REASONING_EFFORTS = frozenset({"medium", "high"})
CANONICAL_LEGACY_PREFLIGHT_ROUTE_FLAG = (
    "--canonical-legacy-preflight-route"
)
GROK_PRIMARY_SANDBOX_PROFILE = "ipfs-accelerate-provider-isolated"
GROK_ISOLATION_GROK_SANDBOX = "grok-sandbox"
GROK_ISOLATION_DOCKER = "docker"
DEFAULT_GROK_ISOLATION_IMAGE = "ubuntu:24.04"
_DOCKER_LOCAL_HOST = "unix:///var/run/docker.sock"
_DOCKER_CLEANUP_WATCHDOG_ARG = "--internal-docker-cleanup-watchdog"
_CODEX_CONTAINER_HOME = Path("/opt/codex-home")
_CODEX_CONTAINER_AUTH_PATH = _CODEX_CONTAINER_HOME / "auth.json"
class _AgentRouteEffectDenied(ValueError):
    """The canonical route lost authority before the provider effect."""
_CODEX_TASK_TOOLCHAIN_IMAGE_ID = AGENT_IMPLEMENTATION_CODEX_IMAGE_ID
_CODEX_TASK_TOOLCHAIN_IMAGE_LABEL = AGENT_IMPLEMENTATION_CODEX_IMAGE_LABEL
_CODEX_TASK_TOOLCHAIN_SITE_PACKAGES = Path(
    "/opt/ipfs-validation-site-packages"
)
_CODEX_TASK_TOOLCHAIN_BIN = Path("/opt/ipfs-task-tools/bin")
_CODEX_TASK_TOOLCHAIN_PYTHON = _CODEX_TASK_TOOLCHAIN_BIN / "python"
_HOST_CODEX_TASK_TOOLCHAIN_PYTHON = Path("/usr/bin/python3.12")
_CODEX_DOCKER_IMAGE_ENV_OVERRIDES = (
    "BASH_ENV=",
    "CUDA_VISIBLE_DEVICES=-1",
    "ENV=",
    "LD_LIBRARY_PATH=",
    "LD_PRELOAD=",
    "LIBRARY_PATH=",
    "NVIDIA_DRIVER_CAPABILITIES=",
    "NVIDIA_REQUIRE_CUDA=",
    "NVIDIA_REQUIRE_JETPACK_HOST_MOUNTS=",
    "NVIDIA_VISIBLE_DEVICES=void",
)
_DOCKER_CONTAINER_NAME_RE = re.compile(
    r"ipfs-accelerate-(?:grok|codex)-[0-9]+-[0-9a-f]{32}"
)
_DOCKER_ISOLATION_PROVIDERS = frozenset({"grok", "codex"})
_DOCKER_CLEANUP_TIMEOUT_SECONDS = 8.0
_DOCKER_CAS_ABSENT_GRACE_SECONDS = 120.0
_DOCKER_INSPECTION_MAX_BYTES = 256 * 1024
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
    fallback_reasoning_effort: str = (
        DEFAULT_CODEX_QUOTA_FALLBACK_REASONING_EFFORT
    ),
    enable_codex_fallback: bool = True,
    enable_internal_legacy_preflight: bool = False,
    accepted_runner_path: str | Path = "",
) -> list[str]:
    """Build a sealed Grok-4.5 then typed-failure Terra route.

    The returned parent runner owns the Codex argv.  Grok receives neither the
    executable/auth authority nor any way to invoke this fallback directly.
    """

    workspace_text = str(workspace)
    reasoning_effort = str(fallback_reasoning_effort).strip()
    if reasoning_effort not in CODEX_QUOTA_FALLBACK_REASONING_EFFORTS:
        raise ValueError("Codex fallback reasoning must be medium or high")
    codex = (
        resolve_codex_quota_fallback_executable(
            workspace=workspace,
            configured=codex_bin,
        )
        if enable_codex_fallback
        else ""
    )
    runner = str(accepted_runner_path or "").strip()
    runner_argv = (
        ["-I", runner]
        if runner
        else ["-m", "ipfs_accelerate_py.agent_supervisor.grok_cli_runner"]
    )
    if runner and (not Path(runner).is_absolute() or not Path(runner).is_file()):
        raise ValueError("accepted Grok runner must be an absolute file")
    command = [
        str(python_executable or sys.executable),
        *runner_argv,
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
            f'model_reasoning_effort="{reasoning_effort}"',
            "-",
        ]
        command.extend(
            [
                "--codex-fallback-command-json",
                json.dumps(fallback, separators=(",", ":")),
            ]
        )
        if enable_internal_legacy_preflight:
            command.append(CANONICAL_LEGACY_PREFLIGHT_ROUTE_FLAG)
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
# Compatibility export for the legacy physical runner used by the supervised
# Grok-to-Codex adapter.  ``agent_supervisor.__init__`` redirects the public
# ``grok_cli_runner`` module name here, while that adapter still launches the
# physical entrypoint and passes its private failure-receipt descriptor.
TRUSTED_FAILURE_RECEIPT_FD_ENV = (
    "IPFS_ACCELERATE_AGENT_TRUSTED_FAILURE_RECEIPT_FD"
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


def _repository_head(workspace: Path) -> str:
    """Return the exact repository HEAD without accepting symbolic prose."""

    git_environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("GIT_")
    }
    git_environment.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "LC_ALL": "C",
            "LANG": "C",
        }
    )
    completed = subprocess.run(
        [
            "git",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.hooksPath=/dev/null",
            "rev-parse",
            "--verify",
            "HEAD^{commit}",
        ],
        cwd=workspace,
        env=git_environment,
        stdin=subprocess.DEVNULL,
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )
    head = completed.stdout.strip()
    if completed.returncode != 0 or re.fullmatch(r"[0-9a-f]{40}", head) is None:
        raise ValueError("agent implementation route requires a pinned repository HEAD")
    return head


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


def _docker_codex_task_toolchain_image_id(
    docker_bin: str,
    *,
    docker_config: Path,
) -> str:
    """Verify the immutable image that supplies the bounded test toolchain."""

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
                (
                    '{{.Id}}|{{.Os}}|{{.Architecture}}|'
                    '{{index .Config.Labels '
                    '"org.ipfs-accelerate.authority-validation"}}'
                ),
                AGENT_IMPLEMENTATION_CODEX_IMAGE_ID,
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
    expected = (
        f"{AGENT_IMPLEMENTATION_CODEX_IMAGE_ID}|linux|arm64|"
        f"{AGENT_IMPLEMENTATION_CODEX_IMAGE_LABEL}"
    )
    return (
        AGENT_IMPLEMENTATION_CODEX_IMAGE_ID
        if completed.returncode == 0 and completed.stdout.strip() == expected
        else ""
    )


def _host_codex_task_toolchain_python() -> Path:
    """Resolve the exact root-owned Python ABI used by the pinned toolchain."""

    entry = _HOST_CODEX_TASK_TOOLCHAIN_PYTHON
    try:
        entry_stat = entry.lstat()
        resolved = entry.resolve(strict=True)
        resolved_stat = resolved.stat()
    except OSError as exc:
        raise ValueError("Codex task Python toolchain is unavailable") from exc
    if (
        entry != resolved
        or not stat.S_ISREG(entry_stat.st_mode)
        or not stat.S_ISREG(resolved_stat.st_mode)
        or resolved_stat.st_uid != 0
        or resolved_stat.st_mode & 0o022
        or not os.access(resolved, os.X_OK)
    ):
        raise ValueError("Codex task Python toolchain is not trusted")
    return resolved


def _codex_task_container_environment() -> dict[str, str]:
    """Return the complete non-secret environment admitted past ``env -i``."""

    return {
        "BASH_ENV": "",
        "CODEX_HOME": str(_CODEX_CONTAINER_HOME),
        "ENV": "",
        "HOME": str(_CODEX_CONTAINER_HOME),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": f"{_CODEX_TASK_TOOLCHAIN_BIN}:/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": str(_CODEX_TASK_TOOLCHAIN_SITE_PACKAGES),
        "TERM": "dumb",
    }


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


def _effect_receipt_identity(value: object) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _runner_process_start_ticks(pid: int) -> int:
    """Return one Linux process birth identity without trusting its argv."""

    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise ValueError("process identity is invalid")
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(
            encoding="ascii"
        ).split()
        ticks = int(fields[21])
    except (OSError, IndexError, UnicodeError, ValueError) as exc:
        raise ValueError("process identity is unavailable") from exc
    if ticks < 0:
        raise ValueError("process identity is invalid")
    return ticks


def _runner_process_identity_alive(pid: int, start_ticks: int) -> bool:
    try:
        return _runner_process_start_ticks(pid) == start_ticks
    except ValueError:
        return False


def _docker_runtime_receipt_identity(docker_bin: str) -> str:
    return _effect_receipt_identity(_docker_runtime_receipt(docker_bin))


def _docker_runtime_receipt(docker_bin: str) -> dict[str, object]:
    """Return the full stable identity behind a Docker runtime digest."""

    runtime_path = Path(docker_bin).resolve(strict=True)
    runtime_stat = runtime_path.stat()
    if (
        runtime_path
        not in {Path("/usr/bin/docker"), Path("/usr/local/bin/docker")}
        or runtime_stat.st_uid != 0
        or runtime_stat.st_mode & 0o022
    ):
        raise ValueError("Docker runtime identity is not trusted")
    return {
        "path": str(runtime_path),
        "device": runtime_stat.st_dev,
        "inode": runtime_stat.st_ino,
        "mode": runtime_stat.st_mode,
        "uid": runtime_stat.st_uid,
        "size": runtime_stat.st_size,
        "mtime_ns": runtime_stat.st_mtime_ns,
        "ctime_ns": runtime_stat.st_ctime_ns,
    }


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
        # ``docker rm`` also returns nonzero when the exact container is
        # already absent.  Prove that benign/idempotent case with a separate
        # bounded exact-name listing; a Docker control-plane failure must not
        # be mistaken for successful cleanup.
        try:
            observed = subprocess.run(
                [
                    docker_bin,
                    f"--host={_DOCKER_LOCAL_HOST}",
                    "--config",
                    str(docker_config),
                    "container",
                    "ls",
                    "--all",
                    "--no-trunc",
                    "--filter",
                    f"name=^/{container_name}$",
                    "--format",
                    "{{.Names}}",
                ],
                env=_docker_control_env(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=2,
                check=False,
            )
            if (
                observed.returncode == 0
                and len(observed.stdout) <= _DOCKER_INSPECTION_MAX_BYTES
                and not observed.stdout.strip()
            ):
                return
        except (OSError, subprocess.TimeoutExpired):
            pass
        now = time.monotonic()
        if now >= deadline or not settle_for_creation:
            raise ValueError(
                "exact Docker container cleanup could not be verified"
            )
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
    parser.add_argument(
        "--provider",
        required=True,
        choices=tuple(sorted(_DOCKER_ISOLATION_PROVIDERS)),
    )
    parser.add_argument("--docker-bin", required=True)
    parser.add_argument("--container-name", required=True)
    parser.add_argument("--cidfile", type=Path, required=True)
    parser.add_argument("--lease-root", type=Path, required=True)
    parser.add_argument("--provider-home", type=Path, required=True)
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
    provider_home = args.provider_home.absolute()
    prompt_path = args.prompt_path.absolute()
    cas_marker = lease_root / "cas-owned"
    terminal_marker = lease_root / "cas-terminal"
    temporary_root = Path(tempfile.gettempdir()).resolve()
    expected_container_prefix = f"ipfs-accelerate-{args.provider}-"
    if (
        docker_path not in {Path("/usr/bin/docker"), Path("/usr/local/bin/docker")}
        or docker_path.name not in {"docker", "docker.exe"}
        or docker_stat.st_uid != 0
        or docker_stat.st_mode & 0o022
        or _DOCKER_CONTAINER_NAME_RE.fullmatch(args.container_name) is None
        or not args.container_name.startswith(expected_container_prefix)
        or lease_root.parent != temporary_root
        or not lease_root.name.startswith(
            f"asref-{args.provider}-container-"
        )
        or cidfile.parent != lease_root
        or cidfile.name != "container.cid"
        or not docker_config.is_dir()
        or provider_home.parent != temporary_root
        or not provider_home.name.startswith(
            f"asref-{args.provider}-home-"
        )
        or prompt_path.parent != temporary_root
        or not prompt_path.name.startswith("asref-grok-prompt-")
    ):
        return 2

    # A clean marker means docker-run returned, but the final rm remains a
    # defensive idempotent action.  Empty input means the runner was killed;
    # retry briefly to cover a daemon-side container-creation race.
    cleanup_started = False
    cleanup_succeeded = False
    cleanup_failed = False

    def cas_owned() -> bool:
        try:
            metadata = os.lstat(cas_marker)
        except FileNotFoundError:
            return False
        except OSError:
            return True
        return bool(
            stat.S_ISREG(metadata.st_mode)
            and metadata.st_uid == os.geteuid()
            and metadata.st_nlink == 1
            and stat.S_IMODE(metadata.st_mode) == 0o600
        )

    def cas_terminal() -> bool:
        try:
            metadata = os.lstat(terminal_marker)
        except FileNotFoundError:
            return False
        except OSError:
            return False
        return bool(
            stat.S_ISREG(metadata.st_mode)
            and metadata.st_uid == os.geteuid()
            and metadata.st_nlink == 1
            and stat.S_IMODE(metadata.st_mode) == 0o600
        )

    def cleanup(*, settle_for_creation: bool) -> bool:
        nonlocal cleanup_started, cleanup_succeeded, cleanup_failed
        if cleanup_started:
            return cleanup_succeeded
        cleanup_started = True
        try:
            _remove_exact_docker_container(
                docker_bin=str(docker_path),
                docker_config=docker_config,
                container_name=args.container_name,
                settle_for_creation=settle_for_creation,
            )
        except ValueError:
            cleanup_failed = True
            return False
        cleanup_succeeded = True
        return True

    def terminate_watchdog(signum: int, _frame: object) -> None:
        # The supervisor deliberately terminates separately owned descendant
        # process groups before the runner group.  Reap synchronously here so
        # that ordering cannot strand the runner-owned workspace mount.
        if cas_owned() and not cas_terminal():
            # Recovery owns every path needed to inspect/start the inert or
            # running exact container.  Supervisor shutdown must not delete
            # those bind/config inputs before a durable terminal transition.
            return
        # A terminal marker is durable accounting authority.  Reap the exact
        # container synchronously *before* finally removes the Docker config
        # and bind sources.  This covers marker->SIGTERM interleavings where
        # the polling loop has not observed the terminal transition yet.
        if not cleanup(settle_for_creation=True):
            raise SystemExit(125)
        raise SystemExit(128 + signum)

    try:
        signal.signal(signal.SIGTERM, terminate_watchdog)
        signal.signal(signal.SIGINT, terminate_watchdog)
        markers = sys.stdin.buffer.read(16)
        clean_exit = b"C" in markers
        if not cas_owned():
            cleanup(settle_for_creation=not clean_exit)
        elif cas_terminal():
            cleanup(settle_for_creation=False)
        else:
            # A durable effect_started claim transfers cleanup priority to
            # recovery.  Container absence or exit is not proof that the
            # provider effect never ran, so the watchdog must preserve the
            # exact container until the CAS terminal record has been written.
            while True:
                if cas_terminal():
                    cleanup(settle_for_creation=True)
                    break
                time.sleep(1.0)
    finally:
        # Never destroy the inputs required for a later exact retry unless
        # absence/removal of the receipt-bound container was proven.  A dead
        # reaper with preserved private inputs is recoverable; deleting those
        # inputs after an unverified rm is not.
        if cleanup_succeeded:
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
            _robust_remove_runner_temp_tree(provider_home)
            try:
                cidfile.unlink()
            except FileNotFoundError:
                pass
            try:
                cas_marker.unlink()
            except FileNotFoundError:
                pass
            try:
                terminal_marker.unlink()
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
    return 125 if cleanup_failed or not cleanup_succeeded else 0


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
        provider_home: Path,
        prompt_path: Path,
        write_fd: int,
        watchdog: subprocess.Popen[bytes],
    ) -> None:
        self.docker_bin = docker_bin
        self.container_name = container_name
        self.lease_root = lease_root
        self.docker_config = docker_config
        self.cidfile = cidfile
        self.provider_home = provider_home
        self.prompt_path = prompt_path
        self._write_fd = write_fd
        self._watchdog = watchdog
        self._closed = False
        self._cas_owned = False
        self._cas_terminal = False
        self.preserve_for_recovery = False

    @classmethod
    def create(
        cls,
        docker_bin: str,
        *,
        provider: str,
        provider_home: Path,
        prompt_path: Path,
    ) -> "_DockerContainerLease":
        if provider not in _DOCKER_ISOLATION_PROVIDERS:
            raise ValueError("Docker isolation provider is invalid")
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
            tempfile.mkdtemp(prefix=f"asref-{provider}-container-")
        ).resolve()
        cidfile = lease_root / "container.cid"
        docker_config = lease_root / "docker-config"
        docker_config.mkdir(mode=0o700)
        container_name = (
            f"ipfs-accelerate-{provider}-{os.getpid()}-{uuid.uuid4().hex}"
        )
        read_fd, write_fd = os.pipe()
        sealed_match = re.fullmatch(
            r"/proc/self/fd/([0-9]+)",
            str(sys.argv[0]),
        )
        runner_entry = (
            str(sys.argv[0])
            if sealed_match is not None
            else str(Path(__file__).resolve())
        )
        inherited_control_plane = (
            (int(sealed_match.group(1)),)
            if sealed_match is not None
            else ()
        )
        try:
            watchdog = subprocess.Popen(
                [
                    sys.executable,
                    "-I",
                    runner_entry,
                    _DOCKER_CLEANUP_WATCHDOG_ARG,
                    "--provider",
                    provider,
                    "--docker-bin",
                    str(docker_path),
                    "--container-name",
                    container_name,
                    "--cidfile",
                    str(cidfile),
                    "--lease-root",
                    str(lease_root),
                    "--provider-home",
                    str(provider_home),
                    "--prompt-path",
                    str(prompt_path),
                ],
                stdin=read_fd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd="/",
                env=_docker_control_env(),
                start_new_session=True,
                close_fds=True,
                pass_fds=inherited_control_plane,
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
            provider_home=provider_home,
            prompt_path=prompt_path,
            write_fd=write_fd,
            watchdog=watchdog,
        )

    def mark_cas_owned(self) -> None:
        if self._cas_owned:
            return
        marker = self.lease_root / "cas-owned"
        descriptor = os.open(
            marker,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            os.write(descriptor, self.container_name.encode("ascii"))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self._cas_owned = True
        try:
            os.write(self._write_fd, b"A")
        except OSError as exc:
            raise ValueError("Docker CAS watchdog marker failed") from exc

    def mark_cas_terminal(self) -> None:
        """Release cleanup only after the durable CAS terminal write."""

        if not self._cas_owned:
            raise ValueError("Docker CAS terminal marker precedes effect claim")
        if self._cas_terminal:
            return
        marker = self.lease_root / "cas-terminal"
        descriptor = os.open(
            marker,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            payload = self.container_name.encode("ascii")
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written <= 0:
                    raise ValueError("Docker CAS terminal marker write failed")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self._cas_terminal = True
        try:
            os.write(self._write_fd, b"T")
        except OSError as exc:
            raise ValueError("Docker CAS terminal watchdog marker failed") from exc

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
        if self._cas_owned and not self._cas_terminal:
            # The runner may have died after the provider effect but before
            # durable completion.  Recovery owns both the container evidence
            # and its private Docker configuration from this point onward.
            self.preserve_for_recovery = True
            return
        try:
            self._watchdog.wait(timeout=_DOCKER_CLEANUP_TIMEOUT_SECONDS + 2)
        except subprocess.TimeoutExpired:
            try:
                _remove_exact_docker_container(
                    docker_bin=self.docker_bin,
                    docker_config=self.docker_config,
                    container_name=self.container_name,
                    settle_for_creation=False,
                )
            except ValueError:
                # Preserve the exact private cleanup inputs.  A later sealed
                # recovery can retry; destroying them here would make the
                # still-possible container unaccountable.
                self.preserve_for_recovery = True
                return
            self._watchdog.terminate()
            try:
                self._watchdog.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._watchdog.kill()
                self._watchdog.wait(timeout=2)
        if self.lease_root.exists():
            # The watchdog can exit between receiving its marker and proving
            # cleanup.  Do not infer success merely from process death.
            try:
                _remove_exact_docker_container(
                    docker_bin=self.docker_bin,
                    docker_config=self.docker_config,
                    container_name=self.container_name,
                    settle_for_creation=False,
                )
            except ValueError:
                self.preserve_for_recovery = True
                return
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
            (self.lease_root / "cas-owned").unlink()
        except FileNotFoundError:
            pass
        try:
            (self.lease_root / "cas-terminal").unlink()
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
        "create",
        "--pull=never",
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


def _run_created_grok_container_with_typed_failure_capture(
    create_command: Sequence[str],
    *,
    docker_bin: str,
    docker_config: Path,
    cidfile: Path,
    workspace: Path,
    env: dict[str, str],
) -> int:
    """Create the inert Grok container, then run that exact container.

    ``_docker_grok_command`` deliberately returns a ``docker create`` command
    so the container identity exists before any provider effect starts.  The
    ordinary task path must not mistake the successful create command's
    64-byte container ID for Grok output.  Validate both Docker's response and
    the runner-owned cidfile before attaching to the exact created container.
    """

    try:
        created = subprocess.run(
            list(create_command),
            cwd=workspace,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120.0,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError("Grok container creation timed out") from exc
    if (
        created.returncode != 0
        or len(created.stdout) > _DOCKER_INSPECTION_MAX_BYTES
        or len(created.stderr) > _DOCKER_INSPECTION_MAX_BYTES
    ):
        raise ValueError("Grok container could not be created")
    try:
        created_fields = created.stdout.decode("ascii", errors="strict").split()
        recorded_container_id = cidfile.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as exc:
        raise ValueError("Grok container identity is unavailable") from exc
    if (
        len(created_fields) != 1
        or re.fullmatch(r"[0-9a-f]{64}", created_fields[0]) is None
        or recorded_container_id != created_fields[0]
    ):
        raise ValueError("Grok container identity is invalid")
    start_command = [
        docker_bin,
        f"--host={_DOCKER_LOCAL_HOST}",
        "--config",
        str(docker_config),
        "start",
        "--attach",
        "--interactive",
        created_fields[0],
    ]
    return _run_grok_with_typed_failure_capture(start_command, env=env)


def _docker_codex_fallback_command(
    *,
    codex_command: Sequence[str],
    workspace: Path,
    source_auth: Path,
    child_env: dict[str, str],
    docker_config: Path,
    container_name: str,
    cidfile: Path,
    docker_bin: str,
    isolation_image: str,
) -> list[str]:
    """Wrap the pinned Codex fallback in a host-write-confined container."""

    docker = str(docker_bin)
    image = str(isolation_image).strip()
    if not docker or image != AGENT_IMPLEMENTATION_CODEX_IMAGE_ID:
        raise ValueError(
            "Codex fallback requires the exact pinned task-toolchain image"
        )
    if (
        _DOCKER_CONTAINER_NAME_RE.fullmatch(container_name) is None
        or not container_name.startswith("ipfs-accelerate-codex-")
    ):
        raise ValueError("Codex fallback container name is invalid")
    source_auth = _validated_codex_auth_path(
        source_auth=source_auth,
        workspace=workspace,
    )
    host_python = _host_codex_task_toolchain_python()
    expected_environment = _codex_task_container_environment()
    if child_env != expected_environment:
        raise ValueError("Codex fallback container environment is not sealed")

    _validate_codex_quota_fallback_command(
        codex_command,
        workspace=workspace,
    )
    inner = list(codex_command)
    sandbox_index = inner.index("-s")
    if inner[sandbox_index : sandbox_index + 2] != ["-s", "workspace-write"]:
        raise ValueError("Codex fallback sandbox descriptor is invalid")
    # Danger-full-access is safe only because Docker is now the enforcing
    # sandbox: the root filesystem and host /usr are read-only, only this
    # disposable worktree is writable, and no Docker socket or host home is
    # projected into the container. This avoids nested bwrap/userns failures
    # without widening host write authority. The container must be used only
    # for a trusted repository: API network access and exact Codex auth are
    # necessarily available to commands inside this external boundary.
    inner[sandbox_index + 1] = "danger-full-access"

    command = [
        docker,
        f"--host={_DOCKER_LOCAL_HOST}",
        "--config",
        str(docker_config),
        "create",
        "--pull=never",
        "--interactive",
        "--read-only",
        "--network=bridge",
        "--runtime=runc",
        "--entrypoint=/usr/bin/env",
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
        "--tmpfs",
        (
            f"{_CODEX_CONTAINER_HOME}:rw,nosuid,nodev,noexec,mode=0700,"
            f"uid={os.getuid()},gid={os.getgid()}"
        ),
        "--name",
        container_name,
        "--cidfile",
        str(cidfile),
        "--label",
        "ipfs_accelerate.codex_fallback_isolation=true",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=1024",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--workdir",
        str(workspace),
    ]
    for override in _CODEX_DOCKER_IMAGE_ENV_OVERRIDES:
        command.extend(["--env", override])

    host_usr = _existing_path(Path("/usr"))
    if host_usr is None:
        raise ValueError("Codex fallback requires the pinned host /usr toolchain")
    command.extend(_docker_mount(host_usr, read_only=True))
    host_ca_certificates = _existing_path(Path("/etc/ssl/certs"))
    if host_ca_certificates is None:
        raise ValueError("Codex fallback requires pinned host CA certificates")
    command.extend(_docker_mount(host_ca_certificates, read_only=True))
    command.extend(
        _docker_mount(
            host_python,
            destination=_CODEX_TASK_TOOLCHAIN_PYTHON,
            read_only=True,
        )
    )
    for git_root in _git_metadata_roots(workspace):
        command.extend(_docker_mount(git_root, read_only=True))
    command.extend(_docker_mount(workspace, read_only=False))
    git_control_path = _existing_path(workspace / ".git")
    if git_control_path is not None:
        command.extend(_docker_mount(git_control_path, read_only=True))
    command.extend(
        _docker_mount(
            source_auth,
            destination=_CODEX_CONTAINER_AUTH_PATH,
            read_only=True,
        )
    )
    # The authority-validation image contains a large CUDA-oriented Config.Env.
    # Clearing it here prevents ENV/BASH_ENV hooks and every unrelated image
    # default from reaching Codex or repository commands.  Only these fixed,
    # non-secret values are serialized; provider authority remains file-based.
    environment_assignments = [
        f"{name}={value}" for name, value in sorted(expected_environment.items())
    ]
    command.extend([image, "-i", *environment_assignments, *inner])
    return command


def _run_codex_quota_fallback_in_docker(
    codex_command: Sequence[str],
    *,
    workspace: Path,
    prompt: str,
    prompt_path: Path,
    base_env: dict[str, str],
    pre_effect_validator: Callable[[], None] | None = None,
    effect_claim: Callable[[Mapping[str, object]], None] | None = None,
    effect_terminal: Callable[[int], None] | None = None,
) -> int:
    """Run Codex only inside the available pinned external sandbox."""

    trusted_codex = resolve_codex_quota_fallback_executable(
        workspace=workspace,
        configured=str(codex_command[0] if codex_command else ""),
    )
    if not trusted_codex or trusted_codex != str(codex_command[0]):
        raise ValueError("Codex fallback executable lost its trusted identity")
    docker_bin = _docker_isolation_binary()
    if not docker_bin:
        raise ValueError("Codex fallback requires local Docker isolation")

    isolated_home: tempfile.TemporaryDirectory[str] | None = None
    docker_lease: _DockerContainerLease | None = None
    docker_run_finished = False
    try:
        isolated_home, child_env, source_auth = (
            _isolated_codex_quota_fallback_home(
                workspace=workspace,
                base_env=base_env,
            )
        )
        codex_home = Path(isolated_home.name)
        docker_lease = _DockerContainerLease.create(
            docker_bin,
            provider="codex",
            provider_home=codex_home,
            prompt_path=prompt_path,
        )
        isolation_image = _docker_codex_task_toolchain_image_id(
            docker_bin,
            docker_config=docker_lease.docker_config,
        )
        if not isolation_image:
            raise ValueError(
                "Codex fallback task-toolchain image is not pinned locally"
            )
        command = _docker_codex_fallback_command(
            codex_command=codex_command,
            workspace=workspace,
            source_auth=source_auth,
            child_env=child_env,
            docker_config=docker_lease.docker_config,
            container_name=docker_lease.container_name,
            cidfile=docker_lease.cidfile,
            docker_bin=docker_bin,
            isolation_image=isolation_image,
        )
        if pre_effect_validator is not None:
            # Validate the route before the final auth check so an auth swap
            # performed during route validation is caught below.
            pre_effect_validator()
        _validated_codex_auth_path(
            source_auth=source_auth,
            workspace=workspace,
        )
        if pre_effect_validator is not None:
            # Revalidate the route again as the final operation before the
            # only external implementation effect. The preceding auth check
            # and this route check form the narrowest fail-closed boundary
            # available to the path-based Docker CLI handoff.
            pre_effect_validator()
        docker_environment = _docker_control_env(child_env)
        try:
            created = subprocess.run(
                command,
                cwd=workspace,
                env=docker_environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=120.0,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ValueError("Codex fallback container creation timed out") from exc
        if (
            created.returncode != 0
            or len(created.stdout) > _DOCKER_INSPECTION_MAX_BYTES
            or len(created.stderr) > _DOCKER_INSPECTION_MAX_BYTES
        ):
            raise ValueError("Codex fallback container could not be created")
        created_fields = created.stdout.decode("ascii", errors="strict").split()
        if (
            len(created_fields) != 1
            or re.fullmatch(r"[0-9a-f]{64}", created_fields[0]) is None
        ):
            raise ValueError("Codex fallback container identity is invalid")
        container_id = "sha256:" + created_fields[0]
        start_command = [
            docker_bin,
            f"--host={_DOCKER_LOCAL_HOST}",
            "--config",
            str(docker_lease.docker_config),
            "start",
            "--attach",
            "--interactive",
            created_fields[0],
        ]
        # Container creation is inert.  Only after its immutable identity is
        # known do we durably claim the exact logical effect and release the
        # start gate.  A crash can therefore adopt this same container; no
        # later Docker child can materialize an as-yet-unrecorded name.
        if effect_claim is not None:
            mount_arguments = [
                command[index + 1]
                for index, item in enumerate(command[:-1])
                if item in {"--mount", "--volume", "-v"}
            ]
            runtime_receipt = _docker_runtime_receipt(docker_bin)
            command_receipt = {
                "create_argv": list(command),
                "start_argv": list(start_command),
                "provider_argv": [str(item) for item in codex_command],
            }
            mount_receipt = list(mount_arguments)
            environment_receipt = {
                "docker_cli": dict(sorted(docker_environment.items())),
                "container": dict(
                    sorted(_codex_task_container_environment().items())
                ),
            }
            image_receipt = {
                "image_id": isolation_image,
                "image_label": AGENT_IMPLEMENTATION_CODEX_IMAGE_LABEL,
            }
            cleanup_receipt: dict[str, object] = {
                "schema": (
                    "ipfs_accelerate_py.agent_supervisor."
                    "provider-effect-cleanup@1"
                ),
                "lease_root": str(docker_lease.lease_root),
                "docker_config": str(docker_lease.docker_config),
                "cidfile": str(docker_lease.cidfile),
                "provider_home": str(docker_lease.provider_home),
                "prompt_path": str(docker_lease.prompt_path),
                "watchdog_pid": docker_lease._watchdog.pid,
                "watchdog_start_ticks": _runner_process_start_ticks(
                    docker_lease._watchdog.pid
                ),
            }
            cleanup_receipt["receipt_id"] = _effect_receipt_identity(
                cleanup_receipt
            )
            launch_context = {
                "provider_id": "codex",
                "command_id": _effect_receipt_identity(command_receipt),
                "runtime_id": _effect_receipt_identity(runtime_receipt),
                "image_id": isolation_image,
                "mount_id": _effect_receipt_identity(mount_receipt),
                "environment_id": _effect_receipt_identity(
                    environment_receipt
                ),
                "cleanup_id": cleanup_receipt["receipt_id"],
                "container_name": docker_lease.container_name,
                "container_id": container_id,
                "runtime_receipt": runtime_receipt,
                "image_receipt": image_receipt,
                "command_receipt": command_receipt,
                "mount_receipt": mount_receipt,
                "environment_receipt": environment_receipt,
                "cleanup_receipt": cleanup_receipt,
            }
            _validated_codex_auth_path(
                source_auth=source_auth,
                workspace=workspace,
            )
            if pre_effect_validator is not None:
                # Docker creation may block for the full timeout.  Revalidate
                # freshness, lifecycle, HEAD, and the router decision only
                # after the exact inert container exists and immediately
                # before the once-only CAS/start boundary.
                pre_effect_validator()
            effect_claim(launch_context)
            # A concurrent loser must remain an ordinary inert lease so its
            # own container is removed.  Cleanup ownership transfers only
            # after this process has won the durable effect_started CAS.
            docker_lease.mark_cas_owned()
        process = subprocess.Popen(
            start_command,
            cwd=workspace,
            env=docker_environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        if process.stdin is None or process.stdout is None or process.stderr is None:
            raise RuntimeError("Codex fallback process pipes were not created")
        stdout_thread = threading.Thread(
            target=_stream_provider_pipe_without_reserved_records,
            args=(process.stdout, sys.stdout),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_stream_provider_pipe_without_reserved_records,
            args=(process.stderr, sys.stderr),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()
        try:
            process.stdin.write(prompt)
            process.stdin.close()
        except BrokenPipeError:
            pass
        returncode = int(process.wait())
        stdout_thread.join()
        stderr_thread.join()
        if effect_terminal is not None:
            # Persist the exact terminal outcome while Docker still retains
            # inspectable exit evidence.  Cleanup is released only after the
            # durable CAS transition succeeds.
            effect_terminal(returncode)
            if effect_claim is not None:
                docker_lease.mark_cas_terminal()
        docker_run_finished = True
        return returncode
    finally:
        if docker_lease is not None:
            docker_lease.close(docker_run_finished=docker_run_finished)
        if isolated_home is not None and not bool(
            getattr(docker_lease, "preserve_for_recovery", False)
        ):
            _robust_remove_runner_temp_tree(Path(isolated_home.name))
            isolated_home.cleanup()


def _bounded_docker_query(
    command: Sequence[str],
    *,
    timeout: float = 15.0,
) -> tuple[int, bytes, bytes]:
    try:
        completed = subprocess.run(
            list(command),
            env=_docker_control_env(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("recorded Docker effect inspection failed") from exc
    if (
        len(completed.stdout) > _DOCKER_INSPECTION_MAX_BYTES
        or len(completed.stderr) > _DOCKER_INSPECTION_MAX_BYTES
    ):
        raise ValueError("recorded Docker effect inspection was oversized")
    return int(completed.returncode), completed.stdout, completed.stderr


def _recorded_codex_lease_root(
    launch_receipt: Mapping[str, object],
) -> tuple[Path, Path, str]:
    """Recover the winner's private watchdog lease from its exact argv."""

    command = launch_receipt.get("command_receipt")
    if not isinstance(command, Mapping):
        raise ValueError("recorded Docker cleanup command is unavailable")
    create_argv = command.get("create_argv")
    if not isinstance(create_argv, list) or any(
        not isinstance(item, str) for item in create_argv
    ):
        raise ValueError("recorded Docker cleanup command is invalid")
    try:
        config_index = create_argv.index("--config") + 1
        cidfile_index = create_argv.index("--cidfile") + 1
        config_path = Path(create_argv[config_index])
        cidfile_path = Path(create_argv[cidfile_index])
    except (IndexError, ValueError) as exc:
        raise ValueError("recorded Docker cleanup lease is invalid") from exc
    container_name = str(launch_receipt.get("container_name") or "")
    lease_root = config_path.parent
    if (
        not config_path.is_absolute()
        or config_path.name != "docker-config"
        or cidfile_path != lease_root / "container.cid"
        or lease_root.parent != Path(tempfile.gettempdir()).resolve()
        or not lease_root.name.startswith("asref-codex-container-")
        or _DOCKER_CONTAINER_NAME_RE.fullmatch(container_name) is None
        or container_name not in create_argv
    ):
        raise ValueError("recorded Docker cleanup lease identity is invalid")
    cursor = Path(lease_root.anchor)
    for component in lease_root.parts[1:]:
        cursor /= component
        metadata = os.lstat(cursor)
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError("recorded Docker cleanup lease contains a symlink")
    root_stat = os.lstat(lease_root)
    config_stat = os.lstat(config_path)
    if (
        not stat.S_ISDIR(root_stat.st_mode)
        or root_stat.st_uid != os.geteuid()
        or stat.S_IMODE(root_stat.st_mode) != 0o700
        or not stat.S_ISDIR(config_stat.st_mode)
        or config_stat.st_uid != os.geteuid()
        or stat.S_IMODE(config_stat.st_mode) != 0o700
    ):
        raise ValueError("recorded Docker cleanup lease is not private")
    return lease_root, config_path, container_name


def _release_recorded_codex_effect_cleanup(
    launch_receipt: Mapping[str, object],
) -> None:
    """Idempotently reap exact receipt-bound resources after CAS terminal."""

    cleanup = launch_receipt.get("cleanup_receipt")
    if not isinstance(cleanup, Mapping) or set(cleanup) != {
        "schema",
        "lease_root",
        "docker_config",
        "cidfile",
        "provider_home",
        "prompt_path",
        "watchdog_pid",
        "watchdog_start_ticks",
        "receipt_id",
    }:
        raise ValueError("recorded Docker cleanup receipt is invalid")
    cleanup_body = {
        key: item for key, item in cleanup.items() if key != "receipt_id"
    }
    if (
        cleanup.get("schema")
        != "ipfs_accelerate_py.agent_supervisor.provider-effect-cleanup@1"
        or cleanup.get("receipt_id") != _effect_receipt_identity(cleanup_body)
        or launch_receipt.get("cleanup_id") != cleanup.get("receipt_id")
    ):
        raise ValueError("recorded Docker cleanup receipt drifted")
    lease_root = Path(str(cleanup.get("lease_root") or ""))
    docker_config = Path(str(cleanup.get("docker_config") or ""))
    cidfile = Path(str(cleanup.get("cidfile") or ""))
    provider_home = Path(str(cleanup.get("provider_home") or ""))
    prompt_path = Path(str(cleanup.get("prompt_path") or ""))
    watchdog_pid = cleanup.get("watchdog_pid")
    watchdog_start_ticks = cleanup.get("watchdog_start_ticks")
    temporary_root = Path(tempfile.gettempdir()).resolve()
    if (
        lease_root.parent != temporary_root
        or not lease_root.name.startswith("asref-codex-container-")
        or docker_config != lease_root / "docker-config"
        or cidfile != lease_root / "container.cid"
        or provider_home.parent != temporary_root
        or not provider_home.name.startswith("asref-codex-home-")
        or prompt_path.parent != temporary_root
        or not prompt_path.name.startswith("asref-grok-prompt-")
        or isinstance(watchdog_pid, bool)
        or not isinstance(watchdog_pid, int)
        or watchdog_pid <= 0
        or isinstance(watchdog_start_ticks, bool)
        or not isinstance(watchdog_start_ticks, int)
        or watchdog_start_ticks < 0
    ):
        raise ValueError("recorded Docker cleanup paths are invalid")
    container_name = str(launch_receipt.get("container_name") or "")
    if not lease_root.exists():
        if provider_home.exists() or prompt_path.exists():
            raise ValueError("recorded Docker cleanup is partially missing")
        return
    observed_root, observed_config, observed_name = (
        _recorded_codex_lease_root(launch_receipt)
    )
    if (
        observed_root != lease_root
        or observed_config != docker_config
        or observed_name != container_name
    ):
        raise ValueError("recorded Docker cleanup lease drifted")
    marker = lease_root / "cas-terminal"
    payload = container_name.encode("ascii")
    try:
        descriptor = os.open(
            marker,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except FileExistsError:
        descriptor = os.open(
            marker,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            metadata = os.fstat(descriptor)
            observed = os.read(descriptor, len(payload) + 1)
        finally:
            os.close(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or observed != payload
        ):
            raise ValueError("recorded Docker terminal marker drifted")
    else:
        try:
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written <= 0:
                    raise ValueError("recorded Docker terminal marker write failed")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    directory = os.open(
        lease_root,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    if _runner_process_identity_alive(watchdog_pid, watchdog_start_ticks):
        try:
            os.kill(watchdog_pid, signal.SIGTERM)
        except OSError:
            pass
        for _ in range(20):
            if not _runner_process_identity_alive(
                watchdog_pid, watchdog_start_ticks
            ):
                break
            time.sleep(0.05)
    if _runner_process_identity_alive(watchdog_pid, watchdog_start_ticks):
        # A verified live watchdog owns the same exact receipt and will
        # consume the durable marker.  A later terminal replay retries if it
        # dies before cleanup.
        return
    # A terminal-aware watchdog performs Docker rm before deleting its
    # private inputs.  Recheck after its process has exited so a successful
    # synchronous signal-handler cleanup is not followed by an impossible
    # retry through a now-removed Docker config.
    if not lease_root.exists():
        if provider_home.exists() or prompt_path.exists():
            raise ValueError("recorded Docker cleanup is partially missing")
        return
    observed_root, observed_config, observed_name = (
        _recorded_codex_lease_root(launch_receipt)
    )
    if (
        observed_root != lease_root
        or observed_config != docker_config
        or observed_name != container_name
    ):
        raise ValueError("recorded Docker cleanup lease drifted")
    runtime = launch_receipt.get("runtime_receipt")
    docker_bin = str(runtime.get("path") or "") if isinstance(runtime, Mapping) else ""
    if docker_bin not in {"/usr/bin/docker", "/usr/local/bin/docker"}:
        raise ValueError("recorded Docker cleanup runtime is invalid")
    _remove_exact_docker_container(
        docker_bin=docker_bin,
        docker_config=docker_config,
        container_name=container_name,
        settle_for_creation=False,
    )
    try:
        prompt_path.unlink()
    except FileNotFoundError:
        pass
    _robust_remove_runner_temp_tree(provider_home)
    for candidate in (
        cidfile,
        lease_root / "cas-owned",
        marker,
    ):
        try:
            candidate.unlink()
        except FileNotFoundError:
            pass
    _robust_remove_runner_temp_tree(docker_config)
    try:
        lease_root.rmdir()
    except FileNotFoundError:
        pass


def _inspect_recorded_codex_effect(
    launch_receipt: Mapping[str, object],
    observed_at_ms: int,
) -> Mapping[str, object]:
    """Inspect only the CAS winner's exact Docker container and runtime."""

    container_name = str(launch_receipt.get("container_name") or "")
    recorded_container_id = str(launch_receipt.get("container_id") or "")
    if (
        _DOCKER_CONTAINER_NAME_RE.fullmatch(container_name) is None
        or not container_name.startswith("ipfs-accelerate-codex-")
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}", recorded_container_id
        ) is None
        or isinstance(observed_at_ms, bool)
        or not isinstance(observed_at_ms, int)
        or observed_at_ms <= 0
    ):
        raise ValueError("recorded Docker effect identity is invalid")
    docker_bin = _docker_isolation_binary()
    if not docker_bin:
        raise ValueError("recorded Docker runtime is unavailable")
    runtime_id = _docker_runtime_receipt_identity(docker_bin)
    if runtime_id != launch_receipt.get("runtime_id"):
        raise ValueError("recorded Docker runtime identity drifted")
    semantic_inspection = {
        "runtime_id": runtime_id,
        "host": _DOCKER_LOCAL_HOST,
        "operation": "container_inspect",
        "container_name": container_name,
        "container_id": recorded_container_id,
    }
    inspection_command_id = _effect_receipt_identity(semantic_inspection)
    with tempfile.TemporaryDirectory(
        prefix="asref-codex-adoption-docker-config-"
    ) as config_root:
        inspect_command = [
            docker_bin,
            f"--host={_DOCKER_LOCAL_HOST}",
            "--config",
            config_root,
            "container",
            "inspect",
            recorded_container_id.removeprefix("sha256:"),
        ]
        returncode, stdout, _stderr = _bounded_docker_query(inspect_command)
        if returncode != 0:
            list_command = [
                docker_bin,
                f"--host={_DOCKER_LOCAL_HOST}",
                "--config",
                config_root,
                "container",
                "ls",
                "--all",
                "--no-trunc",
                "--filter",
                f"name=^{container_name}$",
                "--format",
                "{{.ID}}",
            ]
            list_returncode, listed, _list_stderr = _bounded_docker_query(
                list_command
            )
            if list_returncode != 0 or listed.strip():
                raise ValueError(
                    "recorded Docker container could not be inspected"
                )
            status_value = "absent"
            container_id = ""
            container_returncode: int | None = None
        else:
            try:
                decoded = json.loads(stdout.decode("utf-8"))
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    "recorded Docker inspection is malformed"
                ) from exc
            if (
                not isinstance(decoded, list)
                or len(decoded) != 1
                or not isinstance(decoded[0], Mapping)
            ):
                raise ValueError("recorded Docker inspection is ambiguous")
            record = decoded[0]
            state = record.get("State")
            raw_container_id = str(record.get("Id") or "")
            if (
                record.get("Name") != "/" + container_name
                or record.get("Image") != launch_receipt.get("image_id")
                or not isinstance(state, Mapping)
                or re.fullmatch(r"[0-9a-f]{64}", raw_container_id) is None
            ):
                raise ValueError(
                    "recorded Docker container identity does not match"
                )
            container_id = "sha256:" + raw_container_id
            if container_id != launch_receipt.get("container_id"):
                raise ValueError(
                    "recorded Docker container identity drifted"
                )
            running = state.get("Running")
            if running is True:
                status_value = "running"
                container_returncode = None
            elif running is False and state.get("Status") == "created":
                status_value = "created"
                container_returncode = None
            elif running is False and state.get("Status") in {"exited", "dead"}:
                exit_code = state.get("ExitCode")
                if isinstance(exit_code, bool) or not isinstance(exit_code, int):
                    raise ValueError(
                        "recorded Docker exit status is invalid"
                    )
                status_value = "exited"
                container_returncode = exit_code
            else:
                raise ValueError(
                    "recorded Docker container is not adoptable"
                )
    return {
        "status": status_value,
        "inspection_runtime_id": runtime_id,
        "inspection_command_id": inspection_command_id,
        "observed_at_ms": observed_at_ms,
        "provider_id": launch_receipt.get("provider_id"),
        "command_id": launch_receipt.get("command_id"),
        "runtime_id": launch_receipt.get("runtime_id"),
        "image_id": launch_receipt.get("image_id"),
        "mount_id": launch_receipt.get("mount_id"),
        "environment_id": launch_receipt.get("environment_id"),
        "container_name": container_name,
        "container_id": container_id,
        "returncode": container_returncode,
    }


def _wait_for_recorded_codex_effect(
    launch_receipt: Mapping[str, object],
) -> int:
    """Attach to the exact adopted effect; this path never creates it."""

    docker_bin = _docker_isolation_binary()
    if (
        not docker_bin
        or _docker_runtime_receipt_identity(docker_bin)
        != launch_receipt.get("runtime_id")
    ):
        raise ValueError("recorded Docker runtime identity drifted")
    container_name = str(launch_receipt.get("container_name") or "")
    container_id = str(launch_receipt.get("container_id") or "")
    if (
        _DOCKER_CONTAINER_NAME_RE.fullmatch(container_name) is None
        or re.fullmatch(r"sha256:[0-9a-f]{64}", container_id) is None
    ):
        raise ValueError("recorded Docker container name is invalid")
    with tempfile.TemporaryDirectory(
        prefix="asref-codex-adoption-docker-config-"
    ) as config_root:
        wait_command = [
            docker_bin,
            f"--host={_DOCKER_LOCAL_HOST}",
            "--config",
            config_root,
            "container",
            "wait",
            container_id.removeprefix("sha256:"),
        ]
        returncode, stdout, _stderr = _bounded_docker_query(
            wait_command,
            timeout=7200.0,
        )
        fields = stdout.decode("ascii", errors="strict").split()
        if returncode != 0 or len(fields) != 1:
            raise ValueError("recorded Docker effect wait failed")
        try:
            effect_returncode = int(fields[0])
        except ValueError as exc:
            raise ValueError("recorded Docker effect exit is invalid") from exc
        if not -(2**31) <= effect_returncode < 2**31:
            raise ValueError("recorded Docker effect exit is invalid")
        return effect_returncode


def _start_recorded_codex_effect(
    launch_receipt: Mapping[str, object],
    *,
    prompt: str,
) -> int:
    """Start/attach exactly the inert container named in the CAS receipt."""

    docker_bin = _docker_isolation_binary()
    if (
        not docker_bin
        or _docker_runtime_receipt_identity(docker_bin)
        != launch_receipt.get("runtime_id")
    ):
        raise ValueError("recorded Docker runtime identity drifted")
    container_name = str(launch_receipt.get("container_name") or "")
    container_id = str(launch_receipt.get("container_id") or "")
    if (
        _DOCKER_CONTAINER_NAME_RE.fullmatch(container_name) is None
        or re.fullmatch(r"sha256:[0-9a-f]{64}", container_id) is None
    ):
        raise ValueError("recorded Docker container name is invalid")
    command_receipt = launch_receipt.get("command_receipt")
    command = (
        command_receipt.get("start_argv")
        if isinstance(command_receipt, Mapping)
        else None
    )
    if (
        not isinstance(command, list)
        or any(not isinstance(item, str) for item in command)
        or command
        != [
            docker_bin,
            f"--host={_DOCKER_LOCAL_HOST}",
            "--config",
            str(_recorded_codex_lease_root(launch_receipt)[1]),
            "start",
            "--attach",
            "--interactive",
            container_id.removeprefix("sha256:"),
        ]
    ):
        raise ValueError("recorded Docker start command drifted")
    process = subprocess.Popen(
        list(command),
        env=_docker_control_env(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    if process.stdin is None or process.stdout is None or process.stderr is None:
        raise ValueError("recorded Docker start pipes were not created")
    stdout_thread = threading.Thread(
        target=_stream_provider_pipe_without_reserved_records,
        args=(process.stdout, sys.stdout),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_stream_provider_pipe_without_reserved_records,
        args=(process.stderr, sys.stderr),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()
    try:
        process.stdin.write(prompt)
        process.stdin.close()
    except BrokenPipeError:
        pass
    returncode = int(process.wait())
    stdout_thread.join()
    stderr_thread.join()
    return returncode


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
    required_reasoning_effort: str | None = None,
) -> None:
    """Require an authorized daemon-owned Terra fallback shape."""

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
    # ``os.access(..., X_OK)`` is false on a noexec test mount even for a
    # correctly pinned executable.  The Docker boundary executes the pinned
    # image command, so verify immutable executable mode here instead.
    if not executable.is_file() or not (resolved_executable.stat().st_mode & 0o111):
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
    if configs.get("model_reasoning_effort") not in {'"medium"', '"high"'}:
        raise ValueError("Codex fallback reasoning is not medium or high")
    if required_reasoning_effort is not None and configs.get(
        "model_reasoning_effort"
    ) != json.dumps(required_reasoning_effort):
        raise ValueError(
            "Codex fallback reasoning does not match the sealed provider route"
        )
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
    except OSError as exc:
        raise ValueError("Codex quota fallback requires a validated auth.json") from exc
    _validated_codex_auth_path(
        source_auth=codex_home / "auth.json",
        workspace=workspace,
    )
    if (
        not codex_home.is_dir()
        or Path(os.path.abspath(candidate)).is_relative_to(workspace)
        or codex_home.is_relative_to(workspace)
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


def _validated_codex_auth_path(
    *,
    source_auth: Path,
    workspace: Path,
) -> Path:
    """Pin a private, single-link regular credential owned by this account."""

    auth_entry = Path(source_auth).expanduser()
    try:
        entry_stat = auth_entry.lstat()
        resolved_auth = auth_entry.resolve(strict=True)
        resolved_workspace = workspace.resolve(strict=True)
    except OSError as exc:
        raise ValueError("Codex quota fallback requires a validated auth.json") from exc
    if (
        not auth_entry.is_absolute()
        or auth_entry != resolved_auth
        or not stat.S_ISREG(entry_stat.st_mode)
        or entry_stat.st_uid != os.getuid()
        or stat.S_IMODE(entry_stat.st_mode) != 0o600
        or entry_stat.st_nlink != 1
        or resolved_auth.name != "auth.json"
        or resolved_auth.is_relative_to(resolved_workspace)
    ):
        raise ValueError(
            "Codex quota fallback auth must be a private, owned, regular auth.json"
        )
    return resolved_auth


def _isolated_codex_quota_fallback_home(
    *,
    workspace: Path,
    base_env: dict[str, str],
) -> tuple[
    tempfile.TemporaryDirectory[str],
    dict[str, str],
    Path,
]:
    """Create an ephemeral Codex home containing only pinned auth authority."""

    host_environment = _codex_quota_fallback_env(
        workspace=workspace,
        base_env=base_env,
    )
    source_auth = (
        Path(host_environment["CODEX_HOME"]) / "auth.json"
    ).resolve(strict=True)
    temporary_home = tempfile.TemporaryDirectory(
        prefix="asref-codex-home-"
    )
    try:
        temporary_home_path = Path(temporary_home.name)
        temporary_home_path.chmod(0o700)
        isolated_environment = _codex_task_container_environment()
        return temporary_home, isolated_environment, source_auth
    except Exception:
        temporary_home.cleanup()
        raise


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


def _stream_provider_pipe_without_reserved_records(
    source: TextIO,
    destination: TextIO,
) -> None:
    """Tee provider output while escaping runner-reserved record prefixes."""

    reserved = (
        GROK_FAILURE_RECEIPT_PREFIX,
        GROK_ROUTE_OUTCOME_PREFIX,
        AGENT_IMPLEMENTATION_ROUTE_OUTCOME_PREFIX,
    )
    maximum_prefix = max(map(len, reserved))
    prefix_buffer = ""
    at_line_start = True

    def sanitized(value: str) -> str:
        # The authority parser is LF-framed.  Remove every other character
        # Python's splitlines() could reinterpret as a record boundary.
        replacements = {
            "\0": "[provider-child-control-00]",
            "\r": "[provider-child-control-0d]",
            "\v": "[provider-child-control-0b]",
            "\f": "[provider-child-control-0c]",
            "\x1c": "[provider-child-control-1c]",
            "\x1d": "[provider-child-control-1d]",
            "\x1e": "[provider-child-control-1e]",
            "\x85": "[provider-child-control-85]",
            "\u2028": "[provider-child-control-2028]",
            "\u2029": "[provider-child-control-2029]",
        }
        return "".join(replacements.get(character, character) for character in value)

    def flush_prefix(*, line_complete: bool) -> None:
        nonlocal prefix_buffer, at_line_start
        if not prefix_buffer and not line_complete:
            return
        still_possible = any(
            item.startswith(prefix_buffer) for item in reserved
        )
        if (
            not line_complete
            and len(prefix_buffer) < maximum_prefix
            and still_possible
        ):
            return
        output = prefix_buffer
        if output.startswith(reserved):
            output = "[provider-child-output-escaped] " + output
        destination.write(output)
        prefix_buffer = ""
        at_line_start = False

    while True:
        chunk = source.read(16 * 1024)
        if not chunk:
            if prefix_buffer:
                flush_prefix(line_complete=True)
            destination.flush()
            return
        parts = sanitized(chunk).split("\n")
        for index, piece in enumerate(parts):
            line_complete = index < len(parts) - 1
            if at_line_start:
                prefix_buffer += piece
                flush_prefix(line_complete=line_complete)
            else:
                destination.write(piece)
            if line_complete:
                if prefix_buffer:
                    flush_prefix(line_complete=True)
                destination.write("\n")
                at_line_start = True
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


def _validate_quota_evidence_in_accepted_child(
    *,
    grok_home: Path,
    expected_session_id: str,
    verifier_returncode: int,
    failure_receipt: Mapping[str, object],
    invocation_binding: object,
    verifier_command: list[str],
    verifier_workspace: Path,
    verifier_prompt_path: Path,
    observed_at_ms: int,
) -> object:
    """Sign native evidence in a separate fork of the accepted runner.

    Forking preserves the already-validated sealed generation without a new
    path-based Python import.  The parent accepts only the bounded signed JSON
    emitted by the exact child PID and independently re-verifies it.
    """

    if not hasattr(os, "fork"):
        return ""
    read_fd, write_fd = os.pipe()
    child_pid = os.fork()
    if child_pid == 0:
        try:
            os.close(read_fd)
            from ipfs_accelerate_py.llm_router import (
                validate_agent_implementation_quota_evidence,
            )

            evidence = validate_agent_implementation_quota_evidence(
                grok_home=grok_home,
                expected_session_id=expected_session_id,
                verifier_returncode=verifier_returncode,
                failure_receipt=failure_receipt,
                invocation_binding=invocation_binding,
                verifier_command=verifier_command,
                verifier_workspace=verifier_workspace,
                verifier_prompt_path=verifier_prompt_path,
                observed_at_ms=observed_at_ms,
                max_age_ms=_SCOPED_ROUTE_MAX_AGE_MS,
            )
            if evidence is None:
                os._exit(2)
            raw = json.dumps(
                evidence.audit_dict(),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
            if not raw or len(raw) > 256 * 1024:
                os._exit(3)
            offset = 0
            while offset < len(raw):
                written = os.write(write_fd, raw[offset:])
                if written <= 0:
                    os._exit(4)
                offset += written
            os.close(write_fd)
            os._exit(0)
        except BaseException:
            os._exit(5)
    os.close(write_fd)
    try:
        chunks: list[bytes] = []
        remaining = 256 * 1024 + 1
        while remaining:
            chunk = os.read(read_fd, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
    finally:
        os.close(read_fd)
    _waited_pid, status = os.waitpid(child_pid, 0)
    raw = b"".join(chunks)
    if (
        not os.WIFEXITED(status)
        or os.WEXITSTATUS(status) != 0
        or not raw
        or len(raw) > 256 * 1024
    ):
        return ""

    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        decoded: dict[str, object] = {}
        for key, value in pairs:
            if key in decoded:
                raise ValueError("duplicate quota evidence field")
            decoded[key] = value
        return decoded

    try:
        payload = json.loads(raw, object_pairs_hook=unique)
    except (UnicodeError, ValueError, json.JSONDecodeError):
        return ""
    if (
        not isinstance(payload, Mapping)
        or payload.get("signer_process_pid") != child_pid
        or payload.get("signer_parent_pid") != os.getpid()
    ):
        return ""
    from ipfs_accelerate_py.llm_router import (
        parse_agent_implementation_quota_evidence,
    )

    return parse_agent_implementation_quota_evidence(
        payload,
        failure_receipt=failure_receipt,
        invocation_binding=invocation_binding,
        now_ms=observed_at_ms,
        max_age_ms=_SCOPED_ROUTE_MAX_AGE_MS,
        expected_signer_parent_pid=os.getpid(),
        expected_signer_process_pid=child_pid,
    ) or ""


def _independently_verify_grok_quota(
    *,
    grok_bin: str,
    base_env: dict[str, str],
    failure_receipt: Mapping[str, object],
    invocation_binding: object | None = None,
) -> object:
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
                "PWD": str(verifier_workspace),
            }
        )
        verifier_env.pop("OLDPWD", None)
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
                AGENT_IMPLEMENTATION_QUOTA_VERIFIER_DISALLOWED_TOOLS,
            ]
        )
        output_index = command.index("--output-format") + 1
        command[output_index] = "streaming-json"
        try:
            completed = subprocess.run(
                command,
                cwd=verifier_workspace,
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
        if invocation_binding is None:
            from ipfs_accelerate_py.llm_router import (
                validate_agent_implementation_quota_evidence,
            )

            return validate_agent_implementation_quota_evidence(
                grok_home=verifier_home,
                expected_session_id=verifier_session_id,
                verifier_returncode=int(completed.returncode),
                failure_receipt=failure_receipt,
            )
        return _validate_quota_evidence_in_accepted_child(
            grok_home=verifier_home,
            expected_session_id=verifier_session_id,
            verifier_returncode=int(completed.returncode),
            failure_receipt=failure_receipt,
            invocation_binding=invocation_binding,
            verifier_command=command,
            verifier_workspace=verifier_workspace,
            verifier_prompt_path=prompt_path,
            observed_at_ms=int(time.time() * 1000),
        )
    finally:
        if isolated_home is not None:
            _robust_remove_runner_temp_tree(Path(isolated_home.name))
            isolated_home.cleanup()
        _robust_remove_runner_temp_tree(verifier_root)


def _run_typed_grok_preflight_once(
    *,
    grok_bin: str,
    base_env: dict[str, str],
    nonce: str,
) -> tuple[int, dict[str, object], bool, str]:
    """Run the fixed no-tools probe and return its runner-authored receipt.

    The probe has no task prompt or task workspace and runs before the primary
    implementation dispatch.  Its bounded stderr is classified locally, then
    bound to the daemon-provided nonce.  Caller code must still validate the
    returned receipt before granting any fallback effect.
    """

    if re.fullmatch(r"[0-9a-f]{64}", str(nonce or "")) is None:
        raise ValueError("typed Grok preflight requires a 256-bit nonce")

    from ipfs_accelerate_py.llm_router import build_grok_cli_command, build_grok_cli_env

    probe_root = Path(tempfile.mkdtemp(prefix="asref-grok-failure-probe-"))
    isolated_home: tempfile.TemporaryDirectory[str] | None = None
    try:
        probe_workspace = probe_root / "workspace"
        probe_workspace.mkdir(mode=0o700)
        prompt_path = probe_root / "prompt.txt"
        prompt_path.write_text(GROK_QUOTA_PROBE_PROMPT, encoding="utf-8")
        child_env = build_grok_cli_env(
            base_env=base_env,
            isolate_alternate_providers=True,
        )
        isolated_home, probe_env, _policy, _denied = _isolated_grok_home(
            base_env=base_env,
            child_env=child_env,
            codex_fallback_command=(),
            workspace=probe_workspace,
        )
        probe_home = Path(probe_env["GROK_HOME"])
        probe_env.update(
            {
                "HOME": str(probe_home),
                "XDG_CONFIG_HOME": str(probe_home / "xdg-config"),
                "XDG_DATA_HOME": str(probe_home / "xdg-data"),
                "XDG_STATE_HOME": str(probe_home / "xdg-state"),
                "PWD": str(probe_workspace),
            }
        )
        probe_env.pop("OLDPWD", None)
        command = build_grok_cli_command(
            mode="chat",
            workspace=probe_workspace,
            model_name=DEFAULT_GROK_MODEL,
            max_turns=1,
            grok_bin=grok_bin,
            prompt_file=prompt_path,
            permission_mode="dontAsk",
            tools="",
        )
        command.extend(
            ["--disallowed-tools", _SEALED_GROK_DISALLOWED_TOOLS]
        )
        returncode, stderr_text, stderr_size, stderr_overflow = (
            _run_isolated_grok_quota_probe(
            command,
            env=probe_env,
            cwd=probe_workspace,
        )
        )
        if returncode == 0:
            return 0, {}, stderr_overflow, ""
        receipt_evidence = (
            "isolated Grok quota probe stderr exceeded the trusted evidence "
            f"limit ({stderr_size} bytes)"
            if stderr_overflow
            else stderr_text
        )
        receipt = build_grok_failure_receipt(
            probe_stderr_text=receipt_evidence,
            nonce=nonce,
            model=DEFAULT_GROK_MODEL,
            probe_returncode=returncode,
            primary_dispatched=False,
            evidence_size=stderr_size,
            evidence_overflow=stderr_overflow,
        )
        if not valid_grok_failure_receipt(
            receipt,
            nonce=nonce,
            model=DEFAULT_GROK_MODEL,
            returncode=returncode,
        ):
            return returncode, {}, stderr_overflow, receipt_evidence
        return returncode, receipt, stderr_overflow, receipt_evidence
    finally:
        if isolated_home is not None:
            _robust_remove_runner_temp_tree(Path(isolated_home.name))
            isolated_home.cleanup()
        _robust_remove_runner_temp_tree(probe_root)


def _run_typed_grok_preflight(
    *,
    grok_bin: str,
    base_env: dict[str, str],
    nonce: str,
) -> tuple[int, dict[str, object], bool]:
    """Run the typed probe, retrying only its exact transient turn artifact."""

    from ipfs_accelerate_py.llm_router import (
        retryable_agent_implementation_preflight_failure,
    )

    returncode, receipt, overflow, evidence = _run_typed_grok_preflight_once(
        grok_bin=grok_bin,
        base_env=base_env,
        nonce=nonce,
    )
    if returncode == 0 or not receipt:
        return returncode, receipt, overflow
    if not retryable_agent_implementation_preflight_failure(
        evidence,
        receipt,
        nonce=nonce,
        model=DEFAULT_GROK_MODEL,
        probe_returncode=returncode,
    ):
        return returncode, receipt, overflow
    retry_returncode, retry_receipt, retry_overflow, _retry_evidence = (
        _run_typed_grok_preflight_once(
            grok_bin=grok_bin,
            base_env=base_env,
            nonce=nonce,
        )
    )
    return retry_returncode, retry_receipt, retry_overflow


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


_PROTECTED_EFFECT_RECOVERY_LOCATOR_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor."
    "provider-effect-recovery-locator@1"
)


def _parse_protected_effect_recovery_locator(
    raw: str,
    *,
    workspace: Path,
) -> dict[str, object]:
    """Decode the daemon's narrow non-dispatch CAS recovery locator."""

    if not isinstance(raw, str) or not raw or len(raw.encode("utf-8")) > 16 * 1024:
        raise ValueError("protected effect recovery locator is invalid")

    def unique(pairs):
        decoded = {}
        for key, value in pairs:
            if key in decoded:
                raise ValueError("protected effect recovery locator has duplicate keys")
            decoded[key] = value
        return decoded

    try:
        value = json.loads(raw, object_pairs_hook=unique)
    except json.JSONDecodeError as exc:
        raise ValueError("protected effect recovery locator is invalid JSON") from exc
    expected = {
        "schema",
        "task_id",
        "attempt",
        "task_revision_cid",
        "board_namespace",
        "logical_attempt_id",
        "worktree_id",
        "prompt_cid",
        "workspace_path",
        "provider_attempt_store",
        "provider_attempt_store_identity",
        "locator_id",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected
        or value.get("schema") != _PROTECTED_EFFECT_RECOVERY_LOCATOR_SCHEMA
        or any(
            not isinstance(value.get(name), str) or not value.get(name)
            for name in expected - {"attempt", "locator_id"}
        )
        or isinstance(value.get("attempt"), bool)
        or not isinstance(value.get("attempt"), int)
        or int(value.get("attempt") or 0) < 1
        or value.get("workspace_path") != str(workspace)
        or value.get("locator_id")
        != _effect_receipt_identity(
            {key: item for key, item in value.items() if key != "locator_id"}
        )
    ):
        raise ValueError("protected effect recovery locator fields are invalid")
    return value


def _run_protected_effect_recovery(
    *,
    raw_locator: str,
    workspace: Path,
) -> int:
    """Account one existing protected effect without dispatching a provider."""

    from ipfs_accelerate_py.agent_supervisor.control.provider_attempt_store import (
        DurableProviderAttemptCAS,
        ProviderAttemptStoreError,
    )
    from ipfs_accelerate_py.llm_router import (
        build_agent_implementation_route_outcome,
        parse_agent_implementation_effect_authorization_context,
        render_agent_implementation_route_outcome,
        valid_agent_implementation_route_outcome,
        verify_agent_implementation_sealed_control_plane,
    )

    try:
        locator = _parse_protected_effect_recovery_locator(
            raw_locator,
            workspace=workspace,
        )
        prompt = sys.stdin.read()
        if (
            not prompt.strip()
            or _agent_prompt_cid(prompt) != locator.get("prompt_cid")
        ):
            raise ValueError("protected effect recovery prompt identity drifted")
        store = DurableProviderAttemptCAS(
            str(locator["provider_attempt_store"]),
            expected_directory_identity=str(
                locator["provider_attempt_store_identity"]
            ),
        )
        reservation = store.read(str(locator["logical_attempt_id"]))
        if reservation is None or reservation.state not in {
            "effect_started",
            "quarantined",
            "terminal",
        }:
            raise ValueError("protected effect recovery CAS is unavailable")
        launch_owner_pid = reservation.effect_launch_receipt.get(
            "effect_owner_pid"
        )
        if (
            isinstance(launch_owner_pid, bool)
            or not isinstance(launch_owner_pid, int)
            or launch_owner_pid <= 0
            or reservation.effect_started_at_ms is None
        ):
            raise ValueError("protected effect recovery launch authority is invalid")
        context = parse_agent_implementation_effect_authorization_context(
            reservation.authorization_context,
            repo_root=workspace,
            effect_started_at_ms=reservation.effect_started_at_ms,
            expected_signer_parent_pid=launch_owner_pid,
            max_age_ms=_SCOPED_ROUTE_MAX_AGE_MS,
        )
        if context is None or context.route.invocation_binding is None:
            raise ValueError("protected effect historical authority is invalid")
        invocation = context.route.invocation_binding
        exact_locator = {
            "task_id": invocation.task_id,
            "attempt": invocation.attempt,
            "task_revision_cid": invocation.task_revision_cid,
            "logical_attempt_id": invocation.logical_attempt_id,
            "worktree_id": invocation.worktree_id,
            "prompt_cid": invocation.prompt_cid,
            "workspace_path": invocation.workspace_path,
            "provider_attempt_store": invocation.provider_attempt_store,
            "provider_attempt_store_identity": (
                invocation.provider_attempt_store_identity
            ),
        }
        if (
            any(locator.get(name) != item for name, item in exact_locator.items())
            or reservation.task_id != invocation.task_id
            or reservation.worktree_id != invocation.worktree_id
            or reservation.route_id != invocation.route_id
            or reservation.decision_id != context.decision.content_id
        ):
            raise ValueError("protected effect recovery identity drifted")
        sealed_match = re.fullmatch(r"/proc/self/fd/([0-9]+)", str(sys.argv[0]))
        if sealed_match is None or verify_agent_implementation_sealed_control_plane(
            invocation.control_plane,
            int(sealed_match.group(1)),
        ) != str(sys.argv[0]):
            raise ValueError("protected effect recovery is not sealed")

        if reservation.terminal:
            outcome = reservation.terminal_outcome
            returncode = reservation.terminal_returncode
            if (
                not isinstance(outcome, Mapping)
                or isinstance(returncode, bool)
                or not isinstance(returncode, int)
                or outcome.get("decision_id") != reservation.decision_id
                or outcome.get("reservation_id") != reservation.reservation_id
                or outcome.get("effect_launch_receipt")
                != reservation.effect_launch_receipt
                or outcome.get("effect_adoption_receipt")
                != reservation.effect_adoption_receipt
                or outcome.get("effect_quarantine_receipt")
                != reservation.quarantine_receipt
                or outcome.get(
                    "effect_quarantine_terminalization_receipt"
                )
                != reservation.quarantine_terminalization_receipt
                or outcome.get("fallback_returncode") != returncode
                or not valid_agent_implementation_route_outcome(
                    outcome,
                    receipt=context.failure_receipt,
                    route=context.route,
                    runner_returncode=returncode,
                )
            ):
                raise ValueError("protected terminal recovery outcome is invalid")
            _release_recorded_codex_effect_cleanup(
                reservation.effect_launch_receipt
            )
            print(render_agent_implementation_route_outcome(outcome), file=sys.stderr)
            return returncode

        quarantined_repair = reservation.state == "quarantined"
        adopted = (
            store.claim_quarantined_terminalization(reservation)
            if quarantined_repair
            else store.adopt_effect(reservation)
        )
        if not adopted.adoption_authorized:
            raise ProviderAttemptStoreError(
                (
                    "quarantined effect remains created/running; exact "
                    "operator reinspection is required"
                    if quarantined_repair
                    else "protected effect recovery owner transfer was denied"
                )
            )
        active = adopted.reservation
        inspection_receipt = (
            active.quarantine_terminalization_receipt
            if quarantined_repair
            else active.effect_adoption_receipt
        )
        status_value = inspection_receipt.get("inspection_status")
        if status_value == "absent":
            returncode = 125
            outcome_decision = "effect_not_created"
            dispatched = False
        elif status_value == "created":
            if quarantined_repair:
                raise ProviderAttemptStoreError(
                    "quarantined created effect cannot be started"
                )
            returncode = _start_recorded_codex_effect(
                active.effect_launch_receipt,
                prompt=prompt,
            )
            outcome_decision = (
                "fallback_succeeded" if returncode == 0 else "fallback_failed"
            )
            dispatched = True
        elif status_value == "exited":
            returncode = inspection_receipt.get("container_returncode")
            if isinstance(returncode, bool) or not isinstance(returncode, int):
                raise ValueError("protected effect recovery exit is invalid")
            outcome_decision = (
                "fallback_succeeded" if returncode == 0 else "fallback_failed"
            )
            dispatched = True
        elif status_value == "running":
            if quarantined_repair:
                raise ProviderAttemptStoreError(
                    "quarantined running effect requires later reinspection"
                )
            returncode = _wait_for_recorded_codex_effect(
                active.effect_launch_receipt
            )
            outcome_decision = (
                "fallback_succeeded" if returncode == 0 else "fallback_failed"
            )
            dispatched = True
        else:
            raise ValueError("protected effect recovery inspection is invalid")
        outcome = build_agent_implementation_route_outcome(
            receipt=context.failure_receipt,
            route=context.route,
            decision=outcome_decision,
            verifier_status=context.decision.verifier_status,
            fallback_dispatched=dispatched,
            fallback_returncode=returncode,
            decision_id=context.decision.content_id,
            quota_evidence=context.quota_evidence,
            reservation_id=active.reservation_id,
            effect_launch_receipt=active.effect_launch_receipt,
            effect_adoption_receipt=active.effect_adoption_receipt,
            effect_quarantine_receipt=(
                active.quarantine_receipt if quarantined_repair else None
            ),
            effect_quarantine_terminalization_receipt=(
                active.quarantine_terminalization_receipt
                if quarantined_repair
                else None
            ),
        )
        terminal = store.complete(
            active,
            returncode=returncode,
            outcome=outcome,
            completion_capability=adopted.completion_capability,
        )
        _release_recorded_codex_effect_cleanup(terminal.effect_launch_receipt)
        print(render_agent_implementation_route_outcome(outcome), file=sys.stderr)
        return returncode
    except (OSError, TypeError, ValueError, ProviderAttemptStoreError) as exc:
        print(f"protected effect recovery denied: {exc}", file=sys.stderr)
        return 125


def _run(args: argparse.Namespace, receipt_fd: int) -> int:
    from ipfs_accelerate_py.agent_supervisor.control.provider_attempt_store import (
        DurableProviderAttemptCAS,
        ProviderAttemptReservation,
        ProviderAttemptStoreError,
    )
    from ipfs_accelerate_py.llm_router import (
        LLMRouterError,
        build_agent_implementation_effect_authorization_context,
        build_agent_implementation_route_outcome,
        build_grok_cli_command,
        build_grok_cli_env,
        create_legacy_agent_implementation_route_invocation,
        decide_agent_implementation_fallback,
        find_grok_cli,
        parse_agent_implementation_effect_authorization_context,
        render_agent_implementation_route_outcome,
        resolve_agent_implementation_route,
        resolve_agent_implementation_route_binding,
        valid_agent_implementation_route_outcome,
        verify_agent_implementation_sealed_control_plane,
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
    internal_legacy_preflight = bool(
        args.canonical_legacy_preflight_route
    )
    if internal_legacy_preflight and not codex_fallback_command:
        print(
            "canonical legacy preflight requires a Codex fallback command",
            file=sys.stderr,
        )
        return 2

    workspace = args.workspace.expanduser().resolve()
    if not workspace.is_dir():
        print(f"workspace is not a directory: {workspace}", file=sys.stderr)
        return 2
    recovery_locator_raw = str(
        args.agent_implementation_recovery_json or ""
    ).strip()
    if recovery_locator_raw:
        if (
            codex_fallback_command
            or str(args.grok_failure_receipt_nonce or "").strip()
            or str(args.agent_implementation_route_json or "").strip()
            or bool(args.canonical_legacy_preflight_route)
            or str(args.grok_bin or "").strip()
            or str(args.model or "").strip()
            or args.require_command
        ):
            print(
                "protected effect recovery forbids provider dispatch options",
                file=sys.stderr,
            )
            return 2
        return _run_protected_effect_recovery(
            raw_locator=recovery_locator_raw,
            workspace=workspace,
        )
    protected_recovery_reservation: ProviderAttemptReservation | None = None
    protected_recovery_context = None
    if codex_fallback_command:
        route_repository_head = ""
        preflight_nonce = str(args.grok_failure_receipt_nonce or "").strip()
        route_binding_raw = str(
            args.agent_implementation_route_json or ""
        ).strip()
        route_plan = None
        if preflight_nonce:
            if internal_legacy_preflight:
                print(
                    "canonical legacy preflight cannot be combined with an "
                    "external nonce or route binding",
                    file=sys.stderr,
                )
                return 2
            if not route_binding_raw:
                print(
                    "typed Grok preflight requires a scoped canonical route "
                    "binding",
                    file=sys.stderr,
                )
                return 2
            if len(route_binding_raw.encode("utf-8")) > 16 * 1024:
                print("agent implementation route binding is oversized", file=sys.stderr)
                return 2

            def reject_route_duplicate_keys(pairs):
                result = {}
                for key, value in pairs:
                    if key in result:
                        raise ValueError(
                            "agent implementation route binding has duplicate keys"
                        )
                    result[key] = value
                return result

            try:
                route_binding = json.loads(
                    route_binding_raw,
                    object_pairs_hook=reject_route_duplicate_keys,
                )
                if not isinstance(route_binding, dict):
                    raise ValueError(
                        "agent implementation route binding must be an object"
                    )
                route_plan = resolve_agent_implementation_route_binding(
                    route_binding,
                    repo_root=workspace,
                    now_ms=int(time.time() * 1000),
                    max_age_ms=_SCOPED_ROUTE_MAX_AGE_MS,
                )
                invocation = route_plan.invocation_binding
                sealed_match = re.fullmatch(
                    r"/proc/self/fd/([0-9]+)",
                    str(sys.argv[0]),
                )
                if invocation is not None:
                    if sealed_match is None:
                        raise ValueError(
                            "protected route requires the sealed accepted-generation archive"
                        )
                    sealed_descriptor = int(sealed_match.group(1))
                    if verify_agent_implementation_sealed_control_plane(
                        invocation.control_plane,
                        sealed_descriptor,
                    ) != str(sys.argv[0]):
                        raise ValueError(
                            "protected route sealed archive identity drifted"
                        )
                    recovery_store = DurableProviderAttemptCAS(
                        invocation.provider_attempt_store,
                        expected_directory_identity=(
                            invocation.provider_attempt_store_identity
                        ),
                    )
                    existing = recovery_store.read(
                        invocation.logical_attempt_id
                    )
                    if existing is not None and existing.state in {
                        "effect_started",
                        "terminal",
                    }:
                        launch_owner_pid = existing.effect_launch_receipt.get(
                            "effect_owner_pid"
                        )
                        if (
                            isinstance(launch_owner_pid, bool)
                            or not isinstance(launch_owner_pid, int)
                            or launch_owner_pid <= 0
                            or existing.effect_started_at_ms is None
                        ):
                            raise ValueError(
                                "protected recovery effect authority is invalid"
                            )
                        protected_recovery_context = (
                            parse_agent_implementation_effect_authorization_context(
                                existing.authorization_context,
                                repo_root=workspace,
                                effect_started_at_ms=(
                                    existing.effect_started_at_ms
                                ),
                                expected_signer_parent_pid=launch_owner_pid,
                                max_age_ms=_SCOPED_ROUTE_MAX_AGE_MS,
                            )
                        )
                        if protected_recovery_context is None:
                            raise ValueError(
                                "protected recovery authority could not be verified"
                            )
                        if (
                            protected_recovery_context.route.invocation_binding
                            is None
                            or protected_recovery_context.route.invocation_binding.logical_attempt_id
                            != invocation.logical_attempt_id
                            or protected_recovery_context.decision.content_id
                            != existing.decision_id
                        ):
                            raise ValueError(
                                "protected recovery authority changed logical attempt"
                            )
                        historical_route = protected_recovery_context.route
                        if existing.terminal:
                            terminal_outcome = existing.terminal_outcome
                            terminal_returncode = existing.terminal_returncode
                            if (
                                not isinstance(terminal_outcome, Mapping)
                                or isinstance(terminal_returncode, bool)
                                or not isinstance(terminal_returncode, int)
                                or terminal_outcome.get("decision_id")
                                != existing.decision_id
                                or terminal_outcome.get("reservation_id")
                                != existing.reservation_id
                                or terminal_outcome.get("effect_launch_receipt")
                                != existing.effect_launch_receipt
                                or terminal_outcome.get("effect_adoption_receipt")
                                != existing.effect_adoption_receipt
                                or terminal_outcome.get(
                                    "effect_quarantine_receipt"
                                )
                                != existing.quarantine_receipt
                                or terminal_outcome.get(
                                    "effect_quarantine_terminalization_receipt"
                                )
                                != existing.quarantine_terminalization_receipt
                                or terminal_outcome.get("fallback_returncode")
                                != terminal_returncode
                                or not valid_agent_implementation_route_outcome(
                                    terminal_outcome,
                                    receipt=(
                                        protected_recovery_context.failure_receipt
                                    ),
                                    route=historical_route,
                                    runner_returncode=terminal_returncode,
                                )
                            ):
                                raise ValueError(
                                    "protected terminal recovery outcome is invalid"
                                )
                            try:
                                _release_recorded_codex_effect_cleanup(
                                    existing.effect_launch_receipt
                                )
                            except FileNotFoundError:
                                pass
                            print(
                                render_agent_implementation_route_outcome(
                                    terminal_outcome
                                ),
                                file=sys.stderr,
                            )
                            return terminal_returncode
                        route_plan = historical_route
                        protected_recovery_reservation = existing
                    elif existing is not None:
                        # A pre-effect reservation never authorizes a provider
                        # restart.  Keep the logical attempt latched until its
                        # exact original authority can be resumed or abandoned
                        # by a dedicated reserved-only transition.
                        raise ValueError(
                            "protected recovery reservation is incomplete"
                        )
                route_repository_head = _repository_head(workspace)
            except (json.JSONDecodeError, OSError, ValueError) as exc:
                print(str(exc), file=sys.stderr)
                return 2
        else:
            if route_binding_raw:
                print(
                    "legacy quota route forbids an auth/high route binding",
                    file=sys.stderr,
                )
                return 2
            if internal_legacy_preflight:
                legacy_invocation = (
                    create_legacy_agent_implementation_route_invocation()
                )
                route_plan = legacy_invocation.route_plan
                preflight_nonce = (
                    legacy_invocation.failure_receipt_nonce
                )
                route_repository_head = _repository_head(workspace)
            else:
                route_plan = resolve_agent_implementation_route(
                    default_route="legacy"
                )

        def route_outcome_record(
            *,
            active_route,
            receipt: Mapping[str, object],
            quota_evidence_id: str,
            decision: str,
            verifier_status: str,
            fallback_dispatched: bool,
            fallback_returncode: int | None,
            decision_id: str = "",
            reservation: ProviderAttemptReservation | None = None,
        ) -> dict[str, object]:
            if active_route.invocation_binding is not None:
                return build_agent_implementation_route_outcome(
                    receipt=receipt,
                    route=active_route,
                    decision=decision,
                    verifier_status=verifier_status,
                    fallback_dispatched=fallback_dispatched,
                    fallback_returncode=fallback_returncode,
                    decision_id=(
                        reservation.decision_id
                        if reservation is not None
                        else decision_id or preflight_decision_id
                    ),
                    quota_evidence=(
                        preflight_quota_evidence
                        if verifier_status == "confirmed_quota"
                        else None
                    ),
                    reservation_id=(
                        reservation.reservation_id if reservation else ""
                    ),
                    effect_launch_receipt=(
                        reservation.effect_launch_receipt if reservation else {}
                    ),
                    effect_adoption_receipt=(
                        getattr(reservation, "effect_adoption_receipt", {})
                        if reservation
                        else {}
                    ),
                    effect_quarantine_receipt=(
                        getattr(reservation, "quarantine_receipt", {})
                        if reservation
                        else {}
                    ),
                    effect_quarantine_terminalization_receipt=(
                        getattr(
                            reservation,
                            "quarantine_terminalization_receipt",
                            {},
                        )
                        if reservation
                        else {}
                    ),
                )
            return build_grok_route_outcome(
                receipt=receipt,
                route_plan=active_route.as_outcome_dict(),
                quota_evidence_id=quota_evidence_id,
                decision=decision,
                verifier_status=verifier_status,
                fallback_dispatched=fallback_dispatched,
                fallback_returncode=fallback_returncode,
            )

        def render_route_outcome_record(
            outcome: Mapping[str, object],
        ) -> str:
            if outcome.get("schema") == (
                "ipfs_accelerate_py.agent_supervisor."
                "protected-route-outcome@1"
            ):
                return render_agent_implementation_route_outcome(outcome)
            return render_grok_route_outcome(outcome)
        try:
            _validate_codex_quota_fallback_command(
                codex_fallback_command,
                workspace=workspace,
                required_reasoning_effort=(
                    route_plan.fallback_reasoning_effort
                ),
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

    grok_bin = (
        ""
        if protected_recovery_reservation is not None
        else str(args.grok_bin).strip() or find_grok_cli() or ""
    )
    if not grok_bin and protected_recovery_reservation is None:
        print("grok CLI not found on PATH", file=sys.stderr)
        return 127
    if codex_fallback_command and protected_recovery_reservation is None:
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
    if protected_recovery_context is not None:
        model = str(
            protected_recovery_reservation.authorization_context.get(
                "expected_model"
            )
            or ""
        )
    if codex_fallback_command and model != DEFAULT_GROK_MODEL:
        print(
            "Default Grok/Codex route requires primary model grok-4.6",
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

    prompt: str | None = None
    if (
        codex_fallback_command
        and route_plan.invocation_binding is not None
    ):
        # The scoped route signs the task prompt. Read and verify it before
        # even the supposedly tool-free primary preflight so no provider call
        # can be made under a prompt authority that the runner did not receive.
        prompt = sys.stdin.read()
        if _agent_prompt_cid(prompt) != route_plan.invocation_binding.prompt_cid:
            print(
                "Signed invocation does not match the task prompt; provider "
                "dispatch is forbidden",
                file=sys.stderr,
            )
            return 2

    workspace_baseline = ""
    preflight_fallback_reason = ""
    preflight_returncode = 0
    preflight_receipt: dict[str, object] = {}
    preflight_verifier_status = "not_run"
    preflight_quota_evidence: object | None = None
    preflight_decision_id = ""
    if codex_fallback_command:
        if protected_recovery_context is not None:
            preflight_receipt = dict(
                protected_recovery_context.failure_receipt
            )
            preflight_quota_evidence = (
                protected_recovery_context.quota_evidence
            )
            preflight_verifier_status = (
                protected_recovery_context.decision.verifier_status
            )
            preflight_decision_id = (
                protected_recovery_context.decision.content_id
            )
            preflight_returncode = int(
                protected_recovery_reservation.authorization_context.get(
                    "expected_probe_returncode"
                )
            )
            preflight_nonce = str(
                protected_recovery_reservation.authorization_context.get(
                    "expected_nonce"
                )
            )
            preflight_fallback_reason = "recovering a claimed provider effect"
        if protected_recovery_context is None:
            try:
                workspace_baseline = _workspace_content_fingerprint(workspace)
            except ValueError as exc:
                print(str(exc), file=sys.stderr)
                return 2
        if preflight_nonce and protected_recovery_context is None:
            try:
                (
                    preflight_returncode,
                    preflight_receipt,
                    _preflight_overflow,
                ) = (
                    _run_typed_grok_preflight(
                        grok_bin=grok_bin,
                        base_env=os.environ.copy(),
                        nonce=preflight_nonce,
                    )
                )
            except (OSError, RuntimeError, ValueError) as exc:
                print(
                    f"unable to run typed Grok preflight: {exc}",
                    file=sys.stderr,
                )
                return 2
            if preflight_receipt:
                print(
                    render_grok_failure_receipt(preflight_receipt),
                    file=sys.stderr,
                )
            if preflight_returncode != 0:
                decision = decide_agent_implementation_fallback(
                    route_plan,
                    repo_root=workspace,
                    failure_receipt=preflight_receipt,
                    expected_nonce=preflight_nonce,
                    expected_model=model,
                    expected_probe_returncode=preflight_returncode,
                    expected_invocation_binding=(
                        route_plan.invocation_binding.signed_payload()
                        if route_plan.invocation_binding is not None
                        else None
                    ),
                    now_ms=int(time.time() * 1000),
                    max_age_ms=_SCOPED_ROUTE_MAX_AGE_MS,
                )
                preflight_decision_id = decision.content_id
                if decision.requires_independent_quota_verification:
                    preflight_quota_evidence = (
                        _independently_verify_grok_quota(
                            grok_bin=grok_bin,
                            base_env=os.environ.copy(),
                            failure_receipt=preflight_receipt,
                            invocation_binding=(
                                route_plan.invocation_binding
                            ),
                        )
                    )
                    decision = decide_agent_implementation_fallback(
                        route_plan,
                        repo_root=workspace,
                        failure_receipt=preflight_receipt,
                        expected_nonce=preflight_nonce,
                        expected_model=model,
                        expected_probe_returncode=preflight_returncode,
                        independent_quota_evidence=(
                            preflight_quota_evidence
                        ),
                        expected_invocation_binding=(
                            route_plan.invocation_binding.signed_payload()
                            if route_plan.invocation_binding is not None
                            else None
                        ),
                        now_ms=int(time.time() * 1000),
                        max_age_ms=_SCOPED_ROUTE_MAX_AGE_MS,
                    )
                    preflight_decision_id = decision.content_id
                preflight_verifier_status = decision.verifier_status
                if not decision.authorized:
                    print(
                        "Typed Grok preflight did not authorize fallback; "
                        "Codex fallback is forbidden",
                        file=sys.stderr,
                    )
                    if preflight_receipt:
                        print(
                            render_route_outcome_record(
                                route_outcome_record(
                                    active_route=route_plan,
                                    receipt=preflight_receipt,
                                    quota_evidence_id=str(
                                        getattr(
                                            preflight_quota_evidence,
                                            "evidence_id",
                                            "",
                                        )
                                    ),
                                    decision="denied",
                                    verifier_status=(
                                        preflight_verifier_status
                                    ),
                                    fallback_dispatched=False,
                                    fallback_returncode=None,
                                )
                            ),
                            file=sys.stderr,
                        )
                    return preflight_returncode
                try:
                    workspace_after_preflight = _workspace_content_fingerprint(
                        workspace
                    )
                except ValueError as exc:
                    print(str(exc), file=sys.stderr)
                    return preflight_returncode
                if workspace_after_preflight != workspace_baseline:
                    print(
                        "The workspace changed during the typed Grok preflight; "
                        "Codex fallback is forbidden",
                        file=sys.stderr,
                    )
                    print(
                        render_route_outcome_record(
                            route_outcome_record(
                                active_route=route_plan,
                                receipt=preflight_receipt,
                                quota_evidence_id=str(
                                    getattr(
                                        preflight_quota_evidence,
                                        "evidence_id",
                                        "",
                                    )
                                ),
                                decision="denied",
                                verifier_status=preflight_verifier_status,
                                fallback_dispatched=False,
                                fallback_returncode=None,
                            )
                        ),
                        file=sys.stderr,
                    )
                    return preflight_returncode
                preflight_fallback_reason = (
                    "authentication is unavailable"
                    if decision.reason_code == "authentication_unavailable"
                    else "quota is exhausted"
                )

    def run_authorized_preflight_fallback(
        *,
        prompt: str,
        prompt_file: Path,
    ) -> int:
        """Revalidate the typed route and dispatch without initializing Grok."""

        outcome_route = route_plan
        effect_verifier_status = preflight_verifier_status
        effect_decision = None
        attempt_store: DurableProviderAttemptCAS | None = None
        attempt_reservation: ProviderAttemptReservation | None = None
        completion_capability = ""
        completed_terminal_outcome: dict[str, object] | None = None

        invocation_binding = route_plan.invocation_binding
        if invocation_binding is not None:
            if (
                _agent_prompt_cid(prompt) != invocation_binding.prompt_cid
                or str(workspace) != invocation_binding.workspace_path
                or (
                    protected_recovery_reservation is None
                    and _repository_head(workspace)
                    != invocation_binding.baseline_commit
                )
            ):
                print(
                    "Signed invocation does not match task prompt/workspace baseline; "
                    "Codex fallback is forbidden",
                    file=sys.stderr,
                )
                return preflight_returncode
            try:
                attempt_store = DurableProviderAttemptCAS(
                    invocation_binding.provider_attempt_store,
                    expected_directory_identity=(
                        invocation_binding.provider_attempt_store_identity
                    ),
                )
            except ProviderAttemptStoreError as exc:
                print(f"provider attempt CAS is unavailable: {exc}", file=sys.stderr)
                return preflight_returncode

        def validate_effect_boundary() -> None:
            nonlocal outcome_route, effect_verifier_status, effect_decision
            try:
                hardlink_violations = _workspace_regular_file_hardlinks(
                    workspace
                )
                if hardlink_violations:
                    raise _AgentRouteEffectDenied(
                        "Codex fallback refuses multiply linked regular "
                        "workspace files: "
                        + ", ".join(str(path) for path in hardlink_violations)
                    )
                descendant_mounts = _workspace_descendant_mountpoints(workspace)
                if descendant_mounts:
                    raise _AgentRouteEffectDenied(
                        "Codex fallback refuses descendant workspace "
                        "mountpoints: "
                        + ", ".join(str(path) for path in descendant_mounts)
                    )
                if (
                    _workspace_content_fingerprint(workspace)
                    != workspace_baseline
                ):
                    raise _AgentRouteEffectDenied(
                        "workspace changed after the typed Grok preflight"
                    )
                fresh_route = resolve_agent_implementation_route_binding(
                    route_plan.as_binding_dict(),
                    repo_root=workspace,
                    now_ms=int(time.time() * 1000),
                    max_age_ms=_SCOPED_ROUTE_MAX_AGE_MS,
                )
                if _repository_head(workspace) != route_repository_head:
                    raise _AgentRouteEffectDenied(
                        "agent implementation route HEAD drifted"
                    )
                effect_decision = decide_agent_implementation_fallback(
                    fresh_route,
                    repo_root=workspace,
                    failure_receipt=preflight_receipt,
                    expected_nonce=preflight_nonce,
                    expected_model=model,
                    expected_probe_returncode=preflight_returncode,
                    independent_quota_evidence=preflight_quota_evidence,
                    expected_invocation_binding=(
                        invocation_binding.signed_payload()
                        if invocation_binding is not None
                        else None
                    ),
                    now_ms=int(time.time() * 1000),
                    max_age_ms=_SCOPED_ROUTE_MAX_AGE_MS,
                )
                if not effect_decision.authorized:
                    raise _AgentRouteEffectDenied(
                        "canonical typed fallback decision is no longer "
                        "authorized"
                    )
                if effect_decision.content_id != preflight_decision_id:
                    raise _AgentRouteEffectDenied(
                        "canonical typed fallback decision identity drifted"
                    )
                _validate_codex_quota_fallback_command(
                    codex_fallback_command,
                    workspace=workspace,
                    required_reasoning_effort=(
                        fresh_route.fallback_reasoning_effort
                    ),
                )
            except _AgentRouteEffectDenied:
                raise
            except (OSError, ValueError) as exc:
                raise _AgentRouteEffectDenied(str(exc)) from exc
            outcome_route = fresh_route
            effect_verifier_status = effect_decision.verifier_status

        def claim_provider_effect(
            launch_context: Mapping[str, object],
        ) -> None:
            nonlocal attempt_reservation, completion_capability
            if attempt_store is None:
                return
            if invocation_binding is None or effect_decision is None:
                raise _AgentRouteEffectDenied(
                    "provider effect lacks a fresh signed route decision"
                )
            # The reservation and effect_started CAS are intentionally
            # adjacent and occur only after inert Docker creation plus the
            # final router/lifecycle/freshness validation.  A failed post-
            # create validation therefore cannot poison this logical attempt
            # with a stale reserved decision.
            authorization_context = (
                build_agent_implementation_effect_authorization_context(
                    route=outcome_route,
                    repo_root=workspace,
                    failure_receipt=preflight_receipt,
                    decision=effect_decision,
                    expected_nonce=preflight_nonce,
                    expected_model=model,
                    expected_probe_returncode=preflight_returncode,
                    quota_evidence=(
                        preflight_quota_evidence
                        if effect_decision.verifier_status
                        == "confirmed_quota"
                        else None
                    ),
                )
            )
            reserved = attempt_store.reserve_or_adopt(
                logical_attempt_id=invocation_binding.logical_attempt_id,
                route_id=route_plan.route_id,
                decision_id=effect_decision.content_id,
                task_id=invocation_binding.task_id,
                worktree_id=invocation_binding.worktree_id,
                authorized=effect_decision.authorized,
                authorization_context=authorization_context,
                launch_context=launch_context,
            )
            attempt_reservation = reserved.reservation
            completion_capability = reserved.completion_capability
            if not reserved.launch_authorized:
                raise _AgentRouteEffectDenied(
                    "provider attempt was already claimed by another process"
                )

        def complete_provider_effect(returncode: int) -> None:
            """Persist the terminal route record before Docker cleanup."""

            nonlocal attempt_reservation, completed_terminal_outcome
            if attempt_store is None:
                return
            if attempt_reservation is None or not completion_capability:
                raise ProviderAttemptStoreError(
                    "provider effect terminal completion lacks the CAS winner"
                )
            terminal_outcome = route_outcome_record(
                active_route=outcome_route,
                receipt=preflight_receipt,
                quota_evidence_id=str(
                    getattr(preflight_quota_evidence, "evidence_id", "")
                ),
                decision=(
                    "fallback_succeeded" if returncode == 0 else "fallback_failed"
                ),
                verifier_status=effect_verifier_status,
                fallback_dispatched=True,
                fallback_returncode=returncode,
                reservation=attempt_reservation,
            )
            attempt_reservation = attempt_store.complete(
                attempt_reservation,
                returncode=returncode,
                outcome=terminal_outcome,
                completion_capability=completion_capability,
            )
            completed_terminal_outcome = terminal_outcome

        def adopt_started_effect(
            reservation: ProviderAttemptReservation,
            *,
            winner_capability: str = "",
        ) -> int:
            """Adopt/terminalize the exact winner without starting Docker."""

            nonlocal attempt_reservation, completion_capability
            assert attempt_store is not None
            adopted = attempt_store.adopt_effect(
                reservation,
                completion_capability=winner_capability,
            )
            if not adopted.adoption_authorized:
                if adopted.reservation.terminal:
                    terminal = adopted.reservation
                    if terminal.terminal_outcome:
                        print(
                            render_route_outcome_record(
                                terminal.terminal_outcome
                            ),
                            file=sys.stderr,
                        )
                    return int(terminal.terminal_returncode or 0)
                raise ProviderAttemptStoreError(
                    "provider effect adoption was not authorized"
                )
            attempt_reservation = adopted.reservation
            completion_capability = adopted.completion_capability
            adoption_receipt = attempt_reservation.effect_adoption_receipt
            inspection_status = adoption_receipt.get("inspection_status")
            if inspection_status == "absent":
                fallback_returncode = 125
                decision = "effect_not_created"
                fallback_dispatched = False
            elif inspection_status == "created":
                fallback_returncode = _start_recorded_codex_effect(
                    attempt_reservation.effect_launch_receipt,
                    prompt=prompt,
                )
                decision = (
                    "fallback_succeeded"
                    if fallback_returncode == 0
                    else "fallback_failed"
                )
                fallback_dispatched = True
            elif inspection_status == "exited":
                recorded_returncode = adoption_receipt.get(
                    "container_returncode"
                )
                if (
                    isinstance(recorded_returncode, bool)
                    or not isinstance(recorded_returncode, int)
                ):
                    raise ProviderAttemptStoreError(
                        "adopted Docker exit is invalid"
                    )
                fallback_returncode = recorded_returncode
                decision = (
                    "fallback_succeeded"
                    if fallback_returncode == 0
                    else "fallback_failed"
                )
                fallback_dispatched = True
            elif inspection_status == "running":
                while True:
                    try:
                        fallback_returncode = (
                            _wait_for_recorded_codex_effect(
                                attempt_reservation.effect_launch_receipt
                            )
                        )
                        break
                    except ValueError:
                        # A transient wait error is not permission to replay
                        # or fabricate completion. Re-inspect the same exact
                        # container; only an observed terminal/absence can end
                        # this owner generation.
                        latest = _inspect_recorded_codex_effect(
                            attempt_reservation.effect_launch_receipt,
                            int(time.time() * 1000),
                        )
                        if latest.get("status") == "exited":
                            fallback_returncode = int(
                                latest.get("returncode")
                            )
                            break
                        if latest.get("status") == "absent":
                            raise ProviderAttemptStoreError(
                                "running Docker effect disappeared without "
                                "an exact terminal returncode"
                            )
                        time.sleep(1.0)
                decision = (
                    "fallback_succeeded"
                    if fallback_returncode == 0
                    else "fallback_failed"
                )
                fallback_dispatched = True
            else:
                raise ProviderAttemptStoreError(
                    "effect adoption receipt is invalid"
                )
            terminal_outcome = route_outcome_record(
                active_route=outcome_route,
                receipt=preflight_receipt,
                quota_evidence_id=str(
                    getattr(preflight_quota_evidence, "evidence_id", "")
                ),
                decision=decision,
                verifier_status=effect_verifier_status,
                fallback_dispatched=fallback_dispatched,
                fallback_returncode=fallback_returncode,
                reservation=attempt_reservation,
            )
            attempt_reservation = attempt_store.complete(
                attempt_reservation,
                returncode=fallback_returncode,
                outcome=terminal_outcome,
                completion_capability=completion_capability,
            )
            _release_recorded_codex_effect_cleanup(
                attempt_reservation.effect_launch_receipt
            )
            print(
                render_route_outcome_record(terminal_outcome),
                file=sys.stderr,
            )
            return fallback_returncode

        if protected_recovery_reservation is not None:
            attempt_reservation = protected_recovery_reservation
            if attempt_reservation.state != "effect_started":
                print(
                    "protected provider recovery state cannot dispatch",
                    file=sys.stderr,
                )
                return 125
            print(
                "Adopted an effect-started provider attempt before provider "
                "preflight; replay is forbidden",
                file=sys.stderr,
            )
            try:
                return adopt_started_effect(attempt_reservation)
            except (OSError, ProviderAttemptStoreError, ValueError) as exc:
                print(
                    f"unable to adopt exact provider effect: {exc}",
                    file=sys.stderr,
                )
                return 125

        try:
            validate_effect_boundary()
        except _AgentRouteEffectDenied as exc:
            print(
                "Canonical route authority changed before fallback: "
                f"{exc}; Codex fallback is forbidden",
                file=sys.stderr,
            )
            print(
                render_route_outcome_record(
                    route_outcome_record(
                        active_route=outcome_route,
                        receipt=preflight_receipt,
                        quota_evidence_id=str(
                            getattr(
                                preflight_quota_evidence,
                                "evidence_id",
                                "",
                            )
                        ),
                        decision="denied",
                        verifier_status=effect_verifier_status,
                        fallback_dispatched=False,
                        fallback_returncode=None,
                    )
                ),
                file=sys.stderr,
            )
            return preflight_returncode

        if invocation_binding is not None:
            assert attempt_store is not None
            assert effect_decision is not None
            try:
                existing_reservation = attempt_store.read(
                    invocation_binding.logical_attempt_id
                )
            except ProviderAttemptStoreError as exc:
                print(f"provider attempt recovery denied: {exc}", file=sys.stderr)
                return preflight_returncode
            attempt_reservation = existing_reservation
            if attempt_reservation is not None and attempt_reservation.terminal:
                if attempt_reservation.terminal_outcome:
                    print(
                        render_route_outcome_record(
                            attempt_reservation.terminal_outcome
                        ),
                        file=sys.stderr,
                    )
                return int(attempt_reservation.terminal_returncode or 0)
            if (
                attempt_reservation is not None
                and attempt_reservation.state == "effect_started"
            ):
                print(
                    "Adopted an effect-started provider attempt; Docker replay is forbidden",
                    file=sys.stderr,
                )
                try:
                    return adopt_started_effect(attempt_reservation)
                except (OSError, ProviderAttemptStoreError, ValueError) as exc:
                    print(
                        f"unable to adopt exact provider effect: {exc}",
                        file=sys.stderr,
                    )
                    return 125

        print(
            "Grok "
            + preflight_fallback_reason
            + "; invoking the pinned Terra fallback",
            file=sys.stderr,
        )
        try:
            fallback_returncode = _run_codex_quota_fallback_in_docker(
                codex_fallback_command,
                workspace=workspace,
                prompt=prompt,
                prompt_path=prompt_file,
                base_env=os.environ.copy(),
                pre_effect_validator=validate_effect_boundary,
                effect_claim=claim_provider_effect,
                effect_terminal=(
                    complete_provider_effect
                    if invocation_binding is not None
                    else None
                ),
            )
            terminal_outcome = completed_terminal_outcome or route_outcome_record(
                active_route=outcome_route,
                receipt=preflight_receipt,
                quota_evidence_id=str(
                    getattr(preflight_quota_evidence, "evidence_id", "")
                ),
                decision=(
                    "fallback_succeeded"
                    if fallback_returncode == 0
                    else "fallback_failed"
                ),
                verifier_status=effect_verifier_status,
                fallback_dispatched=True,
                fallback_returncode=fallback_returncode,
                reservation=attempt_reservation,
            )
            print(
                render_route_outcome_record(terminal_outcome),
                file=sys.stderr,
            )
            return fallback_returncode
        except _AgentRouteEffectDenied as exc:
            print(
                "Canonical route authority changed at the provider effect "
                f"boundary: {exc}; Codex fallback is forbidden",
                file=sys.stderr,
            )
            print(
                render_route_outcome_record(
                    route_outcome_record(
                        active_route=outcome_route,
                        receipt=preflight_receipt,
                        quota_evidence_id=str(
                            getattr(
                                preflight_quota_evidence,
                                "evidence_id",
                                "",
                            )
                        ),
                        decision="denied",
                        verifier_status=effect_verifier_status,
                        fallback_dispatched=False,
                        fallback_returncode=None,
                    )
                ),
                file=sys.stderr,
            )
            return preflight_returncode
        except (OSError, ValueError) as exc:
            print(f"unable to launch Codex fallback: {exc}", file=sys.stderr)
            if (
                attempt_store is not None
                and attempt_reservation is not None
                and attempt_reservation.state == "effect_started"
                and completion_capability
            ):
                try:
                    # The CAS winner must reconcile the exact container.  It
                    # may not disguise a post-claim failure as an unlaunched
                    # generic route error.
                    return adopt_started_effect(
                        attempt_reservation,
                        winner_capability=completion_capability,
                    )
                except (
                    OSError,
                    ProviderAttemptStoreError,
                    ValueError,
                ) as reconciliation_error:
                    print(
                        "unable to reconcile claimed provider effect: "
                        f"{reconciliation_error}",
                        file=sys.stderr,
                    )
                    return 125
            print(
                render_route_outcome_record(
                    route_outcome_record(
                        active_route=outcome_route,
                        receipt=preflight_receipt,
                        quota_evidence_id=str(
                            getattr(
                                preflight_quota_evidence,
                                "evidence_id",
                                "",
                            )
                        ),
                        decision=(
                            "denied"
                            if invocation_binding is not None
                            else "fallback_failed"
                        ),
                        verifier_status=effect_verifier_status,
                        fallback_dispatched=False,
                        fallback_returncode=(
                            None if invocation_binding is not None else 127
                        ),
                    )
                ),
                file=sys.stderr,
            )
            return (
                preflight_returncode
                if invocation_binding is not None
                else 127
            )

    if prompt is None:
        prompt = sys.stdin.read()
    if not prompt.strip():
        print("empty implementation prompt on stdin", file=sys.stderr)
        return 2

    prompt_path = ""
    isolated_home: tempfile.TemporaryDirectory[str] | None = None
    docker_lease: _DockerContainerLease | None = None
    docker_run_finished = False
    grok_launch_env: dict[str, str] = {}
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

        if preflight_fallback_reason:
            # The fixed preflight has already established that the primary
            # cannot run.  Do not select a task-Grok sandbox, image, home, or
            # lease before entering the separately pinned Codex boundary.
            return run_authorized_preflight_fallback(
                prompt=prompt,
                prompt_file=Path(prompt_path),
            )

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
            if codex_fallback_command:
                try:
                    output_index = cmd.index("--output-format") + 1
                    cmd[output_index] = "streaming-json"
                except (ValueError, IndexError) as exc:
                    raise LLMRouterError(
                        "Grok agent command has no output-format slot"
                    ) from exc
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
                    provider="grok",
                    provider_home=_policy_path.parent,
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
                cmd = _create_grok_container_and_build_start_command(
                    cmd,
                    workspace=workspace,
                    docker_environment=grok_launch_env,
                    docker_lease=docker_lease,
                )
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
        # Without an authorized Codex fallback, project typed quota receipts and
        # exit. With a fallback, take the workspace-fenced + independent-verify
        # path so Terra may run only after verified typed provider evidence.
        if not codex_fallback_command:
            child_returncode, error_bytes, error_size, error_overflow = (
                _run_grok_with_bounded_stderr(cmd, env=grok_launch_env)
            )
            docker_run_finished = True
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

        try:
            if docker_lease is not None:
                primary_returncode = (
                    _run_created_grok_container_with_typed_failure_capture(
                        cmd,
                        docker_bin=docker_lease.docker_bin,
                        docker_config=docker_lease.docker_config,
                        cidfile=docker_lease.cidfile,
                        workspace=workspace,
                        env=grok_launch_env,
                    )
                )
            else:
                primary_returncode = _run_grok_with_typed_failure_capture(
                    cmd,
                    env=grok_launch_env,
                )
            docker_run_finished = True
        except (OSError, ValueError) as exc:
            print(f"unable to launch Grok CLI: {exc}", file=sys.stderr)
            return 127
        if primary_returncode == 0:
            return primary_returncode

        if preflight_nonce:
            print(
                "Task Grok failed after a successful typed preflight; the "
                "canonical pre-effect route does not authorize post-dispatch "
                "Codex fallback",
                file=sys.stderr,
            )
            return primary_returncode
        print(
            "Direct no-nonce Grok failure cannot authorize cross-provider "
            "fallback; use a canonical nonce-bound route",
            file=sys.stderr,
        )
        return primary_returncode
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



def _run_grok_streaming(
    command: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    **_kwargs: object,
) -> tuple[int, str]:
    """Compatibility alias for regression tests (stderr/stdout probe path)."""

    return _run_grok_with_stderr_probe(list(command), env=dict(env or {}))


def _grok_quota_exhausted(transcript: str) -> bool:
    """Return True only for complete, typed hard-quota diagnostic envelopes."""

    return bool(parse_grok_quota_error(str(transcript or "")))

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
    parser.add_argument(
        "--grok-failure-receipt-nonce",
        default="",
        help="Internal 256-bit nonce binding a runner-owned failure receipt.",
    )
    parser.add_argument(
        "--agent-implementation-route-json",
        default="",
        help="Internal frozen llm_router side-effecting route binding.",
    )
    parser.add_argument(
        "--agent-implementation-recovery-json",
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        CANONICAL_LEGACY_PREFLIGHT_ROUTE_FLAG,
        action="store_true",
        help=argparse.SUPPRESS,
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

    # Delegate to the full isolation/fallback implementation. Terra is
    # dispatched only after typed preflight auth/quota evidence or terminal
    # quota correlation plus independent verification.
    try:
        try:
            return _run(args, receipt_fd)
        except NameError as exc:
            # Infer and bind missing provider-command symbols, then retry once.
            healed = recover_provider_command_name_error(exc, globals())
            if healed is None or not healed.bound_now:
                raise
            ensure_provider_command_bindings(
                globals(),
                required=_REQUIRED_PROVIDER_COMMAND_SYMBOLS,
                namespace_name=__name__,
                strict=False,
            )
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
