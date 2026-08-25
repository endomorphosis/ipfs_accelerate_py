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
import errno
import fcntl
import hashlib
import json
import os
import re
import secrets
import select
import shutil
import signal
import socket
import stat
import struct
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

from ipfs_accelerate_py.agent_supervisor.control.lifecycle_orchestrator import (
    CONFIGURATION_ROOT_ENV,
    FENCING_EPOCH_ENV,
    PROFILE_ID_ENV,
    REPOSITORY_ROOT_ENV,
    RUN_ID_ENV,
    RUN_ROOT_ENV,
    STATE_ROOT_ENV,
    TARGET_ID_ENV,
)
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
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
    read_process_birth,
)
from ipfs_accelerate_py.agent_supervisor.validation.validation_runtime import (
    ValidationRuntimeError,
)
from ipfs_accelerate_py.llm_router import (
    AGENT_IMPLEMENTATION_CODEX_IMAGE_ID,
    AGENT_IMPLEMENTATION_CODEX_IMAGE_LABEL,
    AGENT_IMPLEMENTATION_ROUTE_OUTCOME_PREFIX,
    AGENT_IMPLEMENTATION_QUOTA_VERIFIER_DISALLOWED_TOOLS,
    find_codex_vendor_binaries,
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
    provider_stdin: socket.socket | None = None,
) -> tuple[int, bytes, int, bool]:
    """Drain child stderr without unbounded memory or disk growth."""

    try:
        process = subprocess.Popen(
            list(command),
            env=env,
            stderr=subprocess.PIPE,
            **({"stdin": provider_stdin} if provider_stdin is not None else {}),
        )
    finally:
        if provider_stdin is not None:
            provider_stdin.close()
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
# Compatibility name retained by the checked-out deterministic-repair route.
CODEX_QUOTA_FALLBACK_DEFAULT_REASONING_EFFORT = (
    DEFAULT_CODEX_QUOTA_FALLBACK_REASONING_EFFORT
)
CODEX_QUOTA_FALLBACK_REASONING_EFFORTS = frozenset({"medium", "high"})
CODEX_QUOTA_FALLBACK_REASONING = 'model_reasoning_effort="medium"'
CANONICAL_LEGACY_PREFLIGHT_ROUTE_FLAG = (
    "--canonical-legacy-preflight-route"
)
GROK_PRIMARY_SANDBOX_PROFILE = "ipfs-accelerate-provider-isolated"
GROK_ISOLATION_GROK_SANDBOX = "grok-sandbox"
GROK_ISOLATION_DOCKER = "docker"
DEFAULT_GROK_ISOLATION_IMAGE = "ubuntu:24.04"
_SEALED_PROVIDER_ISOLATION_ENV = (
    "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_EXTERNAL_ISOLATION_JSON"
)
_DOCKER_LOCAL_HOST = "unix:///var/run/docker.sock"
_DOCKER_CREATE_TIMEOUT_SECONDS = 120.0


def _sealed_provider_isolation_image_id() -> str:
    """Prefer the PCPC sealed isolation image when the daemon pinned one."""

    raw = os.environ.get(_SEALED_PROVIDER_ISOLATION_ENV, "").strip()
    if not raw:
        return ""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return ""
    if not isinstance(payload, dict):
        return ""
    image = str(payload.get("image_id") or "").strip()
    if re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        return image
    return ""


def _docker_isolation_host() -> str:
    raw = os.environ.get(_SEALED_PROVIDER_ISOLATION_ENV, "").strip()
    if raw:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            endpoint = str(payload.get("runtime_endpoint") or "").strip()
            if endpoint in {
                "unix:///var/run/docker.sock",
                f"unix:///run/user/{os.getuid()}/docker.sock",
            }:
                return endpoint
    return _DOCKER_LOCAL_HOST
_DOCKER_CLEANUP_WATCHDOG_ARG = "--internal-docker-cleanup-watchdog"
_DOCKER_CLEANUP_WATCHDOG_LAUNCHER_ARG = (
    "--internal-docker-cleanup-watchdog-launcher"
)
_DOCKER_REMOVAL_ISSUER_ARG = "--internal-docker-removal-issuer"
_DOCKER_REMOVAL_ISSUER_LAUNCHER_ARG = (
    "--internal-docker-removal-issuer-launcher"
)
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
_DOCKER_BINDING_LOCK_TIMEOUT_SECONDS = 5.0
_DOCKER_INSPECTION_MAX_BYTES = 256 * 1024
_DOCKER_CREATE_JOURNAL_NAME = "create-journal.json"
_DOCKER_CREATE_JOURNAL_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/docker-create-journal@4"
)
_DOCKER_CLEANUP_COMPLETION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/docker-cleanup-completion@5"
)
_DOCKER_CLEANUP_INTENT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/terminal-cleanup-intent@1"
)
_DOCKER_CLEANUP_BINDING_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/docker-cleanup-binding@6"
)
_DOCKER_TERMINATION_FENCE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/docker-termination-fence@1"
)
_DOCKER_REMOVAL_DISPATCH_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/docker-removal-dispatch@3"
)
_DOCKER_CLEANUP_BINDING_DIRECTORY = "provider-cleanup-bindings"
_DOCKER_PROVIDER_START_MARKER = b"ASEH_PROVIDER_START_FENCE_V2\n"
_DOCKER_PROVIDER_START_SCRIPT = (
    "IFS= read -r aseh_provider_start && "
    "[ \"$aseh_provider_start\" = ASEH_PROVIDER_START_FENCE_V2 ] "
    "|| exit 125; exec \"$@\""
)
_DOCKER_PRIVATE_CONTROL_MAX_BYTES = 512 * 1024
_DOCKER_CREATE_ENVIRONMENT_MAX_BYTES = 1024 * 1024
_DOCKER_CREATE_HANDOFF_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/docker-create-private-handoff@2"
)
_DOCKER_CREATE_HANDOFF_MAX_BYTES = 2 * 1024 * 1024
_DOCKER_CREATE_RESULT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/docker-create-private-result@2"
)
_DOCKER_CREATE_RESULT_MAX_BYTES = 2 * 1024 * 1024
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
    # The installed-module route imports from the candidate worktree.  Keep
    # those supervisor imports from writing ``__pycache__`` after the runner
    # seals its workspace fingerprint.  The accepted descriptor route stays
    # under its existing isolated-interpreter argv contract.
    runner_argv = (
        ["-I", runner]
        if runner
        else [
            "-B",
            "-m",
            "ipfs_accelerate_py.agent_supervisor.grok_cli_runner",
        ]
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
        "--codex-fallback-reasoning-effort",
        reasoning_effort,
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
    populate_credentials: bool = True,
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

        if populate_credentials:
            _populate_isolated_grok_credentials(
                base_env=base_env,
                grok_home=grok_home,
            )

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


def _populate_isolated_grok_credentials(
    *,
    base_env: Mapping[str, str],
    grok_home: Path,
) -> None:
    """Populate credentials only after ``grok_home`` has durable ownership."""

    source_home_raw = str(base_env.get("GROK_HOME") or "").strip()
    source_home = (
        Path(source_home_raw).expanduser()
        if source_home_raw
        else user_home_from_env(dict(base_env)) / ".grok"
    )
    source_auth = source_home / "auth.json"
    if not source_auth.is_file():
        try:
            import pwd

            source_auth = (
                Path(pwd.getpwuid(os.getuid()).pw_dir) / ".grok" / "auth.json"
            )
        except Exception:
            source_auth = Path.home() / ".grok" / "auth.json"
    if not source_auth.is_file():
        return
    # Copy, do not bind-mount, the operator credential.  For Docker routes the
    # caller invokes this only after the watchdog has durably bound the exact
    # provider-home inode, so SIGKILL cannot create an untracked secret tree.
    nested_home = grok_home / ".grok"
    nested_home.mkdir(mode=0o700, exist_ok=True)
    _install_ephemeral_credential(source_auth, grok_home / "auth.json")
    _install_ephemeral_credential(source_auth, nested_home / "auth.json")
    for name in ("config.toml", "agent_id"):
        extra = source_auth.parent / name
        if extra.is_file():
            try:
                _install_ephemeral_credential(extra, nested_home / name)
            except ValueError:
                continue


_MAX_ISOLATED_CREDENTIAL_BYTES = 256 * 1024


def _install_ephemeral_credential(source: Path, destination: Path) -> None:
    """Copy a bounded operator credential into ephemeral isolated state.

    The operator file is never bind-mounted and never appears in argv.  The
    container may refresh tokens only on this copy, so host login state cannot
    be mutated and deny-masks of ``~/.grok`` cannot hide auth.
    """

    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        source_fd = os.open(source, flags)
    except OSError as exc:
        raise ValueError("isolated credential is unavailable") from exc
    try:
        info = os.fstat(source_fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("isolated credential is not a regular file")
        if info.st_size > _MAX_ISOLATED_CREDENTIAL_BYTES or info.st_size < 1:
            raise ValueError("isolated credential is not bounded regular data")
        data = os.read(source_fd, info.st_size)
        if len(data) != info.st_size:
            raise ValueError("isolated credential changed while read")
    finally:
        os.close(source_fd)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    write_flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        dest_fd = os.open(destination, write_flags, 0o600)
    except OSError as exc:
        raise ValueError("isolated credential copy could not be created") from exc
    try:
        written = os.write(dest_fd, data)
        if written != len(data):
            raise ValueError("isolated credential copy is incomplete")
        os.fchmod(dest_fd, 0o600)
        os.fsync(dest_fd)
    finally:
        os.close(dest_fd)


def _populate_bound_ephemeral_prompt(path: Path, prompt: str) -> None:
    """Write task context through the exact empty inode bound by cleanup."""

    payload = prompt.encode("utf-8")
    if not payload or len(payload) > 8 * 1024 * 1024:
        raise ValueError("implementation prompt is not bounded")
    try:
        before = os.lstat(path)
        descriptor = os.open(
            path,
            os.O_WRONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise ValueError("ephemeral prompt inode is unavailable") from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or before.st_uid != os.geteuid()
            or before.st_nlink != 1
            or (before.st_dev, before.st_ino)
            != (opened.st_dev, opened.st_ino)
            or opened.st_uid != os.geteuid()
            or opened.st_nlink != 1
        ):
            raise ValueError("ephemeral prompt ownership changed")
        os.fchmod(descriptor, 0o600)
        os.ftruncate(descriptor, 0)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise ValueError("ephemeral prompt write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        after = os.lstat(path)
        if (
            (after.st_dev, after.st_ino) != (opened.st_dev, opened.st_ino)
            or after.st_uid != os.geteuid()
            or after.st_nlink != 1
            or not stat.S_ISREG(after.st_mode)
        ):
            raise ValueError("ephemeral prompt inode changed while written")
    finally:
        os.close(descriptor)


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

    # Prefer the sealed PCPC isolation image when the daemon pinned one; the
    # ubuntu:24.04 tag remains the standalone Grok-runner default. Create and
    # start stay on the same Docker host so the container ID remains visible.
    del base_env
    images = []
    sealed = _sealed_provider_isolation_image_id()
    if sealed:
        images.append(sealed)
    images.append(DEFAULT_GROK_ISOLATION_IMAGE)
    for image in images:
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
            continue
        candidate = completed.stdout.strip()
        if (
            completed.returncode == 0
            and re.fullmatch(r"sha256:[0-9a-f]{64}", candidate)
        ):
            return candidate
    return ""


def _docker_codex_task_toolchain_image_id(
    docker_bin: str,
    *,
    docker_config: Path,
) -> str:
    """Verify the immutable image that supplies the bounded test toolchain."""

    sealed = _sealed_provider_isolation_image_id()
    if sealed:
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
                    sealed,
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
            sealed
            if completed.returncode == 0 and candidate == sealed
            else ""
        )
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


_DOCKER_WATCHDOG_LIFECYCLE_ENV_NAMES = (
    RUN_ID_ENV,
    PROFILE_ID_ENV,
    TARGET_ID_ENV,
    REPOSITORY_ROOT_ENV,
    STATE_ROOT_ENV,
    RUN_ROOT_ENV,
    FENCING_EPOCH_ENV,
    CONFIGURATION_ROOT_ENV,
)
_DOCKER_EFFECT_OBSERVATION_FIELDS = frozenset(
    {
        "logical_attempt_id",
        "provider_attempt_store",
        "provider_attempt_store_identity",
    }
)


def _docker_cleanup_watchdog_env() -> dict[str, str]:
    """Project lifecycle identity, never state credentials, to the reaper.

    The configured multi-supervisor stop path snapshots every profile member,
    so this detached auxiliary root becomes a synchronous cleanup barrier.
    Generic single-root health checks see it as non-healthy and therefore fail
    closed; they never mistake an auxiliary reaper for a second supervisor.
    """

    environment = _docker_control_env()
    projected = {
        name: str(os.environ[name])
        for name in _DOCKER_WATCHDOG_LIFECYCLE_ENV_NAMES
        if name in os.environ
    }
    if projected and len(projected) != len(
        _DOCKER_WATCHDOG_LIFECYCLE_ENV_NAMES
    ):
        raise ValueError("Docker cleanup watchdog lifecycle identity is partial")
    environment.update(projected)
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


_DOCKER_TERMINATION_FENCE_FIELDS = frozenset(
    {
        "schema",
        "provider",
        "container_id",
        "container_name",
        "image_id",
        "isolation_label",
        "docker_state",
        "init_pid",
        "kernel_scope",
        "fence_id",
    }
)


def _validated_docker_termination_fence(
    value: Mapping[str, object],
    *,
    provider: str,
    container_name: str,
    expected_container_id: str = "",
    expected_image_id: str = "",
) -> dict[str, object]:
    """Validate one exact Docker effect and its captured kernel scope."""

    from ipfs_accelerate_py.agent_supervisor.runtime.process_security import (
        StateAuthorityProcessIsolationError,
        validate_linux_process_scope,
    )

    body = {name: item for name, item in value.items() if name != "fence_id"}
    container_id = str(value.get("container_id") or "")
    image_id = str(value.get("image_id") or "")
    init_pid = value.get("init_pid")
    kernel_scope = value.get("kernel_scope")
    expected_label = (
        "ipfs_accelerate.grok_isolation"
        if provider == "grok"
        else "ipfs_accelerate.codex_fallback_isolation"
    )
    if (
        set(value) != _DOCKER_TERMINATION_FENCE_FIELDS
        or value.get("schema") != _DOCKER_TERMINATION_FENCE_SCHEMA
        or provider not in _DOCKER_ISOLATION_PROVIDERS
        or value.get("provider") != provider
        or value.get("container_name") != container_name
        or _DOCKER_CONTAINER_NAME_RE.fullmatch(container_name) is None
        or not container_name.startswith(f"ipfs-accelerate-{provider}-")
        or re.fullmatch(r"[0-9a-f]{64}", container_id) is None
        or (
            expected_container_id
            and container_id != expected_container_id
        )
        or re.fullmatch(r"sha256:[0-9a-f]{64}", image_id) is None
        or (expected_image_id and image_id != expected_image_id)
        or value.get("isolation_label") != expected_label
        or value.get("docker_state")
        not in {"created", "running", "paused", "restarting"}
        or type(init_pid) is not int
        or int(init_pid) < 0
        or value.get("fence_id") != _effect_receipt_identity(body)
    ):
        raise ValueError("Docker termination fence identity is invalid")
    if init_pid:
        if not isinstance(kernel_scope, Mapping):
            raise ValueError("Docker termination kernel scope is absent")
        try:
            scope = validate_linux_process_scope(kernel_scope)
        except StateAuthorityProcessIsolationError as exc:
            raise ValueError("Docker termination kernel scope is invalid") from exc
        if (
            scope.get("pid") != init_pid
            or value.get("docker_state") == "created"
        ):
            raise ValueError("Docker termination kernel scope differs")
    elif kernel_scope != {} or value.get("docker_state") != "created":
        # Docker reports State.Pid=0 after an executed container exits.  At
        # that point a detached descendant can still populate the old cgroup,
        # so a post-hoc name/CID observation cannot mint a cleanup fence.  The
        # only safe zero-PID case is an inert container that never crossed its
        # start boundary.
        raise ValueError("inactive Docker termination scope is not inert")
    return dict(value)


def _attest_exact_docker_execution(
    *,
    docker_bin: str,
    docker_config: str | Path,
    provider: str,
    container_name: str,
    container_id: str,
    image_id: str,
    timeout: float,
    pass_fds: tuple[int, ...] = (),
) -> dict[str, object]:
    """Capture immutable Docker and Linux process-scope identity before rm."""

    from .process_security import (
        StateAuthorityProcessIsolationError,
        capture_linux_process_scope,
    )

    try:
        observed = subprocess.run(
            [
                docker_bin,
                f"--host={_DOCKER_LOCAL_HOST}",
                "--config",
                str(docker_config),
                "container",
                "inspect",
                container_id,
            ],
            env=_docker_control_env(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=max(0.05, timeout),
            check=False,
            pass_fds=pass_fds,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("Docker execution identity is unavailable") from exc
    if (
        observed.returncode != 0
        or not observed.stdout
        or len(observed.stdout) > _DOCKER_INSPECTION_MAX_BYTES
    ):
        raise ValueError("Docker execution could not be inspected")
    try:
        decoded = json.loads(observed.stdout.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Docker execution inspection is malformed") from exc
    if (
        not isinstance(decoded, list)
        or len(decoded) != 1
        or not isinstance(decoded[0], Mapping)
    ):
        raise ValueError("Docker execution inspection shape is invalid")
    inspection = decoded[0]
    config = inspection.get("Config")
    labels = config.get("Labels") if isinstance(config, Mapping) else None
    state = inspection.get("State")
    expected_label = (
        "ipfs_accelerate.grok_isolation"
        if provider == "grok"
        else "ipfs_accelerate.codex_fallback_isolation"
    )
    if not isinstance(state, Mapping):
        raise ValueError("Docker execution state is unavailable")
    raw_pid = state.get("Pid")
    docker_state = str(state.get("Status") or "")
    if (
        inspection.get("Id") != container_id
        or inspection.get("Name") != "/" + container_name
        or inspection.get("Image") != image_id
        or not isinstance(labels, Mapping)
        or labels.get(expected_label) != "true"
        or type(raw_pid) is not int
        or raw_pid < 0
    ):
        raise ValueError("Docker execution identity differs")
    try:
        kernel_scope: Mapping[str, object] = (
            capture_linux_process_scope(raw_pid) if raw_pid else {}
        )
    except StateAuthorityProcessIsolationError as exc:
        raise ValueError("Docker execution kernel scope is unavailable") from exc
    # Close the inspect -> /proc capture race.  A PID that exited or was
    # reused between those observations must never be bound to this container.
    try:
        confirmed = subprocess.run(
            [
                docker_bin,
                f"--host={_DOCKER_LOCAL_HOST}",
                "--config",
                str(docker_config),
                "container",
                "inspect",
                container_id,
            ],
            env=_docker_control_env(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=max(0.05, timeout),
            check=False,
            pass_fds=pass_fds,
        )
        confirmed_decoded = json.loads(confirmed.stdout.decode("utf-8"))
    except (
        OSError,
        subprocess.TimeoutExpired,
        UnicodeError,
        json.JSONDecodeError,
    ) as exc:
        raise ValueError("Docker execution confirmation is unavailable") from exc
    if (
        confirmed.returncode != 0
        or len(confirmed.stdout) > _DOCKER_INSPECTION_MAX_BYTES
        or not isinstance(confirmed_decoded, list)
        or len(confirmed_decoded) != 1
        or not isinstance(confirmed_decoded[0], Mapping)
    ):
        raise ValueError("Docker execution confirmation is malformed")
    confirmed_inspection = confirmed_decoded[0]
    confirmed_config = confirmed_inspection.get("Config")
    confirmed_labels = (
        confirmed_config.get("Labels")
        if isinstance(confirmed_config, Mapping)
        else None
    )
    confirmed_state = confirmed_inspection.get("State")
    if (
        confirmed_inspection.get("Id") != container_id
        or confirmed_inspection.get("Name") != "/" + container_name
        or confirmed_inspection.get("Image") != image_id
        or not isinstance(confirmed_labels, Mapping)
        or confirmed_labels.get(expected_label) != "true"
        or not isinstance(confirmed_state, Mapping)
        or confirmed_state.get("Pid") != raw_pid
        or str(confirmed_state.get("Status") or "") != docker_state
    ):
        raise ValueError("Docker execution changed during kernel capture")
    body: dict[str, object] = {
        "schema": _DOCKER_TERMINATION_FENCE_SCHEMA,
        "provider": provider,
        "container_id": container_id,
        "container_name": container_name,
        "image_id": image_id,
        "isolation_label": expected_label,
        "docker_state": docker_state,
        "init_pid": raw_pid,
        "kernel_scope": dict(kernel_scope),
    }
    body["fence_id"] = _effect_receipt_identity(body)
    return _validated_docker_termination_fence(
        body,
        provider=provider,
        container_name=container_name,
        expected_container_id=container_id,
        expected_image_id=image_id,
    )


def _docker_termination_scope_quiescent(
    fence: Mapping[str, object],
) -> bool:
    """Require the captured init, PID namespace, and cgroup to be empty."""

    from .process_security import (
        StateAuthorityProcessIsolationError,
        linux_process_scope_quiescent,
    )

    provider = str(fence.get("provider") or "")
    container_name = str(fence.get("container_name") or "")
    admitted = _validated_docker_termination_fence(
        fence,
        provider=provider,
        container_name=container_name,
    )
    if admitted["init_pid"] == 0:
        return True
    scope = admitted["kernel_scope"]
    try:
        return bool(
            isinstance(scope, Mapping)
            and linux_process_scope_quiescent(scope)
        )
    except StateAuthorityProcessIsolationError:
        return False


def _cleanup_path_identity(path: Path, *, directory: bool) -> dict[str, int]:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise ValueError("Docker cleanup path identity is unavailable") from exc
    if (
        metadata.st_uid != os.geteuid()
        or stat.S_ISLNK(metadata.st_mode)
        or (directory and not stat.S_ISDIR(metadata.st_mode))
        or (not directory and not stat.S_ISREG(metadata.st_mode))
    ):
        raise ValueError("Docker cleanup path identity is unsafe")
    return {
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        # Bind the object kind, not mutable permission bits.  The provider is
        # allowed to chmod files inside its own isolated home; cleanup still
        # owns the same inode and normalizes permissions only after moving it
        # into the private quarantine below.
        "mode": stat.S_IFMT(metadata.st_mode),
        "uid": metadata.st_uid,
    }


def _docker_cleanup_root_identity(path: Path) -> dict[str, int]:
    """Bind one trusted shared or private provider allocator root."""

    root = path.absolute()
    descriptor = -1
    try:
        descriptor = os.open(
            root,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        metadata = os.fstat(descriptor)
        named = os.lstat(root)
        resolved = root.resolve(strict=True)
        final = os.lstat(root)
    except OSError as exc:
        raise ValueError("Docker cleanup root identity is unavailable") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    permissions = stat.S_IMODE(metadata.st_mode)
    private_owned = metadata.st_uid == os.geteuid() and permissions == 0o700
    trusted_shared = metadata.st_uid == 0 and permissions == 0o1777
    if (
        resolved != root
        or stat.S_ISLNK(named.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
            metadata.st_uid,
        )
        != (
            named.st_dev,
            named.st_ino,
            named.st_mode,
            named.st_uid,
        )
        or (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
            metadata.st_uid,
        )
        != (
            final.st_dev,
            final.st_ino,
            final.st_mode,
            final.st_uid,
        )
        or not (private_owned or trusted_shared)
    ):
        raise ValueError("Docker cleanup root identity is unsafe")
    return {
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "mode": metadata.st_mode,
        "uid": metadata.st_uid,
    }


def _validated_docker_cleanup_root(
    *,
    lease_root: Path,
    provider_home: Path,
    prompt_path: Path,
    expected_root: Path | None = None,
    expected_identity: Mapping[str, object] | None = None,
) -> tuple[Path, dict[str, int]]:
    """Admit the exact direct parent shared by all disposable resources."""

    cleanup_root = lease_root.parent.absolute()
    if (
        not lease_root.is_absolute()
        or not provider_home.is_absolute()
        or not prompt_path.is_absolute()
        or provider_home.parent != cleanup_root
        or prompt_path.parent != cleanup_root
        or (expected_root is not None and expected_root != cleanup_root)
    ):
        raise ValueError("Docker cleanup resources do not share one root")
    identity = _docker_cleanup_root_identity(cleanup_root)
    if expected_identity is not None:
        if (
            not isinstance(expected_identity, Mapping)
            or set(expected_identity) != {"device", "inode", "mode", "uid"}
            or any(
                isinstance(expected_identity.get(name), bool)
                or not isinstance(expected_identity.get(name), int)
                or int(expected_identity[name]) < 0
                for name in ("device", "inode", "mode", "uid")
            )
            or dict(expected_identity) != identity
        ):
            raise ValueError("Docker cleanup root identity drifted")
    return cleanup_root, identity


def _provider_start_socketpair() -> tuple[socket.socket, socket.socket]:
    """Mint one anonymous, procfs-nonreopenable provider-start capability."""

    sender, docker_stdin = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        for channel in (sender, docker_stdin):
            channel.set_inheritable(False)
            if (
                channel.family != socket.AF_UNIX
                or channel.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE)
                != socket.SOCK_STREAM
            ):
                raise ValueError("Docker provider start capability is invalid")
        return sender, docker_stdin
    except BaseException:
        sender.close()
        docker_stdin.close()
        raise


def _docker_cleanup_binding_value(
    *,
    binding_state: str,
    provider: str,
    docker_bin: str,
    container_name: str,
    lease_root: Path,
    docker_config: Path,
    cidfile: Path,
    provider_home: Path,
    prompt_path: Path,
    effect_observation: Mapping[str, str],
    path_identities: Mapping[str, Mapping[str, int]],
    binding_path: Path,
    runner_pid: int,
    runner_start_ticks: int,
    watchdog_pid: int,
    watchdog_start_ticks: int,
    create_command_id: str = "",
    create_cwd: Path | None = None,
    create_environment_id: str = "",
    termination_fence: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build either the pre-dispatch or command-bound durable authority."""

    if binding_state not in {"prepared_no_dispatch", "command_bound"}:
        raise ValueError("Docker cleanup binding state is invalid")
    cleanup_root, cleanup_root_identity = _validated_docker_cleanup_root(
        lease_root=lease_root,
        provider_home=provider_home,
        prompt_path=prompt_path,
    )
    lifecycle = {
        name: str(os.environ.get(name, "") or "").strip()
        for name in _DOCKER_WATCHDOG_LIFECYCLE_ENV_NAMES
    }
    try:
        fencing_epoch = int(lifecycle[FENCING_EPOCH_ENV])
        docker_metadata = Path(docker_bin).stat()
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="ascii"
        ).strip()
    except (OSError, ValueError) as exc:
        raise ValueError("Docker cleanup binding identity is unavailable") from exc
    command_bound = binding_state == "command_bound"
    fence = dict(termination_fence or {})
    if (
        not all(lifecycle.values())
        or fencing_epoch < 0
        or not boot_id
        or runner_pid <= 0
        or runner_start_ticks <= 0
        or watchdog_pid <= 0
        or watchdog_start_ticks <= 0
        or command_bound
        != bool(
            re.fullmatch(r"sha256:[0-9a-f]{64}", create_command_id)
            and create_cwd is not None
            and create_cwd.is_absolute()
            and re.fullmatch(r"sha256:[0-9a-f]{64}", create_environment_id)
        )
    ):
        raise ValueError("Docker cleanup binding identity is incomplete")
    if fence:
        if not command_bound:
            raise ValueError(
                "Docker termination fence requires command-bound cleanup"
            )
        _validated_docker_termination_fence(
            fence,
            provider=provider,
            container_name=container_name,
        )
    body: dict[str, object] = {
        "schema": _DOCKER_CLEANUP_BINDING_SCHEMA,
        "binding_state": binding_state,
        "run_id": lifecycle[RUN_ID_ENV],
        "profile_id": lifecycle[PROFILE_ID_ENV],
        "target_id": lifecycle[TARGET_ID_ENV],
        "repository_root": lifecycle[REPOSITORY_ROOT_ENV],
        "state_root": lifecycle[STATE_ROOT_ENV],
        "run_root": lifecycle[RUN_ROOT_ENV],
        "configuration_root": lifecycle[CONFIGURATION_ROOT_ENV],
        "fencing_epoch": fencing_epoch,
        "runner_pid": runner_pid,
        "runner_start_ticks": runner_start_ticks,
        "watchdog_pid": watchdog_pid,
        "watchdog_start_ticks": watchdog_start_ticks,
        "boot_id": boot_id,
        "provider": provider,
        "docker_bin": docker_bin,
        "docker_device": docker_metadata.st_dev,
        "docker_inode": docker_metadata.st_ino,
        "docker_mode": docker_metadata.st_mode,
        "docker_uid": docker_metadata.st_uid,
        "container_name": container_name,
        "cleanup_root": str(cleanup_root),
        "cleanup_root_identity": cleanup_root_identity,
        "lease_root": str(lease_root),
        "docker_config": str(docker_config),
        "cidfile": str(cidfile),
        "provider_home": str(provider_home),
        "prompt_path": str(prompt_path),
        "effect_observation": dict(sorted(effect_observation.items())),
        "create_command_id": create_command_id if command_bound else "",
        "create_cwd": str(create_cwd) if command_bound else "",
        "create_environment_id": (
            create_environment_id if command_bound else ""
        ),
        "termination_fence": fence,
        "path_identities": {
            name: dict(identity)
            for name, identity in path_identities.items()
        },
        "binding_path": str(binding_path),
    }
    body["record_id"] = _effect_receipt_identity(body)
    return body


def _reject_duplicate_control_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("private Docker control record repeats a key")
        value[key] = item
    return value


def _validate_private_control_directory_descriptor(
    path: Path,
    descriptor: int,
) -> None:
    """Require a retained directory descriptor to remain the canonical name."""

    absolute = path.absolute()
    try:
        if absolute.resolve(strict=True) != absolute:
            raise ValueError("private Docker control directory is aliased")
        metadata = os.fstat(descriptor)
        final = os.lstat(absolute)
    except OSError as exc:
        raise ValueError("private Docker control directory is unavailable") from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
            metadata.st_uid,
        )
        != (
            final.st_dev,
            final.st_ino,
            final.st_mode,
            final.st_uid,
        )
    ):
        raise ValueError("private Docker control directory is not owned")


def _private_control_directory(path: Path) -> int:
    """Bind one exact private directory without following a replacement."""

    absolute = path.absolute()
    descriptor: int | None = None
    try:
        descriptor = os.open(
            absolute,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        _validate_private_control_directory_descriptor(absolute, descriptor)
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise ValueError(
            "private Docker control directory is unavailable"
        ) from exc
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        raise
    return descriptor


def _private_control_bytes(value: Mapping[str, object]) -> bytes:
    encoded = (
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    if len(encoded) > _DOCKER_PRIVATE_CONTROL_MAX_BYTES:
        raise ValueError("private Docker control record is oversized")
    return encoded


def _write_private_control_record(
    directory: Path,
    name: str,
    value: Mapping[str, object],
    *,
    replace_existing: bool,
    directory_fd: int | None = None,
) -> None:
    """Publish canonical private JSON with create-only or atomic CAS shape."""

    if re.fullmatch(r"[a-z0-9][a-z0-9.-]{0,127}", name) is None:
        raise ValueError("private Docker control record name is invalid")
    encoded = _private_control_bytes(value)
    owns_directory_fd = directory_fd is None
    bound_directory_fd = (
        _private_control_directory(directory)
        if directory_fd is None
        else directory_fd
    )
    if not owns_directory_fd:
        _validate_private_control_directory_descriptor(
            directory,
            bound_directory_fd,
        )
    temporary_name = "." + name + "." + secrets.token_hex(8)
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=bound_directory_fd,
        )
        try:
            view = memoryview(encoded)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("private Docker control write made no progress")
                view = view[written:]
            os.fchmod(descriptor, 0o600)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if replace_existing:
            os.replace(
                temporary_name,
                name,
                src_dir_fd=bound_directory_fd,
                dst_dir_fd=bound_directory_fd,
            )
        else:
            os.link(
                temporary_name,
                name,
                src_dir_fd=bound_directory_fd,
                dst_dir_fd=bound_directory_fd,
                follow_symlinks=False,
            )
            os.unlink(temporary_name, dir_fd=bound_directory_fd)
        os.fsync(bound_directory_fd)
        _validate_private_control_directory_descriptor(
            directory,
            bound_directory_fd,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError("private Docker control record cannot be published") from exc
    finally:
        try:
            os.unlink(temporary_name, dir_fd=bound_directory_fd)
        except FileNotFoundError:
            pass
        finally:
            if owns_directory_fd:
                os.close(bound_directory_fd)


def _read_private_control_record(
    directory: Path,
    name: str,
    *,
    directory_fd: int | None = None,
) -> dict[str, object] | None:
    """Read one exact canonical private JSON record without mutating it."""

    owns_directory_fd = directory_fd is None
    bound_directory_fd = (
        _private_control_directory(directory)
        if directory_fd is None
        else directory_fd
    )
    if not owns_directory_fd:
        _validate_private_control_directory_descriptor(
            directory,
            bound_directory_fd,
        )
    try:
        try:
            descriptor = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NONBLOCK", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=bound_directory_fd,
            )
        except FileNotFoundError:
            _validate_private_control_directory_descriptor(
                directory,
                bound_directory_fd,
            )
            return None
        before = os.fstat(descriptor)
        try:
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_uid != os.geteuid()
                or before.st_nlink != 1
                or stat.S_IMODE(before.st_mode) != 0o600
                or before.st_size > _DOCKER_PRIVATE_CONTROL_MAX_BYTES
            ):
                raise ValueError("private Docker control record is unsafe")
            remaining = _DOCKER_PRIVATE_CONTROL_MAX_BYTES + 1
            chunks: list[bytes] = []
            while remaining:
                chunk = os.read(descriptor, min(64 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
            after = os.fstat(descriptor)
            final = os.stat(
                name,
                dir_fd=bound_directory_fd,
                follow_symlinks=False,
            )
        finally:
            os.close(descriptor)
        snapshots = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_uid,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        if snapshots != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_uid,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) or snapshots != (
            final.st_dev,
            final.st_ino,
            final.st_mode,
            final.st_uid,
            final.st_nlink,
            final.st_size,
            final.st_mtime_ns,
            final.st_ctime_ns,
        ):
            raise ValueError("private Docker control record changed while read")
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_control_keys,
        )
        _validate_private_control_directory_descriptor(
            directory,
            bound_directory_fd,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("private Docker control record is unreadable") from exc
    finally:
        if owns_directory_fd:
            os.close(bound_directory_fd)
    if type(value) is not dict or raw != _private_control_bytes(value):
        raise ValueError("private Docker control record is noncanonical")
    return value


def _unlink_private_control_record(directory: Path, name: str) -> None:
    directory_fd = _private_control_directory(directory)
    try:
        try:
            metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            raise ValueError("private Docker control record is unsafe")
        os.unlink(name, dir_fd=directory_fd)
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _docker_cleanup_binding_path(
    container_name: str,
    *,
    create_directory: bool = True,
) -> Path | None:
    """Return the lifecycle-owned durable binding path, when supervised."""

    lifecycle = {
        name: str(os.environ.get(name, "") or "").strip()
        for name in _DOCKER_WATCHDOG_LIFECYCLE_ENV_NAMES
    }
    if not any(lifecycle.values()):
        return None
    if not all(lifecycle.values()):
        raise ValueError("Docker cleanup lifecycle binding is partial")
    run_root = Path(lifecycle[RUN_ROOT_ENV])
    state_root = Path(lifecycle[STATE_ROOT_ENV])
    try:
        resolved_run_root = run_root.resolve(strict=True)
        resolved_state_root = state_root.resolve(strict=True)
    except OSError as exc:
        raise ValueError("Docker cleanup lifecycle root is unavailable") from exc
    if (
        resolved_run_root != run_root.absolute()
        or resolved_state_root != state_root.absolute()
        or not resolved_run_root.is_relative_to(resolved_state_root)
    ):
        raise ValueError("Docker cleanup lifecycle root is invalid")
    binding_directory = resolved_run_root / _DOCKER_CLEANUP_BINDING_DIRECTORY
    token = hashlib.sha256(container_name.encode("ascii")).hexdigest()
    if not create_directory and not binding_directory.exists():
        return binding_directory / (token + ".json")
    try:
        if create_directory:
            binding_directory.mkdir(mode=0o700, exist_ok=True)
        if stat.S_IMODE(binding_directory.stat().st_mode) != 0o700:
            raise ValueError("Docker cleanup binding directory is not private")
        descriptor = _private_control_directory(binding_directory)
        os.close(descriptor)
    except OSError as exc:
        raise ValueError("Docker cleanup binding directory is unavailable") from exc
    return binding_directory / (token + ".json")


def _runner_process_start_ticks(pid: int) -> int:
    """Return one Linux process birth identity without trusting its argv."""

    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise ValueError("process identity is invalid")
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        closing_parenthesis = raw.rfind(")")
        fields = raw[closing_parenthesis + 2 :].split()
        ticks = int(fields[19])
    except (OSError, IndexError, UnicodeError, ValueError) as exc:
        raise ValueError("process identity is unavailable") from exc
    if closing_parenthesis < 0 or ticks < 0:
        raise ValueError("process identity is invalid")
    return ticks


def _runner_process_identity_alive(pid: int, start_ticks: int) -> bool:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        closing_parenthesis = raw.rfind(")")
        fields = raw[closing_parenthesis + 2 :].split()
        return (
            closing_parenthesis >= 0
            and fields[0] != "Z"
            and int(fields[19]) == start_ticks
        )
    except (OSError, IndexError, UnicodeError, ValueError):
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

    docker = _docker_isolation_binary()
    if docker:
        # Docker routes have a detached, lifecycle-bound cleanup owner before
        # prompt/auth bytes are populated.  Prefer that recoverable boundary
        # even when bubblewrap is available; a native temporary home cannot be
        # reclaimed after an uncatchable runner death.
        return GROK_ISOLATION_DOCKER
    if require_container_boundary:
        raise ValueError(
            "Default Grok quota route requires the pinned local Docker "
            "isolation image"
        )
    if _grok_custom_sandbox_available():
        return GROK_ISOLATION_GROK_SANDBOX
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


def _docker_codex_host_vendor_mounts() -> list[str]:
    """Project the router-admitted native Codex pair as one read-only mount."""

    vendor = find_codex_vendor_binaries()
    if vendor is None:
        return []
    host_codex, host_companion = vendor
    if host_codex.parent != host_companion.parent:
        return []
    try:
        return _docker_mount(
            host_codex.parent,
            destination=Path("/usr/local/bin"),
            read_only=True,
        )
    except (OSError, ValueError):
        return []


def _remove_exact_docker_container(
    *,
    docker_bin: str,
    docker_config: Path,
    container_name: str,
    settle_for_creation: bool,
    deadline: float | None = None,
    termination_fence: Mapping[str, object] | None = None,
    pass_fds: tuple[int, ...] = (),
    issue_removal: bool = False,
) -> None:
    """Remove one fenced CID and prove Docker plus kernel-scope quiescence."""

    started_at = time.monotonic()
    settle_deadline = started_at + _DOCKER_CLEANUP_TIMEOUT_SECONDS
    hard_deadline = min(
        settle_deadline + 2.0,
        deadline if deadline is not None else float("inf"),
    )
    fence: dict[str, object] | None = None
    if termination_fence:
        provider = str(termination_fence.get("provider") or "")
        fence = _validated_docker_termination_fence(
            termination_fence,
            provider=provider,
            container_name=container_name,
        )
    absence_samples = 0
    removal_attempted = not issue_removal
    while True:
        remaining = hard_deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError(
                "exact Docker container cleanup could not be verified"
            )
        if fence is not None and not removal_attempted:
            # A timed-out ``docker rm`` has an unknown daemon-side outcome.
            # For a known completed create, issue the idempotent request only
            # once and reconcile by observation for the rest of the bound.
            removal_attempted = True
            try:
                completed = subprocess.run(
                    [
                        docker_bin,
                        f"--host={_DOCKER_LOCAL_HOST}",
                        "--config",
                        str(docker_config),
                        "rm",
                        "--force",
                        str(fence["container_id"]),
                    ],
                    env=_docker_control_env(),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=min(2.0, max(0.05, remaining)),
                    check=False,
                    pass_fds=pass_fds,
                )
                del completed
            except (OSError, subprocess.TimeoutExpired):
                pass
        # ``docker rm`` can time out after daemon acceptance.  Never replay
        # that unknown request.  Reconcile the immutable ID, reserved name,
        # captured init birth, PID namespace, and exact cgroup instead.
        remaining = hard_deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError(
                "exact Docker container cleanup could not be verified"
            )
        identity_absent = fence is None
        name_absent = False
        try:
            if fence is not None:
                observed_id = subprocess.run(
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
                        f"id={fence['container_id']}",
                        "--format",
                        "{{.ID}}",
                    ],
                    env=_docker_control_env(),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    timeout=min(2.0, max(0.05, remaining)),
                    check=False,
                    pass_fds=pass_fds,
                )
                identity_absent = bool(
                    observed_id.returncode == 0
                    and len(observed_id.stdout)
                    <= _DOCKER_INSPECTION_MAX_BYTES
                    and not observed_id.stdout.strip()
                )
            remaining = hard_deadline - time.monotonic()
            if remaining <= 0:
                raise ValueError(
                    "exact Docker container cleanup could not be verified"
                )
            observed_name = subprocess.run(
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
                timeout=min(2.0, max(0.05, remaining)),
                check=False,
                pass_fds=pass_fds,
            )
            name_absent = bool(
                observed_name.returncode == 0
                and len(observed_name.stdout) <= _DOCKER_INSPECTION_MAX_BYTES
                and not observed_name.stdout.strip()
            )
        except (OSError, subprocess.TimeoutExpired):
            identity_absent = False
            name_absent = False
        kernel_quiescent = bool(
            fence is None or _docker_termination_scope_quiescent(fence)
        )
        if identity_absent and name_absent and kernel_quiescent:
            absence_samples += 1
        else:
            absence_samples = 0
        now = time.monotonic()
        if absence_samples >= 2 and (
            not settle_for_creation or now >= settle_deadline
        ):
            return
        if now >= hard_deadline:
            raise ValueError(
                "exact Docker container cleanup could not be verified"
            )
        time.sleep(
            min(
                0.1,
                max(
                    0.0,
                    hard_deadline - now,
                ),
            )
        )


def _docker_create_command_identity(
    *,
    provider: str,
    docker_bin: str,
    docker_config: Path,
    container_name: str,
    cidfile: Path,
    cwd: Path,
    environment_id: str,
    expected_image: str,
    argv: Sequence[str],
) -> tuple[str, dict[str, object]]:
    values = [str(item) for item in argv]
    try:
        resolved_cwd = cwd.resolve(strict=True)
    except OSError as exc:
        raise ValueError("Docker create working directory is unavailable") from exc
    expected_label = (
        "ipfs_accelerate.grok_isolation=true"
        if provider == "grok"
        else "ipfs_accelerate.codex_fallback_isolation=true"
    )
    if (
        provider not in _DOCKER_ISOLATION_PROVIDERS
        or resolved_cwd != cwd.absolute()
        or not resolved_cwd.is_dir()
        or re.fullmatch(r"sha256:[0-9a-f]{64}", environment_id) is None
        or len(values) < 6
        or len(values) > 512
        or any(not item or "\x00" in item for item in values)
        or values[:5]
        != [
            docker_bin,
            f"--host={_DOCKER_LOCAL_HOST}",
            "--config",
            str(docker_config),
            "create",
        ]
    ):
        raise ValueError("Docker create command is not exact and inert")

    # Parse Docker's option region with a positive grammar.  The first
    # non-option is the IMAGE operand; a digest appearing later in container
    # argv can never satisfy image pinning.
    fixed_options = {
        "--pull=never",
        "--interactive",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=1024",
    }
    codex_fixed_options = {
        "--network=bridge",
        "--runtime=runc",
        "--entrypoint=/usr/bin/env",
    }
    grok_fixed_options = {"--entrypoint=/bin/sh"}
    value_options = {
        "--name",
        "--cidfile",
        "--label",
        "--user",
        "--workdir",
        "--tmpfs",
        "--env",
        "--mount",
    }
    seen: dict[str, list[str]] = {}
    index = 5
    image = ""
    while index < len(values):
        item = values[index]
        if (
            item in fixed_options
            or (provider == "codex" and item in codex_fixed_options)
            or (provider == "grok" and item in grok_fixed_options)
        ):
            seen.setdefault(item, []).append("")
            index += 1
            continue
        if item in value_options:
            if index + 1 >= len(values):
                raise ValueError("Docker create option value is absent")
            seen.setdefault(item, []).append(values[index + 1])
            index += 2
            continue
        if item.startswith("-"):
            raise ValueError("Docker create option is not allowlisted")
        image = item
        index += 1
        break
    container_argv = values[index:]
    if (
        re.fullmatch(r"sha256:[0-9a-f]{64}", image) is None
        or image != expected_image
        or any(len(seen.get(name, ())) != 1 for name in fixed_options)
        or seen.get("--name") != [container_name]
        or seen.get("--cidfile") != [str(cidfile)]
        or seen.get("--label") != [expected_label]
        or seen.get("--user") != [f"{os.getuid()}:{os.getgid()}"]
        or seen.get("--workdir") != [str(resolved_cwd)]
        or (
            provider == "codex"
            and any(len(seen.get(name, ())) != 1 for name in codex_fixed_options)
        )
        or (
            provider == "grok"
            and any(len(seen.get(name, ())) != 1 for name in grok_fixed_options)
        )
    ):
        raise ValueError("Docker create command grammar is not canonical")
    expected_tmpfs = {
        (
            "/tmp:rw,nosuid,nodev,noexec,mode=0700,"
            f"uid={os.getuid()},gid={os.getgid()}"
        ),
        (
            "/var/tmp:rw,nosuid,nodev,noexec,mode=0700,"
            f"uid={os.getuid()},gid={os.getgid()}"
        ),
    }
    if provider == "codex":
        expected_tmpfs.add(
            (
                f"{_CODEX_CONTAINER_HOME}:rw,nosuid,nodev,noexec,mode=0700,"
                f"uid={os.getuid()},gid={os.getgid()}"
            )
        )
    if sorted(seen.get("--tmpfs", ())) != sorted(expected_tmpfs):
        raise ValueError("Docker create tmpfs is outside provider policy")
    if provider == "grok":
        expected_prefix = [
            "-c",
            _DOCKER_PROVIDER_START_SCRIPT,
            "aseh-provider-start",
        ]
    else:
        expected_prefix = [
            "-i",
            *[
                f"{name}={value}"
                for name, value in sorted(
                    _codex_task_container_environment().items()
                )
            ],
            "/bin/sh",
            "-c",
            _DOCKER_PROVIDER_START_SCRIPT,
            "aseh-provider-start",
        ]
    if (
        container_argv[: len(expected_prefix)] != expected_prefix
        or len(container_argv) <= len(expected_prefix)
    ):
        raise ValueError("Docker provider start command is not canonical")

    denied_environment_fragments = (
        "TOKEN",
        "SECRET",
        "PASSWORD",
        "CREDENTIAL",
        "STATE_OWNER",
        "GRANT_BROKER",
        "QUACK",
        "DOCKER_",
        "CONTAINER_",
        "PODMAN_",
        "BUILDAH_",
    )
    for assignment in seen.get("--env", ()):
        name = assignment.partition("=")[0]
        if (
            re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None
            or any(fragment in name.upper() for fragment in denied_environment_fragments)
        ):
            raise ValueError("Docker create environment projection is unsafe")

    lease_root = docker_config.parent
    cleanup_root = lease_root.parent
    _docker_cleanup_root_identity(cleanup_root)
    allowed_git_sources = set(_git_metadata_roots(resolved_cwd))
    git_control_path = _existing_path(resolved_cwd / ".git")
    if git_control_path is not None:
        allowed_git_sources.add(git_control_path)
    allowed_codex_vendor_mounts: set[str] = set()
    if provider == "codex":
        try:
            vendor_arguments = _docker_codex_host_vendor_mounts()
            if len(vendor_arguments) % 2:
                raise ValueError("Codex vendor mount arguments are incomplete")
            allowed_codex_vendor_mounts = {
                vendor_arguments[offset + 1]
                for offset in range(0, len(vendor_arguments), 2)
                if vendor_arguments[offset] == "--mount"
            }
            if len(allowed_codex_vendor_mounts) * 2 != len(vendor_arguments):
                raise ValueError("Codex vendor mount arguments are noncanonical")
        except (ImportError, OSError, RuntimeError, ValueError):
            # An unavailable vendor authority means no vendor mount is
            # admissible.  The ordinary image-contained Codex path remains
            # valid; a command carrying an unverified host mount fails below.
            allowed_codex_vendor_mounts = set()
    for mount in seen.get("--mount", ()):
        fields = mount.split(",")
        if len(fields) not in {3, 4} or fields[0] != "type=bind":
            raise ValueError("Docker create mount grammar is invalid")
        pairs: dict[str, str] = {}
        readonly = False
        for field in fields[1:]:
            if field == "readonly":
                if readonly:
                    raise ValueError("Docker create mount repeats readonly")
                readonly = True
                continue
            key, separator, value = field.partition("=")
            if key not in {"src", "dst"} or not separator or key in pairs:
                raise ValueError("Docker create mount fields are invalid")
            pairs[key] = value
        if set(pairs) != {"src", "dst"}:
            raise ValueError("Docker create mount is incomplete")
        source = Path(pairs["src"])
        destination = Path(pairs["dst"])
        try:
            resolved_source = source.resolve(strict=True)
        except OSError as exc:
            raise ValueError("Docker create mount source is unavailable") from exc
        if (
            not source.is_absolute()
            or resolved_source != source.absolute()
            or not destination.is_absolute()
            or destination
            in {
                Path("/var/run/docker.sock"),
                Path("/run/docker.sock"),
            }
        ):
            raise ValueError("Docker create mount path is unsafe")
        writable_workspace = bool(
            resolved_source == resolved_cwd and destination == resolved_cwd
        )
        writable_grok_home = bool(
            provider == "grok"
            and resolved_source.parent == cleanup_root
            and resolved_source.name.startswith("asref-grok-home-")
            and destination == resolved_source
        )
        writable_codex_auth = bool(
            provider == "codex"
            and resolved_source.parent.parent == cleanup_root
            and resolved_source.parent.name.startswith("asref-codex-home-")
            and resolved_source.name == "auth.json"
            and destination == _CODEX_CONTAINER_AUTH_PATH
        )
        if not readonly and not (
            writable_workspace or writable_grok_home or writable_codex_auth
        ):
            raise ValueError("Docker create writable mount is outside policy")
        if readonly:
            readonly_source_allowed = bool(
                (
                    resolved_source == Path("/usr")
                    and destination == Path("/usr")
                )
                or (
                    provider == "codex"
                    and resolved_source == Path("/etc/ssl/certs")
                    and destination == Path("/etc/ssl/certs")
                )
                or (
                    provider == "codex"
                    and resolved_source == _HOST_CODEX_TASK_TOOLCHAIN_PYTHON
                    and destination == _CODEX_TASK_TOOLCHAIN_PYTHON
                )
                or (
                    provider == "codex"
                    and mount in allowed_codex_vendor_mounts
                )
                or (
                    destination == resolved_source
                    and resolved_source in allowed_git_sources
                )
                or (
                    resolved_source.parent == cleanup_root
                    and resolved_source.name.startswith("asref-grok-prompt-")
                    and destination == resolved_source
                )
                or resolved_source.is_relative_to(
                    lease_root / "provider-masks"
                )
                or (
                    provider == "grok"
                    and destination == Path("/opt/ipfs-accelerate/grok")
                    and _resolve_trusted_grok_bin(
                        configured=str(resolved_source),
                        workspace=resolved_cwd,
                    )
                    == str(resolved_source)
                )
            )
            if not readonly_source_allowed:
                raise ValueError("Docker create read-only mount is outside policy")

    body: dict[str, object] = {
        "provider": provider,
        "docker_bin": docker_bin,
        "docker_config": str(docker_config),
        "container_name": container_name,
        "cidfile": str(cidfile),
        "cwd": str(resolved_cwd),
        "environment_id": environment_id,
        "image_id": image,
        "argv": values,
    }
    return _effect_receipt_identity(body), body


def _docker_create_private_handoff_payload(
    *,
    command_id: str,
    command_body: Mapping[str, object],
    environment: Mapping[str, str],
) -> bytes:
    """Bind the watchdog's executable tuple to its private runner pipe."""

    body: dict[str, object] = {
        "schema": _DOCKER_CREATE_HANDOFF_SCHEMA,
        "command_id": command_id,
        "command_body": dict(command_body),
        "environment": dict(sorted(environment.items())),
    }
    body["handoff_id"] = _effect_receipt_identity(body)
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    if not encoded or len(encoded) > _DOCKER_CREATE_HANDOFF_MAX_BYTES:
        raise ValueError("Docker create private handoff is oversized")
    return encoded


def _docker_create_private_result_payload(
    journal: Mapping[str, object],
) -> bytes:
    """Encode one watchdog-observed terminal result for its private pipe."""

    state = str(journal.get("state") or "")
    if state not in {
        "create_observed",
        "create_failed_observed",
        "create_outcome_unknown",
    }:
        raise ValueError("Docker create private result is not terminal")
    body: dict[str, object] = {
        "schema": _DOCKER_CREATE_RESULT_SCHEMA,
        "journal": dict(journal),
    }
    body["result_id"] = _effect_receipt_identity(body)
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    if not encoded or len(encoded) > _DOCKER_CREATE_RESULT_MAX_BYTES:
        raise ValueError("Docker create private result is oversized")
    return encoded


def _write_docker_create_private_result(
    channel: socket.socket,
    journal: Mapping[str, object],
) -> None:
    payload = _docker_create_private_result_payload(journal)
    message = b"R" + len(payload).to_bytes(8, "big") + payload
    channel.sendall(message)


def _docker_control_read_exact(
    channel: socket.socket,
    size: int,
    *,
    deadline: float | None = None,
) -> bytes:
    payload = bytearray()
    while len(payload) < size:
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ValueError("Docker private control read timed out")
            readable, _writable, _exceptional = select.select(
                [channel],
                [],
                [],
                remaining,
            )
            if not readable:
                raise ValueError("Docker private control read timed out")
        chunk = channel.recv(size - len(payload))
        if not chunk:
            raise ValueError("Docker private control message is incomplete")
        payload.extend(chunk)
    return bytes(payload)


def _read_docker_create_private_result(
    channel: socket.socket,
    *,
    deadline: float,
) -> dict[str, object]:
    """Read one bounded canonical watchdog result from its private socket."""

    if _docker_control_read_exact(channel, 1, deadline=deadline) != b"R":
        raise ValueError("Docker create private result marker is invalid")
    size = int.from_bytes(
        _docker_control_read_exact(channel, 8, deadline=deadline),
        "big",
    )
    if not 0 < size <= _DOCKER_CREATE_RESULT_MAX_BYTES:
        raise ValueError("Docker create private result length is invalid")
    raw = _docker_control_read_exact(channel, size, deadline=deadline)
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_control_keys,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Docker create private result is invalid JSON") from exc
    if (
        type(value) is not dict
        or set(value) != {"schema", "journal", "result_id"}
        or value.get("schema") != _DOCKER_CREATE_RESULT_SCHEMA
        or not isinstance(value.get("journal"), dict)
        or value.get("result_id")
        != _effect_receipt_identity(
            {key: item for key, item in value.items() if key != "result_id"}
        )
        or raw != _docker_create_private_result_payload(value["journal"])
    ):
        raise ValueError("Docker create private result is noncanonical")
    return dict(value["journal"])


def _docker_create_environment_payload(
    environment: Mapping[str, str],
) -> tuple[str, bytes, dict[str, str]]:
    """Seal the exact Docker CLI environment for an in-memory handoff."""

    values = dict(environment)
    if (
        not values
        or any(
            re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None
            or not isinstance(value, str)
            or "\x00" in value
            or name.upper().startswith(
                ("DOCKER_", "CONTAINER_", "PODMAN_", "BUILDAH_")
            )
            for name, value in values.items()
        )
        or not values.get("PATH")
        or not values.get("HOME")
    ):
        raise ValueError("Docker create environment is not sanitized")
    encoded = json.dumps(
        dict(sorted(values.items())),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    if len(encoded) > _DOCKER_CREATE_ENVIRONMENT_MAX_BYTES:
        raise ValueError("Docker create environment is oversized")
    return _effect_receipt_identity(values), encoded, values


def _docker_create_journal_value(
    *,
    command_body: Mapping[str, object],
    command_id: str,
    state: str,
    issuer_process_birth: Mapping[str, object] | None = None,
    returncode: int | None = None,
    stdout: bytes = b"",
    stderr: bytes = b"",
) -> dict[str, object]:
    if state not in {
        "prepared",
        "create_armed",
        "create_inflight",
        "create_observed",
        "create_failed_observed",
        "create_outcome_unknown",
        "prepared_abandoned",
    }:
        raise ValueError("Docker create journal state is invalid")
    if state == "create_observed" and (
        type(returncode) is not int or returncode != 0
    ):
        raise ValueError("observed Docker create must have return code zero")
    if state == "create_outcome_unknown" and (
        isinstance(returncode, bool)
        or not isinstance(returncode, int)
        or returncode == 0
    ):
        raise ValueError("unknown Docker create must have a nonzero result")
    if state == "create_failed_observed" and (
        type(returncode) is not int or returncode == 0
    ):
        raise ValueError("failed Docker create must have a nonzero result")
    if state not in {
        "create_observed",
        "create_failed_observed",
        "create_outcome_unknown",
    } and (
        returncode is not None or stdout or stderr
    ):
        raise ValueError("nonterminal Docker create journal has output")
    issuer_required = state in {
        "create_inflight",
        "create_observed",
        "create_failed_observed",
        "create_outcome_unknown",
    }
    issuer = dict(issuer_process_birth or {})
    valid_issuer = bool(
        set(issuer) == {"pid", "start_time_ticks", "boot_id", "parent_pid"}
        and type(issuer.get("pid")) is int
        and issuer["pid"] > 0
        and type(issuer.get("start_time_ticks")) is int
        and issuer["start_time_ticks"] > 0
        and type(issuer.get("parent_pid")) is int
        and issuer["parent_pid"] > 0
        and isinstance(issuer.get("boot_id"), str)
        and re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            str(issuer.get("boot_id") or ""),
        )
        is not None
    )
    if issuer_required != valid_issuer:
        raise ValueError("Docker create issuer identity is invalid")
    if not issuer_required and issuer:
        raise ValueError("pre-dispatch Docker journal has an issuer")
    if (
        len(stdout) > _DOCKER_INSPECTION_MAX_BYTES
        or len(stderr) > _DOCKER_INSPECTION_MAX_BYTES
    ):
        raise ValueError("Docker create result is oversized")
    value: dict[str, object] = {
        "schema": _DOCKER_CREATE_JOURNAL_SCHEMA,
        **dict(command_body),
        "command_id": command_id,
        "state": state,
        "issuer_process_birth": issuer,
        "returncode": returncode,
        "stdout_hex": stdout.hex(),
        "stderr_hex": stderr.hex(),
    }
    value["journal_id"] = _effect_receipt_identity(value)
    return value


def _validated_docker_create_journal(
    *,
    lease_root: Path,
    provider: str,
    docker_bin: str,
    docker_config: Path,
    container_name: str,
    cidfile: Path,
) -> dict[str, object] | None:
    value = _read_private_control_record(
        lease_root,
        _DOCKER_CREATE_JOURNAL_NAME,
    )
    if value is None:
        return None
    expected_fields = {
        "schema",
        "provider",
        "docker_bin",
        "docker_config",
        "container_name",
        "cidfile",
        "cwd",
        "environment_id",
        "image_id",
        "argv",
        "command_id",
        "state",
        "issuer_process_birth",
        "returncode",
        "stdout_hex",
        "stderr_hex",
        "journal_id",
    }
    if set(value) != expected_fields or value.get("schema") != (
        _DOCKER_CREATE_JOURNAL_SCHEMA
    ):
        raise ValueError("Docker create journal shape is invalid")
    argv = value.get("argv")
    if not isinstance(argv, list) or not all(
        isinstance(item, str) for item in argv
    ):
        raise ValueError("Docker create journal argv is invalid")
    command_id, command_body = _docker_create_command_identity(
        provider=provider,
        docker_bin=docker_bin,
        docker_config=docker_config,
        container_name=container_name,
        cidfile=cidfile,
        cwd=Path(str(value.get("cwd") or "")),
        environment_id=str(value.get("environment_id") or ""),
        expected_image=str(value.get("image_id") or ""),
        argv=argv,
    )
    try:
        stdout = bytes.fromhex(str(value.get("stdout_hex") or ""))
        stderr = bytes.fromhex(str(value.get("stderr_hex") or ""))
    except ValueError as exc:
        raise ValueError("Docker create journal output is invalid") from exc
    expected = _docker_create_journal_value(
        command_body=command_body,
        command_id=command_id,
        state=str(value.get("state") or ""),
        issuer_process_birth=value.get("issuer_process_birth"),  # type: ignore[arg-type]
        returncode=value.get("returncode"),  # type: ignore[arg-type]
        stdout=stdout,
        stderr=stderr,
    )
    if (
        value != expected
        or value.get("command_id") != command_id
        or value.get("journal_id") != expected["journal_id"]
    ):
        raise ValueError("Docker create journal identity drifted")
    return value


def _transition_docker_create_journal(
    journal: Mapping[str, object],
    *,
    lease_root: Path,
    state: str,
    issuer_process_birth: Mapping[str, object] | None = None,
    returncode: int | None = None,
    stdout: bytes = b"",
    stderr: bytes = b"",
) -> dict[str, object]:
    command_body = {
        name: journal[name]
        for name in (
            "provider",
            "docker_bin",
            "docker_config",
            "container_name",
            "cidfile",
            "cwd",
            "environment_id",
            "image_id",
            "argv",
        )
    }
    value = _docker_create_journal_value(
        command_body=command_body,
        command_id=str(journal.get("command_id") or ""),
        state=state,
        issuer_process_birth=(
            issuer_process_birth
            if issuer_process_birth is not None
            else journal.get("issuer_process_birth")  # type: ignore[arg-type]
        ),
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )
    _write_private_control_record(
        lease_root,
        _DOCKER_CREATE_JOURNAL_NAME,
        value,
        replace_existing=True,
    )
    return value


def _run_fenced_docker_create_issuer(
    journal: Mapping[str, object],
    *,
    lease_root: Path,
    cwd: Path,
    environment: Mapping[str, str],
) -> tuple[dict[str, object], int, bytes, bytes, bool, bool]:
    """Run one gated Docker-create issuer whose exact birth is durable.

    The forked child cannot cross the Docker ``exec`` boundary until its PID,
    start ticks, and boot identity have been fsynced in ``create_inflight``.
    If the watchdog dies, EOF closes the pre-dispatch gate or recovery can
    prove the exact post-dispatch issuer birth is gone before settling the
    external Docker name.  The create command is never replayed.
    """

    # Ordinary pipes are reopenable through /proc/<pid>/fd by a same-UID
    # process and therefore cannot carry either the post-fsync dispatch gate
    # or observed provider output.  Unnamed Unix socketpairs are inherited
    # capabilities whose descriptors Linux refuses to reopen through procfs.
    gate_parent, gate_child = socket.socketpair(
        socket.AF_UNIX,
        socket.SOCK_STREAM,
    )
    stdout_parent, stdout_child = socket.socketpair(
        socket.AF_UNIX,
        socket.SOCK_STREAM,
    )
    stderr_parent, stderr_child = socket.socketpair(
        socket.AF_UNIX,
        socket.SOCK_STREAM,
    )
    child_pid = -1
    try:
        child_pid = os.fork()
    except OSError:
        for channel in (
            gate_parent,
            gate_child,
            stdout_parent,
            stdout_child,
            stderr_parent,
            stderr_child,
        ):
            channel.close()
        raise
    if child_pid == 0:
        try:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            gate_parent.close()
            stdout_parent.close()
            stderr_parent.close()
            marker = gate_child.recv(1)
            gate_child.close()
            if marker != b"D":
                os._exit(125)
            null_fd = os.open(os.devnull, os.O_RDONLY)
            os.dup2(null_fd, 0)
            os.dup2(stdout_child.fileno(), 1)
            os.dup2(stderr_child.fileno(), 2)
            for descriptor in (
                null_fd,
                stdout_child.fileno(),
                stderr_child.fileno(),
            ):
                if descriptor > 2:
                    os.close(descriptor)
            os.chdir(cwd)
            argv = [str(item) for item in journal["argv"]]  # type: ignore[index]
            os.execve(argv[0], argv, dict(environment))
        except BaseException:
            os._exit(125)

    gate_child.close()
    stdout_child.close()
    stderr_child.close()
    issuer_exited = False
    forced_kill = False
    dispatched = False
    status: int | None = None
    try:
        issuer_birth = read_process_birth(child_pid)
        if (
            issuer_birth is None
            or issuer_birth.parent_pid != os.getpid()
            or not issuer_birth.boot_id
        ):
            raise ValueError("Docker create issuer birth is unavailable")
        issuer_start_ticks = issuer_birth.start_time_ticks
        journal = _transition_docker_create_journal(
            journal,
            lease_root=lease_root,
            state="create_inflight",
            issuer_process_birth=issuer_birth.to_dict(),
        )
        try:
            gate_parent.sendall(b"D")
            dispatched = True
        except OSError:
            dispatched = False
        finally:
            gate_parent.close()

        for channel in (stdout_parent, stderr_parent):
            channel.setblocking(False)
        streams = {
            stdout_parent: bytearray(),
            stderr_parent: bytearray(),
        }
        open_streams = set(streams)
        deadline = time.monotonic() + _DOCKER_CREATE_TIMEOUT_SECONDS
        while open_streams or status is None:
            now = time.monotonic()
            if status is None:
                waited_pid, observed_status = os.waitpid(child_pid, os.WNOHANG)
                if waited_pid == child_pid:
                    status = observed_status
                    issuer_exited = True
            if status is None and now >= deadline:
                # This PID is our unreaped direct child, so it cannot be
                # recycled.  Do not collapse an unreadable /proc record into
                # "gone" and then block forever in waitpid.
                forced_kill = True
                try:
                    os.kill(child_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                kill_deadline = time.monotonic() + 2.0
                while status is None and time.monotonic() < kill_deadline:
                    waited_pid, observed_status = os.waitpid(
                        child_pid,
                        os.WNOHANG,
                    )
                    if waited_pid == child_pid:
                        status = observed_status
                        issuer_exited = True
                        break
                    time.sleep(0.02)
                if status is None:
                    raise ValueError("Docker create issuer could not be reaped")
            if status is not None and now >= deadline and open_streams:
                open_streams.clear()
            if open_streams:
                readable, _writable, _exceptional = select.select(
                    tuple(open_streams),
                    (),
                    (),
                    0.05,
                )
                for channel in readable:
                    try:
                        chunk = channel.recv(64 * 1024)
                    except BlockingIOError:
                        continue
                    if not chunk:
                        open_streams.remove(channel)
                        continue
                    buffer = streams[channel]
                    if len(buffer) <= _DOCKER_INSPECTION_MAX_BYTES:
                        buffer.extend(chunk)
            elif status is None:
                time.sleep(0.02)
        stdout = bytes(streams[stdout_parent])
        stderr = bytes(streams[stderr_parent])
        returncode = (
            os.waitstatus_to_exitcode(status)
            if status is not None
            else 125
        )
        if not dispatched and returncode == 0:
            returncode = 125
        return (
            dict(journal),
            returncode,
            stdout,
            stderr,
            dispatched,
            forced_kill,
        )
    finally:
        for channel in (gate_parent, stdout_parent, stderr_parent):
            try:
                channel.close()
            except OSError:
                pass
        if child_pid > 0 and not issuer_exited:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
            reap_deadline = time.monotonic() + 2.0
            while time.monotonic() < reap_deadline:
                try:
                    waited_pid, _status = os.waitpid(child_pid, os.WNOHANG)
                except (ChildProcessError, OSError):
                    break
                if waited_pid == child_pid:
                    break
                time.sleep(0.02)


def _validated_cleanup_binding_record(
    record_path: Path,
    *,
    provider: str,
    docker_bin: str,
    docker_config: Path,
    container_name: str,
    cidfile: Path,
    lease_root: Path,
    provider_home: Path,
    prompt_path: Path,
    effect_observation: Mapping[str, str],
    binding_state: str,
    runner_pid: int,
    runner_start_ticks: int,
    watchdog_pid: int,
    watchdog_start_ticks: int,
    create_command_id: str = "",
    create_cwd: Path | None = None,
    create_environment_id: str = "",
    termination_fence: Mapping[str, object] | None = None,
    control_directory_fd: int | None = None,
) -> dict[str, object]:
    value = _read_private_control_record(
        record_path.parent,
        record_path.name,
        directory_fd=control_directory_fd,
    )
    expected_fields = {
        "schema",
        "binding_state",
        "run_id",
        "profile_id",
        "target_id",
        "repository_root",
        "state_root",
        "run_root",
        "configuration_root",
        "fencing_epoch",
        "runner_pid",
        "runner_start_ticks",
        "watchdog_pid",
        "watchdog_start_ticks",
        "boot_id",
        "provider",
        "docker_bin",
        "docker_device",
        "docker_inode",
        "docker_mode",
        "docker_uid",
        "container_name",
        "cleanup_root",
        "cleanup_root_identity",
        "lease_root",
        "docker_config",
        "cidfile",
        "provider_home",
        "prompt_path",
        "effect_observation",
        "create_command_id",
        "create_cwd",
        "create_environment_id",
        "termination_fence",
        "path_identities",
        "binding_path",
        "record_id",
    }
    if value is None or set(value) != expected_fields:
        raise ValueError("Docker cleanup binding record shape is invalid")
    body = {key: item for key, item in value.items() if key != "record_id"}
    lifecycle = {
        RUN_ID_ENV: "run_id",
        PROFILE_ID_ENV: "profile_id",
        TARGET_ID_ENV: "target_id",
        REPOSITORY_ROOT_ENV: "repository_root",
        STATE_ROOT_ENV: "state_root",
        RUN_ROOT_ENV: "run_root",
        CONFIGURATION_ROOT_ENV: "configuration_root",
    }
    try:
        fencing_epoch = int(os.environ[FENCING_EPOCH_ENV])
        observed_runner_pid = int(value["runner_pid"])
        observed_runner_start_ticks = int(value["runner_start_ticks"])
        docker_metadata = Path(docker_bin).stat()
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="ascii"
        ).strip()
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise ValueError("Docker cleanup binding identity is unavailable") from exc
    command_bound = binding_state == "command_bound"
    expected_termination_fence = dict(termination_fence or {})
    if (
        binding_state not in {"prepared_no_dispatch", "command_bound"}
        or runner_pid <= 0
        or runner_start_ticks <= 0
        or watchdog_pid <= 0
        or watchdog_start_ticks <= 0
        or command_bound
        != bool(
            re.fullmatch(r"sha256:[0-9a-f]{64}", create_command_id)
            and create_cwd is not None
            and create_cwd.is_absolute()
            and re.fullmatch(r"sha256:[0-9a-f]{64}", create_environment_id)
        )
    ):
        raise ValueError("Docker cleanup binding expectation is invalid")
    if expected_termination_fence:
        if not command_bound:
            raise ValueError(
                "Docker termination fence requires command-bound cleanup"
            )
        _validated_docker_termination_fence(
            expected_termination_fence,
            provider=provider,
            container_name=container_name,
        )
    try:
        cleanup_root, cleanup_root_identity = _validated_docker_cleanup_root(
            lease_root=lease_root,
            provider_home=provider_home,
            prompt_path=prompt_path,
            expected_root=Path(str(value.get("cleanup_root") or "")),
            expected_identity=value.get("cleanup_root_identity"),  # type: ignore[arg-type]
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Docker cleanup binding root is invalid") from exc
    if (
        value.get("schema") != _DOCKER_CLEANUP_BINDING_SCHEMA
        or value.get("binding_state") != binding_state
        or value.get("record_id") != _effect_receipt_identity(body)
        or any(value.get(field) != os.environ.get(name) for name, field in lifecycle.items())
        or value.get("fencing_epoch") != fencing_epoch
        or value.get("watchdog_pid") != watchdog_pid
        or value.get("watchdog_start_ticks") != watchdog_start_ticks
        or value.get("boot_id") != boot_id
        or observed_runner_pid != runner_pid
        or observed_runner_start_ticks != runner_start_ticks
        or value.get("provider") != provider
        or value.get("docker_bin") != docker_bin
        or value.get("docker_device") != docker_metadata.st_dev
        or value.get("docker_inode") != docker_metadata.st_ino
        or value.get("docker_mode") != docker_metadata.st_mode
        or value.get("docker_uid") != docker_metadata.st_uid
        or value.get("container_name") != container_name
        or value.get("cleanup_root") != str(cleanup_root)
        or value.get("cleanup_root_identity") != cleanup_root_identity
        or value.get("lease_root") != str(lease_root)
        or value.get("docker_config") != str(docker_config)
        or value.get("cidfile") != str(cidfile)
        or value.get("provider_home") != str(provider_home)
        or value.get("prompt_path") != str(prompt_path)
        or value.get("effect_observation") != dict(sorted(effect_observation.items()))
        or value.get("create_command_id")
        != (create_command_id if command_bound else "")
        or value.get("create_cwd")
        != (str(create_cwd) if command_bound else "")
        or value.get("create_environment_id")
        != (create_environment_id if command_bound else "")
        or value.get("termination_fence") != expected_termination_fence
        or value.get("path_identities")
        != {
            "docker_config": _cleanup_path_identity(
                docker_config,
                directory=True,
            ),
            "lease_root": _cleanup_path_identity(
                lease_root,
                directory=True,
            ),
            "prompt_path": _cleanup_path_identity(
                prompt_path,
                directory=False,
            ),
            "provider_home": _cleanup_path_identity(
                provider_home,
                directory=True,
            ),
        }
        or value.get("binding_path") != str(record_path)
        or record_path.parent
        != Path(str(value.get("run_root")))
        / _DOCKER_CLEANUP_BINDING_DIRECTORY
    ):
        raise ValueError("Docker cleanup binding record identity drifted")
    return value


class _DockerBindingLock:
    """Composite per-binding lock tied to one exact private directory."""

    def __init__(
        self,
        *,
        binding_path: Path,
        directory_fd: int,
        descriptor: int,
        uniqueness_socket: socket.socket,
    ) -> None:
        self.binding_path = binding_path.absolute()
        self.directory_fd = directory_fd
        self.descriptor = descriptor
        self.uniqueness_socket = uniqueness_socket
        self._closed = False

    def assert_current(self) -> None:
        if self._closed:
            raise ValueError("Docker cleanup binding lock is closed")
        _validate_private_control_directory_descriptor(
            self.binding_path.parent,
            self.directory_fd,
        )

    def read(self, path: Path) -> dict[str, object] | None:
        if path.parent.absolute() != self.binding_path.parent:
            raise ValueError("Docker control read escaped binding directory")
        return _read_private_control_record(
            path.parent,
            path.name,
            directory_fd=self.directory_fd,
        )

    def write(
        self,
        path: Path,
        value: Mapping[str, object],
        *,
        replace_existing: bool,
    ) -> None:
        if path.parent.absolute() != self.binding_path.parent:
            raise ValueError("Docker control write escaped binding directory")
        _write_private_control_record(
            path.parent,
            path.name,
            value,
            replace_existing=replace_existing,
            directory_fd=self.directory_fd,
        )

    def path_identity(self, path: Path, *, directory: bool = False) -> dict[str, int]:
        if path.parent.absolute() != self.binding_path.parent:
            raise ValueError("Docker control identity escaped binding directory")
        self.assert_current()
        try:
            metadata = os.stat(
                path.name,
                dir_fd=self.directory_fd,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise ValueError("Docker cleanup path identity is unavailable") from exc
        if (
            metadata.st_uid != os.geteuid()
            or stat.S_ISLNK(metadata.st_mode)
            or (directory and not stat.S_ISDIR(metadata.st_mode))
            or (not directory and not stat.S_ISREG(metadata.st_mode))
        ):
            raise ValueError("Docker cleanup path identity is unsafe")
        self.assert_current()
        return {
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
            "mode": stat.S_IFMT(metadata.st_mode),
            "uid": metadata.st_uid,
        }

    def exists(self, path: Path) -> bool:
        if path.parent.absolute() != self.binding_path.parent:
            raise ValueError("Docker control lookup escaped binding directory")
        self.assert_current()
        try:
            os.stat(
                path.name,
                dir_fd=self.directory_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            self.assert_current()
            return False
        self.assert_current()
        return True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            fcntl.flock(self.descriptor, fcntl.LOCK_UN)
        finally:
            try:
                os.close(self.descriptor)
            finally:
                try:
                    os.close(self.directory_fd)
                finally:
                    self.uniqueness_socket.close()


def _docker_binding_uniqueness_socket(
    binding_path: Path,
    *,
    deadline: float,
) -> socket.socket:
    """Acquire one kernel-resident lease independent of the lock inode."""

    normalized = os.path.normpath(os.path.abspath(os.fspath(binding_path)))
    identity = hashlib.sha256(
        f"{os.geteuid()}\0{normalized}".encode("utf-8")
    ).hexdigest()
    address = b"\0ipfs-accelerate-docker-binding-" + identity.encode("ascii")
    while True:
        channel = socket.socket(
            socket.AF_UNIX,
            socket.SOCK_DGRAM | getattr(socket, "SOCK_CLOEXEC", 0),
        )
        try:
            channel.set_inheritable(False)
            channel.bind(address)
            return channel
        except OSError as exc:
            channel.close()
            remaining = deadline - time.monotonic()
            if exc.errno != errno.EADDRINUSE or remaining <= 0:
                if exc.errno == errno.EADDRINUSE:
                    raise ValueError(
                        "Docker cleanup binding lock is contended"
                    ) from None
                raise ValueError(
                    "Docker cleanup binding uniqueness lease is unavailable"
                ) from exc
            time.sleep(min(0.01, remaining))


def _docker_binding_lock_descriptor(
    binding_path: Path,
    *,
    deadline: float | None = None,
) -> _DockerBindingLock:
    """Boundedly lock one stable binding name without global contention.

    The lock file is permanent.  After acquiring its inode, re-resolve the
    name relative to the already-admitted private directory so a renamed or
    replaced lock can never split the per-binding exclusion domain.
    """

    if re.fullmatch(r"[0-9a-f]{64}\.json", binding_path.name) is None:
        raise ValueError("Docker cleanup binding name is invalid")
    directory_fd = _private_control_directory(binding_path.parent)
    lock_name = binding_path.with_suffix(".lock").name
    descriptor = -1
    uniqueness_socket: socket.socket | None = None
    try:
        descriptor = os.open(
            lock_name,
            os.O_RDWR
            | os.O_CREAT
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_fd,
        )
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            raise ValueError("Docker cleanup binding lock is unsafe")
        os.fsync(directory_fd)
        lock_deadline = (
            time.monotonic() + _DOCKER_BINDING_LOCK_TIMEOUT_SECONDS
            if deadline is None
            else float(deadline)
        )
        while True:
            try:
                fcntl.flock(
                    descriptor,
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
                break
            except BlockingIOError:
                remaining = lock_deadline - time.monotonic()
                if remaining <= 0:
                    raise ValueError(
                        "Docker cleanup binding lock is contended"
                    ) from None
                time.sleep(min(0.01, remaining))
        uniqueness_socket = _docker_binding_uniqueness_socket(
            binding_path,
            deadline=lock_deadline,
        )
        named_metadata = os.stat(
            lock_name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        locked_metadata = os.fstat(descriptor)
        if (
            named_metadata.st_dev != locked_metadata.st_dev
            or named_metadata.st_ino != locked_metadata.st_ino
            or named_metadata.st_mode != locked_metadata.st_mode
            or named_metadata.st_uid != locked_metadata.st_uid
            or named_metadata.st_nlink != locked_metadata.st_nlink
        ):
            raise ValueError("Docker cleanup binding lock name changed")
        _validate_private_control_directory_descriptor(
            binding_path.parent,
            directory_fd,
        )
        return _DockerBindingLock(
            binding_path=binding_path,
            directory_fd=directory_fd,
            descriptor=descriptor,
            uniqueness_socket=uniqueness_socket,
        )
    except BaseException:
        if uniqueness_socket is not None:
            uniqueness_socket.close()
        if descriptor >= 0:
            os.close(descriptor)
        os.close(directory_fd)
        raise


def _publish_docker_termination_fence_binding(
    *,
    record_path: Path,
    expected_record_id: str,
    expected_identity: Mapping[str, int],
    provider: str,
    docker_bin: str,
    docker_config: Path,
    container_name: str,
    cidfile: Path,
    lease_root: Path,
    provider_home: Path,
    prompt_path: Path,
    effect_observation: Mapping[str, str],
    runner_pid: int,
    runner_start_ticks: int,
    watchdog_pid: int,
    watchdog_start_ticks: int,
    create_command_id: str,
    create_cwd: Path,
    create_environment_id: str,
    termination_fence: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, int]]:
    """CAS-upgrade the one canonical cleanup binding before Docker rm."""

    fence = _validated_docker_termination_fence(
        termination_fence,
        provider=provider,
        container_name=container_name,
    )
    lock_handle = _docker_binding_lock_descriptor(record_path)
    try:
        raw = lock_handle.read(record_path)
        if raw is None or not isinstance(raw.get("termination_fence"), Mapping):
            raise ValueError("Docker cleanup binding disappeared")
        current_fence = dict(raw["termination_fence"])
        current = _validated_cleanup_binding_record(
            record_path,
            provider=provider,
            docker_bin=docker_bin,
            docker_config=docker_config,
            container_name=container_name,
            cidfile=cidfile,
            lease_root=lease_root,
            provider_home=provider_home,
            prompt_path=prompt_path,
            effect_observation=effect_observation,
            binding_state="command_bound",
            runner_pid=runner_pid,
            runner_start_ticks=runner_start_ticks,
            watchdog_pid=watchdog_pid,
            watchdog_start_ticks=watchdog_start_ticks,
            create_command_id=create_command_id,
            create_cwd=create_cwd,
            create_environment_id=create_environment_id,
            termination_fence=current_fence,
            control_directory_fd=lock_handle.directory_fd,
        )
        current_identity = lock_handle.path_identity(record_path)
        if current_fence:
            if current_fence != fence:
                raise ValueError("Docker termination fence changed")
            # Another exact publisher may have won while this caller waited
            # for the stable lock.  Coalesce only the deterministic successor
            # of the caller's admitted unfenced record; never accept an
            # unrelated already-fenced binding merely because its fields look
            # plausible.
            predecessor_body = {
                name: item
                for name, item in current.items()
                if name != "record_id"
            }
            predecessor_body["termination_fence"] = {}
            if _effect_receipt_identity(predecessor_body) != expected_record_id:
                raise ValueError("Docker termination fence predecessor changed")
            return current, current_identity
        if (
            current.get("record_id") != expected_record_id
            or current_identity != dict(expected_identity)
        ):
            raise ValueError("Docker cleanup binding lost its CAS identity")
        fenced = _docker_cleanup_binding_value(
            binding_state="command_bound",
            provider=provider,
            docker_bin=docker_bin,
            container_name=container_name,
            lease_root=lease_root,
            docker_config=docker_config,
            cidfile=cidfile,
            provider_home=provider_home,
            prompt_path=prompt_path,
            effect_observation=effect_observation,
            path_identities=current["path_identities"],  # type: ignore[arg-type]
            binding_path=record_path,
            runner_pid=runner_pid,
            runner_start_ticks=runner_start_ticks,
            watchdog_pid=watchdog_pid,
            watchdog_start_ticks=watchdog_start_ticks,
            create_command_id=create_command_id,
            create_cwd=create_cwd,
            create_environment_id=create_environment_id,
            termination_fence=fence,
        )
        lock_handle.write(
            record_path,
            fenced,
            replace_existing=True,
        )
        admitted = _validated_cleanup_binding_record(
            record_path,
            provider=provider,
            docker_bin=docker_bin,
            docker_config=docker_config,
            container_name=container_name,
            cidfile=cidfile,
            lease_root=lease_root,
            provider_home=provider_home,
            prompt_path=prompt_path,
            effect_observation=effect_observation,
            binding_state="command_bound",
            runner_pid=runner_pid,
            runner_start_ticks=runner_start_ticks,
            watchdog_pid=watchdog_pid,
            watchdog_start_ticks=watchdog_start_ticks,
            create_command_id=create_command_id,
            create_cwd=create_cwd,
            create_environment_id=create_environment_id,
            termination_fence=fence,
            control_directory_fd=lock_handle.directory_fd,
        )
        return admitted, lock_handle.path_identity(record_path)
    finally:
        lock_handle.close()


def _docker_removal_dispatch_path(binding_path: Path) -> Path:
    if re.fullmatch(r"[0-9a-f]{64}\.json", binding_path.name) is None:
        raise ValueError("Docker cleanup binding name is invalid")
    return binding_path.with_suffix(".remove-dispatched")


def _docker_removal_dispatch_value(
    *,
    binding_path: Path,
    binding_record: Mapping[str, object],
    termination_fence: Mapping[str, object],
    issuer_process_birth: Mapping[str, object],
    state: str,
    generation: int,
    previous_dispatch_id: str,
    docker_returncode: int | None,
    failure_kind: str,
) -> dict[str, object]:
    provider = str(binding_record.get("provider") or "")
    container_name = str(binding_record.get("container_name") or "")
    fence = _validated_docker_termination_fence(
        termination_fence,
        provider=provider,
        container_name=container_name,
    )
    record_id = str(binding_record.get("record_id") or "")
    if (
        binding_record.get("binding_path") != str(binding_path)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", record_id) is None
        or binding_record.get("termination_fence") != fence
    ):
        raise ValueError("Docker removal dispatch authority is invalid")
    issuer = dict(issuer_process_birth)
    if (
        set(issuer) != {"pid", "start_time_ticks", "boot_id", "parent_pid"}
        or type(issuer.get("pid")) is not int
        or int(issuer["pid"]) <= 0
        or type(issuer.get("start_time_ticks")) is not int
        or int(issuer["start_time_ticks"]) <= 0
        or type(issuer.get("parent_pid")) is not int
        or int(issuer["parent_pid"]) <= 0
        or re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            str(issuer.get("boot_id") or ""),
        )
        is None
    ):
        raise ValueError("Docker removal issuer identity is invalid")
    if (
        isinstance(generation, bool)
        or not isinstance(generation, int)
        or generation < 1
        or state
        not in {
            "prepared",
            "request_started",
            "request_completed",
            "request_outcome_unknown",
        }
        or (
            generation == 1
            and state == "prepared"
            and previous_dispatch_id
        )
        or (
            (generation > 1 or state != "prepared")
            and re.fullmatch(
                r"sha256:[0-9a-f]{64}", previous_dispatch_id
            )
            is None
        )
        or (
            state in {"prepared", "request_started"}
            and (docker_returncode is not None or failure_kind)
        )
        or (
            state == "request_completed"
            and (
                isinstance(docker_returncode, bool)
                or not isinstance(docker_returncode, int)
                or failure_kind
            )
        )
        or (
            state == "request_outcome_unknown"
            and (
                docker_returncode is not None
                or failure_kind not in {"timeout", "os_error", "issuer_failure"}
            )
        )
    ):
        raise ValueError("Docker removal dispatch transition is invalid")
    body: dict[str, object] = {
        "schema": _DOCKER_REMOVAL_DISPATCH_SCHEMA,
        "binding_path": str(binding_path),
        "binding_record_id": record_id,
        "provider": provider,
        "container_name": container_name,
        "container_id": fence["container_id"],
        "fence_id": fence["fence_id"],
        "issuer_process_birth": issuer,
        "state": state,
        "generation": generation,
        "previous_dispatch_id": previous_dispatch_id,
        "docker_returncode": docker_returncode,
        "failure_kind": failure_kind,
    }
    body["dispatch_id"] = _effect_receipt_identity(body)
    return body


def _validated_docker_removal_dispatch(
    value: Mapping[str, object],
    *,
    binding_path: Path,
    binding_record: Mapping[str, object],
    termination_fence: Mapping[str, object],
) -> dict[str, object]:
    """Validate one exact, lineage-linked removal-request transition."""

    if not isinstance(value, Mapping):
        raise ValueError("Docker removal dispatch record is absent")
    try:
        expected = _docker_removal_dispatch_value(
            binding_path=binding_path,
            binding_record=binding_record,
            termination_fence=termination_fence,
            issuer_process_birth=value["issuer_process_birth"],
            state=str(value["state"]),
            generation=value["generation"],  # type: ignore[arg-type]
            previous_dispatch_id=str(value["previous_dispatch_id"]),
            docker_returncode=value["docker_returncode"],  # type: ignore[arg-type]
            failure_kind=str(value["failure_kind"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Docker removal dispatch record is invalid") from exc
    if dict(value) != expected:
        raise ValueError("Docker removal dispatch record drifted")
    return expected


def _docker_removal_issuer_live(
    issuer_process_birth: Mapping[str, object],
) -> bool | None:
    """Return exact issuer liveness; never confuse PID reuse with ownership."""

    try:
        pid = issuer_process_birth.get("pid")
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            return None
        observed = read_process_birth(pid)
    except (OSError, TypeError, ValueError):
        return None
    if observed is None:
        return False
    return observed.to_dict() == dict(issuer_process_birth)


def _open_docker_removal_execution_authority(
    binding_record: Mapping[str, object],
) -> tuple[int, int]:
    """Return exact executable/config descriptors for the removal effect."""

    docker_bin = Path(str(binding_record.get("docker_bin") or ""))
    docker_config = Path(str(binding_record.get("docker_config") or ""))
    identities = binding_record.get("path_identities")
    executable_fd = -1
    config_fd = -1
    try:
        if not docker_bin.is_absolute() or "\x00" in os.fspath(docker_bin):
            raise ValueError("Docker removal executable path is invalid")
        executable_fd = os.open(
            docker_bin,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        config_fd = os.open(
            docker_config,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        executable = os.fstat(executable_fd)
        config = os.fstat(config_fd)
        expected_config = (
            identities.get("docker_config")
            if isinstance(identities, Mapping)
            else None
        )
        if (
            not stat.S_ISREG(executable.st_mode)
            or executable.st_dev != binding_record.get("docker_device")
            or executable.st_ino != binding_record.get("docker_inode")
            or executable.st_mode != binding_record.get("docker_mode")
            or executable.st_uid != binding_record.get("docker_uid")
            or not isinstance(expected_config, Mapping)
            or config.st_dev != expected_config.get("device")
            or config.st_ino != expected_config.get("inode")
            or stat.S_IFMT(config.st_mode) != expected_config.get("mode")
            or config.st_uid != expected_config.get("uid")
            or not stat.S_ISDIR(config.st_mode)
            or not Path(f"/proc/self/fd/{executable_fd}").exists()
            or not Path(f"/proc/self/fd/{config_fd}").exists()
        ):
            raise ValueError("Docker removal execution authority drifted")
        return executable_fd, config_fd
    except BaseException:
        if config_fd >= 0:
            os.close(config_fd)
        if executable_fd >= 0:
            os.close(executable_fd)
        raise


def _docker_removal_issuer_main(
    argv: Sequence[str],
    *,
    control_socket: socket.socket | None = None,
) -> int:
    """Issue one Docker removal only after a durable request-start CAS."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--binding-path", type=Path, required=True)
    parser.add_argument("--runner-pid", type=int, required=True)
    parser.add_argument("--control-fd", type=int, default=-1)
    try:
        args = parser.parse_args(list(argv))
        if control_socket is None:
            control_socket = _docker_cleanup_control_socket(args.control_fd)
        elif args.control_fd >= 3:
            raise ValueError("Docker removal control descriptor is duplicated")
        peer_pid, peer_uid, peer_gid = _docker_control_peer_credentials(
            control_socket
        )
        if (
            args.runner_pid <= 0
            or peer_pid != args.runner_pid
            or peer_uid != os.geteuid()
            or peer_gid != os.getegid()
        ):
            raise ValueError("Docker removal control peer identity drifted")
        binding_path = args.binding_path.absolute()
        if (
            not binding_path.is_absolute()
            or re.fullmatch(r"[0-9a-f]{64}\.json", binding_path.name) is None
        ):
            raise ValueError("Docker removal binding path is invalid")
        issuer_birth = read_process_birth(os.getpid())
        if (
            issuer_birth is None
            or not issuer_birth.boot_id
            or issuer_birth.parent_pid != 1
        ):
            raise ValueError("Docker removal issuer is not detached")
        issuer_identity = issuer_birth.to_dict()
        control_socket.sendall(
            json.dumps(
                issuer_identity,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
            + b"\n"
        )
        # Explicit release and EOF have identical semantics.  The durable
        # prepared record below—not the byte—is the sole effect authority.
        control_socket.recv(1)
        control_socket.close()
        control_socket = None
    except (OSError, TypeError, ValueError):
        if control_socket is not None:
            control_socket.close()
        return 125

    lock_handle: _DockerBindingLock | None = None
    started: dict[str, object] | None = None
    try:
        lock_handle = _docker_binding_lock_descriptor(binding_path)
        binding_record = lock_handle.read(binding_path)
        dispatch_path = _docker_removal_dispatch_path(binding_path)
        observed = lock_handle.read(dispatch_path)
        if binding_record is None or observed is None:
            return 125
        raw_fence = binding_record.get("termination_fence")
        if not isinstance(raw_fence, Mapping) or not raw_fence:
            return 125
        prepared = _validated_docker_removal_dispatch(
            observed,
            binding_path=binding_path,
            binding_record=binding_record,
            termination_fence=raw_fence,
        )
        if (
            prepared.get("state") != "prepared"
            or prepared.get("issuer_process_birth") != issuer_identity
        ):
            return 125
        started = _docker_removal_dispatch_value(
            binding_path=binding_path,
            binding_record=binding_record,
            termination_fence=raw_fence,
            issuer_process_birth=issuer_identity,
            state="request_started",
            generation=int(prepared["generation"]),
            previous_dispatch_id=str(prepared["dispatch_id"]),
            docker_returncode=None,
            failure_kind="",
        )
        lock_handle.write(
            dispatch_path,
            started,
            replace_existing=True,
        )
        if lock_handle.read(dispatch_path) != started:
            return 125

        executable_fd = -1
        config_fd = -1
        try:
            lock_handle.assert_current()
            executable_fd, config_fd = (
                _open_docker_removal_execution_authority(binding_record)
            )
            completed = subprocess.run(
                [
                    f"/proc/self/fd/{executable_fd}",
                    f"--host={_DOCKER_LOCAL_HOST}",
                    "--config",
                    f"/proc/self/fd/{config_fd}",
                    "rm",
                    "--force",
                    str(started["container_id"]),
                ],
                env=_docker_control_env(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2.0,
                check=False,
                close_fds=True,
                pass_fds=(executable_fd, config_fd),
            )
            terminal_state = "request_completed"
            returncode: int | None = int(completed.returncode)
            failure_kind = ""
        except subprocess.TimeoutExpired:
            terminal_state = "request_outcome_unknown"
            returncode = None
            failure_kind = "timeout"
        except OSError:
            terminal_state = "request_outcome_unknown"
            returncode = None
            failure_kind = "os_error"
        finally:
            if config_fd >= 0:
                os.close(config_fd)
            if executable_fd >= 0:
                os.close(executable_fd)
        terminal = _docker_removal_dispatch_value(
            binding_path=binding_path,
            binding_record=binding_record,
            termination_fence=raw_fence,
            issuer_process_birth=issuer_identity,
            state=terminal_state,
            generation=int(started["generation"]),
            previous_dispatch_id=str(started["dispatch_id"]),
            docker_returncode=returncode,
            failure_kind=failure_kind,
        )
        lock_handle.write(
            dispatch_path,
            terminal,
            replace_existing=True,
        )
        return 0
    except BaseException:
        if started is not None and lock_handle is not None:
            try:
                binding_record = lock_handle.read(binding_path)
                raw_fence = (
                    binding_record.get("termination_fence")
                    if isinstance(binding_record, Mapping)
                    else None
                )
                if isinstance(raw_fence, Mapping) and raw_fence:
                    unknown = _docker_removal_dispatch_value(
                        binding_path=binding_path,
                        binding_record=binding_record,
                        termination_fence=raw_fence,
                        issuer_process_birth=issuer_identity,
                        state="request_outcome_unknown",
                        generation=int(started["generation"]),
                        previous_dispatch_id=str(started["dispatch_id"]),
                        docker_returncode=None,
                        failure_kind="issuer_failure",
                    )
                    lock_handle.write(
                        _docker_removal_dispatch_path(binding_path),
                        unknown,
                        replace_existing=True,
                    )
            except BaseException:
                pass
        return 125
    finally:
        if lock_handle is not None:
            lock_handle.close()


def _docker_removal_issuer_launcher_main(argv: Sequence[str]) -> int:
    """Exec-clean, double-fork launcher for one kill-tree-safe rm issuer."""

    items = list(argv)
    if (
        len(items) < 7
        or items[0] != "--control-fd"
        or items[2] != _DOCKER_REMOVAL_ISSUER_ARG
    ):
        return 2
    channel: socket.socket | None = None
    try:
        channel = _docker_cleanup_control_socket(int(items[1]))
        runner_index = items.index("--runner-pid", 3)
        runner_pid = int(items[runner_index + 1])
        peer_pid, peer_uid, peer_gid = _docker_control_peer_credentials(channel)
    except (IndexError, OSError, ValueError):
        if channel is not None:
            channel.close()
        return 2
    if (
        runner_pid <= 0
        or peer_pid != runner_pid
        or peer_uid != os.geteuid()
        or peer_gid != os.getegid()
    ):
        channel.close()
        return 2
    launcher_pid = os.getpid()
    try:
        child_pid = os.fork()
    except OSError:
        channel.close()
        return 2
    if child_pid:
        channel.close()
        return 0
    try:
        os.setsid()
    except OSError:
        channel.close()
        return 2
    deadline = time.monotonic() + 2.0
    while os.getppid() == launcher_pid and time.monotonic() < deadline:
        time.sleep(0.005)
    if os.getppid() != 1:
        channel.close()
        return 2
    return _docker_removal_issuer_main(items[3:], control_socket=channel)


def _arm_docker_removal_once(
    *,
    binding_path: Path,
    expected_binding_identity: Mapping[str, int],
    binding_record: Mapping[str, object],
    termination_fence: Mapping[str, object],
) -> bool:
    """Commit one recoverable rm request; callers may only reconcile.

    The multithreaded supervisor never forks.  It execs a clean launcher that
    double-forks an issuer outside the managed kill tree.  The issuer durably
    transitions ``prepared`` to ``request_started`` before Docker sees the
    request and publishes the exact CLI outcome afterwards.  A dead prepared
    issuer can be replaced by one lineage-linked generation; a started or
    unknown request is never replayed.
    """

    dispatch_path = _docker_removal_dispatch_path(binding_path)
    lock_handle = _docker_binding_lock_descriptor(binding_path)
    control_socket: socket.socket | None = None
    launcher_socket: socket.socket | None = None
    launcher: subprocess.Popen[bytes] | None = None
    try:
        current = lock_handle.read(binding_path)
        if (
            current != dict(binding_record)
            or lock_handle.path_identity(binding_path)
            != dict(expected_binding_identity)
        ):
            raise ValueError("Docker removal binding lost its CAS identity")
        observed = lock_handle.read(dispatch_path)
        generation = 1
        previous_dispatch_id = ""
        if observed is not None:
            admitted = _validated_docker_removal_dispatch(
                observed,
                binding_path=binding_path,
                binding_record=binding_record,
                termination_fence=termination_fence,
            )
            state = str(admitted["state"])
            issuer = admitted["issuer_process_birth"]
            if not isinstance(issuer, Mapping):
                raise ValueError("Docker removal issuer identity is absent")
            live = _docker_removal_issuer_live(issuer)
            if live is None:
                raise ValueError("Docker removal issuer liveness is unknown")
            if state == "prepared":
                if live:
                    return False
                generation = int(admitted["generation"]) + 1
                previous_dispatch_id = str(admitted["dispatch_id"])
            elif state == "request_started" and not live:
                unknown = _docker_removal_dispatch_value(
                    binding_path=binding_path,
                    binding_record=binding_record,
                    termination_fence=termination_fence,
                    issuer_process_birth=issuer,
                    state="request_outcome_unknown",
                    generation=int(admitted["generation"]),
                    previous_dispatch_id=str(admitted["dispatch_id"]),
                    docker_returncode=None,
                    failure_kind="issuer_failure",
                )
                lock_handle.write(
                    dispatch_path,
                    unknown,
                    replace_existing=True,
                )
                return False
            else:
                return False

        from .process_security import (
            require_state_authority_handoff_ptrace_protection,
        )

        require_state_authority_handoff_ptrace_protection()
        control_socket, launcher_socket = socket.socketpair(
            socket.AF_UNIX,
            socket.SOCK_STREAM,
        )
        sealed_match = re.fullmatch(r"/proc/self/fd/([0-9]+)", str(sys.argv[0]))
        runner_entry = (
            str(sys.argv[0])
            if sealed_match is not None
            else str(Path(__file__).resolve())
        )
        inherited_control_plane = (
            (int(sealed_match.group(1)),) if sealed_match is not None else ()
        )
        launcher = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-B",
                runner_entry,
                _DOCKER_REMOVAL_ISSUER_LAUNCHER_ARG,
                "--control-fd",
                str(launcher_socket.fileno()),
                _DOCKER_REMOVAL_ISSUER_ARG,
                "--binding-path",
                str(binding_path),
                "--runner-pid",
                str(os.getpid()),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd="/",
            env=_docker_cleanup_watchdog_env(),
            start_new_session=True,
            close_fds=True,
            pass_fds=tuple(
                sorted({*inherited_control_plane, launcher_socket.fileno()})
            ),
        )
        launcher_socket.close()
        launcher_socket = None
        if launcher.wait(timeout=3.0) != 0:
            raise ValueError("Docker removal issuer launcher failed")
        control_socket.settimeout(3.0)
        encoded_birth = bytearray()
        while b"\n" not in encoded_birth and len(encoded_birth) <= 512:
            chunk = control_socket.recv(513 - len(encoded_birth))
            if not chunk:
                break
            encoded_birth.extend(chunk)
        if not encoded_birth.endswith(b"\n") or len(encoded_birth) > 512:
            raise ValueError("Docker removal issuer birth is unavailable")
        try:
            reported_birth = json.loads(encoded_birth[:-1].decode("ascii"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("Docker removal issuer birth is malformed") from exc
        issuer_pid = (
            reported_birth.get("pid")
            if isinstance(reported_birth, Mapping)
            else None
        )
        observed_birth = (
            read_process_birth(issuer_pid)
            if isinstance(issuer_pid, int) and not isinstance(issuer_pid, bool)
            else None
        )
        if (
            observed_birth is None
            or reported_birth != observed_birth.to_dict()
            or observed_birth.parent_pid != 1
            or not observed_birth.boot_id
        ):
            raise ValueError("Docker removal issuer birth differs")
        prepared = _docker_removal_dispatch_value(
            binding_path=binding_path,
            binding_record=binding_record,
            termination_fence=termination_fence,
            issuer_process_birth=reported_birth,
            state="prepared",
            generation=generation,
            previous_dispatch_id=previous_dispatch_id,
            docker_returncode=None,
            failure_kind="",
        )
        lock_handle.write(
            dispatch_path,
            prepared,
            replace_existing=observed is not None,
        )
        admitted = lock_handle.read(dispatch_path)
        if admitted != prepared:
            raise ValueError("Docker removal dispatch was not durably admitted")
        try:
            control_socket.sendall(b"D")
        except OSError:
            # EOF releases the already self-bound issuer too.
            pass
        control_socket.close()
        control_socket = None
        return False
    finally:
        if control_socket is not None:
            try:
                control_socket.close()
            except OSError:
                pass
        if launcher_socket is not None:
            try:
                launcher_socket.close()
            except OSError:
                pass
        if launcher is not None and launcher.poll() is None:
            try:
                launcher.kill()
            except OSError:
                pass
            try:
                launcher.wait(timeout=1.0)
            except (OSError, subprocess.TimeoutExpired):
                pass
        lock_handle.close()


def _robust_remove_runner_temp_tree(
    path: Path,
    *,
    expected_identity: Mapping[str, int] | None = None,
) -> bool:
    """Remove one exact temp-tree inode using only descriptor-relative walks.

    A provider may change permissions or create symlinks below its private
    home.  Cleanup may unlink those directory entries, but it must never follow
    them and must never turn a rename/replacement race into authority over a
    different tree.  The retained descriptors and inode checks below make each
    directory boundary fail closed under concurrent namespace changes.
    """

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)

    def same_inode(left: os.stat_result, right: os.stat_result) -> bool:
        return bool(
            left.st_dev == right.st_dev
            and left.st_ino == right.st_ino
            and stat.S_IFMT(left.st_mode) == stat.S_IFMT(right.st_mode)
            and left.st_uid == right.st_uid == os.geteuid()
        )

    def remove_contents(directory_fd: int) -> bool:
        try:
            names = tuple(os.listdir(directory_fd))
        except OSError:
            return False
        for name in names:
            if not name or name in {".", ".."} or "/" in name:
                return False
            try:
                before = os.stat(
                    name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
                if stat.S_ISDIR(before.st_mode) and not stat.S_ISLNK(
                    before.st_mode
                ):
                    os.chmod(
                        name,
                        0o700,
                        dir_fd=directory_fd,
                        follow_symlinks=False,
                    )
                    after_chmod = os.stat(
                        name,
                        dir_fd=directory_fd,
                        follow_symlinks=False,
                    )
                    if not same_inode(before, after_chmod):
                        return False
                    child_fd = os.open(name, flags, dir_fd=directory_fd)
                    try:
                        opened = os.fstat(child_fd)
                        if not same_inode(after_chmod, opened):
                            return False
                        if not remove_contents(child_fd):
                            return False
                        current = os.stat(
                            name,
                            dir_fd=directory_fd,
                            follow_symlinks=False,
                        )
                        if not same_inode(opened, current):
                            return False
                    finally:
                        os.close(child_fd)
                    os.rmdir(name, dir_fd=directory_fd)
                else:
                    # unlinkat removes the entry itself.  A symlink or special
                    # file is never opened and therefore cannot redirect the
                    # cleanup walk outside this descriptor-bound directory.
                    os.unlink(name, dir_fd=directory_fd)
            except (FileNotFoundError, NotImplementedError, OSError):
                return False
        try:
            os.fsync(directory_fd)
            return not os.listdir(directory_fd)
        except OSError:
            return False

    parent_fd = -1
    directory_fd = -1
    try:
        parent_fd = os.open(path.parent, flags)
        before = os.stat(
            path.name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or before.st_uid != os.geteuid()
            or (
                expected_identity is not None
                and not _owned_cleanup_path_matches(
                    before,
                    directory=True,
                    identity=expected_identity,
                )
            )
        ):
            return False
        os.chmod(
            path.name,
            0o700,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        after_chmod = os.stat(
            path.name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if not same_inode(before, after_chmod):
            return False
        directory_fd = os.open(path.name, flags, dir_fd=parent_fd)
        opened = os.fstat(directory_fd)
        if not same_inode(after_chmod, opened):
            return False
        if not remove_contents(directory_fd):
            return False
        current = os.stat(
            path.name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if not same_inode(opened, current):
            return False
        os.rmdir(path.name, dir_fd=parent_fd)
        os.fsync(parent_fd)
        try:
            os.stat(
                path.name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return True
        return False
    except (FileNotFoundError, NotImplementedError, OSError, ValueError):
        return False
    finally:
        if directory_fd >= 0:
            os.close(directory_fd)
        if parent_fd >= 0:
            os.close(parent_fd)


def _owned_cleanup_path_matches(
    metadata: os.stat_result,
    *,
    directory: bool,
    identity: Mapping[str, int],
) -> bool:
    return bool(
        metadata.st_dev == identity.get("device")
        and metadata.st_ino == identity.get("inode")
        and stat.S_IFMT(metadata.st_mode) == identity.get("mode")
        and metadata.st_uid == identity.get("uid") == os.geteuid()
        and not stat.S_ISLNK(metadata.st_mode)
        and (
            stat.S_ISDIR(metadata.st_mode)
            if directory
            else stat.S_ISREG(metadata.st_mode)
        )
    )


def _cleanup_path_quarantine(
    path: Path,
    *,
    directory: bool,
    identity: Mapping[str, int],
) -> tuple[Path, Path, Path, dict[str, object]]:
    """Derive the replayable private quarantine for one admitted inode."""

    body: dict[str, object] = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "cleanup-path-tombstone@1"
        ),
        "path": str(path.absolute()),
        "directory": directory,
        "identity": {
            name: int(identity.get(name, -1))
            for name in ("device", "inode", "mode", "uid")
        },
        "transition": "exact_inode_quarantined_for_removal",
    }
    body["tombstone_id"] = _effect_receipt_identity(body)
    quarantine = path.with_name(
        ".aseh-cleanup-"
        + hashlib.sha256(
            json.dumps(
                body,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
    )
    return quarantine, quarantine / "owned", quarantine / "removed.json", body


def _cleanup_tombstone_matches(
    marker: Path,
    expected: Mapping[str, object],
) -> bool:
    try:
        observed = _read_private_control_record(marker.parent, marker.name)
    except (OSError, ValueError):
        return False
    return observed == dict(expected)


def _remove_owned_cleanup_path(
    path: Path,
    *,
    directory: bool,
    identity: Mapping[str, int],
    admitted_tombstone_id: str = "",
) -> bool:
    """Replayably quarantine and remove only the admitted inode.

    The identity-bound tombstone is written and fsynced before ``owned`` is
    deleted.  Consequently a crash can distinguish an admitted removal from
    a same-UID/provider rename-away: an initially absent path without the
    exact tombstone fails closed.
    """

    def matches(metadata: os.stat_result) -> bool:
        return _owned_cleanup_path_matches(
            metadata,
            directory=directory,
            identity=identity,
        )

    quarantine, owned, marker, tombstone = _cleanup_path_quarantine(
        path,
        directory=directory,
        identity=identity,
    )
    try:
        original_exists = os.path.lexists(path)
        owned_exists = os.path.lexists(owned)
        marker_exists = os.path.lexists(marker)
        if original_exists:
            if not matches(os.lstat(path)) or owned_exists or marker_exists:
                return False
            path.chmod(0o700 if directory else 0o600, follow_symlinks=False)
            if not matches(os.lstat(path)):
                return False
            try:
                quarantine.mkdir(mode=0o700)
            except FileExistsError:
                quarantine_metadata = os.lstat(quarantine)
                if (
                    not stat.S_ISDIR(quarantine_metadata.st_mode)
                    or stat.S_ISLNK(quarantine_metadata.st_mode)
                    or quarantine_metadata.st_uid != os.geteuid()
                    or stat.S_IMODE(quarantine_metadata.st_mode) != 0o700
                    or any(quarantine.iterdir())
                ):
                    return False
            os.rename(path, owned)
            for directory_path in (path.parent, quarantine):
                descriptor = os.open(
                    directory_path,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                )
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            owned_exists = True
        elif not owned_exists:
            # A deterministic, self-hashed marker is an audit journal, not
            # mutation authority.  Marker-only replay is admitted exclusively
            # when the canonical terminal attempt CAS precommitted this exact
            # tombstone before any inode mutation.  Unscoped cleanup therefore
            # remains fail-closed across this crash gap.
            if (
                admitted_tombstone_id != tombstone.get("tombstone_id")
                or not marker_exists
                or not _cleanup_tombstone_matches(marker, tombstone)
            ):
                return False
        if owned_exists and not matches(os.lstat(owned)):
            # A replacement won between the pre-rename lstat and rename.
            # Restore that unrelated inode to its original name; never delete
            # it under cleanup authority for the displaced admitted inode.
            if not os.path.lexists(path):
                try:
                    os.rename(owned, path)
                except OSError:
                    pass
            return False
        if not marker_exists:
            _write_private_control_record(
                quarantine,
                marker.name,
                tombstone,
                replace_existing=False,
            )
        elif not _cleanup_tombstone_matches(marker, tombstone):
            return False
        if owned_exists:
            if directory:
                if not _robust_remove_runner_temp_tree(
                    owned,
                    expected_identity=identity,
                ):
                    return False
            else:
                current = os.lstat(owned)
                if not matches(current):
                    return False
                owned.chmod(0o600, follow_symlinks=False)
                owned.unlink()
            descriptor = os.open(
                quarantine,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    except (FileNotFoundError, OSError, ValueError):
        return False
    return bool(
        not os.path.lexists(path)
        and not os.path.lexists(owned)
        and _cleanup_tombstone_matches(marker, tombstone)
    )


def _remove_or_admit_cleanup_tombstone(
    path: Path,
    *,
    directory: bool,
    identity: Mapping[str, int],
) -> bool:
    """Remove a live exact inode; never promote a public marker to authority.

    Crash-gap admission is intentionally confined to the active binding CAS
    while its stable lock is held by
    :func:`_finalize_verified_cleanup_completion`.  This public compatibility
    helper therefore cannot turn a predictable self-hashed tombstone into
    deletion evidence.
    """

    return _remove_owned_cleanup_path(
        path,
        directory=directory,
        identity=identity,
    )


def _discard_owned_cleanup_tombstone(
    path: Path,
    *,
    directory: bool,
    identity: Mapping[str, int],
) -> bool:
    """Best-effort finalization after the durable binding is gone."""

    quarantine, owned, marker, tombstone = _cleanup_path_quarantine(
        path,
        directory=directory,
        identity=identity,
    )
    try:
        if os.path.lexists(path) or os.path.lexists(owned):
            return False
        if not _cleanup_tombstone_matches(marker, tombstone):
            return False
        marker.unlink()
        descriptor = os.open(
            quarantine,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        quarantine.rmdir()
        parent_descriptor = os.open(
            path.parent,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
        return True
    except (FileNotFoundError, OSError):
        return False


def _unlink_owned_cleanup_record(
    path: Path,
    *,
    identity: Mapping[str, int],
) -> bool:
    """Unlink one exact private control record and durably fsync its parent."""

    directory_fd = -1
    try:
        metadata = os.lstat(path)
        if not _owned_cleanup_path_matches(
            metadata,
            directory=False,
            identity=identity,
        ):
            return False
        directory_fd = os.open(
            path.parent,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        current = os.stat(
            path.name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        if not _owned_cleanup_path_matches(
            current,
            directory=False,
            identity=identity,
        ):
            return False
        os.unlink(path.name, dir_fd=directory_fd)
        os.fsync(directory_fd)
        return not os.path.lexists(path)
    except (FileNotFoundError, OSError):
        return False
    finally:
        if directory_fd >= 0:
            os.close(directory_fd)


def _cleanup_completion_path(binding_path: Path) -> Path:
    if re.fullmatch(r"[0-9a-f]{64}\.json", binding_path.name) is None:
        raise ValueError("Docker cleanup binding name is invalid")
    return binding_path.with_suffix(".complete")


def _cleanup_authority_path(binding_path: Path) -> Path:
    """Return the retained exact binding inode after cleanup retirement."""

    if re.fullmatch(r"[0-9a-f]{64}\.json", binding_path.name) is None:
        raise ValueError("Docker cleanup binding name is invalid")
    return binding_path.with_suffix(".authority")


def _cleanup_binding_authority_present(
    binding_path: Path,
    *,
    binding_identity: Mapping[str, int],
    binding_record: Mapping[str, object],
    binding_lock: _DockerBindingLock | None = None,
) -> bool:
    """Validate exactly one live or retired copy of the original inode."""

    authority_path = _cleanup_authority_path(binding_path)
    live = (
        binding_lock.exists(binding_path)
        if binding_lock is not None
        else os.path.lexists(binding_path)
    )
    retired = (
        binding_lock.exists(authority_path)
        if binding_lock is not None
        else os.path.lexists(authority_path)
    )
    if live is retired:
        return False
    path = binding_path if live else authority_path
    try:
        return bool(
            (
                binding_lock.read(path)
                if binding_lock is not None
                else _read_private_control_record(path.parent, path.name)
            )
            == dict(binding_record)
            and (
                binding_lock.path_identity(path)
                if binding_lock is not None
                else _cleanup_path_identity(path, directory=False)
            )
            == dict(binding_identity)
        )
    except (OSError, ValueError):
        return False


def _cleanup_binding_retirement_pair_matches(
    directory_fd: int,
    *,
    binding_name: str,
    authority_name: str,
    binding_identity: Mapping[str, int],
    binding_record: Mapping[str, object],
) -> bool:
    """Validate the transient two-link retirement state without mutation."""

    descriptor = -1
    try:
        descriptor = os.open(
            authority_name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_fd,
        )
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or before.st_nlink != 2
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_size > _DOCKER_PRIVATE_CONTROL_MAX_BYTES
            or before.st_dev != binding_identity.get("device")
            or before.st_ino != binding_identity.get("inode")
            or stat.S_IFMT(before.st_mode) != binding_identity.get("mode")
            or before.st_uid != binding_identity.get("uid")
        ):
            return False
        remaining = _DOCKER_PRIVATE_CONTROL_MAX_BYTES + 1
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        binding = os.stat(
            binding_name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        authority = os.stat(
            authority_name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        snapshots = tuple(
            (
                item.st_dev,
                item.st_ino,
                item.st_mode,
                item.st_uid,
                item.st_nlink,
                item.st_size,
                item.st_mtime_ns,
                item.st_ctime_ns,
            )
            for item in (before, after, binding, authority)
        )
        return bool(
            all(snapshot == snapshots[0] for snapshot in snapshots[1:])
            and raw == _private_control_bytes(binding_record)
        )
    except (FileNotFoundError, OSError, TypeError, ValueError):
        return False
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _retire_cleanup_binding_authority(
    binding_path: Path,
    *,
    binding_identity: Mapping[str, int],
    binding_record: Mapping[str, object],
    binding_lock: _DockerBindingLock | None = None,
) -> bool:
    """Retain the exact binding inode under a create-only authority name.

    The hard-link/fsync/unlink sequence cannot overwrite a raced foreign
    authority file.  If the process exits after link publication, a later
    call validates the exact two-link inode and finishes only the pending
    source unlink.
    """

    authority_path = _cleanup_authority_path(binding_path)
    directory_fd = -1
    owns_directory_fd = binding_lock is None
    try:
        directory_fd = (
            _private_control_directory(binding_path.parent)
            if binding_lock is None
            else binding_lock.directory_fd
        )
        if binding_lock is not None:
            if binding_lock.binding_path != binding_path.absolute():
                return False
            binding_lock.assert_current()
        try:
            binding_metadata = os.stat(
                binding_path.name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            binding_metadata = None
        try:
            authority_metadata = os.stat(
                authority_path.name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            authority_metadata = None

        if authority_metadata is not None:
            if binding_metadata is None:
                return _cleanup_binding_authority_present(
                    binding_path,
                    binding_identity=binding_identity,
                    binding_record=binding_record,
                    binding_lock=binding_lock,
                )
            if not _cleanup_binding_retirement_pair_matches(
                directory_fd,
                binding_name=binding_path.name,
                authority_name=authority_path.name,
                binding_identity=binding_identity,
                binding_record=binding_record,
            ):
                return False
            os.unlink(binding_path.name, dir_fd=directory_fd)
            os.fsync(directory_fd)
            return _cleanup_binding_authority_present(
                binding_path,
                binding_identity=binding_identity,
                binding_record=binding_record,
                binding_lock=binding_lock,
            )

        if (
            binding_metadata is None
            or (
                binding_lock.read(binding_path)
                if binding_lock is not None
                else _read_private_control_record(
                    binding_path.parent,
                    binding_path.name,
                )
            )
            != dict(binding_record)
            or (
                binding_lock.path_identity(binding_path)
                if binding_lock is not None
                else _cleanup_path_identity(binding_path, directory=False)
            )
            != dict(binding_identity)
        ):
            return False
        os.link(
            binding_path.name,
            authority_path.name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
        os.fsync(directory_fd)
        if not _cleanup_binding_retirement_pair_matches(
            directory_fd,
            binding_name=binding_path.name,
            authority_name=authority_path.name,
            binding_identity=binding_identity,
            binding_record=binding_record,
        ):
            return False
        os.unlink(binding_path.name, dir_fd=directory_fd)
        os.fsync(directory_fd)
        if binding_lock is not None:
            binding_lock.assert_current()
        return _cleanup_binding_authority_present(
            binding_path,
            binding_identity=binding_identity,
            binding_record=binding_record,
            binding_lock=binding_lock,
        )
    except (FileNotFoundError, OSError, ValueError):
        return False
    finally:
        if owns_directory_fd and directory_fd >= 0:
            os.close(directory_fd)


def _cleanup_intent_value(
    *,
    binding_path: Path,
    binding_identity: Mapping[str, int],
    binding_record: Mapping[str, object],
    docker_absence: Mapping[str, object],
    terminal_cleanup_authority: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Bind the exact inodes and lifecycle before any path is removed."""

    record_body = {
        key: item for key, item in binding_record.items() if key != "record_id"
    }
    record_id = str(binding_record.get("record_id") or "")
    raw_fence = binding_record.get("termination_fence")
    fence_id = str(
        raw_fence.get("fence_id")
        if isinstance(raw_fence, Mapping) and raw_fence
        else ""
    )
    path_identities = binding_record.get("path_identities")
    authority = dict(terminal_cleanup_authority or {})
    if authority:
        authority_body = {
            key: item for key, item in authority.items() if key != "authority_id"
        }
        if (
            set(authority)
            != {
                "schema",
                "logical_attempt_id",
                "reservation_id",
                "cleanup_id",
                "binding_path",
                "binding_record_id",
                "termination_fence_id",
                "authority_id",
            }
            or authority.get("schema")
            != (
                "ipfs_accelerate_py/agent-supervisor/"
                "terminal-cleanup-authority@1"
            )
            or authority.get("binding_path") != str(binding_path)
            or authority.get("binding_record_id") != record_id
            or authority.get("termination_fence_id") != fence_id
            or authority.get("authority_id")
            != _effect_receipt_identity(authority_body)
        ):
            raise ValueError("terminal cleanup intent authority is invalid")
    else:
        authority = {
            "logical_attempt_id": "",
            "reservation_id": "",
            "cleanup_id": "",
            "authority_id": record_id,
        }
    lifecycle_names = (
        "run_id",
        "profile_id",
        "target_id",
        "repository_root",
        "state_root",
        "run_root",
        "configuration_root",
        "fencing_epoch",
    )
    if (
        binding_record.get("record_id") != _effect_receipt_identity(record_body)
        or binding_record.get("binding_path") != str(binding_path)
        or not isinstance(path_identities, Mapping)
        or set(path_identities)
        != {
            "docker_config",
            "lease_root",
            "prompt_path",
            "provider_home",
        }
        or set(binding_identity) != {"device", "inode", "mode", "uid"}
        or not stat.S_ISREG(int(binding_identity.get("mode", 0)))
        or binding_identity.get("uid") != os.geteuid()
        or not isinstance(docker_absence, Mapping)
        or docker_absence.get("binding_record_id") != record_id
        or any(binding_record.get(name) in (None, "") for name in lifecycle_names)
    ):
        raise ValueError("Docker cleanup intent inputs are invalid")
    resources: list[dict[str, object]] = []
    for name, field, directory in (
        ("prompt_path", "prompt_path", False),
        ("provider_home", "provider_home", True),
        ("lease_root", "lease_root", True),
    ):
        identity = path_identities.get(name)
        path = Path(str(binding_record.get(field) or ""))
        if not isinstance(identity, Mapping):
            raise ValueError("Docker cleanup resource identity is invalid")
        _quarantine, _owned, _marker, tombstone = _cleanup_path_quarantine(
            path,
            directory=directory,
            identity=identity,
        )
        resources.append(
            {
                "name": name,
                "path": str(path),
                "directory": directory,
                "identity": dict(identity),
                "tombstone_id": tombstone["tombstone_id"],
            }
        )
    body: dict[str, object] = {
        "schema": _DOCKER_CLEANUP_INTENT_SCHEMA,
        "logical_attempt_id": authority["logical_attempt_id"],
        "reservation_id": authority["reservation_id"],
        "cleanup_id": authority["cleanup_id"],
        "authority_id": authority["authority_id"],
        "binding_path": str(binding_path),
        "binding_identity": dict(binding_identity),
        "binding_record_id": record_id,
        "termination_fence_id": fence_id,
        "lifecycle": {
            name: binding_record[name] for name in lifecycle_names
        },
        "resources": resources,
        "docker_absence": dict(docker_absence),
    }
    body["intent_id"] = _effect_receipt_identity(body)
    return body


def _cleanup_completion_value(
    *,
    binding_path: Path,
    binding_identity: Mapping[str, int],
    binding_record: Mapping[str, object],
    terminal_cleanup_authority: Mapping[str, object] | None = None,
    binding_lock: _DockerBindingLock | None = None,
) -> dict[str, object]:
    """Build the durable transition from exact absence to path removal."""

    record_body = {
        key: item for key, item in binding_record.items() if key != "record_id"
    }
    path_identities = binding_record.get("path_identities")
    if (
        binding_record.get("record_id") != _effect_receipt_identity(record_body)
        or binding_record.get("binding_path") != str(binding_path)
        or not isinstance(path_identities, dict)
        or set(path_identities)
        != {
            "docker_config",
            "lease_root",
            "prompt_path",
            "provider_home",
        }
    ):
        raise ValueError("Docker cleanup completion authority is invalid")
    resources = [
        {
            "name": name,
            "path": str(binding_record[path_field]),
            "directory": directory,
            "identity": dict(path_identities[name]),
        }
        for name, path_field, directory in (
            ("prompt_path", "prompt_path", False),
            ("provider_home", "provider_home", True),
            ("lease_root", "lease_root", True),
        )
    ]
    raw_fence = binding_record.get("termination_fence")
    dispatch_path = _docker_removal_dispatch_path(binding_path)
    dispatch = (
        binding_lock.read(dispatch_path)
        if binding_lock is not None
        else _read_private_control_record(
            dispatch_path.parent,
            dispatch_path.name,
        )
    )
    if isinstance(raw_fence, Mapping) and raw_fence:
        fence = _validated_docker_termination_fence(
            raw_fence,
            provider=str(binding_record.get("provider") or ""),
            container_name=str(binding_record.get("container_name") or ""),
        )
        expected_dispatch = _validated_docker_removal_dispatch(
            dispatch if isinstance(dispatch, Mapping) else {},
            binding_path=binding_path,
            binding_record=binding_record,
            termination_fence=fence,
        )
        dispatch_state = expected_dispatch.get("state")
        if dispatch_state == "request_started":
            issuer_live = _docker_removal_issuer_live(
                expected_dispatch["issuer_process_birth"]  # type: ignore[arg-type]
            )
            if issuer_live is not False:
                raise ValueError(
                    "Docker removal request outcome is not reconcilable"
                )
        elif dispatch_state not in {
            "request_completed",
            "request_outcome_unknown",
        }:
            raise ValueError("Docker cleanup dispatch is not effect-terminal")
        docker_absence: dict[str, object] = {
            "kind": "fenced_effect_absence",
            "binding_record_id": binding_record["record_id"],
            "container_id": fence["container_id"],
            "fence_id": fence["fence_id"],
            "dispatch_id": expected_dispatch["dispatch_id"],
            "observation": "exact_cid_name_and_kernel_scope_quiescent",
        }
    else:
        if dispatch is not None:
            raise ValueError("unfenced Docker cleanup has a dispatch record")
        docker_absence = {
            "kind": "unmaterialized_name_absence",
            "binding_record_id": binding_record["record_id"],
            "binding_state": binding_record.get("binding_state"),
            "observation": "exact_name_absent_twice",
        }
    cleanup_intent = _cleanup_intent_value(
        binding_path=binding_path,
        binding_identity=binding_identity,
        binding_record=binding_record,
        docker_absence=docker_absence,
        terminal_cleanup_authority=terminal_cleanup_authority,
    )
    body: dict[str, object] = {
        "schema": _DOCKER_CLEANUP_COMPLETION_SCHEMA,
        "binding_path": str(binding_path),
        "binding_identity": dict(binding_identity),
        "binding_record": dict(binding_record),
        "resources": resources,
        "docker_absence": docker_absence,
        "cleanup_intent": cleanup_intent,
    }
    body["completion_id"] = _effect_receipt_identity(body)
    return body


def _publish_cleanup_completion(
    *,
    binding_path: Path,
    binding_identity: Mapping[str, int],
    binding_record: Mapping[str, object],
    terminal_cleanup_authority: Mapping[str, object] | None = None,
    binding_lock: _DockerBindingLock | None = None,
) -> Path:
    completion_path = _cleanup_completion_path(binding_path)
    expected = _cleanup_completion_value(
        binding_path=binding_path,
        binding_identity=binding_identity,
        binding_record=binding_record,
        terminal_cleanup_authority=terminal_cleanup_authority,
        binding_lock=binding_lock,
    )
    observed = (
        binding_lock.read(completion_path)
        if binding_lock is not None
        else _read_private_control_record(
            completion_path.parent,
            completion_path.name,
        )
    )
    if observed is None:
        if binding_lock is not None:
            binding_lock.write(
                completion_path,
                expected,
                replace_existing=False,
            )
        else:
            _write_private_control_record(
                completion_path.parent,
                completion_path.name,
                expected,
                replace_existing=False,
            )
    elif observed != expected:
        raise ValueError("Docker cleanup completion record drifted")
    return completion_path


def _cleanup_intent_terminal_authority(
    intent: Mapping[str, object],
) -> Mapping[str, object] | None:
    """Reconstruct the exact terminal authority embedded by a scoped intent."""

    if not str(intent.get("logical_attempt_id") or ""):
        return None
    authority: dict[str, object] = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "terminal-cleanup-authority@1"
        ),
        "logical_attempt_id": intent.get("logical_attempt_id"),
        "reservation_id": intent.get("reservation_id"),
        "cleanup_id": intent.get("cleanup_id"),
        "binding_path": intent.get("binding_path"),
        "binding_record_id": intent.get("binding_record_id"),
        "termination_fence_id": intent.get("termination_fence_id"),
    }
    authority["authority_id"] = _effect_receipt_identity(authority)
    if authority.get("authority_id") != intent.get("authority_id"):
        raise ValueError("terminal cleanup intent authority drifted")
    return authority


def _cleanup_progress_matches(
    progress: Mapping[str, object] | None,
    *,
    intent: Mapping[str, object],
    completion_id: str = "",
) -> bool:
    """Validate authority-observed monotonic cleanup progress bytes."""

    if not isinstance(progress, Mapping) or set(progress) != {
        "schema",
        "logical_attempt_id",
        "reservation_id",
        "authority_id",
        "phase",
        "intent_id",
        "intent",
        "completion_id",
        "previous_progress_id",
        "progress_id",
    }:
        return False
    body = {
        key: item for key, item in progress.items() if key != "progress_id"
    }
    phase = progress.get("phase")
    if (
        progress.get("schema")
        != (
            "ipfs_accelerate_py/agent-supervisor/"
            "terminal-cleanup-progress@1"
        )
        or phase not in {"intent_committed", "completion_committed"}
        or progress.get("logical_attempt_id")
        != intent.get("logical_attempt_id")
        or progress.get("reservation_id") != intent.get("reservation_id")
        or progress.get("authority_id") != intent.get("authority_id")
        or progress.get("intent_id") != intent.get("intent_id")
        or progress.get("intent") != dict(intent)
        or progress.get("progress_id") != _effect_receipt_identity(body)
        or (
            phase == "intent_committed"
            and (
                progress.get("completion_id") != ""
                or progress.get("previous_progress_id") != ""
            )
        )
        or (
            phase == "completion_committed"
            and (
                re.fullmatch(
                    r"sha256:[0-9a-f]{64}",
                    str(progress.get("completion_id") or ""),
                )
                is None
                or re.fullmatch(
                    r"sha256:[0-9a-f]{64}",
                    str(progress.get("previous_progress_id") or ""),
                )
                is None
            )
        )
        or completion_id
        and (
            phase != "completion_committed"
            or progress.get("completion_id") != completion_id
        )
    ):
        return False
    return True


def _recover_cleanup_completion(
    completion_path: Path,
    *,
    expected_lifecycle: Mapping[str, object] | None = None,
    docker_absence_verified: bool = False,
    binding_lock: _DockerBindingLock | None = None,
    terminal_cleanup_progress: Mapping[str, object] | None = None,
) -> bool:
    """Converge an already-authorized post-Docker cleanup transition.

    A completion record has a public integrity hash, not mutation authority.
    Replay therefore requires the retained original binding inode and, for a
    marker-only crash gap, the exact terminal attempt-CAS intent.  Tombstones
    are discarded only after that CAS also binds these exact completion bytes
    and the binding inode has been renamed to its retained authority name.
    """

    lock_handle = binding_lock
    owns_lock = binding_lock is None
    try:
        if re.fullmatch(r"[0-9a-f]{64}\.complete", completion_path.name) is None:
            return False
        expected_binding_path = completion_path.with_suffix(".json")
        if lock_handle is None:
            lock_handle = _docker_binding_lock_descriptor(expected_binding_path)
        value = lock_handle.read(completion_path)
        if value is None or set(value) != {
            "schema",
            "binding_path",
            "binding_identity",
            "binding_record",
            "resources",
            "docker_absence",
            "cleanup_intent",
            "completion_id",
        }:
            return False
        body = {key: item for key, item in value.items() if key != "completion_id"}
        binding_path = Path(str(value.get("binding_path") or ""))
        binding_identity = value.get("binding_identity")
        binding_record = value.get("binding_record")
        resources = value.get("resources")
        cleanup_intent = value.get("cleanup_intent")
        record_container = str(
            binding_record.get("container_name")
            if isinstance(binding_record, dict)
            else ""
        )
        record_provider = str(
            binding_record.get("provider")
            if isinstance(binding_record, dict)
            else ""
        )
        record_lease = Path(
            str(
                binding_record.get("lease_root")
                if isinstance(binding_record, dict)
                else ""
            )
        )
        record_home = Path(
            str(
                binding_record.get("provider_home")
                if isinstance(binding_record, dict)
                else ""
            )
        )
        record_prompt = Path(
            str(
                binding_record.get("prompt_path")
                if isinstance(binding_record, dict)
                else ""
            )
        )
        cleanup_root_valid = False
        if isinstance(binding_record, dict):
            try:
                observed_root, observed_root_identity = (
                    _validated_docker_cleanup_root(
                        lease_root=record_lease,
                        provider_home=record_home,
                        prompt_path=record_prompt,
                        expected_root=Path(
                            str(binding_record.get("cleanup_root") or "")
                        ),
                        expected_identity=binding_record.get(
                            "cleanup_root_identity"
                        ),
                    )
                )
                cleanup_root_valid = bool(
                    binding_record.get("cleanup_root") == str(observed_root)
                    and binding_record.get("cleanup_root_identity")
                    == observed_root_identity
                )
            except (TypeError, ValueError):
                cleanup_root_valid = False
        lifecycle_valid = bool(
            expected_lifecycle is None
            or (
                isinstance(binding_record, dict)
                and all(
                    binding_record.get(name) == expected
                    for name, expected in expected_lifecycle.items()
                )
            )
        )
        if (
            value.get("schema") != _DOCKER_CLEANUP_COMPLETION_SCHEMA
            or value.get("completion_id") != _effect_receipt_identity(body)
            or not isinstance(value.get("docker_absence"), dict)
            or completion_path != _cleanup_completion_path(binding_path)
            or binding_path != expected_binding_path
            or completion_path.parent != binding_path.parent
            or binding_path.parent.name != _DOCKER_CLEANUP_BINDING_DIRECTORY
            or not isinstance(binding_identity, dict)
            or set(binding_identity) != {"device", "inode", "mode", "uid"}
            or not isinstance(binding_record, dict)
            or not isinstance(cleanup_intent, dict)
            or binding_record.get("schema") != _DOCKER_CLEANUP_BINDING_SCHEMA
            or not cleanup_root_valid
            or binding_record.get("binding_path") != str(binding_path)
            or binding_record.get("provider") not in _DOCKER_ISOLATION_PROVIDERS
            or record_provider not in _DOCKER_ISOLATION_PROVIDERS
            or _DOCKER_CONTAINER_NAME_RE.fullmatch(record_container) is None
            or not record_container.startswith(
                f"ipfs-accelerate-{record_provider}-"
            )
            or binding_path
            != binding_path.parent
            / (hashlib.sha256(record_container.encode("ascii")).hexdigest() + ".json")
            or not record_lease.name.startswith(
                f"asref-{record_provider}-container-"
            )
            or not record_home.name.startswith(
                f"asref-{record_provider}-home-"
            )
            or not record_prompt.name.startswith("asref-grok-prompt-")
            or binding_record.get("docker_config")
            != str(record_lease / "docker-config")
            or binding_record.get("cidfile") != str(record_lease / "container.cid")
            or not lifecycle_valid
            or _cleanup_completion_value(
                binding_path=binding_path,
                binding_identity=binding_identity,
                binding_record=binding_record,
                terminal_cleanup_authority=(
                    _cleanup_intent_terminal_authority(cleanup_intent)
                ),
                binding_lock=lock_handle,
            )
            != value
            or not isinstance(resources, list)
        ):
            return False
        if not docker_absence_verified:
            return False
        lock_handle.path_identity(completion_path)
        scoped_intent = bool(cleanup_intent.get("logical_attempt_id"))
        if scoped_intent and not _cleanup_progress_matches(
            terminal_cleanup_progress,
            intent=cleanup_intent,
            completion_id=str(value.get("completion_id") or ""),
        ):
            return False
        if (
            not _cleanup_binding_authority_present(
                binding_path,
                binding_identity=binding_identity,
                binding_record=binding_record,
                binding_lock=lock_handle,
            )
        ):
            return False
        resource_specs: list[tuple[Path, bool, Mapping[str, int]]] = []
        for item in resources:
            if (
                not isinstance(item, dict)
                or set(item) != {"name", "path", "directory", "identity"}
                or not isinstance(item.get("directory"), bool)
                or not isinstance(item.get("identity"), dict)
            ):
                return False
            resource_specs.append(
                (
                    Path(str(item["path"])),
                    bool(item["directory"]),
                    item["identity"],
                )
            )
        resource_states: list[
            tuple[Path, bool, Mapping[str, int], bool]
        ] = []
        for path, directory, identity in resource_specs:
            quarantine, owned, marker, tombstone = _cleanup_path_quarantine(
                path,
                directory=directory,
                identity=identity,
            )
            # A self-hashed completion is never authority to rename or delete
            # a live original (or an incompletely quarantined ``owned``
            # inode).  Only the pre-existing exact tombstone admits final
            # tombstone disposal below.
            if os.path.lexists(path) or os.path.lexists(owned):
                return False
            tombstone_present = _cleanup_tombstone_matches(marker, tombstone)
            fully_absent = bool(
                not tombstone_present
                and not os.path.lexists(marker)
                and not os.path.lexists(quarantine)
            )
            if not tombstone_present and not fully_absent:
                return False
            resource_states.append(
                (path, directory, identity, tombstone_present)
            )
        # A crash can occur between any two tombstone disposals.  Each exact
        # resource therefore converges independently from either its admitted
        # tombstone or a fully absent prior completion state.
        for path, directory, identity, tombstone_present in resource_states:
            if tombstone_present:
                if not _discard_owned_cleanup_tombstone(
                    path,
                    directory=directory,
                    identity=identity,
                ):
                    return False
        for path, directory, identity, _present in resource_states:
            quarantine, owned, marker, _tombstone = _cleanup_path_quarantine(
                path,
                directory=directory,
                identity=identity,
            )
            if any(
                os.path.lexists(candidate)
                for candidate in (path, owned, marker, quarantine)
            ):
                return False
        return True
    except (KeyError, OSError, TypeError, ValueError):
        return False
    finally:
        if owns_lock and lock_handle is not None:
            lock_handle.close()


def _reobserve_cleanup_binding_docker_absence(
    binding_record: Mapping[str, object],
) -> bool:
    """Independently recheck a completed binding without issuing an effect."""

    raw_fence = binding_record.get("termination_fence")
    fence = raw_fence if isinstance(raw_fence, Mapping) and raw_fence else None
    docker_bin = str(binding_record.get("docker_bin") or "")
    container_name = str(binding_record.get("container_name") or "")
    if (
        docker_bin not in {"/usr/bin/docker", "/usr/local/bin/docker"}
        or _DOCKER_CONTAINER_NAME_RE.fullmatch(container_name) is None
    ):
        return False
    recovery_config = Path(
        tempfile.mkdtemp(prefix="aseh-docker-completion-recheck-")
    )
    recovery_config.chmod(0o700)
    try:
        _remove_exact_docker_container(
            docker_bin=docker_bin,
            docker_config=recovery_config,
            container_name=container_name,
            settle_for_creation=False,
            deadline=time.monotonic() + _DOCKER_CLEANUP_TIMEOUT_SECONDS,
            termination_fence=fence,
            issue_removal=False,
        )
        return True
    except (OSError, TypeError, ValueError):
        return False
    finally:
        shutil.rmtree(recovery_config, ignore_errors=True)


def _finalize_verified_cleanup_completion(
    *,
    binding_path: Path,
    binding_identity: Mapping[str, int],
    binding_record: Mapping[str, object],
    expected_lifecycle: Mapping[str, object] | None = None,
    terminal_cleanup_store: object | None = None,
    terminal_cleanup_reservation: object | None = None,
) -> bool:
    """Retire an independently admitted binding after Docker absence.

    Callers invoke this only after they have established exact Docker identity
    and kernel-scope quiescence. Protected provider cleanup first commits the
    exact removal intent through its existing terminal attempt CAS. The stable
    binding lock then covers inode quarantine, completion publication, the
    monotonic CAS completion, and retirement of the original binding inode.
    Unscoped cleanup can complete the uninterrupted live path, but never
    admits a marker-only crash gap.
    """

    lock_handle: _DockerBindingLock | None = None
    try:
        scoped_cleanup = bool(
            terminal_cleanup_store is not None
            and terminal_cleanup_reservation is not None
        )
        if (terminal_cleanup_store is None) != (
            terminal_cleanup_reservation is None
        ):
            return False

        def observed_terminal() -> object | None:
            if not scoped_cleanup:
                return None
            logical_attempt_id = str(
                getattr(
                    terminal_cleanup_reservation,
                    "logical_attempt_id",
                    "",
                )
                or ""
            )
            observe = getattr(terminal_cleanup_store, "observe", None)
            if not logical_attempt_id or not callable(observe):
                raise ValueError("terminal cleanup CAS observer is unavailable")
            observed = observe(logical_attempt_id)
            if (
                observed is None
                or getattr(observed, "state", "") != "terminal"
                or getattr(observed, "reservation_id", "")
                != getattr(terminal_cleanup_reservation, "reservation_id", "")
                or getattr(observed, "terminal_cleanup_authority", None)
                != getattr(
                    terminal_cleanup_reservation,
                    "terminal_cleanup_authority",
                    None,
                )
            ):
                raise ValueError("terminal cleanup CAS authority changed")
            return observed

        terminal = observed_terminal()
        terminal_authority = (
            getattr(terminal, "terminal_cleanup_authority", None)
            if terminal is not None
            else None
        )
        lock_handle = _docker_binding_lock_descriptor(binding_path)
        completion_path = _cleanup_completion_path(binding_path)
        authority_path = _cleanup_authority_path(binding_path)
        if lock_handle.exists(binding_path) and lock_handle.exists(authority_path):
            # Recover only the exact link-before-unlink retirement crash.
            # The completion bytes and (when scoped) terminal CAS completion
            # must already be authoritative; a second name alone grants no
            # ability to issue or replay any removal effect.
            expected_completion = _cleanup_completion_value(
                binding_path=binding_path,
                binding_identity=binding_identity,
                binding_record=binding_record,
                terminal_cleanup_authority=terminal_authority,
                binding_lock=lock_handle,
            )
            completion = lock_handle.read(completion_path)
            cleanup_intent = (
                completion.get("cleanup_intent")
                if isinstance(completion, Mapping)
                else None
            )
            progress = (
                getattr(terminal, "terminal_cleanup_progress", None)
                if terminal is not None
                else None
            )
            if (
                completion != expected_completion
                or not isinstance(cleanup_intent, Mapping)
                or (
                    scoped_cleanup
                    and not _cleanup_progress_matches(
                        progress,
                        intent=cleanup_intent,
                        completion_id=str(
                            completion.get("completion_id") or ""
                        ),
                    )
                )
                or not _reobserve_cleanup_binding_docker_absence(
                    binding_record
                )
                or not _retire_cleanup_binding_authority(
                    binding_path,
                    binding_identity=binding_identity,
                    binding_record=binding_record,
                    binding_lock=lock_handle,
                )
            ):
                return False
            return _recover_cleanup_completion(
                completion_path,
                expected_lifecycle=expected_lifecycle,
                docker_absence_verified=True,
                binding_lock=lock_handle,
                terminal_cleanup_progress=progress,
            )
        current = lock_handle.read(binding_path)
        if current is None:
            expected_completion = _cleanup_completion_value(
                binding_path=binding_path,
                binding_identity=binding_identity,
                binding_record=binding_record,
                terminal_cleanup_authority=terminal_authority,
                binding_lock=lock_handle,
            )
            if (
                lock_handle.read(completion_path)
                != expected_completion
                or not _cleanup_binding_authority_present(
                    binding_path,
                    binding_identity=binding_identity,
                    binding_record=binding_record,
                    binding_lock=lock_handle,
                )
                or not _reobserve_cleanup_binding_docker_absence(binding_record)
            ):
                return False
            return _recover_cleanup_completion(
                completion_path,
                expected_lifecycle=expected_lifecycle,
                docker_absence_verified=True,
                binding_lock=lock_handle,
                terminal_cleanup_progress=(
                    getattr(terminal, "terminal_cleanup_progress", None)
                    if terminal is not None
                    else None
                ),
            )
        if (
            current != dict(binding_record)
            or lock_handle.path_identity(binding_path)
            != dict(binding_identity)
            or not _reobserve_cleanup_binding_docker_absence(binding_record)
        ):
            return False
        path_identities = binding_record.get("path_identities")
        if not isinstance(path_identities, dict):
            return False
        expected_completion = _cleanup_completion_value(
            binding_path=binding_path,
            binding_identity=binding_identity,
            binding_record=binding_record,
            terminal_cleanup_authority=terminal_authority,
            binding_lock=lock_handle,
        )
        cleanup_intent = expected_completion.get("cleanup_intent")
        if not isinstance(cleanup_intent, Mapping):
            return False
        if scoped_cleanup:
            prior_progress = getattr(
                terminal,
                "terminal_cleanup_progress",
                None,
            )
            if not prior_progress:
                # The CAS intent must precede every inode mutation. If a
                # same-UID writer renamed a credential and forged its public
                # tombstone first, do not convert that absence into cleanup
                # authority.
                initial_resources = cleanup_intent.get("resources")
                if not isinstance(initial_resources, list):
                    return False
                for item in initial_resources:
                    if (
                        not isinstance(item, Mapping)
                        or not isinstance(item.get("identity"), Mapping)
                        or not isinstance(item.get("directory"), bool)
                    ):
                        return False
                    initial_path = Path(str(item.get("path") or ""))
                    try:
                        metadata = os.lstat(initial_path)
                    except OSError:
                        return False
                    quarantine, owned, marker, _tombstone = (
                        _cleanup_path_quarantine(
                            initial_path,
                            directory=bool(item["directory"]),
                            identity=item["identity"],  # type: ignore[arg-type]
                        )
                    )
                    if (
                        not _owned_cleanup_path_matches(
                            metadata,
                            directory=bool(item["directory"]),
                            identity=item["identity"],  # type: ignore[arg-type]
                        )
                        or any(
                            os.path.lexists(candidate)
                            for candidate in (owned, marker, quarantine)
                        )
                    ):
                        return False
            commit_intent = getattr(
                terminal_cleanup_store,
                "commit_terminal_cleanup_intent",
                None,
            )
            if not callable(commit_intent):
                return False
            terminal = commit_intent(terminal, intent=cleanup_intent)
            progress = getattr(terminal, "terminal_cleanup_progress", None)
            if not _cleanup_progress_matches(
                progress,
                intent=cleanup_intent,
            ):
                return False
        else:
            progress = None
        completion_already_exact = bool(
            lock_handle.read(completion_path)
            == expected_completion
        )
        completion_already_committed = bool(
            completion_already_exact
            and (
                not scoped_cleanup
                or _cleanup_progress_matches(
                    progress,
                    intent=cleanup_intent,
                    completion_id=str(
                        expected_completion.get("completion_id") or ""
                    ),
                )
            )
        )
        intent_resources = cleanup_intent.get("resources")
        if not isinstance(intent_resources, list) or len(intent_resources) != 3:
            return False
        admitted_tombstones = {
            str(item.get("name") or ""): str(item.get("tombstone_id") or "")
            for item in intent_resources
            if isinstance(item, Mapping)
        }
        for name, field, directory in (
            ("prompt_path", "prompt_path", False),
            ("provider_home", "provider_home", True),
            ("lease_root", "lease_root", True),
        ):
            identity = path_identities.get(name)
            path = Path(str(binding_record.get(field) or ""))
            if not isinstance(identity, dict):
                return False
            quarantine, owned, marker, _tombstone = _cleanup_path_quarantine(
                path,
                directory=directory,
                identity=identity,
            )
            fully_absent = not any(
                os.path.lexists(candidate)
                for candidate in (path, owned, marker, quarantine)
            )
            if fully_absent and completion_already_committed:
                continue
            if not _remove_owned_cleanup_path(
                path,
                directory=directory,
                identity=identity,
                admitted_tombstone_id=(
                    admitted_tombstones.get(name, "")
                    if scoped_cleanup
                    else ""
                ),
            ):
                return False
        completion_path = _publish_cleanup_completion(
            binding_path=binding_path,
            binding_identity=binding_identity,
            binding_record=binding_record,
            terminal_cleanup_authority=terminal_authority,
            binding_lock=lock_handle,
        )
        completion = lock_handle.read(completion_path)
        if completion != expected_completion:
            return False
        if scoped_cleanup:
            commit_completion = getattr(
                terminal_cleanup_store,
                "commit_terminal_cleanup_completion",
                None,
            )
            if not callable(commit_completion):
                return False
            terminal = commit_completion(
                terminal,
                intent_id=str(cleanup_intent.get("intent_id") or ""),
                completion_id=str(completion.get("completion_id") or ""),
            )
            progress = getattr(terminal, "terminal_cleanup_progress", None)
            if not _cleanup_progress_matches(
                progress,
                intent=cleanup_intent,
                completion_id=str(completion.get("completion_id") or ""),
            ):
                return False
        if not _recover_cleanup_completion(
            completion_path,
            expected_lifecycle=expected_lifecycle,
            docker_absence_verified=True,
            binding_lock=lock_handle,
            terminal_cleanup_progress=progress,
        ):
            return False
        if not _retire_cleanup_binding_authority(
            binding_path,
            binding_identity=binding_identity,
            binding_record=binding_record,
            binding_lock=lock_handle,
        ):
            return False
        return _recover_cleanup_completion(
            completion_path,
            expected_lifecycle=expected_lifecycle,
            docker_absence_verified=True,
            binding_lock=lock_handle,
            terminal_cleanup_progress=progress,
        )
    except (FileNotFoundError, OSError, TypeError, ValueError):
        return False
    finally:
        if lock_handle is not None:
            lock_handle.close()


def _docker_cleanup_control_socket(descriptor: int) -> socket.socket:
    """Take ownership of one connected, procfs-nonreopenable Unix socket."""

    if descriptor < 3:
        raise ValueError("Docker cleanup control descriptor is invalid")
    try:
        metadata = os.fstat(descriptor)
    except OSError as exc:
        raise ValueError("Docker cleanup control descriptor is unavailable") from exc
    if not stat.S_ISSOCK(metadata.st_mode):
        raise ValueError("Docker cleanup control descriptor is not a socket")
    channel = socket.socket(fileno=descriptor)
    try:
        if (
            channel.family != socket.AF_UNIX
            or channel.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE)
            != socket.SOCK_STREAM
        ):
            raise ValueError("Docker cleanup control socket has the wrong type")
        channel.getpeername()
        channel.settimeout(None)
        os.set_inheritable(descriptor, False)
        return channel
    except BaseException:
        channel.close()
        raise


def _docker_control_peer_credentials(
    channel: socket.socket,
) -> tuple[int, int, int]:
    try:
        payload = channel.getsockopt(
            socket.SOL_SOCKET,
            socket.SO_PEERCRED,
            struct.calcsize("3i"),
        )
        pid, uid, gid = struct.unpack("3i", payload)
    except (OSError, struct.error) as exc:
        raise ValueError("Docker cleanup control peer is unavailable") from exc
    if pid <= 0 or uid < 0 or gid < 0:
        raise ValueError("Docker cleanup control peer is invalid")
    return pid, uid, gid


def _docker_cleanup_watchdog_launcher_main(argv: Sequence[str]) -> int:
    """Detach the existing cleanup watchdog beyond the fenced runner tree.

    The supervisor's strict shutdown path freezes and kills every descendant
    before it releases a lane.  A mere ``setsid`` watchdog is still a
    descendant and therefore cannot run its exact-container cleanup handler.
    This single-purpose launcher double-forks, and reports readiness only
    after the watchdog has been reparented to PID 1.  If that invariant is not
    available (for example under an unexpected subreaper), provider launch
    fails closed instead of creating an unreapable external effect.

    The inherited control capability is a connected Unix socket, not a pipe.
    Linux refuses reopening socket FDs through ``/proc/<pid>/fd``; combined
    with the qualified ptrace prerequisite this prevents a same-UID peer from
    injecting command or result bytes into the post-exec channel.
    """

    items = list(argv)
    if (
        len(items) < 4
        or items[0] != "--control-fd"
        or items[2] != _DOCKER_CLEANUP_WATCHDOG_ARG
    ):
        return 2
    channel: socket.socket | None = None
    try:
        channel = _docker_cleanup_control_socket(int(items[1]))
        runner_index = items.index("--runner-pid", 3)
        runner_pid = int(items[runner_index + 1])
        peer_pid, peer_uid, peer_gid = _docker_control_peer_credentials(channel)
    except (IndexError, OSError, ValueError):
        if channel is not None:
            channel.close()
        return 2
    if (
        peer_pid != runner_pid
        or peer_uid != os.geteuid()
        or peer_gid != os.getegid()
    ):
        channel.close()
        return 2

    launcher_pid = os.getpid()
    try:
        child_pid = os.fork()
    except OSError:
        channel.close()
        return 2
    if child_pid:
        channel.close()
        return 0

    try:
        os.setsid()
    except OSError:
        channel.close()
        return 2
    deadline = time.monotonic() + 2.0
    while os.getppid() == launcher_pid and time.monotonic() < deadline:
        time.sleep(0.005)
    if os.getppid() != 1:
        # The nearest subreaper is still inside an unknown ownership tree.
        # Refuse to claim kill-safe cleanup rather than weakening strict
        # descendant fencing or silently leaking a provider effect.
        channel.close()
        return 2
    return _docker_cleanup_watchdog_main(items[3:], control_socket=channel)


def _docker_cleanup_watchdog_main(
    argv: Sequence[str],
    *,
    control_socket: socket.socket | None = None,
) -> int:
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
    parser.add_argument("--logical-attempt-id", default="")
    parser.add_argument("--provider-attempt-store", default="")
    parser.add_argument("--provider-attempt-store-identity", default="")
    parser.add_argument("--cleanup-binding-record", default="")
    parser.add_argument("--runner-pid", type=int, required=True)
    parser.add_argument("--runner-start-ticks", type=int, required=True)
    parser.add_argument("--control-fd", type=int, default=-1)
    args = parser.parse_args(list(argv))

    try:
        if control_socket is None:
            control_socket = _docker_cleanup_control_socket(args.control_fd)
        elif args.control_fd >= 3:
            raise ValueError("Docker cleanup control descriptor is duplicated")
        peer_pid, peer_uid, peer_gid = _docker_control_peer_credentials(
            control_socket
        )
        if (
            peer_pid != args.runner_pid
            or peer_uid != os.geteuid()
            or peer_gid != os.getegid()
            or _runner_process_start_ticks(peer_pid) != args.runner_start_ticks
        ):
            raise ValueError("Docker cleanup control peer identity drifted")
    except (OSError, ValueError):
        if control_socket is not None:
            control_socket.close()
        return 2

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
    cleanup_binding_record = (
        Path(args.cleanup_binding_record).absolute()
        if args.cleanup_binding_record
        else None
    )
    observation_values = (
        args.logical_attempt_id,
        args.provider_attempt_store,
        args.provider_attempt_store_identity,
    )
    observation_configured = all(observation_values)
    if any(observation_values) and not observation_configured:
        return 2
    attempt_observer = None
    if observation_configured:
        try:
            from ipfs_accelerate_py.agent_supervisor.control.provider_attempt_store import (
                DurableProviderAttemptCAS,
                ProviderAttemptStoreError,
            )

            attempt_observer = DurableProviderAttemptCAS(
                args.provider_attempt_store,
                expected_directory_identity=(
                    args.provider_attempt_store_identity
                ),
                create_if_missing=False,
            )
            # Validate the stable logical identifier before provider creation.
            attempt_observer.observe(args.logical_attempt_id)
        except (OSError, ProviderAttemptStoreError, ValueError):
            return 2
    cleanup_effect_observation = (
        {
            "logical_attempt_id": args.logical_attempt_id,
            "provider_attempt_store": args.provider_attempt_store,
            "provider_attempt_store_identity": (
                args.provider_attempt_store_identity
            ),
        }
        if observation_configured
        else {}
    )
    cas_marker = lease_root / "cas-owned"
    terminal_marker = lease_root / "cas-terminal"
    expected_container_prefix = f"ipfs-accelerate-{args.provider}-"
    try:
        _cleanup_root, _cleanup_root_identity = (
            _validated_docker_cleanup_root(
                lease_root=lease_root,
                provider_home=provider_home,
                prompt_path=prompt_path,
            )
        )
    except ValueError:
        return 2
    if (
        docker_path not in {Path("/usr/bin/docker"), Path("/usr/local/bin/docker")}
        or docker_path.name not in {"docker", "docker.exe"}
        or docker_stat.st_uid != 0
        or docker_stat.st_mode & 0o022
        or _DOCKER_CONTAINER_NAME_RE.fullmatch(args.container_name) is None
        or not args.container_name.startswith(expected_container_prefix)
        or not lease_root.name.startswith(
            f"asref-{args.provider}-container-"
        )
        or cidfile.parent != lease_root
        or cidfile.name != "container.cid"
        or not docker_config.is_dir()
        or not provider_home.name.startswith(
            f"asref-{args.provider}-home-"
        )
        or not prompt_path.name.startswith("asref-grok-prompt-")
        or (
            cleanup_binding_record is not None
            and (
                cleanup_binding_record.parent.name
                != _DOCKER_CLEANUP_BINDING_DIRECTORY
                or re.fullmatch(
                    r"[0-9a-f]{64}\.json",
                    cleanup_binding_record.name,
                )
                is None
            )
        )
    ):
        return 2
    try:
        cleanup_path_identities: dict[str, Mapping[str, int]] = {
            "docker_config": _cleanup_path_identity(
                docker_config,
                directory=True,
            ),
            "lease_root": _cleanup_path_identity(
                lease_root,
                directory=True,
            ),
            "prompt_path": _cleanup_path_identity(
                prompt_path,
                directory=False,
            ),
            "provider_home": _cleanup_path_identity(
                provider_home,
                directory=True,
            ),
        }
    except ValueError:
        return 2
    cleanup_binding_identity: Mapping[str, int] | None = None
    cleanup_binding_value: Mapping[str, object] | None = None

    # A clean marker means docker-run returned, but the final rm remains a
    # defensive idempotent action.  Empty input means the runner was killed;
    # retry briefly to cover a daemon-side container-creation race.
    cleanup_started = False
    cleanup_succeeded = False
    cleanup_failed = False
    create_request_received = False
    create_worker_active = False
    create_environment: dict[str, str] | None = None
    private_create_command_id = ""
    private_create_command_body: dict[str, object] | None = None
    private_result_open = True

    def close_private_result() -> None:
        nonlocal private_result_open
        if not private_result_open:
            return
        try:
            control_socket.shutdown(socket.SHUT_WR)
        except OSError:
            pass
        private_result_open = False

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

    def durable_cas_state() -> str:
        """Observe only the exact existing attempt CAS; never mutate it."""

        return _observed_provider_attempt_cleanup_state(
            attempt_observer,
            logical_attempt_id=args.logical_attempt_id,
            lease_root=lease_root,
            docker_config=docker_config,
            container_name=args.container_name,
            watchdog_pid=os.getpid(),
            watchdog_start_ticks=_runner_process_start_ticks(os.getpid()),
        )

    def create_journal() -> dict[str, object] | None:
        return _validated_docker_create_journal(
            lease_root=lease_root,
            provider=args.provider,
            docker_bin=str(docker_path),
            docker_config=docker_config,
            container_name=args.container_name,
            cidfile=cidfile,
        )

    def admit_cleanup_authority(
        journal: Mapping[str, object] | None,
    ) -> bool:
        nonlocal cleanup_binding_identity, cleanup_binding_value
        if cleanup_binding_record is None:
            return True
        try:
            raw_binding = _read_private_control_record(
                cleanup_binding_record.parent,
                cleanup_binding_record.name,
            )
            if raw_binding is None:
                return False
            binding_state = str(raw_binding.get("binding_state") or "")
            if binding_state == "prepared_no_dispatch":
                if journal is not None and journal.get("state") != "prepared":
                    return False
                create_command_id = ""
                create_cwd = None
                create_environment_id = ""
                termination_fence: Mapping[str, object] = {}
            elif binding_state == "command_bound" and journal is not None:
                create_command_id = str(journal["command_id"])
                create_cwd = Path(str(journal["cwd"]))
                create_environment_id = str(journal["environment_id"])
                raw_termination_fence = raw_binding.get("termination_fence")
                if not isinstance(raw_termination_fence, Mapping):
                    return False
                termination_fence = raw_termination_fence
                if (
                    not private_create_command_id
                    or private_create_command_body is None
                    or create_command_id != private_create_command_id
                    or any(
                        journal.get(name) != expected
                        for name, expected in private_create_command_body.items()
                    )
                ):
                    return False
            else:
                return False
            value = _validated_cleanup_binding_record(
                cleanup_binding_record,
                provider=args.provider,
                docker_bin=str(docker_path),
                docker_config=docker_config,
                container_name=args.container_name,
                cidfile=cidfile,
                lease_root=lease_root,
                provider_home=provider_home,
                prompt_path=prompt_path,
                effect_observation=cleanup_effect_observation,
                binding_state=binding_state,
                runner_pid=args.runner_pid,
                runner_start_ticks=args.runner_start_ticks,
                watchdog_pid=os.getpid(),
                watchdog_start_ticks=_runner_process_start_ticks(os.getpid()),
                create_command_id=create_command_id,
                create_cwd=create_cwd,
                create_environment_id=create_environment_id,
                termination_fence=termination_fence,
            )
            if value.get("path_identities") != cleanup_path_identities:
                return False
            observed_binding = _cleanup_path_identity(
                cleanup_binding_record,
                directory=False,
            )
        except (KeyError, ValueError):
            return False
        expected_upgrade = bool(
            cleanup_binding_value is not None
            and (
                (
                    cleanup_binding_value.get("binding_state")
                    == "prepared_no_dispatch"
                    and value.get("binding_state") == "command_bound"
                )
                or (
                    cleanup_binding_value.get("binding_state")
                    == value.get("binding_state")
                    == "command_bound"
                    and cleanup_binding_value.get("termination_fence") == {}
                    and bool(value.get("termination_fence"))
                )
            )
        )
        if (
            cleanup_binding_identity is not None
            and observed_binding != cleanup_binding_identity
            and not expected_upgrade
        ):
            return False
        cleanup_binding_identity = observed_binding
        cleanup_binding_value = value
        return True

    def publish_command_bound_cleanup_authority(
        journal: Mapping[str, object],
    ) -> None:
        """Let the pipe-bound watchdog perform the sole binding upgrade."""

        nonlocal cleanup_binding_identity, cleanup_binding_value
        if (
            cleanup_binding_record is None
            or cleanup_binding_identity is None
            or cleanup_binding_value is None
            or cleanup_binding_value.get("binding_state")
            != "prepared_no_dispatch"
            or not private_create_command_id
            or private_create_command_body is None
            or journal.get("command_id") != private_create_command_id
            or any(
                journal.get(name) != expected
                for name, expected in private_create_command_body.items()
            )
        ):
            raise ValueError("Docker create command lacks private pipe authority")
        lock_handle = _docker_binding_lock_descriptor(cleanup_binding_record)
        try:
            prepared_value = _validated_cleanup_binding_record(
                cleanup_binding_record,
                provider=args.provider,
                docker_bin=str(docker_path),
                docker_config=docker_config,
                container_name=args.container_name,
                cidfile=cidfile,
                lease_root=lease_root,
                provider_home=provider_home,
                prompt_path=prompt_path,
                effect_observation=cleanup_effect_observation,
                binding_state="prepared_no_dispatch",
                runner_pid=args.runner_pid,
                runner_start_ticks=args.runner_start_ticks,
                watchdog_pid=os.getpid(),
                watchdog_start_ticks=_runner_process_start_ticks(os.getpid()),
                control_directory_fd=lock_handle.directory_fd,
            )
            prepared_identity = lock_handle.path_identity(cleanup_binding_record)
            if (
                prepared_value != cleanup_binding_value
                or prepared_identity != cleanup_binding_identity
            ):
                raise ValueError(
                    "Docker cleanup pre-dispatch binding changed"
                )
            command_value = _docker_cleanup_binding_value(
                binding_state="command_bound",
                provider=args.provider,
                docker_bin=str(docker_path),
                container_name=args.container_name,
                lease_root=lease_root,
                docker_config=docker_config,
                cidfile=cidfile,
                provider_home=provider_home,
                prompt_path=prompt_path,
                effect_observation=cleanup_effect_observation,
                path_identities=cleanup_path_identities,
                binding_path=cleanup_binding_record,
                runner_pid=args.runner_pid,
                runner_start_ticks=args.runner_start_ticks,
                watchdog_pid=os.getpid(),
                watchdog_start_ticks=_runner_process_start_ticks(os.getpid()),
                create_command_id=private_create_command_id,
                create_cwd=Path(str(private_create_command_body["cwd"])),
                create_environment_id=str(
                    private_create_command_body["environment_id"]
                ),
            )
            lock_handle.write(
                cleanup_binding_record,
                command_value,
                replace_existing=True,
            )
            cleanup_binding_value = _validated_cleanup_binding_record(
                cleanup_binding_record,
                provider=args.provider,
                docker_bin=str(docker_path),
                docker_config=docker_config,
                container_name=args.container_name,
                cidfile=cidfile,
                lease_root=lease_root,
                provider_home=provider_home,
                prompt_path=prompt_path,
                effect_observation=cleanup_effect_observation,
                binding_state="command_bound",
                runner_pid=args.runner_pid,
                runner_start_ticks=args.runner_start_ticks,
                watchdog_pid=os.getpid(),
                watchdog_start_ticks=_runner_process_start_ticks(os.getpid()),
                create_command_id=private_create_command_id,
                create_cwd=Path(str(private_create_command_body["cwd"])),
                create_environment_id=str(
                    private_create_command_body["environment_id"]
                ),
                control_directory_fd=lock_handle.directory_fd,
            )
            cleanup_binding_identity = lock_handle.path_identity(
                cleanup_binding_record
            )
        finally:
            lock_handle.close()

    def publish_termination_fence(
        journal: Mapping[str, object],
    ) -> Mapping[str, object]:
        """CAS-publish exact init/namespace/cgroup identity before Docker rm."""

        nonlocal cleanup_binding_identity, cleanup_binding_value
        if journal.get("state") != "create_observed":
            raise ValueError("Docker termination fence requires observed create")
        try:
            container_id = cidfile.read_text(encoding="ascii").strip()
        except (OSError, UnicodeError) as exc:
            raise ValueError("Docker termination CID is unavailable") from exc
        image_id = str(journal.get("image_id") or "")
        if (
            re.fullmatch(r"[0-9a-f]{64}", container_id) is None
            or re.fullmatch(r"sha256:[0-9a-f]{64}", image_id) is None
        ):
            raise ValueError("Docker termination identity is invalid")
        existing = (
            cleanup_binding_value.get("termination_fence")
            if cleanup_binding_value is not None
            else None
        )
        if isinstance(existing, Mapping) and existing:
            return _validated_docker_termination_fence(
                existing,
                provider=args.provider,
                container_name=args.container_name,
                expected_container_id=container_id,
                expected_image_id=image_id,
            )
        fence = _attest_exact_docker_execution(
            docker_bin=str(docker_path),
            docker_config=docker_config,
            provider=args.provider,
            container_name=args.container_name,
            container_id=container_id,
            image_id=image_id,
            timeout=2.0,
        )
        if cleanup_binding_record is None:
            return fence
        if (
            cleanup_binding_identity is None
            or cleanup_binding_value is None
            or cleanup_binding_value.get("binding_state") != "command_bound"
            or cleanup_binding_value.get("termination_fence") != {}
        ):
            raise ValueError("Docker termination binding was not admitted")
        cleanup_binding_value, cleanup_binding_identity = (
            _publish_docker_termination_fence_binding(
                record_path=cleanup_binding_record,
                expected_record_id=str(cleanup_binding_value["record_id"]),
                expected_identity=cleanup_binding_identity,
                provider=args.provider,
                docker_bin=str(docker_path),
                docker_config=docker_config,
                container_name=args.container_name,
                cidfile=cidfile,
                lease_root=lease_root,
                provider_home=provider_home,
                prompt_path=prompt_path,
                effect_observation=cleanup_effect_observation,
                runner_pid=args.runner_pid,
                runner_start_ticks=args.runner_start_ticks,
                watchdog_pid=os.getpid(),
                watchdog_start_ticks=_runner_process_start_ticks(os.getpid()),
                create_command_id=str(journal["command_id"]),
                create_cwd=Path(str(journal["cwd"])),
                create_environment_id=str(journal["environment_id"]),
                termination_fence=fence,
            )
        )
        return fence

    def run_durable_create() -> None:
        nonlocal create_request_received, create_worker_active
        if create_request_received:
            raise ValueError("Docker create worker received duplicate dispatch")
        create_request_received = True
        journal = create_journal()
        if (
            journal is None
            or journal.get("state") != "create_armed"
            or create_environment is None
            or not private_create_command_id
            or private_create_command_body is None
            or journal.get("command_id") != private_create_command_id
            or any(
                journal.get(name) != expected
                for name, expected in private_create_command_body.items()
            )
        ):
            raise ValueError("Docker create worker lacks one armed journal")
        environment_id, _payload, admitted_environment = (
            _docker_create_environment_payload(create_environment)
        )
        create_cwd = Path(str(journal.get("cwd") or ""))
        if journal.get("environment_id") != environment_id:
            raise ValueError("Docker create environment identity drifted")
        if cleanup_binding_record is None:
            raise ValueError("Docker create worker lacks a durable binding")
        publish_command_bound_cleanup_authority(journal)
        if not admit_cleanup_authority(journal):
            raise ValueError("Docker create worker lacks cleanup authority")
        create_worker_active = True
        request_dispatched = False
        forced_kill = False
        try:
            try:
                (
                    journal,
                    returncode,
                    stdout,
                    stderr,
                    request_dispatched,
                    forced_kill,
                ) = _run_fenced_docker_create_issuer(
                    journal,
                    lease_root=lease_root,
                    cwd=create_cwd,
                    environment=admitted_environment,
                )
            except (OSError, TypeError, ValueError):
                stdout = b""
                stderr = b""
                returncode = 125
                current = create_journal()
                if current is not None:
                    journal = current
                    if current.get("state") in {
                        "create_inflight",
                        "create_outcome_unknown",
                    }:
                        # The issuer crossed the durable inflight transition.
                        # If its result path then failed, absence at one instant
                        # cannot prove Docker did not accept the request.
                        request_dispatched = True
                        forced_kill = True
        finally:
            create_worker_active = False
        observed = bool(
            request_dispatched
            and not forced_kill
            and returncode == 0
            and len(stdout) <= _DOCKER_INSPECTION_MAX_BYTES
            and len(stderr) <= _DOCKER_INSPECTION_MAX_BYTES
        )
        if not observed and (
            len(stdout) > _DOCKER_INSPECTION_MAX_BYTES
            or len(stderr) > _DOCKER_INSPECTION_MAX_BYTES
        ):
            stdout = b""
            stderr = b""
            returncode = 125
        # Only a dispatch that never crossed the private gate is a proven
        # failed create.  A natural nonzero CLI exit (and especially a forced
        # kill) can follow daemon acceptance, so fixed-time name absence must
        # not turn it into a releasable observation.
        failed_observed = bool(not request_dispatched and not observed)
        terminal_journal = _transition_docker_create_journal(
            journal,
            lease_root=lease_root,
            state=(
                "create_observed"
                if observed
                else (
                    "create_failed_observed"
                    if failed_observed
                    else "create_outcome_unknown"
                )
            ),
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
        )
        _write_docker_create_private_result(
            control_socket,
            terminal_journal,
        )
        close_private_result()

    def cleanup(*, settle_for_creation: bool) -> bool:
        nonlocal cleanup_started, cleanup_succeeded, cleanup_failed
        if cleanup_started:
            return cleanup_succeeded
        try:
            journal = create_journal()
        except ValueError:
            cleanup_failed = True
            return False
        if not admit_cleanup_authority(journal):
            cleanup_failed = True
            return False
        if journal is not None and journal.get("state") in {
            "create_armed",
            "create_inflight",
            "create_outcome_unknown",
        }:
            cleanup_failed = True
            return False
        cleanup_started = True
        try:
            termination_fence = (
                publish_termination_fence(journal)
                if journal is not None
                and journal.get("state") == "create_observed"
                else None
            )
            issue_removal = False
            if termination_fence is not None:
                if (
                    cleanup_binding_record is None
                    or cleanup_binding_value is None
                    or cleanup_binding_identity is None
                ):
                    raise ValueError(
                        "Docker rm lacks a durable dispatch authority"
                    )
                issue_removal = _arm_docker_removal_once(
                    binding_path=cleanup_binding_record,
                    expected_binding_identity=cleanup_binding_identity,
                    binding_record=cleanup_binding_value,
                    termination_fence=termination_fence,
                )
            _remove_exact_docker_container(
                docker_bin=str(docker_path),
                docker_config=docker_config,
                container_name=args.container_name,
                settle_for_creation=settle_for_creation,
                termination_fence=termination_fence,
                issue_removal=issue_removal,
            )
        except ValueError:
            cleanup_failed = True
            cleanup_started = False
            return False
        cleanup_succeeded = True
        cleanup_failed = False
        return True

    def terminate_watchdog(signum: int, _frame: object) -> None:
        # The supervisor deliberately terminates separately owned descendant
        # process groups before the runner group.  Reap synchronously here so
        # that ordering cannot strand the runner-owned workspace mount.
        if cleanup_started:
            # EOF and lifecycle TERM can arrive together after a strict
            # ancestry fence.  Never turn the second trigger into a
            # re-entrant false failure while the first exact Docker rm is
            # still in progress; that in-flight cleanup retains authority and
            # reports its own terminal result.
            return
        try:
            journal = create_journal()
        except ValueError:
            return
        if create_worker_active or (
            journal is not None
            and journal.get("state")
            in {
                "create_armed",
                "create_inflight",
                "create_outcome_unknown",
            }
        ):
            # The detached worker is the sole Docker-create issuer.  Never
            # kill it or free its exact name while that request lacks a
            # definite successful observation.
            return
        durable_state = durable_cas_state()
        if (
            (cas_owned() or durable_state in {"owned", "unknown"})
            and not (cas_terminal() or durable_state == "terminal")
        ):
            # Recovery owns every path needed to inspect/start the inert or
            # running exact container.  Supervisor shutdown must not delete
            # those bind/config inputs before a durable terminal transition.
            return
        # A terminal marker is durable accounting authority.  Reap the exact
        # container synchronously *before* finally removes the Docker config
        # and bind sources.  This covers marker->SIGTERM interleavings where
        # the polling loop has not observed the terminal transition yet.
        if observation_configured and durable_state == "absent":
            # A signal can arrive while the live runner is still inside the
            # durable claim.  Wait for runner EOF, then observe the final CAS.
            return
        if not cleanup(settle_for_creation=journal is not None):
            # Preserve the private cleanup inputs and return a distinct
            # failure status.  The lifecycle owner independently verifies
            # the exact lease root and container absence, so a detached
            # proxy cannot convert this failed reaper into lane release.
            raise SystemExit(125)
        raise SystemExit(128 + signum)

    try:
        signal.signal(signal.SIGTERM, terminate_watchdog)
        signal.signal(signal.SIGINT, terminate_watchdog)
        if cleanup_binding_record is not None:
            try:
                runner_birth = read_process_birth(args.runner_pid)
                if (
                    runner_birth is None
                    or runner_birth.start_time_ticks != args.runner_start_ticks
                    or runner_birth.boot_id
                    != Path(
                        "/proc/sys/kernel/random/boot_id"
                    ).read_text(encoding="ascii").strip()
                ):
                    raise ValueError("Docker cleanup runner birth changed")
                cleanup_binding_value = _docker_cleanup_binding_value(
                    binding_state="prepared_no_dispatch",
                    provider=args.provider,
                    docker_bin=str(docker_path),
                    container_name=args.container_name,
                    lease_root=lease_root,
                    docker_config=docker_config,
                    cidfile=cidfile,
                    provider_home=provider_home,
                    prompt_path=prompt_path,
                    effect_observation=cleanup_effect_observation,
                    path_identities=cleanup_path_identities,
                    binding_path=cleanup_binding_record,
                    runner_pid=args.runner_pid,
                    runner_start_ticks=args.runner_start_ticks,
                    watchdog_pid=os.getpid(),
                    watchdog_start_ticks=_runner_process_start_ticks(
                        os.getpid()
                    ),
                )
                _write_private_control_record(
                    cleanup_binding_record.parent,
                    cleanup_binding_record.name,
                    cleanup_binding_value,
                    replace_existing=False,
                )
                cleanup_binding_identity = _cleanup_path_identity(
                    cleanup_binding_record,
                    directory=False,
                )
            except (OSError, ValueError):
                return 2
        sealed_match = re.fullmatch(r"/proc/self/fd/([0-9]+)", sys.argv[0])
        if sealed_match is not None:
            sealed_descriptor = int(sealed_match.group(1))
            if (
                sealed_descriptor >= 3
                and sealed_descriptor != control_socket.fileno()
            ):
                try:
                    os.close(sealed_descriptor)
                except OSError:
                    pass
        ready_payload = (
            f"{os.getpid()}:{_runner_process_start_ticks(os.getpid())}\n"
        ).encode("ascii")
        try:
            control_socket.sendall(ready_payload)
        except OSError:
            # Readiness precedes the caller's Docker create boundary, so there
            # is no creation race to settle on a rejected launch.
            cleanup(settle_for_creation=False)
            return 2
        markers = bytearray()
        while True:
            marker = control_socket.recv(1)
            if not marker:
                break
            if marker == b"Q":
                try:
                    if (
                        create_environment is not None
                        or private_create_command_id
                        or private_create_command_body is not None
                        or create_request_received
                    ):
                        raise ValueError(
                            "Docker create private handoff is duplicated"
                        )
                    raw_size = _docker_control_read_exact(control_socket, 8)
                    payload_size = int.from_bytes(raw_size, "big")
                    if not 0 < payload_size <= _DOCKER_CREATE_HANDOFF_MAX_BYTES:
                        raise ValueError(
                            "Docker create private handoff length is invalid"
                        )
                    payload = _docker_control_read_exact(
                        control_socket,
                        payload_size,
                    )
                    decoded = json.loads(
                        payload.decode("utf-8"),
                        object_pairs_hook=_reject_duplicate_control_keys,
                    )
                    if (
                        type(decoded) is not dict
                        or set(decoded)
                        != {
                            "schema",
                            "command_id",
                            "command_body",
                            "environment",
                            "handoff_id",
                        }
                        or decoded.get("schema")
                        != _DOCKER_CREATE_HANDOFF_SCHEMA
                        or not isinstance(decoded.get("command_body"), dict)
                        or not isinstance(decoded.get("environment"), dict)
                    ):
                        raise ValueError(
                            "Docker create private handoff is invalid"
                        )
                    environment_id, _canonical_environment, admitted = (
                        _docker_create_environment_payload(
                            decoded["environment"]
                        )
                    )
                    command_body = decoded["command_body"]
                    argv = command_body.get("argv")
                    if not isinstance(argv, list) or not all(
                        isinstance(item, str) for item in argv
                    ):
                        raise ValueError(
                            "Docker create private command argv is invalid"
                        )
                    command_id, admitted_command_body = (
                        _docker_create_command_identity(
                            provider=args.provider,
                            docker_bin=str(docker_path),
                            docker_config=docker_config,
                            container_name=args.container_name,
                            cidfile=cidfile,
                            cwd=Path(str(command_body.get("cwd") or "")),
                            environment_id=environment_id,
                            expected_image=str(
                                command_body.get("image_id") or ""
                            ),
                            argv=argv,
                        )
                    )
                    canonical = _docker_create_private_handoff_payload(
                        command_id=command_id,
                        command_body=admitted_command_body,
                        environment=admitted,
                    )
                    if (
                        canonical != payload
                        or decoded.get("command_id") != command_id
                        or command_body != admitted_command_body
                        or decoded.get("handoff_id")
                        != _effect_receipt_identity(
                            {
                                key: value
                                for key, value in decoded.items()
                                if key != "handoff_id"
                            }
                        )
                    ):
                        raise ValueError(
                            "Docker create private handoff is noncanonical"
                        )
                    create_environment = admitted
                    private_create_command_id = command_id
                    private_create_command_body = admitted_command_body
                except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
                    journal = create_journal()
                    if journal is not None and journal.get("state") == (
                        "create_armed"
                    ):
                        _transition_docker_create_journal(
                            journal,
                            lease_root=lease_root,
                            state="prepared_abandoned",
                        )
                    return 125
                continue
            if marker == b"D":
                try:
                    run_durable_create()
                except (OSError, ValueError):
                    journal = create_journal()
                    if journal is not None and journal.get("state") == (
                        "create_armed"
                    ):
                        _transition_docker_create_journal(
                            journal,
                            lease_root=lease_root,
                            state="prepared_abandoned",
                        )
                    elif journal is not None and journal.get("state") == (
                        "create_inflight"
                    ):
                        _transition_docker_create_journal(
                            journal,
                            lease_root=lease_root,
                            state="create_outcome_unknown",
                            returncode=125,
                        )
                    # This dispatch can never produce an admitted private
                    # result. Close the sole writer now so the runner observes
                    # EOF immediately instead of waiting its 120-second bound.
                    close_private_result()
                continue
            if marker not in {b"A", b"T", b"C"}:
                return 125
            markers.extend(marker)
        journal = create_journal()
        if (
            journal is not None
            and journal.get("state") == "create_armed"
            and not create_request_received
        ):
            journal = _transition_docker_create_journal(
                journal,
                lease_root=lease_root,
                state="prepared_abandoned",
            )
        if journal is not None and journal.get("state") in {
            "create_inflight",
            "create_outcome_unknown",
        }:
            return 125
        clean_exit = b"C" in markers
        durable_state = durable_cas_state()
        if (
            not cas_owned()
            and durable_state in {"unscoped", "absent", "foreign"}
        ):
            settle_for_creation = bool(
                journal is not None
                and not clean_exit
                and journal.get("state") != "create_failed_observed"
            )
            while not cleanup(
                settle_for_creation=settle_for_creation
            ):
                settle_for_creation = True
                time.sleep(0.25)
        elif cas_terminal() or durable_state == "terminal":
            settle_for_creation = False
            while not cleanup(
                settle_for_creation=settle_for_creation
            ):
                settle_for_creation = True
                time.sleep(0.25)
        else:
            # A durable effect_started claim transfers cleanup priority to
            # recovery.  Container absence or exit is not proof that the
            # provider effect never ran, so the watchdog must preserve the
            # exact container until the CAS terminal record has been written.
            while True:
                durable_state = durable_cas_state()
                if cas_terminal() or durable_state == "terminal":
                    while not cleanup(settle_for_creation=True):
                        time.sleep(0.25)
                    break
                time.sleep(1.0)
    finally:
        # Never destroy the inputs required for a later exact retry unless
        # absence/removal of the receipt-bound container was proven.  A dead
        # reaper with preserved private inputs is recoverable; deleting those
        # inputs after an unverified rm is not.
        cleanup_resources = (
            (
                prompt_path,
                False,
                cleanup_path_identities["prompt_path"],
            ),
            (
                provider_home,
                True,
                cleanup_path_identities["provider_home"],
            ),
            (
                lease_root,
                True,
                cleanup_path_identities["lease_root"],
            ),
        )
        if cleanup_succeeded and cleanup_binding_record is not None:
            try:
                if (
                    cleanup_binding_identity is None
                    or cleanup_binding_value is None
                ):
                    raise ValueError("Docker cleanup binding was not admitted")
                if not _finalize_verified_cleanup_completion(
                    binding_path=cleanup_binding_record,
                    binding_identity=cleanup_binding_identity,
                    binding_record=cleanup_binding_value,
                ):
                    raise ValueError("Docker cleanup completion did not converge")
            except ValueError:
                cleanup_succeeded = False
                cleanup_failed = True
        if cleanup_succeeded and cleanup_binding_record is None:
            for path, directory, identity in cleanup_resources:
                if not _remove_owned_cleanup_path(
                    path,
                    directory=directory,
                    identity=identity,
                ):
                    cleanup_succeeded = False
                    cleanup_failed = True
                    break
        if cleanup_succeeded and cleanup_binding_record is None:
            # The binding was the last durable authority.  Tombstones contain
            # no credentials and are discarded only after its exact removal;
            # a crash here is harmless and replayable.
            for path, directory, identity in cleanup_resources:
                _discard_owned_cleanup_tombstone(
                    path,
                    directory=directory,
                    identity=identity,
                )
        control_socket.close()
    return 125 if cleanup_failed or not cleanup_succeeded else 0


class _DetachedDockerCleanupWatchdog:
    """Exact birth identity for the non-child kill-safe cleanup process."""

    def __init__(self, pid: int, start_ticks: int) -> None:
        self.pid = pid
        self.start_ticks = start_ticks
        if not _runner_process_identity_alive(pid, start_ticks):
            raise ValueError("Docker cleanup watchdog birth is not alive")
        try:
            self._pidfd = os.pidfd_open(pid, 0)
        except (AttributeError, OSError) as exc:
            raise ValueError("Docker cleanup watchdog pidfd is unavailable") from exc
        self._exited = False
        if not _runner_process_identity_alive(pid, start_ticks):
            os.close(self._pidfd)
            self._pidfd = -1
            raise ValueError("Docker cleanup watchdog birth changed")

    def poll(self) -> int | None:
        if self._exited:
            return 0
        readable, _writable, _exceptional = select.select(
            [self._pidfd],
            [],
            [],
            0,
        )
        if not readable:
            return None
        self._exited = True
        os.close(self._pidfd)
        self._pidfd = -1
        return 0

    def wait(self, timeout: float | None = None) -> int:
        deadline = (
            None
            if timeout is None
            else time.monotonic() + max(0.0, float(timeout))
        )
        while self.poll() is None:
            if deadline is not None and time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(
                    cmd=["docker-cleanup-watchdog", str(self.pid)],
                    timeout=timeout,
                )
            time.sleep(0.02)
        return 0

    def _signal(self, signum: int) -> None:
        if self.poll() is not None:
            return
        try:
            signal.pidfd_send_signal(self._pidfd, signum)
        except (AttributeError, ProcessLookupError):
            return

    def terminate(self) -> None:
        self._signal(signal.SIGTERM)

    def kill(self) -> None:
        self._signal(signal.SIGKILL)

    def __del__(self) -> None:
        descriptor = getattr(self, "_pidfd", -1)
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _read_detached_docker_cleanup_watchdog(
    channel: socket.socket,
    *,
    timeout: float = 5.0,
) -> _DetachedDockerCleanupWatchdog:
    """Read and verify one bounded detached-watchdog readiness record."""

    deadline = time.monotonic() + max(0.0, timeout)
    payload = bytearray()
    while b"\n" not in payload and len(payload) <= 128:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError("Docker cleanup watchdog readiness timed out")
        readable, _writable, _exceptional = select.select(
            [channel],
            [],
            [],
            remaining,
        )
        if not readable:
            raise ValueError("Docker cleanup watchdog readiness timed out")
        block = channel.recv(129 - len(payload))
        if not block:
            break
        payload.extend(block)
    match = re.fullmatch(rb"([1-9][0-9]*):([0-9]+)\n", bytes(payload))
    if match is None:
        raise ValueError("Docker cleanup watchdog readiness is invalid")
    pid = int(match.group(1))
    start_ticks = int(match.group(2))
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        closing_parenthesis = raw.rfind(")")
        fields = raw[closing_parenthesis + 2 :].split()
        parent_pid = int(fields[1])
    except (OSError, IndexError, UnicodeError, ValueError) as exc:
        raise ValueError("Docker cleanup watchdog parent is unavailable") from exc
    if closing_parenthesis < 0 or parent_pid != 1:
        raise ValueError("Docker cleanup watchdog is not detached from the fence")
    watchdog = _DetachedDockerCleanupWatchdog(pid, start_ticks)
    if watchdog.poll() is not None:
        raise ValueError("Docker cleanup watchdog exited before admission")
    return watchdog


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
        control_socket: socket.socket,
        watchdog: _DetachedDockerCleanupWatchdog,
        provider: str,
        effect_observation: Mapping[str, str],
        cleanup_binding_record: Path | None,
    ) -> None:
        self.docker_bin = docker_bin
        self.container_name = container_name
        self.lease_root = lease_root
        self.docker_config = docker_config
        self.cidfile = cidfile
        self.provider_home = provider_home
        self.prompt_path = prompt_path
        self._control_socket = control_socket
        self._watchdog = watchdog
        self.provider = provider
        self.effect_observation = dict(effect_observation)
        self.cleanup_binding_record = cleanup_binding_record
        self._cleanup_path_identities: dict[str, Mapping[str, int]] = {
            "docker_config": _cleanup_path_identity(
                docker_config,
                directory=True,
            ),
            "lease_root": _cleanup_path_identity(
                lease_root,
                directory=True,
            ),
            "prompt_path": _cleanup_path_identity(
                prompt_path,
                directory=False,
            ),
            "provider_home": _cleanup_path_identity(
                provider_home,
                directory=True,
            ),
        }
        self._provider_start_sender: socket.socket | None = None
        self._provider_start_stdin: socket.socket | None = None
        self._provider_start_stdin_taken = False
        self._provider_start_released = False
        self._cleanup_binding_identity: Mapping[str, int] | None = None
        self._cleanup_binding_value: Mapping[str, object] | None = None
        self._create_command_id = ""
        self._create_command_body: Mapping[str, object] | None = None
        self._create_lock = threading.Lock()
        self._create_started = False
        self._authorized_image_id = ""
        self._termination_fence: Mapping[str, object] = {}
        self._closed = False
        self._cas_owned = False
        self._cas_terminal = False
        self._create_outcome_unknown = False
        self.preserve_for_recovery = False
        if self.cleanup_binding_record is not None:
            try:
                value = _validated_cleanup_binding_record(
                    self.cleanup_binding_record,
                    provider=self.provider,
                    docker_bin=self.docker_bin,
                    docker_config=self.docker_config,
                    container_name=self.container_name,
                    cidfile=self.cidfile,
                    lease_root=self.lease_root,
                    provider_home=self.provider_home,
                    prompt_path=self.prompt_path,
                    effect_observation=self.effect_observation,
                    binding_state="prepared_no_dispatch",
                    runner_pid=os.getpid(),
                    runner_start_ticks=_runner_process_start_ticks(os.getpid()),
                    watchdog_pid=self._watchdog.pid,
                    watchdog_start_ticks=self._watchdog.start_ticks,
                )
                identity = _cleanup_path_identity(
                    self.cleanup_binding_record,
                    directory=False,
                )
            except (OSError, ValueError) as exc:
                raise ValueError(
                    "Docker cleanup pre-dispatch binding is unavailable"
                ) from exc
            if value.get("path_identities") != self._cleanup_path_identities:
                raise ValueError("Docker cleanup pre-dispatch paths drifted")
            self._cleanup_binding_identity = identity
            self._cleanup_binding_value = value
        (
            self._provider_start_sender,
            self._provider_start_stdin,
        ) = _provider_start_socketpair()

    @classmethod
    def create(
        cls,
        docker_bin: str,
        *,
        provider: str,
        provider_home: Path,
        prompt_path: Path,
        effect_observation: Mapping[str, str] | None = None,
    ) -> "_DockerContainerLease":
        if provider not in _DOCKER_ISOLATION_PROVIDERS:
            raise ValueError("Docker isolation provider is invalid")
        observation = dict(effect_observation or {})
        if (
            set(observation)
            not in (set(), set(_DOCKER_EFFECT_OBSERVATION_FIELDS))
            or any(
                not isinstance(value, str) or not value
                for value in observation.values()
            )
            or (observation and provider != "codex")
        ):
            raise ValueError("Docker effect observation binding is invalid")
        observation_arguments: list[str] = []
        if observation:
            observation_arguments = [
                "--logical-attempt-id",
                observation["logical_attempt_id"],
                "--provider-attempt-store",
                observation["provider_attempt_store"],
                "--provider-attempt-store-identity",
                observation["provider_attempt_store_identity"],
            ]
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
        cleanup_binding_record = _docker_cleanup_binding_path(container_name)
        from .process_security import (
            require_state_authority_handoff_ptrace_protection,
        )

        # Unlike pipes, connected Unix socket descriptors cannot be reopened
        # through /proc/<pid>/fd by a same-UID process. The ptrace prerequisite
        # closes the remaining descriptor-duplication path before this one-shot
        # launcher capability exists.
        require_state_authority_handoff_ptrace_protection()
        control_socket, launcher_socket = socket.socketpair(
            socket.AF_UNIX,
            socket.SOCK_STREAM,
        )
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
        launcher: subprocess.Popen[bytes] | None = None
        watchdog: _DetachedDockerCleanupWatchdog | None = None
        runner_start_ticks = _runner_process_start_ticks(os.getpid())
        try:
            launcher = subprocess.Popen(
                [
                    sys.executable,
                    "-I",
                    "-B",
                    runner_entry,
                    _DOCKER_CLEANUP_WATCHDOG_LAUNCHER_ARG,
                    "--control-fd",
                    str(launcher_socket.fileno()),
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
                    "--runner-pid",
                    str(os.getpid()),
                    "--runner-start-ticks",
                    str(runner_start_ticks),
                    *observation_arguments,
                    *(
                        [
                            "--cleanup-binding-record",
                            str(cleanup_binding_record),
                        ]
                        if cleanup_binding_record is not None
                        else []
                    ),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd="/",
                env=_docker_cleanup_watchdog_env(),
                start_new_session=True,
                close_fds=True,
                pass_fds=tuple(
                    sorted(
                        {
                            *inherited_control_plane,
                            launcher_socket.fileno(),
                        }
                    )
                ),
            )
            launcher_socket.close()
            if launcher.wait(timeout=5.0) != 0:
                raise ValueError("Docker cleanup watchdog launcher failed")
            watchdog = _read_detached_docker_cleanup_watchdog(
                control_socket,
                timeout=5.0,
            )
        except Exception:
            control_socket.close()
            launcher_socket.close()
            if launcher is not None and launcher.poll() is None:
                launcher.kill()
                try:
                    launcher.wait(timeout=1.0)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            if watchdog is not None:
                watchdog.terminate()
                try:
                    watchdog.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    watchdog.kill()
            try:
                _remove_exact_docker_container(
                    docker_bin=str(docker_path),
                    docker_config=docker_config,
                    container_name=container_name,
                    settle_for_creation=False,
                )
            except ValueError:
                pass
            try:
                docker_config.rmdir()
            except (FileNotFoundError, OSError):
                pass
            try:
                lease_root.rmdir()
            except (FileNotFoundError, OSError):
                pass
            raise
        if watchdog is None:
            control_socket.close()
            raise ValueError("Docker cleanup watchdog failed to start")
        try:
            return cls(
                docker_bin=str(docker_path),
                container_name=container_name,
                lease_root=lease_root,
                docker_config=docker_config,
                cidfile=cidfile,
                provider_home=provider_home,
                prompt_path=prompt_path,
                control_socket=control_socket,
                watchdog=watchdog,
                provider=provider,
                effect_observation=observation,
                cleanup_binding_record=cleanup_binding_record,
            )
        except BaseException:
            # Constructor admission can fail after the detached watchdog has
            # published its prepared record. Closing the private socket lets
            # that exact watchdog perform the ordinary no-dispatch cleanup;
            # killing it here would strand the now-durable resource lease.
            control_socket.close()
            try:
                watchdog.wait(timeout=_DOCKER_CLEANUP_TIMEOUT_SECONDS + 2.0)
            except subprocess.TimeoutExpired:
                pass
            raise

    def _publish_cleanup_binding(
        self,
        *,
        create_command_id: str,
        create_cwd: Path,
        create_environment_id: str,
    ) -> None:
        if self.cleanup_binding_record is None:
            return
        lock_handle = _docker_binding_lock_descriptor(
            self.cleanup_binding_record
        )
        try:
            expected_value = self._cleanup_binding_value
            expected_identity = self._cleanup_binding_identity
            if expected_value is None or expected_identity is None:
                raise ValueError("Docker cleanup pre-dispatch binding was not admitted")
            current = _validated_cleanup_binding_record(
                self.cleanup_binding_record,
                provider=self.provider,
                docker_bin=self.docker_bin,
                docker_config=self.docker_config,
                container_name=self.container_name,
                cidfile=self.cidfile,
                lease_root=self.lease_root,
                provider_home=self.provider_home,
                prompt_path=self.prompt_path,
                effect_observation=self.effect_observation,
                binding_state="prepared_no_dispatch",
                runner_pid=os.getpid(),
                runner_start_ticks=_runner_process_start_ticks(os.getpid()),
                watchdog_pid=self._watchdog.pid,
                watchdog_start_ticks=self._watchdog.start_ticks,
                control_directory_fd=lock_handle.directory_fd,
            )
            current_identity = lock_handle.path_identity(
                self.cleanup_binding_record
            )
            if current != expected_value or current_identity != expected_identity:
                raise ValueError("Docker cleanup pre-dispatch binding changed")
            body = _docker_cleanup_binding_value(
                binding_state="command_bound",
                provider=self.provider,
                docker_bin=self.docker_bin,
                container_name=self.container_name,
                lease_root=self.lease_root,
                docker_config=self.docker_config,
                cidfile=self.cidfile,
                provider_home=self.provider_home,
                prompt_path=self.prompt_path,
                effect_observation=self.effect_observation,
                path_identities=self._cleanup_path_identities,
                binding_path=self.cleanup_binding_record,
                runner_pid=os.getpid(),
                runner_start_ticks=_runner_process_start_ticks(os.getpid()),
                watchdog_pid=self._watchdog.pid,
                watchdog_start_ticks=self._watchdog.start_ticks,
                create_command_id=create_command_id,
                create_cwd=create_cwd,
                create_environment_id=create_environment_id,
            )
            lock_handle.write(
                self.cleanup_binding_record,
                body,
                replace_existing=True,
            )
            admitted = _validated_cleanup_binding_record(
                self.cleanup_binding_record,
                provider=self.provider,
                docker_bin=self.docker_bin,
                docker_config=self.docker_config,
                container_name=self.container_name,
                cidfile=self.cidfile,
                lease_root=self.lease_root,
                provider_home=self.provider_home,
                prompt_path=self.prompt_path,
                effect_observation=self.effect_observation,
                binding_state="command_bound",
                runner_pid=os.getpid(),
                runner_start_ticks=_runner_process_start_ticks(os.getpid()),
                watchdog_pid=self._watchdog.pid,
                watchdog_start_ticks=self._watchdog.start_ticks,
                create_command_id=create_command_id,
                create_cwd=create_cwd,
                create_environment_id=create_environment_id,
                control_directory_fd=lock_handle.directory_fd,
            )
            self._cleanup_binding_identity = lock_handle.path_identity(
                self.cleanup_binding_record
            )
            self._cleanup_binding_value = admitted
        finally:
            lock_handle.close()

    def _admit_cleanup_authority(
        self,
        journal: Mapping[str, object] | None,
    ) -> bool:
        if self.cleanup_binding_record is None:
            return True
        try:
            raw_binding = _read_private_control_record(
                self.cleanup_binding_record.parent,
                self.cleanup_binding_record.name,
            )
            if raw_binding is None:
                return False
            binding_state = str(raw_binding.get("binding_state") or "")
            if binding_state == "prepared_no_dispatch":
                if journal is not None and journal.get("state") != "prepared":
                    return False
                create_command_id = ""
                create_cwd = None
                create_environment_id = ""
                termination_fence: Mapping[str, object] = {}
            elif binding_state == "command_bound" and journal is not None:
                create_command_id = str(journal["command_id"])
                create_cwd = Path(str(journal["cwd"]))
                create_environment_id = str(journal["environment_id"])
                raw_termination_fence = raw_binding.get("termination_fence")
                if not isinstance(raw_termination_fence, Mapping):
                    return False
                termination_fence = raw_termination_fence
                if (
                    not self._create_command_id
                    or self._create_command_body is None
                    or create_command_id != self._create_command_id
                    or any(
                        journal.get(name) != expected
                        for name, expected in self._create_command_body.items()
                    )
                ):
                    return False
            else:
                return False
            value = _validated_cleanup_binding_record(
                self.cleanup_binding_record,
                provider=self.provider,
                docker_bin=self.docker_bin,
                docker_config=self.docker_config,
                container_name=self.container_name,
                cidfile=self.cidfile,
                lease_root=self.lease_root,
                provider_home=self.provider_home,
                prompt_path=self.prompt_path,
                effect_observation=self.effect_observation,
                binding_state=binding_state,
                runner_pid=os.getpid(),
                runner_start_ticks=_runner_process_start_ticks(os.getpid()),
                watchdog_pid=self._watchdog.pid,
                watchdog_start_ticks=self._watchdog.start_ticks,
                create_command_id=create_command_id,
                create_cwd=create_cwd,
                create_environment_id=create_environment_id,
                termination_fence=termination_fence,
            )
            current_binding_identity = _cleanup_path_identity(
                self.cleanup_binding_record,
                directory=False,
            )
        except (KeyError, ValueError):
            return False
        if value.get("path_identities") != self._cleanup_path_identities:
            return False
        expected_upgrade = bool(
            self._cleanup_binding_value is not None
            and (
                (
                    self._cleanup_binding_value.get("binding_state")
                    == "prepared_no_dispatch"
                    and value.get("binding_state") == "command_bound"
                )
                or (
                    self._cleanup_binding_value.get("binding_state")
                    == value.get("binding_state")
                    == "command_bound"
                    and self._cleanup_binding_value.get("termination_fence")
                    == {}
                    and bool(value.get("termination_fence"))
                )
            )
        )
        if (
            self._cleanup_binding_identity is not None
            and current_binding_identity != self._cleanup_binding_identity
            and not expected_upgrade
        ):
            return False
        self._cleanup_binding_identity = current_binding_identity
        self._cleanup_binding_value = value
        return True

    def bind_isolation_image(self, image_id: str) -> None:
        """Bind the separately resolved immutable image before create."""

        image = str(image_id or "").strip()
        allowed_codex_images = {_CODEX_TASK_TOOLCHAIN_IMAGE_ID}
        sealed = _sealed_provider_isolation_image_id()
        if sealed:
            allowed_codex_images.add(sealed)
        if (
            re.fullmatch(r"sha256:[0-9a-f]{64}", image) is None
            or (self.provider == "codex" and image not in allowed_codex_images)
        ):
            raise ValueError("Docker isolation image authority is invalid")
        with self._create_lock:
            if self._create_started or self._authorized_image_id:
                raise ValueError("Docker isolation image is already bound")
            self._authorized_image_id = image

    def create_inert_container(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
    ) -> subprocess.CompletedProcess[bytes]:
        """Have the durable watchdog issue the sole Docker create request."""

        with self._create_lock:
            if self._create_started:
                raise ValueError("Docker create lease permits exactly one dispatch")
            if not self._authorized_image_id:
                raise ValueError("Docker create lease has no bound image authority")
            authorized_image = self._authorized_image_id
            self._create_started = True
        if self.cleanup_binding_record is None:
            return subprocess.run(
                list(command),
                cwd=cwd,
                env=dict(env),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=_DOCKER_CREATE_TIMEOUT_SECONDS,
                check=False,
            )
        environment_id, _environment_payload, create_environment = (
            _docker_create_environment_payload(env)
        )
        command_id, command_body = _docker_create_command_identity(
            provider=self.provider,
            docker_bin=self.docker_bin,
            docker_config=self.docker_config,
            container_name=self.container_name,
            cidfile=self.cidfile,
            cwd=cwd,
            environment_id=environment_id,
            expected_image=authorized_image,
            argv=command,
        )
        prepared = _docker_create_journal_value(
            command_body=command_body,
            command_id=command_id,
            state="prepared",
        )
        self._create_command_id = command_id
        self._create_command_body = command_body
        private_payload = _docker_create_private_handoff_payload(
            command_id=command_id,
            command_body=command_body,
            environment=create_environment,
        )
        private_message = (
            b"Q"
            + len(private_payload).to_bytes(8, "big")
            + private_payload
        )
        try:
            self._control_socket.sendall(private_message)
        except OSError as exc:
            self.preserve_for_recovery = True
            raise ValueError("Docker private command handoff failed") from exc
        _write_private_control_record(
            self.lease_root,
            _DOCKER_CREATE_JOURNAL_NAME,
            prepared,
            replace_existing=False,
        )
        _transition_docker_create_journal(
            prepared,
            lease_root=self.lease_root,
            state="create_armed",
        )
        try:
            self._control_socket.sendall(b"D")
        except OSError as exc:
            self.preserve_for_recovery = True
            raise ValueError("Docker create worker dispatch failed") from exc
        deadline = time.monotonic() + _DOCKER_CREATE_TIMEOUT_SECONDS
        try:
            journal = _read_docker_create_private_result(
                self._control_socket,
                deadline=deadline,
            )
            state = str(journal.get("state") or "")
            stdout = bytes.fromhex(str(journal.get("stdout_hex") or ""))
            stderr = bytes.fromhex(str(journal.get("stderr_hex") or ""))
            returncode = journal.get("returncode")
            issuer = journal.get("issuer_process_birth")
            if isinstance(returncode, bool) or not isinstance(returncode, int):
                raise ValueError("Docker create private return code is invalid")
            expected = _docker_create_journal_value(
                command_body=command_body,
                command_id=command_id,
                state=state,
                issuer_process_birth=(issuer if isinstance(issuer, Mapping) else None),
                returncode=returncode,
                stdout=stdout,
                stderr=stderr,
            )
            if (
                journal != expected
                or not isinstance(issuer, Mapping)
                or issuer.get("parent_pid") != self._watchdog.pid
                or any(
                    journal.get(name) != item
                    for name, item in command_body.items()
                )
                or not self._admit_cleanup_authority(journal)
            ):
                raise ValueError("Docker create private result identity drifted")
        except (OSError, TypeError, ValueError) as exc:
            self.preserve_for_recovery = True
            raise ValueError("Docker create worker result is unavailable") from exc
        if state == "create_outcome_unknown":
            self._create_outcome_unknown = True
            self.preserve_for_recovery = True
        return subprocess.CompletedProcess(
            list(command),
            returncode,
            stdout=stdout,
            stderr=stderr,
        )

    def take_provider_start_stdin(self) -> socket.socket:
        """Transfer the sole anonymous Docker-stdin endpoint to one start."""

        if (
            not self._create_started
            or self._create_outcome_unknown
            or self._provider_start_stdin_taken
            or self._provider_start_stdin is None
            or self._provider_start_sender is None
        ):
            raise ValueError("Docker provider start capability is unavailable")
        channel = self._provider_start_stdin
        self._provider_start_stdin = None
        self._provider_start_stdin_taken = True
        return channel

    def finish_provider_input(self, payload: str | bytes = b"") -> None:
        """Send post-fence provider input once, then make stdin observe EOF."""

        if not self._provider_start_released:
            raise ValueError("Docker provider input precedes its running fence")
        channel = self._provider_start_sender
        if channel is None:
            raise ValueError("Docker provider input capability is unavailable")
        self._provider_start_sender = None
        encoded = (
            payload.encode("utf-8")
            if isinstance(payload, str)
            else bytes(payload)
        )
        try:
            if encoded:
                channel.sendall(encoded)
            try:
                channel.shutdown(socket.SHUT_WR)
            except OSError:
                pass
        finally:
            channel.close()

    def _abort_provider_start(self) -> None:
        """Close every local endpoint so an unreleased wrapper observes EOF."""

        for name in ("_provider_start_sender", "_provider_start_stdin"):
            channel = getattr(self, name, None)
            setattr(self, name, None)
            if channel is not None:
                try:
                    channel.close()
                except OSError:
                    pass

    def capture_running_termination_fence(
        self,
        *,
        timeout: float = 5.0,
    ) -> Mapping[str, object]:
        """Persist the exact init/cgroup identity while the container runs.

        Docker clears ``State.Pid`` after exit.  Capturing only in ``close``
        therefore permits a detached cgroup member to outlive a disappeared
        Docker name.  The attached start path calls this method immediately
        after dispatch and before it supplies provider input or waits for the
        provider outcome.
        """

        if (
            not self._create_started
            or not self._authorized_image_id
            or self._create_outcome_unknown
        ):
            raise ValueError("Docker execution was not exactly created")
        try:
            raw_container_id = self.cidfile.read_text(encoding="ascii").strip()
        except (OSError, UnicodeError) as exc:
            self.preserve_for_recovery = True
            raise ValueError("Docker execution CID is unavailable") from exc
        if re.fullmatch(r"[0-9a-f]{64}", raw_container_id) is None:
            self.preserve_for_recovery = True
            raise ValueError("Docker execution CID is invalid")
        try:
            journal = _validated_docker_create_journal(
                lease_root=self.lease_root,
                provider=self.provider,
                docker_bin=self.docker_bin,
                docker_config=self.docker_config,
                container_name=self.container_name,
                cidfile=self.cidfile,
            )
            if journal is None or journal.get("state") != "create_observed":
                raise ValueError("Docker create is not durably observed")
            if not self._admit_cleanup_authority(journal):
                raise ValueError("Docker cleanup authority is unavailable")
            existing = (
                self._cleanup_binding_value.get("termination_fence")
                if self._cleanup_binding_value is not None
                else self._termination_fence
            )
            if isinstance(existing, Mapping) and existing:
                admitted_existing = _validated_docker_termination_fence(
                    existing,
                    provider=self.provider,
                    container_name=self.container_name,
                    expected_container_id=raw_container_id,
                    expected_image_id=self._authorized_image_id,
                )
                self._termination_fence = admitted_existing
                self._release_provider_start_capability(admitted_existing)
                return admitted_existing

            deadline = time.monotonic() + max(0.05, timeout)
            fence: dict[str, object] | None = None
            while time.monotonic() < deadline:
                try:
                    candidate = _attest_exact_docker_execution(
                        docker_bin=self.docker_bin,
                        docker_config=self.docker_config,
                        provider=self.provider,
                        container_name=self.container_name,
                        container_id=raw_container_id,
                        image_id=self._authorized_image_id,
                        timeout=min(0.5, max(0.05, deadline - time.monotonic())),
                    )
                except ValueError:
                    candidate = {}
                if candidate and candidate.get("init_pid"):
                    fence = candidate
                    break
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
            if fence is None:
                raise ValueError("Docker running kernel scope was not captured")

            if self.cleanup_binding_record is not None:
                expected_value = self._cleanup_binding_value
                expected_identity = self._cleanup_binding_identity
                if expected_value is None or expected_identity is None:
                    raise ValueError("Docker cleanup binding CAS is unavailable")
                published, published_identity = (
                    _publish_docker_termination_fence_binding(
                        record_path=self.cleanup_binding_record,
                        expected_record_id=str(expected_value.get("record_id") or ""),
                        expected_identity=expected_identity,
                        provider=self.provider,
                        docker_bin=self.docker_bin,
                        docker_config=self.docker_config,
                        container_name=self.container_name,
                        cidfile=self.cidfile,
                        lease_root=self.lease_root,
                        provider_home=self.provider_home,
                        prompt_path=self.prompt_path,
                        effect_observation=self.effect_observation,
                        runner_pid=os.getpid(),
                        runner_start_ticks=_runner_process_start_ticks(os.getpid()),
                        watchdog_pid=self._watchdog.pid,
                        watchdog_start_ticks=self._watchdog.start_ticks,
                        create_command_id=str(journal["command_id"]),
                        create_cwd=Path(str(journal["cwd"])),
                        create_environment_id=str(journal["environment_id"]),
                        termination_fence=fence,
                    )
                )
                self._cleanup_binding_value = published
                self._cleanup_binding_identity = published_identity
            self._termination_fence = fence
            self._release_provider_start_capability(fence)
            return fence
        except (KeyError, OSError, TypeError, ValueError):
            self.preserve_for_recovery = True
            raise

    def _release_provider_start_capability(
        self,
        termination_fence: Mapping[str, object],
    ) -> None:
        """Send the one-shot stdin marker only after the fence CAS is durable."""

        if self._provider_start_released:
            return
        admitted = _validated_docker_termination_fence(
            termination_fence,
            provider=self.provider,
            container_name=self.container_name,
            expected_image_id=self._authorized_image_id,
        )
        channel = self._provider_start_sender
        if (
            admitted.get("docker_state") != "running"
            or int(admitted.get("init_pid") or 0) <= 0
            or not self._provider_start_stdin_taken
            or channel is None
            or channel.family != socket.AF_UNIX
            or channel.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE)
            != socket.SOCK_STREAM
            or not stat.S_ISSOCK(os.fstat(channel.fileno()).st_mode)
        ):
            raise ValueError("Docker provider start release lacks a live fence")
        # This is a one-shot external handoff.  Mark it consumed before the
        # fallible send so an unknown partial write can never be replayed.
        self._provider_start_released = True
        try:
            channel.sendall(_DOCKER_PROVIDER_START_MARKER)
        except OSError:
            self._provider_start_sender = None
            channel.close()
            raise
        if self.provider == "grok":
            self.finish_provider_input()

    def mark_cas_owned(self) -> None:
        if self._cas_owned:
            return
        # This method is entered only after the durable effect_started CAS
        # commits.  Transfer cleanup ownership in memory before the first
        # fallible marker operation so ENOSPC/I/O failure cannot make close()
        # reap a winner whose external outcome is still recoverable.
        self._cas_owned = True
        marker = self.lease_root / "cas-owned"
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
            try:
                os.write(descriptor, self.container_name.encode("ascii"))
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError:
            self.preserve_for_recovery = True
            raise
        try:
            self._control_socket.sendall(b"A")
        except OSError as exc:
            self.preserve_for_recovery = True
            raise ValueError("Docker CAS watchdog marker failed") from exc

    def _arm_fenced_removal(
        self,
        termination_fence: Mapping[str, object],
    ) -> bool:
        if (
            self.cleanup_binding_record is None
            or self._cleanup_binding_value is None
            or self._cleanup_binding_identity is None
        ):
            raise ValueError("Docker rm lacks a durable binding")
        return _arm_docker_removal_once(
            binding_path=self.cleanup_binding_record,
            expected_binding_identity=self._cleanup_binding_identity,
            binding_record=self._cleanup_binding_value,
            termination_fence=termination_fence,
        )

    def _durable_cas_state(self) -> str:
        if not self.effect_observation:
            return "unscoped"
        try:
            from ipfs_accelerate_py.agent_supervisor.control.provider_attempt_store import (
                DurableProviderAttemptCAS,
            )

            observer = DurableProviderAttemptCAS(
                self.effect_observation["provider_attempt_store"],
                expected_directory_identity=self.effect_observation[
                    "provider_attempt_store_identity"
                ],
                create_if_missing=False,
            )
        except (KeyError, OSError, ValueError):
            return "unknown"
        return _observed_provider_attempt_cleanup_state(
            observer,
            logical_attempt_id=self.effect_observation["logical_attempt_id"],
            lease_root=self.lease_root,
            docker_config=self.docker_config,
            container_name=self.container_name,
            watchdog_pid=self._watchdog.pid,
            watchdog_start_ticks=self._watchdog.start_ticks,
        )

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
            self._control_socket.sendall(b"T")
        except OSError as exc:
            raise ValueError("Docker CAS terminal watchdog marker failed") from exc

    def close(self, *, docker_run_finished: bool) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if docker_run_finished:
                self._control_socket.sendall(b"C")
        except OSError:
            pass
        finally:
            self._control_socket.close()
        self._abort_provider_start()
        if self.preserve_for_recovery:
            return
        journal: Mapping[str, object] | None = None
        if self.cleanup_binding_record is not None:
            try:
                journal = _validated_docker_create_journal(
                    lease_root=self.lease_root,
                    provider=self.provider,
                    docker_bin=self.docker_bin,
                    docker_config=self.docker_config,
                    container_name=self.container_name,
                    cidfile=self.cidfile,
                )
            except ValueError:
                self.preserve_for_recovery = True
                return
            if journal is not None and journal.get("state") in {
                "create_armed",
                "create_inflight",
                "create_outcome_unknown",
            }:
                self.preserve_for_recovery = True
                return
            if not self._admit_cleanup_authority(journal):
                self.preserve_for_recovery = True
                return
        durable_state = self._durable_cas_state()
        if (
            (self._cas_owned or durable_state in {"owned", "unknown"})
            and not (self._cas_terminal or durable_state == "terminal")
        ):
            # The runner may have died after the provider effect but before
            # durable completion.  Recovery owns both the container evidence
            # and its private Docker configuration from this point onward.
            self.preserve_for_recovery = True
            return
        try:
            self._watchdog.wait(timeout=_DOCKER_CLEANUP_TIMEOUT_SECONDS + 2)
        except subprocess.TimeoutExpired:
            if (
                self.cleanup_binding_record is not None
                and not self._admit_cleanup_authority(journal)
            ):
                self.preserve_for_recovery = True
                return
            raw_fence = (
                self._cleanup_binding_value.get("termination_fence")
                if self._cleanup_binding_value is not None
                else self._termination_fence
            )
            termination_fence = (
                raw_fence if isinstance(raw_fence, Mapping) and raw_fence else None
            )
            if (
                journal is not None
                and journal.get("state") == "create_observed"
                and termination_fence is None
            ):
                self.preserve_for_recovery = True
                return
            try:
                issue_removal = bool(
                    termination_fence is not None
                    and self._arm_fenced_removal(termination_fence)
                )
                _remove_exact_docker_container(
                    docker_bin=self.docker_bin,
                    docker_config=self.docker_config,
                    container_name=self.container_name,
                    settle_for_creation=False,
                    termination_fence=termination_fence,
                    issue_removal=issue_removal,
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
        cleanup_resources = (
            (
                self.prompt_path,
                False,
                self._cleanup_path_identities["prompt_path"],
            ),
            (
                self.provider_home,
                True,
                self._cleanup_path_identities["provider_home"],
            ),
            (
                self.lease_root,
                True,
                self._cleanup_path_identities["lease_root"],
            ),
        )
        if self.lease_root.exists():
            # The watchdog can exit between receiving its marker and proving
            # cleanup.  Do not infer success merely from process death.
            try:
                if (
                    self.cleanup_binding_record is not None
                    and not self._admit_cleanup_authority(journal)
                ):
                    raise ValueError("Docker cleanup binding could not be refreshed")
                raw_fence = (
                    self._cleanup_binding_value.get("termination_fence")
                    if self._cleanup_binding_value is not None
                    else self._termination_fence
                )
                termination_fence = (
                    raw_fence
                    if isinstance(raw_fence, Mapping) and raw_fence
                    else None
                )
                if (
                    journal is not None
                    and journal.get("state") == "create_observed"
                    and termination_fence is None
                ):
                    raise ValueError(
                        "Docker executed effect has no kernel cleanup fence"
                    )
                issue_removal = bool(
                    termination_fence is not None
                    and self._arm_fenced_removal(termination_fence)
                )
                _remove_exact_docker_container(
                    docker_bin=self.docker_bin,
                    docker_config=self.docker_config,
                    container_name=self.container_name,
                    settle_for_creation=False,
                    termination_fence=termination_fence,
                    issue_removal=issue_removal,
                )
            except ValueError:
                self.preserve_for_recovery = True
                return
        if self.cleanup_binding_record is not None:
            if (
                self._cleanup_binding_identity is None
                or self._cleanup_binding_value is None
            ):
                self.preserve_for_recovery = True
                return
            try:
                completed = _finalize_verified_cleanup_completion(
                    binding_path=self.cleanup_binding_record,
                    binding_identity=self._cleanup_binding_identity,
                    binding_record=self._cleanup_binding_value,
                )
            except ValueError:
                self.preserve_for_recovery = True
                return
            if not completed:
                self.preserve_for_recovery = True
            return
        # Legacy/unbound pre-dispatch paths have no durable per-binding lock.
        # They retain the older exact-inode transition but cannot coalesce a
        # tombstone replay into authoritative cleanup.
        for path, directory, identity in cleanup_resources:
            if not _remove_owned_cleanup_path(
                path,
                directory=directory,
                identity=identity,
            ):
                self.preserve_for_recovery = True
                return
        for path, directory, identity in cleanup_resources:
            _discard_owned_cleanup_tombstone(
                path,
                directory=directory,
                identity=identity,
            )


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
        "--entrypoint=/bin/sh",
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

    # Operator secrets stay in the ephemeral grok_home copy.  Do not bind-mount
    # ~/.grok/auth.json: argv would name the operator path, and a later deny
    # mask of ~/.grok would hide it.

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
    command.extend(
        [
            image,
            "-c",
            _DOCKER_PROVIDER_START_SCRIPT,
            "aseh-provider-start",
            *inner,
        ]
    )
    return command


def _validated_created_grok_container_id(
    created: subprocess.CompletedProcess[bytes],
    *,
    cidfile: Path,
) -> str:
    """Match Docker's create response to the runner-owned cidfile."""

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
    return created_fields[0]


def _create_grok_container_and_build_start_command(
    create_command: Sequence[str],
    *,
    workspace: Path,
    docker_environment: dict[str, str],
    docker_lease: _DockerContainerLease,
) -> list[str]:
    """Create inert Grok container, then bind its exact ID to attached start."""

    try:
        durable_create = getattr(
            docker_lease,
            "create_inert_container",
            None,
        )
        created = (
            durable_create(
                list(create_command),
                cwd=workspace,
                env=docker_environment,
            )
            if callable(durable_create)
            else subprocess.run(
                list(create_command),
                cwd=workspace,
                env=docker_environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=_DOCKER_CREATE_TIMEOUT_SECONDS,
                check=False,
            )
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError("Grok container creation timed out") from exc
    container_id = _validated_created_grok_container_id(
        created,
        cidfile=docker_lease.cidfile,
    )
    return [
        docker_lease.docker_bin,
        f"--host={_DOCKER_LOCAL_HOST}",
        "--config",
        str(docker_lease.docker_config),
        "start",
        "--attach",
        "--interactive",
        container_id,
    ]


def _run_created_grok_container_with_typed_failure_capture(
    create_command: Sequence[str],
    *,
    workspace: Path,
    env: dict[str, str],
    docker_lease: _DockerContainerLease,
) -> int:
    """Create the inert Grok container, then run that exact container.

    ``_docker_grok_command`` deliberately returns a ``docker create`` command
    so the container identity exists before any provider effect starts.  The
    ordinary task path must not mistake the successful create command's
    64-byte container ID for Grok output.  Validate both Docker's response and
    the runner-owned cidfile before attaching to the exact created container.
    """

    start_command = _create_grok_container_and_build_start_command(
        create_command,
        workspace=workspace,
        docker_environment=env,
        docker_lease=docker_lease,
    )
    provider_stdin = docker_lease.take_provider_start_stdin()
    failures: list[BaseException] = []

    def capture_fence() -> None:
        try:
            docker_lease.capture_running_termination_fence()
        except BaseException as exc:
            failures.append(exc)
            docker_lease._abort_provider_start()

    capture_thread = threading.Thread(
        target=capture_fence,
        name="docker-grok-compat-fence-capture",
        daemon=True,
    )
    capture_thread.start()
    returncode = _run_grok_with_typed_failure_capture(
        start_command,
        env=env,
        provider_stdin=provider_stdin,
    )
    capture_thread.join(timeout=6.0)
    if capture_thread.is_alive() or failures:
        docker_lease.preserve_for_recovery = True
        raise ValueError("Grok Docker kernel cleanup fence was not captured")
    return returncode


def _docker_codex_fallback_command(
    *,
    codex_command: Sequence[str],
    workspace: Path,
    source_auth: Path,
    provider_home: Path,
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
    allowed_images = {AGENT_IMPLEMENTATION_CODEX_IMAGE_ID}
    sealed = _sealed_provider_isolation_image_id()
    if sealed:
        allowed_images.add(sealed)
    if not docker or image not in allowed_images:
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
    cleanup_root = docker_config.parent.parent
    _docker_cleanup_root_identity(cleanup_root)
    try:
        provider_home_metadata = os.lstat(provider_home)
    except OSError as exc:
        raise ValueError("Codex fallback provider home is unavailable") from exc
    if (
        provider_home.parent != cleanup_root
        or not provider_home.name.startswith("asref-codex-home-")
        or not stat.S_ISDIR(provider_home_metadata.st_mode)
        or stat.S_ISLNK(provider_home_metadata.st_mode)
        or provider_home_metadata.st_uid != os.geteuid()
    ):
        raise ValueError("Codex fallback provider home is not cleanup-bound")
    # Keep the mutable credential copy inside the exact provider_home inode
    # recorded by the watchdog, durable binding, and CAS cleanup receipt.
    # SIGKILL can therefore preserve it only behind the same strict fence; no
    # unrecorded /tmp credential lease exists.
    isolated_auth = provider_home / "auth.json"
    _install_ephemeral_credential(source_auth, isolated_auth)
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
    vendor_mounts = _docker_codex_host_vendor_mounts()
    if vendor_mounts:
        command.extend(vendor_mounts)
        inner[0] = "/usr/local/bin/codex"
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
            isolated_auth,
            destination=_CODEX_CONTAINER_AUTH_PATH,
            read_only=False,
        )
    )
    # The authority-validation image contains a large CUDA-oriented Config.Env.
    # Clearing it here prevents ENV/BASH_ENV hooks and every unrelated image
    # default from reaching Codex or repository commands.  Only these fixed,
    # non-secret values are serialized; provider authority remains file-based.
    environment_assignments = [
        f"{name}={value}" for name, value in sorted(expected_environment.items())
    ]
    command.extend(
        [
            image,
            "-i",
            *environment_assignments,
            "/bin/sh",
            "-c",
            _DOCKER_PROVIDER_START_SCRIPT,
            "aseh-provider-start",
            *inner,
        ]
    )
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
    effect_observation: Mapping[str, str] | None = None,
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
        lease_observation = (
            {}
            if effect_observation is None
            else {"effect_observation": effect_observation}
        )
        docker_lease = _DockerContainerLease.create(
            docker_bin,
            provider="codex",
            provider_home=codex_home,
            prompt_path=prompt_path,
            **lease_observation,
        )
        _populate_bound_ephemeral_prompt(prompt_path, prompt)
        isolation_image = _docker_codex_task_toolchain_image_id(
            docker_bin,
            docker_config=docker_lease.docker_config,
        )
        if not isolation_image:
            raise ValueError(
                "Codex fallback task-toolchain image is not pinned locally"
            )
        docker_lease.bind_isolation_image(isolation_image)
        command = _docker_codex_fallback_command(
            codex_command=codex_command,
            workspace=workspace,
            source_auth=source_auth,
            provider_home=docker_lease.provider_home,
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
            durable_create = getattr(
                docker_lease,
                "create_inert_container",
                None,
            )
            created = (
                durable_create(
                    command,
                    cwd=workspace,
                    env=docker_environment,
                )
                if callable(durable_create)
                else subprocess.run(
                    command,
                    cwd=workspace,
                    env=docker_environment,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=_DOCKER_CREATE_TIMEOUT_SECONDS,
                    check=False,
                )
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
        # anonymous start capability. A crash can therefore adopt this same
        # created container with a fresh capability; no
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
                "watchdog_start_ticks": docker_lease._watchdog.start_ticks,
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
            # Publishing effect_started is an ambiguous durable boundary: a
            # directory fsync can fail after os.replace made the winner
            # visible.  Preserve the inert container and all recovery inputs
            # before entering that call.  Clear only when a read-only exact
            # observation proves this lease is absent/foreign, or after the
            # local ownership handoff is complete.
            docker_lease.preserve_for_recovery = True
            try:
                effect_claim(launch_context)
            except BaseException:
                if docker_lease._durable_cas_state() in {"absent", "foreign"}:
                    docker_lease.preserve_for_recovery = False
                raise
            # A concurrent loser must remain an ordinary inert lease so its
            # own container is removed.  Cleanup ownership transfers only
            # after this process has won the durable effect_started CAS.
            docker_lease.mark_cas_owned()
            docker_lease.preserve_for_recovery = False
        provider_stdin = docker_lease.take_provider_start_stdin()
        try:
            process = subprocess.Popen(
                start_command,
                cwd=workspace,
                env=docker_environment,
                stdin=provider_stdin,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        finally:
            provider_stdin.close()
        if process.stdout is None or process.stderr is None:
            raise RuntimeError("Codex fallback process pipes were not created")
        try:
            # Codex has not received its prompt yet, so its init remains live
            # while the host captures and durably publishes the exact PID,
            # namespace, and cgroup fence used by every later cleanup path.
            capture_running_fence = getattr(
                docker_lease,
                "capture_running_termination_fence",
                None,
            )
            if callable(capture_running_fence):
                capture_running_fence()
            elif type(docker_lease) is _DockerContainerLease:
                raise ValueError("Docker cleanup fence capture is unavailable")
        except (OSError, TypeError, ValueError) as exc:
            docker_lease.preserve_for_recovery = True
            docker_lease._abort_provider_start()
            try:
                process.wait(timeout=5.0)
            except (OSError, subprocess.TimeoutExpired):
                pass
            raise ValueError(
                "Codex fallback kernel cleanup fence was not captured"
            ) from exc
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
            docker_lease.finish_provider_input(prompt)
        except OSError:
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


def _recorded_codex_cleanup_identity(
    launch_receipt: Mapping[str, object],
) -> tuple[Path, Path, str]:
    """Validate one immutable cleanup receipt without requiring live paths."""

    command = launch_receipt.get("command_receipt")
    cleanup = launch_receipt.get("cleanup_receipt")
    if not isinstance(command, Mapping):
        raise ValueError("recorded Docker cleanup command is unavailable")
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
    create_argv = command.get("create_argv")
    if not isinstance(create_argv, list) or any(
        not isinstance(item, str) for item in create_argv
    ):
        raise ValueError("recorded Docker cleanup command is invalid")
    try:
        if create_argv.count("--config") != 1 or create_argv.count(
            "--cidfile"
        ) != 1:
            raise ValueError("recorded Docker cleanup command is ambiguous")
        config_index = create_argv.index("--config") + 1
        cidfile_index = create_argv.index("--cidfile") + 1
        config_path = Path(create_argv[config_index])
        cidfile_path = Path(create_argv[cidfile_index])
    except (IndexError, ValueError) as exc:
        raise ValueError("recorded Docker cleanup lease is invalid") from exc
    container_name = str(launch_receipt.get("container_name") or "")
    lease_root = config_path.parent
    provider_home = Path(str(cleanup.get("provider_home") or ""))
    prompt_path = Path(str(cleanup.get("prompt_path") or ""))
    watchdog_pid = cleanup.get("watchdog_pid")
    watchdog_start_ticks = cleanup.get("watchdog_start_ticks")
    cleanup_root = lease_root.parent
    if (
        not config_path.is_absolute()
        or config_path.name != "docker-config"
        or cidfile_path != lease_root / "container.cid"
        or not cleanup_root.is_absolute()
        or not lease_root.name.startswith("asref-codex-container-")
        or _DOCKER_CONTAINER_NAME_RE.fullmatch(container_name) is None
        or create_argv.count("--name") != 1
        or create_argv.index("--name") + 1 >= len(create_argv)
        or create_argv[create_argv.index("--name") + 1] != container_name
        or cleanup.get("lease_root") != str(lease_root)
        or cleanup.get("docker_config") != str(config_path)
        or cleanup.get("cidfile") != str(cidfile_path)
        or not provider_home.is_absolute()
        or provider_home.parent != cleanup_root
        or not provider_home.name.startswith("asref-codex-home-")
        or not prompt_path.is_absolute()
        or prompt_path.parent != cleanup_root
        or not prompt_path.name.startswith("asref-grok-prompt-")
        or type(watchdog_pid) is not int
        or watchdog_pid <= 0
        or type(watchdog_start_ticks) is not int
        or watchdog_start_ticks < 0
    ):
        raise ValueError("recorded Docker cleanup lease identity is invalid")
    return lease_root, config_path, container_name


def _recorded_codex_lease_root(
    launch_receipt: Mapping[str, object],
) -> tuple[Path, Path, str]:
    """Recover the winner's live private watchdog lease from exact bytes."""

    lease_root, config_path, container_name = (
        _recorded_codex_cleanup_identity(launch_receipt)
    )
    cleanup = launch_receipt.get("cleanup_receipt")
    if not isinstance(cleanup, Mapping):
        raise ValueError("recorded Docker cleanup receipt is unavailable")
    _validated_docker_cleanup_root(
        lease_root=lease_root,
        provider_home=Path(str(cleanup.get("provider_home") or "")),
        prompt_path=Path(str(cleanup.get("prompt_path") or "")),
    )
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


def _recorded_codex_terminal_cleanup_evidence(
    launch_receipt: Mapping[str, object],
) -> dict[str, object]:
    """Read the exact live binding identities sealed by terminal CAS.

    This observation happens before the terminal provider transition, while
    the winner's private lease and command-bound cleanup record must still be
    present. It never treats a retained completion record as input authority.
    """

    lease_root, docker_config, container_name = _recorded_codex_lease_root(
        launch_receipt
    )
    cleanup = launch_receipt.get("cleanup_receipt")
    runtime = launch_receipt.get("runtime_receipt")
    if not isinstance(cleanup, Mapping) or not isinstance(runtime, Mapping):
        raise ValueError("recorded Docker cleanup authority is unavailable")
    docker_bin = str(runtime.get("path") or "")
    if docker_bin not in {"/usr/bin/docker", "/usr/local/bin/docker"}:
        raise ValueError("recorded Docker cleanup runtime is invalid")
    cidfile = lease_root / "container.cid"
    provider_home = Path(str(cleanup.get("provider_home") or ""))
    prompt_path = Path(str(cleanup.get("prompt_path") or ""))
    binding_path = _docker_cleanup_binding_path(
        container_name,
        create_directory=False,
    )
    if binding_path is None or not os.path.lexists(binding_path):
        raise ValueError("recorded Docker cleanup binding is unavailable")
    candidate = _read_private_control_record(
        binding_path.parent,
        binding_path.name,
    )
    if candidate is None:
        raise ValueError("recorded Docker cleanup binding disappeared")
    journal = _validated_docker_create_journal(
        lease_root=lease_root,
        provider="codex",
        docker_bin=docker_bin,
        docker_config=docker_config,
        container_name=container_name,
        cidfile=cidfile,
    )
    observation = candidate.get("effect_observation")
    raw_fence = candidate.get("termination_fence")
    try:
        runner_pid = int(candidate.get("runner_pid"))
        runner_start_ticks = int(candidate.get("runner_start_ticks"))
        watchdog_pid = int(cleanup.get("watchdog_pid"))
        watchdog_start_ticks = int(cleanup.get("watchdog_start_ticks"))
    except (TypeError, ValueError) as exc:
        raise ValueError("recorded Docker cleanup process identity is invalid") from exc
    if (
        journal is None
        or journal.get("state") != "create_observed"
        or not isinstance(observation, Mapping)
        or any(
            not isinstance(name, str) or not isinstance(value, str)
            for name, value in observation.items()
        )
        or not isinstance(raw_fence, Mapping)
    ):
        raise ValueError("recorded Docker cleanup binding is not terminalizable")
    admitted = _validated_cleanup_binding_record(
        binding_path,
        provider="codex",
        docker_bin=docker_bin,
        docker_config=docker_config,
        container_name=container_name,
        cidfile=cidfile,
        lease_root=lease_root,
        provider_home=provider_home,
        prompt_path=prompt_path,
        effect_observation=observation,  # type: ignore[arg-type]
        binding_state="command_bound",
        runner_pid=runner_pid,
        runner_start_ticks=runner_start_ticks,
        watchdog_pid=watchdog_pid,
        watchdog_start_ticks=watchdog_start_ticks,
        create_command_id=str(journal["command_id"]),
        create_cwd=Path(str(journal["cwd"])),
        create_environment_id=str(journal["environment_id"]),
        termination_fence=raw_fence,
    )
    fence_id = ""
    if raw_fence:
        fence = _validated_docker_termination_fence(
            raw_fence,
            provider="codex",
            container_name=container_name,
            expected_container_id=str(
                launch_receipt.get("container_id") or ""
            ).removeprefix("sha256:"),
            expected_image_id=str(launch_receipt.get("image_id") or ""),
        )
        fence_id = str(fence["fence_id"])
    return {
        "binding_path": str(binding_path),
        "binding_record_id": str(admitted["record_id"]),
        "termination_fence_id": fence_id,
    }


def _observed_provider_attempt_cleanup_state(
    attempt_observer: object | None,
    *,
    logical_attempt_id: str,
    lease_root: Path,
    docker_config: Path,
    container_name: str,
    watchdog_pid: int,
    watchdog_start_ticks: int,
) -> str:
    """Read one exact provider CAS without acquiring mutation authority."""

    if attempt_observer is None:
        return "unscoped"
    try:
        reservation = attempt_observer.observe(  # type: ignore[attr-defined]
            logical_attempt_id
        )
    except (OSError, ValueError):
        return "unknown"
    if reservation is None or reservation.state == "reserved":
        return "absent"
    if reservation.state not in {
        "effect_started",
        "quarantined",
        "terminal",
    }:
        return "unknown"
    launch_receipt = reservation.effect_launch_receipt
    cleanup = launch_receipt.get("cleanup_receipt")
    if not isinstance(cleanup, Mapping):
        return "unknown"
    try:
        observed_root, observed_config, observed_name = (
            _recorded_codex_cleanup_identity(launch_receipt)
        )
    except (OSError, ValueError):
        return "unknown"
    if (
        observed_root != lease_root
        or observed_config != docker_config
        or observed_name != container_name
        or cleanup.get("lease_root") != str(lease_root)
        or cleanup.get("docker_config") != str(docker_config)
        or cleanup.get("watchdog_pid") != watchdog_pid
        or cleanup.get("watchdog_start_ticks") != watchdog_start_ticks
    ):
        # The CAS admits exactly one cleanup receipt for this logical
        # attempt.  A different, fully validated immutable receipt proves
        # that this local lease lost before provider start and remains inert.
        # Local marker ownership still wins fail-closed in the caller.
        return "foreign"
    return "terminal" if reservation.state == "terminal" else "owned"


def _admit_terminal_cleanup_authority(
    *,
    launch_receipt: Mapping[str, object],
    terminal_observer: object,
    terminal_reservation: object,
) -> object:
    """Re-observe the immutable terminal provider CAS before destruction."""

    logical_attempt_id = str(
        getattr(terminal_reservation, "logical_attempt_id", "") or ""
    )
    observe = getattr(terminal_observer, "observe", None)
    if not logical_attempt_id or not callable(observe):
        raise ValueError("terminal Docker cleanup observer is unavailable")
    try:
        observed = observe(logical_attempt_id)
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError("terminal Docker cleanup CAS is unavailable") from exc
    observed_outcome = getattr(observed, "terminal_outcome", None)
    observed_returncode = getattr(observed, "terminal_returncode", None)
    observed_cleanup_authority = getattr(
        observed,
        "terminal_cleanup_authority",
        None,
    )
    cleanup_authority_fields = {
        "schema",
        "logical_attempt_id",
        "reservation_id",
        "cleanup_id",
        "binding_path",
        "binding_record_id",
        "termination_fence_id",
        "authority_id",
    }
    authority_body = (
        {
            name: item
            for name, item in observed_cleanup_authority.items()
            if name != "authority_id"
        }
        if isinstance(observed_cleanup_authority, Mapping)
        else {}
    )
    if (
        observed is None
        or getattr(observed, "state", "") != "terminal"
        or getattr(observed, "terminal", False) is not True
        or (
            getattr(observed, "content_id", "")
            != getattr(terminal_reservation, "content_id", "")
            and (
                getattr(observed, "terminal_outcome_id", "")
                != getattr(terminal_reservation, "terminal_outcome_id", "")
                or getattr(observed, "terminal_cleanup_authority", None)
                != getattr(
                    terminal_reservation,
                    "terminal_cleanup_authority",
                    None,
                )
            )
        )
        or getattr(observed, "reservation_id", "")
        != getattr(terminal_reservation, "reservation_id", "")
        or getattr(observed, "effect_launch_receipt", None)
        != dict(launch_receipt)
        or not isinstance(observed_outcome, Mapping)
        or observed_outcome.get("reservation_id")
        != getattr(observed, "reservation_id", "")
        or observed_outcome.get("effect_launch_receipt")
        != getattr(observed, "effect_launch_receipt", None)
        or not isinstance(
            observed_outcome.get("fallback_dispatched"),
            bool,
        )
        or isinstance(observed_returncode, bool)
        or not isinstance(observed_returncode, int)
        or observed_outcome.get("fallback_returncode") != observed_returncode
        or not isinstance(observed_cleanup_authority, Mapping)
        or set(observed_cleanup_authority) != cleanup_authority_fields
        or observed_cleanup_authority.get("schema")
        != (
            "ipfs_accelerate_py/agent-supervisor/"
            "terminal-cleanup-authority@1"
        )
        or observed_cleanup_authority.get("logical_attempt_id")
        != logical_attempt_id
        or observed_cleanup_authority.get("reservation_id")
        != getattr(observed, "reservation_id", "")
        or observed_cleanup_authority.get("cleanup_id")
        != launch_receipt.get("cleanup_id")
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(observed_cleanup_authority.get("binding_record_id") or ""),
        )
        is None
        or (
            observed_outcome.get("fallback_dispatched") is True
            and re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                str(
                    observed_cleanup_authority.get(
                        "termination_fence_id"
                    )
                    or ""
                ),
            )
            is None
        )
        or (
            observed_outcome.get("fallback_dispatched") is False
            and observed_cleanup_authority.get("termination_fence_id") != ""
        )
        or observed_cleanup_authority.get("authority_id")
        != _effect_receipt_identity(authority_body)
    ):
        raise ValueError("terminal Docker cleanup CAS authority drifted")
    return observed


def _release_recorded_codex_effect_cleanup(
    launch_receipt: Mapping[str, object],
    *,
    terminal_observer: object,
    terminal_reservation: object,
) -> None:
    """Idempotently reap exact receipt-bound resources after CAS terminal."""

    admitted_terminal = _admit_terminal_cleanup_authority(
        launch_receipt=launch_receipt,
        terminal_observer=terminal_observer,
        terminal_reservation=terminal_reservation,
    )
    terminal_cleanup_authority = getattr(
        admitted_terminal,
        "terminal_cleanup_authority",
    )

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
    try:
        cleanup_root, cleanup_root_identity = (
            _validated_docker_cleanup_root(
                lease_root=lease_root,
                provider_home=provider_home,
                prompt_path=prompt_path,
            )
        )
    except ValueError as exc:
        raise ValueError("recorded Docker cleanup root is invalid") from exc
    if (
        not lease_root.name.startswith("asref-codex-container-")
        or docker_config != lease_root / "docker-config"
        or cidfile != lease_root / "container.cid"
        or not provider_home.name.startswith("asref-codex-home-")
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
    runtime = launch_receipt.get("runtime_receipt")
    docker_bin = str(runtime.get("path") or "") if isinstance(runtime, Mapping) else ""
    if docker_bin not in {"/usr/bin/docker", "/usr/local/bin/docker"}:
        raise ValueError("recorded Docker cleanup runtime is invalid")
    binding_path = _docker_cleanup_binding_path(
        container_name,
        create_directory=False,
    )
    if binding_path is None:
        raise ValueError("recorded Docker cleanup binding path is absent")
    completion_path = _cleanup_completion_path(binding_path)
    try:
        expected_lifecycle: dict[str, object] = {
            "run_id": os.environ[RUN_ID_ENV],
            "profile_id": os.environ[PROFILE_ID_ENV],
            "target_id": os.environ[TARGET_ID_ENV],
            "repository_root": os.environ[REPOSITORY_ROOT_ENV],
            "state_root": os.environ[STATE_ROOT_ENV],
            "run_root": os.environ[RUN_ROOT_ENV],
            "configuration_root": os.environ[CONFIGURATION_ROOT_ENV],
            "fencing_epoch": int(os.environ[FENCING_EPOCH_ENV]),
        }
    except (KeyError, ValueError) as exc:
        raise ValueError("recorded Docker cleanup lifecycle is unavailable") from exc
    if not os.path.lexists(binding_path):
        if (
            not os.path.lexists(completion_path)
            or not os.path.lexists(_cleanup_authority_path(binding_path))
        ):
            raise ValueError(
                "recorded Docker cleanup lost its terminal fence receipt"
            )
        completion = _read_private_control_record(
            completion_path.parent,
            completion_path.name,
        )
        completed_binding = (
            completion.get("binding_record")
            if isinstance(completion, Mapping)
            else None
        )
        completed_fence = (
            completed_binding.get("termination_fence")
            if isinstance(completed_binding, Mapping)
            else None
        )
        recorded_container_id = str(
            launch_receipt.get("container_id") or ""
        ).removeprefix("sha256:")
        recorded_image_id = str(launch_receipt.get("image_id") or "")
        terminal_fence_id = str(
            terminal_cleanup_authority.get("termination_fence_id") or ""
        )
        if (
            not isinstance(completed_binding, Mapping)
            or not isinstance(completed_fence, Mapping)
            or bool(completed_fence) != bool(terminal_fence_id)
            or completed_binding.get("record_id")
            != terminal_cleanup_authority.get("binding_record_id")
            or completed_binding.get("binding_path") != str(binding_path)
            or terminal_cleanup_authority.get("binding_path")
            != str(binding_path)
            or completed_binding.get("provider") != "codex"
            or completed_binding.get("docker_bin") != docker_bin
            or completed_binding.get("container_name") != container_name
            or completed_binding.get("cleanup_root") != str(cleanup_root)
            or completed_binding.get("cleanup_root_identity")
            != cleanup_root_identity
            or completed_binding.get("lease_root") != str(lease_root)
            or completed_binding.get("docker_config") != str(docker_config)
            or completed_binding.get("cidfile") != str(cidfile)
            or completed_binding.get("provider_home") != str(provider_home)
            or completed_binding.get("prompt_path") != str(prompt_path)
            or completed_binding.get("watchdog_pid") != watchdog_pid
            or completed_binding.get("watchdog_start_ticks")
            != watchdog_start_ticks
            or any(
                completed_binding.get(name) != expected
                for name, expected in expected_lifecycle.items()
            )
        ):
            raise ValueError("recorded Docker cleanup completion lacks authority")
        admitted_fence: Mapping[str, object] | None = None
        if completed_fence:
            admitted_fence = _validated_docker_termination_fence(
                completed_fence,
                provider="codex",
                container_name=container_name,
                expected_container_id=recorded_container_id,
                expected_image_id=recorded_image_id,
            )
            if admitted_fence.get("fence_id") != terminal_fence_id:
                raise ValueError(
                    "recorded Docker cleanup completion fence drifted"
                )
        # This is an idempotent replay after a durable completion.  It may
        # observe the exact CID/name/scope but must never dispatch rm again.
        recovery_config = Path(
            tempfile.mkdtemp(prefix="aseh-docker-recovery-config-")
        )
        recovery_config.chmod(0o700)
        try:
            _remove_exact_docker_container(
                docker_bin=docker_bin,
                docker_config=recovery_config,
                container_name=container_name,
                settle_for_creation=False,
                termination_fence=admitted_fence,
                issue_removal=False,
            )
        finally:
            shutil.rmtree(recovery_config, ignore_errors=True)
        completed_identity = completion.get("binding_identity")
        if not isinstance(completed_identity, Mapping) or not (
            _finalize_verified_cleanup_completion(
                binding_path=binding_path,
                binding_identity=completed_identity,
                binding_record=completed_binding,
                expected_lifecycle=expected_lifecycle,
                terminal_cleanup_store=terminal_observer,
                terminal_cleanup_reservation=admitted_terminal,
            )
        ):
            raise ValueError("recorded Docker cleanup completion is invalid")
        return
    candidate = _read_private_control_record(
        binding_path.parent,
        binding_path.name,
    )
    if candidate is None:
        raise ValueError("recorded Docker cleanup binding disappeared")
    candidate_body = {
        key: item for key, item in candidate.items() if key != "record_id"
    }
    try:
        runner_pid = int(candidate.get("runner_pid"))
        runner_start_ticks = int(candidate.get("runner_start_ticks"))
    except (TypeError, ValueError) as exc:
        raise ValueError("recorded Docker cleanup runner is invalid") from exc
    path_identities = candidate.get("path_identities")
    if (
        candidate.get("schema") != _DOCKER_CLEANUP_BINDING_SCHEMA
        or candidate.get("record_id") != _effect_receipt_identity(candidate_body)
        or candidate.get("record_id")
        != terminal_cleanup_authority.get("binding_record_id")
        or candidate.get("binding_path") != str(binding_path)
        or terminal_cleanup_authority.get("binding_path") != str(binding_path)
        or candidate.get("provider") != "codex"
        or candidate.get("docker_bin") != docker_bin
        or candidate.get("container_name") != container_name
        or candidate.get("cleanup_root") != str(cleanup_root)
        or candidate.get("cleanup_root_identity") != cleanup_root_identity
        or candidate.get("lease_root") != str(lease_root)
        or candidate.get("docker_config") != str(docker_config)
        or candidate.get("cidfile") != str(cidfile)
        or candidate.get("provider_home") != str(provider_home)
        or candidate.get("prompt_path") != str(prompt_path)
        or candidate.get("watchdog_pid") != watchdog_pid
        or candidate.get("watchdog_start_ticks") != watchdog_start_ticks
        or any(
            candidate.get(name) != expected
            for name, expected in expected_lifecycle.items()
        )
        or not isinstance(path_identities, dict)
    ):
        raise ValueError("recorded Docker cleanup binding drifted")
    raw_fence = candidate.get("termination_fence")
    terminal_fence_id = str(
        terminal_cleanup_authority.get("termination_fence_id") or ""
    )
    if (
        not isinstance(raw_fence, Mapping)
        or bool(raw_fence) != bool(terminal_fence_id)
    ):
        raise ValueError("recorded Docker effect cleanup fence differs")
    admitted_fence = None
    if raw_fence:
        admitted_fence = _validated_docker_termination_fence(
            raw_fence,
            provider="codex",
            container_name=container_name,
            expected_container_id=str(
                launch_receipt.get("container_id") or ""
            ).removeprefix("sha256:"),
            expected_image_id=str(launch_receipt.get("image_id") or ""),
        )
        if admitted_fence.get("fence_id") != terminal_fence_id:
            raise ValueError("recorded Docker cleanup terminal fence drifted")
    binding_identity = _cleanup_path_identity(
        binding_path,
        directory=False,
    )
    if not lease_root.exists():
        # A prior cleanup may have removed the lease and then crashed before
        # retiring its binding.  Only exact inode tombstones—not mere path
        # absence—admit that crash gap.
        recovery_config = Path(
            tempfile.mkdtemp(prefix="aseh-docker-recovery-config-")
        )
        recovery_config.chmod(0o700)
        try:
            _remove_exact_docker_container(
                docker_bin=docker_bin,
                docker_config=recovery_config,
                container_name=container_name,
                settle_for_creation=False,
                termination_fence=admitted_fence,
                issue_removal=False,
            )
        finally:
            shutil.rmtree(recovery_config, ignore_errors=True)
        if not _finalize_verified_cleanup_completion(
            binding_path=binding_path,
            binding_identity=binding_identity,
            binding_record=candidate,
            expected_lifecycle=expected_lifecycle,
            terminal_cleanup_store=terminal_observer,
            terminal_cleanup_reservation=admitted_terminal,
        ):
            raise ValueError("recorded Docker cleanup did not converge")
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
    journal = _validated_docker_create_journal(
        lease_root=lease_root,
        provider="codex",
        docker_bin=docker_bin,
        docker_config=docker_config,
        container_name=container_name,
        cidfile=cidfile,
    )
    if journal is None or journal.get("state") != "create_observed":
        raise ValueError("recorded Docker create is not terminal")
    observation = candidate.get("effect_observation")
    candidate_termination_fence = candidate.get("termination_fence")
    if not isinstance(observation, dict) or any(
        not isinstance(name, str) or not isinstance(value, str)
        for name, value in observation.items()
    ) or not isinstance(candidate_termination_fence, Mapping):
        raise ValueError("recorded Docker cleanup observation is invalid")
    admitted = _validated_cleanup_binding_record(
        binding_path,
        provider="codex",
        docker_bin=docker_bin,
        docker_config=docker_config,
        container_name=container_name,
        cidfile=cidfile,
        lease_root=lease_root,
        provider_home=provider_home,
        prompt_path=prompt_path,
        effect_observation=observation,
        binding_state="command_bound",
        runner_pid=runner_pid,
        runner_start_ticks=runner_start_ticks,
        create_command_id=str(journal["command_id"]),
        create_cwd=Path(str(journal["cwd"])),
        create_environment_id=str(journal["environment_id"]),
        watchdog_pid=watchdog_pid,
        watchdog_start_ticks=watchdog_start_ticks,
        termination_fence=candidate_termination_fence,
    )
    path_identities = admitted.get("path_identities")
    if not isinstance(path_identities, dict):
        raise ValueError("recorded Docker cleanup path authority is invalid")
    watchdog_boot_id = str(admitted.get("boot_id") or "")
    try:
        current_boot_id = Path(
            "/proc/sys/kernel/random/boot_id"
        ).read_text(encoding="ascii").strip()
        watchdog_birth = read_process_birth(watchdog_pid)
    except (OSError, UnicodeError) as exc:
        raise ValueError("recorded Docker watchdog liveness is unknown") from exc
    exact_watchdog: _DetachedDockerCleanupWatchdog | None = None
    if (
        current_boot_id == watchdog_boot_id
        and watchdog_birth is not None
        and watchdog_birth.start_time_ticks == watchdog_start_ticks
        and watchdog_birth.boot_id == watchdog_boot_id
    ):
        exact_watchdog = _DetachedDockerCleanupWatchdog(
            watchdog_pid,
            watchdog_start_ticks,
        )
        exact_watchdog.terminate()
        for _ in range(20):
            if exact_watchdog.poll() is not None:
                break
            time.sleep(0.05)
        try:
            still_live = read_process_birth(watchdog_pid)
        except OSError as exc:
            raise ValueError("recorded Docker watchdog exit is unknown") from exc
        if (
            still_live is not None
            and still_live.start_time_ticks == watchdog_start_ticks
            and still_live.boot_id == watchdog_boot_id
        ):
            raise ValueError("recorded Docker watchdog cleanup remains pending")
    if not lease_root.exists():
        recovery_config = Path(
            tempfile.mkdtemp(prefix="aseh-docker-recovery-config-")
        )
        recovery_config.chmod(0o700)
        try:
            _remove_exact_docker_container(
                docker_bin=docker_bin,
                docker_config=recovery_config,
                container_name=container_name,
                settle_for_creation=False,
                termination_fence=admitted_fence,
                issue_removal=False,
            )
        finally:
            shutil.rmtree(recovery_config, ignore_errors=True)
        if not _finalize_verified_cleanup_completion(
            binding_path=binding_path,
            binding_identity=binding_identity,
            binding_record=candidate,
            expected_lifecycle=expected_lifecycle,
            terminal_cleanup_store=terminal_observer,
            terminal_cleanup_reservation=admitted_terminal,
        ):
            raise ValueError("recorded Docker cleanup did not converge")
        return
    # Revalidate all inode authority after any concurrent watchdog activity
    # and immediately before the exact Docker/path cleanup boundary.
    current = _validated_cleanup_binding_record(
        binding_path,
        provider="codex",
        docker_bin=docker_bin,
        docker_config=docker_config,
        container_name=container_name,
        cidfile=cidfile,
        lease_root=lease_root,
        provider_home=provider_home,
        prompt_path=prompt_path,
        effect_observation=observation,
        binding_state="command_bound",
        runner_pid=runner_pid,
        runner_start_ticks=runner_start_ticks,
        create_command_id=str(journal["command_id"]),
        create_cwd=Path(str(journal["cwd"])),
        create_environment_id=str(journal["environment_id"]),
        watchdog_pid=watchdog_pid,
        watchdog_start_ticks=watchdog_start_ticks,
        termination_fence=admitted_fence,
    )
    if current.get("record_id") != admitted.get("record_id"):
        raise ValueError("recorded Docker cleanup binding changed")
    current_identity = _cleanup_path_identity(
        binding_path,
        directory=False,
    )
    issue_removal = bool(
        admitted_fence is not None
        and _arm_docker_removal_once(
            binding_path=binding_path,
            expected_binding_identity=current_identity,
            binding_record=current,
            termination_fence=admitted_fence,
        )
    )
    _remove_exact_docker_container(
        docker_bin=docker_bin,
        docker_config=docker_config,
        container_name=container_name,
        settle_for_creation=False,
        termination_fence=admitted_fence,
        issue_removal=issue_removal,
    )
    if not _finalize_verified_cleanup_completion(
        binding_path=binding_path,
        binding_identity=current_identity,
        binding_record=current,
        expected_lifecycle=expected_lifecycle,
        terminal_cleanup_store=terminal_observer,
        terminal_cleanup_reservation=admitted_terminal,
    ):
        raise ValueError("recorded Docker cleanup did not converge")


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


def _recorded_codex_running_fence(
    launch_receipt: Mapping[str, object],
    *,
    capture_if_absent: bool,
) -> Mapping[str, object]:
    """Admit a persisted fence, or capture one for a newly started container."""

    lease_root, docker_config, container_name = _recorded_codex_lease_root(
        launch_receipt
    )
    binding_path = _docker_cleanup_binding_path(
        container_name,
        create_directory=False,
    )
    if binding_path is None:
        raise ValueError("recorded Docker start binding is absent")
    candidate = _read_private_control_record(
        binding_path.parent,
        binding_path.name,
    )
    if candidate is None:
        raise ValueError("recorded Docker start binding disappeared")
    cleanup = launch_receipt.get("cleanup_receipt")
    if not isinstance(cleanup, Mapping):
        raise ValueError("recorded Docker cleanup receipt is absent")
    provider_home = Path(str(cleanup.get("provider_home") or ""))
    prompt_path = Path(str(cleanup.get("prompt_path") or ""))
    cidfile = Path(str(cleanup.get("cidfile") or ""))
    journal = _validated_docker_create_journal(
        lease_root=lease_root,
        provider="codex",
        docker_bin=str(candidate.get("docker_bin") or ""),
        docker_config=docker_config,
        container_name=container_name,
        cidfile=cidfile,
    )
    effect_observation = candidate.get("effect_observation")
    raw_fence = candidate.get("termination_fence")
    if (
        journal is None
        or journal.get("state") != "create_observed"
        or candidate.get("binding_state") != "command_bound"
        or not isinstance(effect_observation, dict)
        or not isinstance(raw_fence, Mapping)
    ):
        raise ValueError("recorded Docker start authority is invalid")
    admitted = _validated_cleanup_binding_record(
        binding_path,
        provider="codex",
        docker_bin=str(candidate.get("docker_bin") or ""),
        docker_config=docker_config,
        container_name=container_name,
        cidfile=cidfile,
        lease_root=lease_root,
        provider_home=provider_home,
        prompt_path=prompt_path,
        effect_observation=effect_observation,
        binding_state="command_bound",
        runner_pid=int(candidate.get("runner_pid") or 0),
        runner_start_ticks=int(candidate.get("runner_start_ticks") or 0),
        watchdog_pid=int(candidate.get("watchdog_pid") or 0),
        watchdog_start_ticks=int(candidate.get("watchdog_start_ticks") or 0),
        create_command_id=str(journal["command_id"]),
        create_cwd=Path(str(journal["cwd"])),
        create_environment_id=str(journal["environment_id"]),
        termination_fence=raw_fence,
    )
    container_id = str(launch_receipt.get("container_id") or "").removeprefix(
        "sha256:"
    )
    image_id = str(launch_receipt.get("image_id") or "")
    if raw_fence:
        fence = _validated_docker_termination_fence(
            raw_fence,
            provider="codex",
            container_name=container_name,
            expected_container_id=container_id,
            expected_image_id=image_id,
        )
    else:
        if not capture_if_absent:
            raise ValueError(
                "running Docker adoption lacks its persisted termination fence"
            )
        fence = _attest_exact_docker_execution(
            docker_bin=str(admitted["docker_bin"]),
            docker_config=docker_config,
            provider="codex",
            container_name=container_name,
            container_id=container_id,
            image_id=image_id,
            timeout=2.0,
        )
        if fence.get("docker_state") != "running" or int(
            fence.get("init_pid") or 0
        ) <= 0:
            raise ValueError("recorded Docker provider start is not running")
        admitted, _admitted_identity = _publish_docker_termination_fence_binding(
            record_path=binding_path,
            expected_record_id=str(admitted["record_id"]),
            expected_identity=_cleanup_path_identity(
                binding_path,
                directory=False,
            ),
            provider="codex",
            docker_bin=str(admitted["docker_bin"]),
            docker_config=docker_config,
            container_name=container_name,
            cidfile=cidfile,
            lease_root=lease_root,
            provider_home=provider_home,
            prompt_path=prompt_path,
            effect_observation=effect_observation,
            runner_pid=int(admitted["runner_pid"]),
            runner_start_ticks=int(admitted["runner_start_ticks"]),
            watchdog_pid=int(admitted["watchdog_pid"]),
            watchdog_start_ticks=int(admitted["watchdog_start_ticks"]),
            create_command_id=str(journal["command_id"]),
            create_cwd=Path(str(journal["cwd"])),
            create_environment_id=str(journal["environment_id"]),
            termination_fence=fence,
        )
    return fence


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
    from .process_security import (
        require_state_authority_handoff_ptrace_protection,
    )

    # This adoption owner mints a new, process-local capability. Apply the
    # same same-UID descriptor-duplication prerequisite as an ordinary lease
    # immediately before the socket exists.
    require_state_authority_handoff_ptrace_protection()
    provider_sender, provider_stdin = _provider_start_socketpair()
    try:
        try:
            process = subprocess.Popen(
                list(command),
                env=_docker_control_env(),
                stdin=provider_stdin,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        finally:
            provider_stdin.close()
    except BaseException:
        provider_sender.close()
        raise
    if process.stdout is None or process.stderr is None:
        provider_sender.close()
        raise ValueError("recorded Docker start pipes were not created")
    try:
        _recorded_codex_running_fence(
            launch_receipt,
            capture_if_absent=True,
        )
        provider_sender.sendall(_DOCKER_PROVIDER_START_MARKER)
        if prompt:
            provider_sender.sendall(prompt.encode("utf-8"))
        try:
            provider_sender.shutdown(socket.SHUT_WR)
        except OSError:
            pass
    except BaseException:
        provider_sender.close()
        try:
            process.terminate()
            process.wait(timeout=2.0)
        except (OSError, subprocess.TimeoutExpired):
            try:
                process.kill()
            except OSError:
                pass
        raise
    provider_sender.close()
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
    returncode = int(process.wait())
    stdout_thread.join()
    stderr_thread.join()
    return returncode


def _validate_codex_quota_fallback_reasoning_effort(value: object) -> str:
    """Return a closed Codex fallback effort or reject configuration drift."""

    effort = str(value).strip() if isinstance(value, str) else ""
    if effort not in CODEX_QUOTA_FALLBACK_REASONING_EFFORTS:
        allowed = ", ".join(sorted(CODEX_QUOTA_FALLBACK_REASONING_EFFORTS))
        raise ValueError(
            "Codex quota fallback reasoning must be one of: " + allowed
        )
    return effort


def _parse_codex_fallback_command(
    raw: str,
    *,
    expected_fallback_reasoning_effort: str = (
        DEFAULT_CODEX_QUOTA_FALLBACK_REASONING_EFFORT
    ),
) -> list[str]:
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
    _validate_codex_quota_fallback_command(
        command,
        expected_fallback_reasoning_effort=expected_fallback_reasoning_effort,
    )
    return command


def _validate_codex_quota_fallback_command(
    command: Sequence[str],
    *,
    workspace: Path | None = None,
    required_reasoning_effort: str | None = None,
    expected_fallback_reasoning_effort: str | None = None,
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
    if (
        required_reasoning_effort is not None
        and expected_fallback_reasoning_effort is not None
        and required_reasoning_effort != expected_fallback_reasoning_effort
    ):
        raise ValueError("Codex fallback reasoning requirements disagree")
    selected_reasoning_effort = (
        required_reasoning_effort
        if required_reasoning_effort is not None
        else expected_fallback_reasoning_effort
    )
    if selected_reasoning_effort is not None:
        selected_reasoning_effort = (
            _validate_codex_quota_fallback_reasoning_effort(
                selected_reasoning_effort
            )
        )
    if selected_reasoning_effort is not None and configs.get(
        "model_reasoning_effort"
    ) != json.dumps(selected_reasoning_effort):
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
    provider_stdin: socket.socket | None = None,
) -> int:
    """Run Grok with live output; stdout never grants fallback authority."""

    try:
        process = subprocess.Popen(
            list(command),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            **({"stdin": provider_stdin} if provider_stdin is not None else {}),
        )
    finally:
        if provider_stdin is not None:
            provider_stdin.close()
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
                verifier_workspace=verifier_workspace,
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
                reservation.effect_launch_receipt,
                terminal_observer=store,
                terminal_reservation=reservation,
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
            _recorded_codex_running_fence(
                active.effect_launch_receipt,
                capture_if_absent=False,
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
            terminal_cleanup_evidence=(
                _recorded_codex_terminal_cleanup_evidence(
                    active.effect_launch_receipt
                )
            ),
        )
        _release_recorded_codex_effect_cleanup(
            terminal.effect_launch_receipt,
            terminal_observer=store,
            terminal_reservation=terminal,
        )
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
            str(args.codex_fallback_command_json),
            expected_fallback_reasoning_effort=(
                args.codex_fallback_reasoning_effort
            ),
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
                                    existing.effect_launch_receipt,
                                    terminal_observer=attempt_store,
                                    terminal_reservation=existing,
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
                terminal_cleanup_evidence=(
                    _recorded_codex_terminal_cleanup_evidence(
                        attempt_reservation.effect_launch_receipt
                    )
                ),
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
                _recorded_codex_running_fence(
                    attempt_reservation.effect_launch_receipt,
                    capture_if_absent=False,
                )
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
                terminal_cleanup_evidence=(
                    _recorded_codex_terminal_cleanup_evidence(
                        attempt_reservation.effect_launch_receipt
                    )
                ),
            )
            _release_recorded_codex_effect_cleanup(
                attempt_reservation.effect_launch_receipt,
                terminal_observer=attempt_store,
                terminal_reservation=attempt_reservation,
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
                effect_claim=(
                    claim_provider_effect
                    if invocation_binding is not None
                    else None
                ),
                effect_terminal=(
                    complete_provider_effect
                    if invocation_binding is not None
                    else None
                ),
                effect_observation=(
                    {
                        "logical_attempt_id": (
                            invocation_binding.logical_attempt_id
                        ),
                        "provider_attempt_store": (
                            invocation_binding.provider_attempt_store
                        ),
                        "provider_attempt_store_identity": (
                            invocation_binding.provider_attempt_store_identity
                        ),
                    }
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
    docker_fence_thread: threading.Thread | None = None
    docker_fence_failures: list[BaseException] = []
    grok_launch_env: dict[str, str] = {}
    command_environment_stack = ExitStack()
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix="asref-grok-prompt-",
            suffix=".txt",
            delete=False,
        ) as handle:
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
                _populate_bound_ephemeral_prompt(Path(prompt_path), prompt)
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
                _populate_bound_ephemeral_prompt(Path(prompt_path), prompt)
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
                populate_credentials=False,
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
                _populate_bound_ephemeral_prompt(Path(prompt_path), prompt)
                _populate_isolated_grok_credentials(
                    base_env=base_env,
                    grok_home=_policy_path.parent,
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
                docker_lease.bind_isolation_image(isolation_image)
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
            else:
                # The native sandbox is used only when no pinned Docker
                # boundary is available.  Keep the population immediately
                # adjacent to launch; Docker routes always use the durable
                # prepared cleanup binding above.
                _populate_bound_ephemeral_prompt(Path(prompt_path), prompt)
                _populate_isolated_grok_credentials(
                    base_env=base_env,
                    grok_home=_policy_path.parent,
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
        docker_provider_stdin: socket.socket | None = None
        if docker_lease is not None:
            try:
                docker_provider_stdin = docker_lease.take_provider_start_stdin()
            except (OSError, ValueError) as exc:
                docker_lease.preserve_for_recovery = True
                print(
                    f"Grok Docker provider-start capability failed: {exc}",
                    file=sys.stderr,
                )
                return 125
        capture_running_fence = (
            getattr(docker_lease, "capture_running_termination_fence", None)
            if docker_lease is not None
            else None
        )
        if callable(capture_running_fence):

            def capture_docker_fence() -> None:
                try:
                    capture_running_fence()
                except BaseException as exc:
                    docker_fence_failures.append(exc)
                    docker_lease._abort_provider_start()

            # The capture thread begins against the inert ``created`` state
            # and observes the active init immediately after the attached
            # start command below crosses Docker's start boundary.
            docker_fence_thread = threading.Thread(
                target=capture_docker_fence,
                name="docker-kernel-fence-capture",
                daemon=True,
            )
            docker_fence_thread.start()
        # Without an authorized Codex fallback, project typed quota receipts and
        # exit. With a fallback, take the workspace-fenced + independent-verify
        # path so Terra may run only after verified typed provider evidence.
        if not codex_fallback_command:
            child_returncode, error_bytes, error_size, error_overflow = (
                _run_grok_with_bounded_stderr(
                    cmd,
                    env=grok_launch_env,
                    provider_stdin=docker_provider_stdin,
                )
            )
            if docker_fence_thread is not None:
                docker_fence_thread.join(timeout=6.0)
                if docker_fence_thread.is_alive() or docker_fence_failures:
                    docker_lease.preserve_for_recovery = True
                    print(
                        "Grok Docker kernel cleanup fence was not captured",
                        file=sys.stderr,
                    )
                    return 125
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
            primary_returncode = _run_grok_with_typed_failure_capture(
                cmd,
                env=grok_launch_env,
                provider_stdin=docker_provider_stdin,
            )
            if docker_fence_thread is not None:
                docker_fence_thread.join(timeout=6.0)
                if docker_fence_thread.is_alive() or docker_fence_failures:
                    docker_lease.preserve_for_recovery = True
                    print(
                        "Grok Docker kernel cleanup fence was not captured",
                        file=sys.stderr,
                    )
                    return 125
            docker_run_finished = True
        except OSError as exc:
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
        if docker_fence_thread is not None and docker_fence_thread.is_alive():
            docker_fence_thread.join(timeout=0.5)
            if docker_fence_thread.is_alive() and docker_lease is not None:
                docker_lease.preserve_for_recovery = True
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
        "--codex-fallback-reasoning-effort",
        default=DEFAULT_CODEX_QUOTA_FALLBACK_REASONING_EFFORT,
        choices=tuple(sorted(CODEX_QUOTA_FALLBACK_REASONING_EFFORTS)),
        help="Exact closed reasoning effort bound into the Codex fallback argv.",
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
    if (
        len(sys.argv) > 1
        and sys.argv[1] == _DOCKER_REMOVAL_ISSUER_LAUNCHER_ARG
    ):
        raise SystemExit(_docker_removal_issuer_launcher_main(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == _DOCKER_REMOVAL_ISSUER_ARG:
        raise SystemExit(_docker_removal_issuer_main(sys.argv[2:]))
    if (
        len(sys.argv) > 1
        and sys.argv[1] == _DOCKER_CLEANUP_WATCHDOG_LAUNCHER_ARG
    ):
        raise SystemExit(_docker_cleanup_watchdog_launcher_main(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == _DOCKER_CLEANUP_WATCHDOG_ARG:
        raise SystemExit(_docker_cleanup_watchdog_main(sys.argv[2:]))
    raise SystemExit(main())
