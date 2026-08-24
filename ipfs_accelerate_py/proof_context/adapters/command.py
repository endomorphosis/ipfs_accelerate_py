"""Hermetic argv-only local coding-agent adapter (PCCE-032).

This is deliberately a small subprocess boundary: a policy names exact binary
paths and working directories, no parent environment is inherited, and every
child is placed in its own process group.  It accepts exactly one JSON object
on stdout; diagnostics are never interpreted as a proposal.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from ipfs_accelerate_py.proof_context.adapters.base import (
    APPROVAL_AUTHORITY, CANONICAL_BRANCH_AUTHORITY, AdapterResult,
    CancellationToken, admit_adapter_result, bind_adapter_request,
)
from ipfs_accelerate_py.proof_context.adapters.models import (
    CODING_AGENT_INVOCATION_SCHEMA, MAX_LOG_BYTES, MAX_PROVIDER_OUTPUT_BYTES,
    PATCH_PROPOSAL_SCHEMA, CodingAgentInvocation, ContextPack,
    ModelRouteDecision, PatchProposal, TaskSpecification, admit_bounded_patch,
    admit_cid, admit_non_negative_int, admit_path_list, assert_declared_scope,
    admit_relative_path, wire_canonical_utf8,
)
from ipfs_accelerate_py.proof_context.errors import (
    BoundaryViolationError, IdentityInconsistentError, MalformedError,
    ProofCancelledError, ProofTimeoutError, UnavailableCapabilityError,
    redact_text,
)

ADAPTER: Final[str] = "CommandAdapter@0.1"
COMMAND_CONTRACT: Final[str] = "local-agent-json-argv@1"
REQUEST_SCHEMA: Final[str] = "ipfs-accelerate.proof-context.v0.1/command-request@1"
MAX_ARGUMENTS: Final[int] = 64
MAX_ARGUMENT_BYTES: Final[int] = 16_384
_OUTPUT_FIELDS: Final[frozenset[str]] = frozenset({
    "task_id", "repository_state_cid", "pack_cid", "route_cid", "declared_files",
    "patch", "model", "revision", "token_count", "cached_token_count", "latency_ms",
    "cost_micros",
})
_SECRET = re.compile(r"(?i)(api[_-]?key|access[_-]?token|refresh[_-]?token|token|secret|password|authorization|credential)s?\\s*[:=]\\s*\\S+")
_PATCH_PATH = re.compile(r"^diff --git a/(.+?) b/(.+)$", re.MULTILINE)
_SAFE_ENVIRONMENT_KEYS: Final[frozenset[str]] = frozenset({"TERM", "TZ", "NO_COLOR"})


def _cid(value: Any) -> str:
    raw = wire_canonical_utf8(value).encode("utf-8")
    digest = hashlib.sha256(raw).digest()
    return "b" + base64.b32encode(b"\x01\x55\x12\x20" + digest).decode().lower().rstrip("=")


def _path(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise MalformedError(f"{field_name} must be a non-empty path")
    if not os.path.isabs(value):
        raise BoundaryViolationError(f"{field_name} must be absolute")
    resolved = os.path.realpath(value)
    if resolved != value or not os.path.exists(resolved):
        raise BoundaryViolationError(f"{field_name} must be an existing canonical path")
    return resolved


def redact_command_log(value: bytes | str) -> bytes:
    text = value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)
    text = _SECRET.sub("[redacted]", text.replace("\x00", ""))
    text = redact_text(text)
    return text.encode("utf-8")[:MAX_LOG_BYTES]


@dataclass(frozen=True)
class CommandPolicy:
    """Immutable explicit inputs controlling the sole executable and cwd."""

    executable: str
    allowed_executables: tuple[str, ...]
    cwd: str
    allowed_cwds: tuple[str, ...]
    arguments: tuple[str, ...] = ()
    environment: Mapping[str, str] = field(default_factory=dict)
    timeout_seconds: float = 120.0

    def __post_init__(self) -> None:
        executable = _path(self.executable, field_name="executable")
        allowed = tuple(_path(item, field_name="allowed_executables") for item in self.allowed_executables)
        cwd = _path(self.cwd, field_name="cwd")
        cwds = tuple(_path(item, field_name="allowed_cwds") for item in self.allowed_cwds)
        if (executable not in allowed or cwd not in cwds or not os.path.isdir(cwd)
                or not os.path.isfile(executable) or not os.access(executable, os.X_OK)):
            raise BoundaryViolationError("command policy is outside its immutable allowlist")
        if not 0 < self.timeout_seconds <= 3600:
            raise MalformedError("timeout_seconds is outside the safe bound")
        if len(self.arguments) > MAX_ARGUMENTS:
            raise BoundaryViolationError("too many command arguments")
        args: list[str] = []
        for arg in self.arguments:
            if not isinstance(arg, str) or "\x00" in arg or len(arg.encode("utf-8")) > MAX_ARGUMENT_BYTES:
                raise MalformedError("command argument is malformed")
            args.append(arg)
        safe_env: dict[str, str] = {}
        for key, value in self.environment.items():
            if not isinstance(key, str) or not re.fullmatch(r"[A-Z_][A-Z0-9_]{0,63}", key):
                raise MalformedError("environment key is malformed")
            if key not in _SAFE_ENVIRONMENT_KEYS or _SECRET.search(key) or not isinstance(value, str) or "\x00" in value or len(value) > 4096:
                raise BoundaryViolationError("command environment contains a forbidden value")
            safe_env[key] = value
        object.__setattr__(self, "executable", executable)
        object.__setattr__(self, "allowed_executables", allowed)
        object.__setattr__(self, "cwd", cwd)
        object.__setattr__(self, "allowed_cwds", cwds)
        object.__setattr__(self, "arguments", tuple(args))
        object.__setattr__(self, "environment", MappingProxyType(safe_env))


@dataclass(frozen=True)
class CommandExecution:
    stdout: bytes
    stderr: bytes
    returncode: int
    latency_ms: int

    @property
    def log_bytes(self) -> bytes:
        return redact_command_log(self.stderr)


def _kill_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=0.25)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _reader(stream: Any, destination: bytearray, budget: list[int], lock: threading.Lock, exceeded: threading.Event) -> None:
    while True:
        chunk = stream.read(8192)
        if not chunk:
            return
        with lock:
            available = budget[0]
            if available > 0:
                destination.extend(chunk[:available])
                budget[0] -= min(available, len(chunk))
        if len(chunk) > available:
            exceeded.set()


def invoke_command(policy: CommandPolicy, request: Mapping[str, Any], cancellation: CancellationToken | None = None) -> CommandExecution:
    """Execute one allowlisted argv process with no inherited environment."""
    if cancellation is not None:
        cancellation.check()
    payload = wire_canonical_utf8(dict(request)).encode("utf-8")
    if len(payload) > MAX_PROVIDER_OUTPUT_BYTES:
        raise BoundaryViolationError("command request exceeds the frozen byte bound")
    with tempfile.TemporaryDirectory(prefix="pcce-command-") as home:
        env = {"HOME": home, "XDG_CACHE_HOME": f"{home}/.cache", "XDG_CONFIG_HOME": f"{home}/.config", "XDG_DATA_HOME": f"{home}/.local/share", "XDG_STATE_HOME": f"{home}/.local/state", "PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}
        env.update(policy.environment)
        try:
            started = time.monotonic()
            process = subprocess.Popen([policy.executable, *policy.arguments], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=policy.cwd, env=env, shell=False, start_new_session=True)
        except FileNotFoundError as exc:
            raise UnavailableCapabilityError("allowlisted executable is unavailable") from exc
        except OSError as exc:
            raise UnavailableCapabilityError("allowlisted executable cannot be started") from exc
        stdout, stderr = bytearray(), bytearray()
        budget, lock = [MAX_PROVIDER_OUTPUT_BYTES], threading.Lock()
        exceeded = threading.Event()
        threads = [threading.Thread(target=_reader, args=(process.stdout, stdout, budget, lock, exceeded), daemon=True), threading.Thread(target=_reader, args=(process.stderr, stderr, budget, lock, exceeded), daemon=True)]
        for thread in threads: thread.start()
        assert process.stdin is not None
        try:
            process.stdin.write(payload); process.stdin.close()
            deadline = started + policy.timeout_seconds
            reason: str | None = None
            while process.poll() is None:
                if cancellation is not None and cancellation.cancelled: reason = "cancelled"
                elif exceeded.is_set(): reason = "output"
                elif time.monotonic() >= deadline: reason = "timeout"
                if reason:
                    _kill_group(process); break
                time.sleep(0.01)
            process.wait(timeout=1)
        finally:
            _kill_group(process)
            for thread in threads: thread.join(timeout=1)
        latency = max(0, int((time.monotonic() - started) * 1000))
    if cancellation is not None and cancellation.cancelled:
        raise ProofCancelledError("command invocation cancelled")
    if exceeded.is_set():
        raise BoundaryViolationError("command output exceeds the frozen byte bound")
    # A successful process may finish at the deadline; only a still-running process was killed.
    if 'reason' in locals() and reason == "timeout":
        raise ProofTimeoutError("command invocation timed out")
    return CommandExecution(bytes(stdout), bytes(stderr), int(process.returncode or 0), latency)


def decode_structured_output(stdout: bytes | str) -> Mapping[str, Any]:
    """Decode exactly one closed JSON proposal object; no fences or prose."""
    raw = stdout.encode("utf-8") if isinstance(stdout, str) else bytes(stdout)
    if not raw or len(raw) > MAX_PROVIDER_OUTPUT_BYTES:
        raise MalformedError("command stdout is empty or exceeds its bound")
    try:
        text = raw.decode("utf-8")
        decoder = json.JSONDecoder()
        value, index = decoder.raw_decode(text.lstrip())
        if text.lstrip()[index:].strip(): raise ValueError("trailing data")
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise MalformedError("command stdout is not exactly one JSON object") from exc
    if not isinstance(value, dict) or set(value) != _OUTPUT_FIELDS:
        raise MalformedError("command proposal has an invalid closed field set")
    return MappingProxyType(value)


def _admit_patch_scope(patch: bytes, declared: tuple[str, ...], task: TaskSpecification) -> None:
    """Require every changed path to be both declared and owned."""
    text = patch.decode("utf-8", "strict")
    matches = list(_PATCH_PATH.finditer(text))
    if not matches:
        raise MalformedError("command patch must be a unified diff")
    for match in matches:
        for raw in match.groups():
            if raw == "/dev/null":
                continue
            path = admit_relative_path(raw, field="patch_path")
            assert_declared_scope((path,), task.owned_paths, task.declared_files)
            if path not in declared:
                raise BoundaryViolationError("command patch path is not declared", details={"field": "declared_files"})


def build_command_request(task: TaskSpecification, pack: ContextPack, route: ModelRouteDecision) -> Mapping[str, Any]:
    bind_adapter_request(task, pack, route)
    return MappingProxyType({"schema": REQUEST_SCHEMA, "task_id": task.task_id, "objective_id": task.objective_id, "repository_state_cid": task.repository_state_cid, "pack_cid": pack.pack_cid, "route_cid": route.decision_cid, "owned_paths": list(task.owned_paths), "declared_files": list(task.declared_files or task.owned_paths), "provider": route.provider, "model": route.model, "revision": route.revision or "unspecified", "tier": route.tier, "instruction": "Return exactly one JSON proposal object and no prose; never approve or apply a patch."})


@dataclass
class CommandAdapter:
    policy: CommandPolicy

    def __post_init__(self) -> None:
        self.accepted = False; self.approved = False
        self.approval_authority = APPROVAL_AUTHORITY; self.canonical_branch_authority = CANONICAL_BRANCH_AUTHORITY

    def cancel(self, cancellation: CancellationToken) -> None:
        cancellation.cancel()

    def propose(self, task: TaskSpecification, context_pack: ContextPack, route: ModelRouteDecision, cancellation: CancellationToken | None = None) -> AdapterResult:
        request = build_command_request(task, context_pack, route)
        execution = invoke_command(self.policy, request, cancellation)
        if execution.returncode != 0:
            raise UnavailableCapabilityError("local command returned a non-zero exit status")
        output = decode_structured_output(execution.stdout)
        for name, expected in (("task_id", task.task_id), ("repository_state_cid", task.repository_state_cid), ("pack_cid", context_pack.pack_cid), ("route_cid", route.decision_cid), ("model", route.model), ("revision", route.revision or "unspecified")):
            if output[name] != expected: raise IdentityInconsistentError(f"command response {name} drifted", details={"field": name})
        declared = admit_path_list(output["declared_files"], field="declared_files", min_items=1, max_items=1024)
        assert_declared_scope(declared, task.owned_paths, task.declared_files)
        patch = admit_bounded_patch(output["patch"])
        _admit_patch_scope(patch, declared, task)
        # Validate every reported counter even though elapsed wall time is authoritative.
        admit_non_negative_int(output["latency_ms"], field="latency_ms")
        invocation_body = {"schema": CODING_AGENT_INVOCATION_SCHEMA, "task_id": task.task_id, "repository_state_cid": task.repository_state_cid, "route_cid": route.decision_cid, "provider": route.provider, "model": route.model, "revision": route.revision or "unspecified", "tier": route.tier, "token_count": admit_non_negative_int(output["token_count"], field="token_count"), "cached_token_count": admit_non_negative_int(output["cached_token_count"], field="cached_token_count"), "latency_ms": execution.latency_ms, "cost_micros": admit_non_negative_int(output["cost_micros"], field="cost_micros"), "response_artifact_cid": _cid({"request": dict(request), "stdout": execution.stdout.decode("utf-8")}), "provenance": "live"}
        invocation_body["invocation_cid"] = _cid(invocation_body)
        proposal_body = {"schema": PATCH_PROPOSAL_SCHEMA, "task_id": task.task_id, "repository_state_cid": task.repository_state_cid, "declared_files": list(declared), "invocation_cid": invocation_body["invocation_cid"], "patch_cid": _cid(patch.decode("utf-8", "replace")), "provenance": "live"}
        proposal_body["proposal_cid"] = _cid(proposal_body)
        result = AdapterResult(PatchProposal.from_mapping(proposal_body), CodingAgentInvocation.from_mapping(invocation_body), patch_bytes=patch, log_bytes=execution.log_bytes)
        return admit_adapter_result(task, context_pack, route, result, cancellation=cancellation)


__all__ = ["ADAPTER", "COMMAND_CONTRACT", "CommandAdapter", "CommandExecution", "CommandPolicy", "build_command_request", "decode_structured_output", "invoke_command", "redact_command_log"]
