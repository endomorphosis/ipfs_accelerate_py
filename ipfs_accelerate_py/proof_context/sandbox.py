"""Fail-closed, descriptor-rooted proof-context sandbox boundary (PCCE-071).

This module is an independently testable boundary, not a claim that every v0.1
execution path is sandboxed.  In particular, the adapter registry, InstalledCodex,
bootstrap, lifecycle, verification, and publication paths do not import it at the
PCCE-071 revision.  Consequently every receipt denies approval, canonical-branch,
and publication authority and the descriptor remains ``observed_tested_limited`` /
``not_integrated`` / production-ineligible.

The only executable backend composes a trusted isolated Python gate with the
existing :func:`invoke_command` PID/user/mount/network namespace supervisor.  The
inner gate adds descriptor revalidation, read/execute Landlock rules, exact-FD
``execve``, and hard resource limits.  There is no unsandboxed fallback.  The
route-endpoint allowlist shape is frozen for later integration, but execution in
that mode is deliberately unavailable because the current backend can enforce
deny-all networking only.
"""

from __future__ import annotations

import ast
import base64
import ctypes
import errno
import hashlib
import json
import os
import re
import resource
import secrets
import stat
import struct
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Self

from ipfs_accelerate_py.proof_context.adapters.base import CancellationToken
from ipfs_accelerate_py.proof_context.adapters.command import (
    CommandExecution,
    CommandPolicy,
    invoke_command,
)
from ipfs_accelerate_py.proof_context.adapters.models import (
    MAX_LOG_BYTES,
    MAX_PROVIDER_OUTPUT_BYTES,
    MAX_SAFE_INTEGER,
    admit_cid,
    wire_canonical_utf8,
)
from ipfs_accelerate_py.proof_context.errors import (
    BoundaryViolationError,
    MalformedError,
    PartialEffectError,
    ProofCancelledError,
    ProofContextError,
    ProofTimeoutError,
    RepairRequiredError,
    SchemaMismatchError,
    UnavailableCapabilityError,
    UnknownFieldError,
)

_REVIEWED_COMMAND_POLICY: Final[type[CommandPolicy]] = CommandPolicy
_REVIEWED_INVOKE_COMMAND: Final[Any] = invoke_command

INTERFACE: Final[str] = "SandboxBoundary@0.1"
SANDBOX_DESCRIPTOR_SCHEMA: Final[str] = "pcce/proof-context/v0.1/sandbox-descriptor@1"
SANDBOX_POLICY_SCHEMA: Final[str] = "pcce/proof-context/v0.1/sandbox-policy@1"
SANDBOX_EXECUTION_PERMIT_SCHEMA: Final[str] = "pcce/proof-context/v0.1/sandbox-execution-permit@1"
SANDBOX_CAPABILITY_REPORT_SCHEMA: Final[str] = "pcce/proof-context/v0.1/sandbox-capability-report@1"
SANDBOX_DENIAL_TRACE_SCHEMA: Final[str] = "pcce/proof-context/v0.1/sandbox-denial-trace@1"
SANDBOX_EXECUTION_RECEIPT_SCHEMA: Final[str] = "pcce/proof-context/v0.1/sandbox-execution-receipt@1"
SANDBOX_EXECUTION_RESULT_SCHEMA: Final[str] = "pcce/proof-context/v0.1/sandbox-execution-result@1"

RUNTIME_INTEGRATION_STATUS: Final[str] = "not_integrated"
ENFORCEMENT_DISPOSITION: Final[str] = "observed_tested_limited"
PRODUCTION_ELIGIBLE: Final[bool] = False
APPROVAL_AUTHORITY: Final[bool] = False
CANONICAL_BRANCH_AUTHORITY: Final[bool] = False
PUBLICATION_AUTHORITY: Final[bool] = False

NETWORK_MODES: Final[tuple[str, ...]] = ("deny_all", "route_endpoint_allowlist")
EXECUTION_STATUSES: Final[tuple[str, ...]] = (
    "completed_unpublished",
    "failed",
    "denied",
    "unavailable",
    "timeout",
    "cancelled",
    "partial_effect",
    "repair_required",
)
DENIAL_REASONS: Final[tuple[str, ...]] = (
    "non_disposable_worktree",
    "protected_ref",
    "root_overlap",
    "path_escape",
    "path_symlink",
    "path_identity_drift",
    "path_magiclink",
    "path_cross_device",
    "path_hardlink",
    "git_metadata_untrusted",
    "worktree_dirty",
    "base_mismatch",
    "executable_not_allowlisted",
    "executable_identity_drift",
    "argv_mismatch",
    "credential_forbidden",
    "route_mismatch",
    "endpoint_enforcement_unavailable",
    "permit_expired",
    "permit_replayed",
    "capability_unavailable",
    "timeout",
    "cancelled",
    "output_limit",
    "resource_limit",
    "secret_detected",
    "cleanup_unproven",
    "canonical_drift",
    "publication_forbidden",
)
THREAT_IDS: Final[tuple[str, ...]] = (
    "TH-001",
    "TH-002",
    "TH-003",
    "TH-004",
    "TH-005",
    "TH-006",
    "TH-007",
    "TH-011",
    "TH-013",
    "TH-014",
)
TRUST_BOUNDARIES: Final[tuple[str, ...]] = (
    "TB-02",
    "TB-03",
    "TB-04",
    "TB-05",
    "TB-10",
    "TB-11",
)
MAX_RECORD_BYTES: Final[int] = 1_048_576
MAX_PATH_BYTES: Final[int] = 4096
MAX_ARGUMENTS: Final[int] = 128
MAX_ARGUMENT_BYTES: Final[int] = 16_384
MAX_POLICY_PATHS: Final[int] = 1024
MAX_PERMIT_TTL_SECONDS: Final[int] = 3600
MAX_EXECUTABLE_BYTES: Final[int] = 268_435_456
MIN_MEMORY_BYTES: Final[int] = 134_217_728
MAX_MEMORY_BYTES: Final[int] = 17_179_869_184
MAX_OUTPUT_FILE_BYTES: Final[int] = 2_500_000
MAX_OPEN_FILES: Final[int] = 256
MAX_PROCESSES: Final[int] = 128
MAX_CPU_SECONDS: Final[int] = 3600
MAX_PROCESS_LOCAL_NONCES: Final[int] = 65_536
MAX_JSON_SECRET_FIELDS: Final[int] = 4096
MAX_PYTHON_SECRET_FIELDS: Final[int] = 4096
MAX_PARENT_SECRET_VALUES: Final[int] = 64
MAX_PARENT_SECRET_BYTES: Final[int] = 65_536
_GIT_TIMEOUT_SECONDS: Final[int] = 30
_HEX_SHA256: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")
_HEX_COMMIT: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{40,64}$")
_NONCE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_SECRET_KEY: Final[re.Pattern[str]] = re.compile(
    r"(?i)(api.?key|access.?token|refresh.?token|token|secret|password|passwd|"
    r"authorization|bearer|credential|cookie|private.?key|session)"
)
_SECRET_TEXT: Final[re.Pattern[str]] = re.compile(
    r"(?i)(api[_-]?key|access[_-]?token|refresh[_-]?token|token|secret|password|"
    r"authorization|bearer|credential)s?\s*[:=]\s*[^\s,;]{4,}"
)
_JSON_FIELD: Final[re.Pattern[str]] = re.compile(r'\\?"(?P<key>(?:\\.|[^"\\])*)\\?"\s*:')
_PYTHON_FIELD: Final[re.Pattern[str]] = re.compile(r"\\?'(?P<key>(?:\\.|[^'\\])*)\\?'\s*:")
_BEARER_TEXT: Final[re.Pattern[str]] = re.compile(
    r"(?i)\b(?:bearer\s+|sk-|ghp_|github_pat_|xox[baprs]-)[A-Za-z0-9_./+=-]{8,}"
)
_HOST_PATH_TEXT: Final[re.Pattern[str]] = re.compile(
    r"(?<![A-Za-z0-9])/(?:home|root|tmp|var|run|proc|sys|etc|opt|srv)/[^\s,;\]\[{}\"']+"
)
_FORBIDDEN_ENV_EXACT: Final[frozenset[str]] = frozenset(
    {
        "BASH_ENV",
        "ENV",
        "GIT_CONFIG",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_SYSTEM",
        "GIT_SSH",
        "GIT_SSH_COMMAND",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "SSH_AUTH_SOCK",
    }
)
_FORBIDDEN_ENV_PREFIXES: Final[tuple[str, ...]] = ("LD_", "DYLD_")
_INNER_GATE_SCHEMA: Final[str] = "pcce-sandbox-inner-gate@1"
_OPENAT2_SYSCALL: Final[int] = 437
_OPENAT2_RESOLVE: Final[int] = 0x01 | 0x02 | 0x04 | 0x08
_LANDLOCK_CREATE_RULESET: Final[int] = 444
_LANDLOCK_CREATE_RULESET_VERSION: Final[int] = 1
_LANDLOCK_MINIMUM_ABI: Final[int] = 6
_USED_PERMIT_NONCES: set[str] = set()
_PERMIT_NONCE_LOCK: Final[threading.Lock] = threading.Lock()


def _raw_cid(value: bytes) -> str:
    digest = hashlib.sha256(value).digest()
    multihash = b"\x01\x55\x12\x20" + digest
    return "b" + base64.b32encode(multihash).decode("ascii").lower().rstrip("=")


def _canonical_bytes(value: Any) -> bytes:
    return wire_canonical_utf8(value).encode("utf-8")


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _json_secret_material(text: str) -> tuple[bool, bool]:
    """Locate JSON secret-field values, including JSON embedded in argv/prose."""

    decoder = json.JSONDecoder()
    detected = False
    secret_fields = 0
    for match in _JSON_FIELD.finditer(text):
        try:
            key = json.loads(f'"{match.group("key")}"')
        except ValueError:
            continue
        if not isinstance(key, str) or not _SECRET_KEY.search(key):
            continue
        secret_fields += 1
        if secret_fields > MAX_JSON_SECRET_FIELDS:
            return True, True
        value_start = match.end()
        while value_start < len(text) and text[value_start].isspace():
            value_start += 1
        try:
            parsed, _value_end = decoder.raw_decode(text, value_start)
        except (ValueError, RecursionError):
            return True, True
        if parsed == "[redacted]":
            detected = True
            continue
        # Conservatively hide the entire preview.  Re-serializing individual
        # values cannot enumerate every equivalent JSON escape spelling, and
        # repeated per-field rewrites make postflight cost attacker-controlled.
        return True, True
    return detected, False


def _python_secret_material(text: str) -> tuple[bool, bool]:
    """Locate bounded single-quoted mapping keys without evaluating containers."""

    field_count = 0
    for match in _PYTHON_FIELD.finditer(text):
        field_count += 1
        if field_count > MAX_PYTHON_SECRET_FIELDS:
            # The scanner cannot prove an oversized mapping credential-free.
            return True, True
        encoded_key = match.group("key")
        if len(encoded_key.encode("utf-8")) > MAX_ARGUMENT_BYTES:
            return True, True
        if _SECRET_KEY.search(encoded_key):
            return True, True
        if "\\" not in encoded_key:
            continue
        try:
            key = ast.literal_eval(f"'{encoded_key}'")
        except (MemoryError, RecursionError):
            return True, True
        except (SyntaxError, ValueError):
            continue
        if isinstance(key, str) and _SECRET_KEY.search(key):
            # As with JSON, wiping the whole preview avoids escape-equivalent
            # copies of a decoded secret value and attacker-controlled rewrites.
            return True, True
    return False, False


def _structured_secret_material(text: str) -> tuple[bool, bool]:
    json_detected, json_wipe_all = _json_secret_material(text)
    if json_wipe_all:
        return True, True
    python_detected, python_wipe_all = _python_secret_material(text)
    return json_detected or python_detected, json_wipe_all or python_wipe_all


def _contains_secret_material(text: str) -> bool:
    structured_detected, _wipe_all = _structured_secret_material(text)
    return bool(structured_detected or _SECRET_TEXT.search(text) or _BEARER_TEXT.search(text))


def _bounded_identifier(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise MalformedError(f"{field_name} is not a bounded identifier")
    return value


def _bounded_int(
    value: Any,
    *,
    field_name: str,
    minimum: int,
    maximum: int,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise MalformedError(f"{field_name} is outside its frozen integer bound")
    return value


def _optional_cid(value: Any, *, field_name: str) -> str | None:
    if value is None:
        return None
    return admit_cid(value, field=field_name)


def _sha256(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not _HEX_SHA256.fullmatch(value):
        raise MalformedError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _commit(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not _HEX_COMMIT.fullmatch(value):
        raise MalformedError(f"{field_name} must be a full lowercase Git object id")
    return value


def _absolute_path(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise MalformedError(f"{field_name} must be a non-empty path")
    if len(os.fsencode(value)) > MAX_PATH_BYTES or not os.path.isabs(value):
        raise BoundaryViolationError(f"{field_name} must be a bounded absolute path")
    normalized = os.path.normpath(value)
    if normalized != value or value.startswith("//"):
        raise BoundaryViolationError(f"{field_name} must be a normalized absolute path")
    return value


def _relative_path(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise MalformedError(f"{field_name} must be a non-empty relative path")
    if len(value.encode("utf-8")) > MAX_PATH_BYTES:
        raise MalformedError(f"{field_name} exceeds its frozen byte bound")
    if value.startswith(("/", "\\", "~")) or ":" in value[:2]:
        raise _violation("path_escape", f"{field_name} is not repository-relative")
    parts = value.replace("\\", "/").split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise _violation("path_escape", f"{field_name} escapes its descriptor root")
    if any(part == ".git" for part in parts):
        raise _violation("protected_ref", f"{field_name} enters protected Git metadata")
    return "/".join(parts)


def _path_tuple(value: Any, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise MalformedError(f"{field_name} must be an immutable path sequence")
    if len(value) > MAX_POLICY_PATHS:
        raise MalformedError(f"{field_name} exceeds its frozen item bound")
    admitted = tuple(
        _relative_path(item, field_name=f"{field_name}[{index}]")
        for index, item in enumerate(value)
    )
    if len(set(admitted)) != len(admitted):
        raise MalformedError(f"{field_name} contains duplicate paths")
    return admitted


def _argv(value: Any, *, field_name: str = "argv") -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise MalformedError(f"{field_name} must be an argv sequence")
    if not 1 <= len(value) <= MAX_ARGUMENTS:
        raise MalformedError(f"{field_name} count is outside its frozen bound")
    admitted: list[str] = []
    for index, item in enumerate(value):
        if (
            not isinstance(item, str)
            or "\x00" in item
            or len(item.encode("utf-8")) > MAX_ARGUMENT_BYTES
        ):
            raise MalformedError(f"{field_name}[{index}] is malformed")
        if _contains_secret_material(item):
            raise _violation(
                "credential_forbidden", f"{field_name}[{index}] contains credential material"
            )
        admitted.append(item)
    return tuple(admitted)


def _closed_mapping(
    payload: Any,
    *,
    schema: str,
    fields: frozenset[str],
    record_name: str,
) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise MalformedError(f"{record_name} must be a mapping")
    extra = set(payload) - fields
    if extra:
        raise UnknownFieldError(f"unknown {record_name} field {sorted(extra)[0]!r}")
    missing = fields - set(payload)
    if missing:
        raise MalformedError(f"{record_name} is missing field {sorted(missing)[0]!r}")
    if payload.get("schema") != schema:
        raise SchemaMismatchError(f"{record_name} schema {payload.get('schema')!r} is not {schema}")
    return payload


def _decode_json_object(value: bytes | str) -> Mapping[str, Any]:
    raw = value.encode("utf-8") if isinstance(value, str) else bytes(value)
    if not raw or len(raw) > MAX_RECORD_BYTES:
        raise MalformedError("sandbox record bytes are empty or exceed the frozen bound")
    try:
        text = raw.decode("utf-8")

        def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, item in items:
                if key in result:
                    raise ValueError("duplicate object key")
                result[key] = item
            return result

        def reject_constant(item: str) -> None:
            raise ValueError(f"non-finite number {item}")

        decoder = json.JSONDecoder(object_pairs_hook=pairs, parse_constant=reject_constant)
        stripped = text.lstrip()
        decoded, index = decoder.raw_decode(stripped)
        if stripped[index:].strip():
            raise ValueError("trailing data")
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise MalformedError("sandbox record is not exactly one strict JSON object") from exc
    if not isinstance(decoded, dict):
        raise MalformedError("sandbox record must be a JSON object")
    return decoded


def _violation(
    reason: str,
    message: str,
    error_type: type[ProofContextError] = BoundaryViolationError,
) -> ProofContextError:
    if reason not in DENIAL_REASONS:
        raise ValueError(f"unknown sandbox denial reason {reason!r}")
    error = error_type(message, details={"reason": reason})
    object.__setattr__(error, "sandbox_reason", reason)
    return error


def _reason_for_exception(exc: BaseException) -> str:
    reason = getattr(exc, "sandbox_reason", None)
    if isinstance(reason, str) and reason in DENIAL_REASONS:
        return reason
    if isinstance(exc, ProofCancelledError):
        return "cancelled"
    if isinstance(exc, ProofTimeoutError):
        return "timeout"
    if isinstance(exc, (PartialEffectError, RepairRequiredError)):
        return "cleanup_unproven"
    if isinstance(exc, UnavailableCapabilityError):
        return "capability_unavailable"
    if isinstance(exc, BoundaryViolationError) and "output" in str(exc).lower():
        return "output_limit"
    return "capability_unavailable"


class _WireRecord:
    def to_mapping(self) -> Mapping[str, Any]:
        raise NotImplementedError

    @property
    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_mapping())

    @property
    def cid(self) -> str:
        return _raw_cid(self.canonical_bytes)

    def to_json(self) -> str:
        return self.canonical_bytes.decode("utf-8")


@dataclass(frozen=True, slots=True)
class SandboxPolicy(_WireRecord):
    """Closed immutable process, path, network, and resource policy."""

    repository_state_cid: str
    allowed_executable: str
    executable_sha256: str
    executable_identity_cid: str
    allowed_argv: tuple[str, ...]
    allowed_read_paths: tuple[str, ...] = ()
    allowed_write_paths: tuple[str, ...] = ()
    network_mode: str = "deny_all"
    route_cid: str | None = None
    endpoint_generation_cid: str | None = None
    timeout_seconds: int = 120
    cpu_seconds: int = 120
    memory_bytes: int = 1_073_741_824
    output_file_bytes: int = MAX_OUTPUT_FILE_BYTES
    open_files: int = 64
    processes: int = 32
    aggregate_output_bytes: int = MAX_PROVIDER_OUTPUT_BYTES
    redacted_log_bytes: int = MAX_LOG_BYTES
    schema: str = field(init=False, default=SANDBOX_POLICY_SCHEMA)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "repository_state_cid",
            admit_cid(self.repository_state_cid, field="repository_state_cid"),
        )
        executable = _absolute_path(self.allowed_executable, field_name="allowed_executable")
        argv = _argv(self.allowed_argv, field_name="allowed_argv")
        if argv[0] != executable:
            raise _violation(
                "argv_mismatch", "allowed_argv[0] must be the exact allowlisted executable"
            )
        object.__setattr__(self, "allowed_executable", executable)
        object.__setattr__(
            self,
            "executable_sha256",
            _sha256(self.executable_sha256, field_name="executable_sha256"),
        )
        object.__setattr__(
            self,
            "executable_identity_cid",
            admit_cid(self.executable_identity_cid, field="executable_identity_cid"),
        )
        object.__setattr__(self, "allowed_argv", argv)
        object.__setattr__(
            self,
            "allowed_read_paths",
            _path_tuple(self.allowed_read_paths, field_name="allowed_read_paths"),
        )
        object.__setattr__(
            self,
            "allowed_write_paths",
            _path_tuple(self.allowed_write_paths, field_name="allowed_write_paths"),
        )
        if self.allowed_write_paths:
            raise BoundaryViolationError(
                "the PCCE-071 backend grants no worktree-write authority; use bounded stdout"
            )
        if self.network_mode not in NETWORK_MODES:
            raise MalformedError("network_mode is not in the closed sandbox policy")
        route_cid = _optional_cid(self.route_cid, field_name="route_cid")
        endpoint_cid = _optional_cid(
            self.endpoint_generation_cid, field_name="endpoint_generation_cid"
        )
        if self.network_mode == "deny_all" and (route_cid is not None or endpoint_cid is not None):
            raise _violation("route_mismatch", "deny_all cannot carry an endpoint authority")
        if self.network_mode == "route_endpoint_allowlist" and (
            route_cid is None or endpoint_cid is None
        ):
            raise _violation(
                "route_mismatch", "route endpoint mode requires route and generation CIDs"
            )
        object.__setattr__(self, "route_cid", route_cid)
        object.__setattr__(self, "endpoint_generation_cid", endpoint_cid)
        for name, lower, upper in (
            ("timeout_seconds", 1, MAX_PERMIT_TTL_SECONDS),
            ("cpu_seconds", 1, MAX_CPU_SECONDS),
            ("memory_bytes", MIN_MEMORY_BYTES, MAX_MEMORY_BYTES),
            ("output_file_bytes", 1, MAX_OUTPUT_FILE_BYTES),
            ("open_files", 16, MAX_OPEN_FILES),
            ("processes", 1, MAX_PROCESSES),
        ):
            object.__setattr__(
                self,
                name,
                _bounded_int(getattr(self, name), field_name=name, minimum=lower, maximum=upper),
            )
        if self.aggregate_output_bytes != MAX_PROVIDER_OUTPUT_BYTES:
            raise MalformedError("aggregate_output_bytes must equal the enforced backend bound")
        if self.redacted_log_bytes != MAX_LOG_BYTES:
            raise MalformedError("redacted_log_bytes must equal the enforced backend bound")

    @classmethod
    def capture(
        cls,
        *,
        repository_state_cid: str,
        executable: str,
        argv: Sequence[str],
        allowed_read_paths: Sequence[str] = (),
        allowed_write_paths: Sequence[str] = (),
        network_mode: str = "deny_all",
        route_cid: str | None = None,
        endpoint_generation_cid: str | None = None,
        timeout_seconds: int = 120,
        cpu_seconds: int = 120,
        memory_bytes: int = 1_073_741_824,
        output_file_bytes: int = MAX_OUTPUT_FILE_BYTES,
        open_files: int = 64,
        processes: int = 32,
    ) -> Self:
        absolute = _absolute_path(executable, field_name="executable")
        digest, identity = _capture_executable(absolute)
        return cls(
            repository_state_cid=repository_state_cid,
            allowed_executable=absolute,
            executable_sha256=digest,
            executable_identity_cid=identity,
            allowed_argv=tuple(argv),
            allowed_read_paths=tuple(allowed_read_paths),
            allowed_write_paths=tuple(allowed_write_paths),
            network_mode=network_mode,
            route_cid=route_cid,
            endpoint_generation_cid=endpoint_generation_cid,
            timeout_seconds=timeout_seconds,
            cpu_seconds=cpu_seconds,
            memory_bytes=memory_bytes,
            output_file_bytes=output_file_bytes,
            open_files=open_files,
            processes=processes,
        )

    def to_mapping(self) -> Mapping[str, Any]:
        return _deep_freeze(
            {
                "schema": self.schema,
                "repository_state_cid": self.repository_state_cid,
                "allowed_executable": self.allowed_executable,
                "executable_sha256": self.executable_sha256,
                "executable_identity_cid": self.executable_identity_cid,
                "allowed_argv": self.allowed_argv,
                "allowed_read_paths": self.allowed_read_paths,
                "allowed_write_paths": self.allowed_write_paths,
                "network_mode": self.network_mode,
                "route_cid": self.route_cid,
                "endpoint_generation_cid": self.endpoint_generation_cid,
                "timeout_seconds": self.timeout_seconds,
                "cpu_seconds": self.cpu_seconds,
                "memory_bytes": self.memory_bytes,
                "output_file_bytes": self.output_file_bytes,
                "open_files": self.open_files,
                "processes": self.processes,
                "aggregate_output_bytes": self.aggregate_output_bytes,
                "redacted_log_bytes": self.redacted_log_bytes,
            }
        )

    @classmethod
    def from_mapping(cls, payload: Any) -> Self:
        fields = frozenset(
            {
                "schema",
                "repository_state_cid",
                "allowed_executable",
                "executable_sha256",
                "executable_identity_cid",
                "allowed_argv",
                "allowed_read_paths",
                "allowed_write_paths",
                "network_mode",
                "route_cid",
                "endpoint_generation_cid",
                "timeout_seconds",
                "cpu_seconds",
                "memory_bytes",
                "output_file_bytes",
                "open_files",
                "processes",
                "aggregate_output_bytes",
                "redacted_log_bytes",
            }
        )
        raw = _closed_mapping(
            payload,
            schema=SANDBOX_POLICY_SCHEMA,
            fields=fields,
            record_name="sandbox policy",
        )
        return cls(**{key: raw[key] for key in fields if key != "schema"})

    @classmethod
    def from_json(cls, value: bytes | str) -> Self:
        return cls.from_mapping(_decode_json_object(value))


@dataclass(frozen=True, slots=True)
class SandboxExecutionPermit(_WireRecord):
    """Unauthenticated, process-local single-use execution binding."""

    task_id: str
    objective_id: str
    repository_state_cid: str
    policy_cid: str
    worktree_base_commit: str
    executable_identity_cid: str
    argv: tuple[str, ...]
    network_mode: str
    route_cid: str | None
    endpoint_generation_cid: str | None
    issued_at_epoch: int
    expires_at_epoch: int
    nonce: str
    single_use: bool = True
    approval_authority: bool = False
    canonical_branch_authority: bool = False
    publication_authority: bool = False
    schema: str = field(init=False, default=SANDBOX_EXECUTION_PERMIT_SCHEMA)

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _bounded_identifier(self.task_id, field_name="task_id"))
        object.__setattr__(
            self, "objective_id", _bounded_identifier(self.objective_id, field_name="objective_id")
        )
        for name in (
            "repository_state_cid",
            "policy_cid",
            "executable_identity_cid",
        ):
            object.__setattr__(self, name, admit_cid(getattr(self, name), field=name))
        object.__setattr__(
            self,
            "worktree_base_commit",
            _commit(self.worktree_base_commit, field_name="worktree_base_commit"),
        )
        object.__setattr__(self, "argv", _argv(self.argv))
        if self.network_mode not in NETWORK_MODES:
            raise MalformedError("permit network_mode is unknown")
        route = _optional_cid(self.route_cid, field_name="route_cid")
        endpoint = _optional_cid(self.endpoint_generation_cid, field_name="endpoint_generation_cid")
        if self.network_mode == "deny_all" and (route is not None or endpoint is not None):
            raise _violation("route_mismatch", "deny_all permit carries endpoint authority")
        if self.network_mode == "route_endpoint_allowlist" and (route is None or endpoint is None):
            raise _violation("route_mismatch", "route permit is missing its endpoint binding")
        object.__setattr__(self, "route_cid", route)
        object.__setattr__(self, "endpoint_generation_cid", endpoint)
        issued = _bounded_int(
            self.issued_at_epoch,
            field_name="issued_at_epoch",
            minimum=0,
            maximum=MAX_SAFE_INTEGER,
        )
        expires = _bounded_int(
            self.expires_at_epoch,
            field_name="expires_at_epoch",
            minimum=1,
            maximum=MAX_SAFE_INTEGER,
        )
        if not issued < expires <= issued + MAX_PERMIT_TTL_SECONDS:
            raise MalformedError("permit lifetime is outside its frozen bound")
        if not isinstance(self.nonce, str) or not _NONCE.fullmatch(self.nonce):
            raise MalformedError("permit nonce must be 32 random bytes in lowercase hex")
        if self.single_use is not True:
            raise BoundaryViolationError("sandbox permits are single-use only")
        if any(
            value is not False
            for value in (
                self.approval_authority,
                self.canonical_branch_authority,
                self.publication_authority,
            )
        ):
            raise _violation("publication_forbidden", "sandbox permit cannot grant authority")

    @classmethod
    def issue(
        cls,
        policy: SandboxPolicy,
        *,
        task_id: str,
        objective_id: str,
        worktree_base_commit: str,
        now_epoch: int | None = None,
        ttl_seconds: int = 300,
        nonce: str | None = None,
    ) -> Self:
        now = int(time.time()) if now_epoch is None else now_epoch
        ttl = _bounded_int(
            ttl_seconds,
            field_name="ttl_seconds",
            minimum=1,
            maximum=MAX_PERMIT_TTL_SECONDS,
        )
        return cls(
            task_id=task_id,
            objective_id=objective_id,
            repository_state_cid=policy.repository_state_cid,
            policy_cid=policy.cid,
            worktree_base_commit=worktree_base_commit,
            executable_identity_cid=policy.executable_identity_cid,
            argv=policy.allowed_argv,
            network_mode=policy.network_mode,
            route_cid=policy.route_cid,
            endpoint_generation_cid=policy.endpoint_generation_cid,
            issued_at_epoch=now,
            expires_at_epoch=now + ttl,
            nonce=nonce or secrets.token_hex(32),
        )

    def to_mapping(self) -> Mapping[str, Any]:
        return _deep_freeze(
            {
                "schema": self.schema,
                "task_id": self.task_id,
                "objective_id": self.objective_id,
                "repository_state_cid": self.repository_state_cid,
                "policy_cid": self.policy_cid,
                "worktree_base_commit": self.worktree_base_commit,
                "executable_identity_cid": self.executable_identity_cid,
                "argv": self.argv,
                "network_mode": self.network_mode,
                "route_cid": self.route_cid,
                "endpoint_generation_cid": self.endpoint_generation_cid,
                "issued_at_epoch": self.issued_at_epoch,
                "expires_at_epoch": self.expires_at_epoch,
                "nonce": self.nonce,
                "single_use": self.single_use,
                "approval_authority": False,
                "canonical_branch_authority": False,
                "publication_authority": False,
            }
        )

    @classmethod
    def from_mapping(cls, payload: Any) -> Self:
        fields = frozenset(
            {
                "schema",
                "task_id",
                "objective_id",
                "repository_state_cid",
                "policy_cid",
                "worktree_base_commit",
                "executable_identity_cid",
                "argv",
                "network_mode",
                "route_cid",
                "endpoint_generation_cid",
                "issued_at_epoch",
                "expires_at_epoch",
                "nonce",
                "single_use",
                "approval_authority",
                "canonical_branch_authority",
                "publication_authority",
            }
        )
        raw = _closed_mapping(
            payload,
            schema=SANDBOX_EXECUTION_PERMIT_SCHEMA,
            fields=fields,
            record_name="sandbox execution permit",
        )
        return cls(**{key: raw[key] for key in fields if key != "schema"})

    @classmethod
    def from_json(cls, value: bytes | str) -> Self:
        return cls.from_mapping(_decode_json_object(value))


@dataclass(frozen=True, slots=True)
class SandboxCapabilityReport(_WireRecord):
    captured_at_epoch: int
    linux: bool
    descriptor_root: bool
    openat2: bool
    landlock_abi: int
    namespace_launcher: bool
    pidfd_supervision: bool
    seccomp: bool
    hard_rlimits: bool
    deny_all_network: bool
    process_tree_cleanup: bool
    route_endpoint_allowlist_enforcement: bool
    direct_execution_supported: bool
    runtime_integration_status: str = RUNTIME_INTEGRATION_STATUS
    enforcement_disposition: str = ENFORCEMENT_DISPOSITION
    production_eligible: bool = False
    limitations: tuple[str, ...] = (
        "authoritative runtime paths do not invoke this module",
        "route-scoped endpoint enforcement is unavailable",
        "permit replay memory is process-local and issuer authentication is unavailable",
        "descriptor fallback has no production qualification credit",
        "same-UID host actors remain outside direct child confinement",
        "Git worktree materialization occurs on the host before sandbox admission",
    )
    schema: str = field(init=False, default=SANDBOX_CAPABILITY_REPORT_SCHEMA)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "captured_at_epoch",
            _bounded_int(
                self.captured_at_epoch,
                field_name="captured_at_epoch",
                minimum=0,
                maximum=MAX_SAFE_INTEGER,
            ),
        )
        for name in (
            "linux",
            "descriptor_root",
            "openat2",
            "namespace_launcher",
            "pidfd_supervision",
            "seccomp",
            "hard_rlimits",
            "deny_all_network",
            "process_tree_cleanup",
            "route_endpoint_allowlist_enforcement",
            "direct_execution_supported",
        ):
            if type(getattr(self, name)) is not bool:
                raise MalformedError(f"{name} must be a boolean capability observation")
        object.__setattr__(
            self,
            "landlock_abi",
            _bounded_int(
                self.landlock_abi,
                field_name="landlock_abi",
                minimum=0,
                maximum=MAX_SAFE_INTEGER,
            ),
        )
        if self.route_endpoint_allowlist_enforcement:
            raise BoundaryViolationError("route endpoint enforcement cannot be claimed at PCCE-071")
        if self.runtime_integration_status != RUNTIME_INTEGRATION_STATUS:
            raise BoundaryViolationError("sandbox runtime integration is not established")
        if self.enforcement_disposition != ENFORCEMENT_DISPOSITION:
            raise BoundaryViolationError("sandbox disposition cannot be upgraded")
        if self.production_eligible:
            raise BoundaryViolationError("sandbox is not production eligible")
        if (
            not isinstance(self.limitations, tuple)
            or not self.limitations
            or len(self.limitations) > 16
            or len(set(self.limitations)) != len(self.limitations)
            or any(
                not isinstance(item, str) or not item or "\x00" in item or len(item) > 240
                for item in self.limitations
            )
        ):
            raise MalformedError("capability limitations must be a bounded immutable tuple")

    @classmethod
    def probe(cls, *, captured_at_epoch: int | None = None) -> Self:
        linux = sys.platform.startswith("linux") and hasattr(os, "O_PATH")
        descriptor_root = linux and all(
            hasattr(os, name) for name in ("O_NOFOLLOW", "O_DIRECTORY", "O_CLOEXEC")
        )
        openat2 = _probe_openat2() if linux else False
        landlock_abi = _probe_landlock_abi() if linux else 0
        launchers = all(
            _trusted_executable_available(path, require_setuid=setuid)
            for path, setuid in (
                ("/usr/bin/busybox", False),
                ("/usr/bin/unshare", False),
                ("/usr/bin/newuidmap", True),
                ("/usr/bin/newgidmap", True),
                (str(Path(sys.executable).resolve()), False),
            )
        )
        pidfd = (
            all(
                hasattr(owner, name)
                for owner, name in (
                    (os, "pidfd_open"),
                    (os, "waitid"),
                    (__import__("signal"), "pidfd_send_signal"),
                )
            )
            and Path("/proc/self/stat").is_file()
        )
        try:
            ctypes.CDLL("libseccomp.so.2")
        except OSError:
            seccomp = False
        else:
            seccomp = True
        hard_rlimits = all(
            hasattr(resource, name)
            for name in (
                "RLIMIT_CPU",
                "RLIMIT_AS",
                "RLIMIT_FSIZE",
                "RLIMIT_NOFILE",
                "RLIMIT_NPROC",
                "RLIMIT_CORE",
                "RLIMIT_MEMLOCK",
            )
        )
        direct = all(
            (
                linux,
                descriptor_root,
                landlock_abi >= _LANDLOCK_MINIMUM_ABI,
                launchers,
                pidfd,
                seccomp,
                hard_rlimits,
            )
        )
        return cls(
            captured_at_epoch=int(time.time()) if captured_at_epoch is None else captured_at_epoch,
            linux=linux,
            descriptor_root=descriptor_root,
            openat2=openat2,
            landlock_abi=landlock_abi,
            namespace_launcher=launchers,
            pidfd_supervision=pidfd,
            seccomp=seccomp,
            hard_rlimits=hard_rlimits,
            deny_all_network=direct,
            process_tree_cleanup=direct,
            route_endpoint_allowlist_enforcement=False,
            direct_execution_supported=direct,
        )

    def to_mapping(self) -> Mapping[str, Any]:
        return _deep_freeze(
            {
                "schema": self.schema,
                "captured_at_epoch": self.captured_at_epoch,
                "linux": self.linux,
                "descriptor_root": self.descriptor_root,
                "openat2": self.openat2,
                "landlock_abi": self.landlock_abi,
                "namespace_launcher": self.namespace_launcher,
                "pidfd_supervision": self.pidfd_supervision,
                "seccomp": self.seccomp,
                "hard_rlimits": self.hard_rlimits,
                "deny_all_network": self.deny_all_network,
                "process_tree_cleanup": self.process_tree_cleanup,
                "route_endpoint_allowlist_enforcement": False,
                "direct_execution_supported": self.direct_execution_supported,
                "runtime_integration_status": RUNTIME_INTEGRATION_STATUS,
                "enforcement_disposition": ENFORCEMENT_DISPOSITION,
                "production_eligible": False,
                "limitations": self.limitations,
            }
        )

    @classmethod
    def from_mapping(cls, payload: Any) -> Self:
        fields = frozenset(
            {
                "schema",
                "captured_at_epoch",
                "linux",
                "descriptor_root",
                "openat2",
                "landlock_abi",
                "namespace_launcher",
                "pidfd_supervision",
                "seccomp",
                "hard_rlimits",
                "deny_all_network",
                "process_tree_cleanup",
                "route_endpoint_allowlist_enforcement",
                "direct_execution_supported",
                "runtime_integration_status",
                "enforcement_disposition",
                "production_eligible",
                "limitations",
            }
        )
        raw = _closed_mapping(
            payload,
            schema=SANDBOX_CAPABILITY_REPORT_SCHEMA,
            fields=fields,
            record_name="sandbox capability report",
        )
        if not isinstance(raw["limitations"], Sequence) or isinstance(
            raw["limitations"], (str, bytes, bytearray)
        ):
            raise MalformedError("capability limitations must be an array")
        values = {key: raw[key] for key in fields if key != "schema"}
        values["limitations"] = tuple(values["limitations"])
        return cls(**values)

    @classmethod
    def from_json(cls, value: bytes | str) -> Self:
        return cls.from_mapping(_decode_json_object(value))


@dataclass(frozen=True, slots=True)
class SandboxDenialTrace(_WireRecord):
    reason: str
    stage: str
    observed_at_epoch: int
    subject_cid: str | None = None
    detail: str = ""
    publication_allowed: bool = False
    schema: str = field(init=False, default=SANDBOX_DENIAL_TRACE_SCHEMA)

    def __post_init__(self) -> None:
        if self.reason not in DENIAL_REASONS:
            raise MalformedError("denial reason is not in the closed sandbox vocabulary")
        object.__setattr__(self, "stage", _bounded_identifier(self.stage, field_name="stage"))
        object.__setattr__(
            self,
            "observed_at_epoch",
            _bounded_int(
                self.observed_at_epoch,
                field_name="observed_at_epoch",
                minimum=0,
                maximum=MAX_SAFE_INTEGER,
            ),
        )
        object.__setattr__(
            self, "subject_cid", _optional_cid(self.subject_cid, field_name="subject_cid")
        )
        if not isinstance(self.detail, str) or len(self.detail.encode("utf-8")) > MAX_LOG_BYTES:
            raise MalformedError("denial detail must be bounded text")
        redacted, _detected = _redact_preview(self.detail.encode("utf-8"), (), limit=240)
        object.__setattr__(self, "detail", redacted)
        if self.publication_allowed:
            raise _violation("publication_forbidden", "denial traces cannot authorize publication")

    @classmethod
    def from_exception(
        cls,
        exc: BaseException,
        *,
        stage: str,
        observed_at_epoch: int | None = None,
        subject_cid: str | None = None,
    ) -> Self:
        return cls(
            reason=_reason_for_exception(exc),
            stage=stage,
            observed_at_epoch=int(time.time()) if observed_at_epoch is None else observed_at_epoch,
            subject_cid=subject_cid,
            detail=str(exc),
        )

    def to_mapping(self) -> Mapping[str, Any]:
        return _deep_freeze(
            {
                "schema": self.schema,
                "reason": self.reason,
                "stage": self.stage,
                "observed_at_epoch": self.observed_at_epoch,
                "subject_cid": self.subject_cid,
                "detail": self.detail,
                "publication_allowed": False,
            }
        )

    @classmethod
    def from_mapping(cls, payload: Any) -> Self:
        fields = frozenset(
            {
                "schema",
                "reason",
                "stage",
                "observed_at_epoch",
                "subject_cid",
                "detail",
                "publication_allowed",
            }
        )
        raw = _closed_mapping(
            payload,
            schema=SANDBOX_DENIAL_TRACE_SCHEMA,
            fields=fields,
            record_name="sandbox denial trace",
        )
        return cls(**{key: raw[key] for key in fields if key != "schema"})

    @classmethod
    def from_json(cls, value: bytes | str) -> Self:
        return cls.from_mapping(_decode_json_object(value))


@dataclass(frozen=True, slots=True)
class SandboxExecutionReceipt(_WireRecord):
    permit_cid: str
    policy_cid: str
    capability_report_cid: str
    status: str
    reason: str | None
    started_at_epoch: int
    finished_at_epoch: int
    latency_ms: int
    returncode: int | None
    stdout_cid: str
    stderr_cid: str
    stdout_bytes: int
    stderr_bytes: int
    denial_trace_cid: str | None
    worktree_cleanup_proven: bool
    canonical_unchanged: bool
    secret_scan_passed: bool
    runtime_integration_status: str = RUNTIME_INTEGRATION_STATUS
    enforcement_disposition: str = ENFORCEMENT_DISPOSITION
    approval_authority: bool = False
    canonical_branch_authority: bool = False
    publication_allowed: bool = False
    production_eligible: bool = False
    schema: str = field(init=False, default=SANDBOX_EXECUTION_RECEIPT_SCHEMA)

    def __post_init__(self) -> None:
        for name in (
            "permit_cid",
            "policy_cid",
            "capability_report_cid",
            "stdout_cid",
            "stderr_cid",
        ):
            object.__setattr__(self, name, admit_cid(getattr(self, name), field=name))
        if self.status not in EXECUTION_STATUSES:
            raise MalformedError("sandbox receipt status is unknown")
        if self.reason is not None and self.reason not in DENIAL_REASONS:
            raise MalformedError("sandbox receipt denial reason is unknown")
        for name in (
            "started_at_epoch",
            "finished_at_epoch",
            "latency_ms",
            "stdout_bytes",
            "stderr_bytes",
        ):
            object.__setattr__(
                self,
                name,
                _bounded_int(
                    getattr(self, name),
                    field_name=name,
                    minimum=0,
                    maximum=MAX_SAFE_INTEGER,
                ),
            )
        if self.finished_at_epoch < self.started_at_epoch:
            raise MalformedError("sandbox receipt time range is inverted")
        if self.returncode is not None:
            object.__setattr__(
                self,
                "returncode",
                _bounded_int(self.returncode, field_name="returncode", minimum=-255, maximum=255),
            )
        object.__setattr__(
            self,
            "denial_trace_cid",
            _optional_cid(self.denial_trace_cid, field_name="denial_trace_cid"),
        )
        for name in ("worktree_cleanup_proven", "canonical_unchanged", "secret_scan_passed"):
            if type(getattr(self, name)) is not bool:
                raise MalformedError(f"{name} must be boolean")
        if self.reason == "secret_detected" and self.secret_scan_passed:
            raise BoundaryViolationError(
                "secret-detected receipt cannot claim that its secret scan passed"
            )
        if self.reason == "secret_detected" and self.status != "denied":
            raise MalformedError("secret-detected receipt must carry denied status")
        if self.runtime_integration_status != RUNTIME_INTEGRATION_STATUS:
            raise BoundaryViolationError("receipt cannot claim runtime integration")
        if self.enforcement_disposition != ENFORCEMENT_DISPOSITION:
            raise BoundaryViolationError("receipt disposition cannot be upgraded")
        if any(
            value is not False
            for value in (
                self.approval_authority,
                self.canonical_branch_authority,
                self.publication_allowed,
                self.production_eligible,
            )
        ):
            raise _violation("publication_forbidden", "sandbox receipts cannot grant authority")
        if self.status == "completed_unpublished" and (
            not self.worktree_cleanup_proven
            or not self.canonical_unchanged
            or not self.secret_scan_passed
        ):
            raise BoundaryViolationError("completed receipt lacks its direct enforcement proofs")
        if self.status == "completed_unpublished" and (
            self.reason is not None or self.denial_trace_cid is not None
        ):
            raise MalformedError("completed receipt cannot carry a denial")
        if self.status in {
            "denied",
            "unavailable",
            "timeout",
            "cancelled",
            "partial_effect",
            "repair_required",
        } and (self.reason is None or self.denial_trace_cid is None):
            raise MalformedError("nonterminal-success receipt requires a denial trace")

    def to_mapping(self) -> Mapping[str, Any]:
        return _deep_freeze(
            {
                "schema": self.schema,
                "permit_cid": self.permit_cid,
                "policy_cid": self.policy_cid,
                "capability_report_cid": self.capability_report_cid,
                "status": self.status,
                "reason": self.reason,
                "started_at_epoch": self.started_at_epoch,
                "finished_at_epoch": self.finished_at_epoch,
                "latency_ms": self.latency_ms,
                "returncode": self.returncode,
                "stdout_cid": self.stdout_cid,
                "stderr_cid": self.stderr_cid,
                "stdout_bytes": self.stdout_bytes,
                "stderr_bytes": self.stderr_bytes,
                "denial_trace_cid": self.denial_trace_cid,
                "worktree_cleanup_proven": self.worktree_cleanup_proven,
                "canonical_unchanged": self.canonical_unchanged,
                "secret_scan_passed": self.secret_scan_passed,
                "runtime_integration_status": RUNTIME_INTEGRATION_STATUS,
                "enforcement_disposition": ENFORCEMENT_DISPOSITION,
                "approval_authority": False,
                "canonical_branch_authority": False,
                "publication_allowed": False,
                "production_eligible": False,
            }
        )

    @classmethod
    def from_mapping(cls, payload: Any) -> Self:
        fields = frozenset(
            {
                "schema",
                "permit_cid",
                "policy_cid",
                "capability_report_cid",
                "status",
                "reason",
                "started_at_epoch",
                "finished_at_epoch",
                "latency_ms",
                "returncode",
                "stdout_cid",
                "stderr_cid",
                "stdout_bytes",
                "stderr_bytes",
                "denial_trace_cid",
                "worktree_cleanup_proven",
                "canonical_unchanged",
                "secret_scan_passed",
                "runtime_integration_status",
                "enforcement_disposition",
                "approval_authority",
                "canonical_branch_authority",
                "publication_allowed",
                "production_eligible",
            }
        )
        raw = _closed_mapping(
            payload,
            schema=SANDBOX_EXECUTION_RECEIPT_SCHEMA,
            fields=fields,
            record_name="sandbox execution receipt",
        )
        return cls(**{key: raw[key] for key in fields if key != "schema"})

    @classmethod
    def from_json(cls, value: bytes | str) -> Self:
        return cls.from_mapping(_decode_json_object(value))


@dataclass(frozen=True, slots=True)
class SandboxExecutionResult(_WireRecord):
    receipt: SandboxExecutionReceipt
    stdout_preview: str
    stderr_preview: str
    denial_trace: SandboxDenialTrace | None = None
    schema: str = field(init=False, default=SANDBOX_EXECUTION_RESULT_SCHEMA)

    def __post_init__(self) -> None:
        if not isinstance(self.receipt, SandboxExecutionReceipt):
            raise MalformedError("sandbox result requires a typed receipt")
        for name in ("stdout_preview", "stderr_preview"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value.encode("utf-8")) > MAX_LOG_BYTES:
                raise MalformedError(f"{name} exceeds its redacted bound")
            preview, detected = _redact_preview(value.encode("utf-8"), (), limit=MAX_LOG_BYTES)
            if detected and preview != value:
                raise BoundaryViolationError(f"{name} contains unredacted credential material")
            if detected and self.receipt.secret_scan_passed:
                raise BoundaryViolationError(
                    f"{name} contains redacted credential fields but the receipt claims a pass"
                )
        if self.denial_trace is None and self.receipt.denial_trace_cid is not None:
            raise MalformedError("receipt names a denial trace absent from the result")
        if self.denial_trace is not None and self.denial_trace.cid != self.receipt.denial_trace_cid:
            raise MalformedError("receipt denial trace identity is inconsistent")
        if self.denial_trace is not None and self.denial_trace.reason != self.receipt.reason:
            raise MalformedError("receipt and denial trace reasons are inconsistent")

    def to_mapping(self) -> Mapping[str, Any]:
        return _deep_freeze(
            {
                "schema": self.schema,
                "receipt": self.receipt.to_mapping(),
                "stdout_preview": self.stdout_preview,
                "stderr_preview": self.stderr_preview,
                "denial_trace": None
                if self.denial_trace is None
                else self.denial_trace.to_mapping(),
            }
        )

    @classmethod
    def from_mapping(cls, payload: Any) -> Self:
        fields = frozenset(
            {"schema", "receipt", "stdout_preview", "stderr_preview", "denial_trace"}
        )
        raw = _closed_mapping(
            payload,
            schema=SANDBOX_EXECUTION_RESULT_SCHEMA,
            fields=fields,
            record_name="sandbox execution result",
        )
        denial = raw["denial_trace"]
        if denial is not None and not isinstance(denial, Mapping):
            raise MalformedError("sandbox execution result denial_trace must be an object or null")
        return cls(
            receipt=SandboxExecutionReceipt.from_mapping(raw["receipt"]),
            stdout_preview=raw["stdout_preview"],
            stderr_preview=raw["stderr_preview"],
            denial_trace=None if denial is None else SandboxDenialTrace.from_mapping(denial),
        )

    @classmethod
    def from_json(cls, value: bytes | str) -> Self:
        return cls.from_mapping(_decode_json_object(value))


def _open_absolute_directory(path: str) -> int:
    components = Path(path).parts
    descriptor = os.open("/", os.O_PATH | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        for component in components[1:]:
            next_descriptor = os.open(
                component,
                os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except OSError as exc:
        os.close(descriptor)
        if path.startswith("/proc/") and exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            reason = "path_magiclink"
        else:
            reason = "path_symlink" if exc.errno in {errno.ELOOP, errno.ENOTDIR} else "path_escape"
        raise _violation(
            reason, "descriptor root cannot be opened without symlink traversal"
        ) from exc


class _OpenHow(ctypes.Structure):
    _fields_ = (
        ("flags", ctypes.c_uint64),
        ("mode", ctypes.c_uint64),
        ("resolve", ctypes.c_uint64),
    )


def _openat2_beneath(root_fd: int, relative: str, flags: int) -> int | None:
    """Prefer openat2; return None only when the syscall shape is unavailable."""

    if not sys.platform.startswith("linux"):
        return None
    library = ctypes.CDLL(None, use_errno=True)
    library.syscall.restype = ctypes.c_long
    how = _OpenHow(flags, 0, _OPENAT2_RESOLVE)
    result = int(
        library.syscall(
            _OPENAT2_SYSCALL,
            ctypes.c_int(root_fd),
            ctypes.c_char_p(os.fsencode(relative)),
            ctypes.byref(how),
            ctypes.sizeof(how),
        )
    )
    if result >= 0:
        return result
    error = ctypes.get_errno()
    if error in {errno.ENOSYS, errno.EINVAL, errno.EPERM}:
        return None
    raise OSError(error, os.strerror(error), relative)


class DescriptorRoot:
    """An absolute directory identity used for component-wise openat operations."""

    __slots__ = ("path", "_fd", "_identity", "_closed")

    def __init__(self, path: str | os.PathLike[str]) -> None:
        admitted = _absolute_path(os.fspath(path), field_name="descriptor_root")
        try:
            descriptor = _open_absolute_directory(admitted)
            metadata = os.fstat(descriptor)
            visible = os.stat(admitted, follow_symlinks=False)
        except ProofContextError:
            raise
        except OSError as exc:
            raise _violation("path_escape", "descriptor root is unavailable") from exc
        identity = _directory_identity(metadata)
        if not stat.S_ISDIR(metadata.st_mode) or _directory_identity(visible) != identity:
            os.close(descriptor)
            raise _violation("path_identity_drift", "descriptor root identity is inconsistent")
        if os.path.realpath(f"/proc/self/fd/{descriptor}") != admitted:
            os.close(descriptor)
            raise _violation(
                "path_magiclink", "descriptor root resolves through an unexpected path"
            )
        self.path = admitted
        self._fd = descriptor
        self._identity = identity
        self._closed = False

    @property
    def identity_cid(self) -> str:
        self._ensure_stable()
        payload = {
            "device": self._identity[0],
            "inode": self._identity[1],
            "mode": self._identity[2],
            "uid": self._identity[3],
            "gid": self._identity[4],
        }
        return _raw_cid(_canonical_bytes(payload))

    @property
    def device(self) -> int:
        self._ensure_stable()
        return self._identity[0]

    def _ensure_stable(self) -> None:
        if self._closed:
            raise _violation("path_identity_drift", "descriptor root is closed")
        try:
            anchored = os.fstat(self._fd)
            visible = os.stat(self.path, follow_symlinks=False)
            resolved = os.path.realpath(f"/proc/self/fd/{self._fd}")
        except OSError as exc:
            raise _violation(
                "path_identity_drift", "descriptor root cannot be revalidated"
            ) from exc
        if (
            _directory_identity(anchored) != self._identity
            or _directory_identity(visible) != self._identity
            or resolved != self.path
        ):
            raise _violation("path_identity_drift", "descriptor root identity drifted")

    def _open_components(self, relative: str, *, final_flags: int) -> int:
        admitted = _relative_path(relative, field_name="path")
        self._ensure_stable()
        try:
            preferred = _openat2_beneath(
                self._fd,
                admitted,
                final_flags | os.O_NOFOLLOW | os.O_CLOEXEC,
            )
        except OSError as exc:
            if exc.errno == errno.EXDEV:
                reason = "path_cross_device"
            elif exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                reason = "path_symlink"
            else:
                reason = "path_escape"
            raise _violation(reason, "descriptor-relative path cannot be opened") from exc
        if preferred is not None:
            metadata = os.fstat(preferred)
            if metadata.st_dev != self._identity[0]:
                os.close(preferred)
                raise _violation("path_cross_device", "path crosses the descriptor filesystem")
            return preferred
        parts = admitted.split("/")
        descriptor = os.dup(self._fd)
        try:
            for component in parts[:-1]:
                next_descriptor = os.open(
                    component,
                    os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                    dir_fd=descriptor,
                )
                metadata = os.fstat(next_descriptor)
                if metadata.st_dev != self._identity[0]:
                    os.close(next_descriptor)
                    raise _violation("path_cross_device", "path crosses the descriptor filesystem")
                os.close(descriptor)
                descriptor = next_descriptor
            result = os.open(
                parts[-1], final_flags | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=descriptor
            )
            metadata = os.fstat(result)
            if metadata.st_dev != self._identity[0]:
                os.close(result)
                raise _violation("path_cross_device", "path crosses the descriptor filesystem")
            return result
        except ProofContextError:
            raise
        except OSError as exc:
            reason = "path_symlink" if exc.errno in {errno.ELOOP, errno.ENOTDIR} else "path_escape"
            raise _violation(reason, "descriptor-relative path cannot be opened") from exc
        finally:
            os.close(descriptor)

    def open_readonly(self, relative: str) -> int:
        descriptor = self._open_components(relative, final_flags=os.O_RDONLY)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            os.close(descriptor)
            raise _violation("path_escape", "descriptor-relative object is not a regular file")
        if metadata.st_nlink != 1:
            os.close(descriptor)
            raise _violation("path_hardlink", "hard-linked files are not admitted")
        return descriptor

    def read_bytes(self, relative: str, *, max_bytes: int = MAX_RECORD_BYTES) -> bytes:
        bound = _bounded_int(
            max_bytes, field_name="max_bytes", minimum=1, maximum=MAX_PROVIDER_OUTPUT_BYTES
        )
        descriptor = self.open_readonly(relative)
        try:
            data = bytearray()
            while len(data) <= bound:
                chunk = os.read(descriptor, min(65_536, bound + 1 - len(data)))
                if not chunk:
                    return bytes(data)
                data.extend(chunk)
            raise _violation("output_limit", "descriptor-rooted read exceeds its bound")
        finally:
            os.close(descriptor)

    def require_directory(self, relative: str) -> tuple[int, int, int, int, int]:
        descriptor = self._open_components(relative, final_flags=os.O_PATH | os.O_DIRECTORY)
        try:
            return _directory_identity(os.fstat(descriptor))
        finally:
            os.close(descriptor)

    def atomic_write(
        self,
        relative: str,
        data: bytes,
        *,
        mode: int = 0o600,
        max_bytes: int = MAX_RECORD_BYTES,
    ) -> str:
        admitted = _relative_path(relative, field_name="path")
        bound = _bounded_int(
            max_bytes,
            field_name="max_bytes",
            minimum=1,
            maximum=MAX_PROVIDER_OUTPUT_BYTES,
        )
        if not isinstance(data, bytes) or len(data) > bound:
            raise _violation("output_limit", "anchored write exceeds its byte bound")
        if mode not in {0o600, 0o640, 0o644}:
            raise BoundaryViolationError("anchored write mode is not in the closed allowlist")
        self._ensure_stable()
        parts = admitted.split("/")
        parent_fd = os.open(
            ".",
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=self._fd,
        )
        temporary = f".pcce-sandbox-{secrets.token_hex(16)}.tmp"
        temporary_created = False
        parent_relative = "/".join(parts[:-1])
        try:
            for component in parts[:-1]:
                next_fd = os.open(
                    component,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                    dir_fd=parent_fd,
                )
                if os.fstat(next_fd).st_dev != self._identity[0]:
                    os.close(next_fd)
                    raise _violation("path_cross_device", "write parent crosses filesystem")
                os.close(parent_fd)
                parent_fd = next_fd
            parent_identity = _directory_identity(os.fstat(parent_fd))

            def ensure_parent_visible() -> None:
                self._ensure_stable()
                if not parent_relative:
                    return
                visible_fd = self._open_components(
                    parent_relative,
                    final_flags=os.O_PATH | os.O_DIRECTORY,
                )
                try:
                    if _directory_identity(os.fstat(visible_fd)) != parent_identity:
                        raise _violation(
                            "path_identity_drift", "anchored write parent identity drifted"
                        )
                finally:
                    os.close(visible_fd)

            try:
                current = os.stat(parts[-1], dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                current = None
            if current is not None and (not stat.S_ISREG(current.st_mode) or current.st_nlink != 1):
                reason = "path_symlink" if stat.S_ISLNK(current.st_mode) else "path_hardlink"
                raise _violation(reason, "anchored write target is not a single-link regular file")
            output_fd = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                mode,
                dir_fd=parent_fd,
            )
            temporary_created = True
            try:
                metadata = os.fstat(output_fd)
                if metadata.st_dev != self._identity[0] or metadata.st_nlink != 1:
                    raise _violation("path_identity_drift", "temporary output identity is unsafe")
                view = memoryview(data)
                offset = 0
                while offset < len(view):
                    offset += os.write(output_fd, view[offset:])
                os.fsync(output_fd)
            finally:
                os.close(output_fd)
            ensure_parent_visible()
            os.rename(temporary, parts[-1], src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
            temporary_created = False
            os.fsync(parent_fd)
            ensure_parent_visible()
            final = os.stat(parts[-1], dir_fd=parent_fd, follow_symlinks=False)
            if not stat.S_ISREG(final.st_mode) or final.st_nlink != 1:
                raise _violation("path_identity_drift", "anchored output identity is inconsistent")
            visible_fd = self._open_components(admitted, final_flags=os.O_RDONLY)
            try:
                visible = os.fstat(visible_fd)
                if (
                    visible.st_dev,
                    visible.st_ino,
                    visible.st_mode,
                    visible.st_uid,
                    visible.st_gid,
                    visible.st_nlink,
                    visible.st_size,
                ) != (
                    final.st_dev,
                    final.st_ino,
                    final.st_mode,
                    final.st_uid,
                    final.st_gid,
                    final.st_nlink,
                    final.st_size,
                ):
                    raise _violation(
                        "path_identity_drift", "anchored output visibility is inconsistent"
                    )
            finally:
                os.close(visible_fd)
            return _raw_cid(data)
        except ProofContextError:
            raise
        except OSError as exc:
            raise _violation("path_identity_drift", "anchored atomic write failed") from exc
        finally:
            if temporary_created:
                try:
                    os.unlink(temporary, dir_fd=parent_fd)
                except OSError:
                    pass
            os.close(parent_fd)

    def close(self) -> None:
        if not self._closed:
            os.close(self._fd)
            self._closed = True

    def __enter__(self) -> Self:
        self._ensure_stable()
        return self

    def __exit__(self, _kind: Any, _value: Any, _traceback: Any) -> None:
        self.close()


@dataclass(frozen=True, slots=True)
class _GitSnapshot:
    head: str
    head_tree: str
    status_sha256: str
    head_file_sha256: str
    index_sha256: str
    config_sha256: str
    packed_refs_sha256: str
    protected_refs: tuple[tuple[str, str | None], ...]


class DisposableWorktreeGuard:
    """Fence a caller-created detached exact-base worktree and remove it on exit."""

    __slots__ = (
        "canonical_repository",
        "worktree",
        "expected_base_commit",
        "protected_refs",
        "remove_on_exit",
        "_canonical_root",
        "_worktree_root",
        "_snapshot",
        "_entered",
        "cleanup_proven",
        "canonical_unchanged",
    )

    def __init__(
        self,
        canonical_repository: str | os.PathLike[str],
        worktree: str | os.PathLike[str],
        *,
        expected_base_commit: str,
        protected_refs: Sequence[str] = ("refs/heads/main", "refs/heads/master"),
        remove_on_exit: bool = True,
    ) -> None:
        self.canonical_repository = _absolute_path(
            os.fspath(canonical_repository), field_name="canonical_repository"
        )
        self.worktree = _absolute_path(os.fspath(worktree), field_name="worktree")
        self.expected_base_commit = _commit(expected_base_commit, field_name="expected_base_commit")
        admitted_refs: list[str] = []
        for ref in protected_refs:
            if (
                not isinstance(ref, str)
                or not ref.startswith("refs/heads/")
                or ".." in ref
                or "\x00" in ref
                or len(ref) > 255
            ):
                raise _violation("protected_ref", "protected ref name is malformed")
            admitted_refs.append(ref)
        if len(set(admitted_refs)) != len(admitted_refs):
            raise MalformedError("protected refs contain duplicates")
        if remove_on_exit is not True:
            raise BoundaryViolationError("sandbox worktrees must be removed on exit")
        self.protected_refs = tuple(admitted_refs)
        self.remove_on_exit = True
        self._canonical_root: DescriptorRoot | None = None
        self._worktree_root: DescriptorRoot | None = None
        self._snapshot: _GitSnapshot | None = None
        self._entered = False
        self.cleanup_proven = False
        self.canonical_unchanged = False

    @classmethod
    def create(
        cls,
        canonical_repository: str | os.PathLike[str],
        disposable_parent: str | os.PathLike[str],
        *,
        expected_base_commit: str,
        protected_refs: Sequence[str] = ("refs/heads/main", "refs/heads/master"),
    ) -> Self:
        canonical = _absolute_path(
            os.fspath(canonical_repository), field_name="canonical_repository"
        )
        parent_path = _absolute_path(os.fspath(disposable_parent), field_name="disposable_parent")
        base = _commit(expected_base_commit, field_name="expected_base_commit")
        worktree = os.path.join(parent_path, f"pcce-sandbox-{secrets.token_hex(16)}")
        attempted_add = False
        try:
            with DescriptorRoot(parent_path) as parent:
                _reject_root_overlap(canonical, worktree)
                if os.path.lexists(worktree):
                    raise _violation("non_disposable_worktree", "disposable worktree target exists")
                attempted_add = True
                _run_git(
                    canonical,
                    ("worktree", "add", "--detach", worktree, base),
                    expected=(0,),
                )
                parent._ensure_stable()
            return cls(
                canonical,
                worktree,
                expected_base_commit=base,
                protected_refs=protected_refs,
            )
        except BaseException as original:
            if not attempted_add:
                raise
            parent_identity_uncertain = (
                getattr(original, "sandbox_reason", None) == "path_identity_drift"
            )
            try:
                _run_git(
                    canonical,
                    ("worktree", "remove", "--force", worktree),
                    expected=(0, 128),
                )
                inventory = _run_git(
                    canonical,
                    ("worktree", "list", "--porcelain", "-z"),
                    expected=(0,),
                ).stdout.split(b"\x00")
                registered = b"worktree " + os.fsencode(worktree) in inventory
                if os.path.lexists(worktree) or registered:
                    raise _violation(
                        "cleanup_unproven",
                        "failed-creation worktree cleanup is unproven",
                        RepairRequiredError,
                    )
            except RepairRequiredError:
                raise
            except BaseException as cleanup_exc:
                raise _violation(
                    "cleanup_unproven",
                    "failed-creation worktree cleanup is unproven",
                    RepairRequiredError,
                ) from cleanup_exc
            if parent_identity_uncertain:
                raise _violation(
                    "cleanup_unproven",
                    "failed-creation worktree parent identity drifted",
                    RepairRequiredError,
                ) from original
            raise

    def __enter__(self) -> Self:
        if self._entered:
            raise _violation("non_disposable_worktree", "worktree guard cannot be re-entered")
        _reject_root_overlap(self.canonical_repository, self.worktree)
        canonical_root = DescriptorRoot(self.canonical_repository)
        worktree_root: DescriptorRoot | None = None
        linked_worktree_verified = False
        try:
            worktree_root = DescriptorRoot(self.worktree)
            if canonical_root.device == worktree_root.device and (
                canonical_root.identity_cid == worktree_root.identity_cid
            ):
                raise _violation("root_overlap", "canonical and disposable roots are identical")
            canonical_top = _git_text(self.canonical_repository, ("rev-parse", "--show-toplevel"))
            worktree_top = _git_text(self.worktree, ("rev-parse", "--show-toplevel"))
            if canonical_top != self.canonical_repository or worktree_top != self.worktree:
                raise _violation("git_metadata_untrusted", "Git top-level identity is inconsistent")
            _validate_linked_git_metadata(self.canonical_repository, self.worktree)
            linked_worktree_verified = True
            symbolic = _run_git(
                self.worktree,
                ("symbolic-ref", "-q", "HEAD"),
                expected=(0, 1),
            )
            if symbolic.stdout:
                raise _violation("protected_ref", "sandbox worktree is attached to a branch")
            head = _git_text(self.worktree, ("rev-parse", "--verify", "HEAD"))
            if head != self.expected_base_commit:
                raise _violation("base_mismatch", "sandbox worktree does not match its exact base")
            if _run_git(
                self.worktree,
                ("status", "--porcelain=v2", "-z", "--untracked-files=all"),
                expected=(0,),
            ).stdout:
                raise _violation("worktree_dirty", "sandbox worktree must begin clean")
            self._canonical_root = canonical_root
            self._worktree_root = worktree_root
            self._snapshot = _capture_git_snapshot(self.canonical_repository, self.protected_refs)
            self._entered = True
            return self
        except BaseException:
            if worktree_root is not None:
                worktree_root.close()
            canonical_root.close()
            if linked_worktree_verified:
                try:
                    _run_git(
                        self.canonical_repository,
                        ("worktree", "remove", "--force", self.worktree),
                        expected=(0,),
                    )
                except BaseException as cleanup_exc:
                    raise _violation(
                        "cleanup_unproven",
                        f"failed-admission worktree cleanup is unproven: {cleanup_exc}",
                        RepairRequiredError,
                    ) from cleanup_exc
            raise

    def __exit__(self, kind: Any, value: Any, traceback: Any) -> bool:
        if not self._entered or self._snapshot is None:
            raise _violation(
                "cleanup_unproven", "worktree guard exited without admission", RepairRequiredError
            )
        postflight_error: BaseException | None = None
        cleanup_error: BaseException | None = None
        try:
            assert self._canonical_root is not None and self._worktree_root is not None
            self._canonical_root._ensure_stable()
            self._worktree_root._ensure_stable()
            before_cleanup = _capture_git_snapshot(self.canonical_repository, self.protected_refs)
            if before_cleanup != self._snapshot:
                postflight_error = _violation(
                    "canonical_drift",
                    "canonical Git state drifted during sandbox execution",
                    PartialEffectError,
                )
        except BaseException as exc:
            postflight_error = exc
        try:
            # Removal is still attempted after ref/index drift when both anchored
            # roots remain the admitted identities.  This prevents a hostile
            # command from defeating disposable cleanup by first touching a ref.
            assert self._canonical_root is not None and self._worktree_root is not None
            self._canonical_root._ensure_stable()
            self._worktree_root._ensure_stable()
            _run_git(
                self.canonical_repository,
                ("worktree", "remove", "--force", self.worktree),
                expected=(0,),
            )
            if os.path.lexists(self.worktree):
                raise _violation(
                    "cleanup_unproven",
                    "disposable worktree remains after removal",
                    RepairRequiredError,
                )
            self.cleanup_proven = True
        except BaseException as exc:
            cleanup_error = _violation(
                "cleanup_unproven",
                f"disposable worktree cleanup is unproven: {exc}",
                RepairRequiredError,
            )
            self.cleanup_proven = False
        try:
            after_cleanup = _capture_git_snapshot(self.canonical_repository, self.protected_refs)
            if after_cleanup != self._snapshot:
                postflight_error = postflight_error or _violation(
                    "canonical_drift",
                    "canonical Git state drifted during cleanup",
                    PartialEffectError,
                )
            self.canonical_unchanged = after_cleanup == self._snapshot
        except BaseException as exc:
            postflight_error = postflight_error or exc
            self.canonical_unchanged = False
        finally:
            if self._worktree_root is not None:
                self._worktree_root.close()
            if self._canonical_root is not None:
                self._canonical_root.close()
            self._entered = False
        if cleanup_error is not None:
            raise cleanup_error
        if postflight_error is not None:
            raise postflight_error
        return False


def _directory_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_gid,
    )


def _reject_root_overlap(canonical: str, disposable: str) -> None:
    try:
        common = os.path.commonpath((canonical, disposable))
    except ValueError as exc:
        raise _violation("root_overlap", "sandbox roots cannot be compared") from exc
    if common in {canonical, disposable}:
        raise _violation("root_overlap", "canonical and disposable roots overlap")


def _git_environment() -> Mapping[str, str]:
    return {
        "HOME": "/dev/null",
        "PATH": "/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_OPTIONAL_LOCKS": "0",
    }


def _run_git(
    cwd: str,
    arguments: tuple[str, ...],
    *,
    expected: tuple[int, ...] = (0,),
) -> subprocess.CompletedProcess[bytes]:
    try:
        result = subprocess.run(
            (
                "/usr/bin/git",
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                "core.fsmonitor=false",
                *arguments,
            ),
            cwd=cwd,
            env=dict(_git_environment()),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            shell=False,
            check=False,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise _violation("git_metadata_untrusted", "bounded local Git inspection failed") from exc
    if result.returncode not in expected:
        raise _violation(
            "git_metadata_untrusted", "local Git inspection returned an unexpected status"
        )
    if len(result.stdout) + len(result.stderr) > MAX_PROVIDER_OUTPUT_BYTES:
        raise _violation("output_limit", "local Git inspection output exceeds its bound")
    return result


def _git_text(cwd: str, arguments: tuple[str, ...]) -> str:
    result = _run_git(cwd, arguments)
    try:
        text = result.stdout.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise _violation("git_metadata_untrusted", "Git identity output is not UTF-8") from exc
    if not text or "\x00" in text or "\n" in text:
        raise _violation("git_metadata_untrusted", "Git identity output is malformed")
    return text


def _hash_optional_file(path: str) -> str:
    parent, name = os.path.split(_absolute_path(path, field_name="git_metadata_path"))
    with DescriptorRoot(parent) as root:
        try:
            descriptor = os.open(
                name,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=root._fd,
            )
        except FileNotFoundError:
            return hashlib.sha256(b"").hexdigest()
        except OSError as exc:
            raise _violation(
                "git_metadata_untrusted", "Git metadata file cannot be anchored"
            ) from exc
        try:
            metadata = os.fstat(descriptor)
            identity = (
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_mode,
                metadata.st_uid,
                metadata.st_gid,
                metadata.st_nlink,
                metadata.st_size,
                metadata.st_mtime_ns,
                metadata.st_ctime_ns,
            )
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_size > MAX_RECORD_BYTES
                or os.path.realpath(f"/proc/self/fd/{descriptor}") != path
            ):
                raise _violation(
                    "git_metadata_untrusted", "Git metadata file is not bounded regular data"
                )
            digest = hashlib.sha256()
            while True:
                chunk = os.read(descriptor, 65_536)
                if not chunk:
                    break
                digest.update(chunk)
            after = os.fstat(descriptor)
            if identity != (
                after.st_dev,
                after.st_ino,
                after.st_mode,
                after.st_uid,
                after.st_gid,
                after.st_nlink,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                raise _violation("git_metadata_untrusted", "Git metadata identity drifted")
            root._ensure_stable()
            return digest.hexdigest()
        finally:
            os.close(descriptor)


def _capture_git_snapshot(repository: str, protected_refs: tuple[str, ...]) -> _GitSnapshot:
    git_dir = _git_text(repository, ("rev-parse", "--absolute-git-dir"))
    common_dir = _git_text(repository, ("rev-parse", "--git-common-dir"))
    if not os.path.isabs(common_dir):
        common_dir = os.path.normpath(os.path.join(repository, common_dir))
    common_dir = _absolute_path(common_dir, field_name="git_common_dir")
    head = _git_text(repository, ("rev-parse", "--verify", "HEAD"))
    tree = _git_text(repository, ("rev-parse", "--verify", "HEAD^{tree}"))
    status = _run_git(
        repository,
        ("status", "--porcelain=v2", "-z", "--untracked-files=all"),
    ).stdout
    refs_by_name: dict[str, str | None] = {}
    all_heads = _run_git(
        repository,
        ("for-each-ref", "--format=%(refname)%00%(objectname)", "refs/heads"),
    ).stdout
    try:
        for row in all_heads.decode("ascii").splitlines():
            name, oid = row.split("\x00", 1)
            if not name.startswith("refs/heads/") or not _HEX_COMMIT.fullmatch(oid):
                raise ValueError("malformed local branch row")
            refs_by_name[name] = oid
    except (UnicodeDecodeError, ValueError) as exc:
        raise _violation("git_metadata_untrusted", "local branch inventory is malformed") from exc
    for ref in protected_refs:
        result = _run_git(
            repository,
            ("show-ref", "--verify", "--hash", ref),
            expected=(0, 128),
        )
        refs_by_name[ref] = result.stdout.decode("ascii").strip() or None
    return _GitSnapshot(
        head=head,
        head_tree=tree,
        status_sha256=hashlib.sha256(status).hexdigest(),
        head_file_sha256=_hash_optional_file(os.path.join(git_dir, "HEAD")),
        index_sha256=_hash_optional_file(os.path.join(git_dir, "index")),
        config_sha256=_hash_optional_file(os.path.join(common_dir, "config")),
        packed_refs_sha256=_hash_optional_file(os.path.join(common_dir, "packed-refs")),
        protected_refs=tuple(sorted(refs_by_name.items())),
    )


def _validate_linked_git_metadata(canonical: str, worktree: str) -> None:
    marker = os.path.join(worktree, ".git")
    try:
        descriptor = os.open(marker, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError as exc:
        raise _violation(
            "git_metadata_untrusted", "linked worktree Git marker is unavailable"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size > MAX_PATH_BYTES
            or os.path.realpath(f"/proc/self/fd/{descriptor}") != marker
        ):
            raise _violation("git_metadata_untrusted", "linked worktree Git marker is unsafe")
        raw = bytearray()
        while len(raw) <= MAX_PATH_BYTES:
            chunk = os.read(descriptor, min(4096, MAX_PATH_BYTES + 1 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
        if len(raw) > MAX_PATH_BYTES:
            raise _violation("git_metadata_untrusted", "linked worktree Git marker is unsafe")
    finally:
        os.close(descriptor)
    try:
        line = bytes(raw).decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise _violation(
            "git_metadata_untrusted", "linked worktree Git marker is malformed"
        ) from exc
    if not line.startswith("gitdir: "):
        raise _violation("git_metadata_untrusted", "linked worktree Git marker is malformed")
    admin = _absolute_path(line[8:], field_name="worktree_gitdir")
    common = _git_text(canonical, ("rev-parse", "--git-common-dir"))
    if not os.path.isabs(common):
        common = os.path.normpath(os.path.join(canonical, common))
    common = _absolute_path(common, field_name="git_common_dir")
    admin_parent = os.path.join(common, "worktrees")
    if os.path.commonpath((admin_parent, admin)) != admin_parent or admin == admin_parent:
        raise _violation(
            "git_metadata_untrusted", "linked worktree metadata is outside Git authority"
        )
    with DescriptorRoot(common), DescriptorRoot(admin):
        pass


def _capture_executable(path: str) -> tuple[str, str]:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError as exc:
        reason = "path_symlink" if exc.errno == errno.ELOOP else "executable_not_allowlisted"
        raise _violation(reason, "executable cannot be identity-anchored") from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or not metadata.st_mode & 0o111
            or metadata.st_mode & 0o022
            or not 0 < metadata.st_size <= MAX_EXECUTABLE_BYTES
            or os.path.realpath(f"/proc/self/fd/{descriptor}") != path
        ):
            raise _violation("executable_not_allowlisted", "executable identity is not trusted")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1_048_576)
            if not chunk:
                break
            digest.update(chunk)
        sha = digest.hexdigest()
        identity = {
            "sha256": sha,
            "size": metadata.st_size,
            "mode": stat.S_IMODE(metadata.st_mode),
            "uid": metadata.st_uid,
            "gid": metadata.st_gid,
        }
        return sha, _raw_cid(_canonical_bytes(identity))
    finally:
        os.close(descriptor)


def _is_static_elf(path: str) -> bool:
    """Return true only for a native ELF image without a PT_INTERP loader."""

    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError:
        return False
    try:
        header = os.pread(descriptor, 64, 0)
        if len(header) < 52 or header[:4] != b"\x7fELF" or header[6] != 1:
            return False
        elf_class = header[4]
        byte_order = header[5]
        if elf_class not in {1, 2} or byte_order not in {1, 2}:
            return False
        endian = "<" if byte_order == 1 else ">"
        if elf_class == 2:
            if len(header) < 64:
                return False
            program_offset = struct.unpack_from(f"{endian}Q", header, 32)[0]
            program_entry_size = struct.unpack_from(f"{endian}H", header, 54)[0]
            program_count = struct.unpack_from(f"{endian}H", header, 56)[0]
        else:
            program_offset = struct.unpack_from(f"{endian}I", header, 28)[0]
            program_entry_size = struct.unpack_from(f"{endian}H", header, 42)[0]
            program_count = struct.unpack_from(f"{endian}H", header, 44)[0]
        if not 0 < program_entry_size <= 1024 or not 0 < program_count <= 4096:
            return False
        for index in range(program_count):
            entry = os.pread(
                descriptor,
                program_entry_size,
                program_offset + index * program_entry_size,
            )
            if len(entry) != program_entry_size:
                return False
            if struct.unpack_from(f"{endian}I", entry, 0)[0] == 3:  # PT_INTERP
                return False
        return True
    except (OSError, struct.error):
        return False
    finally:
        os.close(descriptor)


def _trusted_executable_available(path: str, *, require_setuid: bool) -> bool:
    try:
        metadata = os.stat(path, follow_symlinks=False)
    except OSError:
        return False
    return bool(
        stat.S_ISREG(metadata.st_mode)
        and metadata.st_uid == 0
        and metadata.st_mode & 0o111
        and not metadata.st_mode & 0o022
        and (not require_setuid or metadata.st_mode & stat.S_ISUID)
    )


def _probe_landlock_abi() -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    result = int(
        libc.syscall(
            _LANDLOCK_CREATE_RULESET,
            ctypes.c_void_p(),
            ctypes.c_size_t(0),
            ctypes.c_uint(_LANDLOCK_CREATE_RULESET_VERSION),
        )
    )
    return max(0, result)


def _probe_openat2() -> bool:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    root = os.open("/", os.O_PATH | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        how = _OpenHow(
            os.O_PATH | os.O_DIRECTORY | os.O_CLOEXEC,
            0,
            _OPENAT2_RESOLVE,
        )
        descriptor = int(
            libc.syscall(
                _OPENAT2_SYSCALL,
                ctypes.c_int(root),
                ctypes.c_char_p(b"."),
                ctypes.byref(how),
                ctypes.sizeof(how),
            )
        )
        if descriptor < 0:
            return False
        os.close(descriptor)
        return True
    finally:
        os.close(root)


def _redact_preview(
    value: bytes,
    parent_secret_values: Sequence[bytes],
    *,
    limit: int,
) -> tuple[str, bool]:
    text = value.decode("utf-8", "replace").replace("\x00", "")
    structured_detected, wipe_all = _structured_secret_material(text)
    detected = structured_detected
    if wipe_all and structured_detected:
        text = '"[redacted]"'
    secrets_to_replace: set[str] = set()
    for raw in parent_secret_values:
        if len(raw) < 4:
            continue
        secret = raw.decode("utf-8", "ignore")
        if secret:
            secrets_to_replace.add(secret)
    variants: set[str] = set()
    for secret in secrets_to_replace:
        variants.add(secret)
        variants.add(json.dumps(secret, ensure_ascii=True)[1:-1])
        variants.add(json.dumps(secret, ensure_ascii=False)[1:-1])
    if variants:
        literal_pattern = re.compile(
            "|".join(re.escape(secret) for secret in sorted(variants, key=len, reverse=True))
        )
        if literal_pattern.search(text):
            text = literal_pattern.sub("[redacted]", text)
            detected = True
    if _SECRET_TEXT.search(text) or _BEARER_TEXT.search(text):
        detected = True
    text = _SECRET_TEXT.sub("[redacted]", text)
    text = _BEARER_TEXT.sub("[redacted]", text)
    text = _HOST_PATH_TEXT.sub("[path]", text)
    encoded = text.encode("utf-8")[:limit]
    return encoded.decode("utf-8", "ignore"), detected


def _ambient_secret_values(environment: Mapping[str, str]) -> tuple[bytes, ...]:
    values: list[bytes] = []
    total_bytes = 0
    secret_bytes = 0
    if len(environment) > 1024:
        raise MalformedError("parent environment exceeds its scan item bound")
    for key, value in environment.items():
        if not isinstance(key, str) or not isinstance(value, str) or "\x00" in key + value:
            raise MalformedError("parent environment contains malformed text")
        encoded = value.encode("utf-8", "ignore")
        total_bytes += len(key.encode("utf-8")) + len(encoded)
        if total_bytes > MAX_RECORD_BYTES:
            raise MalformedError("parent environment exceeds its scan byte bound")
        normalized = key.upper()
        if (
            normalized in _FORBIDDEN_ENV_EXACT
            or normalized.startswith(_FORBIDDEN_ENV_PREFIXES)
            or normalized.endswith("_PROXY")
            or _SECRET_KEY.search(normalized)
        ):
            if value:
                secret_bytes += len(encoded)
                if (
                    len(values) >= MAX_PARENT_SECRET_VALUES
                    or secret_bytes > MAX_PARENT_SECRET_BYTES
                ):
                    raise MalformedError("parent credential scan inputs exceed their frozen bound")
                values.append(encoded)
    return tuple(values)


_INNER_GATE_SOURCE: Final[str] = r"""
import base64, ctypes, errno, hashlib, json, os, resource, stat, sys

SCHEMA = "pcce-sandbox-inner-gate@1"
FAILURE = 125
CREATE = 444
ADD = 445
RESTRICT = 446
VERSION = 1
OPENAT2 = 437
RESOLVE = 0x01 | 0x02 | 0x04 | 0x08
RULE_PATH = 1
EXECUTE = 1 << 0
WRITE_FILE = 1 << 1
READ_FILE = 1 << 2
READ_DIR = 1 << 3
REMOVE_DIR = 1 << 4
REMOVE_FILE = 1 << 5
MAKE_CHAR = 1 << 6
MAKE_DIR = 1 << 7
MAKE_REG = 1 << 8
MAKE_SOCK = 1 << 9
MAKE_FIFO = 1 << 10
MAKE_BLOCK = 1 << 11
MAKE_SYM = 1 << 12
REFER = 1 << 13
TRUNCATE = 1 << 14
IOCTL_DEV = 1 << 15
READ = READ_FILE | READ_DIR
WRITE = WRITE_FILE | REMOVE_DIR | REMOVE_FILE | MAKE_CHAR | MAKE_DIR | MAKE_REG | MAKE_SOCK | MAKE_FIFO | MAKE_BLOCK | MAKE_SYM | REFER | TRUNCATE | IOCTL_DEV
ALL = EXECUTE | READ | WRITE
O_PATH = getattr(os, "O_PATH", 0)

class Ruleset(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64), ("handled_access_net", ctypes.c_uint64), ("scoped", ctypes.c_uint64)]

class PathRule(ctypes.Structure):
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]

class OpenHow(ctypes.Structure):
    _fields_ = [("flags", ctypes.c_uint64), ("mode", ctypes.c_uint64), ("resolve", ctypes.c_uint64)]

def fail():
    os._exit(FAILURE)

def sha256_fd(fd):
    os.lseek(fd, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while True:
        chunk = os.read(fd, 1048576)
        if not chunk:
            break
        digest.update(chunk)
    os.lseek(fd, 0, os.SEEK_SET)
    return digest.hexdigest()

def open_beneath(root_fd, relative, directory=False):
    parts = relative.split("/")
    if not relative or any(p in ("", ".", "..", ".git") for p in parts):
        raise RuntimeError("unsafe relative path")
    flags = O_PATH | os.O_NOFOLLOW | os.O_CLOEXEC
    if directory:
        flags |= os.O_DIRECTORY
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    how = OpenHow(flags, 0, RESOLVE)
    preferred = int(libc.syscall(OPENAT2, ctypes.c_int(root_fd), ctypes.c_char_p(relative.encode("utf-8")), ctypes.byref(how), ctypes.sizeof(how)))
    if preferred >= 0:
        return preferred
    if ctypes.get_errno() not in (errno.ENOSYS, errno.EINVAL, errno.EPERM):
        raise OSError(ctypes.get_errno(), "openat2 beneath")
    current = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            nxt = os.open(part, O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=current)
            os.close(current)
            current = nxt
        return os.open(parts[-1], flags, dir_fd=current)
    finally:
        os.close(current)

def lower_limit(kind, value):
    soft, hard = resource.getrlimit(kind)
    admitted = value if hard == resource.RLIM_INFINITY else min(value, hard)
    resource.setrlimit(kind, (admitted, admitted))

try:
    if len(sys.argv) != 2:
        raise RuntimeError("inner gate argv")
    raw = base64.b64decode(sys.argv[1].encode("ascii"), validate=True)
    config = json.loads(raw.decode("utf-8"))
    fields = {"schema", "worktree", "executable", "sha256", "argv", "read_paths", "write_paths", "cpu", "memory", "file", "nofile", "nproc"}
    if not isinstance(config, dict) or set(config) != fields or config["schema"] != SCHEMA:
        raise RuntimeError("inner gate config")
    root_path = config["worktree"]
    executable = config["executable"]
    argv = config["argv"]
    if not isinstance(root_path, str) or not root_path.startswith("/") or not isinstance(executable, str) or not executable.startswith("/"):
        raise RuntimeError("inner gate paths")
    if not isinstance(argv, list) or not argv or argv[0] != executable or any(not isinstance(item, str) or "\0" in item for item in argv):
        raise RuntimeError("inner gate command")
    expected_env = {"HOME", "TMPDIR", "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME", "PATH", "LANG", "LC_ALL"}
    if set(os.environ) != expected_env or os.environ["TMPDIR"] != os.environ["HOME"]:
        raise RuntimeError("inner gate environment")
    for raw_fd in os.listdir("/proc/self/fd"):
        try:
            inherited_fd = int(raw_fd)
        except ValueError:
            continue
        if inherited_fd > 2:
            try:
                os.close(inherited_fd)
            except OSError:
                pass
    root_fd = os.open(root_path, O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    executable_fd = os.open(executable, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    root_stat = os.fstat(root_fd)
    exe_stat = os.fstat(executable_fd)
    if not stat.S_ISDIR(root_stat.st_mode) or not stat.S_ISREG(exe_stat.st_mode) or not exe_stat.st_mode & 0o111 or exe_stat.st_mode & 0o022 or os.path.realpath("/proc/self/fd/" + str(root_fd)) != root_path or os.path.realpath("/proc/self/fd/" + str(executable_fd)) != executable or sha256_fd(executable_fd) != config["sha256"]:
        raise RuntimeError("inner gate identity")
    exe_after_hash = os.fstat(executable_fd)
    if (exe_after_hash.st_dev, exe_after_hash.st_ino, exe_after_hash.st_mode, exe_after_hash.st_uid, exe_after_hash.st_gid, exe_after_hash.st_size, exe_after_hash.st_mtime_ns, exe_after_hash.st_ctime_ns) != (exe_stat.st_dev, exe_stat.st_ino, exe_stat.st_mode, exe_stat.st_uid, exe_stat.st_gid, exe_stat.st_size, exe_stat.st_mtime_ns, exe_stat.st_ctime_ns):
        raise RuntimeError("inner gate executable drift")
    opened = []
    read_fds = []
    write_fds = []
    root_read_fds = []
    for entry in os.listdir(root_path):
        if entry == ".git" or entry in (".", ".."):
            continue
        fd = os.open(entry, O_PATH | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=root_fd)
        entry_stat = os.fstat(fd)
        if entry_stat.st_dev != root_stat.st_dev:
            raise RuntimeError("root read scope device")
        if stat.S_ISREG(entry_stat.st_mode) or stat.S_ISDIR(entry_stat.st_mode):
            root_read_fds.append(fd)
        else:
            os.close(fd)
    for relative in config["read_paths"]:
        fd = open_beneath(root_fd, relative)
        if os.fstat(fd).st_dev != root_stat.st_dev:
            raise RuntimeError("read scope device")
        read_fds.append(fd)
    for relative in config["write_paths"]:
        fd = open_beneath(root_fd, relative, directory=True)
        if os.fstat(fd).st_dev != root_stat.st_dev:
            raise RuntimeError("write scope device")
        write_fds.append(fd)
    lower_limit(resource.RLIMIT_CPU, config["cpu"])
    lower_limit(resource.RLIMIT_AS, config["memory"])
    lower_limit(resource.RLIMIT_FSIZE, config["file"])
    lower_limit(resource.RLIMIT_NOFILE, config["nofile"])
    lower_limit(resource.RLIMIT_NPROC, config["nproc"])
    lower_limit(resource.RLIMIT_CORE, 0)
    lower_limit(resource.RLIMIT_MEMLOCK, 0)
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    abi = int(libc.syscall(CREATE, ctypes.c_void_p(), ctypes.c_size_t(0), ctypes.c_uint(VERSION)))
    if abi < 6:
        raise RuntimeError("Landlock ABI")
    ruleset_attr = Ruleset(ALL, 0, 0)
    ruleset_fd = int(libc.syscall(CREATE, ctypes.byref(ruleset_attr), ctypes.sizeof(ruleset_attr), ctypes.c_uint(0)))
    if ruleset_fd < 0:
        raise RuntimeError("Landlock ruleset")
    def add(fd, access):
        rule = PathRule(access, fd)
        if int(libc.syscall(ADD, ruleset_fd, RULE_PATH, ctypes.byref(rule), ctypes.c_uint(0))) != 0:
            raise RuntimeError("Landlock rule")
    add(root_fd, READ_DIR)
    for fd in root_read_fds:
        add(fd, READ if stat.S_ISDIR(os.fstat(fd).st_mode) else READ_FILE)
    add(executable_fd, EXECUTE | READ_FILE)
    for fd in read_fds:
        add(fd, READ if stat.S_ISDIR(os.fstat(fd).st_mode) else READ_FILE)
    for fd in write_fds:
        add(fd, READ | WRITE)
    home_fd = os.open(os.environ["HOME"], O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    opened.append(home_fd)
    add(home_fd, READ | WRITE)
    for device in ("/dev/null", "/dev/zero", "/dev/full"):
        fd = os.open(device, O_PATH | os.O_NOFOLLOW | os.O_CLOEXEC)
        opened.append(fd)
        add(fd, READ_FILE | WRITE_FILE | IOCTL_DEV)
    if int(libc.syscall(RESTRICT, ruleset_fd, ctypes.c_uint(0))) != 0:
        raise RuntimeError("Landlock restrict")
    os.close(ruleset_fd)
    for fd in root_read_fds + read_fds + write_fds + opened:
        os.close(fd)
    os.close(root_fd)
    os.set_inheritable(executable_fd, True)
    os.execve(executable_fd, argv, os.environ)
except BaseException:
    fail()
"""

_INNER_GATE_SHA256: Final[str] = hashlib.sha256(_INNER_GATE_SOURCE.encode("utf-8")).hexdigest()


class SandboxExecutor:
    """Execute one exact permit through the deny-all direct sandbox boundary."""

    __slots__ = ("policy", "permit", "capabilities")

    def __init__(
        self,
        policy: SandboxPolicy,
        permit: SandboxExecutionPermit,
        *,
        capabilities: SandboxCapabilityReport | None = None,
    ) -> None:
        if not isinstance(policy, SandboxPolicy) or not isinstance(permit, SandboxExecutionPermit):
            raise MalformedError("sandbox executor requires typed policy and permit")
        if capabilities is not None and not isinstance(capabilities, SandboxCapabilityReport):
            raise MalformedError("sandbox executor requires a typed capability report")
        self.policy = policy
        self.permit = permit
        self.capabilities = capabilities or SandboxCapabilityReport.probe()
        self._bind_permit()

    def _bind_permit(self) -> None:
        if (
            self.permit.repository_state_cid != self.policy.repository_state_cid
            or self.permit.policy_cid != self.policy.cid
        ):
            raise _violation("route_mismatch", "sandbox permit does not bind the exact policy")
        if self.permit.executable_identity_cid != self.policy.executable_identity_cid:
            raise _violation(
                "executable_identity_drift",
                "sandbox permit does not bind the executable identity",
            )
        if self.permit.argv != self.policy.allowed_argv:
            raise _violation("argv_mismatch", "sandbox permit does not bind the exact argv")
        if (
            self.permit.network_mode != self.policy.network_mode
            or self.permit.route_cid != self.policy.route_cid
            or self.permit.endpoint_generation_cid != self.policy.endpoint_generation_cid
        ):
            raise _violation("route_mismatch", "sandbox permit does not bind the exact policy")

    def _reserve_permit(self, now_epoch: int) -> None:
        if not self.permit.issued_at_epoch <= now_epoch < self.permit.expires_at_epoch:
            raise _violation("permit_expired", "sandbox permit is outside its validity window")
        with _PERMIT_NONCE_LOCK:
            if self.permit.nonce in _USED_PERMIT_NONCES:
                raise _violation("permit_replayed", "sandbox permit was already used")
            if len(_USED_PERMIT_NONCES) >= MAX_PROCESS_LOCAL_NONCES:
                raise _violation(
                    "capability_unavailable",
                    "bounded process-local permit replay memory is exhausted",
                    UnavailableCapabilityError,
                )
            _USED_PERMIT_NONCES.add(self.permit.nonce)

    def _inner_command_policy(self, worktree: str) -> CommandPolicy:
        if (
            CommandPolicy is not _REVIEWED_COMMAND_POLICY
            or invoke_command is not _REVIEWED_INVOKE_COMMAND
            or hashlib.sha256(_INNER_GATE_SOURCE.encode("utf-8")).hexdigest() != _INNER_GATE_SHA256
        ):
            raise _violation(
                "capability_unavailable",
                "reviewed sandbox backend identity is unavailable",
                UnavailableCapabilityError,
            )
        digest, identity = _capture_executable(self.policy.allowed_executable)
        if (
            digest != self.policy.executable_sha256
            or identity != self.policy.executable_identity_cid
        ):
            raise _violation("executable_identity_drift", "allowlisted executable identity drifted")
        if not _is_static_elf(self.policy.allowed_executable):
            raise _violation(
                "capability_unavailable",
                "exact descendant-exec confinement requires a static ELF target",
                UnavailableCapabilityError,
            )
        root = DescriptorRoot(worktree)
        try:
            for relative in self.policy.allowed_read_paths:
                descriptor = root._open_components(relative, final_flags=os.O_PATH)
                os.close(descriptor)
            for relative in self.policy.allowed_write_paths:
                root.require_directory(relative)
        finally:
            root.close()
        interpreter = str(Path(sys.executable).resolve(strict=True))
        config = {
            "schema": _INNER_GATE_SCHEMA,
            "worktree": worktree,
            "executable": self.policy.allowed_executable,
            "sha256": self.policy.executable_sha256,
            "argv": self.policy.allowed_argv,
            "read_paths": self.policy.allowed_read_paths,
            "write_paths": self.policy.allowed_write_paths,
            "cpu": self.policy.cpu_seconds,
            "memory": self.policy.memory_bytes,
            "file": self.policy.output_file_bytes,
            "nofile": self.policy.open_files,
            "nproc": self.policy.processes,
        }
        encoded = base64.b64encode(_canonical_bytes(config)).decode("ascii")
        return CommandPolicy(
            executable=interpreter,
            allowed_executables=(interpreter,),
            cwd=worktree,
            allowed_cwds=(worktree,),
            arguments=("-I", "-c", _INNER_GATE_SOURCE, encoded),
            environment={},
            timeout_seconds=float(self.policy.timeout_seconds),
        )

    def execute(
        self,
        request: Mapping[str, Any],
        guard: DisposableWorktreeGuard,
        *,
        cancellation: CancellationToken | None = None,
        now_epoch: int | None = None,
        parent_environment: Mapping[str, str] | None = None,
    ) -> SandboxExecutionResult:
        if not isinstance(request, Mapping):
            raise MalformedError("sandbox request must be a mapping")
        if not isinstance(guard, DisposableWorktreeGuard):
            raise MalformedError("sandbox executor requires a disposable worktree guard")
        if cancellation is not None and not isinstance(cancellation, CancellationToken):
            raise MalformedError("sandbox cancellation must be a typed token")
        if guard.expected_base_commit != self.permit.worktree_base_commit:
            raise _violation("base_mismatch", "worktree guard base differs from permit")
        request_snapshot = _deep_freeze(_decode_json_object(_canonical_bytes(request)))
        now = (
            int(time.time())
            if now_epoch is None
            else _bounded_int(
                now_epoch,
                field_name="now_epoch",
                minimum=0,
                maximum=MAX_SAFE_INTEGER,
            )
        )
        started_wall = now
        started_mono = time.monotonic()
        execution: CommandExecution | None = None
        caught: ProofContextError | None = None
        stage = "preflight"
        secret_values = _ambient_secret_values(
            os.environ if parent_environment is None else parent_environment
        )
        stdout = b""
        stderr = b""
        try:
            with guard:
                if self.policy.network_mode == "route_endpoint_allowlist":
                    raise _violation(
                        "endpoint_enforcement_unavailable",
                        "route-scoped endpoint enforcement is unavailable before spawn",
                        UnavailableCapabilityError,
                    )
                if not self.capabilities.direct_execution_supported:
                    raise _violation(
                        "capability_unavailable",
                        "required direct sandbox capabilities are unavailable before spawn",
                        UnavailableCapabilityError,
                    )
                self._reserve_permit(now)
                if cancellation is not None:
                    cancellation.check()
                stage = "execution"
                command_policy = self._inner_command_policy(guard.worktree)
                try:
                    execution = _REVIEWED_INVOKE_COMMAND(
                        command_policy,
                        request_snapshot,
                        cancellation,
                    )
                except ProofContextError:
                    raise
                except Exception as exc:
                    raise _violation(
                        "capability_unavailable",
                        "sandbox execution backend failed closed",
                        UnavailableCapabilityError,
                    ) from exc
                stdout = execution.stdout
                stderr = execution.stderr
                if len(stdout) + len(stderr) > self.policy.aggregate_output_bytes:
                    raise _violation("output_limit", "sandbox output exceeds its aggregate bound")
                stage = "postflight"
        except (
            BoundaryViolationError,
            UnavailableCapabilityError,
            ProofTimeoutError,
            ProofCancelledError,
            PartialEffectError,
            RepairRequiredError,
        ) as exc:
            caught = exc
        finished = int(time.time()) if now_epoch is None else max(now, now_epoch)
        latency = max(0, int((time.monotonic() - started_mono) * 1000))
        stdout_preview, stdout_secret = _redact_preview(stdout, secret_values, limit=MAX_LOG_BYTES)
        stderr_preview, stderr_secret = _redact_preview(stderr, secret_values, limit=MAX_LOG_BYTES)
        secret_passed = not (stdout_secret or stderr_secret)
        trace: SandboxDenialTrace | None = None
        reason: str | None = None
        if caught is not None:
            reason = _reason_for_exception(caught)
            trace = SandboxDenialTrace.from_exception(
                caught,
                stage=stage,
                observed_at_epoch=finished,
                subject_cid=self.permit.cid,
            )
            if isinstance(caught, ProofTimeoutError):
                status = "timeout"
            elif isinstance(caught, ProofCancelledError):
                status = "cancelled"
            elif isinstance(caught, UnavailableCapabilityError):
                status = "unavailable"
            elif isinstance(caught, RepairRequiredError):
                status = "repair_required"
            elif isinstance(caught, PartialEffectError):
                status = "partial_effect"
            else:
                status = "denied"
        elif not secret_passed:
            reason = "secret_detected"
            trace = SandboxDenialTrace(
                reason=reason,
                stage="postflight",
                observed_at_epoch=finished,
                subject_cid=self.permit.cid,
                detail="sandbox output matched forbidden ambient credential material",
            )
            status = "denied"
        elif execution is None:
            reason = "capability_unavailable"
            trace = SandboxDenialTrace(
                reason=reason,
                stage="postflight",
                observed_at_epoch=finished,
                subject_cid=self.permit.cid,
                detail="sandbox produced no execution observation",
            )
            status = "unavailable"
        elif execution.returncode != 0:
            reason = "resource_limit" if execution.returncode in {125, 137, 152, 153} else None
            if reason is not None:
                trace = SandboxDenialTrace(
                    reason=reason,
                    stage="execution",
                    observed_at_epoch=finished,
                    subject_cid=self.permit.cid,
                    detail="sandbox process terminated at an enforced resource boundary",
                )
            status = "failed"
        elif guard.cleanup_proven and guard.canonical_unchanged:
            status = "completed_unpublished"
        else:
            reason = "cleanup_unproven"
            trace = SandboxDenialTrace(
                reason=reason,
                stage="cleanup",
                observed_at_epoch=finished,
                subject_cid=self.permit.cid,
                detail="worktree cleanup or canonical-state preservation is unproven",
            )
            status = "repair_required"
        receipt = SandboxExecutionReceipt(
            permit_cid=self.permit.cid,
            policy_cid=self.policy.cid,
            capability_report_cid=self.capabilities.cid,
            status=status,
            reason=reason,
            started_at_epoch=started_wall,
            finished_at_epoch=finished,
            latency_ms=latency if execution is None else execution.latency_ms,
            returncode=None if execution is None else execution.returncode,
            stdout_cid=_raw_cid(stdout),
            stderr_cid=_raw_cid(stderr),
            stdout_bytes=len(stdout),
            stderr_bytes=len(stderr),
            denial_trace_cid=None if trace is None else trace.cid,
            worktree_cleanup_proven=guard.cleanup_proven,
            canonical_unchanged=guard.canonical_unchanged,
            secret_scan_passed=secret_passed,
        )
        return SandboxExecutionResult(
            receipt=receipt,
            stdout_preview=stdout_preview,
            stderr_preview=stderr_preview,
            denial_trace=trace,
        )


class SandboxCommandAdapter:
    """Direct wrapper only; it is intentionally absent from the adapter registry."""

    __slots__ = ("executor",)
    runtime_integration_status: Final[str] = RUNTIME_INTEGRATION_STATUS
    approval_authority: Final[bool] = False
    canonical_branch_authority: Final[bool] = False
    publication_authority: Final[bool] = False

    def __init__(self, executor: SandboxExecutor) -> None:
        if not isinstance(executor, SandboxExecutor):
            raise MalformedError("sandbox command adapter requires a SandboxExecutor")
        self.executor = executor

    def execute(
        self,
        request: Mapping[str, Any],
        guard: DisposableWorktreeGuard,
        *,
        cancellation: CancellationToken | None = None,
        now_epoch: int | None = None,
        parent_environment: Mapping[str, str] | None = None,
    ) -> SandboxExecutionResult:
        return self.executor.execute(
            request,
            guard,
            cancellation=cancellation,
            now_epoch=now_epoch,
            parent_environment=parent_environment,
        )


_PUBLIC_SYMBOLS: Final[tuple[str, ...]] = (
    "SandboxPolicy",
    "SandboxExecutionPermit",
    "SandboxCapabilityReport",
    "SandboxExecutionResult",
    "SandboxExecutionReceipt",
    "SandboxDenialTrace",
    "DescriptorRoot",
    "DisposableWorktreeGuard",
    "SandboxExecutor",
    "SandboxCommandAdapter",
    "sandbox_descriptor",
    "sandbox_descriptor_cid",
)


def sandbox_descriptor() -> Mapping[str, Any]:
    """Return the immutable truthful PCCE-071 boundary descriptor."""

    return _deep_freeze(
        {
            "schema": SANDBOX_DESCRIPTOR_SCHEMA,
            "interface": INTERFACE,
            "member_schemas": {
                "policy": SANDBOX_POLICY_SCHEMA,
                "execution_permit": SANDBOX_EXECUTION_PERMIT_SCHEMA,
                "capability_report": SANDBOX_CAPABILITY_REPORT_SCHEMA,
                "denial_trace": SANDBOX_DENIAL_TRACE_SCHEMA,
                "execution_receipt": SANDBOX_EXECUTION_RECEIPT_SCHEMA,
                "execution_result": SANDBOX_EXECUTION_RESULT_SCHEMA,
            },
            "backend": "CommandPolicy/invoke_command+descriptor-inner-gate@1",
            "public_symbols": _PUBLIC_SYMBOLS,
            "fixed_bounds": {
                "aggregate_output_bytes": MAX_PROVIDER_OUTPUT_BYTES,
                "redacted_log_bytes": MAX_LOG_BYTES,
                "max_record_bytes": MAX_RECORD_BYTES,
                "max_arguments": MAX_ARGUMENTS,
                "max_argument_bytes": MAX_ARGUMENT_BYTES,
                "max_permit_ttl_seconds": MAX_PERMIT_TTL_SECONDS,
                "max_cpu_seconds": MAX_CPU_SECONDS,
                "max_memory_bytes": MAX_MEMORY_BYTES,
                "max_processes": MAX_PROCESSES,
                "max_process_local_nonces": MAX_PROCESS_LOCAL_NONCES,
                "max_json_secret_fields": MAX_JSON_SECRET_FIELDS,
                "max_python_secret_fields": MAX_PYTHON_SECRET_FIELDS,
                "max_parent_secret_values": MAX_PARENT_SECRET_VALUES,
                "max_parent_secret_bytes": MAX_PARENT_SECRET_BYTES,
            },
            "required_capabilities": (
                "Linux O_PATH/O_NOFOLLOW descriptor roots",
                "openat2 preferred; component-open fallback carries no production credit",
                "Landlock ABI 6 read/write/execute mediation",
                "PID/user/mount/network namespaces",
                "pidfd process-tree supervision",
                "libseccomp provider socket and namespace denial",
                "hard POSIX resource limits",
                "static ELF target without a PT_INTERP loader",
            ),
            "threat_ids": THREAT_IDS,
            "trust_boundaries": TRUST_BOUNDARIES,
            "control_dependencies": (
                "PC-071",
                "OC-04 existing CommandAdapter direct boundary",
                "PCCE-070 frozen threat model",
                "PCCE-075 hostile integrated execution evidence",
                "PCCE-076 security gate",
            ),
            "network_modes": {
                "deny_all": "direct-observed-testable",
                "route_endpoint_allowlist": "typed-unavailable-before-spawn",
            },
            "denial_reasons": DENIAL_REASONS,
            "unsupported_features": (
                "authoritative adapter-registry routing",
                "InstalledCodex credential and network mediation",
                "bootstrap/lifecycle/verification/apply integration",
                "route-scoped live endpoint enforcement",
                "kernel/container isolation on non-Linux platforms",
                "issuer-authenticated permits",
                "durable cross-process nonce replay prevention",
                "same-UID host-actor mutation exclusion",
                "sandboxed Git worktree materialization",
                "dynamically linked or shebang target executables",
                "worktree writes (the direct backend is proposal/read-only)",
                "publication or canonical-branch authority",
                "benchmark, CI, release, or security qualification",
            ),
            "approval_authority": False,
            "canonical_branch_authority": False,
            "publication_authority": False,
            "runtime_integration_status": RUNTIME_INTEGRATION_STATUS,
            "enforcement_disposition": ENFORCEMENT_DISPOSITION,
            "production_eligible": False,
        }
    )


def sandbox_descriptor_cid() -> str:
    return _raw_cid(_canonical_bytes(sandbox_descriptor()))


__all__: Final[tuple[str, ...]] = _PUBLIC_SYMBOLS
