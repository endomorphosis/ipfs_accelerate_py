"""Residual provider invocation under sealed packet only (WPD-023).

Interface: ``ResidualProviderInvocation@1``

Wraps existing provider execution so argv and context receive **only** sealed
fields from :class:`~ipfs_accelerate_py.agent_supervisor.planning.residual_llm_packet.ResidualLlmPacket`.
This module does not register or invent providers; callers inject the same
provider callable the daemon already uses.

Fail-closed rules:

* Oversized sealed packets / prompt bodies are rejected before dispatch.
* Path-escaping write paths (absolute, ``..``, or outside a path lease) are
  rejected before dispatch.
* Provider process environment excludes secret-shaped variables.
* Full-task prose dumps are forbidden on the provider surface.
* Every successful preparation and invocation logs the residual packet CID.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Final

from ..planning.residual_llm_packet import (
    DEFAULT_MAX_PACKET_BYTES,
    DEFAULT_MAX_PACKET_TOKENS,
    ResidualLlmPacket,
    ResidualLlmPacketBudgetError,
    ResidualLlmPacketError,
    packet_satisfies_residual_llm_contract,
)
from ..proof.formal_verification_contracts import content_identity


# ---------------------------------------------------------------------------
# Interface / evidence
# ---------------------------------------------------------------------------

RESIDUAL_PROVIDER_INVOCATION_INTERFACE: Final[str] = "ResidualProviderInvocation@1"
RESIDUAL_PROVIDER_INVOCATION_VERSION: Final[int] = 1
RESIDUAL_PROVIDER_INVOCATION_EVIDENCE: Final[str] = (
    "wpd/residual-provider-invocation@1"
)
RESIDUAL_PROVIDER_INVOCATION_PRODUCER: Final[str] = "residual-provider-invocation@1"

RESIDUAL_PROVIDER_CONTEXT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/residual-provider-context@1"
)
RESIDUAL_PROVIDER_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/residual-provider-invocation-receipt@1"
)
RESIDUAL_PROVIDER_INVOCATION_DISCOVERY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/residual-provider-invocation/discovery@1"
)

# Prompt / context bounds (aligned with ResidualLlmPacket defaults; not larger).
DEFAULT_MAX_PROMPT_BYTES: Final[int] = DEFAULT_MAX_PACKET_BYTES
DEFAULT_MAX_PROMPT_TOKENS: Final[int] = DEFAULT_MAX_PACKET_TOKENS
DEFAULT_MAX_PATH_LEASE: Final[int] = 256
BYTES_PER_TOKEN: Final[int] = 3

EVENT_RESIDUAL_PROVIDER_PREPARED: Final[str] = "residual_provider_prepared"
EVENT_RESIDUAL_PROVIDER_INVOKED: Final[str] = "residual_provider_invoked"
EVENT_RESIDUAL_PROVIDER_REJECTED: Final[str] = "residual_provider_rejected"

# Environment keys always retained for a hermetic residual provider process.
_SAFE_ENV_EXACT: Final[frozenset[str]] = frozenset(
    {
        "COLORTERM",
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "LOGNAME",
        "NO_COLOR",
        "PATH",
        "PWD",
        "TERM",
        "TMPDIR",
        "TZ",
        "USER",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_RUNTIME_DIR",
        "XDG_STATE_HOME",
    }
)

_SAFE_ENV_PREFIXES: Final[tuple[str, ...]] = (
    "LC_",
    "IPFS_ACCELERATE_AGENT_TASK_",
    "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_",
)

# Secret-shaped markers for environment keys (fail-closed exclusion).
_SECRET_ENV_MARKERS: Final[tuple[str, ...]] = (
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "authorization",
    "auth",
    "credential",
    "private_key",
    "access_key",
    "session_key",
    "cookie",
    "bearer",
    "ssh_key",
    "privatekey",
)

_SECRET_ENV_KEY_RE = re.compile(
    r"(?:^|[_\-.])(?:password|passwd|secret|api[_-]?key|access[_-]?token|"
    r"refresh[_-]?token|session[_-]?token|credential|authorization|cookie|"
    r"private[_-]?key|bearer|ssh[_-]?key)(?:$|[_\-.])",
    re.IGNORECASE,
)

# Keys that must never appear as free-form task dumps on the provider surface.
_FORBIDDEN_TASK_DUMP_KEYS: Final[frozenset[str]] = frozenset(
    {
        "full_task",
        "full_task_body",
        "full_task_dump",
        "full_task_prose",
        "prompt_body",
        "prompt_text",
        "raw_prompt",
        "task_body",
        "task_dump",
        "task_prose",
        "task_text",
        "unbounded_context",
        "repository_dump",
        "repository_corpus",
        "source_body",
        "source_code",
        "source_text",
        "file_contents",
        "file_text",
    }
)

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reason codes
# ---------------------------------------------------------------------------


class ResidualProviderReason(str, Enum):
    """Stable, machine-readable residual provider rejection reasons."""

    MALFORMED = "malformed"
    PACKET_REQUIRED = "residual_packet_required"
    PACKET_CONTRACT_FAILED = "residual_packet_contract_failed"
    OVERSIZED_PACKET = "oversized_packet"
    OVERSIZED_PROMPT = "oversized_prompt"
    PATH_ESCAPE = "path_escape"
    PATH_LEASE_MISMATCH = "path_lease_mismatch"
    FULL_TASK_DUMP_FORBIDDEN = "full_task_dump_forbidden"
    SECRET_IN_ENV = "secret_in_provider_env"
    PROVIDER_NOT_CONFIGURED = "provider_not_configured"
    DISPOSITION_NOT_AUTHORIZED = "disposition_not_authorized"
    PREPARED = "residual_provider_prepared"
    INVOKED = "residual_provider_invoked"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ResidualProviderInvocationError(RuntimeError):
    """Fail-closed rejection for an unsafe residual provider hand-off."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: ResidualProviderReason | str = ResidualProviderReason.MALFORMED,
    ) -> None:
        super().__init__(message)
        if isinstance(reason_code, ResidualProviderReason):
            self.reason_code = reason_code.value
        else:
            self.reason_code = str(reason_code or ResidualProviderReason.MALFORMED.value)


class ResidualProviderPathError(ResidualProviderInvocationError, ValueError):
    """Write path escapes the lease or is otherwise unsafe."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: ResidualProviderReason | str = ResidualProviderReason.PATH_ESCAPE,
    ) -> None:
        super().__init__(message, reason_code=reason_code)


class ResidualProviderBudgetError(ResidualProviderInvocationError):
    """Sealed packet or prompt exceeds configured size bounds."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: ResidualProviderReason | str = ResidualProviderReason.OVERSIZED_PACKET,
    ) -> None:
        super().__init__(message, reason_code=reason_code)


class ResidualProviderEnvError(ResidualProviderInvocationError):
    """Provider environment retained secret-shaped material."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: ResidualProviderReason | str = ResidualProviderReason.SECRET_IN_ENV,
    ) -> None:
        super().__init__(message, reason_code=reason_code)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _estimate_tokens(byte_count: int) -> int:
    if byte_count <= 0:
        return 0
    return (int(byte_count) + BYTES_PER_TOKEN - 1) // BYTES_PER_TOKEN


def _normalize_env_key(key: str) -> str:
    return str(key).strip().lower().replace("-", "_")


def is_secret_env_key(key: str) -> bool:
    """Return True when an environment key looks secret-shaped."""

    lowered = _normalize_env_key(key)
    if not lowered:
        return False
    if _SECRET_ENV_KEY_RE.search(lowered):
        return True
    return any(marker in lowered for marker in _SECRET_ENV_MARKERS)


def _is_safe_env_key(key: str) -> bool:
    name = str(key)
    if name in _SAFE_ENV_EXACT:
        return True
    if any(name.startswith(prefix) for prefix in _SAFE_ENV_PREFIXES):
        # Still reject secret-shaped agent keys.
        return not is_secret_env_key(name)
    return False


def _exact_path(value: Any, name: str = "path") -> str:
    if not isinstance(value, str):
        raise ResidualProviderPathError(
            f"{name} must be a string path",
            reason_code=ResidualProviderReason.PATH_ESCAPE,
        )
    raw = value.strip().replace("\\", "/")
    if not raw:
        raise ResidualProviderPathError(
            f"{name} must not be empty",
            reason_code=ResidualProviderReason.PATH_ESCAPE,
        )
    candidate = PurePosixPath(raw)
    if (
        candidate.is_absolute()
        or ".." in candidate.parts
        or raw in {".", ""}
        or raw.startswith("./")
        or any(char in raw for char in "*?[]{}")
        or "//" in raw
        or raw.endswith("/")
    ):
        raise ResidualProviderPathError(
            f"{name} escapes path lease or is not exact: {value!r}",
            reason_code=ResidualProviderReason.PATH_ESCAPE,
        )
    normalized = candidate.as_posix()
    if normalized != raw:
        raise ResidualProviderPathError(
            f"{name} must be a normalized repository-relative path: {value!r}",
            reason_code=ResidualProviderReason.PATH_ESCAPE,
        )
    return normalized


def _reject_full_task_dump(payload: Any, *, where: str) -> None:
    if isinstance(payload, Mapping):
        for key, item in payload.items():
            norm = _normalize_env_key(str(key))
            if norm in _FORBIDDEN_TASK_DUMP_KEYS:
                raise ResidualProviderInvocationError(
                    f"{where} contains forbidden full-task dump key {key!r}",
                    reason_code=ResidualProviderReason.FULL_TASK_DUMP_FORBIDDEN,
                )
            _reject_full_task_dump(item, where=where)
    elif isinstance(payload, Sequence) and not isinstance(
        payload, (str, bytes, bytearray, memoryview)
    ):
        for item in payload:
            _reject_full_task_dump(item, where=where)


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


# ---------------------------------------------------------------------------
# Path lease
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PathLease:
    """Bounded set of repository-relative write paths the provider may touch."""

    permitted_write_paths: tuple[str, ...]
    lease_id: str = ""

    def __post_init__(self) -> None:
        paths: list[str] = []
        seen: set[str] = set()
        for item in self.permitted_write_paths or ():
            path = _exact_path(item, "permitted_write_paths")
            if path not in seen:
                seen.add(path)
                paths.append(path)
        if not paths:
            raise ResidualProviderPathError(
                "path lease requires at least one permitted write path",
                reason_code=ResidualProviderReason.PATH_LEASE_MISMATCH,
            )
        if len(paths) > DEFAULT_MAX_PATH_LEASE:
            raise ResidualProviderPathError(
                "path lease exceeds maximum permitted write paths",
                reason_code=ResidualProviderReason.PATH_LEASE_MISMATCH,
            )
        object.__setattr__(self, "permitted_write_paths", tuple(paths))
        object.__setattr__(self, "lease_id", str(self.lease_id or "").strip())

    def contains(self, path: str) -> bool:
        return _exact_path(path, "path") in self.permitted_write_paths

    def assert_covers(self, write_paths: Sequence[str]) -> None:
        for path in write_paths:
            exact = _exact_path(path, "write_paths")
            if exact not in self.permitted_write_paths:
                raise ResidualProviderPathError(
                    f"write path {exact!r} is outside the path lease",
                    reason_code=ResidualProviderReason.PATH_LEASE_MISMATCH,
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "lease_id": self.lease_id,
            "permitted_write_paths": list(self.permitted_write_paths),
        }


# ---------------------------------------------------------------------------
# Sealed context + receipt
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SealedProviderContext:
    """Provider-facing argv/context built exclusively from sealed residual fields.

    The prompt body is the canonical sealed packet surface (no free-form task
    prose). Environment is secret-free. Packet CID is always present.
    """

    packet_cid: str
    task_id: str
    repository_id: str
    tree_id: str
    write_paths: tuple[str, ...]
    obligation_ids: tuple[str, ...]
    validation_commands: tuple[str, ...]
    counterexample_capsule: Mapping[str, Any]
    acceptance_ids: tuple[str, ...]
    authority_roots: Mapping[str, str]
    policy_id: str
    policy_revision: str
    forest_id: str
    prompt_body: str
    prompt_bytes: int
    prompt_tokens: int
    environment: Mapping[str, str]
    argv_bindings: Mapping[str, str]
    path_lease_id: str = ""
    nomination_only: bool = True
    semantic_authority: bool = False
    write_authority: bool = False
    completion_authority: bool = False
    producer_id: str = RESIDUAL_PROVIDER_INVOCATION_PRODUCER

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "environment",
            MappingProxyType({str(k): str(v) for k, v in dict(self.environment).items()}),
        )
        object.__setattr__(
            self,
            "argv_bindings",
            MappingProxyType(
                {str(k): str(v) for k, v in dict(self.argv_bindings).items()}
            ),
        )
        object.__setattr__(
            self,
            "counterexample_capsule",
            MappingProxyType(dict(self.counterexample_capsule)),
        )
        object.__setattr__(
            self,
            "authority_roots",
            MappingProxyType({str(k): str(v) for k, v in dict(self.authority_roots).items()}),
        )
        # Hard-zero authority on the sealed surface.
        object.__setattr__(self, "nomination_only", True)
        object.__setattr__(self, "semantic_authority", False)
        object.__setattr__(self, "write_authority", False)
        object.__setattr__(self, "completion_authority", False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": RESIDUAL_PROVIDER_CONTEXT_SCHEMA,
            "interface": RESIDUAL_PROVIDER_INVOCATION_INTERFACE,
            "evidence": RESIDUAL_PROVIDER_INVOCATION_EVIDENCE,
            "packet_cid": self.packet_cid,
            "task_id": self.task_id,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "write_paths": list(self.write_paths),
            "obligation_ids": list(self.obligation_ids),
            "validation_commands": list(self.validation_commands),
            "counterexample_capsule": dict(self.counterexample_capsule),
            "acceptance_ids": list(self.acceptance_ids),
            "authority_roots": dict(self.authority_roots),
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
            "forest_id": self.forest_id,
            "prompt_bytes": self.prompt_bytes,
            "prompt_tokens": self.prompt_tokens,
            "environment_keys": sorted(self.environment.keys()),
            "argv_bindings": dict(self.argv_bindings),
            "path_lease_id": self.path_lease_id,
            "nomination_only": True,
            "semantic_authority": False,
            "write_authority": False,
            "completion_authority": False,
            "producer_id": self.producer_id,
            "contains_full_task_dump": False,
            "contains_secrets_in_env": False,
        }

    @property
    def content_id(self) -> str:
        # Exclude raw prompt_body / environment values from identity so
        # diagnostics remain body-free while still binding the sealed fields.
        return content_identity(self.to_dict())


@dataclass(frozen=True)
class ResidualProviderInvocationReceipt:
    """Body-free receipt of a residual provider prepare/invoke step."""

    packet_cid: str
    prepared: bool
    invoked: bool
    provider_hook_count: int
    reason_code: str
    prompt_bytes: int
    prompt_tokens: int
    environment_keys: tuple[str, ...]
    write_paths: tuple[str, ...]
    path_lease_id: str = ""
    context_cid: str = ""
    provider_result_present: bool = False
    producer_id: str = RESIDUAL_PROVIDER_INVOCATION_PRODUCER
    log_records: tuple[str, ...] = ()

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": RESIDUAL_PROVIDER_RECEIPT_SCHEMA,
            "interface": RESIDUAL_PROVIDER_INVOCATION_INTERFACE,
            "evidence": RESIDUAL_PROVIDER_INVOCATION_EVIDENCE,
            "contract_version": RESIDUAL_PROVIDER_INVOCATION_VERSION,
            "packet_cid": self.packet_cid,
            "prepared": self.prepared,
            "invoked": self.invoked,
            "provider_hook_count": self.provider_hook_count,
            "reason_code": self.reason_code,
            "prompt_bytes": self.prompt_bytes,
            "prompt_tokens": self.prompt_tokens,
            "environment_keys": list(self.environment_keys),
            "write_paths": list(self.write_paths),
            "path_lease_id": self.path_lease_id,
            "context_cid": self.context_cid,
            "provider_result_present": self.provider_result_present,
            "producer_id": self.producer_id,
            "log_records": list(self.log_records),
        }

    def to_event_payload(self, *, task_id: str = "", attempt: int = 1) -> dict[str, Any]:
        event = (
            EVENT_RESIDUAL_PROVIDER_INVOKED
            if self.invoked
            else (
                EVENT_RESIDUAL_PROVIDER_PREPARED
                if self.prepared
                else EVENT_RESIDUAL_PROVIDER_REJECTED
            )
        )
        return {
            "event": event,
            "task_id": task_id,
            "attempt": int(attempt),
            "packet_cid": self.packet_cid,
            "prepared": self.prepared,
            "invoked": self.invoked,
            "provider_hook_count": self.provider_hook_count,
            "reason_code": self.reason_code,
            "prompt_bytes": self.prompt_bytes,
            "prompt_tokens": self.prompt_tokens,
            "path_lease_id": self.path_lease_id,
            "context_cid": self.context_cid,
            "interface": RESIDUAL_PROVIDER_INVOCATION_INTERFACE,
        }


# ---------------------------------------------------------------------------
# Environment sanitization
# ---------------------------------------------------------------------------


def build_provider_environment(
    source: Mapping[str, str] | None = None,
    *,
    extra: Mapping[str, str] | None = None,
    packet_cid: str = "",
    task_id: str = "",
    attempt: int = 1,
) -> dict[str, str]:
    """Build a provider process environment that excludes secret-shaped keys.

    Only safe process keys and residual-invocation metadata are retained.
    Secret markers (``API_KEY``, ``TOKEN``, ``PASSWORD``, …) are never copied.
    """

    inherited = os.environ if source is None else source
    environment: dict[str, str] = {}
    for key, value in inherited.items():
        name = str(key)
        if is_secret_env_key(name):
            continue
        if not _is_safe_env_key(name):
            continue
        environment[name] = str(value)

    if extra:
        for key, value in extra.items():
            name = str(key)
            if is_secret_env_key(name):
                raise ResidualProviderEnvError(
                    f"provider env extra must not include secret key {name!r}",
                    reason_code=ResidualProviderReason.SECRET_IN_ENV,
                )
            environment[name] = str(value)

    if packet_cid:
        environment["IPFS_ACCELERATE_AGENT_RESIDUAL_PACKET_CID"] = str(packet_cid)
    if task_id:
        environment["IPFS_ACCELERATE_AGENT_TASK_ID"] = str(task_id)
    environment["IPFS_ACCELERATE_AGENT_IMPLEMENTATION_ATTEMPT"] = str(int(attempt))

    # Final fail-closed scan: no secret keys may remain.
    for key in environment:
        if is_secret_env_key(key):
            raise ResidualProviderEnvError(
                f"provider env retained secret key {key!r}",
                reason_code=ResidualProviderReason.SECRET_IN_ENV,
            )
    return environment


def assert_provider_env_excludes_secrets(environment: Mapping[str, str]) -> None:
    """Fail closed when any secret-shaped key is present in *environment*."""

    for key in environment:
        if is_secret_env_key(str(key)):
            raise ResidualProviderEnvError(
                f"provider env contains secret key {key!r}",
                reason_code=ResidualProviderReason.SECRET_IN_ENV,
            )


# ---------------------------------------------------------------------------
# Core invocation wrapper
# ---------------------------------------------------------------------------


@dataclass
class ResidualProviderInvocation:
    """Wrap existing provider execution under a sealed residual packet.

    Conflict policy: does **not** add providers. Callers supply
    ``provider_invoke`` (the same callable the daemon already uses).
    """

    max_prompt_bytes: int = DEFAULT_MAX_PROMPT_BYTES
    max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS
    require_path_lease: bool = False
    log: logging.Logger = field(default=_logger)

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_prompt_bytes, bool)
            or not isinstance(self.max_prompt_bytes, int)
            or self.max_prompt_bytes < 256
        ):
            raise ResidualProviderInvocationError(
                "max_prompt_bytes must be an integer >= 256",
                reason_code=ResidualProviderReason.MALFORMED,
            )
        if (
            isinstance(self.max_prompt_tokens, bool)
            or not isinstance(self.max_prompt_tokens, int)
            or self.max_prompt_tokens < 64
        ):
            raise ResidualProviderInvocationError(
                "max_prompt_tokens must be an integer >= 64",
                reason_code=ResidualProviderReason.MALFORMED,
            )

    @classmethod
    def discovery(cls) -> dict[str, Any]:
        return {
            "schema": RESIDUAL_PROVIDER_INVOCATION_DISCOVERY_SCHEMA,
            "interface": RESIDUAL_PROVIDER_INVOCATION_INTERFACE,
            "version": RESIDUAL_PROVIDER_INVOCATION_VERSION,
            "evidence_key": RESIDUAL_PROVIDER_INVOCATION_EVIDENCE,
            "adds_providers": False,
            "wraps_existing_provider_execution": True,
            "sealed_fields_only": True,
            "full_task_dump_forbidden": True,
            "secrets_excluded_from_env": True,
            "packet_cid_logged": True,
            "oversized_packets_rejected": True,
            "path_escaping_packets_rejected": True,
            "max_prompt_bytes": DEFAULT_MAX_PROMPT_BYTES,
            "max_prompt_tokens": DEFAULT_MAX_PROMPT_TOKENS,
        }

    def coerce_packet(
        self, packet: ResidualLlmPacket | Mapping[str, Any]
    ) -> ResidualLlmPacket:
        """Admit a sealed residual packet or fail closed."""

        if packet is None:
            raise ResidualProviderInvocationError(
                "residual packet is required for provider invocation",
                reason_code=ResidualProviderReason.PACKET_REQUIRED,
            )
        if isinstance(packet, ResidualLlmPacket):
            sealed = packet
        elif isinstance(packet, Mapping):
            try:
                sealed = ResidualLlmPacket.from_dict(packet)
            except ResidualLlmPacketBudgetError as exc:
                raise ResidualProviderBudgetError(
                    f"oversized residual packet: {exc}",
                    reason_code=ResidualProviderReason.OVERSIZED_PACKET,
                ) from exc
            except ResidualLlmPacketError as exc:
                reason = str(getattr(exc, "reason_code", "") or "")
                if reason in {
                    "path_not_exact",
                    ResidualProviderReason.PATH_ESCAPE.value,
                }:
                    raise ResidualProviderPathError(
                        f"path-escaping residual packet: {exc}",
                        reason_code=ResidualProviderReason.PATH_ESCAPE,
                    ) from exc
                if reason == "over_budget":
                    raise ResidualProviderBudgetError(
                        f"oversized residual packet: {exc}",
                        reason_code=ResidualProviderReason.OVERSIZED_PACKET,
                    ) from exc
                raise ResidualProviderInvocationError(
                    f"residual packet contract failed: {exc}",
                    reason_code=ResidualProviderReason.PACKET_CONTRACT_FAILED,
                ) from exc
        else:
            raise ResidualProviderInvocationError(
                "residual packet must be ResidualLlmPacket or mapping",
                reason_code=ResidualProviderReason.MALFORMED,
            )

        if not packet_satisfies_residual_llm_contract(sealed):
            raise ResidualProviderInvocationError(
                "residual packet failed ResidualLlmPacket@1 contract",
                reason_code=ResidualProviderReason.PACKET_CONTRACT_FAILED,
            )
        return sealed

    def assert_write_paths_safe(
        self,
        packet: ResidualLlmPacket,
        *,
        path_lease: PathLease | Mapping[str, Any] | Sequence[str] | None = None,
    ) -> PathLease | None:
        """Reject path-escaping write paths and enforce an optional path lease."""

        # ResidualLlmPacket already rejects absolute / ``..`` paths; re-check
        # here so this module fails closed even if a forged mapping slips past.
        for path in packet.write_paths:
            _exact_path(path, "write_paths")

        lease = self._coerce_path_lease(path_lease)
        if lease is not None:
            lease.assert_covers(packet.write_paths)
        elif self.require_path_lease:
            raise ResidualProviderPathError(
                "path lease is required for residual provider invocation",
                reason_code=ResidualProviderReason.PATH_LEASE_MISMATCH,
            )
        return lease

    def build_sealed_prompt(self, packet: ResidualLlmPacket) -> tuple[str, int, int]:
        """Serialize sealed residual fields only; reject oversized prompts."""

        surface = {
            "schema": RESIDUAL_PROVIDER_CONTEXT_SCHEMA,
            "interface": RESIDUAL_PROVIDER_INVOCATION_INTERFACE,
            "stage": "residual_llm_authorized",
            "packet_cid": packet.packet_id or packet.content_id,
            "task_id": packet.task_id,
            "repository_id": packet.repository_id,
            "tree_id": packet.tree_id,
            "policy_id": packet.policy_id,
            "policy_revision": packet.policy_revision,
            "forest_id": packet.forest_id,
            "write_paths": list(packet.write_paths),
            "obligation_ids": list(packet.obligation_ids),
            "validation_commands": list(packet.validation_commands),
            "acceptance_ids": list(packet.acceptance_ids),
            "authority_roots": dict(packet.authority_roots or {}),
            "counterexample_capsule": dict(packet.counterexample_capsule),
            "codex_packet_ref": packet.codex_packet_ref,
            "transition_ref": packet.transition_ref,
            "limits": {
                "max_bytes": packet.max_bytes,
                "max_tokens": packet.max_tokens,
                "max_capsule_bytes": packet.max_capsule_bytes,
            },
            "model_constraints": {
                "nomination_only": True,
                "semantic_authority": False,
                "write_authority": False,
                "completion_authority": False,
                "full_task_dump_forbidden": True,
                "sealed_fields_only": True,
            },
            "provider_instructions": [
                "Operate only on the sealed residual fields in this packet.",
                "Do not request or invent full-task prose or repository dumps.",
                "Do not claim write, semantic, or completion authority.",
                f"Residual packet CID: {packet.packet_id or packet.content_id}",
            ],
        }
        _reject_full_task_dump(surface, where="sealed residual prompt")
        encoded = _canonical_json_bytes(surface)
        prompt_bytes = len(encoded)
        prompt_tokens = _estimate_tokens(prompt_bytes)
        if prompt_bytes > self.max_prompt_bytes:
            raise ResidualProviderBudgetError(
                "residual provider prompt exceeds max_prompt_bytes",
                reason_code=ResidualProviderReason.OVERSIZED_PROMPT,
            )
        if prompt_tokens > self.max_prompt_tokens:
            raise ResidualProviderBudgetError(
                "residual provider prompt exceeds max_prompt_tokens",
                reason_code=ResidualProviderReason.OVERSIZED_PROMPT,
            )
        # Packet-level bound still applies (acceptance: oversized packets rejected).
        if packet.byte_size > packet.max_bytes:
            raise ResidualProviderBudgetError(
                "residual packet exceeds its max_bytes bound",
                reason_code=ResidualProviderReason.OVERSIZED_PACKET,
            )
        return encoded.decode("utf-8"), prompt_bytes, prompt_tokens

    def build_argv_bindings(
        self,
        packet: ResidualLlmPacket,
        *,
        base_argv: Sequence[str] | None = None,
    ) -> dict[str, str]:
        """Return argv metadata bindings limited to sealed residual fields."""

        packet_cid = packet.packet_id or packet.content_id
        bindings = {
            "packet_cid": packet_cid,
            "task_id": packet.task_id,
            "repository_id": packet.repository_id,
            "tree_id": packet.tree_id,
            "write_paths": ",".join(packet.write_paths),
            "obligation_ids": ",".join(packet.obligation_ids),
        }
        if base_argv:
            # Record length only — never re-inject free-form argv payload bodies.
            bindings["base_argv_len"] = str(len(tuple(base_argv)))
        _reject_full_task_dump(bindings, where="provider argv bindings")
        return bindings

    def prepare(
        self,
        packet: ResidualLlmPacket | Mapping[str, Any],
        *,
        path_lease: PathLease | Mapping[str, Any] | Sequence[str] | None = None,
        base_env: Mapping[str, str] | None = None,
        extra_env: Mapping[str, str] | None = None,
        base_argv: Sequence[str] | None = None,
        attempt: int = 1,
    ) -> SealedProviderContext:
        """Prepare sealed provider context without invoking any provider."""

        sealed = self.coerce_packet(packet)
        lease = self.assert_write_paths_safe(sealed, path_lease=path_lease)
        prompt_body, prompt_bytes, prompt_tokens = self.build_sealed_prompt(sealed)
        packet_cid = sealed.packet_id or sealed.content_id
        environment = build_provider_environment(
            base_env,
            extra=extra_env,
            packet_cid=packet_cid,
            task_id=sealed.task_id,
            attempt=attempt,
        )
        assert_provider_env_excludes_secrets(environment)
        argv_bindings = self.build_argv_bindings(sealed, base_argv=base_argv)

        context = SealedProviderContext(
            packet_cid=packet_cid,
            task_id=sealed.task_id,
            repository_id=sealed.repository_id,
            tree_id=sealed.tree_id,
            write_paths=tuple(sealed.write_paths),
            obligation_ids=tuple(sealed.obligation_ids),
            validation_commands=tuple(sealed.validation_commands),
            counterexample_capsule=dict(sealed.counterexample_capsule),
            acceptance_ids=tuple(sealed.acceptance_ids),
            authority_roots=dict(sealed.authority_roots or {}),
            policy_id=sealed.policy_id,
            policy_revision=sealed.policy_revision,
            forest_id=sealed.forest_id,
            prompt_body=prompt_body,
            prompt_bytes=prompt_bytes,
            prompt_tokens=prompt_tokens,
            environment=environment,
            argv_bindings=argv_bindings,
            path_lease_id=lease.lease_id if lease is not None else "",
        )
        self._log_packet_cid(packet_cid, stage="prepared")
        return context

    def invoke(
        self,
        packet: ResidualLlmPacket | Mapping[str, Any],
        provider_invoke: Callable[..., Any],
        *,
        path_lease: PathLease | Mapping[str, Any] | Sequence[str] | None = None,
        base_env: Mapping[str, str] | None = None,
        extra_env: Mapping[str, str] | None = None,
        base_argv: Sequence[str] | None = None,
        attempt: int = 1,
        provider_kwargs: Mapping[str, Any] | None = None,
    ) -> tuple[ResidualProviderInvocationReceipt, Any]:
        """Prepare sealed context and invoke the injected existing provider.

        ``provider_invoke`` is called as::

            provider_invoke(
                prompt=context.prompt_body,
                env=dict(context.environment),
                argv_bindings=dict(context.argv_bindings),
                packet_cid=context.packet_cid,
                **provider_kwargs,
            )
        """

        if provider_invoke is None or not callable(provider_invoke):
            raise ResidualProviderInvocationError(
                "provider_invoke callable is required (wrap existing providers only)",
                reason_code=ResidualProviderReason.PROVIDER_NOT_CONFIGURED,
            )
        if provider_kwargs:
            _reject_full_task_dump(
                dict(provider_kwargs), where="provider_kwargs"
            )

        context = self.prepare(
            packet,
            path_lease=path_lease,
            base_env=base_env,
            extra_env=extra_env,
            base_argv=base_argv,
            attempt=attempt,
        )
        log_records = (
            f"residual_packet_cid={context.packet_cid}",
            f"stage=invoked prompt_bytes={context.prompt_bytes}",
        )
        self._log_packet_cid(context.packet_cid, stage="invoked")

        kwargs = dict(provider_kwargs or {})
        result = provider_invoke(
            prompt=context.prompt_body,
            env=dict(context.environment),
            argv_bindings=dict(context.argv_bindings),
            packet_cid=context.packet_cid,
            **kwargs,
        )
        receipt = ResidualProviderInvocationReceipt(
            packet_cid=context.packet_cid,
            prepared=True,
            invoked=True,
            provider_hook_count=1,
            reason_code=ResidualProviderReason.INVOKED.value,
            prompt_bytes=context.prompt_bytes,
            prompt_tokens=context.prompt_tokens,
            environment_keys=tuple(sorted(context.environment.keys())),
            write_paths=context.write_paths,
            path_lease_id=context.path_lease_id,
            context_cid=context.content_id,
            provider_result_present=result is not None,
            log_records=log_records,
        )
        return receipt, result

    def prepare_receipt(
        self,
        context: SealedProviderContext,
    ) -> ResidualProviderInvocationReceipt:
        """Body-free receipt for a prepared (not yet invoked) context."""

        return ResidualProviderInvocationReceipt(
            packet_cid=context.packet_cid,
            prepared=True,
            invoked=False,
            provider_hook_count=0,
            reason_code=ResidualProviderReason.PREPARED.value,
            prompt_bytes=context.prompt_bytes,
            prompt_tokens=context.prompt_tokens,
            environment_keys=tuple(sorted(context.environment.keys())),
            write_paths=context.write_paths,
            path_lease_id=context.path_lease_id,
            context_cid=context.content_id,
            provider_result_present=False,
            log_records=(f"residual_packet_cid={context.packet_cid}",),
        )

    def _coerce_path_lease(
        self,
        path_lease: PathLease | Mapping[str, Any] | Sequence[str] | None,
    ) -> PathLease | None:
        if path_lease is None:
            return None
        if isinstance(path_lease, PathLease):
            return path_lease
        if isinstance(path_lease, Mapping):
            return PathLease(
                permitted_write_paths=tuple(
                    path_lease.get("permitted_write_paths")
                    or path_lease.get("write_paths")
                    or ()
                ),
                lease_id=str(path_lease.get("lease_id") or ""),
            )
        if isinstance(path_lease, Sequence) and not isinstance(
            path_lease, (str, bytes, bytearray)
        ):
            return PathLease(permitted_write_paths=tuple(path_lease))
        raise ResidualProviderPathError(
            "path_lease must be PathLease, mapping, or path sequence",
            reason_code=ResidualProviderReason.PATH_LEASE_MISMATCH,
        )

    def _log_packet_cid(self, packet_cid: str, *, stage: str) -> None:
        # Acceptance: packet CID logged on every residual provider hand-off.
        self.log.info(
            "ResidualProviderInvocation: stage=%s residual_packet_cid=%s",
            stage,
            packet_cid,
        )


def build_residual_provider_invocation(
    *,
    max_prompt_bytes: int = DEFAULT_MAX_PROMPT_BYTES,
    max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS,
    require_path_lease: bool = False,
) -> ResidualProviderInvocation:
    """Construct a residual provider invocation wrapper."""

    return ResidualProviderInvocation(
        max_prompt_bytes=max_prompt_bytes,
        max_prompt_tokens=max_prompt_tokens,
        require_path_lease=require_path_lease,
    )


def prepare_residual_provider_context(
    packet: ResidualLlmPacket | Mapping[str, Any],
    *,
    path_lease: PathLease | Mapping[str, Any] | Sequence[str] | None = None,
    base_env: Mapping[str, str] | None = None,
    extra_env: Mapping[str, str] | None = None,
    base_argv: Sequence[str] | None = None,
    attempt: int = 1,
    max_prompt_bytes: int = DEFAULT_MAX_PROMPT_BYTES,
    max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS,
    require_path_lease: bool = False,
) -> SealedProviderContext:
    """Module-level helper: prepare sealed provider context."""

    invoker = build_residual_provider_invocation(
        max_prompt_bytes=max_prompt_bytes,
        max_prompt_tokens=max_prompt_tokens,
        require_path_lease=require_path_lease,
    )
    return invoker.prepare(
        packet,
        path_lease=path_lease,
        base_env=base_env,
        extra_env=extra_env,
        base_argv=base_argv,
        attempt=attempt,
    )


def invoke_residual_provider(
    packet: ResidualLlmPacket | Mapping[str, Any],
    provider_invoke: Callable[..., Any],
    *,
    path_lease: PathLease | Mapping[str, Any] | Sequence[str] | None = None,
    base_env: Mapping[str, str] | None = None,
    extra_env: Mapping[str, str] | None = None,
    base_argv: Sequence[str] | None = None,
    attempt: int = 1,
    provider_kwargs: Mapping[str, Any] | None = None,
    max_prompt_bytes: int = DEFAULT_MAX_PROMPT_BYTES,
    max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS,
    require_path_lease: bool = False,
) -> tuple[ResidualProviderInvocationReceipt, Any]:
    """Module-level helper: wrap an existing provider under a sealed packet."""

    invoker = build_residual_provider_invocation(
        max_prompt_bytes=max_prompt_bytes,
        max_prompt_tokens=max_prompt_tokens,
        require_path_lease=require_path_lease,
    )
    return invoker.invoke(
        packet,
        provider_invoke,
        path_lease=path_lease,
        base_env=base_env,
        extra_env=extra_env,
        base_argv=base_argv,
        attempt=attempt,
        provider_kwargs=provider_kwargs,
    )


__all__ = [
    "BYTES_PER_TOKEN",
    "DEFAULT_MAX_PATH_LEASE",
    "DEFAULT_MAX_PROMPT_BYTES",
    "DEFAULT_MAX_PROMPT_TOKENS",
    "EVENT_RESIDUAL_PROVIDER_INVOKED",
    "EVENT_RESIDUAL_PROVIDER_PREPARED",
    "EVENT_RESIDUAL_PROVIDER_REJECTED",
    "PathLease",
    "RESIDUAL_PROVIDER_CONTEXT_SCHEMA",
    "RESIDUAL_PROVIDER_INVOCATION_DISCOVERY_SCHEMA",
    "RESIDUAL_PROVIDER_INVOCATION_EVIDENCE",
    "RESIDUAL_PROVIDER_INVOCATION_INTERFACE",
    "RESIDUAL_PROVIDER_INVOCATION_PRODUCER",
    "RESIDUAL_PROVIDER_INVOCATION_VERSION",
    "RESIDUAL_PROVIDER_RECEIPT_SCHEMA",
    "ResidualProviderBudgetError",
    "ResidualProviderEnvError",
    "ResidualProviderInvocation",
    "ResidualProviderInvocationError",
    "ResidualProviderInvocationReceipt",
    "ResidualProviderPathError",
    "ResidualProviderReason",
    "SealedProviderContext",
    "assert_provider_env_excludes_secrets",
    "build_provider_environment",
    "build_residual_provider_invocation",
    "invoke_residual_provider",
    "is_secret_env_key",
    "prepare_residual_provider_context",
]
