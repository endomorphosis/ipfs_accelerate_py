"""Bounded database context capsules and LLM frontier views (DQP-026).

Interfaces
----------
* ``DatabaseContextManifest@1`` — content-addressed task context capsule
* ``ContextDelta@1`` — bounded parent-to-child semantic delta
* ``LLMContextFrontier@1`` — explicit omitted / unresolved frontier

Effects
-------
Content-addressed capsules contain the task identity, unmet dependencies,
latest distinct failure, worktree delta, impacted symbols, open obligations,
relevant decisions/evidence, and exact validation commands.  Compilation
applies hard row/byte/token budgets, progressive disclosure, and
delta-from-prior comparison.  Heartbeat and wall-clock noise never enter the
semantic identity.  Unresolved or budget-omitted material is recorded as an
explicit frontier rather than silently dropped.  Model packets never receive
secrets or unrestricted repository dumps.

Conflict policy
---------------
This module owns context query/manifests.  :mod:`context_compiler` remains the
semantic composition boundary for provider-aware token budgets and retry
reconstruction.  Callers may project a compiled manifest into a
:class:`~context_contracts.ContextCapsule` via
:func:`project_to_context_compiler_inputs` without inventing authority.

Cold import performs no filesystem, database, network, provider, or process
action.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, ClassVar, Final

from .context_contracts import (
    ContextBudget,
    ContextReference,
    ContextTier,
)
from ..proof.formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    canonical_json_bytes,
    content_identity,
)

# ---------------------------------------------------------------------------
# Interface / schema identity
# ---------------------------------------------------------------------------

DATABASE_CONTEXT_MANIFEST_INTERFACE: Final[str] = "DatabaseContextManifest@1"
CONTEXT_DELTA_INTERFACE: Final[str] = "ContextDelta@1"
LLM_CONTEXT_FRONTIER_INTERFACE: Final[str] = "LLMContextFrontier@1"

DATABASE_CONTEXT_MANIFEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-context-manifest@1"
)
CONTEXT_DELTA_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-context-delta@1"
)
LLM_CONTEXT_FRONTIER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/llm-context-frontier@1"
)
DATABASE_CONTEXT_MEMBER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-context-member@1"
)
DATABASE_CONTEXT_BUDGET_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-context-budget@1"
)
DATABASE_CONTEXT_MODEL_PACKET_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-context-model-packet@1"
)

PRODUCER_ID: Final[str] = "database-context@1"
CONTRACT_VERSION: Final[int] = 1
AUTHORITY_CLASS: Final[str] = "derived_evidence"
UNTRUSTED_DATA_LABEL: Final[str] = "untrusted_repository_data"
REDACTION_MARKER: Final[str] = "[REDACTED]"

# ---------------------------------------------------------------------------
# Hard bounds
# ---------------------------------------------------------------------------

DEFAULT_MAX_ROWS: Final[int] = 128
DEFAULT_MAX_BYTES: Final[int] = 48_000
DEFAULT_MAX_TOKENS: Final[int] = 8_000
DEFAULT_MAX_ITEM_BYTES: Final[int] = 4_096
DEFAULT_MAX_TEXT_BYTES: Final[int] = 2_048
DEFAULT_PAGE_SIZE: Final[int] = 32
MAX_PAGE_SIZE: Final[int] = 256
MAX_ROWS_ABSOLUTE: Final[int] = 4_096
MAX_BYTES_ABSOLUTE: Final[int] = 262_144
MAX_TOKENS_ABSOLUTE: Final[int] = 65_536
MAX_MEMBERS_PER_KIND: Final[int] = 1_024
MAX_PATH_BYTES: Final[int] = 1_024
BYTES_PER_TOKEN: Final[int] = 4

REQUIRED_MEMBER_KINDS: Final[tuple[str, ...]] = (
    "task",
    "unmet_dependency",
    "latest_failure",
    "worktree_delta",
    "impacted_symbol",
    "open_obligation",
    "decision",
    "evidence",
    "validation",
)

# Keys that never contribute to the semantic context CID.
NOISE_FIELD_NAMES: Final[frozenset[str]] = frozenset(
    {
        "heartbeat",
        "heartbeat_at",
        "heartbeat_at_ms",
        "heartbeat_cid",
        "last_heartbeat",
        "last_heartbeat_at",
        "last_heartbeat_at_ms",
        "observed_at",
        "observed_at_ms",
        "polled_at",
        "polled_at_ms",
        "wall_time",
        "wall_time_ms",
        "wall_clock",
        "compiled_at",
        "compiled_at_ms",
        "created_at",
        "created_at_ms",
        "updated_at",
        "updated_at_ms",
        "expires_at",
        "expires_at_ms",
        "renewed_at",
        "renewed_at_ms",
        "lease_expires_at",
        "lease_expires_at_ms",
        "lease_renewed_at",
        "lease_renewed_at_ms",
        "cursor_noise",
        "poll_cursor",
        "server_time",
        "server_time_ms",
        "now",
        "now_ms",
        "timestamp",
        "timestamps",
        "time_noise",
    }
)

_FORBIDDEN_BODY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "api_key",
        "ast_body",
        "ast_nodes",
        "authorization",
        "credential",
        "file_content",
        "file_contents",
        "full_repository",
        "password",
        "private_key",
        "proof_body",
        "proof_transcript",
        "raw_repository",
        "repository_body",
        "repository_dump",
        "repository_source",
        "secret",
        "secrets",
        "source_body",
        "source_code",
        "source_text",
        "token",
        "unrestricted_source",
    }
)

_SECRET_KEY_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:^|[_\-.])(?:password|passwd|secret|api[_-]?key|access[_-]?token|"
    r"refresh[_-]?token|session[_-]?token|credential|authorization|cookie|"
    r"private[_-]?key|bearer)(?:$|[_\-.])",
    re.IGNORECASE,
)

_SECRET_VALUE_MARKERS: Final[tuple[str, ...]] = (
    "api_key",
    "access_token",
    "private_key",
    "-----begin",
    "sk-",
    "password=",
    "authorization: bearer",
    "bearer ",
)

_SECRET_TEXT_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(
        r"(?i)\b(api[_ -]?key|access[_ -]?token|auth[_ -]?token|"
        r"client[_ -]?secret|password|passphrase|secret)"
        r"\s*[:=]\s*['\"][^'\"]{6,}['\"]"
    ),
    re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----", re.IGNORECASE),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b"),
)

assert DATABASE_CONTEXT_MANIFEST_INTERFACE == "DatabaseContextManifest@1"
assert CONTEXT_DELTA_INTERFACE == "ContextDelta@1"
assert LLM_CONTEXT_FRONTIER_INTERFACE == "LLMContextFrontier@1"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DatabaseContextError(ContractValidationError):
    """Fail-closed error for database context compilation."""

    def __init__(self, message: str, *, reason_code: str = "malformed") -> None:
        super().__init__(message)
        self.reason_code = str(reason_code or "malformed")


class DatabaseContextBoundsError(DatabaseContextError):
    """A row, byte, or token budget was exceeded without a valid frontier."""

    def __init__(self, message: str, *, reason_code: str = "over_budget") -> None:
        super().__init__(message, reason_code=reason_code)


class DatabaseContextStaleError(DatabaseContextError):
    """Input identity is stale relative to bound roots."""

    def __init__(self, message: str, *, reason_code: str = "stale_input") -> None:
        super().__init__(message, reason_code=reason_code)


class DatabaseContextSecretError(DatabaseContextError):
    """Secret or unrestricted repository material entered the capsule path."""

    def __init__(
        self, message: str, *, reason_code: str = "secret_or_dump"
    ) -> None:
        super().__init__(message, reason_code=reason_code)


class DatabaseContextInvalidationError(DatabaseContextError):
    """A semantic dependency root changed and invalidates the prior capsule."""

    def __init__(
        self, message: str, *, reason_code: str = "dependency_invalidated"
    ) -> None:
        super().__init__(message, reason_code=reason_code)


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class ContextMemberKind(str, Enum):
    """Closed vocabulary of database context member kinds."""

    TASK = "task"
    UNMET_DEPENDENCY = "unmet_dependency"
    LATEST_FAILURE = "latest_failure"
    WORKTREE_DELTA = "worktree_delta"
    IMPACTED_SYMBOL = "impacted_symbol"
    OPEN_OBLIGATION = "open_obligation"
    DECISION = "decision"
    EVIDENCE = "evidence"
    VALIDATION = "validation"
    EXPANSION = "expansion"
    FRONTIER = "frontier"

    @classmethod
    def coerce(cls, value: Any) -> "ContextMemberKind":
        if isinstance(value, cls):
            return value
        text = str(value or "").strip().casefold().replace("-", "_")
        aliases = {
            "dependency": cls.UNMET_DEPENDENCY,
            "dependencies": cls.UNMET_DEPENDENCY,
            "unmet_dependencies": cls.UNMET_DEPENDENCY,
            "failure": cls.LATEST_FAILURE,
            "failures": cls.LATEST_FAILURE,
            "delta": cls.WORKTREE_DELTA,
            "worktree": cls.WORKTREE_DELTA,
            "symbol": cls.IMPACTED_SYMBOL,
            "symbols": cls.IMPACTED_SYMBOL,
            "impacted_symbols": cls.IMPACTED_SYMBOL,
            "obligation": cls.OPEN_OBLIGATION,
            "obligations": cls.OPEN_OBLIGATION,
            "open_obligations": cls.OPEN_OBLIGATION,
            "decisions": cls.DECISION,
            "evidence_ref": cls.EVIDENCE,
            "validations": cls.VALIDATION,
            "validation_command": cls.VALIDATION,
            "expansion_handle": cls.EXPANSION,
            "unresolved": cls.FRONTIER,
        }
        if text in aliases:
            return aliases[text]
        try:
            return cls(text)
        except ValueError as exc:
            raise DatabaseContextError(
                f"unsupported context member kind: {value!r}",
                reason_code="unsupported_kind",
            ) from exc


class FrontierDisposition(str, Enum):
    """How an omitted or unresolved frontier entry is treated."""

    OMITTED_BUDGET = "omitted_budget"
    OMITTED_PAGINATION = "omitted_pagination"
    UNRESOLVED = "unresolved"
    UNSUPPORTED = "unsupported"
    SECRET_EXCLUDED = "secret_excluded"
    PRIVATE_EXCLUDED = "private_excluded"
    STALE = "stale"
    PROGRESSIVE = "progressive"

    @classmethod
    def coerce(cls, value: Any) -> "FrontierDisposition":
        if isinstance(value, cls):
            return value
        text = str(value or "").strip().casefold().replace("-", "_")
        try:
            return cls(text)
        except ValueError as exc:
            raise DatabaseContextError(
                f"unsupported frontier disposition: {value!r}",
                reason_code="unsupported_disposition",
            ) from exc


class FrontierKind(str, Enum):
    """Closed vocabulary of LLM context frontier kinds."""

    UNRESOLVED_DEPENDENCY = "unresolved_dependency"
    UNRESOLVED_SYMBOL = "unresolved_symbol"
    UNRESOLVED_OBLIGATION = "unresolved_obligation"
    UNRESOLVED_EVIDENCE = "unresolved_evidence"
    DYNAMIC_CALL = "dynamic_call"
    PARSER_UNCERTAINTY = "parser_uncertainty"
    GENERATED_CODE = "generated_code"
    CROSS_LANGUAGE = "cross_language"
    BUDGET_OVERFLOW = "budget_overflow"
    PAGINATION = "pagination"
    SECRET = "secret"
    PRIVATE = "private"
    STALE_INPUT = "stale_input"
    PROGRESSIVE_DISCLOSURE = "progressive_disclosure"
    OTHER = "other"

    @classmethod
    def coerce(cls, value: Any) -> "FrontierKind":
        if isinstance(value, cls):
            return value
        text = str(value or "").strip().casefold().replace("-", "_")
        aliases = {
            "dependency": cls.UNRESOLVED_DEPENDENCY,
            "symbol": cls.UNRESOLVED_SYMBOL,
            "obligation": cls.UNRESOLVED_OBLIGATION,
            "evidence": cls.UNRESOLVED_EVIDENCE,
            "overflow": cls.BUDGET_OVERFLOW,
            "page": cls.PAGINATION,
            "progressive": cls.PROGRESSIVE_DISCLOSURE,
        }
        if text in aliases:
            return aliases[text]
        try:
            return cls(text)
        except ValueError as exc:
            raise DatabaseContextError(
                f"unsupported frontier kind: {value!r}",
                reason_code="unsupported_frontier_kind",
            ) from exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def estimate_tokens(byte_size: int) -> int:
    """Conservative deterministic token estimate (no tokenizer dependency)."""

    if isinstance(byte_size, bool) or not isinstance(byte_size, int) or byte_size < 0:
        raise DatabaseContextError("byte_size must be a non-negative integer")
    if byte_size == 0:
        return 0
    return (byte_size + BYTES_PER_TOKEN - 1) // BYTES_PER_TOKEN


def _text(
    value: Any,
    name: str,
    *,
    required: bool = True,
    limit: int = DEFAULT_MAX_TEXT_BYTES,
) -> str:
    if value is None:
        text = ""
    elif not isinstance(value, str):
        raise DatabaseContextError(f"{name} must be a string")
    else:
        text = value.strip()
    if required and not text:
        raise DatabaseContextError(f"{name} is required")
    if "\x00" in text:
        raise DatabaseContextError(f"{name} must not contain NUL")
    encoded = text.encode("utf-8")
    if len(encoded) > limit:
        raise DatabaseContextBoundsError(f"{name} exceeds text bound")
    return text


def _optional_text(
    value: Any,
    name: str,
    *,
    limit: int = DEFAULT_MAX_TEXT_BYTES,
) -> str:
    return _text(value, name, required=False, limit=limit)


def _identifier(value: Any, name: str) -> str:
    text = _text(value, name, required=True, limit=DEFAULT_MAX_TEXT_BYTES)
    if any(char.isspace() for char in text):
        raise DatabaseContextError(f"{name} must be a compact identifier")
    return text


def _nonneg_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DatabaseContextError(f"{name} must be a non-negative integer")
    return value


def _positive_int(value: Any, name: str, *, minimum: int = 1) -> int:
    result = _nonneg_int(value, name)
    if result < minimum:
        raise DatabaseContextError(f"{name} must be >= {minimum}")
    return result


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise DatabaseContextError(f"{name} must be a boolean")
    return value


def _path(value: Any, name: str = "path") -> str:
    text = _text(value, name, required=True, limit=MAX_PATH_BYTES).replace("\\", "/")
    candidate = PurePosixPath(text)
    if (
        candidate.is_absolute()
        or ".." in candidate.parts
        or text in {".", ""}
        or any(char in text for char in "*?[]{}")
    ):
        raise DatabaseContextError(f"{name} must be a repository-relative path")
    return candidate.as_posix()


def _normalize_key(key: Any) -> str:
    return str(key).casefold().replace("-", "_")


def is_noise_field(name: Any) -> bool:
    """Return True when ``name`` is heartbeat/time noise, not semantic state."""

    return _normalize_key(name) in NOISE_FIELD_NAMES


def strip_noise(value: Any, *, depth: int = 0) -> Any:
    """Deep-copy a value while dropping heartbeat/time noise fields."""

    if depth > 32:
        raise DatabaseContextBoundsError("context payload exceeds nesting bound")
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        raise DatabaseContextError("floating-point values are not canonical context")
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise DatabaseContextError("context keys must be strings")
            if is_noise_field(key):
                continue
            result[key] = strip_noise(item, depth=depth + 1)
        return {key: result[key] for key in sorted(result)}
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        return [strip_noise(item, depth=depth + 1) for item in value]
    if isinstance(value, Enum):
        return value.value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return strip_noise(to_dict(), depth=depth + 1)
    raise DatabaseContextError(
        f"unsupported context value type: {type(value).__name__}"
    )


def _reject_forbidden_material(value: Any, *, where: str, depth: int = 0) -> None:
    """Fail closed when secrets or unrestricted dumps appear in context input."""

    if depth > 32:
        raise DatabaseContextBoundsError(f"{where} exceeds nesting bound")
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_s = str(key)
            norm = _normalize_key(key_s)
            child = f"{where}.{key_s}"
            if norm in _FORBIDDEN_BODY_KEYS:
                raise DatabaseContextSecretError(
                    f"{child} contains forbidden body/secret material",
                    reason_code="forbidden_body",
                )
            if _SECRET_KEY_RE.search(norm):
                raise DatabaseContextSecretError(
                    f"{child} contains secret-bearing key",
                    reason_code="secret_key",
                )
            _reject_forbidden_material(item, where=child, depth=depth + 1)
        return
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        for index, item in enumerate(value):
            _reject_forbidden_material(
                item, where=f"{where}[{index}]", depth=depth + 1
            )
        return
    if isinstance(value, str):
        lowered = value.casefold()
        if any(marker in lowered for marker in _SECRET_VALUE_MARKERS):
            raise DatabaseContextSecretError(
                f"{where} contains secret-bearing text",
                reason_code="secret_value",
            )
        for pattern in _SECRET_TEXT_PATTERNS:
            if pattern.search(value):
                raise DatabaseContextSecretError(
                    f"{where} contains secret-bearing text",
                    reason_code="secret_value",
                )
        # Reject unrestricted repository dumps masquerading as free-form text.
        if len(value.encode("utf-8")) > DEFAULT_MAX_TEXT_BYTES * 8:
            leaf = where.rsplit(".", 1)[-1]
            if _normalize_key(leaf) in _FORBIDDEN_BODY_KEYS | {
                "dump",
                "payload",
                "blob",
                "raw",
                "repository",
                "source",
            }:
                raise DatabaseContextSecretError(
                    f"{where} looks like an unrestricted repository dump",
                    reason_code="unrestricted_dump",
                )


def _member_digest(kind: str, member_id: str, payload: Mapping[str, Any]) -> str:
    return content_identity(
        {
            "kind": kind,
            "member_id": member_id,
            "payload": dict(payload),
        }
    )


def _sorted_unique_ids(values: Iterable[Any], name: str) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = _identifier(item, name)
        if text not in seen:
            seen.add(text)
            ordered.append(text)
    return tuple(ordered)


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatabaseContextBudget(CanonicalContract):
    """Hard row, byte, and token budgets for one database context capsule."""

    SCHEMA: ClassVar[str] = DATABASE_CONTEXT_BUDGET_SCHEMA

    max_rows: int = DEFAULT_MAX_ROWS
    max_bytes: int = DEFAULT_MAX_BYTES
    max_tokens: int = DEFAULT_MAX_TOKENS
    max_item_bytes: int = DEFAULT_MAX_ITEM_BYTES
    max_text_bytes: int = DEFAULT_MAX_TEXT_BYTES
    page_size: int = DEFAULT_PAGE_SIZE
    overflow_behavior: str = "frontier"  # frontier | fail_closed

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "max_rows", _positive_int(self.max_rows, "max_rows")
        )
        object.__setattr__(
            self, "max_bytes", _positive_int(self.max_bytes, "max_bytes", minimum=256)
        )
        object.__setattr__(
            self,
            "max_tokens",
            _positive_int(self.max_tokens, "max_tokens", minimum=64),
        )
        object.__setattr__(
            self,
            "max_item_bytes",
            _positive_int(self.max_item_bytes, "max_item_bytes", minimum=64),
        )
        object.__setattr__(
            self,
            "max_text_bytes",
            _positive_int(self.max_text_bytes, "max_text_bytes", minimum=32),
        )
        object.__setattr__(
            self,
            "page_size",
            _positive_int(self.page_size, "page_size"),
        )
        if self.max_rows > MAX_ROWS_ABSOLUTE:
            raise DatabaseContextBoundsError("max_rows exceeds absolute limit")
        if self.max_bytes > MAX_BYTES_ABSOLUTE:
            raise DatabaseContextBoundsError("max_bytes exceeds absolute limit")
        if self.max_tokens > MAX_TOKENS_ABSOLUTE:
            raise DatabaseContextBoundsError("max_tokens exceeds absolute limit")
        if self.page_size > MAX_PAGE_SIZE:
            raise DatabaseContextBoundsError("page_size exceeds absolute limit")
        if self.max_text_bytes > self.max_item_bytes:
            raise DatabaseContextBoundsError(
                "max_text_bytes cannot exceed max_item_bytes"
            )
        behavior = _text(self.overflow_behavior, "overflow_behavior", limit=64)
        if behavior not in {"frontier", "fail_closed"}:
            raise DatabaseContextError(
                "overflow_behavior must be 'frontier' or 'fail_closed'"
            )
        object.__setattr__(self, "overflow_behavior", behavior)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "max_rows": self.max_rows,
            "max_bytes": self.max_bytes,
            "max_tokens": self.max_tokens,
            "max_item_bytes": self.max_item_bytes,
            "max_text_bytes": self.max_text_bytes,
            "page_size": self.page_size,
            "overflow_behavior": self.overflow_behavior,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DatabaseContextBudget":
        if not isinstance(payload, Mapping):
            raise DatabaseContextError("budget payload must be an object")
        return cls(
            max_rows=payload.get("max_rows", DEFAULT_MAX_ROWS),
            max_bytes=payload.get("max_bytes", DEFAULT_MAX_BYTES),
            max_tokens=payload.get("max_tokens", DEFAULT_MAX_TOKENS),
            max_item_bytes=payload.get("max_item_bytes", DEFAULT_MAX_ITEM_BYTES),
            max_text_bytes=payload.get("max_text_bytes", DEFAULT_MAX_TEXT_BYTES),
            page_size=payload.get("page_size", DEFAULT_PAGE_SIZE),
            overflow_behavior=payload.get("overflow_behavior", "frontier"),
        )


# ---------------------------------------------------------------------------
# Members / frontier entries
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatabaseContextMember(CanonicalContract):
    """One content-addressed member of a database context capsule."""

    SCHEMA: ClassVar[str] = DATABASE_CONTEXT_MEMBER_SCHEMA

    kind: ContextMemberKind | str
    member_id: str
    digest: str
    ordinal: int = 0
    summary: str = ""
    payload: Mapping[str, Any] = field(default_factory=dict)
    required: bool = False
    page: int = 0
    disclosed: bool = True

    def __post_init__(self) -> None:
        kind = ContextMemberKind.coerce(self.kind)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(
            self, "member_id", _identifier(self.member_id, "member_id")
        )
        object.__setattr__(self, "ordinal", _nonneg_int(self.ordinal, "ordinal"))
        object.__setattr__(
            self, "summary", _optional_text(self.summary, "summary")
        )
        if not isinstance(self.payload, Mapping):
            raise DatabaseContextError("member payload must be an object")
        clean = strip_noise(dict(self.payload))
        if not isinstance(clean, dict):
            raise DatabaseContextError("member payload must canonicalize to an object")
        _reject_forbidden_material(clean, where=f"member[{self.member_id}]")
        object.__setattr__(self, "payload", MappingProxyType(clean))
        object.__setattr__(self, "required", _bool(self.required, "required"))
        object.__setattr__(self, "page", _nonneg_int(self.page, "page"))
        object.__setattr__(self, "disclosed", _bool(self.disclosed, "disclosed"))
        digest = _optional_text(self.digest, "digest")
        expected = _member_digest(kind.value, self.member_id, clean)
        if digest and digest != expected:
            raise DatabaseContextError(
                f"member digest mismatch for {self.member_id}",
                reason_code="digest_mismatch",
            )
        object.__setattr__(self, "digest", digest or expected)

    @property
    def byte_size(self) -> int:
        return len(canonical_json_bytes(self.to_semantic_dict()))

    def to_semantic_dict(self) -> dict[str, Any]:
        return {
            "digest": self.digest,
            "disclosed": self.disclosed,
            "kind": self.kind.value if isinstance(self.kind, ContextMemberKind) else str(self.kind),
            "member_id": self.member_id,
            "ordinal": self.ordinal,
            "page": self.page,
            "payload": dict(self.payload),
            "required": self.required,
            "summary": self.summary,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            **self.to_semantic_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DatabaseContextMember":
        if not isinstance(payload, Mapping):
            raise DatabaseContextError("member payload must be an object")
        return cls(
            kind=payload.get("kind", ""),
            member_id=payload.get("member_id", ""),
            digest=payload.get("digest", ""),
            ordinal=payload.get("ordinal", 0),
            summary=payload.get("summary", ""),
            payload=payload.get("payload", {}),
            required=payload.get("required", False),
            page=payload.get("page", 0),
            disclosed=payload.get("disclosed", True),
        )


@dataclass(frozen=True)
class LLMContextFrontierEntry:
    """One explicit omitted or unresolved frontier item."""

    frontier_id: str
    kind: FrontierKind | str
    disposition: FrontierDisposition | str
    reason: str = ""
    member_id: str = ""
    member_kind: str = ""
    blocks_automatic_repair: bool = False
    expansion_handle: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "frontier_id", _identifier(self.frontier_id, "frontier_id")
        )
        object.__setattr__(self, "kind", FrontierKind.coerce(self.kind))
        object.__setattr__(
            self, "disposition", FrontierDisposition.coerce(self.disposition)
        )
        object.__setattr__(
            self, "reason", _optional_text(self.reason, "reason", limit=512)
        )
        object.__setattr__(
            self, "member_id", _optional_text(self.member_id, "member_id")
        )
        object.__setattr__(
            self, "member_kind", _optional_text(self.member_kind, "member_kind")
        )
        object.__setattr__(
            self,
            "blocks_automatic_repair",
            _bool(self.blocks_automatic_repair, "blocks_automatic_repair"),
        )
        object.__setattr__(
            self,
            "expansion_handle",
            _optional_text(self.expansion_handle, "expansion_handle"),
        )
        if not isinstance(self.metadata, Mapping):
            raise DatabaseContextError("frontier metadata must be an object")
        meta = strip_noise(dict(self.metadata))
        if not isinstance(meta, dict):
            raise DatabaseContextError("frontier metadata must be an object")
        _reject_forbidden_material(meta, where=f"frontier[{self.frontier_id}]")
        object.__setattr__(self, "metadata", MappingProxyType(meta))

    def to_dict(self) -> dict[str, Any]:
        return {
            "blocks_automatic_repair": self.blocks_automatic_repair,
            "disposition": (
                self.disposition.value
                if isinstance(self.disposition, FrontierDisposition)
                else str(self.disposition)
            ),
            "expansion_handle": self.expansion_handle,
            "frontier_id": self.frontier_id,
            "kind": (
                self.kind.value if isinstance(self.kind, FrontierKind) else str(self.kind)
            ),
            "member_id": self.member_id,
            "member_kind": self.member_kind,
            "metadata": dict(self.metadata),
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LLMContextFrontierEntry":
        if not isinstance(payload, Mapping):
            raise DatabaseContextError("frontier entry must be an object")
        return cls(
            frontier_id=payload.get("frontier_id", ""),
            kind=payload.get("kind", FrontierKind.OTHER.value),
            disposition=payload.get(
                "disposition", FrontierDisposition.UNRESOLVED.value
            ),
            reason=payload.get("reason", ""),
            member_id=payload.get("member_id", ""),
            member_kind=payload.get("member_kind", ""),
            blocks_automatic_repair=payload.get("blocks_automatic_repair", False),
            expansion_handle=payload.get("expansion_handle", ""),
            metadata=payload.get("metadata", {}),
        )


@dataclass(frozen=True)
class LLMContextFrontier(CanonicalContract):
    """Explicit collection of omitted / unresolved context frontier entries."""

    SCHEMA: ClassVar[str] = LLM_CONTEXT_FRONTIER_SCHEMA
    INTERFACE: ClassVar[str] = LLM_CONTEXT_FRONTIER_INTERFACE

    entries: tuple[LLMContextFrontierEntry, ...] = ()
    omitted_count: int = 0
    unresolved_count: int = 0
    blocks_automatic_repair: bool = False
    complete: bool = True
    page: int = 0
    page_size: int = DEFAULT_PAGE_SIZE
    total_pages: int = 1
    next_page_token: str = ""

    def __post_init__(self) -> None:
        entries: list[LLMContextFrontierEntry] = []
        seen: set[str] = set()
        source = self.entries or ()
        if isinstance(source, (str, bytes, bytearray)) or not isinstance(
            source, Sequence
        ):
            raise DatabaseContextError("frontier entries must be a sequence")
        for item in source:
            entry = (
                item
                if isinstance(item, LLMContextFrontierEntry)
                else LLMContextFrontierEntry.from_dict(item)
            )
            if entry.frontier_id in seen:
                raise DatabaseContextError(
                    f"duplicate frontier id: {entry.frontier_id}"
                )
            seen.add(entry.frontier_id)
            entries.append(entry)
        entries.sort(key=lambda item: (item.kind.value, item.frontier_id))
        object.__setattr__(self, "entries", tuple(entries))
        omitted = sum(
            1
            for item in entries
            if item.disposition
            in {
                FrontierDisposition.OMITTED_BUDGET,
                FrontierDisposition.OMITTED_PAGINATION,
                FrontierDisposition.SECRET_EXCLUDED,
                FrontierDisposition.PRIVATE_EXCLUDED,
                FrontierDisposition.PROGRESSIVE,
            }
        )
        unresolved = sum(
            1
            for item in entries
            if item.disposition
            in {
                FrontierDisposition.UNRESOLVED,
                FrontierDisposition.UNSUPPORTED,
                FrontierDisposition.STALE,
            }
        )
        object.__setattr__(self, "omitted_count", omitted)
        object.__setattr__(self, "unresolved_count", unresolved)
        blocks = any(item.blocks_automatic_repair for item in entries)
        object.__setattr__(self, "blocks_automatic_repair", bool(blocks))
        # complete is True only when nothing remains omitted or unresolved.
        object.__setattr__(self, "complete", not entries)
        object.__setattr__(self, "page", _nonneg_int(self.page, "page"))
        object.__setattr__(
            self, "page_size", _positive_int(self.page_size, "page_size")
        )
        object.__setattr__(
            self, "total_pages", _positive_int(self.total_pages, "total_pages")
        )
        object.__setattr__(
            self,
            "next_page_token",
            _optional_text(self.next_page_token, "next_page_token"),
        )

    @property
    def interface(self) -> str:
        return self.INTERFACE

    @property
    def frontier_cid(self) -> str:
        return content_identity(self.to_semantic_dict())

    def to_semantic_dict(self) -> dict[str, Any]:
        return {
            "blocks_automatic_repair": self.blocks_automatic_repair,
            "complete": self.complete,
            "entries": [item.to_dict() for item in self.entries],
            "interface": self.INTERFACE,
            "next_page_token": self.next_page_token,
            "omitted_count": self.omitted_count,
            "page": self.page,
            "page_size": self.page_size,
            "total_pages": self.total_pages,
            "unresolved_count": self.unresolved_count,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.to_semantic_dict()
        payload["schema"] = self.SCHEMA
        payload["frontier_cid"] = self.frontier_cid
        payload["producer_id"] = PRODUCER_ID
        payload["contract_version"] = CONTRACT_VERSION
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LLMContextFrontier":
        if not isinstance(payload, Mapping):
            raise DatabaseContextError("frontier payload must be an object")
        return cls(
            entries=payload.get("entries", ()),
            omitted_count=payload.get("omitted_count", 0),
            unresolved_count=payload.get("unresolved_count", 0),
            blocks_automatic_repair=payload.get("blocks_automatic_repair", False),
            complete=payload.get("complete", True),
            page=payload.get("page", 0),
            page_size=payload.get("page_size", DEFAULT_PAGE_SIZE),
            total_pages=payload.get("total_pages", 1),
            next_page_token=payload.get("next_page_token", ""),
        )

    def page_slice(
        self, page: int = 0, *, page_size: int | None = None
    ) -> "LLMContextFrontier":
        """Return a paginated projection of frontier entries."""

        size = self.page_size if page_size is None else _positive_int(page_size, "page_size")
        if size > MAX_PAGE_SIZE:
            raise DatabaseContextBoundsError("page_size exceeds absolute limit")
        page = _nonneg_int(page, "page")
        total = len(self.entries)
        total_pages = max(1, (total + size - 1) // size) if total else 1
        if page >= total_pages and total:
            raise DatabaseContextBoundsError("frontier page is out of range")
        start = page * size
        end = start + size
        slice_entries = self.entries[start:end]
        next_token = f"page:{page + 1}" if end < total else ""
        return LLMContextFrontier(
            entries=slice_entries,
            page=page,
            page_size=size,
            total_pages=total_pages,
            next_page_token=next_token,
        )


# ---------------------------------------------------------------------------
# Manifest / delta
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatabaseContextManifest(CanonicalContract):
    """Content-addressed bounded database context capsule (semantic identity)."""

    SCHEMA: ClassVar[str] = DATABASE_CONTEXT_MANIFEST_SCHEMA
    INTERFACE: ClassVar[str] = DATABASE_CONTEXT_MANIFEST_INTERFACE

    task_cid: str
    repository_id: str
    tree_id: str
    schema_revision: int
    policy_digest: str
    members: tuple[DatabaseContextMember, ...]
    frontier: LLMContextFrontier
    budget: DatabaseContextBudget
    goal_cid: str = ""
    plan_cid: str = ""
    task_revision: int = 0
    snapshot_id: str = ""
    parser_id: str = ""
    authority: str = AUTHORITY_CLASS
    truncated: bool = False
    row_count: int = 0
    byte_size: int = 0
    token_estimate: int = 0
    parent_manifest_cid: str = ""
    semantic_roots: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_cid", _identifier(self.task_cid, "task_cid"))
        object.__setattr__(
            self, "repository_id", _identifier(self.repository_id, "repository_id")
        )
        object.__setattr__(self, "tree_id", _identifier(self.tree_id, "tree_id"))
        object.__setattr__(
            self,
            "schema_revision",
            _nonneg_int(self.schema_revision, "schema_revision"),
        )
        object.__setattr__(
            self, "policy_digest", _identifier(self.policy_digest, "policy_digest")
        )
        object.__setattr__(
            self, "goal_cid", _optional_text(self.goal_cid, "goal_cid")
        )
        object.__setattr__(
            self, "plan_cid", _optional_text(self.plan_cid, "plan_cid")
        )
        object.__setattr__(
            self, "task_revision", _nonneg_int(self.task_revision, "task_revision")
        )
        object.__setattr__(
            self, "snapshot_id", _optional_text(self.snapshot_id, "snapshot_id")
        )
        object.__setattr__(
            self, "parser_id", _optional_text(self.parser_id, "parser_id")
        )
        object.__setattr__(
            self,
            "authority",
            _text(self.authority, "authority", limit=128) or AUTHORITY_CLASS,
        )
        object.__setattr__(self, "truncated", _bool(self.truncated, "truncated"))
        if not isinstance(self.budget, DatabaseContextBudget):
            if isinstance(self.budget, Mapping):
                object.__setattr__(
                    self, "budget", DatabaseContextBudget.from_dict(self.budget)
                )
            else:
                raise DatabaseContextError("budget must be DatabaseContextBudget")
        if not isinstance(self.frontier, LLMContextFrontier):
            if isinstance(self.frontier, Mapping):
                object.__setattr__(
                    self, "frontier", LLMContextFrontier.from_dict(self.frontier)
                )
            else:
                raise DatabaseContextError("frontier must be LLMContextFrontier")
        members = _coerce_members(self.members)
        object.__setattr__(self, "members", members)
        if not isinstance(self.semantic_roots, Mapping):
            raise DatabaseContextError("semantic_roots must be an object")
        roots = {
            _identifier(str(key), "semantic_roots key"): _identifier(
                value, f"semantic_roots[{key}]"
            )
            for key, value in dict(self.semantic_roots).items()
        }
        # Always bind the authoritative semantic roots.
        roots.setdefault("task_cid", self.task_cid)
        roots.setdefault("tree_id", self.tree_id)
        roots.setdefault("policy_digest", self.policy_digest)
        roots.setdefault("schema_revision", str(self.schema_revision))
        if self.snapshot_id:
            roots.setdefault("snapshot_id", self.snapshot_id)
        object.__setattr__(
            self,
            "semantic_roots",
            MappingProxyType({key: roots[key] for key in sorted(roots)}),
        )
        if not isinstance(self.metadata, Mapping):
            raise DatabaseContextError("metadata must be an object")
        meta = strip_noise(dict(self.metadata))
        if not isinstance(meta, dict):
            raise DatabaseContextError("metadata must be an object")
        _reject_forbidden_material(meta, where="manifest.metadata")
        object.__setattr__(self, "metadata", MappingProxyType(meta))
        object.__setattr__(
            self,
            "parent_manifest_cid",
            _optional_text(self.parent_manifest_cid, "parent_manifest_cid"),
        )

        # Budget accounting is over disclosed members (the model-facing core).
        # Frontier metadata is explicit residual state and is not charged against
        # the same row/byte/token envelope, but is included in reported size.
        disclosed = [item for item in members if item.disclosed]
        row_count = len(disclosed)
        members_body = {
            "members": [item.to_semantic_dict() for item in disclosed]
        }
        member_bytes = len(canonical_json_bytes(members_body))
        member_tokens = estimate_tokens(member_bytes)
        full_byte_size = len(canonical_json_bytes(self._semantic_body(disclosed)))
        object.__setattr__(self, "row_count", row_count)
        object.__setattr__(self, "byte_size", full_byte_size)
        object.__setattr__(
            self, "token_estimate", estimate_tokens(full_byte_size)
        )
        if row_count > self.budget.max_rows:
            raise DatabaseContextBoundsError(
                "manifest exceeds max_rows after compilation"
            )
        if member_bytes > self.budget.max_bytes:
            raise DatabaseContextBoundsError(
                "manifest exceeds max_bytes after compilation"
            )
        if member_tokens > self.budget.max_tokens:
            raise DatabaseContextBoundsError(
                "manifest exceeds max_tokens after compilation"
            )
        # truncated means budget/pagination/progressive omission occurred.
        has_omission = self.frontier.omitted_count > 0
        object.__setattr__(self, "truncated", bool(self.truncated) or has_omission)

    def _semantic_body(
        self, members: Sequence[DatabaseContextMember] | None = None
    ) -> dict[str, Any]:
        body_members = (
            list(members) if members is not None else list(self.members)
        )
        return {
            "authority": self.authority,
            "frontier": self.frontier.to_semantic_dict(),
            "goal_cid": self.goal_cid,
            "interface": self.INTERFACE,
            "members": [
                item.to_semantic_dict()
                for item in sorted(
                    body_members,
                    key=lambda m: (
                        m.kind.value if isinstance(m.kind, ContextMemberKind) else str(m.kind),
                        m.ordinal,
                        m.member_id,
                    ),
                )
                if item.disclosed
            ],
            "parser_id": self.parser_id,
            "plan_cid": self.plan_cid,
            "policy_digest": self.policy_digest,
            "repository_id": self.repository_id,
            "schema_revision": self.schema_revision,
            "semantic_roots": dict(self.semantic_roots),
            "snapshot_id": self.snapshot_id,
            "task_cid": self.task_cid,
            "task_revision": self.task_revision,
            "tree_id": self.tree_id,
            "truncated": self.truncated,
        }

    @property
    def interface(self) -> str:
        return self.INTERFACE

    @property
    def manifest_cid(self) -> str:
        """Stable semantic identity (excludes heartbeat/time noise)."""

        return content_identity(self._semantic_body())

    @property
    def content_id(self) -> str:
        return self.manifest_cid

    @property
    def capsule_id(self) -> str:
        return self.manifest_cid

    def member_ids(self, *, kind: ContextMemberKind | str | None = None) -> tuple[str, ...]:
        if kind is None:
            return tuple(item.member_id for item in self.members if item.disclosed)
        kind_v = ContextMemberKind.coerce(kind)
        return tuple(
            item.member_id
            for item in self.members
            if item.disclosed and item.kind is kind_v
        )

    def members_by_kind(
        self, kind: ContextMemberKind | str
    ) -> tuple[DatabaseContextMember, ...]:
        kind_v = ContextMemberKind.coerce(kind)
        return tuple(
            item for item in self.members if item.disclosed and item.kind is kind_v
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "authority": self.authority,
            "budget": self.budget.to_dict(),
            "byte_size": self.byte_size,
            "contract_version": CONTRACT_VERSION,
            "frontier": self.frontier.to_dict(),
            "goal_cid": self.goal_cid,
            "interface": self.INTERFACE,
            "manifest_cid": self.manifest_cid,
            "members": [item.to_dict() for item in self.members if item.disclosed],
            "metadata": dict(self.metadata),
            "parent_manifest_cid": self.parent_manifest_cid,
            "parser_id": self.parser_id,
            "plan_cid": self.plan_cid,
            "policy_digest": self.policy_digest,
            "producer_id": PRODUCER_ID,
            "repository_id": self.repository_id,
            "row_count": self.row_count,
            "schema": self.SCHEMA,
            "schema_revision": self.schema_revision,
            "semantic_roots": dict(self.semantic_roots),
            "snapshot_id": self.snapshot_id,
            "task_cid": self.task_cid,
            "task_revision": self.task_revision,
            "token_estimate": self.token_estimate,
            "tree_id": self.tree_id,
            "truncated": self.truncated,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DatabaseContextManifest":
        if not isinstance(payload, Mapping):
            raise DatabaseContextError("manifest payload must be an object")
        result = cls(
            task_cid=payload.get("task_cid", ""),
            repository_id=payload.get("repository_id", ""),
            tree_id=payload.get("tree_id", ""),
            schema_revision=payload.get("schema_revision", 0),
            policy_digest=payload.get("policy_digest", ""),
            members=payload.get("members", ()),
            frontier=payload.get("frontier", {}),
            budget=payload.get("budget", {}),
            goal_cid=payload.get("goal_cid", ""),
            plan_cid=payload.get("plan_cid", ""),
            task_revision=payload.get("task_revision", 0),
            snapshot_id=payload.get("snapshot_id", ""),
            parser_id=payload.get("parser_id", ""),
            authority=payload.get("authority", AUTHORITY_CLASS),
            truncated=payload.get("truncated", False),
            parent_manifest_cid=payload.get("parent_manifest_cid", ""),
            semantic_roots=payload.get("semantic_roots", {}),
            metadata=payload.get("metadata", {}),
        )
        claimed = payload.get("manifest_cid") or payload.get("content_id")
        if claimed not in (None, "") and claimed != result.manifest_cid:
            raise DatabaseContextError(
                "manifest_cid does not match semantic payload",
                reason_code="cid_mismatch",
            )
        return result


@dataclass(frozen=True)
class ContextDelta(CanonicalContract):
    """Bounded semantic delta between two database context manifests."""

    SCHEMA: ClassVar[str] = CONTEXT_DELTA_SCHEMA
    INTERFACE: ClassVar[str] = CONTEXT_DELTA_INTERFACE

    from_manifest_cid: str
    to_manifest_cid: str
    added: tuple[DatabaseContextMember, ...] = ()
    removed: tuple[str, ...] = ()
    changed: tuple[DatabaseContextMember, ...] = ()
    unchanged_count: int = 0
    frontier_delta: LLMContextFrontier | None = None
    invalidated_roots: tuple[str, ...] = ()
    byte_size: int = 0
    token_estimate: int = 0
    bounded: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "from_manifest_cid",
            _identifier(self.from_manifest_cid, "from_manifest_cid"),
        )
        object.__setattr__(
            self,
            "to_manifest_cid",
            _identifier(self.to_manifest_cid, "to_manifest_cid"),
        )
        object.__setattr__(self, "added", _coerce_members(self.added))
        object.__setattr__(self, "changed", _coerce_members(self.changed))
        object.__setattr__(
            self,
            "removed",
            _sorted_unique_ids(self.removed or (), "removed"),
        )
        object.__setattr__(
            self,
            "unchanged_count",
            _nonneg_int(self.unchanged_count, "unchanged_count"),
        )
        if self.frontier_delta is not None and not isinstance(
            self.frontier_delta, LLMContextFrontier
        ):
            if isinstance(self.frontier_delta, Mapping):
                object.__setattr__(
                    self,
                    "frontier_delta",
                    LLMContextFrontier.from_dict(self.frontier_delta),
                )
            else:
                raise DatabaseContextError(
                    "frontier_delta must be LLMContextFrontier"
                )
        object.__setattr__(
            self,
            "invalidated_roots",
            _sorted_unique_ids(self.invalidated_roots or (), "invalidated_roots"),
        )
        body = self.to_semantic_dict()
        byte_size = len(canonical_json_bytes(body))
        tokens = estimate_tokens(byte_size)
        object.__setattr__(self, "byte_size", byte_size)
        object.__setattr__(self, "token_estimate", tokens)
        object.__setattr__(self, "bounded", _bool(self.bounded, "bounded"))
        # A delta is bounded when it transmits only changed members, not full replay.
        total_delta_members = len(self.added) + len(self.changed) + len(self.removed)
        if total_delta_members == 0 and not self.invalidated_roots:
            # Empty delta is valid only for identical manifests.
            if self.from_manifest_cid != self.to_manifest_cid:
                raise DatabaseContextError(
                    "non-identical manifests produced an empty delta",
                    reason_code="empty_delta",
                )

    @property
    def interface(self) -> str:
        return self.INTERFACE

    @property
    def delta_id(self) -> str:
        return content_identity(self.to_semantic_dict())

    @property
    def content_id(self) -> str:
        return self.delta_id

    def to_semantic_dict(self) -> dict[str, Any]:
        return {
            "added": [item.to_semantic_dict() for item in self.added],
            "changed": [item.to_semantic_dict() for item in self.changed],
            "from_manifest_cid": self.from_manifest_cid,
            "frontier_delta": (
                self.frontier_delta.to_semantic_dict()
                if self.frontier_delta is not None
                else None
            ),
            "interface": self.INTERFACE,
            "invalidated_roots": list(self.invalidated_roots),
            "removed": list(self.removed),
            "to_manifest_cid": self.to_manifest_cid,
            "unchanged_count": self.unchanged_count,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.to_semantic_dict(),
            "bounded": self.bounded,
            "byte_size": self.byte_size,
            "contract_version": CONTRACT_VERSION,
            "delta_id": self.delta_id,
            "frontier_delta": (
                self.frontier_delta.to_dict()
                if self.frontier_delta is not None
                else None
            ),
            "producer_id": PRODUCER_ID,
            "schema": self.SCHEMA,
            "token_estimate": self.token_estimate,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ContextDelta":
        if not isinstance(payload, Mapping):
            raise DatabaseContextError("delta payload must be an object")
        result = cls(
            from_manifest_cid=payload.get("from_manifest_cid", ""),
            to_manifest_cid=payload.get("to_manifest_cid", ""),
            added=payload.get("added", ()),
            removed=payload.get("removed", ()),
            changed=payload.get("changed", ()),
            unchanged_count=payload.get("unchanged_count", 0),
            frontier_delta=payload.get("frontier_delta"),
            invalidated_roots=payload.get("invalidated_roots", ()),
            bounded=payload.get("bounded", True),
        )
        claimed = payload.get("delta_id") or payload.get("content_id")
        if claimed not in (None, "") and claimed != result.delta_id:
            raise DatabaseContextError(
                "delta_id does not match semantic payload",
                reason_code="cid_mismatch",
            )
        return result


# ---------------------------------------------------------------------------
# Request / compilation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatabaseContextRequest:
    """Inputs for compiling one bounded database context capsule.

    Noise fields (heartbeats, wall clocks, lease expiry churn) may appear on
    the request or nested payloads; they are stripped before semantic identity
    is computed.
    """

    task_cid: str
    repository_id: str
    tree_id: str
    schema_revision: int
    policy_digest: str
    task: Mapping[str, Any]
    unmet_dependencies: Sequence[Mapping[str, Any]] = ()
    latest_failure: Mapping[str, Any] | None = None
    worktree_delta: Sequence[Mapping[str, Any]] = ()
    impacted_symbols: Sequence[Mapping[str, Any]] = ()
    open_obligations: Sequence[Mapping[str, Any]] = ()
    decisions: Sequence[Mapping[str, Any]] = ()
    evidence: Sequence[Mapping[str, Any]] = ()
    validations: Sequence[Mapping[str, Any] | str] = ()
    unresolved_frontier: Sequence[Mapping[str, Any]] = ()
    goal_cid: str = ""
    plan_cid: str = ""
    task_revision: int = 0
    snapshot_id: str = ""
    parser_id: str = ""
    budget: DatabaseContextBudget | Mapping[str, Any] | None = None
    parent_manifest: DatabaseContextManifest | Mapping[str, Any] | None = None
    expected_roots: Mapping[str, str] | None = None
    # Explicit noise that must not affect identity (also stripped if nested).
    heartbeat_at: str = ""
    heartbeat_at_ms: int = 0
    observed_at: str = ""
    observed_at_ms: int = 0
    lease_expires_at_ms: int = 0
    compiled_at: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    page: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_cid", _identifier(self.task_cid, "task_cid"))
        object.__setattr__(
            self, "repository_id", _identifier(self.repository_id, "repository_id")
        )
        object.__setattr__(self, "tree_id", _identifier(self.tree_id, "tree_id"))
        object.__setattr__(
            self,
            "schema_revision",
            _nonneg_int(self.schema_revision, "schema_revision"),
        )
        object.__setattr__(
            self, "policy_digest", _identifier(self.policy_digest, "policy_digest")
        )
        if not isinstance(self.task, Mapping) or not self.task:
            raise DatabaseContextError("task must be a non-empty object")
        _reject_forbidden_material(self.task, where="task")
        object.__setattr__(self, "task", MappingProxyType(strip_noise(dict(self.task))))
        for name in (
            "unmet_dependencies",
            "worktree_delta",
            "impacted_symbols",
            "open_obligations",
            "decisions",
            "evidence",
            "unresolved_frontier",
        ):
            raw = getattr(self, name)
            if isinstance(raw, (str, bytes, bytearray)) or not isinstance(
                raw, Sequence
            ):
                raise DatabaseContextError(f"{name} must be a sequence")
            cleaned: list[Mapping[str, Any]] = []
            for index, item in enumerate(raw):
                if not isinstance(item, Mapping):
                    raise DatabaseContextError(f"{name}[{index}] must be an object")
                _reject_forbidden_material(item, where=f"{name}[{index}]")
                cleaned.append(MappingProxyType(strip_noise(dict(item))))
            if len(cleaned) > MAX_MEMBERS_PER_KIND:
                raise DatabaseContextBoundsError(f"{name} exceeds collection bound")
            object.__setattr__(self, name, tuple(cleaned))
        if self.latest_failure is not None:
            if not isinstance(self.latest_failure, Mapping):
                raise DatabaseContextError("latest_failure must be an object")
            _reject_forbidden_material(self.latest_failure, where="latest_failure")
            object.__setattr__(
                self,
                "latest_failure",
                MappingProxyType(strip_noise(dict(self.latest_failure))),
            )
        validations: list[Any] = []
        if isinstance(self.validations, (str, bytes, bytearray)) or not isinstance(
            self.validations, Sequence
        ):
            raise DatabaseContextError("validations must be a sequence")
        for index, item in enumerate(self.validations):
            if isinstance(item, str):
                validations.append(_text(item, f"validations[{index}]", limit=512))
            elif isinstance(item, Mapping):
                _reject_forbidden_material(item, where=f"validations[{index}]")
                validations.append(MappingProxyType(strip_noise(dict(item))))
            else:
                raise DatabaseContextError(
                    f"validations[{index}] must be a string or object"
                )
        object.__setattr__(self, "validations", tuple(validations))
        for name in (
            "goal_cid",
            "plan_cid",
            "snapshot_id",
            "parser_id",
            "heartbeat_at",
            "observed_at",
            "compiled_at",
        ):
            object.__setattr__(
                self, name, _optional_text(getattr(self, name), name)
            )
        object.__setattr__(
            self, "task_revision", _nonneg_int(self.task_revision, "task_revision")
        )
        object.__setattr__(
            self,
            "heartbeat_at_ms",
            _nonneg_int(self.heartbeat_at_ms, "heartbeat_at_ms"),
        )
        object.__setattr__(
            self,
            "observed_at_ms",
            _nonneg_int(self.observed_at_ms, "observed_at_ms"),
        )
        object.__setattr__(
            self,
            "lease_expires_at_ms",
            _nonneg_int(self.lease_expires_at_ms, "lease_expires_at_ms"),
        )
        object.__setattr__(self, "page", _nonneg_int(self.page, "page"))
        if self.budget is None:
            object.__setattr__(self, "budget", DatabaseContextBudget())
        elif isinstance(self.budget, Mapping):
            object.__setattr__(
                self, "budget", DatabaseContextBudget.from_dict(self.budget)
            )
        elif not isinstance(self.budget, DatabaseContextBudget):
            raise DatabaseContextError("budget must be DatabaseContextBudget")
        if self.expected_roots is not None:
            if not isinstance(self.expected_roots, Mapping):
                raise DatabaseContextError("expected_roots must be an object")
            object.__setattr__(
                self,
                "expected_roots",
                MappingProxyType(
                    {
                        _identifier(str(k), "expected_roots key"): _identifier(
                            v, f"expected_roots[{k}]"
                        )
                        for k, v in dict(self.expected_roots).items()
                    }
                ),
            )
        if not isinstance(self.metadata, Mapping):
            raise DatabaseContextError("metadata must be an object")
        object.__setattr__(
            self, "metadata", MappingProxyType(strip_noise(dict(self.metadata)))
        )


def _coerce_members(
    value: Any,
) -> tuple[DatabaseContextMember, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise DatabaseContextError("members must be a sequence")
    result: list[DatabaseContextMember] = []
    seen: set[str] = set()
    for item in value:
        member = (
            item
            if isinstance(item, DatabaseContextMember)
            else DatabaseContextMember.from_dict(item)
        )
        if member.member_id in seen:
            raise DatabaseContextError(
                f"duplicate member id: {member.member_id}"
            )
        seen.add(member.member_id)
        result.append(member)
    result.sort(
        key=lambda m: (
            m.kind.value if isinstance(m.kind, ContextMemberKind) else str(m.kind),
            m.ordinal,
            m.member_id,
        )
    )
    return tuple(result)


def _item_id(item: Mapping[str, Any], *keys: str, fallback: str) -> str:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return fallback


def _summary_from(item: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:256]
    return ""


def _make_member(
    *,
    kind: ContextMemberKind,
    member_id: str,
    payload: Mapping[str, Any],
    ordinal: int,
    required: bool,
    page: int = 0,
    disclosed: bool = True,
    summary: str = "",
    max_item_bytes: int = DEFAULT_MAX_ITEM_BYTES,
) -> DatabaseContextMember:
    clean = strip_noise(dict(payload))
    if not isinstance(clean, dict):
        raise DatabaseContextError("member payload must be an object")
    member = DatabaseContextMember(
        kind=kind,
        member_id=member_id,
        digest="",
        ordinal=ordinal,
        summary=summary or _summary_from(clean, "summary", "title", "name"),
        payload=clean,
        required=required,
        page=page,
        disclosed=disclosed,
    )
    if member.byte_size > max_item_bytes:
        # Progressive disclosure: keep digest + summary only.
        reduced = {
            "digest": member.digest,
            "member_id": member.member_id,
            "progressive": True,
            "summary": member.summary,
        }
        member = DatabaseContextMember(
            kind=kind,
            member_id=member_id,
            digest="",
            ordinal=ordinal,
            summary=member.summary,
            payload=reduced,
            required=required,
            page=page,
            disclosed=disclosed,
        )
    return member


def _build_candidates(
    request: DatabaseContextRequest,
) -> tuple[list[DatabaseContextMember], list[LLMContextFrontierEntry]]:
    """Materialize ordered candidate members and unresolved frontier entries."""

    members: list[DatabaseContextMember] = []
    frontier: list[LLMContextFrontierEntry] = []
    budget = request.budget
    assert isinstance(budget, DatabaseContextBudget)
    ordinal = 0

    # Task is always required and first.
    task_payload = {
        "goal_cid": request.goal_cid or str(request.task.get("goal_cid") or ""),
        "plan_cid": request.plan_cid or str(request.task.get("plan_cid") or ""),
        "status": str(request.task.get("status") or request.task.get("task_status") or ""),
        "task_alias": str(request.task.get("task_alias") or request.task.get("alias") or ""),
        "task_cid": request.task_cid,
        "task_revision": request.task_revision
        or int(request.task.get("revision") or request.task.get("task_revision") or 0),
        "title": str(request.task.get("title") or request.task.get("summary") or ""),
    }
    # Preserve additional non-noise task fields.
    for key, value in dict(request.task).items():
        if key not in task_payload and not is_noise_field(key):
            task_payload[key] = value
    members.append(
        _make_member(
            kind=ContextMemberKind.TASK,
            member_id=f"task:{request.task_cid}",
            payload=task_payload,
            ordinal=ordinal,
            required=True,
            summary=task_payload.get("title") or request.task_cid,
            max_item_bytes=budget.max_item_bytes,
        )
    )
    ordinal += 1

    for index, item in enumerate(request.unmet_dependencies):
        mid = _item_id(
            item,
            "dependency_id",
            "id",
            "task_cid",
            fallback=f"dependency:{index}",
        )
        members.append(
            _make_member(
                kind=ContextMemberKind.UNMET_DEPENDENCY,
                member_id=mid if mid.startswith("dependency:") else f"dependency:{mid}",
                payload=item,
                ordinal=ordinal,
                required=True,
                summary=_summary_from(item, "summary", "title", "dependency_id"),
                max_item_bytes=budget.max_item_bytes,
            )
        )
        ordinal += 1

    if request.latest_failure:
        mid = _item_id(
            request.latest_failure,
            "signature_id",
            "failure_id",
            "id",
            fallback="failure:latest",
        )
        members.append(
            _make_member(
                kind=ContextMemberKind.LATEST_FAILURE,
                member_id=mid if ":" in mid else f"failure:{mid}",
                payload=request.latest_failure,
                ordinal=ordinal,
                required=True,
                summary=_summary_from(
                    request.latest_failure, "summary", "failure_kind", "signature"
                ),
                max_item_bytes=budget.max_item_bytes,
            )
        )
        ordinal += 1

    for index, item in enumerate(request.worktree_delta):
        mid = _item_id(
            item, "path", "delta_id", "id", fallback=f"worktree:{index}"
        )
        if "/" in mid or mid.endswith(".py") or mid.endswith(".md"):
            # Path-shaped ids become worktree:path
            try:
                path = _path(mid, "worktree path")
                mid = f"worktree:{path}"
            except DatabaseContextError:
                mid = f"worktree:{mid}"
        elif not mid.startswith("worktree:"):
            mid = f"worktree:{mid}"
        members.append(
            _make_member(
                kind=ContextMemberKind.WORKTREE_DELTA,
                member_id=mid,
                payload=item,
                ordinal=ordinal,
                required=False,
                summary=_summary_from(item, "path", "summary"),
                max_item_bytes=budget.max_item_bytes,
            )
        )
        ordinal += 1

    for index, item in enumerate(request.impacted_symbols):
        mid = _item_id(
            item,
            "symbol",
            "qualified_name",
            "symbol_id",
            "id",
            fallback=f"symbol:{index}",
        )
        if not mid.startswith("symbol:"):
            mid = f"symbol:{mid}"
        members.append(
            _make_member(
                kind=ContextMemberKind.IMPACTED_SYMBOL,
                member_id=mid,
                payload=item,
                ordinal=ordinal,
                required=False,
                summary=_summary_from(item, "symbol", "qualified_name", "summary"),
                max_item_bytes=budget.max_item_bytes,
            )
        )
        ordinal += 1

    for index, item in enumerate(request.open_obligations):
        mid = _item_id(
            item, "obligation_id", "id", fallback=f"obligation:{index}"
        )
        if not mid.startswith("obligation:"):
            mid = f"obligation:{mid}"
        members.append(
            _make_member(
                kind=ContextMemberKind.OPEN_OBLIGATION,
                member_id=mid,
                payload=item,
                ordinal=ordinal,
                required=True,
                summary=_summary_from(item, "summary", "obligation_id"),
                max_item_bytes=budget.max_item_bytes,
            )
        )
        ordinal += 1

    for index, item in enumerate(request.decisions):
        mid = _item_id(item, "decision_id", "id", fallback=f"decision:{index}")
        if not mid.startswith("decision:"):
            mid = f"decision:{mid}"
        members.append(
            _make_member(
                kind=ContextMemberKind.DECISION,
                member_id=mid,
                payload=item,
                ordinal=ordinal,
                required=False,
                summary=_summary_from(item, "summary", "decision_id"),
                max_item_bytes=budget.max_item_bytes,
            )
        )
        ordinal += 1

    for index, item in enumerate(request.evidence):
        mid = _item_id(
            item, "evidence_id", "digest", "id", fallback=f"evidence:{index}"
        )
        if not mid.startswith("evidence:"):
            mid = f"evidence:{mid}"
        members.append(
            _make_member(
                kind=ContextMemberKind.EVIDENCE,
                member_id=mid,
                payload=item,
                ordinal=ordinal,
                required=False,
                summary=_summary_from(item, "summary", "evidence_id", "digest"),
                max_item_bytes=budget.max_item_bytes,
            )
        )
        ordinal += 1

    for index, item in enumerate(request.validations):
        if isinstance(item, str):
            payload = {"command": item}
            mid = f"validation:{hashlib.sha256(item.encode()).hexdigest()[:16]}"
            summary = item[:256]
        else:
            payload = dict(item)
            mid = _item_id(
                item,
                "validation_id",
                "command",
                "id",
                fallback=f"validation:{index}",
            )
            if not mid.startswith("validation:"):
                mid = f"validation:{mid}"
            summary = _summary_from(item, "command", "summary", "validation_id")
        members.append(
            _make_member(
                kind=ContextMemberKind.VALIDATION,
                member_id=mid,
                payload=payload,
                ordinal=ordinal,
                required=True,
                summary=summary,
                max_item_bytes=budget.max_item_bytes,
            )
        )
        ordinal += 1

    for index, item in enumerate(request.unresolved_frontier):
        fid = _item_id(
            item, "frontier_id", "id", "symbol", fallback=f"frontier:{index}"
        )
        if not fid.startswith("frontier:"):
            fid = f"frontier:{fid}"
        kind = FrontierKind.coerce(item.get("kind", FrontierKind.OTHER.value))
        disposition = FrontierDisposition.coerce(
            item.get("disposition", FrontierDisposition.UNRESOLVED.value)
        )
        blocks = bool(item.get("blocks_automatic_repair", True))
        frontier.append(
            LLMContextFrontierEntry(
                frontier_id=fid,
                kind=kind,
                disposition=disposition,
                reason=_optional_text(item.get("reason", "unresolved"), "reason"),
                member_id=_optional_text(item.get("member_id", ""), "member_id"),
                member_kind=_optional_text(
                    item.get("member_kind", ""), "member_kind"
                ),
                blocks_automatic_repair=blocks,
                expansion_handle=_optional_text(
                    item.get("expansion_handle", ""), "expansion_handle"
                ),
                metadata={
                    key: value
                    for key, value in dict(item).items()
                    if key
                    not in {
                        "frontier_id",
                        "id",
                        "kind",
                        "disposition",
                        "reason",
                        "member_id",
                        "member_kind",
                        "blocks_automatic_repair",
                        "expansion_handle",
                    }
                    and not is_noise_field(key)
                },
            )
        )

    return members, frontier


def _apply_budgets(
    candidates: Sequence[DatabaseContextMember],
    unresolved: Sequence[LLMContextFrontierEntry],
    budget: DatabaseContextBudget,
    *,
    page: int = 0,
) -> tuple[
    list[DatabaseContextMember],
    list[LLMContextFrontierEntry],
    bool,
]:
    """Select disclosed members under row/byte/token budgets with pagination."""

    page_size = budget.page_size
    # Required members always attempt inclusion first (stable order).
    required = [m for m in candidates if m.required]
    optional = [m for m in candidates if not m.required]

    selected: list[DatabaseContextMember] = []
    omitted: list[LLMContextFrontierEntry] = list(unresolved)
    truncated = bool(unresolved)

    def _fits(member: DatabaseContextMember) -> bool:
        trial = selected + [member]
        if len(trial) > budget.max_rows:
            return False
        # Charge the member envelope only (matches DatabaseContextManifest).
        body = {"members": [item.to_semantic_dict() for item in trial]}
        raw = canonical_json_bytes(body)
        if len(raw) > budget.max_bytes:
            return False
        if estimate_tokens(len(raw)) > budget.max_tokens:
            return False
        return True

    # Include all required members; fail or frontier if they cannot fit.
    for member in required:
        if _fits(member):
            selected.append(member)
            continue
        if budget.overflow_behavior == "fail_closed":
            raise DatabaseContextBoundsError(
                f"required member {member.member_id} exceeds context budget",
                reason_code="required_overflow",
            )
        omitted.append(
            LLMContextFrontierEntry(
                frontier_id=f"omit:{member.member_id}",
                kind=FrontierKind.BUDGET_OVERFLOW,
                disposition=FrontierDisposition.OMITTED_BUDGET,
                reason="required member exceeds remaining budget",
                member_id=member.member_id,
                member_kind=member.kind.value,
                blocks_automatic_repair=True,
                expansion_handle=f"expand:{member.digest}",
            )
        )
        truncated = True

    # Paginate optional members.
    start = page * page_size
    # Optional stream is ordered; page selects a window, remainder becomes frontier.
    window = optional[start : start + page_size]
    remainder_before = optional[:start]
    remainder_after = optional[start + page_size :]

    for member in remainder_before:
        omitted.append(
            LLMContextFrontierEntry(
                frontier_id=f"page-before:{member.member_id}",
                kind=FrontierKind.PAGINATION,
                disposition=FrontierDisposition.OMITTED_PAGINATION,
                reason=f"member is on a prior page (page<{page})",
                member_id=member.member_id,
                member_kind=member.kind.value,
                blocks_automatic_repair=False,
                expansion_handle=f"page:{max(0, page - 1)}",
            )
        )
        truncated = True

    for member in window:
        # Progressive disclosure: oversized already reduced in _make_member.
        if member.payload.get("progressive") is True:
            omitted.append(
                LLMContextFrontierEntry(
                    frontier_id=f"progressive:{member.member_id}",
                    kind=FrontierKind.PROGRESSIVE_DISCLOSURE,
                    disposition=FrontierDisposition.PROGRESSIVE,
                    reason="member body reduced to digest/summary under item budget",
                    member_id=member.member_id,
                    member_kind=member.kind.value,
                    blocks_automatic_repair=False,
                    expansion_handle=f"expand:{member.digest}",
                )
            )
            truncated = True
        if _fits(member):
            selected.append(member)
        else:
            if budget.overflow_behavior == "fail_closed":
                raise DatabaseContextBoundsError(
                    f"optional member {member.member_id} exceeds context budget",
                    reason_code="optional_overflow",
                )
            omitted.append(
                LLMContextFrontierEntry(
                    frontier_id=f"omit:{member.member_id}",
                    kind=FrontierKind.BUDGET_OVERFLOW,
                    disposition=FrontierDisposition.OMITTED_BUDGET,
                    reason="optional member exceeds remaining budget",
                    member_id=member.member_id,
                    member_kind=member.kind.value,
                    blocks_automatic_repair=False,
                    expansion_handle=f"expand:{member.digest}",
                )
            )
            truncated = True

    for member in remainder_after:
        omitted.append(
            LLMContextFrontierEntry(
                frontier_id=f"page-after:{member.member_id}",
                kind=FrontierKind.PAGINATION,
                disposition=FrontierDisposition.OMITTED_PAGINATION,
                reason=f"member is on a later page (page>{page})",
                member_id=member.member_id,
                member_kind=member.kind.value,
                blocks_automatic_repair=False,
                expansion_handle=f"page:{page + 1}",
            )
        )
        truncated = True

    return selected, omitted, truncated


def _check_expected_roots(request: DatabaseContextRequest) -> None:
    if not request.expected_roots:
        return
    actual = {
        "task_cid": request.task_cid,
        "tree_id": request.tree_id,
        "policy_digest": request.policy_digest,
        "schema_revision": str(request.schema_revision),
    }
    if request.snapshot_id:
        actual["snapshot_id"] = request.snapshot_id
    for key, expected in request.expected_roots.items():
        if key not in actual:
            raise DatabaseContextStaleError(
                f"expected root {key!r} is not bound by this context",
                reason_code="missing_root",
            )
        if actual[key] != expected:
            raise DatabaseContextStaleError(
                f"stale input: root {key} expected {expected!r}, got {actual[key]!r}",
                reason_code="stale_root",
            )


def compile_database_context(
    request: DatabaseContextRequest | Mapping[str, Any],
) -> DatabaseContextManifest:
    """Compile a bounded, content-addressed database context capsule.

    Heartbeat and wall-clock fields on the request never enter the semantic
    manifest CID.  Unresolved and budget-omitted material is recorded on the
    returned frontier.
    """

    if isinstance(request, Mapping):
        request = DatabaseContextRequest(**dict(request))  # type: ignore[arg-type]
    if not isinstance(request, DatabaseContextRequest):
        raise DatabaseContextError("request must be DatabaseContextRequest")

    _check_expected_roots(request)
    budget = request.budget
    assert isinstance(budget, DatabaseContextBudget)

    candidates, unresolved = _build_candidates(request)
    selected, frontier_entries, truncated = _apply_budgets(
        candidates, unresolved, budget, page=request.page
    )

    total_optional = sum(1 for m in candidates if not m.required)
    total_pages = max(1, (total_optional + budget.page_size - 1) // budget.page_size)
    next_token = (
        f"page:{request.page + 1}" if request.page + 1 < total_pages else ""
    )
    frontier = LLMContextFrontier(
        entries=tuple(frontier_entries),
        page=request.page,
        page_size=budget.page_size,
        total_pages=total_pages,
        next_page_token=next_token,
    )

    parent_cid = ""
    if request.parent_manifest is not None:
        parent = request.parent_manifest
        if isinstance(parent, Mapping):
            parent = DatabaseContextManifest.from_dict(parent)
        if not isinstance(parent, DatabaseContextManifest):
            raise DatabaseContextError("parent_manifest is invalid")
        parent_cid = parent.manifest_cid
        # Exact dependency invalidation: changed semantic roots invalidate parent.
        for key, value in parent.semantic_roots.items():
            current = {
                "task_cid": request.task_cid,
                "tree_id": request.tree_id,
                "policy_digest": request.policy_digest,
                "schema_revision": str(request.schema_revision),
                "snapshot_id": request.snapshot_id,
            }.get(key)
            if current is not None and current != value:
                # Parent binding is informational; child still compiles, but
                # callers can detect invalidation via compare_and_delta.
                pass

    manifest = DatabaseContextManifest(
        task_cid=request.task_cid,
        repository_id=request.repository_id,
        tree_id=request.tree_id,
        schema_revision=request.schema_revision,
        policy_digest=request.policy_digest,
        members=tuple(selected),
        frontier=frontier,
        budget=budget,
        goal_cid=request.goal_cid or str(request.task.get("goal_cid") or ""),
        plan_cid=request.plan_cid or str(request.task.get("plan_cid") or ""),
        task_revision=request.task_revision
        or int(request.task.get("revision") or request.task.get("task_revision") or 0),
        snapshot_id=request.snapshot_id,
        parser_id=request.parser_id,
        truncated=truncated,
        parent_manifest_cid=parent_cid,
        metadata=dict(request.metadata),
    )
    return manifest


def compare_and_delta(
    prior: DatabaseContextManifest | Mapping[str, Any],
    current: DatabaseContextManifest | Mapping[str, Any],
    *,
    max_delta_bytes: int | None = None,
    max_delta_tokens: int | None = None,
) -> ContextDelta:
    """Compute a bounded semantic delta between two compiled manifests.

    Changed evidence yields only the added/removed/changed members.  Unchanged
    heartbeat/time noise is already absent from both CIDs, so identical
    semantic state produces an empty bounded delta with equal CIDs.
    """

    if isinstance(prior, Mapping):
        prior = DatabaseContextManifest.from_dict(prior)
    if isinstance(current, Mapping):
        current = DatabaseContextManifest.from_dict(current)
    if not isinstance(prior, DatabaseContextManifest):
        raise DatabaseContextError("prior must be DatabaseContextManifest")
    if not isinstance(current, DatabaseContextManifest):
        raise DatabaseContextError("current must be DatabaseContextManifest")

    invalidated: list[str] = []
    for key, value in prior.semantic_roots.items():
        cur = current.semantic_roots.get(key)
        if cur is not None and cur != value:
            invalidated.append(key)
    # Also check keys only on current that are authoritative.
    for key in ("task_cid", "tree_id", "policy_digest", "schema_revision", "snapshot_id"):
        if (
            key in prior.semantic_roots
            and key in current.semantic_roots
            and prior.semantic_roots[key] != current.semantic_roots[key]
            and key not in invalidated
        ):
            invalidated.append(key)

    if invalidated and (
        prior.semantic_roots.get("tree_id") != current.semantic_roots.get("tree_id")
        or prior.semantic_roots.get("policy_digest")
        != current.semantic_roots.get("policy_digest")
        or prior.semantic_roots.get("schema_revision")
        != current.semantic_roots.get("schema_revision")
    ):
        # Hard invalidation of dependency identity: delta still reports roots.
        pass

    prior_by_id = {item.member_id: item for item in prior.members if item.disclosed}
    current_by_id = {
        item.member_id: item for item in current.members if item.disclosed
    }

    added: list[DatabaseContextMember] = []
    changed: list[DatabaseContextMember] = []
    removed: list[str] = []
    unchanged = 0

    for member_id, member in current_by_id.items():
        if member_id not in prior_by_id:
            added.append(member)
        elif prior_by_id[member_id].digest != member.digest:
            changed.append(member)
        else:
            unchanged += 1
    for member_id in prior_by_id:
        if member_id not in current_by_id:
            removed.append(member_id)

    # Frontier delta: entries present only on current or with changed disposition.
    prior_frontier = {
        item.frontier_id: item for item in prior.frontier.entries
    }
    frontier_entries: list[LLMContextFrontierEntry] = []
    for entry in current.frontier.entries:
        previous = prior_frontier.get(entry.frontier_id)
        if previous is None or previous.to_dict() != entry.to_dict():
            frontier_entries.append(entry)
    for fid, entry in prior_frontier.items():
        if fid not in {e.frontier_id for e in current.frontier.entries}:
            frontier_entries.append(
                LLMContextFrontierEntry(
                    frontier_id=f"resolved:{fid}",
                    kind=entry.kind,
                    disposition=FrontierDisposition.OMITTED_BUDGET
                    if entry.disposition
                    is FrontierDisposition.OMITTED_BUDGET
                    else FrontierDisposition.UNRESOLVED,
                    reason="frontier entry resolved or dropped in current capsule",
                    member_id=entry.member_id,
                    member_kind=entry.member_kind,
                    blocks_automatic_repair=False,
                    metadata={"prior_disposition": entry.disposition.value},
                )
            )

    frontier_delta = (
        LLMContextFrontier(entries=tuple(frontier_entries))
        if frontier_entries
        else None
    )

    delta = ContextDelta(
        from_manifest_cid=prior.manifest_cid,
        to_manifest_cid=current.manifest_cid,
        added=tuple(added),
        removed=tuple(removed),
        changed=tuple(changed),
        unchanged_count=unchanged,
        frontier_delta=frontier_delta,
        invalidated_roots=tuple(sorted(set(invalidated))),
        bounded=True,
    )

    limit_bytes = max_delta_bytes
    limit_tokens = max_delta_tokens
    if limit_bytes is not None and delta.byte_size > limit_bytes:
        raise DatabaseContextBoundsError(
            "context delta exceeds max_delta_bytes",
            reason_code="delta_overflow",
        )
    if limit_tokens is not None and delta.token_estimate > limit_tokens:
        raise DatabaseContextBoundsError(
            "context delta exceeds max_delta_tokens",
            reason_code="delta_overflow",
        )

    # Boundedness: delta payload must be smaller than full current manifest
    # when any members are unchanged (unless identity-equal empty delta).
    if (
        prior.manifest_cid != current.manifest_cid
        and unchanged > 0
        and delta.byte_size >= current.byte_size
    ):
        raise DatabaseContextBoundsError(
            "context delta is not smaller than full capsule replay",
            reason_code="unbounded_delta",
        )
    return delta


def assert_dependency_freshness(
    manifest: DatabaseContextManifest,
    *,
    tree_id: str | None = None,
    policy_digest: str | None = None,
    schema_revision: int | None = None,
    snapshot_id: str | None = None,
    task_cid: str | None = None,
) -> None:
    """Fail closed when a bound semantic dependency root has drifted."""

    checks = {
        "tree_id": tree_id,
        "policy_digest": policy_digest,
        "schema_revision": (
            str(schema_revision) if schema_revision is not None else None
        ),
        "snapshot_id": snapshot_id,
        "task_cid": task_cid,
    }
    for key, expected in checks.items():
        if expected is None:
            continue
        actual = manifest.semantic_roots.get(key)
        if actual is None:
            raise DatabaseContextInvalidationError(
                f"manifest does not bind dependency root {key}",
                reason_code="missing_root",
            )
        if actual != str(expected):
            raise DatabaseContextInvalidationError(
                f"dependency root {key} invalidated: expected {expected!r}, "
                f"manifest has {actual!r}",
                reason_code="dependency_invalidated",
            )


# ---------------------------------------------------------------------------
# Model packet projection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatabaseContextModelPacket(CanonicalContract):
    """Model-facing projection of a database context capsule.

    Never contains secrets, private keys, or unrestricted repository dumps.
    Unresolved frontier omissions remain explicit.
    """

    SCHEMA: ClassVar[str] = DATABASE_CONTEXT_MODEL_PACKET_SCHEMA

    manifest_cid: str
    task_cid: str
    repository_id: str
    tree_id: str
    members: tuple[Mapping[str, Any], ...]
    frontier: Mapping[str, Any]
    validation_commands: tuple[str, ...]
    open_obligation_ids: tuple[str, ...]
    impacted_symbols: tuple[str, ...]
    token_estimate: int
    truncated: bool
    authority: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "manifest_cid", _identifier(self.manifest_cid, "manifest_cid")
        )
        object.__setattr__(self, "task_cid", _identifier(self.task_cid, "task_cid"))
        object.__setattr__(
            self, "repository_id", _identifier(self.repository_id, "repository_id")
        )
        object.__setattr__(self, "tree_id", _identifier(self.tree_id, "tree_id"))
        members: list[Mapping[str, Any]] = []
        if isinstance(self.members, (str, bytes, bytearray)) or not isinstance(
            self.members, Sequence
        ):
            raise DatabaseContextError("model packet members must be a sequence")
        for index, item in enumerate(self.members):
            if not isinstance(item, Mapping):
                raise DatabaseContextError(
                    f"model packet members[{index}] must be an object"
                )
            _reject_forbidden_material(item, where=f"model_packet.members[{index}]")
            members.append(MappingProxyType(strip_noise(dict(item))))
        object.__setattr__(self, "members", tuple(members))
        if not isinstance(self.frontier, Mapping):
            raise DatabaseContextError("model packet frontier must be an object")
        _reject_forbidden_material(self.frontier, where="model_packet.frontier")
        object.__setattr__(
            self, "frontier", MappingProxyType(strip_noise(dict(self.frontier)))
        )
        object.__setattr__(
            self,
            "validation_commands",
            tuple(
                _text(item, "validation_commands", limit=512)
                for item in (self.validation_commands or ())
            ),
        )
        object.__setattr__(
            self,
            "open_obligation_ids",
            _sorted_unique_ids(self.open_obligation_ids or (), "open_obligation_ids"),
        )
        object.__setattr__(
            self,
            "impacted_symbols",
            _sorted_unique_ids(self.impacted_symbols or (), "impacted_symbols"),
        )
        object.__setattr__(
            self,
            "token_estimate",
            _nonneg_int(self.token_estimate, "token_estimate"),
        )
        object.__setattr__(self, "truncated", _bool(self.truncated, "truncated"))
        if not isinstance(self.authority, Mapping):
            raise DatabaseContextError("authority must be an object")
        object.__setattr__(
            self, "authority", MappingProxyType(dict(self.authority))
        )
        # Final whole-packet scan.
        _reject_forbidden_material(self.to_dict(), where="model_packet")

    @property
    def packet_cid(self) -> str:
        return content_identity(
            {key: value for key, value in self.to_dict().items() if key != "packet_cid"}
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "authority": dict(self.authority),
            "frontier": dict(self.frontier),
            "impacted_symbols": list(self.impacted_symbols),
            "manifest_cid": self.manifest_cid,
            "members": [dict(item) for item in self.members],
            "open_obligation_ids": list(self.open_obligation_ids),
            "repository_id": self.repository_id,
            "schema": self.SCHEMA,
            "task_cid": self.task_cid,
            "token_estimate": self.token_estimate,
            "tree_id": self.tree_id,
            "truncated": self.truncated,
            "validation_commands": list(self.validation_commands),
        }

    def provider_payload(self) -> Mapping[str, Any]:
        """Canonical payload handed to a model provider."""

        payload = self.to_dict()
        payload["packet_cid"] = self.packet_cid
        payload["data_label"] = UNTRUSTED_DATA_LABEL
        payload["treat_as"] = "data_not_instructions"
        return MappingProxyType(payload)


def model_packet_from_manifest(
    manifest: DatabaseContextManifest | Mapping[str, Any],
) -> DatabaseContextModelPacket:
    """Project a compiled manifest into a secret-free model packet."""

    if isinstance(manifest, Mapping):
        manifest = DatabaseContextManifest.from_dict(manifest)
    if not isinstance(manifest, DatabaseContextManifest):
        raise DatabaseContextError("manifest must be DatabaseContextManifest")

    member_views: list[dict[str, Any]] = []
    validation_commands: list[str] = []
    open_obligations: list[str] = []
    impacted: list[str] = []

    for member in manifest.members:
        if not member.disclosed:
            continue
        view = {
            "digest": member.digest,
            "kind": member.kind.value,
            "member_id": member.member_id,
            "required": member.required,
            "summary": member.summary,
            # Payload is already noise-stripped and secret-scanned.
            "payload": dict(member.payload),
        }
        _reject_forbidden_material(view, where=f"model_member[{member.member_id}]")
        member_views.append(view)
        if member.kind is ContextMemberKind.VALIDATION:
            command = member.payload.get("command")
            if isinstance(command, str) and command.strip():
                validation_commands.append(command.strip())
        if member.kind is ContextMemberKind.OPEN_OBLIGATION:
            open_obligations.append(member.member_id)
        if member.kind is ContextMemberKind.IMPACTED_SYMBOL:
            symbol = (
                member.payload.get("symbol")
                or member.payload.get("qualified_name")
                or member.member_id
            )
            impacted.append(str(symbol))

    frontier_view = {
        "blocks_automatic_repair": manifest.frontier.blocks_automatic_repair,
        "complete": manifest.frontier.complete,
        "entries": [
            {
                "disposition": item.disposition.value,
                "expansion_handle": item.expansion_handle,
                "frontier_id": item.frontier_id,
                "kind": item.kind.value,
                "member_id": item.member_id,
                "reason": item.reason,
            }
            for item in manifest.frontier.entries
        ],
        "interface": LLM_CONTEXT_FRONTIER_INTERFACE,
        "omitted_count": manifest.frontier.omitted_count,
        "unresolved_count": manifest.frontier.unresolved_count,
    }
    # Explicit: omitted unresolved frontier is never silent.
    if manifest.frontier.entries and frontier_view["complete"]:
        frontier_view["complete"] = False

    return DatabaseContextModelPacket(
        manifest_cid=manifest.manifest_cid,
        task_cid=manifest.task_cid,
        repository_id=manifest.repository_id,
        tree_id=manifest.tree_id,
        members=tuple(member_views),
        frontier=frontier_view,
        validation_commands=tuple(validation_commands),
        open_obligation_ids=tuple(open_obligations),
        impacted_symbols=tuple(impacted),
        token_estimate=manifest.token_estimate,
        truncated=manifest.truncated,
        authority={
            "class": AUTHORITY_CLASS,
            "completion_authoritative": False,
            "semantic_authority": False,
            "write_authority": False,
            "model_output_is_nomination_only": True,
        },
    )


def project_to_context_compiler_inputs(
    manifest: DatabaseContextManifest,
    *,
    budget: ContextBudget | None = None,
) -> dict[str, Any]:
    """Project a database context manifest into ContextCompiler-facing inputs.

    The ContextCompiler remains the semantic composition boundary; this helper
    only prepares ranked evidence references and invariant core fields.
    """

    if not isinstance(manifest, DatabaseContextManifest):
        raise DatabaseContextError("manifest must be DatabaseContextManifest")

    task_members = manifest.members_by_kind(ContextMemberKind.TASK)
    task_payload = dict(task_members[0].payload) if task_members else {}
    goal = {
        "goal_cid": manifest.goal_cid,
        "task_cid": manifest.task_cid,
        "title": task_payload.get("title") or task_payload.get("task_alias") or "",
    }
    authority = {
        "authority_class": AUTHORITY_CLASS,
        "policy_digest": manifest.policy_digest,
        "repository_id": manifest.repository_id,
        "schema_revision": manifest.schema_revision,
        "tree_id": manifest.tree_id,
    }
    scope = {
        "impacted_symbols": list(manifest.member_ids(kind=ContextMemberKind.IMPACTED_SYMBOL)),
        "plan_cid": manifest.plan_cid,
        "snapshot_id": manifest.snapshot_id,
        "worktree_delta": list(manifest.member_ids(kind=ContextMemberKind.WORKTREE_DELTA)),
    }
    acceptance = {
        "open_obligations": list(
            manifest.member_ids(kind=ContextMemberKind.OPEN_OBLIGATION)
        ),
        "validation_commands": [
            str(item.payload.get("command") or item.summary)
            for item in manifest.members_by_kind(ContextMemberKind.VALIDATION)
        ],
    }

    evidence: list[ContextReference] = []
    for member in manifest.members:
        if member.kind is ContextMemberKind.TASK:
            continue
        tier = (
            ContextTier.INVARIANT
            if member.required
            else ContextTier.EVIDENCE
        )
        evidence.append(
            ContextReference(
                reference_id=member.member_id,
                kind=member.kind.value,
                tier=tier,
                content_id=member.digest,
                summary=member.summary or member.member_id,
                token_count=max(1, estimate_tokens(member.byte_size)),
                repository_id=manifest.repository_id,
                tree_id=manifest.tree_id,
                metadata={
                    "digest": member.digest,
                    "ordinal": member.ordinal,
                    "progressive": bool(member.payload.get("progressive")),
                    "required": member.required,
                },
            )
        )

    expansion_refs: list[ContextReference] = []
    for entry in manifest.frontier.entries:
        if not entry.expansion_handle:
            continue
        expansion_refs.append(
            ContextReference(
                reference_id=entry.frontier_id,
                kind="expansion",
                tier=ContextTier.EXPANSION,
                content_id=content_identity(
                    {
                        "expansion_handle": entry.expansion_handle,
                        "frontier_id": entry.frontier_id,
                    }
                ),
                summary=entry.reason or entry.frontier_id,
                token_count=1,
                repository_id=manifest.repository_id,
                tree_id=manifest.tree_id,
                metadata={
                    "disposition": entry.disposition.value,
                    "expansion_handle": entry.expansion_handle,
                    "frontier_kind": entry.kind.value,
                    "required": False,
                },
            )
        )

    return {
        "acceptance": acceptance,
        "authority": authority,
        "budget": budget,
        "evidence": evidence,
        "expansion_references": expansion_refs,
        "frontier": manifest.frontier.to_dict(),
        "goal": goal,
        "manifest_cid": manifest.manifest_cid,
        "scope": scope,
    }


def page_members(
    manifest: DatabaseContextManifest,
    *,
    page: int = 0,
    page_size: int | None = None,
    kind: ContextMemberKind | str | None = None,
) -> tuple[tuple[DatabaseContextMember, ...], str]:
    """Paginate disclosed members for progressive disclosure."""

    size = (
        manifest.budget.page_size
        if page_size is None
        else _positive_int(page_size, "page_size")
    )
    if size > MAX_PAGE_SIZE:
        raise DatabaseContextBoundsError("page_size exceeds absolute limit")
    page = _nonneg_int(page, "page")
    items = (
        list(manifest.members_by_kind(kind))
        if kind is not None
        else [item for item in manifest.members if item.disclosed]
    )
    start = page * size
    end = start + size
    slice_items = tuple(items[start:end])
    next_token = f"page:{page + 1}" if end < len(items) else ""
    return slice_items, next_token


# Public re-export surface for discoverability.
__all__ = [
    "AUTHORITY_CLASS",
    "CONTEXT_DELTA_INTERFACE",
    "CONTEXT_DELTA_SCHEMA",
    "CONTRACT_VERSION",
    "DATABASE_CONTEXT_MANIFEST_INTERFACE",
    "DATABASE_CONTEXT_MANIFEST_SCHEMA",
    "DATABASE_CONTEXT_MODEL_PACKET_SCHEMA",
    "DEFAULT_MAX_BYTES",
    "DEFAULT_MAX_ROWS",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_PAGE_SIZE",
    "LLM_CONTEXT_FRONTIER_INTERFACE",
    "LLM_CONTEXT_FRONTIER_SCHEMA",
    "NOISE_FIELD_NAMES",
    "PRODUCER_ID",
    "REDACTION_MARKER",
    "REQUIRED_MEMBER_KINDS",
    "UNTRUSTED_DATA_LABEL",
    "ContextDelta",
    "ContextMemberKind",
    "DatabaseContextBudget",
    "DatabaseContextBoundsError",
    "DatabaseContextError",
    "DatabaseContextInvalidationError",
    "DatabaseContextManifest",
    "DatabaseContextMember",
    "DatabaseContextModelPacket",
    "DatabaseContextRequest",
    "DatabaseContextSecretError",
    "DatabaseContextStaleError",
    "FrontierDisposition",
    "FrontierKind",
    "LLMContextFrontier",
    "LLMContextFrontierEntry",
    "assert_dependency_freshness",
    "compare_and_delta",
    "compile_database_context",
    "estimate_tokens",
    "is_noise_field",
    "model_packet_from_manifest",
    "page_members",
    "project_to_context_compiler_inputs",
    "strip_noise",
]
