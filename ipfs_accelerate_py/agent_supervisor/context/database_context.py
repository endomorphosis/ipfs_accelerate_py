"""Bounded database context capsules, deltas, and LLM frontier views.

DQP-026 / Interfaces: ``DatabaseContextManifest@1``, ``ContextDelta@1``,
``LLMContextFrontier@1``
============================================================================

Builds content-addressed task context from database-shaped task/plan/event
and impact inputs.  Capsules carry only the bounded semantic core:

* task identity and status
* unmet dependencies
* latest distinct failure
* worktree delta (paths / digests, never raw dumps)
* impacted symbols
* open obligations
* relevant decisions / evidence handles
* exact validation commands

and enforce hard row / byte / token budgets with explicit progressive
disclosure.  Heartbeat and wall-clock noise are excluded from identity so
unchanged semantic state yields a stable context CID.

Acceptance properties
---------------------
* Unchanged semantic state yields identical context CID despite heartbeat
  and time noise.
* Changed evidence yields a bounded parent-bound delta.
* Omitted unresolved frontier is explicit (``LLMContextFrontier@1``).
* No secret material or unrestricted repository dump enters a model packet.

Cold import of this module performs no filesystem, database, network,
provider, or process action.  Opening a store is the first I/O boundary.

Conflict policy: this module owns context query/manifests.  Existing
:mod:`context_compiler` remains the semantic composition boundary when a
full :class:`ContextCapsule` is required.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Final

from .context_compiler import ContextCompiler
from .context_contracts import (
    ContextBudget,
    ContextCapsule,
    ContextReference,
    ContextTier,
)
from ..task_sources.duckdb_state import open_duckdb_connection

# ---------------------------------------------------------------------------
# Contract identity
# ---------------------------------------------------------------------------

DATABASE_CONTEXT_MANIFEST_INTERFACE: Final[str] = "DatabaseContextManifest@1"
CONTEXT_DELTA_INTERFACE: Final[str] = "ContextDelta@1"
LLM_CONTEXT_FRONTIER_INTERFACE: Final[str] = "LLMContextFrontier@1"
DATABASE_CONTEXT_STORE_INTERFACE: Final[str] = "DatabaseContextStore@1"

DATABASE_CONTEXT_MANIFEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-context-manifest@1"
)
CONTEXT_DELTA_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/context-delta@1"
)
LLM_CONTEXT_FRONTIER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/llm-context-frontier@1"
)
CONTEXT_MEMBER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-context-member@1"
)
CONTEXT_BUDGET_BIND_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-context-budget@1"
)
MODEL_PACKET_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-context-model-packet@1"
)

DEFAULT_POLICY_ID: Final[str] = "database-context-policy@1"
DEFAULT_SCHEMA_REVISION: Final[int] = 1
AUTHORITY_CLASS: Final[str] = "derived_evidence"
PRODUCER_ID: Final[str] = "database-context@1"
REDACTION_MARKER: Final[str] = "secret_material"
UNTRUSTED_DATA_LABEL: Final[str] = "untrusted_repository_data"

MAX_PATH_BYTES: Final[int] = 4_096
MAX_TEXT_BYTES: Final[int] = 4_096
MAX_SUMMARY_BYTES: Final[int] = 1_024
MAX_BODY_JSON_BYTES: Final[int] = 262_144
MAX_MEMBERS: Final[int] = 1_024
MAX_ROWS: Final[int] = 4_096
MAX_PAGE_SIZE: Final[int] = 256
DEFAULT_PAGE_SIZE: Final[int] = 32
DEFAULT_MAX_ROWS: Final[int] = 128
DEFAULT_MAX_BYTES: Final[int] = 32_000
DEFAULT_MAX_TOKENS: Final[int] = 4_096
MAX_VALIDATIONS: Final[int] = 64
MAX_DEPENDENCIES: Final[int] = 512
MAX_SYMBOLS: Final[int] = 1_024
MAX_OBLIGATIONS: Final[int] = 512
MAX_EVIDENCE: Final[int] = 512
MAX_DECISIONS: Final[int] = 256
MAX_DELTA_PATHS: Final[int] = 1_024

# Identity intentionally ignores these volatile keys.
_NOISE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "heartbeat_at",
        "heartbeat_ms",
        "last_heartbeat",
        "last_heartbeat_at",
        "observed_at",
        "observed_at_ms",
        "wall_time",
        "wall_clock",
        "now",
        "timestamp",
        "timestamps",
        "created_at",
        "updated_at",
        "recorded_at",
        "lease_heartbeat",
        "pid",
        "process_id",
        "clock_skew_ms",
    }
)

_SENSITIVE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "auth_token",
        "client_secret",
        "credential",
        "credentials",
        "github_token",
        "access_token",
        "refresh_token",
        "token",
        "password",
        "passphrase",
        "passwd",
        "private_key",
        "secret",
        "secrets",
        "secret_handle",
        "raw_secret",
    }
)

_BODY_FORBIDDEN_KEYS: Final[frozenset[str]] = frozenset(
    {
        "source_body",
        "source_text",
        "source_code",
        "ast_body",
        "proof_body",
        "proof_transcript",
        "file_content",
        "file_contents",
        "repository_body",
        "repository_dump",
        "repository_content",
        "raw_repository",
        "unrestricted_dump",
        "private_key",
        "secret",
        "secrets",
        "password",
        "token",
        "api_key",
        "authorization",
        "credential",
        "credentials",
    }
)

_TEXT_SECRET_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(
        r"(?i)\b(api[_ -]?key|access[_ -]?token|auth[_ -]?token|"
        r"client[_ -]?secret|password|passphrase|secret)"
        r"(\s*[:=]\s*)[^\s,;]{4,}"
    ),
    re.compile(
        r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?"
        r"-----END [A-Z0-9 ]*PRIVATE KEY-----",
        re.DOTALL,
    ),
)

_SECRET_PATH_MARKERS: Final[tuple[str, ...]] = (
    ".env",
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
    "credentials.json",
    "secrets.json",
    "private_key",
    ".pem",
    ".p12",
    ".pfx",
    "kubeconfig",
)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_BOOKKEEPING_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS database_context_metadata (
    key VARCHAR PRIMARY KEY,
    value VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS context_manifests (
    manifest_cid VARCHAR PRIMARY KEY,
    task_cid VARCHAR NOT NULL,
    schema_revision BIGINT NOT NULL,
    repository_tree_id VARCHAR NOT NULL,
    policy_digest VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);
CREATE INDEX IF NOT EXISTS context_manifests_task_idx
    ON context_manifests(task_cid, created_at);

CREATE TABLE IF NOT EXISTS context_members (
    manifest_cid VARCHAR NOT NULL,
    ordinal BIGINT NOT NULL,
    member_kind VARCHAR NOT NULL,
    member_id VARCHAR NOT NULL,
    digest VARCHAR NOT NULL,
    included INTEGER NOT NULL DEFAULT 1,
    byte_count BIGINT NOT NULL DEFAULT 0,
    token_count BIGINT NOT NULL DEFAULT 0,
    body_json VARCHAR NOT NULL DEFAULT '{}',
    PRIMARY KEY (manifest_cid, ordinal)
);

CREATE TABLE IF NOT EXISTS context_deltas (
    delta_id VARCHAR PRIMARY KEY,
    from_manifest_cid VARCHAR NOT NULL,
    to_manifest_cid VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS llm_context_frontiers (
    frontier_id VARCHAR PRIMARY KEY,
    manifest_cid VARCHAR NOT NULL,
    disposition VARCHAR NOT NULL,
    omitted_count BIGINT NOT NULL DEFAULT 0,
    body_json VARCHAR NOT NULL
);
CREATE INDEX IF NOT EXISTS llm_context_frontiers_manifest_idx
    ON llm_context_frontiers(manifest_cid);
"""


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DatabaseContextError(RuntimeError):
    """Base error for database context failures."""

    def __init__(self, message: str, *, reason_code: str = "database_context") -> None:
        super().__init__(message)
        self.reason_code = reason_code


class DatabaseContextNotOpenError(DatabaseContextError):
    """Operation requires an open context store."""

    def __init__(self, message: str = "DatabaseContextStore is not open") -> None:
        super().__init__(message, reason_code="not_open")


class DatabaseContextIntegrityError(DatabaseContextError, ValueError):
    """Identity, path, or payload integrity failure."""

    def __init__(
        self, message: str, *, reason_code: str = "integrity"
    ) -> None:
        super().__init__(message, reason_code=reason_code)


class DatabaseContextBoundsError(DatabaseContextError, ValueError):
    """A resource or payload bound was exceeded."""

    def __init__(self, message: str, *, reason_code: str = "bounds") -> None:
        super().__init__(message, reason_code=reason_code)


class DatabaseContextSecretError(DatabaseContextError, ValueError):
    """Secret or private material was presented for a model packet."""

    def __init__(
        self,
        message: str = "secret or private material is excluded from context",
        *,
        reason_code: str = "secret_material_rejected",
    ) -> None:
        super().__init__(message, reason_code=reason_code)


class DatabaseContextStaleError(DatabaseContextError, ValueError):
    """Input crossed an immutable repository/tree/policy boundary."""

    def __init__(
        self,
        message: str = "context input is stale relative to its binding",
        *,
        reason_code: str = "stale_input",
    ) -> None:
        super().__init__(message, reason_code=reason_code)


class DatabaseContextOverflowError(DatabaseContextError, ValueError):
    """Required core cannot fit within hard budgets."""

    def __init__(
        self,
        message: str = "required context exceeds hard budget",
        *,
        reason_code: str = "overflow",
    ) -> None:
        super().__init__(message, reason_code=reason_code)


class DuckDBUnavailableError(DatabaseContextError):
    """Optional DuckDB dependency is not installed."""

    def __init__(
        self, message: str = "DuckDB is required for DatabaseContextStore"
    ) -> None:
        super().__init__(message, reason_code="duckdb_unavailable")


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class MemberKind(str, Enum):
    """Closed vocabulary of context member kinds."""

    TASK = "task"
    DEPENDENCY = "dependency"
    FAILURE = "failure"
    WORKTREE_DELTA = "worktree_delta"
    IMPACTED_SYMBOL = "impacted_symbol"
    OBLIGATION = "obligation"
    DECISION = "decision"
    EVIDENCE = "evidence"
    VALIDATION = "validation"
    EXPANSION = "expansion"

    @classmethod
    def coerce(cls, value: Any) -> "MemberKind":
        if isinstance(value, cls):
            return value
        raw = str(getattr(value, "value", value) or "").strip().casefold()
        aliases: Mapping[str, MemberKind] = {
            "task": cls.TASK,
            "dependency": cls.DEPENDENCY,
            "dependencies": cls.DEPENDENCY,
            "unmet_dependency": cls.DEPENDENCY,
            "failure": cls.FAILURE,
            "latest_failure": cls.FAILURE,
            "worktree_delta": cls.WORKTREE_DELTA,
            "delta": cls.WORKTREE_DELTA,
            "impacted_symbol": cls.IMPACTED_SYMBOL,
            "impacted_symbols": cls.IMPACTED_SYMBOL,
            "symbol": cls.IMPACTED_SYMBOL,
            "obligation": cls.OBLIGATION,
            "open_obligation": cls.OBLIGATION,
            "decision": cls.DECISION,
            "evidence": cls.EVIDENCE,
            "validation": cls.VALIDATION,
            "validation_command": cls.VALIDATION,
            "expansion": cls.EXPANSION,
            "expansion_handle": cls.EXPANSION,
        }
        try:
            return aliases[raw]
        except KeyError as exc:
            raise DatabaseContextIntegrityError(
                f"unsupported member kind: {value!r}"
            ) from exc


class MemberTier(str, Enum):
    INVARIANT = "invariant"
    EVIDENCE = "evidence"
    SUGGESTION = "suggestion"
    EXPANSION = "expansion"


class FrontierDisposition(str, Enum):
    EMPTY = "empty"
    EXPLICIT_OMISSION = "explicit_omission"
    BUDGET_OVERFLOW = "budget_overflow"
    PAGINATED = "paginated"
    SECRET_EXCLUDED = "secret_excluded"
    PRIVATE_EXCLUDED = "private_excluded"
    STALE_BLOCKED = "stale_blocked"
    UNRESOLVED = "unresolved"


class Completeness(str, Enum):
    COMPLETE = "complete"
    PARTIAL_WITH_FRONTIER = "partial_with_frontier"
    OVERFLOW = "overflow"
    ABSTAINED = "abstained"


class InvalidationKind(str, Enum):
    DEPENDENCY = "dependency"
    TREE = "tree"
    POLICY = "policy"
    SCHEMA = "schema"
    TASK_REVISION = "task_revision"
    EVIDENCE = "evidence"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def duckdb_available() -> bool:
    """Return whether the optional duckdb package can be imported."""

    try:
        import duckdb  # type: ignore  # noqa: F401
    except ImportError:
        return False
    return True


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise DatabaseContextIntegrityError(
            "values must be canonical JSON"
        ) from exc


def _identity(prefix: str, value: Any) -> str:
    encoded = _canonical_json(value).encode("utf-8")
    return f"{prefix}:sha256:" + hashlib.sha256(encoded).hexdigest()


def _content_cid(value: Any) -> str:
    """Stable content identity used as a context CID."""

    encoded = _canonical_json(value).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return f"baguqeera{digest[:52]}"


def _text(value: Any, name: str, *, required: bool = True) -> str:
    text = str(value or "").strip()
    if "\x00" in text:
        raise DatabaseContextIntegrityError(f"{name} contains NUL")
    if required and not text:
        raise DatabaseContextIntegrityError(f"{name} is required")
    if len(text.encode("utf-8")) > MAX_TEXT_BYTES:
        raise DatabaseContextBoundsError(
            f"{name} exceeds {MAX_TEXT_BYTES} UTF-8 bytes"
        )
    return text


def _nonneg_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DatabaseContextBoundsError(
            f"{name} must be a non-negative integer"
        )
    return value


def _positive_int(value: Any, name: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DatabaseContextBoundsError(
            f"{name} must be a positive integer"
        )
    if maximum is not None and value > maximum:
        raise DatabaseContextBoundsError(
            f"{name} exceeds maximum {maximum}"
        )
    return value


def _bounded_text(value: Any, maximum: int = MAX_TEXT_BYTES) -> str:
    text = str(value or "")
    encoded = text.encode("utf-8", "replace")
    if len(encoded) <= maximum:
        return text
    marker = "…[truncated]"
    budget = max(0, maximum - len(marker.encode("utf-8")))
    return encoded[:budget].decode("utf-8", "ignore") + marker


def _repo_path(value: Any, *, required: bool = False) -> str:
    raw = str(value or "").strip().replace("\\", "/")
    while raw.startswith("./"):
        raw = raw[2:]
    if not raw:
        if required:
            raise DatabaseContextIntegrityError("repository path is required")
        return ""
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts or "\x00" in raw:
        raise DatabaseContextIntegrityError(
            f"repository path escapes its root: {value!r}"
        )
    normalized = path.as_posix()
    if len(normalized.encode("utf-8")) > MAX_PATH_BYTES:
        raise DatabaseContextBoundsError(
            f"path exceeds {MAX_PATH_BYTES} bytes: {normalized}"
        )
    return normalized


def _normalized_key(value: str) -> str:
    return value.strip().casefold().replace("-", "_").replace(" ", "_")


def _is_sensitive_key(key: str) -> bool:
    normalized = _normalized_key(key)
    if normalized in _SENSITIVE_KEYS or normalized in _BODY_FORBIDDEN_KEYS:
        return True
    if normalized.endswith("_secret") or normalized.endswith("_password"):
        return True
    if normalized.endswith("_api_key") or normalized.endswith("_private_key"):
        return True
    return False


def _looks_like_secret_path(path: str) -> bool:
    lowered = path.casefold()
    name = PurePosixPath(lowered).name
    if name.startswith(".env") or name.endswith(".env"):
        return True
    return any(marker in lowered for marker in _SECRET_PATH_MARKERS)


def _text_contains_secret_pattern(value: str) -> bool:
    return any(pattern.search(value) for pattern in _TEXT_SECRET_PATTERNS)


def _strip_noise(value: Any) -> Any:
    """Drop heartbeat/time noise so identity is semantic-only."""

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if _normalized_key(key_text) in _NOISE_KEYS:
                continue
            result[key_text] = _strip_noise(item)
        return result
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        return [_strip_noise(item) for item in value]
    return value


def _reject_or_redact_secrets(
    value: Any,
    *,
    path: str = "",
    reject: bool = True,
) -> Any:
    """Fail closed on secret/private material, or redact when allowed."""

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise DatabaseContextIntegrityError(
                    "context object keys must be strings"
                )
            key_path = f"{path}.{key}" if path else key
            if _is_sensitive_key(key) or key.casefold() in {
                str(item).casefold() for item in _BODY_FORBIDDEN_KEYS
            }:
                if reject:
                    raise DatabaseContextSecretError(
                        f"secret or private field excluded: {key_path}",
                        reason_code="secret_material_rejected",
                    )
                result[key] = REDACTION_MARKER
                continue
            result[key] = _reject_or_redact_secrets(
                item, path=key_path, reject=reject
            )
        return result
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        return [
            _reject_or_redact_secrets(
                item, path=f"{path}[{index}]", reject=reject
            )
            for index, item in enumerate(value)
        ]
    if isinstance(value, str):
        if _text_contains_secret_pattern(value):
            if reject:
                raise DatabaseContextSecretError(
                    f"secret pattern excluded at {path or 'value'}",
                    reason_code="secret_material_rejected",
                )
            return REDACTION_MARKER
        return value
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        raise DatabaseContextIntegrityError(
            "floating-point values are not canonical context"
        )
    raise DatabaseContextIntegrityError(
        f"unsupported context value type: {type(value).__name__}"
    )


def _estimate_tokens(value: Any) -> int:
    encoded = _canonical_json(value).encode("utf-8")
    return max(1, (len(encoded) + 3) // 4)


def _byte_count(value: Any) -> int:
    return len(_canonical_json(value).encode("utf-8"))


def _split_sql_statements(sql_text: str) -> list[str]:
    statements: list[str] = []
    for chunk in str(sql_text).split(";"):
        statement = chunk.strip()
        if not statement:
            continue
        lines = [
            line
            for line in statement.splitlines()
            if line.strip() and not line.strip().startswith("--")
        ]
        if lines:
            statements.append("\n".join(lines))
    return statements


def _row_mapping(row: Any) -> dict[str, Any]:
    if isinstance(row, Mapping):
        return {str(key): row[key] for key in row}
    try:
        keys = list(row.keys())  # type: ignore[attr-defined]
    except Exception:
        return {}
    return {str(key): row[key] for key in keys}


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(value))


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContextBudgetSpec:
    """Hard row / byte / token budgets for one context build."""

    max_rows: int = DEFAULT_MAX_ROWS
    max_bytes: int = DEFAULT_MAX_BYTES
    max_tokens: int = DEFAULT_MAX_TOKENS
    page_size: int = DEFAULT_PAGE_SIZE
    page_offset: int = 0
    max_members: int = MAX_MEMBERS

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_rows",
            _positive_int(int(self.max_rows), "max_rows", maximum=MAX_ROWS),
        )
        object.__setattr__(
            self,
            "max_bytes",
            _positive_int(int(self.max_bytes), "max_bytes"),
        )
        object.__setattr__(
            self,
            "max_tokens",
            _positive_int(int(self.max_tokens), "max_tokens"),
        )
        object.__setattr__(
            self,
            "page_size",
            _positive_int(
                int(self.page_size), "page_size", maximum=MAX_PAGE_SIZE
            ),
        )
        object.__setattr__(
            self,
            "page_offset",
            _nonneg_int(int(self.page_offset), "page_offset"),
        )
        object.__setattr__(
            self,
            "max_members",
            _positive_int(
                int(self.max_members), "max_members", maximum=MAX_MEMBERS
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CONTEXT_BUDGET_BIND_SCHEMA,
            "max_rows": self.max_rows,
            "max_bytes": self.max_bytes,
            "max_tokens": self.max_tokens,
            "page_size": self.page_size,
            "page_offset": self.page_offset,
            "max_members": self.max_members,
        }


@dataclass(frozen=True)
class ContextMember:
    """One content-addressed member of a database context manifest."""

    member_id: str
    kind: MemberKind | str
    digest: str
    tier: MemberTier | str = MemberTier.EVIDENCE
    summary: str = ""
    path: str = ""
    included: bool = True
    byte_count: int = 0
    token_count: int = 0
    ordinal: int = 0
    payload: Mapping[str, Any] = field(default_factory=dict)
    omission_reason: str = ""
    expansion_handle: str = ""

    def __post_init__(self) -> None:
        kind = MemberKind.coerce(self.kind)
        object.__setattr__(self, "kind", kind)
        tier = self.tier
        if not isinstance(tier, MemberTier):
            tier = MemberTier(str(tier).strip().casefold())
        object.__setattr__(self, "tier", tier)
        object.__setattr__(
            self, "summary", _bounded_text(self.summary, MAX_SUMMARY_BYTES)
        )
        object.__setattr__(
            self, "path", _repo_path(self.path, required=False)
        )
        if self.path and _looks_like_secret_path(self.path):
            raise DatabaseContextSecretError(
                f"secret-bearing path excluded: {self.path}",
                reason_code="secret_path_excluded",
            )
        payload = _reject_or_redact_secrets(
            _strip_noise(dict(self.payload or {})), reject=True
        )
        if not isinstance(payload, dict):
            raise DatabaseContextIntegrityError("member payload must be object")
        object.__setattr__(self, "payload", _freeze_mapping(payload))
        object.__setattr__(
            self, "byte_count", _nonneg_int(int(self.byte_count), "byte_count")
        )
        object.__setattr__(
            self,
            "token_count",
            _nonneg_int(int(self.token_count), "token_count"),
        )
        object.__setattr__(
            self, "ordinal", _nonneg_int(int(self.ordinal), "ordinal")
        )
        object.__setattr__(
            self,
            "omission_reason",
            _bounded_text(self.omission_reason, MAX_SUMMARY_BYTES),
        )
        object.__setattr__(
            self,
            "expansion_handle",
            _text(self.expansion_handle, "expansion_handle", required=False),
        )
        body_for_digest = {
            "schema": CONTEXT_MEMBER_SCHEMA,
            "kind": kind.value,
            "tier": tier.value,
            "summary": self.summary,
            "path": self.path,
            "payload": payload,
        }
        computed_digest = _identity("member", body_for_digest)
        claimed_digest = str(self.digest or "").strip()
        if claimed_digest and claimed_digest != computed_digest:
            raise DatabaseContextIntegrityError(
                "context member digest does not match payload"
            )
        object.__setattr__(self, "digest", claimed_digest or computed_digest)
        claimed_id = str(self.member_id or "").strip()
        # Identity is semantic (kind + digest). Ordinal is only presentation order
        # and must not churn member IDs when neighboring rows are inserted.
        computed_id = _identity(
            "ctx-member",
            {
                "kind": kind.value,
                "digest": self.digest,
            },
        )
        object.__setattr__(self, "member_id", claimed_id or computed_id)
        if self.byte_count == 0:
            object.__setattr__(self, "byte_count", _byte_count(body_for_digest))
        if self.token_count == 0:
            object.__setattr__(
                self, "token_count", _estimate_tokens(body_for_digest)
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CONTEXT_MEMBER_SCHEMA,
            "member_id": self.member_id,
            "kind": self.kind.value
            if isinstance(self.kind, MemberKind)
            else str(self.kind),
            "tier": self.tier.value
            if isinstance(self.tier, MemberTier)
            else str(self.tier),
            "digest": self.digest,
            "summary": self.summary,
            "path": self.path,
            "included": bool(self.included),
            "byte_count": self.byte_count,
            "token_count": self.token_count,
            "ordinal": self.ordinal,
            "payload": dict(self.payload),
            "omission_reason": self.omission_reason,
            "expansion_handle": self.expansion_handle,
        }


@dataclass(frozen=True)
class LLMContextFrontier:
    """Explicit omitted / unresolved frontier for progressive disclosure.

    Interface: ``LLMContextFrontier@1``.
    """

    frontier_id: str
    disposition: FrontierDisposition | str
    omitted_member_ids: tuple[str, ...] = ()
    omitted_kinds: tuple[str, ...] = ()
    expansion_handles: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    page_offset: int = 0
    page_size: int = DEFAULT_PAGE_SIZE
    has_more: bool = False
    secret_excluded_count: int = 0
    private_excluded_count: int = 0
    schema: str = LLM_CONTEXT_FRONTIER_SCHEMA

    def __post_init__(self) -> None:
        disposition = self.disposition
        if not isinstance(disposition, FrontierDisposition):
            disposition = FrontierDisposition(
                str(disposition).strip().casefold()
            )
        object.__setattr__(self, "disposition", disposition)
        omitted = tuple(
            dict.fromkeys(
                _text(item, "omitted_member_id")
                for item in self.omitted_member_ids
                if str(item).strip()
            )
        )
        object.__setattr__(self, "omitted_member_ids", omitted)
        kinds = tuple(
            dict.fromkeys(
                str(item).strip().casefold()
                for item in self.omitted_kinds
                if str(item).strip()
            )
        )
        object.__setattr__(self, "omitted_kinds", kinds)
        handles = tuple(
            dict.fromkeys(
                _text(item, "expansion_handle")
                for item in self.expansion_handles
                if str(item).strip()
            )
        )
        object.__setattr__(self, "expansion_handles", handles)
        reasons = tuple(
            _bounded_text(item, MAX_SUMMARY_BYTES)
            for item in self.reasons
            if str(item).strip()
        )
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(
            self, "page_offset", _nonneg_int(int(self.page_offset), "page_offset")
        )
        object.__setattr__(
            self,
            "page_size",
            _positive_int(
                int(self.page_size), "page_size", maximum=MAX_PAGE_SIZE
            ),
        )
        object.__setattr__(
            self,
            "secret_excluded_count",
            _nonneg_int(int(self.secret_excluded_count), "secret_excluded_count"),
        )
        object.__setattr__(
            self,
            "private_excluded_count",
            _nonneg_int(
                int(self.private_excluded_count), "private_excluded_count"
            ),
        )
        if self.schema != LLM_CONTEXT_FRONTIER_SCHEMA:
            raise DatabaseContextIntegrityError(
                "unsupported llm context frontier schema"
            )
        claimed = str(self.frontier_id or "").strip()
        computed = _identity(
            "llm-frontier",
            {
                "schema": self.schema,
                "disposition": disposition.value,
                "omitted_member_ids": list(omitted),
                "omitted_kinds": list(kinds),
                "expansion_handles": list(handles),
                "page_offset": self.page_offset,
                "page_size": self.page_size,
                "has_more": bool(self.has_more),
            },
        )
        object.__setattr__(self, "frontier_id", claimed or computed)

    @property
    def interface(self) -> str:
        return LLM_CONTEXT_FRONTIER_INTERFACE

    @property
    def is_explicit(self) -> bool:
        return self.disposition is not FrontierDisposition.EMPTY or bool(
            self.omitted_member_ids or self.reasons
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": LLM_CONTEXT_FRONTIER_INTERFACE,
            "frontier_id": self.frontier_id,
            "disposition": self.disposition.value
            if isinstance(self.disposition, FrontierDisposition)
            else str(self.disposition),
            "omitted_member_ids": list(self.omitted_member_ids),
            "omitted_kinds": list(self.omitted_kinds),
            "expansion_handles": list(self.expansion_handles),
            "reasons": list(self.reasons),
            "page_offset": self.page_offset,
            "page_size": self.page_size,
            "has_more": bool(self.has_more),
            "secret_excluded_count": self.secret_excluded_count,
            "private_excluded_count": self.private_excluded_count,
            "omitted_count": len(self.omitted_member_ids),
            "authority": AUTHORITY_CLASS,
        }


@dataclass(frozen=True)
class DatabaseContextManifest:
    """Content-addressed bounded task context manifest.

    Interface: ``DatabaseContextManifest@1``.
    """

    manifest_cid: str
    task_cid: str
    repository_id: str
    tree_id: str
    policy_id: str
    policy_digest: str
    schema_revision: int
    members: tuple[ContextMember, ...]
    frontier: LLMContextFrontier
    completeness: Completeness | str = Completeness.COMPLETE
    budget: ContextBudgetSpec = field(default_factory=ContextBudgetSpec)
    dependency_digests: tuple[str, ...] = ()
    task_revision: str = ""
    plan_cid: str = ""
    goal_cid: str = ""
    total_bytes: int = 0
    total_tokens: int = 0
    total_rows: int = 0
    producer_id: str = PRODUCER_ID
    authority: str = AUTHORITY_CLASS
    schema: str = DATABASE_CONTEXT_MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_cid", _text(self.task_cid, "task_cid"))
        object.__setattr__(
            self, "repository_id", _text(self.repository_id, "repository_id")
        )
        object.__setattr__(self, "tree_id", _text(self.tree_id, "tree_id"))
        object.__setattr__(
            self, "policy_id", _text(self.policy_id, "policy_id")
        )
        object.__setattr__(
            self, "policy_digest", _text(self.policy_digest, "policy_digest")
        )
        object.__setattr__(
            self,
            "schema_revision",
            _positive_int(int(self.schema_revision), "schema_revision"),
        )
        members = tuple(self.members)
        if len(members) > MAX_MEMBERS:
            raise DatabaseContextBoundsError(
                f"member count exceeds {MAX_MEMBERS}"
            )
        seen_ids: set[str] = set()
        for member in members:
            if not isinstance(member, ContextMember):
                raise DatabaseContextIntegrityError(
                    "members must be ContextMember"
                )
            if member.member_id in seen_ids:
                raise DatabaseContextIntegrityError(
                    f"duplicate member id: {member.member_id}"
                )
            seen_ids.add(member.member_id)
        object.__setattr__(self, "members", members)
        if not isinstance(self.frontier, LLMContextFrontier):
            raise DatabaseContextIntegrityError(
                "frontier must be LLMContextFrontier"
            )
        completeness = self.completeness
        if not isinstance(completeness, Completeness):
            completeness = Completeness(str(completeness).strip().casefold())
        object.__setattr__(self, "completeness", completeness)
        if not isinstance(self.budget, ContextBudgetSpec):
            raise DatabaseContextIntegrityError(
                "budget must be ContextBudgetSpec"
            )
        deps = tuple(
            dict.fromkeys(
                _text(item, "dependency_digest")
                for item in self.dependency_digests
                if str(item).strip()
            )
        )
        object.__setattr__(self, "dependency_digests", deps)
        object.__setattr__(
            self,
            "task_revision",
            _text(self.task_revision, "task_revision", required=False),
        )
        object.__setattr__(
            self, "plan_cid", _text(self.plan_cid, "plan_cid", required=False)
        )
        object.__setattr__(
            self, "goal_cid", _text(self.goal_cid, "goal_cid", required=False)
        )
        included = [item for item in members if item.included]
        total_bytes = sum(item.byte_count for item in included)
        total_tokens = sum(item.token_count for item in included)
        total_rows = len(included)
        object.__setattr__(
            self,
            "total_bytes",
            _nonneg_int(int(self.total_bytes or total_bytes), "total_bytes"),
        )
        object.__setattr__(
            self,
            "total_tokens",
            _nonneg_int(int(self.total_tokens or total_tokens), "total_tokens"),
        )
        object.__setattr__(
            self,
            "total_rows",
            _nonneg_int(int(self.total_rows or total_rows), "total_rows"),
        )
        if self.schema != DATABASE_CONTEXT_MANIFEST_SCHEMA:
            raise DatabaseContextIntegrityError(
                "unsupported database context manifest schema"
            )
        identity_body = self._identity_body()
        computed = _content_cid(identity_body)
        claimed = str(self.manifest_cid or "").strip()
        if claimed and claimed != computed:
            raise DatabaseContextIntegrityError(
                "manifest CID does not match semantic payload"
            )
        object.__setattr__(self, "manifest_cid", claimed or computed)

    def _identity_body(self) -> dict[str, Any]:
        """Semantic body used for stable identity (no wall-clock noise)."""

        return {
            "schema": self.schema,
            "task_cid": self.task_cid,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "policy_id": self.policy_id,
            "policy_digest": self.policy_digest,
            "schema_revision": self.schema_revision,
            "task_revision": self.task_revision,
            "plan_cid": self.plan_cid,
            "goal_cid": self.goal_cid,
            "dependency_digests": list(self.dependency_digests),
            "members": [
                {
                    "member_id": item.member_id,
                    "kind": item.kind.value
                    if isinstance(item.kind, MemberKind)
                    else str(item.kind),
                    "digest": item.digest,
                    "included": bool(item.included),
                    "ordinal": item.ordinal,
                }
                for item in self.members
            ],
            "frontier": {
                "disposition": self.frontier.disposition.value
                if isinstance(self.frontier.disposition, FrontierDisposition)
                else str(self.frontier.disposition),
                "omitted_member_ids": list(self.frontier.omitted_member_ids),
                "has_more": bool(self.frontier.has_more),
            },
            "completeness": self.completeness.value
            if isinstance(self.completeness, Completeness)
            else str(self.completeness),
            "budget": self.budget.to_dict(),
            "producer_id": self.producer_id,
        }

    @property
    def interface(self) -> str:
        return DATABASE_CONTEXT_MANIFEST_INTERFACE

    @property
    def context_cid(self) -> str:
        return self.manifest_cid

    def included_members(self) -> tuple[ContextMember, ...]:
        return tuple(item for item in self.members if item.included)

    def omitted_members(self) -> tuple[ContextMember, ...]:
        return tuple(item for item in self.members if not item.included)

    def member_digest_map(self) -> Mapping[str, str]:
        return MappingProxyType(
            {item.member_id: item.digest for item in self.members if item.included}
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": DATABASE_CONTEXT_MANIFEST_INTERFACE,
            "manifest_cid": self.manifest_cid,
            "context_cid": self.manifest_cid,
            "task_cid": self.task_cid,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "policy_id": self.policy_id,
            "policy_digest": self.policy_digest,
            "schema_revision": self.schema_revision,
            "task_revision": self.task_revision,
            "plan_cid": self.plan_cid,
            "goal_cid": self.goal_cid,
            "members": [item.to_dict() for item in self.members],
            "frontier": self.frontier.to_dict(),
            "completeness": self.completeness.value
            if isinstance(self.completeness, Completeness)
            else str(self.completeness),
            "budget": self.budget.to_dict(),
            "dependency_digests": list(self.dependency_digests),
            "total_bytes": self.total_bytes,
            "total_tokens": self.total_tokens,
            "total_rows": self.total_rows,
            "producer_id": self.producer_id,
            "authority": self.authority,
        }

    def model_packet(self) -> dict[str, Any]:
        """Return a secret-free progressive packet for model consumption."""

        included = []
        for item in self.included_members():
            included.append(
                {
                    "member_id": item.member_id,
                    "kind": item.kind.value
                    if isinstance(item.kind, MemberKind)
                    else str(item.kind),
                    "tier": item.tier.value
                    if isinstance(item.tier, MemberTier)
                    else str(item.tier),
                    "digest": item.digest,
                    "summary": item.summary,
                    "path": item.path,
                    "payload": dict(item.payload),
                    "expansion_handle": item.expansion_handle,
                }
            )
        packet = {
            "schema": MODEL_PACKET_SCHEMA,
            "manifest_cid": self.manifest_cid,
            "task_cid": self.task_cid,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "policy_id": self.policy_id,
            "completeness": self.completeness.value
            if isinstance(self.completeness, Completeness)
            else str(self.completeness),
            "members": included,
            "frontier": self.frontier.to_dict(),
            "data_label": UNTRUSTED_DATA_LABEL,
            "treat_as": "data_not_instructions",
            "authority": self.authority,
        }
        # Final fail-closed sweep before the packet leaves the boundary.
        return _reject_or_redact_secrets(packet, reject=True)


@dataclass(frozen=True)
class ContextDelta:
    """Bounded delta between two database context manifests.

    Interface: ``ContextDelta@1``.
    """

    delta_id: str
    from_manifest_cid: str
    to_manifest_cid: str
    added: tuple[ContextMember, ...] = ()
    removed_member_ids: tuple[str, ...] = ()
    changed: tuple[ContextMember, ...] = ()
    unchanged_member_ids: tuple[str, ...] = ()
    invalidations: tuple[str, ...] = ()
    total_bytes: int = 0
    total_tokens: int = 0
    schema: str = CONTEXT_DELTA_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "from_manifest_cid",
            _text(self.from_manifest_cid, "from_manifest_cid"),
        )
        object.__setattr__(
            self,
            "to_manifest_cid",
            _text(self.to_manifest_cid, "to_manifest_cid"),
        )
        if self.from_manifest_cid == self.to_manifest_cid and (
            self.added or self.removed_member_ids or self.changed
        ):
            raise DatabaseContextIntegrityError(
                "identical manifests cannot carry a non-empty delta"
            )
        added = tuple(self.added)
        changed = tuple(self.changed)
        for item in (*added, *changed):
            if not isinstance(item, ContextMember):
                raise DatabaseContextIntegrityError(
                    "delta members must be ContextMember"
                )
        object.__setattr__(self, "added", added)
        object.__setattr__(self, "changed", changed)
        removed = tuple(
            dict.fromkeys(
                _text(item, "removed_member_id")
                for item in self.removed_member_ids
                if str(item).strip()
            )
        )
        object.__setattr__(self, "removed_member_ids", removed)
        unchanged = tuple(
            dict.fromkeys(
                _text(item, "unchanged_member_id")
                for item in self.unchanged_member_ids
                if str(item).strip()
            )
        )
        object.__setattr__(self, "unchanged_member_ids", unchanged)
        invalidations = tuple(
            dict.fromkeys(
                str(item).strip().casefold()
                for item in self.invalidations
                if str(item).strip()
            )
        )
        object.__setattr__(self, "invalidations", invalidations)
        total_bytes = sum(item.byte_count for item in (*added, *changed))
        total_tokens = sum(item.token_count for item in (*added, *changed))
        object.__setattr__(
            self,
            "total_bytes",
            _nonneg_int(int(self.total_bytes or total_bytes), "total_bytes"),
        )
        object.__setattr__(
            self,
            "total_tokens",
            _nonneg_int(int(self.total_tokens or total_tokens), "total_tokens"),
        )
        if self.schema != CONTEXT_DELTA_SCHEMA:
            raise DatabaseContextIntegrityError(
                "unsupported context delta schema"
            )
        claimed = str(self.delta_id or "").strip()
        computed = _identity(
            "context-delta",
            {
                "schema": self.schema,
                "from_manifest_cid": self.from_manifest_cid,
                "to_manifest_cid": self.to_manifest_cid,
                "added": [item.digest for item in added],
                "removed_member_ids": list(removed),
                "changed": [item.digest for item in changed],
                "invalidations": list(invalidations),
            },
        )
        object.__setattr__(self, "delta_id", claimed or computed)

    @property
    def interface(self) -> str:
        return CONTEXT_DELTA_INTERFACE

    @property
    def is_empty(self) -> bool:
        return not (self.added or self.removed_member_ids or self.changed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": CONTEXT_DELTA_INTERFACE,
            "delta_id": self.delta_id,
            "from_manifest_cid": self.from_manifest_cid,
            "to_manifest_cid": self.to_manifest_cid,
            "added": [item.to_dict() for item in self.added],
            "removed_member_ids": list(self.removed_member_ids),
            "changed": [item.to_dict() for item in self.changed],
            "unchanged_member_ids": list(self.unchanged_member_ids),
            "invalidations": list(self.invalidations),
            "total_bytes": self.total_bytes,
            "total_tokens": self.total_tokens,
            "authority": AUTHORITY_CLASS,
        }


@dataclass(frozen=True)
class TaskContextInput:
    """Normalized inputs used to build one database context manifest."""

    task_cid: str
    repository_id: str
    tree_id: str
    policy_id: str = DEFAULT_POLICY_ID
    policy_digest: str = ""
    schema_revision: int = DEFAULT_SCHEMA_REVISION
    task_revision: str = ""
    plan_cid: str = ""
    goal_cid: str = ""
    task_status: str = "ready"
    task_summary: str = ""
    unmet_dependencies: tuple[Mapping[str, Any] | str, ...] = ()
    latest_failure: Mapping[str, Any] | None = None
    worktree_delta: Mapping[str, Any] | Sequence[Any] | None = None
    impacted_symbols: tuple[Mapping[str, Any] | str, ...] = ()
    open_obligations: tuple[Mapping[str, Any] | str, ...] = ()
    decisions: tuple[Mapping[str, Any] | str, ...] = ()
    evidence: tuple[Mapping[str, Any] | str, ...] = ()
    validations: tuple[Mapping[str, Any] | str, ...] = ()
    expected_tree_id: str = ""
    expected_policy_digest: str = ""
    expected_dependency_digests: tuple[str, ...] = ()
    heartbeat_at: str = ""
    observed_at: str = ""
    budget: ContextBudgetSpec = field(default_factory=ContextBudgetSpec)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_cid", _text(self.task_cid, "task_cid"))
        object.__setattr__(
            self, "repository_id", _text(self.repository_id, "repository_id")
        )
        object.__setattr__(self, "tree_id", _text(self.tree_id, "tree_id"))
        object.__setattr__(
            self,
            "policy_id",
            _text(self.policy_id or DEFAULT_POLICY_ID, "policy_id"),
        )
        policy_digest = str(self.policy_digest or "").strip()
        if not policy_digest:
            policy_digest = _identity(
                "policy",
                {"policy_id": self.policy_id, "schema_revision": self.schema_revision},
            )
        object.__setattr__(self, "policy_digest", policy_digest)
        object.__setattr__(
            self,
            "schema_revision",
            _positive_int(int(self.schema_revision), "schema_revision"),
        )
        object.__setattr__(
            self,
            "task_revision",
            _text(self.task_revision, "task_revision", required=False),
        )
        object.__setattr__(
            self, "plan_cid", _text(self.plan_cid, "plan_cid", required=False)
        )
        object.__setattr__(
            self, "goal_cid", _text(self.goal_cid, "goal_cid", required=False)
        )
        object.__setattr__(
            self,
            "task_status",
            _text(self.task_status or "ready", "task_status"),
        )
        object.__setattr__(
            self,
            "task_summary",
            _bounded_text(self.task_summary, MAX_SUMMARY_BYTES),
        )
        if not isinstance(self.budget, ContextBudgetSpec):
            raise DatabaseContextIntegrityError(
                "budget must be ContextBudgetSpec"
            )
        # Noise fields are accepted but never enter identity.
        object.__setattr__(
            self,
            "heartbeat_at",
            _text(self.heartbeat_at, "heartbeat_at", required=False),
        )
        object.__setattr__(
            self,
            "observed_at",
            _text(self.observed_at, "observed_at", required=False),
        )
        meta = _reject_or_redact_secrets(
            _strip_noise(dict(self.metadata or {})), reject=True
        )
        object.__setattr__(self, "metadata", _freeze_mapping(meta))
        object.__setattr__(
            self,
            "expected_tree_id",
            _text(self.expected_tree_id, "expected_tree_id", required=False),
        )
        object.__setattr__(
            self,
            "expected_policy_digest",
            _text(
                self.expected_policy_digest,
                "expected_policy_digest",
                required=False,
            ),
        )
        expected_deps = tuple(
            dict.fromkeys(
                _text(item, "expected_dependency_digest")
                for item in self.expected_dependency_digests
                if str(item).strip()
            )
        )
        object.__setattr__(self, "expected_dependency_digests", expected_deps)


# ---------------------------------------------------------------------------
# Build pipeline
# ---------------------------------------------------------------------------


def _as_mapping_item(
    item: Mapping[str, Any] | str,
    *,
    id_key: str,
    kind: MemberKind,
) -> dict[str, Any]:
    if isinstance(item, Mapping):
        payload = _reject_or_redact_secrets(
            _strip_noise(dict(item)), reject=True
        )
        if not isinstance(payload, dict):
            raise DatabaseContextIntegrityError(f"{kind.value} must be object")
        member_key = str(
            payload.get(id_key)
            or payload.get("id")
            or payload.get("member_id")
            or ""
        ).strip()
        summary = str(
            payload.get("summary")
            or payload.get("title")
            or payload.get("command")
            or payload.get("name")
            or member_key
            or kind.value
        )
        path = str(payload.get("path") or "")
        return {
            "id": member_key or _identity(kind.value, payload),
            "summary": _bounded_text(summary, MAX_SUMMARY_BYTES),
            "path": _repo_path(path, required=False),
            "payload": payload,
        }
    text = _text(item, kind.value)
    return {
        "id": text,
        "summary": _bounded_text(text, MAX_SUMMARY_BYTES),
        "path": "",
        "payload": {"id": text},
    }


def _worktree_delta_items(
    value: Mapping[str, Any] | Sequence[Any] | None,
) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        paths = value.get("paths") or value.get("changed_paths") or ()
        digests = value.get("digests") or value.get("path_digests") or {}
        if isinstance(paths, (str, bytes)) or not isinstance(paths, Sequence):
            # Single object form: treat as one delta record.
            payload = _reject_or_redact_secrets(
                _strip_noise(dict(value)), reject=True
            )
            path = _repo_path(str(payload.get("path") or ""), required=False)
            return [
                {
                    "id": str(payload.get("id") or path or "worktree-delta"),
                    "summary": _bounded_text(
                        payload.get("summary") or path or "worktree_delta",
                        MAX_SUMMARY_BYTES,
                    ),
                    "path": path,
                    "payload": payload,
                }
            ]
        items: list[dict[str, Any]] = []
        digest_map = digests if isinstance(digests, Mapping) else {}
        for path_value in paths:
            path = _repo_path(path_value, required=False)
            if not path:
                continue
            if _looks_like_secret_path(path):
                raise DatabaseContextSecretError(
                    f"secret-bearing path excluded: {path}",
                    reason_code="secret_path_excluded",
                )
            payload = {
                "path": path,
                "digest": str(digest_map.get(path) or ""),
            }
            items.append(
                {
                    "id": path,
                    "summary": path,
                    "path": path,
                    "payload": payload,
                }
            )
        if len(items) > MAX_DELTA_PATHS:
            raise DatabaseContextBoundsError(
                f"worktree delta exceeds {MAX_DELTA_PATHS} paths"
            )
        return items
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        items = []
        for entry in value:
            if isinstance(entry, Mapping):
                items.extend(_worktree_delta_items(entry))
            else:
                path = _repo_path(entry, required=False)
                if not path:
                    continue
                if _looks_like_secret_path(path):
                    raise DatabaseContextSecretError(
                        f"secret-bearing path excluded: {path}",
                        reason_code="secret_path_excluded",
                    )
                items.append(
                    {
                        "id": path,
                        "summary": path,
                        "path": path,
                        "payload": {"path": path},
                    }
                )
        return items
    raise DatabaseContextIntegrityError("worktree_delta must be object or list")


def _candidate_members(request: TaskContextInput) -> list[ContextMember]:
    candidates: list[tuple[MemberKind, MemberTier, dict[str, Any]]] = []

    task_payload = {
        "task_cid": request.task_cid,
        "status": request.task_status,
        "summary": request.task_summary,
        "task_revision": request.task_revision,
        "plan_cid": request.plan_cid,
        "goal_cid": request.goal_cid,
    }
    candidates.append(
        (
            MemberKind.TASK,
            MemberTier.INVARIANT,
            {
                "id": request.task_cid,
                "summary": request.task_summary or request.task_cid,
                "path": "",
                "payload": task_payload,
            },
        )
    )

    for item in request.unmet_dependencies[:MAX_DEPENDENCIES]:
        candidates.append(
            (
                MemberKind.DEPENDENCY,
                MemberTier.INVARIANT,
                _as_mapping_item(
                    item, id_key="dependency_id", kind=MemberKind.DEPENDENCY
                ),
            )
        )

    if request.latest_failure:
        candidates.append(
            (
                MemberKind.FAILURE,
                MemberTier.EVIDENCE,
                _as_mapping_item(
                    request.latest_failure,
                    id_key="failure_id",
                    kind=MemberKind.FAILURE,
                ),
            )
        )

    for item in _worktree_delta_items(request.worktree_delta):
        candidates.append(
            (MemberKind.WORKTREE_DELTA, MemberTier.EVIDENCE, item)
        )

    for item in request.impacted_symbols[:MAX_SYMBOLS]:
        candidates.append(
            (
                MemberKind.IMPACTED_SYMBOL,
                MemberTier.EVIDENCE,
                _as_mapping_item(
                    item, id_key="symbol", kind=MemberKind.IMPACTED_SYMBOL
                ),
            )
        )

    for item in request.open_obligations[:MAX_OBLIGATIONS]:
        candidates.append(
            (
                MemberKind.OBLIGATION,
                MemberTier.INVARIANT,
                _as_mapping_item(
                    item, id_key="obligation_id", kind=MemberKind.OBLIGATION
                ),
            )
        )

    for item in request.decisions[:MAX_DECISIONS]:
        candidates.append(
            (
                MemberKind.DECISION,
                MemberTier.EVIDENCE,
                _as_mapping_item(
                    item, id_key="decision_id", kind=MemberKind.DECISION
                ),
            )
        )

    for item in request.evidence[:MAX_EVIDENCE]:
        candidates.append(
            (
                MemberKind.EVIDENCE,
                MemberTier.EVIDENCE,
                _as_mapping_item(
                    item, id_key="evidence_id", kind=MemberKind.EVIDENCE
                ),
            )
        )

    for item in request.validations[:MAX_VALIDATIONS]:
        candidates.append(
            (
                MemberKind.VALIDATION,
                MemberTier.INVARIANT,
                _as_mapping_item(
                    item, id_key="command", kind=MemberKind.VALIDATION
                ),
            )
        )

    members: list[ContextMember] = []
    for ordinal, (kind, tier, item) in enumerate(candidates):
        members.append(
            ContextMember(
                member_id="",
                kind=kind,
                digest="",
                tier=tier,
                summary=str(item.get("summary") or ""),
                path=str(item.get("path") or ""),
                included=True,
                ordinal=ordinal,
                payload=item.get("payload") or {},
            )
        )
    return members


def _apply_budgets(
    members: Sequence[ContextMember],
    budget: ContextBudgetSpec,
) -> tuple[
    list[ContextMember],
    list[ContextMember],
    FrontierDisposition,
    list[str],
]:
    """Select included members under hard budgets; preserve invariant core."""

    page = members[budget.page_offset : budget.page_offset + budget.page_size]
    remainder_after_page = members[budget.page_offset + budget.page_size :]
    before_page = members[: budget.page_offset]

    included: list[ContextMember] = []
    omitted: list[ContextMember] = []
    reasons: list[str] = []
    used_bytes = 0
    used_tokens = 0
    used_rows = 0
    disposition = FrontierDisposition.EMPTY

    def _mark(
        member: ContextMember,
        *,
        included_flag: bool,
        omission_reason: str = "",
        expansion_handle: str = "",
    ) -> ContextMember:
        # Preserve digest/payload/tier identity; only flip inclusion metadata.
        # Tier stays immutable so content digests remain stable across omission
        # and persistence round-trips.
        object.__setattr__(member, "included", included_flag)
        if omission_reason:
            object.__setattr__(member, "omission_reason", omission_reason)
        if expansion_handle:
            object.__setattr__(member, "expansion_handle", expansion_handle)
        return member

    def try_include(member: ContextMember, *, force_invariant: bool) -> None:
        nonlocal used_bytes, used_tokens, used_rows, disposition
        next_rows = used_rows + 1
        next_bytes = used_bytes + member.byte_count
        next_tokens = used_tokens + member.token_count
        over = (
            next_rows > budget.max_rows
            or next_bytes > budget.max_bytes
            or next_tokens > budget.max_tokens
            or len(included) >= budget.max_members
        )
        if over and not force_invariant:
            handle = _identity(
                "expand",
                {"member_id": member.member_id, "digest": member.digest},
            )
            omitted.append(
                _mark(
                    member,
                    included_flag=False,
                    omission_reason="budget_overflow",
                    expansion_handle=handle,
                )
            )
            disposition = FrontierDisposition.BUDGET_OVERFLOW
            kind_name = (
                member.kind.value
                if isinstance(member.kind, MemberKind)
                else str(member.kind)
            )
            reasons.append(f"omitted {kind_name} due to budget")
            return
        if over and force_invariant:
            raise DatabaseContextOverflowError(
                "required invariant context exceeds hard budget"
            )
        included.append(_mark(member, included_flag=True))
        used_rows = next_rows
        used_bytes = next_bytes
        used_tokens = next_tokens

    # Always attempt invariant members first (from full set, order preserved).
    invariants = [
        item
        for item in members
        if item.tier is MemberTier.INVARIANT
        or (
            isinstance(item.tier, str)
            and item.tier == MemberTier.INVARIANT.value
        )
    ]
    non_invariants_in_page = [
        item
        for item in page
        if item.tier is not MemberTier.INVARIANT
        and not (
            isinstance(item.tier, str)
            and item.tier == MemberTier.INVARIANT.value
        )
    ]

    for member in invariants:
        try_include(member, force_invariant=True)
    for member in non_invariants_in_page:
        try_include(member, force_invariant=False)

    # Members outside the page window become explicit pagination omissions.
    for member in (*before_page, *remainder_after_page):
        if member.tier is MemberTier.INVARIANT or (
            isinstance(member.tier, str)
            and member.tier == MemberTier.INVARIANT.value
        ):
            continue
        if any(item.member_id == member.member_id for item in included):
            continue
        if any(item.member_id == member.member_id for item in omitted):
            continue
        handle = _identity(
            "expand",
            {"member_id": member.member_id, "digest": member.digest},
        )
        omitted.append(
            _mark(
                member,
                included_flag=False,
                omission_reason="pagination",
                expansion_handle=handle,
            )
        )
        if disposition is FrontierDisposition.EMPTY:
            disposition = FrontierDisposition.PAGINATED
        kind_name = (
            member.kind.value
            if isinstance(member.kind, MemberKind)
            else str(member.kind)
        )
        reasons.append(f"paginated omission of {kind_name}")

    return included, omitted, disposition, reasons


def _dependency_digests(request: TaskContextInput) -> tuple[str, ...]:
    digests: list[str] = []
    for item in request.unmet_dependencies:
        if isinstance(item, Mapping):
            digests.append(
                _identity(
                    "dep",
                    _reject_or_redact_secrets(
                        _strip_noise(dict(item)), reject=True
                    ),
                )
            )
        else:
            digests.append(_identity("dep", {"id": _text(item, "dependency")}))
    return tuple(dict.fromkeys(digests))


def _check_staleness(request: TaskContextInput) -> None:
    if request.expected_tree_id and request.expected_tree_id != request.tree_id:
        raise DatabaseContextStaleError(
            "tree_id drifted from expected binding",
            reason_code="stale_tree",
        )
    if (
        request.expected_policy_digest
        and request.expected_policy_digest != request.policy_digest
    ):
        raise DatabaseContextStaleError(
            "policy_digest drifted from expected binding",
            reason_code="stale_policy",
        )
    if request.expected_dependency_digests:
        actual = set(_dependency_digests(request))
        expected = set(request.expected_dependency_digests)
        if actual != expected:
            raise DatabaseContextStaleError(
                "dependency digests drifted from expected binding",
                reason_code="stale_dependencies",
            )


def build_database_context_manifest(
    request: TaskContextInput,
) -> DatabaseContextManifest:
    """Build one content-addressed bounded context manifest."""

    _check_staleness(request)
    candidates = _candidate_members(request)
    included, omitted, disposition, reasons = _apply_budgets(
        candidates, request.budget
    )
    all_members = tuple(
        sorted(
            (*included, *omitted),
            key=lambda item: (item.ordinal, item.member_id),
        )
    )
    has_more = bool(omitted)
    if not omitted:
        disposition = FrontierDisposition.EMPTY
    frontier = LLMContextFrontier(
        frontier_id="",
        disposition=disposition,
        omitted_member_ids=tuple(item.member_id for item in omitted),
        omitted_kinds=tuple(
            sorted(
                {
                    item.kind.value
                    if isinstance(item.kind, MemberKind)
                    else str(item.kind)
                    for item in omitted
                }
            )
        ),
        expansion_handles=tuple(
            item.expansion_handle for item in omitted if item.expansion_handle
        ),
        reasons=tuple(dict.fromkeys(reasons)),
        page_offset=request.budget.page_offset,
        page_size=request.budget.page_size,
        has_more=has_more,
    )
    if omitted and disposition is FrontierDisposition.BUDGET_OVERFLOW:
        completeness = Completeness.OVERFLOW
    elif omitted:
        completeness = Completeness.PARTIAL_WITH_FRONTIER
    else:
        completeness = Completeness.COMPLETE

    return DatabaseContextManifest(
        manifest_cid="",
        task_cid=request.task_cid,
        repository_id=request.repository_id,
        tree_id=request.tree_id,
        policy_id=request.policy_id,
        policy_digest=request.policy_digest,
        schema_revision=request.schema_revision,
        members=all_members,
        frontier=frontier,
        completeness=completeness,
        budget=request.budget,
        dependency_digests=_dependency_digests(request),
        task_revision=request.task_revision,
        plan_cid=request.plan_cid,
        goal_cid=request.goal_cid,
    )


def build_context_delta(
    prior: DatabaseContextManifest,
    current: DatabaseContextManifest,
) -> ContextDelta:
    """Build a bounded delta between two manifests."""

    if (
        prior.repository_id != current.repository_id
        or prior.tree_id != current.tree_id
    ):
        raise DatabaseContextStaleError(
            "cannot delta across repository/tree boundary",
            reason_code="stale_tree",
        )
    if prior.policy_digest != current.policy_digest:
        raise DatabaseContextStaleError(
            "cannot delta across policy boundary",
            reason_code="stale_policy",
        )

    prior_map = {
        item.member_id: item for item in prior.included_members()
    }
    current_map = {
        item.member_id: item for item in current.included_members()
    }
    added: list[ContextMember] = []
    changed: list[ContextMember] = []
    unchanged: list[str] = []
    for member_id, member in current_map.items():
        previous = prior_map.get(member_id)
        if previous is None:
            added.append(member)
        elif previous.digest != member.digest:
            changed.append(member)
        else:
            unchanged.append(member_id)
    removed = tuple(
        member_id
        for member_id in prior_map
        if member_id not in current_map
    )

    invalidations: list[str] = []
    if prior.task_revision != current.task_revision:
        invalidations.append(InvalidationKind.TASK_REVISION.value)
    if set(prior.dependency_digests) != set(current.dependency_digests):
        invalidations.append(InvalidationKind.DEPENDENCY.value)
    if prior.schema_revision != current.schema_revision:
        invalidations.append(InvalidationKind.SCHEMA.value)
    if added or changed or removed:
        invalidations.append(InvalidationKind.EVIDENCE.value)

    return ContextDelta(
        delta_id="",
        from_manifest_cid=prior.manifest_cid,
        to_manifest_cid=current.manifest_cid,
        added=tuple(added),
        removed_member_ids=removed,
        changed=tuple(changed),
        unchanged_member_ids=tuple(unchanged),
        invalidations=tuple(dict.fromkeys(invalidations)),
    )


def compile_manifest_to_capsule(
    manifest: DatabaseContextManifest,
    *,
    budget: ContextBudget | None = None,
    stage: str = "implementation",
    caller: str = "database-context",
) -> ContextCapsule:
    """Compose a ContextCapsule via the existing ContextCompiler boundary."""

    effective_budget = budget or ContextBudget(
        max_input_tokens=max(256, manifest.budget.max_tokens),
        reserved_output_tokens=128,
        reserved_tool_tokens=64,
        max_items=max(16, manifest.budget.max_members),
        max_item_bytes=16_384,
        max_serialized_bytes=max(
            65_536, manifest.budget.max_bytes * 4
        ),
        max_depth=8,
        max_text_bytes=8_192,
    )
    evidence: list[ContextReference] = []
    for item in manifest.included_members():
        if item.kind is MemberKind.TASK:
            continue
        evidence.append(
            ContextReference(
                reference_id=item.member_id,
                kind=item.kind.value
                if isinstance(item.kind, MemberKind)
                else str(item.kind),
                tier=(
                    ContextTier.INVARIANT
                    if item.tier is MemberTier.INVARIANT
                    else ContextTier.EVIDENCE
                ),
                referenced_content_id=item.digest,
                repository_id=manifest.repository_id,
                tree_id=manifest.tree_id,
                path=item.path,
                summary=item.summary,
                byte_count=item.byte_count,
                token_count=item.token_count,
                metadata={
                    "required": item.tier is MemberTier.INVARIANT,
                    "digest": item.digest,
                    "data_label": UNTRUSTED_DATA_LABEL,
                },
            )
        )

    task_member = next(
        (
            item
            for item in manifest.included_members()
            if item.kind is MemberKind.TASK
        ),
        None,
    )
    goal = {
        "id": manifest.goal_cid or manifest.task_cid,
        "summary": (task_member.summary if task_member else manifest.task_cid),
    }
    authority = {
        "mode": "proposal",
        "policy_id": manifest.policy_id,
        "policy_digest": manifest.policy_digest,
    }
    scope = {
        "paths": sorted(
            {
                item.path
                for item in manifest.included_members()
                if item.path
            }
        ),
        "task_cid": manifest.task_cid,
    }
    acceptance = {
        "manifest_cid": manifest.manifest_cid,
        "completeness": manifest.completeness.value
        if isinstance(manifest.completeness, Completeness)
        else str(manifest.completeness),
        "frontier_explicit": manifest.frontier.is_explicit,
        "frontier_id": manifest.frontier.frontier_id,
        "cannot_include_secrets": True,
        "omitted_member_ids": list(manifest.frontier.omitted_member_ids),
    }
    compiler = ContextCompiler(effective_budget)
    result = compiler.compile(
        repository_id=manifest.repository_id,
        tree_id=manifest.tree_id,
        objective_id=manifest.goal_cid or manifest.task_cid,
        objective_revision=manifest.task_revision or manifest.manifest_cid,
        policy_id=manifest.policy_id,
        policy_revision=manifest.policy_digest,
        caller=caller,
        stage=stage,
        goal=goal,
        authority=authority,
        scope=scope,
        acceptance=acceptance,
        evidence=tuple(evidence),
    )
    return result.capsule


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class DatabaseContextStore:
    """Persist and query database context manifests in DuckDB.

    Interface: ``DatabaseContextStore@1`` (local store surface).
    """

    INTERFACE: Final[str] = DATABASE_CONTEXT_STORE_INTERFACE
    SCHEMA: Final[str] = DATABASE_CONTEXT_MANIFEST_SCHEMA

    def __init__(
        self,
        database_path: Path | str,
        *,
        policy_id: str = DEFAULT_POLICY_ID,
    ) -> None:
        if not duckdb_available():
            raise DuckDBUnavailableError(
                "DuckDB is required for DatabaseContextStore; install the "
                "optional duckdb dependency"
            )
        self._path = Path(database_path)
        self._policy_id = _text(policy_id or DEFAULT_POLICY_ID, "policy_id")
        self._connection: Any | None = None
        self._lock = threading.RLock()
        self._closed = True

    @property
    def database_path(self) -> Path:
        return self._path

    @property
    def policy_id(self) -> str:
        return self._policy_id

    @property
    def is_open(self) -> bool:
        return not self._closed and self._connection is not None

    def open(self) -> "DatabaseContextStore":
        with self._lock:
            if self.is_open:
                return self
            self._path.parent.mkdir(parents=True, exist_ok=True)
            connection = open_duckdb_connection(self._path)
            for statement in _split_sql_statements(_BOOKKEEPING_SQL):
                connection.execute(statement)
            for key, value in (
                ("interface", DATABASE_CONTEXT_STORE_INTERFACE),
                ("schema", DATABASE_CONTEXT_MANIFEST_SCHEMA),
                ("policy_id", self._policy_id),
                ("authority", AUTHORITY_CLASS),
                ("producer_id", PRODUCER_ID),
            ):
                connection.execute(
                    """
                    INSERT OR REPLACE INTO database_context_metadata(key, value)
                    VALUES (?, ?)
                    """,
                    [key, value],
                )
            self._connection = connection
            self._closed = False
            return self

    def close(self) -> None:
        with self._lock:
            connection = self._connection
            self._connection = None
            self._closed = True
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass

    def __enter__(self) -> "DatabaseContextStore":
        return self.open()

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _require(self) -> Any:
        if not self.is_open or self._connection is None:
            raise DatabaseContextNotOpenError()
        return self._connection

    def metadata(self) -> dict[str, str]:
        connection = self._require()
        rows = connection.execute(
            "SELECT key, value FROM database_context_metadata ORDER BY key"
        ).fetchall()
        result: dict[str, str] = {}
        for row in rows:
            mapping = _row_mapping(row)
            if mapping:
                result[str(mapping["key"])] = str(mapping["value"])
            else:
                result[str(row[0])] = str(row[1])
        return result

    def build(
        self,
        request: TaskContextInput,
        *,
        persist: bool = True,
    ) -> DatabaseContextManifest:
        """Build a manifest and optionally persist it."""

        if request.policy_id != self._policy_id and request.policy_id:
            # Allow request override but keep store default for metadata.
            pass
        manifest = build_database_context_manifest(request)
        if persist:
            self.persist_manifest(manifest)
        return manifest

    def persist_manifest(self, manifest: DatabaseContextManifest) -> None:
        connection = self._require()
        body = _canonical_json(manifest.to_dict())
        if len(body.encode("utf-8")) > MAX_BODY_JSON_BYTES:
            raise DatabaseContextBoundsError(
                f"manifest body exceeds {MAX_BODY_JSON_BYTES} bytes"
            )
        with self._lock:
            connection.execute(
                """
                INSERT OR REPLACE INTO context_manifests(
                    manifest_cid, task_cid, schema_revision,
                    repository_tree_id, policy_digest, created_at, body_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    manifest.manifest_cid,
                    manifest.task_cid,
                    int(manifest.schema_revision),
                    manifest.tree_id,
                    manifest.policy_digest,
                    "",  # wall-clock intentionally blank for identity stability
                    body,
                ],
            )
            connection.execute(
                "DELETE FROM context_members WHERE manifest_cid = ?",
                [manifest.manifest_cid],
            )
            for member in manifest.members:
                connection.execute(
                    """
                    INSERT INTO context_members(
                        manifest_cid, ordinal, member_kind, member_id,
                        digest, included, byte_count, token_count, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        manifest.manifest_cid,
                        int(member.ordinal),
                        member.kind.value
                        if isinstance(member.kind, MemberKind)
                        else str(member.kind),
                        member.member_id,
                        member.digest,
                        1 if member.included else 0,
                        int(member.byte_count),
                        int(member.token_count),
                        _canonical_json(member.to_dict()),
                    ],
                )
            connection.execute(
                """
                INSERT OR REPLACE INTO llm_context_frontiers(
                    frontier_id, manifest_cid, disposition,
                    omitted_count, body_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                [
                    manifest.frontier.frontier_id,
                    manifest.manifest_cid,
                    manifest.frontier.disposition.value
                    if isinstance(
                        manifest.frontier.disposition, FrontierDisposition
                    )
                    else str(manifest.frontier.disposition),
                    len(manifest.frontier.omitted_member_ids),
                    _canonical_json(manifest.frontier.to_dict()),
                ],
            )
            commit = getattr(connection, "commit", None)
            if callable(commit):
                try:
                    commit()
                except Exception:
                    pass

    def persist_delta(self, delta: ContextDelta) -> None:
        connection = self._require()
        body = _canonical_json(delta.to_dict())
        with self._lock:
            connection.execute(
                """
                INSERT OR REPLACE INTO context_deltas(
                    delta_id, from_manifest_cid, to_manifest_cid,
                    created_at, body_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                [
                    delta.delta_id,
                    delta.from_manifest_cid,
                    delta.to_manifest_cid,
                    "",
                    body,
                ],
            )
            commit = getattr(connection, "commit", None)
            if callable(commit):
                try:
                    commit()
                except Exception:
                    pass

    def get_manifest(self, manifest_cid: str) -> DatabaseContextManifest | None:
        connection = self._require()
        row = connection.execute(
            """
            SELECT body_json FROM context_manifests
            WHERE manifest_cid = ?
            """,
            [_text(manifest_cid, "manifest_cid")],
        ).fetchone()
        if row is None:
            return None
        mapping = _row_mapping(row)
        body_json = (
            str(mapping.get("body_json") or "")
            if mapping
            else str(row[0])
        )
        payload = json.loads(body_json)
        return manifest_from_dict(payload)

    def list_manifests_for_task(
        self, task_cid: str, *, limit: int = 32
    ) -> tuple[str, ...]:
        connection = self._require()
        limit = _positive_int(int(limit), "limit", maximum=MAX_PAGE_SIZE)
        rows = connection.execute(
            """
            SELECT manifest_cid FROM context_manifests
            WHERE task_cid = ?
            ORDER BY manifest_cid
            LIMIT ?
            """,
            [_text(task_cid, "task_cid"), limit],
        ).fetchall()
        result: list[str] = []
        for row in rows:
            mapping = _row_mapping(row)
            if mapping:
                result.append(str(mapping["manifest_cid"]))
            else:
                result.append(str(row[0]))
        return tuple(result)

    def page_members(
        self,
        manifest_cid: str,
        *,
        offset: int = 0,
        limit: int = DEFAULT_PAGE_SIZE,
        included_only: bool = True,
    ) -> tuple[ContextMember, ...]:
        """Paginate persisted members for progressive disclosure."""

        connection = self._require()
        offset = _nonneg_int(int(offset), "offset")
        limit = _positive_int(int(limit), "limit", maximum=MAX_PAGE_SIZE)
        sql = """
            SELECT body_json FROM context_members
            WHERE manifest_cid = ?
        """
        params: list[Any] = [_text(manifest_cid, "manifest_cid")]
        if included_only:
            sql += " AND included = 1"
        sql += " ORDER BY ordinal LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        rows = connection.execute(sql, params).fetchall()
        members: list[ContextMember] = []
        for row in rows:
            mapping = _row_mapping(row)
            body_json = (
                str(mapping.get("body_json") or "")
                if mapping
                else str(row[0])
            )
            payload = json.loads(body_json)
            members.append(member_from_dict(payload))
        return tuple(members)


def member_from_dict(payload: Mapping[str, Any]) -> ContextMember:
    return ContextMember(
        member_id=str(payload.get("member_id") or ""),
        kind=str(payload.get("kind") or ""),
        digest=str(payload.get("digest") or ""),
        tier=str(payload.get("tier") or MemberTier.EVIDENCE.value),
        summary=str(payload.get("summary") or ""),
        path=str(payload.get("path") or ""),
        included=bool(payload.get("included", True)),
        byte_count=int(payload.get("byte_count") or 0),
        token_count=int(payload.get("token_count") or 0),
        ordinal=int(payload.get("ordinal") or 0),
        payload=payload.get("payload") or {},
        omission_reason=str(payload.get("omission_reason") or ""),
        expansion_handle=str(payload.get("expansion_handle") or ""),
    )


def frontier_from_dict(payload: Mapping[str, Any]) -> LLMContextFrontier:
    return LLMContextFrontier(
        frontier_id=str(payload.get("frontier_id") or ""),
        disposition=str(payload.get("disposition") or FrontierDisposition.EMPTY.value),
        omitted_member_ids=tuple(payload.get("omitted_member_ids") or ()),
        omitted_kinds=tuple(payload.get("omitted_kinds") or ()),
        expansion_handles=tuple(payload.get("expansion_handles") or ()),
        reasons=tuple(payload.get("reasons") or ()),
        page_offset=int(payload.get("page_offset") or 0),
        page_size=int(payload.get("page_size") or DEFAULT_PAGE_SIZE),
        has_more=bool(payload.get("has_more", False)),
        secret_excluded_count=int(payload.get("secret_excluded_count") or 0),
        private_excluded_count=int(payload.get("private_excluded_count") or 0),
    )


def manifest_from_dict(payload: Mapping[str, Any]) -> DatabaseContextManifest:
    budget_payload = payload.get("budget") or {}
    budget = ContextBudgetSpec(
        max_rows=int(budget_payload.get("max_rows") or DEFAULT_MAX_ROWS),
        max_bytes=int(budget_payload.get("max_bytes") or DEFAULT_MAX_BYTES),
        max_tokens=int(budget_payload.get("max_tokens") or DEFAULT_MAX_TOKENS),
        page_size=int(budget_payload.get("page_size") or DEFAULT_PAGE_SIZE),
        page_offset=int(budget_payload.get("page_offset") or 0),
        max_members=int(budget_payload.get("max_members") or MAX_MEMBERS),
    )
    members = tuple(
        member_from_dict(item)
        for item in (payload.get("members") or ())
        if isinstance(item, Mapping)
    )
    frontier_payload = payload.get("frontier") or {}
    frontier = (
        frontier_from_dict(frontier_payload)
        if isinstance(frontier_payload, Mapping)
        else LLMContextFrontier(
            frontier_id="", disposition=FrontierDisposition.EMPTY
        )
    )
    return DatabaseContextManifest(
        manifest_cid=str(payload.get("manifest_cid") or ""),
        task_cid=str(payload.get("task_cid") or ""),
        repository_id=str(payload.get("repository_id") or ""),
        tree_id=str(payload.get("tree_id") or ""),
        policy_id=str(payload.get("policy_id") or DEFAULT_POLICY_ID),
        policy_digest=str(payload.get("policy_digest") or ""),
        schema_revision=int(
            payload.get("schema_revision") or DEFAULT_SCHEMA_REVISION
        ),
        members=members,
        frontier=frontier,
        completeness=str(
            payload.get("completeness") or Completeness.COMPLETE.value
        ),
        budget=budget,
        dependency_digests=tuple(payload.get("dependency_digests") or ()),
        task_revision=str(payload.get("task_revision") or ""),
        plan_cid=str(payload.get("plan_cid") or ""),
        goal_cid=str(payload.get("goal_cid") or ""),
        total_bytes=int(payload.get("total_bytes") or 0),
        total_tokens=int(payload.get("total_tokens") or 0),
        total_rows=int(payload.get("total_rows") or 0),
    )


def open_database_context_store(
    database_path: Path | str,
    *,
    policy_id: str = DEFAULT_POLICY_ID,
) -> DatabaseContextStore:
    """Open a DatabaseContextStore (creates schema on first open)."""

    return DatabaseContextStore(
        database_path, policy_id=policy_id
    ).open()


__all__ = [
    "AUTHORITY_CLASS",
    "CONTEXT_DELTA_INTERFACE",
    "CONTEXT_DELTA_SCHEMA",
    "Completeness",
    "ContextBudgetSpec",
    "ContextDelta",
    "ContextMember",
    "DATABASE_CONTEXT_MANIFEST_INTERFACE",
    "DATABASE_CONTEXT_MANIFEST_SCHEMA",
    "DATABASE_CONTEXT_STORE_INTERFACE",
    "DEFAULT_POLICY_ID",
    "DatabaseContextBoundsError",
    "DatabaseContextError",
    "DatabaseContextIntegrityError",
    "DatabaseContextManifest",
    "DatabaseContextNotOpenError",
    "DatabaseContextOverflowError",
    "DatabaseContextSecretError",
    "DatabaseContextStaleError",
    "DatabaseContextStore",
    "DuckDBUnavailableError",
    "FrontierDisposition",
    "InvalidationKind",
    "LLM_CONTEXT_FRONTIER_INTERFACE",
    "LLM_CONTEXT_FRONTIER_SCHEMA",
    "LLMContextFrontier",
    "MemberKind",
    "MemberTier",
    "PRODUCER_ID",
    "REDACTION_MARKER",
    "TaskContextInput",
    "UNTRUSTED_DATA_LABEL",
    "build_context_delta",
    "build_database_context_manifest",
    "compile_manifest_to_capsule",
    "duckdb_available",
    "frontier_from_dict",
    "manifest_from_dict",
    "member_from_dict",
    "open_database_context_store",
]
