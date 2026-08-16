"""Delta task packets and deterministic-first replay suppression.

DQP-028 / Interfaces: ``DeltaTaskPacket@1``, ``DeterministicFirstDecision@1``
============================================================================

Builds content-addressed, delta-oriented task packets for provider economy:

* Deterministic operators, decision cache, and query hits resolve known work
  **before** any provider dispatch.
* The model-facing packet carries only the unresolved bounded delta plus the
  exact allowed effect scope, validation commands, and proof obligations.
* Repeated unchanged failures open a typed replay circuit until material
  evidence (counterexample, tree, plan, policy, or schema) changes.
* Provider surfaces never receive omitted authority claims or credentials.
* Packet and reply are bound to the exact context CID and effect scope so
  cross-scope reuse is fail-closed.

Evidence subset: packet identity, progressive disclosure, deterministic hit,
cache miss, unchanged reprompt, counterexample, scope/secret escape, context
overflow.

Cold import of this module performs no filesystem, database, network,
provider, or process action. Opening an optional ledger integration is the
first I/O boundary.

Conflict policy: this module owns packet/replay integration. Existing
provider routers remain authority for provider selection; database context
and the provider-call ledger remain authorities for their own contracts.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Final

# ---------------------------------------------------------------------------
# Contract identity
# ---------------------------------------------------------------------------

DELTA_TASK_PACKET_INTERFACE: Final[str] = "DeltaTaskPacket@1"
DETERMINISTIC_FIRST_DECISION_INTERFACE: Final[str] = "DeterministicFirstDecision@1"
PACKET_REPLY_BINDING_INTERFACE: Final[str] = "PacketReplyBinding@1"
DELTA_TASK_PACKET_SERVICE_INTERFACE: Final[str] = "DeltaTaskPacketService@1"

DELTA_TASK_PACKET_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/delta-task-packet@1"
)
DETERMINISTIC_FIRST_DECISION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-first-decision@1"
)
PACKET_REPLY_BINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/packet-reply-binding@1"
)
PROVIDER_FACING_PACKET_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/delta-task-provider-packet@1"
)
CIRCUIT_STATE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/delta-task-replay-circuit@1"
)

DEFAULT_POLICY_ID: Final[str] = "delta-task-packet-policy@1"
DEFAULT_SCHEMA_REVISION: Final[int] = 1
AUTHORITY_CLASS: Final[str] = "derived_evidence"
PRODUCER_ID: Final[str] = "delta-task-packet@1"
REDACTION_MARKER: Final[str] = "secret_material"
UNTRUSTED_DATA_LABEL: Final[str] = "untrusted_repository_data"

MAX_TEXT_BYTES: Final[int] = 4_096
MAX_PATH_BYTES: Final[int] = 4_096
MAX_ID_BYTES: Final[int] = 512
MAX_BODY_JSON_BYTES: Final[int] = 262_144
MAX_DELTA_ITEMS: Final[int] = 512
MAX_EFFECTS: Final[int] = 64
MAX_VALIDATIONS: Final[int] = 64
MAX_PROOFS: Final[int] = 128
MAX_PATHS: Final[int] = 256
MAX_OMISSIONS: Final[int] = 256
DEFAULT_MAX_BYTES: Final[int] = 34_000
DEFAULT_MAX_TOKENS: Final[int] = 8_192
DEFAULT_MAX_ROWS: Final[int] = 128
DEFAULT_RETRY_BUDGET: Final[int] = 2
DEFAULT_CIRCUIT_TTL_MS: Final[int] = 3_600_000
BYTES_PER_TOKEN: Final[int] = 4

# Authority fields that must never be granted to a provider-facing packet.
_PROVIDER_OMITTED_AUTHORITY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "completion_authority",
        "write_authority",
        "semantic_authority",
        "mutation_authority",
        "merge_authority",
        "operator_authority",
        "quack_token",
        "quack_credential",
        "sql_capability",
        "raw_sql",
        "database_credential",
        "provider_credential",
        "api_credential",
    }
)

_SENSITIVE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "authorization",
        "auth_token",
        "client_secret",
        "credential",
        "credentials",
        "github_token",
        "password",
        "passphrase",
        "passwd",
        "private_key",
        "refresh_token",
        "secret",
        "secrets",
        "secret_handle",
        "raw_secret",
        "token",
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
        "prompt",
        "prompt_text",
        "completion",
        "completion_text",
        "messages",
        "private_key",
        "secret",
        "secrets",
        "password",
        "token",
        "api_key",
        "authorization",
        "credential",
        "credentials",
        "quack_token",
        "quack_credential",
    }
)

_SECRET_PATH_MARKERS: Final[tuple[str, ...]] = (
    ".env",
    "credentials.json",
    "secrets.json",
    "id_rsa",
    "id_ed25519",
    ".pem",
    ".key",
    "service-account",
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

DEFAULT_ALLOWED_EFFECTS: Final[tuple[str, ...]] = (
    "inspect_repository",
    "edit_isolated_worktree",
    "run_validation",
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DeltaTaskPacketError(RuntimeError):
    """Base error for delta task packet construction and admission."""

    def __init__(self, message: str, *, reason_code: str = "delta_task_packet") -> None:
        super().__init__(message)
        self.reason_code = reason_code


class DeltaTaskPacketIntegrityError(DeltaTaskPacketError, ValueError):
    def __init__(self, message: str, *, reason_code: str = "integrity") -> None:
        super().__init__(message, reason_code=reason_code)


class DeltaTaskPacketBoundsError(DeltaTaskPacketError, ValueError):
    def __init__(self, message: str, *, reason_code: str = "bounds") -> None:
        super().__init__(message, reason_code=reason_code)


class DeltaTaskPacketSecretError(DeltaTaskPacketError, ValueError):
    """Secret, credential, or private material presented for a model packet."""

    def __init__(
        self, message: str, *, reason_code: str = "secret_material_rejected"
    ) -> None:
        super().__init__(message, reason_code=reason_code)


class DeltaTaskPacketScopeError(DeltaTaskPacketError, ValueError):
    """Effect or path scope escape."""

    def __init__(self, message: str, *, reason_code: str = "scope_escape") -> None:
        super().__init__(message, reason_code=reason_code)


class DeltaTaskPacketAuthorityError(DeltaTaskPacketError, ValueError):
    """Provider-facing packet attempted to carry omitted authority."""

    def __init__(
        self, message: str, *, reason_code: str = "authority_omitted"
    ) -> None:
        super().__init__(message, reason_code=reason_code)


class DeltaTaskPacketOverflowError(DeltaTaskPacketError, ValueError):
    def __init__(
        self, message: str, *, reason_code: str = "context_overflow"
    ) -> None:
        super().__init__(message, reason_code=reason_code)


class DeltaTaskPacketSuppressedError(DeltaTaskPacketError):
    """Unchanged failure is under active replay suppression."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "replay_suppressed",
        circuit_id: str = "",
    ) -> None:
        super().__init__(message, reason_code=reason_code)
        self.circuit_id = circuit_id


class DeltaTaskPacketStaleError(DeltaTaskPacketError, ValueError):
    def __init__(self, message: str, *, reason_code: str = "stale_binding") -> None:
        super().__init__(message, reason_code=reason_code)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DeterministicAction(str, Enum):
    """Outcome of the deterministic-first gate."""

    RESOLVE_HIT = "resolve_hit"
    DISPATCH_PROVIDER = "dispatch_provider"
    SUPPRESS_REPLAY = "suppress_replay"
    REJECT = "reject"

    @classmethod
    def coerce(cls, value: Any) -> "DeterministicAction":
        if isinstance(value, cls):
            return value
        text = str(value or "").strip().casefold()
        aliases = {
            "hit": cls.RESOLVE_HIT,
            "deterministic_hit": cls.RESOLVE_HIT,
            "cache_hit": cls.RESOLVE_HIT,
            "miss": cls.DISPATCH_PROVIDER,
            "cache_miss": cls.DISPATCH_PROVIDER,
            "dispatch": cls.DISPATCH_PROVIDER,
            "suppress": cls.SUPPRESS_REPLAY,
            "suppressed": cls.SUPPRESS_REPLAY,
            "replay_suppress": cls.SUPPRESS_REPLAY,
        }
        if text in aliases:
            return aliases[text]
        try:
            return cls(text)
        except ValueError as exc:
            raise DeltaTaskPacketIntegrityError(
                f"unknown deterministic action: {value!r}"
            ) from exc


class DecisionSource(str, Enum):
    """How a deterministic-first decision was produced."""

    OPERATOR = "operator"
    DECISION_CACHE = "decision_cache"
    QUERY = "query"
    CIRCUIT = "circuit"
    RESIDUAL = "residual"
    POLICY = "policy"

    @classmethod
    def coerce(cls, value: Any) -> "DecisionSource":
        if isinstance(value, cls):
            return value
        text = str(value or "").strip().casefold()
        try:
            return cls(text)
        except ValueError as exc:
            raise DeltaTaskPacketIntegrityError(
                f"unknown decision source: {value!r}"
            ) from exc


class Completeness(str, Enum):
    COMPLETE = "complete"
    PARTIAL_WITH_FRONTIER = "partial_with_frontier"
    OVERFLOW = "overflow"

    @classmethod
    def coerce(cls, value: Any) -> "Completeness":
        if isinstance(value, cls):
            return value
        text = str(value or "").strip().casefold()
        try:
            return cls(text)
        except ValueError as exc:
            raise DeltaTaskPacketIntegrityError(
                f"unknown completeness: {value!r}"
            ) from exc


class FrontierDisposition(str, Enum):
    EMPTY = "empty"
    PAGINATED = "paginated"
    BUDGET_OVERFLOW = "budget_overflow"
    DETERMINISTIC_RESOLVED = "deterministic_resolved"

    @classmethod
    def coerce(cls, value: Any) -> "FrontierDisposition":
        if isinstance(value, cls):
            return value
        text = str(value or "").strip().casefold()
        try:
            return cls(text)
        except ValueError as exc:
            raise DeltaTaskPacketIntegrityError(
                f"unknown frontier disposition: {value!r}"
            ) from exc


class DeltaItemKind(str, Enum):
    TASK = "task"
    DEPENDENCY = "dependency"
    FAILURE = "failure"
    WORKTREE_DELTA = "worktree_delta"
    SYMBOL = "symbol"
    OBLIGATION = "obligation"
    EVIDENCE = "evidence"
    COUNTEREXAMPLE = "counterexample"
    VALIDATION = "validation"
    PROOF = "proof"
    DECISION = "decision"
    EFFECT = "effect"

    @classmethod
    def coerce(cls, value: Any) -> "DeltaItemKind":
        if isinstance(value, cls):
            return value
        text = str(value or "").strip().casefold()
        aliases = {
            "unmet_dependency": cls.DEPENDENCY,
            "dep": cls.DEPENDENCY,
            "worktree": cls.WORKTREE_DELTA,
            "delta": cls.WORKTREE_DELTA,
            "cex": cls.COUNTEREXAMPLE,
            "validation_command": cls.VALIDATION,
            "proof_obligation": cls.PROOF,
        }
        if text in aliases:
            return aliases[text]
        try:
            return cls(text)
        except ValueError as exc:
            raise DeltaTaskPacketIntegrityError(
                f"unknown delta item kind: {value!r}"
            ) from exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
        raise DeltaTaskPacketIntegrityError(
            "values must be canonical JSON"
        ) from exc


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _content_digest(value: Any) -> str:
    return "sha256:" + _sha256_hex(_canonical_json(value).encode("utf-8"))


def _identity(prefix: str, value: Any) -> str:
    return f"{prefix}:" + _content_digest(value)


def _content_cid(value: Any) -> str:
    digest = _sha256_hex(_canonical_json(value).encode("utf-8"))
    return f"baguqeera{digest[:52]}"


def _text(value: Any, name: str, *, required: bool = True, maximum: int = MAX_TEXT_BYTES) -> str:
    text = str(value or "").strip()
    if "\x00" in text:
        raise DeltaTaskPacketIntegrityError(f"{name} contains NUL")
    if required and not text:
        raise DeltaTaskPacketIntegrityError(f"{name} is required")
    if len(text.encode("utf-8")) > maximum:
        raise DeltaTaskPacketBoundsError(
            f"{name} exceeds {maximum} UTF-8 bytes"
        )
    return text


def _nonneg_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DeltaTaskPacketBoundsError(
            f"{name} must be a non-negative integer"
        )
    return value


def _positive_int(value: Any, name: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DeltaTaskPacketBoundsError(
            f"{name} must be a positive integer"
        )
    if maximum is not None and value > maximum:
        raise DeltaTaskPacketBoundsError(
            f"{name} exceeds maximum {maximum}"
        )
    return value


def _normalized_key(value: str) -> str:
    return value.strip().casefold().replace("-", "_").replace(" ", "_")


def _is_authority_key(key: str) -> bool:
    return _normalized_key(key) in _PROVIDER_OMITTED_AUTHORITY_KEYS


def _is_sensitive_key(key: str) -> bool:
    normalized = _normalized_key(key)
    # Authority keys are handled separately so hard-zero flags may appear.
    if normalized in _PROVIDER_OMITTED_AUTHORITY_KEYS:
        return False
    if normalized in _SENSITIVE_KEYS or normalized in _BODY_FORBIDDEN_KEYS:
        return True
    if normalized.endswith("_secret") or normalized.endswith("_password"):
        return True
    if normalized.endswith("_api_key") or normalized.endswith("_private_key"):
        return True
    if normalized.endswith("_token") and normalized not in {
        "task_token_budget",
        "token_estimate",
        "token_count",
        "max_tokens",
    }:
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


def _repo_path(value: Any, *, required: bool = False) -> str:
    raw = str(value or "").strip().replace("\\", "/")
    while raw.startswith("./"):
        raw = raw[2:]
    if not raw:
        if required:
            raise DeltaTaskPacketIntegrityError("repository path is required")
        return ""
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts or "\x00" in raw:
        raise DeltaTaskPacketScopeError(
            f"repository path escapes its root: {value!r}",
            reason_code="path_escape",
        )
    normalized = path.as_posix()
    if len(normalized.encode("utf-8")) > MAX_PATH_BYTES:
        raise DeltaTaskPacketBoundsError(
            f"path exceeds {MAX_PATH_BYTES} bytes: {normalized}"
        )
    if _looks_like_secret_path(normalized):
        raise DeltaTaskPacketSecretError(
            f"secret-bearing path excluded: {normalized}"
        )
    return normalized


def _strip_noise(value: Any) -> Any:
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
    depth: int = 0,
) -> Any:
    if depth > 12:
        raise DeltaTaskPacketBoundsError("payload exceeds recursion depth")
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise DeltaTaskPacketIntegrityError(
                    "packet object keys must be strings"
                )
            key_path = f"{path}.{key}" if path else key
            normalized = _normalized_key(key)
            if _is_authority_key(key):
                # Only hard-zero / empty authority projections are allowed.
                # "omitted_authority" is a documentation list, not a grant.
                if normalized == "omitted_authority":
                    result[key] = _reject_or_redact_secrets(
                        item, path=key_path, reject=reject, depth=depth + 1
                    )
                    continue
                if item not in (False, None, 0, ""):
                    if reject:
                        raise DeltaTaskPacketAuthorityError(
                            f"omitted authority must not reach provider: {key_path}",
                            reason_code="authority_omitted",
                        )
                    result[key] = False
                    continue
                result[key] = False if item in (True, False, None) else item
                continue
            if _is_sensitive_key(key) or normalized in _BODY_FORBIDDEN_KEYS:
                if reject:
                    raise DeltaTaskPacketSecretError(
                        f"secret or private field excluded: {key_path}"
                    )
                result[key] = REDACTION_MARKER
                continue
            result[key] = _reject_or_redact_secrets(
                item, path=key_path, reject=reject, depth=depth + 1
            )
        return result
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        return [
            _reject_or_redact_secrets(
                item, path=f"{path}[{index}]", reject=reject, depth=depth + 1
            )
            for index, item in enumerate(value)
        ]
    if isinstance(value, str):
        if _text_contains_secret_pattern(value):
            if reject:
                raise DeltaTaskPacketSecretError(
                    f"secret pattern excluded at {path or 'value'}"
                )
            return REDACTION_MARKER
        return value
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        raise DeltaTaskPacketIntegrityError(
            "floating-point values are not canonical packet material"
        )
    raise DeltaTaskPacketIntegrityError(
        f"unsupported packet value type: {type(value).__name__}"
    )


def _estimate_tokens(value: Any) -> int:
    encoded = _canonical_json(value).encode("utf-8")
    return max(1, (len(encoded) + BYTES_PER_TOKEN - 1) // BYTES_PER_TOKEN)


def _byte_count(value: Any) -> int:
    return len(_canonical_json(value).encode("utf-8"))


def _now_ms() -> int:
    return int(time.time() * 1000)


def _freeze_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


def _normalize_effect(value: Any) -> str:
    text = _text(value, "allowed_effect", maximum=MAX_ID_BYTES)
    normalized = text.casefold().replace("-", "_").replace(" ", "_")
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", normalized):
        raise DeltaTaskPacketScopeError(
            f"invalid allowed effect: {value!r}",
            reason_code="invalid_effect",
        )
    return normalized


def _normalize_effects(values: Sequence[Any] | None) -> tuple[str, ...]:
    if not values:
        return DEFAULT_ALLOWED_EFFECTS
    effects: list[str] = []
    for item in values:
        effect = _normalize_effect(item)
        if effect not in effects:
            effects.append(effect)
        if len(effects) > MAX_EFFECTS:
            raise DeltaTaskPacketBoundsError(
                f"allowed_effects exceeds {MAX_EFFECTS}"
            )
    return tuple(effects)


def _normalize_paths(values: Sequence[Any] | None) -> tuple[str, ...]:
    paths: list[str] = []
    for item in values or ():
        path = _repo_path(item, required=True)
        if path not in paths:
            paths.append(path)
        if len(paths) > MAX_PATHS:
            raise DeltaTaskPacketBoundsError(f"paths exceed {MAX_PATHS}")
    return tuple(paths)


def _normalize_commands(values: Sequence[Any] | None) -> tuple[str, ...]:
    commands: list[str] = []
    for item in values or ():
        if isinstance(item, Mapping):
            command = _text(
                item.get("command") or item.get("validation_command") or "",
                "validation_command",
            )
        else:
            command = _text(item, "validation_command")
        if command not in commands:
            commands.append(command)
        if len(commands) > MAX_VALIDATIONS:
            raise DeltaTaskPacketBoundsError(
                f"validation_commands exceed {MAX_VALIDATIONS}"
            )
    return tuple(commands)


def _normalize_ids(
    values: Sequence[Any] | None, *, name: str, limit: int
) -> tuple[str, ...]:
    ids: list[str] = []
    for item in values or ():
        if isinstance(item, Mapping):
            raw = (
                item.get("id")
                or item.get(f"{name}_id")
                or item.get("obligation_id")
                or item.get("proof_id")
                or item.get("evidence_id")
                or ""
            )
            identifier = _text(raw, name, maximum=MAX_ID_BYTES)
        else:
            identifier = _text(item, name, maximum=MAX_ID_BYTES)
        if identifier not in ids:
            ids.append(identifier)
        if len(ids) > limit:
            raise DeltaTaskPacketBoundsError(f"{name} exceeds {limit}")
    return tuple(ids)


def _as_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    raise DeltaTaskPacketIntegrityError("expected a mapping")


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PacketBudget:
    """Hard row / byte / token budget for a delta task packet."""

    max_bytes: int = DEFAULT_MAX_BYTES
    max_tokens: int = DEFAULT_MAX_TOKENS
    max_rows: int = DEFAULT_MAX_ROWS
    max_items: int = MAX_DELTA_ITEMS

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_bytes",
            _positive_int(int(self.max_bytes), "max_bytes", maximum=1_048_576),
        )
        object.__setattr__(
            self,
            "max_tokens",
            _positive_int(int(self.max_tokens), "max_tokens", maximum=262_144),
        )
        object.__setattr__(
            self,
            "max_rows",
            _positive_int(int(self.max_rows), "max_rows", maximum=MAX_DELTA_ITEMS),
        )
        object.__setattr__(
            self,
            "max_items",
            _positive_int(int(self.max_items), "max_items", maximum=MAX_DELTA_ITEMS),
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "max_bytes": self.max_bytes,
            "max_tokens": self.max_tokens,
            "max_rows": self.max_rows,
            "max_items": self.max_items,
        }


@dataclass(frozen=True)
class DeltaItem:
    """One bounded unresolved delta member admitted into a packet."""

    item_id: str
    kind: DeltaItemKind
    digest: str
    summary: str = ""
    path: str = ""
    payload: Mapping[str, Any] = field(default_factory=dict)
    included: bool = True
    omission_reason: str = ""
    expansion_handle: str = ""
    resolved_deterministically: bool = False
    ordinal: int = 0

    def __post_init__(self) -> None:
        kind = DeltaItemKind.coerce(self.kind)
        object.__setattr__(self, "kind", kind)
        payload = _reject_or_redact_secrets(
            _strip_noise(_as_mapping(self.payload)), reject=True
        )
        object.__setattr__(self, "payload", _freeze_mapping(payload))
        summary = _text(self.summary, "summary", required=False)
        path = _repo_path(self.path, required=False) if self.path else ""
        object.__setattr__(self, "summary", summary)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "included", bool(self.included))
        object.__setattr__(
            self,
            "omission_reason",
            _text(self.omission_reason, "omission_reason", required=False),
        )
        object.__setattr__(
            self,
            "expansion_handle",
            _text(self.expansion_handle, "expansion_handle", required=False),
        )
        object.__setattr__(
            self,
            "resolved_deterministically",
            bool(self.resolved_deterministically),
        )
        object.__setattr__(
            self, "ordinal", _nonneg_int(int(self.ordinal), "ordinal")
        )
        claimed = str(self.digest or "").strip()
        computed = _content_digest(
            {
                "kind": kind.value,
                "summary": summary,
                "path": path,
                "payload": payload,
            }
        )
        object.__setattr__(self, "digest", claimed or computed)
        item_id = str(self.item_id or "").strip()
        if not item_id:
            item_id = _identity(
                "delta-item",
                {"kind": kind.value, "digest": self.digest, "ordinal": self.ordinal},
            )
        object.__setattr__(
            self, "item_id", _text(item_id, "item_id", maximum=MAX_ID_BYTES + 64)
        )

    @property
    def byte_count(self) -> int:
        return _byte_count(self.to_dict())

    @property
    def token_count(self) -> int:
        return _estimate_tokens(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "kind": self.kind.value
            if isinstance(self.kind, DeltaItemKind)
            else str(self.kind),
            "digest": self.digest,
            "summary": self.summary,
            "path": self.path,
            "payload": dict(self.payload),
            "included": self.included,
            "omission_reason": self.omission_reason,
            "expansion_handle": self.expansion_handle,
            "resolved_deterministically": self.resolved_deterministically,
            "ordinal": self.ordinal,
        }


@dataclass(frozen=True)
class PacketFrontier:
    """Explicit progressive-disclosure frontier for omitted residual work."""

    disposition: FrontierDisposition | str = FrontierDisposition.EMPTY
    omitted_item_ids: tuple[str, ...] = ()
    omitted_kinds: tuple[str, ...] = ()
    expansion_handles: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    has_more: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "disposition", FrontierDisposition.coerce(self.disposition)
        )
        object.__setattr__(
            self,
            "omitted_item_ids",
            tuple(
                dict.fromkeys(
                    _text(item, "omitted_item_id", maximum=MAX_ID_BYTES + 64)
                    for item in self.omitted_item_ids
                    if str(item).strip()
                )
            )[:MAX_OMISSIONS],
        )
        object.__setattr__(
            self,
            "omitted_kinds",
            tuple(
                dict.fromkeys(
                    str(item).strip().casefold()
                    for item in self.omitted_kinds
                    if str(item).strip()
                )
            ),
        )
        object.__setattr__(
            self,
            "expansion_handles",
            tuple(
                dict.fromkeys(
                    _text(item, "expansion_handle", required=False)
                    for item in self.expansion_handles
                    if str(item).strip()
                )
            )[:MAX_OMISSIONS],
        )
        object.__setattr__(
            self,
            "reasons",
            tuple(
                dict.fromkeys(
                    _text(item, "reason", required=False)
                    for item in self.reasons
                    if str(item).strip()
                )
            ),
        )
        object.__setattr__(self, "has_more", bool(self.has_more))

    @property
    def is_explicit(self) -> bool:
        disposition = self.disposition
        if not isinstance(disposition, FrontierDisposition):
            disposition = FrontierDisposition.coerce(disposition)
        return disposition is not FrontierDisposition.EMPTY or bool(
            self.omitted_item_ids
        )

    @property
    def omitted_count(self) -> int:
        return len(self.omitted_item_ids)

    def to_dict(self) -> dict[str, Any]:
        disposition = self.disposition
        if isinstance(disposition, FrontierDisposition):
            disposition_value = disposition.value
        else:
            disposition_value = str(disposition)
        return {
            "disposition": disposition_value,
            "omitted_item_ids": list(self.omitted_item_ids),
            "omitted_kinds": list(self.omitted_kinds),
            "expansion_handles": list(self.expansion_handles),
            "reasons": list(self.reasons),
            "has_more": self.has_more,
            "omitted_count": self.omitted_count,
        }


@dataclass(frozen=True)
class EffectScope:
    """Exact allowed effects and write paths bound into a packet/reply."""

    allowed_effects: tuple[str, ...] = DEFAULT_ALLOWED_EFFECTS
    write_paths: tuple[str, ...] = ()
    repository_id: str = ""
    tree_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "allowed_effects", _normalize_effects(self.allowed_effects)
        )
        object.__setattr__(self, "write_paths", _normalize_paths(self.write_paths))
        object.__setattr__(
            self,
            "repository_id",
            _text(self.repository_id, "repository_id", required=False),
        )
        object.__setattr__(
            self, "tree_id", _text(self.tree_id, "tree_id", required=False)
        )

    @property
    def scope_digest(self) -> str:
        return _content_digest(
            {
                "allowed_effects": list(self.allowed_effects),
                "write_paths": list(self.write_paths),
                "repository_id": self.repository_id,
                "tree_id": self.tree_id,
            }
        )

    def permits_effect(self, effect: str) -> bool:
        return _normalize_effect(effect) in self.allowed_effects

    def permits_path(self, path: str) -> bool:
        candidate = _repo_path(path, required=True)
        if not self.write_paths:
            return True
        return any(
            candidate == allowed
            or candidate.startswith(allowed.rstrip("/") + "/")
            for allowed in self.write_paths
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed_effects": list(self.allowed_effects),
            "write_paths": list(self.write_paths),
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "scope_digest": self.scope_digest,
        }


@dataclass(frozen=True)
class ValidationProofRequirements:
    """Validation commands and proof obligations preserved across resolution."""

    validation_commands: tuple[str, ...] = ()
    proof_obligations: tuple[str, ...] = ()
    acceptance_ids: tuple[str, ...] = ()
    require_validation: bool = True
    require_proof: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "validation_commands",
            _normalize_commands(self.validation_commands),
        )
        object.__setattr__(
            self,
            "proof_obligations",
            _normalize_ids(
                self.proof_obligations, name="proof_obligation", limit=MAX_PROOFS
            ),
        )
        object.__setattr__(
            self,
            "acceptance_ids",
            _normalize_ids(
                self.acceptance_ids, name="acceptance", limit=MAX_PROOFS
            ),
        )
        object.__setattr__(self, "require_validation", bool(self.require_validation))
        object.__setattr__(self, "require_proof", bool(self.require_proof))
        if self.require_validation and not self.validation_commands:
            raise DeltaTaskPacketIntegrityError(
                "validation requirements demand at least one command"
            )
        if self.require_proof and not self.proof_obligations:
            raise DeltaTaskPacketIntegrityError(
                "proof requirements demand at least one obligation"
            )

    @property
    def requirements_digest(self) -> str:
        return _content_digest(
            {
                "validation_commands": list(self.validation_commands),
                "proof_obligations": list(self.proof_obligations),
                "acceptance_ids": list(self.acceptance_ids),
                "require_validation": self.require_validation,
                "require_proof": self.require_proof,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "validation_commands": list(self.validation_commands),
            "proof_obligations": list(self.proof_obligations),
            "acceptance_ids": list(self.acceptance_ids),
            "require_validation": self.require_validation,
            "require_proof": self.require_proof,
            "requirements_digest": self.requirements_digest,
        }


@dataclass(frozen=True)
class DeterministicFirstDecision:
    """Deterministic-first gate decision before provider dispatch.

    Interface: ``DeterministicFirstDecision@1``.
    """

    action: DeterministicAction | str
    reason: str
    source: DecisionSource | str = DecisionSource.RESIDUAL
    cache_key: str = ""
    resolved_item_ids: tuple[str, ...] = ()
    residual_item_ids: tuple[str, ...] = ()
    circuit_id: str = ""
    may_dispatch_provider: bool = False
    preserves_validation_proof: bool = True
    evidence_digest: str = ""
    decision_id: str = ""
    schema: str = DETERMINISTIC_FIRST_DECISION_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", DeterministicAction.coerce(self.action))
        object.__setattr__(self, "source", DecisionSource.coerce(self.source))
        object.__setattr__(self, "reason", _text(self.reason, "reason"))
        object.__setattr__(
            self, "cache_key", _text(self.cache_key, "cache_key", required=False)
        )
        object.__setattr__(
            self,
            "resolved_item_ids",
            tuple(
                dict.fromkeys(
                    _text(item, "resolved_item_id", maximum=MAX_ID_BYTES + 64)
                    for item in self.resolved_item_ids
                    if str(item).strip()
                )
            ),
        )
        object.__setattr__(
            self,
            "residual_item_ids",
            tuple(
                dict.fromkeys(
                    _text(item, "residual_item_id", maximum=MAX_ID_BYTES + 64)
                    for item in self.residual_item_ids
                    if str(item).strip()
                )
            ),
        )
        object.__setattr__(
            self, "circuit_id", _text(self.circuit_id, "circuit_id", required=False)
        )
        action = self.action
        if not isinstance(action, DeterministicAction):
            action = DeterministicAction.coerce(action)
        may_dispatch = bool(self.may_dispatch_provider)
        if action is DeterministicAction.DISPATCH_PROVIDER:
            may_dispatch = True
        elif action in {
            DeterministicAction.RESOLVE_HIT,
            DeterministicAction.SUPPRESS_REPLAY,
            DeterministicAction.REJECT,
        }:
            may_dispatch = False
        object.__setattr__(self, "may_dispatch_provider", may_dispatch)
        object.__setattr__(
            self,
            "preserves_validation_proof",
            bool(self.preserves_validation_proof),
        )
        object.__setattr__(
            self,
            "evidence_digest",
            _text(self.evidence_digest, "evidence_digest", required=False),
        )
        if self.schema != DETERMINISTIC_FIRST_DECISION_SCHEMA:
            raise DeltaTaskPacketIntegrityError(
                "unsupported deterministic-first decision schema"
            )
        claimed = str(self.decision_id or "").strip()
        computed = _identity(
            "dfd",
            {
                "action": action.value,
                "reason": self.reason,
                "source": (
                    self.source.value
                    if isinstance(self.source, DecisionSource)
                    else str(self.source)
                ),
                "cache_key": self.cache_key,
                "resolved_item_ids": list(self.resolved_item_ids),
                "residual_item_ids": list(self.residual_item_ids),
                "circuit_id": self.circuit_id,
                "evidence_digest": self.evidence_digest,
            },
        )
        object.__setattr__(self, "decision_id", claimed or computed)

    @property
    def interface(self) -> str:
        return DETERMINISTIC_FIRST_DECISION_INTERFACE

    @property
    def is_hit(self) -> bool:
        action = self.action
        if not isinstance(action, DeterministicAction):
            action = DeterministicAction.coerce(action)
        return action is DeterministicAction.RESOLVE_HIT

    @property
    def is_suppressed(self) -> bool:
        action = self.action
        if not isinstance(action, DeterministicAction):
            action = DeterministicAction.coerce(action)
        return action is DeterministicAction.SUPPRESS_REPLAY

    def to_dict(self) -> dict[str, Any]:
        action = self.action
        source = self.source
        return {
            "schema": self.schema,
            "interface": DETERMINISTIC_FIRST_DECISION_INTERFACE,
            "decision_id": self.decision_id,
            "action": action.value
            if isinstance(action, DeterministicAction)
            else str(action),
            "reason": self.reason,
            "source": source.value
            if isinstance(source, DecisionSource)
            else str(source),
            "cache_key": self.cache_key,
            "resolved_item_ids": list(self.resolved_item_ids),
            "residual_item_ids": list(self.residual_item_ids),
            "circuit_id": self.circuit_id,
            "may_dispatch_provider": self.may_dispatch_provider,
            "preserves_validation_proof": self.preserves_validation_proof,
            "evidence_digest": self.evidence_digest,
            "authority": AUTHORITY_CLASS,
        }


@dataclass(frozen=True)
class DeltaTaskPacket:
    """Content-addressed delta task packet admitted for residual provider work.

    Interface: ``DeltaTaskPacket@1``.
    """

    packet_id: str
    task_cid: str
    repository_id: str
    tree_id: str
    context_cid: str
    plan_cid: str
    policy_id: str
    policy_digest: str
    schema_revision: int
    evidence_digest: str
    effect_scope: EffectScope
    requirements: ValidationProofRequirements
    unresolved_delta: tuple[DeltaItem, ...] = ()
    resolved_delta: tuple[DeltaItem, ...] = ()
    frontier: PacketFrontier = field(default_factory=PacketFrontier)
    completeness: Completeness | str = Completeness.COMPLETE
    budget: PacketBudget = field(default_factory=PacketBudget)
    counterexample_digest: str = ""
    failure_signature_id: str = ""
    decision: DeterministicFirstDecision | None = None
    total_bytes: int = 0
    total_tokens: int = 0
    producer_id: str = PRODUCER_ID
    authority: str = AUTHORITY_CLASS
    semantic_authority: bool = False
    write_authority: bool = False
    completion_authority: bool = False
    nomination_only: bool = True
    schema: str = DELTA_TASK_PACKET_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_cid", _text(self.task_cid, "task_cid"))
        object.__setattr__(
            self, "repository_id", _text(self.repository_id, "repository_id")
        )
        object.__setattr__(self, "tree_id", _text(self.tree_id, "tree_id"))
        object.__setattr__(
            self, "context_cid", _text(self.context_cid, "context_cid")
        )
        object.__setattr__(
            self, "plan_cid", _text(self.plan_cid, "plan_cid", required=False)
        )
        object.__setattr__(
            self,
            "policy_id",
            _text(self.policy_id or DEFAULT_POLICY_ID, "policy_id"),
        )
        object.__setattr__(
            self, "policy_digest", _text(self.policy_digest, "policy_digest")
        )
        object.__setattr__(
            self,
            "schema_revision",
            _positive_int(int(self.schema_revision), "schema_revision"),
        )
        object.__setattr__(
            self,
            "evidence_digest",
            _text(self.evidence_digest, "evidence_digest"),
        )
        if not isinstance(self.effect_scope, EffectScope):
            raise DeltaTaskPacketIntegrityError("effect_scope must be EffectScope")
        if self.effect_scope.repository_id and (
            self.effect_scope.repository_id != self.repository_id
        ):
            raise DeltaTaskPacketScopeError(
                "effect scope repository_id must match packet",
                reason_code="scope_repository_mismatch",
            )
        if self.effect_scope.tree_id and self.effect_scope.tree_id != self.tree_id:
            raise DeltaTaskPacketScopeError(
                "effect scope tree_id must match packet",
                reason_code="scope_tree_mismatch",
            )
        if not isinstance(self.requirements, ValidationProofRequirements):
            raise DeltaTaskPacketIntegrityError(
                "requirements must be ValidationProofRequirements"
            )
        unresolved = tuple(self.unresolved_delta)
        resolved = tuple(self.resolved_delta)
        for item in (*unresolved, *resolved):
            if not isinstance(item, DeltaItem):
                raise DeltaTaskPacketIntegrityError(
                    "delta members must be DeltaItem"
                )
        object.__setattr__(self, "unresolved_delta", unresolved)
        object.__setattr__(self, "resolved_delta", resolved)
        if not isinstance(self.frontier, PacketFrontier):
            raise DeltaTaskPacketIntegrityError("frontier must be PacketFrontier")
        object.__setattr__(
            self, "completeness", Completeness.coerce(self.completeness)
        )
        if not isinstance(self.budget, PacketBudget):
            raise DeltaTaskPacketIntegrityError("budget must be PacketBudget")
        object.__setattr__(
            self,
            "counterexample_digest",
            _text(
                self.counterexample_digest,
                "counterexample_digest",
                required=False,
            ),
        )
        object.__setattr__(
            self,
            "failure_signature_id",
            _text(
                self.failure_signature_id,
                "failure_signature_id",
                required=False,
            ),
        )
        if self.decision is not None and not isinstance(
            self.decision, DeterministicFirstDecision
        ):
            raise DeltaTaskPacketIntegrityError(
                "decision must be DeterministicFirstDecision"
            )
        object.__setattr__(
            self, "producer_id", _text(self.producer_id or PRODUCER_ID, "producer_id")
        )
        object.__setattr__(
            self, "authority", _text(self.authority or AUTHORITY_CLASS, "authority")
        )
        # Hard-zero provider authority claims.
        if self.nomination_only is not True:
            raise DeltaTaskPacketAuthorityError(
                "delta task packet must remain nomination_only"
            )
        for name in (
            "semantic_authority",
            "write_authority",
            "completion_authority",
        ):
            if getattr(self, name) is not False:
                raise DeltaTaskPacketAuthorityError(
                    f"delta task packet must hard-zero {name}"
                )
            object.__setattr__(self, name, False)
        object.__setattr__(self, "nomination_only", True)
        if self.schema != DELTA_TASK_PACKET_SCHEMA:
            raise DeltaTaskPacketIntegrityError(
                "unsupported delta task packet schema"
            )

        total_bytes = sum(item.byte_count for item in unresolved)
        total_tokens = sum(item.token_count for item in unresolved)
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
        if self.total_bytes > self.budget.max_bytes:
            raise DeltaTaskPacketOverflowError(
                "unresolved delta exceeds max_bytes budget"
            )
        if self.total_tokens > self.budget.max_tokens:
            raise DeltaTaskPacketOverflowError(
                "unresolved delta exceeds max_tokens budget"
            )
        if len(unresolved) > self.budget.max_items:
            raise DeltaTaskPacketOverflowError(
                "unresolved delta exceeds max_items budget"
            )

        claimed = str(self.packet_id or "").strip()
        computed = _content_cid(self._identity_body())
        if claimed and claimed != computed:
            raise DeltaTaskPacketIntegrityError(
                "packet_id does not match content identity"
            )
        object.__setattr__(self, "packet_id", claimed or computed)

        # Final fail-closed secret sweep over the sealed surface.
        _reject_or_redact_secrets(self._identity_body(), reject=True)

    def _identity_body(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": DELTA_TASK_PACKET_INTERFACE,
            "task_cid": self.task_cid,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "context_cid": self.context_cid,
            "plan_cid": self.plan_cid,
            "policy_id": self.policy_id,
            "policy_digest": self.policy_digest,
            "schema_revision": self.schema_revision,
            "evidence_digest": self.evidence_digest,
            "counterexample_digest": self.counterexample_digest,
            "failure_signature_id": self.failure_signature_id,
            "effect_scope": self.effect_scope.to_dict(),
            "requirements": self.requirements.to_dict(),
            "unresolved_delta": [item.digest for item in self.unresolved_delta],
            "resolved_delta": [item.digest for item in self.resolved_delta],
            "frontier": self.frontier.to_dict(),
            "completeness": (
                self.completeness.value
                if isinstance(self.completeness, Completeness)
                else str(self.completeness)
            ),
            "budget": self.budget.to_dict(),
            "semantic_authority": False,
            "write_authority": False,
            "completion_authority": False,
            "nomination_only": True,
            "producer_id": self.producer_id,
            "authority": self.authority,
        }

    @property
    def interface(self) -> str:
        return DELTA_TASK_PACKET_INTERFACE

    @property
    def is_admitted_for_provider(self) -> bool:
        if self.decision is None:
            return bool(self.unresolved_delta)
        return bool(self.decision.may_dispatch_provider) and bool(
            self.unresolved_delta
        )

    def included_unresolved(self) -> tuple[DeltaItem, ...]:
        return tuple(item for item in self.unresolved_delta if item.included)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": DELTA_TASK_PACKET_INTERFACE,
            "packet_id": self.packet_id,
            "task_cid": self.task_cid,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "context_cid": self.context_cid,
            "plan_cid": self.plan_cid,
            "policy_id": self.policy_id,
            "policy_digest": self.policy_digest,
            "schema_revision": self.schema_revision,
            "evidence_digest": self.evidence_digest,
            "counterexample_digest": self.counterexample_digest,
            "failure_signature_id": self.failure_signature_id,
            "effect_scope": self.effect_scope.to_dict(),
            "requirements": self.requirements.to_dict(),
            "unresolved_delta": [item.to_dict() for item in self.unresolved_delta],
            "resolved_delta": [item.to_dict() for item in self.resolved_delta],
            "frontier": self.frontier.to_dict(),
            "completeness": (
                self.completeness.value
                if isinstance(self.completeness, Completeness)
                else str(self.completeness)
            ),
            "budget": self.budget.to_dict(),
            "decision": self.decision.to_dict() if self.decision else None,
            "total_bytes": self.total_bytes,
            "total_tokens": self.total_tokens,
            "producer_id": self.producer_id,
            "authority": self.authority,
            "semantic_authority": False,
            "write_authority": False,
            "completion_authority": False,
            "nomination_only": True,
        }

    def provider_packet(self) -> dict[str, Any]:
        """Return the secret-free, authority-omitted provider-facing packet.

        Omitted frontier material is represented only by explicit handles;
        credentials and completion/write/semantic authority never appear as
        true claims. Validation and proof requirements are always preserved.
        """

        if self.decision is not None and not self.decision.may_dispatch_provider:
            raise DeltaTaskPacketSuppressedError(
                "provider packet refused: deterministic-first decision forbids dispatch",
                circuit_id=self.decision.circuit_id,
            )
        if not self.unresolved_delta:
            raise DeltaTaskPacketIntegrityError(
                "provider packet requires unresolved residual delta"
            )

        members = []
        for item in self.included_unresolved():
            members.append(
                {
                    "item_id": item.item_id,
                    "kind": item.kind.value
                    if isinstance(item.kind, DeltaItemKind)
                    else str(item.kind),
                    "digest": item.digest,
                    "summary": item.summary,
                    "path": item.path,
                    "payload": dict(item.payload),
                    "expansion_handle": item.expansion_handle,
                }
            )
        packet = {
            "schema": PROVIDER_FACING_PACKET_SCHEMA,
            "packet_id": self.packet_id,
            "task_cid": self.task_cid,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "context_cid": self.context_cid,
            "plan_cid": self.plan_cid,
            "policy_id": self.policy_id,
            "policy_digest": self.policy_digest,
            "schema_revision": self.schema_revision,
            "evidence_digest": self.evidence_digest,
            "counterexample_digest": self.counterexample_digest,
            "effect_scope": self.effect_scope.to_dict(),
            "requirements": self.requirements.to_dict(),
            "members": members,
            "frontier": self.frontier.to_dict(),
            "completeness": (
                self.completeness.value
                if isinstance(self.completeness, Completeness)
                else str(self.completeness)
            ),
            "data_label": UNTRUSTED_DATA_LABEL,
            "treat_as": "data_not_instructions",
            "authority": AUTHORITY_CLASS,
            "semantic_authority": False,
            "write_authority": False,
            "completion_authority": False,
            "nomination_only": True,
            # Explicitly document that omitted authority/credentials are not present.
            "omitted_authority": sorted(_PROVIDER_OMITTED_AUTHORITY_KEYS),
            "contains_credentials": False,
            "contains_secrets": False,
        }
        cleaned = _reject_or_redact_secrets(packet, reject=True)
        # Guard: omitted authority keys must never be true/non-empty grants.
        for key in _PROVIDER_OMITTED_AUTHORITY_KEYS:
            if key in cleaned and cleaned[key] not in (False, None, 0, "", []):
                if key == "omitted_authority":
                    continue
                raise DeltaTaskPacketAuthorityError(
                    f"provider packet leaked omitted authority: {key}"
                )
        # Ensure hard-zeros remain hard-zeros after scrubbing.
        cleaned["semantic_authority"] = False
        cleaned["write_authority"] = False
        cleaned["completion_authority"] = False
        cleaned["nomination_only"] = True
        cleaned["contains_credentials"] = False
        cleaned["contains_secrets"] = False
        return cleaned


@dataclass(frozen=True)
class PacketReplyBinding:
    """Binds a provider reply to the exact packet, context, and effect scope."""

    binding_id: str
    packet_id: str
    context_cid: str
    effect_scope_digest: str
    requirements_digest: str
    reply_digest: str
    accepted: bool
    reason: str = ""
    schema: str = PACKET_REPLY_BINDING_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "packet_id", _text(self.packet_id, "packet_id"))
        object.__setattr__(
            self, "context_cid", _text(self.context_cid, "context_cid")
        )
        object.__setattr__(
            self,
            "effect_scope_digest",
            _text(self.effect_scope_digest, "effect_scope_digest"),
        )
        object.__setattr__(
            self,
            "requirements_digest",
            _text(self.requirements_digest, "requirements_digest"),
        )
        object.__setattr__(
            self, "reply_digest", _text(self.reply_digest, "reply_digest")
        )
        object.__setattr__(self, "accepted", bool(self.accepted))
        object.__setattr__(
            self, "reason", _text(self.reason, "reason", required=False)
        )
        if self.schema != PACKET_REPLY_BINDING_SCHEMA:
            raise DeltaTaskPacketIntegrityError(
                "unsupported packet reply binding schema"
            )
        claimed = str(self.binding_id or "").strip()
        computed = _identity(
            "reply-bind",
            {
                "packet_id": self.packet_id,
                "context_cid": self.context_cid,
                "effect_scope_digest": self.effect_scope_digest,
                "requirements_digest": self.requirements_digest,
                "reply_digest": self.reply_digest,
                "accepted": self.accepted,
            },
        )
        object.__setattr__(self, "binding_id", claimed or computed)

    @property
    def interface(self) -> str:
        return PACKET_REPLY_BINDING_INTERFACE

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": PACKET_REPLY_BINDING_INTERFACE,
            "binding_id": self.binding_id,
            "packet_id": self.packet_id,
            "context_cid": self.context_cid,
            "effect_scope_digest": self.effect_scope_digest,
            "requirements_digest": self.requirements_digest,
            "reply_digest": self.reply_digest,
            "accepted": self.accepted,
            "reason": self.reason,
            "authority": AUTHORITY_CLASS,
        }


@dataclass(frozen=True)
class ReplayCircuit:
    """Typed circuit that suppresses unchanged failure reprompts."""

    circuit_id: str
    failure_signature_id: str
    evidence_digest: str
    task_cid: str
    open: bool = True
    failure_count: int = 1
    retry_budget: int = DEFAULT_RETRY_BUDGET
    opened_at_ms: int = 0
    expires_at_ms: int = 0
    reason: str = "unchanged_failure"
    schema: str = CIRCUIT_STATE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "failure_signature_id",
            _text(self.failure_signature_id, "failure_signature_id"),
        )
        object.__setattr__(
            self, "evidence_digest", _text(self.evidence_digest, "evidence_digest")
        )
        object.__setattr__(self, "task_cid", _text(self.task_cid, "task_cid"))
        object.__setattr__(self, "open", bool(self.open))
        object.__setattr__(
            self,
            "failure_count",
            _positive_int(int(self.failure_count), "failure_count"),
        )
        object.__setattr__(
            self,
            "retry_budget",
            _nonneg_int(int(self.retry_budget), "retry_budget"),
        )
        object.__setattr__(
            self, "opened_at_ms", _nonneg_int(int(self.opened_at_ms), "opened_at_ms")
        )
        object.__setattr__(
            self,
            "expires_at_ms",
            _nonneg_int(int(self.expires_at_ms), "expires_at_ms"),
        )
        object.__setattr__(
            self, "reason", _text(self.reason or "unchanged_failure", "reason")
        )
        claimed = str(self.circuit_id or "").strip()
        computed = _identity(
            "circuit",
            {
                "failure_signature_id": self.failure_signature_id,
                "evidence_digest": self.evidence_digest,
                "task_cid": self.task_cid,
            },
        )
        object.__setattr__(self, "circuit_id", claimed or computed)

    def is_active(self, *, now_ms: int | None = None) -> bool:
        if not self.open:
            return False
        if self.expires_at_ms <= 0:
            return True
        current = int(now_ms if now_ms is not None else _now_ms())
        return current < self.expires_at_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "circuit_id": self.circuit_id,
            "failure_signature_id": self.failure_signature_id,
            "evidence_digest": self.evidence_digest,
            "task_cid": self.task_cid,
            "open": self.open,
            "failure_count": self.failure_count,
            "retry_budget": self.retry_budget,
            "opened_at_ms": self.opened_at_ms,
            "expires_at_ms": self.expires_at_ms,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class DeltaTaskPacketRequest:
    """Inputs used to build and gate one delta task packet."""

    task_cid: str
    repository_id: str
    tree_id: str
    context_cid: str = ""
    plan_cid: str = ""
    policy_id: str = DEFAULT_POLICY_ID
    policy_digest: str = ""
    schema_revision: int = DEFAULT_SCHEMA_REVISION
    task_summary: str = ""
    evidence_digest: str = ""
    counterexample: Mapping[str, Any] | None = None
    latest_failure: Mapping[str, Any] | None = None
    unmet_dependencies: tuple[Mapping[str, Any] | str, ...] = ()
    worktree_delta: Mapping[str, Any] | Sequence[Any] | None = None
    impacted_symbols: tuple[Mapping[str, Any] | str, ...] = ()
    open_obligations: tuple[Mapping[str, Any] | str, ...] = ()
    evidence: tuple[Mapping[str, Any] | str, ...] = ()
    validations: tuple[Mapping[str, Any] | str, ...] = ()
    proof_obligations: tuple[Mapping[str, Any] | str, ...] = ()
    acceptance_ids: tuple[str, ...] = ()
    allowed_effects: tuple[str, ...] = DEFAULT_ALLOWED_EFFECTS
    write_paths: tuple[str, ...] = ()
    deterministic_resolutions: tuple[Mapping[str, Any] | str, ...] = ()
    require_validation: bool = True
    require_proof: bool = False
    retry_budget: int = DEFAULT_RETRY_BUDGET
    prior_failure_count: int = 0
    failure_signature_id: str = ""
    budget: PacketBudget = field(default_factory=PacketBudget)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    heartbeat_at: str = ""
    observed_at: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_cid", _text(self.task_cid, "task_cid"))
        object.__setattr__(
            self, "repository_id", _text(self.repository_id, "repository_id")
        )
        object.__setattr__(self, "tree_id", _text(self.tree_id, "tree_id"))
        object.__setattr__(
            self, "plan_cid", _text(self.plan_cid, "plan_cid", required=False)
        )
        object.__setattr__(
            self,
            "policy_id",
            _text(self.policy_id or DEFAULT_POLICY_ID, "policy_id"),
        )
        policy_digest = str(self.policy_digest or "").strip()
        if not policy_digest:
            policy_digest = _identity(
                "policy",
                {
                    "policy_id": self.policy_id,
                    "schema_revision": self.schema_revision,
                },
            )
        object.__setattr__(self, "policy_digest", policy_digest)
        object.__setattr__(
            self,
            "schema_revision",
            _positive_int(int(self.schema_revision), "schema_revision"),
        )
        object.__setattr__(
            self,
            "task_summary",
            _text(self.task_summary, "task_summary", required=False),
        )
        object.__setattr__(
            self,
            "retry_budget",
            _nonneg_int(int(self.retry_budget), "retry_budget"),
        )
        object.__setattr__(
            self,
            "prior_failure_count",
            _nonneg_int(int(self.prior_failure_count), "prior_failure_count"),
        )
        object.__setattr__(
            self,
            "failure_signature_id",
            _text(
                self.failure_signature_id,
                "failure_signature_id",
                required=False,
            ),
        )
        if not isinstance(self.budget, PacketBudget):
            raise DeltaTaskPacketIntegrityError("budget must be PacketBudget")
        # Fail closed on secret metadata early.
        metadata = _reject_or_redact_secrets(
            _strip_noise(_as_mapping(self.metadata)), reject=True
        )
        object.__setattr__(self, "metadata", _freeze_mapping(metadata))
        object.__setattr__(
            self,
            "allowed_effects",
            _normalize_effects(self.allowed_effects),
        )
        object.__setattr__(self, "write_paths", _normalize_paths(self.write_paths))
        object.__setattr__(
            self, "require_validation", bool(self.require_validation)
        )
        object.__setattr__(self, "require_proof", bool(self.require_proof))

        # Normalize sequences.
        object.__setattr__(
            self,
            "unmet_dependencies",
            tuple(self.unmet_dependencies or ()),
        )
        object.__setattr__(
            self, "impacted_symbols", tuple(self.impacted_symbols or ())
        )
        object.__setattr__(
            self, "open_obligations", tuple(self.open_obligations or ())
        )
        object.__setattr__(self, "evidence", tuple(self.evidence or ()))
        object.__setattr__(self, "validations", tuple(self.validations or ()))
        object.__setattr__(
            self, "proof_obligations", tuple(self.proof_obligations or ())
        )
        object.__setattr__(
            self, "acceptance_ids", tuple(self.acceptance_ids or ())
        )
        object.__setattr__(
            self,
            "deterministic_resolutions",
            tuple(self.deterministic_resolutions or ()),
        )

        counterexample = (
            _reject_or_redact_secrets(
                _strip_noise(_as_mapping(self.counterexample)), reject=True
            )
            if self.counterexample is not None
            else None
        )
        object.__setattr__(
            self,
            "counterexample",
            _freeze_mapping(counterexample) if counterexample is not None else None,
        )
        latest_failure = (
            _reject_or_redact_secrets(
                _strip_noise(_as_mapping(self.latest_failure)), reject=True
            )
            if self.latest_failure is not None
            else None
        )
        object.__setattr__(
            self,
            "latest_failure",
            _freeze_mapping(latest_failure) if latest_failure is not None else None,
        )
        if self.worktree_delta is not None:
            if isinstance(self.worktree_delta, Mapping):
                worktree = _reject_or_redact_secrets(
                    _strip_noise(dict(self.worktree_delta)), reject=True
                )
                paths = worktree.get("paths") or ()
                for path in paths:
                    _repo_path(path, required=True)
                object.__setattr__(self, "worktree_delta", _freeze_mapping(worktree))
            elif isinstance(self.worktree_delta, Sequence) and not isinstance(
                self.worktree_delta, (str, bytes, bytearray)
            ):
                paths = [_repo_path(item, required=True) for item in self.worktree_delta]
                object.__setattr__(
                    self,
                    "worktree_delta",
                    _freeze_mapping({"paths": paths}),
                )
            else:
                raise DeltaTaskPacketIntegrityError(
                    "worktree_delta must be a mapping or path sequence"
                )

        evidence_digest = str(self.evidence_digest or "").strip()
        if not evidence_digest:
            evidence_digest = self.compute_evidence_digest()
        object.__setattr__(self, "evidence_digest", evidence_digest)

        context_cid = str(self.context_cid or "").strip()
        if not context_cid:
            context_cid = _content_cid(self._context_identity())
        object.__setattr__(self, "context_cid", context_cid)

    def _context_identity(self) -> dict[str, Any]:
        return {
            "task_cid": self.task_cid,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "plan_cid": self.plan_cid,
            "policy_id": self.policy_id,
            "policy_digest": self.policy_digest,
            "schema_revision": self.schema_revision,
            "task_summary": self.task_summary,
            "evidence_digest": self.evidence_digest,
        }

    def compute_evidence_digest(self) -> str:
        material = _strip_noise(
            {
                "counterexample": dict(self.counterexample or {}),
                "latest_failure": dict(self.latest_failure or {}),
                "unmet_dependencies": list(self.unmet_dependencies),
                "worktree_delta": (
                    dict(self.worktree_delta)
                    if isinstance(self.worktree_delta, Mapping)
                    else self.worktree_delta
                ),
                "impacted_symbols": list(self.impacted_symbols),
                "open_obligations": list(self.open_obligations),
                "evidence": list(self.evidence),
                "validations": list(self.validations),
                "proof_obligations": list(self.proof_obligations),
                "plan_cid": self.plan_cid,
                "policy_digest": self.policy_digest,
                "schema_revision": self.schema_revision,
                "tree_id": self.tree_id,
            }
        )
        return _content_digest(material)

    def counterexample_digest(self) -> str:
        if not self.counterexample:
            return ""
        return _content_digest(dict(self.counterexample))

    def cache_key(self) -> str:
        return _identity(
            "dtp-cache",
            {
                "task_cid": self.task_cid,
                "repository_id": self.repository_id,
                "tree_id": self.tree_id,
                "plan_cid": self.plan_cid,
                "policy_digest": self.policy_digest,
                "schema_revision": self.schema_revision,
                "evidence_digest": self.evidence_digest,
                "context_cid": self.context_cid,
            },
        )


# ---------------------------------------------------------------------------
# Decision cache + circuit store (in-process; cold-import safe)
# ---------------------------------------------------------------------------


class DecisionCache:
    """Exact-identity decision cache for deterministic-first hits."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._entries: dict[str, dict[str, Any]] = {}

    def put(
        self,
        cache_key: str,
        *,
        resolution: Mapping[str, Any],
        requirements_digest: str,
    ) -> None:
        key = _text(cache_key, "cache_key")
        body = _reject_or_redact_secrets(_strip_noise(dict(resolution)), reject=True)
        with self._lock:
            self._entries[key] = {
                "cache_key": key,
                "resolution": body,
                "requirements_digest": _text(
                    requirements_digest, "requirements_digest"
                ),
                "stored_at_ms": _now_ms(),
            }

    def get(self, cache_key: str) -> Mapping[str, Any] | None:
        key = _text(cache_key, "cache_key", required=False)
        if not key:
            return None
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            return MappingProxyType(dict(entry))

    def invalidate(self, cache_key: str) -> None:
        key = _text(cache_key, "cache_key", required=False)
        if not key:
            return
        with self._lock:
            self._entries.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


class ReplayCircuitStore:
    """In-memory typed circuits for unchanged-failure suppression."""

    def __init__(self, *, default_ttl_ms: int = DEFAULT_CIRCUIT_TTL_MS) -> None:
        self._lock = threading.RLock()
        self._circuits: dict[str, ReplayCircuit] = {}
        self._default_ttl_ms = _nonneg_int(int(default_ttl_ms), "default_ttl_ms")

    def get(
        self,
        *,
        failure_signature_id: str,
        evidence_digest: str,
        task_cid: str,
        now_ms: int | None = None,
    ) -> ReplayCircuit | None:
        circuit_id = _identity(
            "circuit",
            {
                "failure_signature_id": _text(
                    failure_signature_id, "failure_signature_id"
                ),
                "evidence_digest": _text(evidence_digest, "evidence_digest"),
                "task_cid": _text(task_cid, "task_cid"),
            },
        )
        with self._lock:
            circuit = self._circuits.get(circuit_id)
            if circuit is None:
                return None
            if not circuit.is_active(now_ms=now_ms):
                closed = ReplayCircuit(
                    circuit_id=circuit.circuit_id,
                    failure_signature_id=circuit.failure_signature_id,
                    evidence_digest=circuit.evidence_digest,
                    task_cid=circuit.task_cid,
                    open=False,
                    failure_count=circuit.failure_count,
                    retry_budget=circuit.retry_budget,
                    opened_at_ms=circuit.opened_at_ms,
                    expires_at_ms=circuit.expires_at_ms,
                    reason=circuit.reason,
                )
                self._circuits[circuit_id] = closed
                return closed
            return circuit

    def record_failure(
        self,
        *,
        failure_signature_id: str,
        evidence_digest: str,
        task_cid: str,
        retry_budget: int = DEFAULT_RETRY_BUDGET,
        now_ms: int | None = None,
        reason: str = "unchanged_failure",
    ) -> ReplayCircuit:
        current_ms = int(now_ms if now_ms is not None else _now_ms())
        existing = self.get(
            failure_signature_id=failure_signature_id,
            evidence_digest=evidence_digest,
            task_cid=task_cid,
            now_ms=current_ms,
        )
        failure_count = 1 if existing is None else existing.failure_count + 1
        budget = _nonneg_int(int(retry_budget), "retry_budget")
        open_circuit = failure_count > budget
        expires = (
            current_ms + self._default_ttl_ms
            if open_circuit and self._default_ttl_ms > 0
            else 0
        )
        circuit = ReplayCircuit(
            circuit_id="",
            failure_signature_id=failure_signature_id,
            evidence_digest=evidence_digest,
            task_cid=task_cid,
            open=open_circuit,
            failure_count=failure_count,
            retry_budget=budget,
            opened_at_ms=current_ms if open_circuit else 0,
            expires_at_ms=expires,
            reason=reason if open_circuit else "within_retry_budget",
        )
        with self._lock:
            self._circuits[circuit.circuit_id] = circuit
        return circuit

    def clear(self) -> None:
        with self._lock:
            self._circuits.clear()


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _item_from_mapping(
    value: Mapping[str, Any] | str,
    *,
    kind: DeltaItemKind,
    id_key: str,
    ordinal: int,
    resolved: bool = False,
) -> DeltaItem:
    if isinstance(value, str):
        payload = {"id": value, "summary": value}
        summary = value
        path = ""
    else:
        payload = _reject_or_redact_secrets(
            _strip_noise(dict(value)), reject=True
        )
        summary = str(
            payload.get("summary")
            or payload.get("command")
            or payload.get(id_key)
            or ""
        )
        path = str(payload.get("path") or "")
    return DeltaItem(
        item_id="",
        kind=kind,
        digest="",
        summary=summary,
        path=path,
        payload=payload,
        included=True,
        resolved_deterministically=resolved,
        ordinal=ordinal,
    )


def _candidate_items(request: DeltaTaskPacketRequest) -> list[DeltaItem]:
    items: list[DeltaItem] = []
    ordinal = 0

    items.append(
        DeltaItem(
            item_id="",
            kind=DeltaItemKind.TASK,
            digest="",
            summary=request.task_summary or request.task_cid,
            payload={
                "task_cid": request.task_cid,
                "summary": request.task_summary,
            },
            ordinal=ordinal,
        )
    )
    ordinal += 1

    for dep in request.unmet_dependencies[:MAX_DELTA_ITEMS]:
        items.append(
            _item_from_mapping(
                dep,
                kind=DeltaItemKind.DEPENDENCY,
                id_key="dependency_id",
                ordinal=ordinal,
            )
        )
        ordinal += 1

    if request.latest_failure:
        items.append(
            DeltaItem(
                item_id="",
                kind=DeltaItemKind.FAILURE,
                digest="",
                summary=str(
                    request.latest_failure.get("summary")
                    or request.latest_failure.get("failure_id")
                    or "latest_failure"
                ),
                payload=dict(request.latest_failure),
                ordinal=ordinal,
            )
        )
        ordinal += 1

    if request.worktree_delta:
        payload = (
            dict(request.worktree_delta)
            if isinstance(request.worktree_delta, Mapping)
            else {"paths": list(request.worktree_delta)}
        )
        items.append(
            DeltaItem(
                item_id="",
                kind=DeltaItemKind.WORKTREE_DELTA,
                digest="",
                summary="worktree_delta",
                payload=payload,
                ordinal=ordinal,
            )
        )
        ordinal += 1

    for symbol in request.impacted_symbols[:MAX_DELTA_ITEMS]:
        items.append(
            _item_from_mapping(
                symbol,
                kind=DeltaItemKind.SYMBOL,
                id_key="symbol",
                ordinal=ordinal,
            )
        )
        ordinal += 1

    for obligation in request.open_obligations[:MAX_DELTA_ITEMS]:
        items.append(
            _item_from_mapping(
                obligation,
                kind=DeltaItemKind.OBLIGATION,
                id_key="obligation_id",
                ordinal=ordinal,
            )
        )
        ordinal += 1

    for evidence in request.evidence[:MAX_DELTA_ITEMS]:
        items.append(
            _item_from_mapping(
                evidence,
                kind=DeltaItemKind.EVIDENCE,
                id_key="evidence_id",
                ordinal=ordinal,
            )
        )
        ordinal += 1

    if request.counterexample:
        items.append(
            DeltaItem(
                item_id="",
                kind=DeltaItemKind.COUNTEREXAMPLE,
                digest="",
                summary=str(
                    request.counterexample.get("summary")
                    or request.counterexample.get("counterexample_id")
                    or "counterexample"
                ),
                payload=dict(request.counterexample),
                ordinal=ordinal,
            )
        )
        ordinal += 1

    for command in request.validations[:MAX_VALIDATIONS]:
        items.append(
            _item_from_mapping(
                command,
                kind=DeltaItemKind.VALIDATION,
                id_key="command",
                ordinal=ordinal,
            )
        )
        ordinal += 1

    for proof in request.proof_obligations[:MAX_PROOFS]:
        items.append(
            _item_from_mapping(
                proof,
                kind=DeltaItemKind.PROOF,
                id_key="obligation_id",
                ordinal=ordinal,
            )
        )
        ordinal += 1

    for effect in request.allowed_effects:
        items.append(
            DeltaItem(
                item_id="",
                kind=DeltaItemKind.EFFECT,
                digest="",
                summary=effect,
                payload={"effect": effect},
                ordinal=ordinal,
            )
        )
        ordinal += 1

    return items


def _resolved_id_set(request: DeltaTaskPacketRequest) -> set[str]:
    resolved: set[str] = set()
    for item in request.deterministic_resolutions:
        if isinstance(item, Mapping):
            for key in (
                "item_id",
                "id",
                "dependency_id",
                "obligation_id",
                "evidence_id",
                "symbol",
                "command",
            ):
                value = item.get(key)
                if value:
                    resolved.add(str(value).strip())
            if item.get("kind") and item.get("digest"):
                resolved.add(f"{item['kind']}:{item['digest']}")
        else:
            resolved.add(str(item).strip())
    return {item for item in resolved if item}


def _apply_budgets(
    items: Sequence[DeltaItem],
    budget: PacketBudget,
    *,
    force_kinds: frozenset[DeltaItemKind],
) -> tuple[list[DeltaItem], list[DeltaItem], FrontierDisposition, list[str]]:
    included: list[DeltaItem] = []
    omitted: list[DeltaItem] = []
    reasons: list[str] = []
    used_bytes = 0
    used_tokens = 0
    disposition = FrontierDisposition.EMPTY

    def mark(
        item: DeltaItem,
        *,
        included_flag: bool,
        omission_reason: str = "",
    ) -> DeltaItem:
        handle = ""
        if not included_flag:
            handle = _identity(
                "expand",
                {"item_id": item.item_id, "digest": item.digest},
            )
        return DeltaItem(
            item_id=item.item_id,
            kind=item.kind,
            digest=item.digest,
            summary=item.summary,
            path=item.path,
            payload=dict(item.payload),
            included=included_flag,
            omission_reason=omission_reason,
            expansion_handle=handle,
            resolved_deterministically=item.resolved_deterministically,
            ordinal=item.ordinal,
        )

    def try_include(item: DeltaItem, *, force: bool) -> None:
        nonlocal used_bytes, used_tokens, disposition
        kind = (
            item.kind
            if isinstance(item.kind, DeltaItemKind)
            else DeltaItemKind.coerce(item.kind)
        )
        next_bytes = used_bytes + item.byte_count
        next_tokens = used_tokens + item.token_count
        over = (
            len(included) >= budget.max_rows
            or len(included) >= budget.max_items
            or next_bytes > budget.max_bytes
            or next_tokens > budget.max_tokens
        )
        if over and force:
            raise DeltaTaskPacketOverflowError(
                f"required invariant delta item exceeds budget: {kind.value}"
            )
        if over:
            omitted.append(
                mark(item, included_flag=False, omission_reason="budget_overflow")
            )
            disposition = FrontierDisposition.BUDGET_OVERFLOW
            reasons.append(f"omitted {kind.value} due to budget")
            return
        included.append(mark(item, included_flag=True))
        used_bytes = next_bytes
        used_tokens = next_tokens

    # Invariant core first so progressive disclosure never drops validations,
    # proofs, effects, or task identity under residual pressure.
    invariants = [
        item
        for item in items
        if (
            item.kind
            if isinstance(item.kind, DeltaItemKind)
            else DeltaItemKind.coerce(item.kind)
        )
        in force_kinds
    ]
    optional = [
        item
        for item in items
        if (
            item.kind
            if isinstance(item.kind, DeltaItemKind)
            else DeltaItemKind.coerce(item.kind)
        )
        not in force_kinds
    ]
    for item in invariants:
        try_include(item, force=True)
    for item in optional:
        try_include(item, force=False)

    return included, omitted, disposition, reasons


def compute_failure_signature_id(
    *,
    task_cid: str,
    evidence_digest: str,
    proposal_digest: str = "",
    failure_class: str = "validation",
    policy_id: str = DEFAULT_POLICY_ID,
) -> str:
    return _identity(
        "fsig",
        {
            "task_cid": _text(task_cid, "task_cid"),
            "evidence_digest": _text(evidence_digest, "evidence_digest"),
            "proposal_digest": _text(
                proposal_digest, "proposal_digest", required=False
            ),
            "failure_class": _text(failure_class or "validation", "failure_class"),
            "policy_id": _text(policy_id or DEFAULT_POLICY_ID, "policy_id"),
        },
    )


def evaluate_deterministic_first(
    request: DeltaTaskPacketRequest,
    *,
    decision_cache: DecisionCache | None = None,
    circuit_store: ReplayCircuitStore | None = None,
    now_ms: int | None = None,
) -> DeterministicFirstDecision:
    """Resolve known work before admitting a provider packet.

    Order of evaluation (fail closed):

    1. Active unchanged-failure circuit → ``suppress_replay``
    2. Exact decision-cache hit with matching requirements → ``resolve_hit``
    3. Operator / query resolutions covering all residual work → ``resolve_hit``
    4. Otherwise residual remains → ``dispatch_provider``
    """

    requirements = ValidationProofRequirements(
        validation_commands=_normalize_commands(request.validations),
        proof_obligations=_normalize_ids(
            request.proof_obligations, name="proof_obligation", limit=MAX_PROOFS
        ),
        acceptance_ids=_normalize_ids(
            request.acceptance_ids, name="acceptance", limit=MAX_PROOFS
        ),
        require_validation=request.require_validation,
        require_proof=request.require_proof,
    )
    cache_key = request.cache_key()
    evidence = request.evidence_digest
    failure_sig = request.failure_signature_id or compute_failure_signature_id(
        task_cid=request.task_cid,
        evidence_digest=evidence,
        policy_id=request.policy_id,
    )

    # 1) Replay circuit for unchanged failures.
    if circuit_store is not None:
        circuit = circuit_store.get(
            failure_signature_id=failure_sig,
            evidence_digest=evidence,
            task_cid=request.task_cid,
            now_ms=now_ms,
        )
        if circuit is not None and circuit.is_active(now_ms=now_ms):
            return DeterministicFirstDecision(
                action=DeterministicAction.SUPPRESS_REPLAY,
                reason="unchanged_failure_circuit_open",
                source=DecisionSource.CIRCUIT,
                cache_key=cache_key,
                circuit_id=circuit.circuit_id,
                may_dispatch_provider=False,
                preserves_validation_proof=True,
                evidence_digest=evidence,
            )
    elif (
        request.prior_failure_count > request.retry_budget
        and request.failure_signature_id
    ):
        return DeterministicFirstDecision(
            action=DeterministicAction.SUPPRESS_REPLAY,
            reason="policy_exhausted_unchanged_evidence",
            source=DecisionSource.POLICY,
            cache_key=cache_key,
            may_dispatch_provider=False,
            preserves_validation_proof=True,
            evidence_digest=evidence,
        )

    # 2) Decision cache hit.
    if decision_cache is not None:
        cached = decision_cache.get(cache_key)
        if cached is not None:
            if cached.get("requirements_digest") != requirements.requirements_digest:
                # Requirements drift invalidates the cache entry.
                decision_cache.invalidate(cache_key)
            else:
                resolution = dict(cached.get("resolution") or {})
                resolved_ids = tuple(
                    str(item)
                    for item in resolution.get("resolved_item_ids", ())
                    if str(item).strip()
                )
                return DeterministicFirstDecision(
                    action=DeterministicAction.RESOLVE_HIT,
                    reason="decision_cache_hit",
                    source=DecisionSource.DECISION_CACHE,
                    cache_key=cache_key,
                    resolved_item_ids=resolved_ids,
                    residual_item_ids=(),
                    may_dispatch_provider=False,
                    preserves_validation_proof=True,
                    evidence_digest=evidence,
                )

    # 3) Operator / query resolutions.
    candidates = _candidate_items(request)
    resolved_keys = _resolved_id_set(request)
    residual_ids: list[str] = []
    resolved_ids: list[str] = []
    for item in candidates:
        kind = item.kind if isinstance(item.kind, DeltaItemKind) else DeltaItemKind.coerce(item.kind)
        # Validation/proof/effect/task identity are never "resolved away".
        if kind in {
            DeltaItemKind.VALIDATION,
            DeltaItemKind.PROOF,
            DeltaItemKind.EFFECT,
            DeltaItemKind.TASK,
        }:
            continue
        markers = {
            item.item_id,
            item.digest,
            str(item.payload.get("id") or ""),
            str(item.payload.get("dependency_id") or ""),
            str(item.payload.get("obligation_id") or ""),
            str(item.payload.get("evidence_id") or ""),
            str(item.payload.get("symbol") or ""),
            str(item.payload.get("command") or ""),
            f"{kind.value}:{item.digest}",
        }
        if markers & resolved_keys:
            resolved_ids.append(item.item_id)
        else:
            # Only residual non-invariant work counts as needing the model.
            if kind not in {DeltaItemKind.VALIDATION, DeltaItemKind.PROOF}:
                residual_ids.append(item.item_id)

    # Dependencies/symbols/etc. that were not explicitly resolved remain residual.
    # If caller provided deterministic resolutions covering every non-invariant
    # residual and no counterexample/failure remains, treat as hit.
    residual_kinds_present = any(
        item.item_id in residual_ids for item in candidates
    )
    has_open_failure = bool(request.latest_failure) and not (
        markers_for_failure_resolved(request, resolved_keys)
    )
    has_open_counterexample = bool(request.counterexample) and not (
        str(request.counterexample.get("counterexample_id") or "") in resolved_keys
        or str(request.counterexample.get("id") or "") in resolved_keys
    )

    if not residual_kinds_present and not has_open_failure and not has_open_counterexample:
        source = (
            DecisionSource.OPERATOR
            if request.deterministic_resolutions
            else DecisionSource.QUERY
        )
        reason = (
            "deterministic_operators_resolved_residual"
            if request.deterministic_resolutions
            else "query_found_no_unresolved_residual"
        )
        return DeterministicFirstDecision(
            action=DeterministicAction.RESOLVE_HIT,
            reason=reason,
            source=source,
            cache_key=cache_key,
            resolved_item_ids=tuple(resolved_ids),
            residual_item_ids=(),
            may_dispatch_provider=False,
            preserves_validation_proof=True,
            evidence_digest=evidence,
        )

    # 4) Cache miss / residual remains → provider dispatch for bounded delta.
    return DeterministicFirstDecision(
        action=DeterministicAction.DISPATCH_PROVIDER,
        reason="cache_miss_unresolved_residual",
        source=DecisionSource.RESIDUAL,
        cache_key=cache_key,
        resolved_item_ids=tuple(resolved_ids),
        residual_item_ids=tuple(residual_ids),
        may_dispatch_provider=True,
        preserves_validation_proof=True,
        evidence_digest=evidence,
    )


def markers_for_failure_resolved(
    request: DeltaTaskPacketRequest, resolved_keys: set[str]
) -> bool:
    if not request.latest_failure:
        return True
    markers = {
        str(request.latest_failure.get("failure_id") or ""),
        str(request.latest_failure.get("id") or ""),
    }
    return bool(markers & resolved_keys)


def build_delta_task_packet(
    request: DeltaTaskPacketRequest,
    *,
    decision: DeterministicFirstDecision | None = None,
    decision_cache: DecisionCache | None = None,
    circuit_store: ReplayCircuitStore | None = None,
    now_ms: int | None = None,
) -> DeltaTaskPacket:
    """Build one content-addressed delta task packet under deterministic-first policy."""

    decision = decision or evaluate_deterministic_first(
        request,
        decision_cache=decision_cache,
        circuit_store=circuit_store,
        now_ms=now_ms,
    )

    effect_scope = EffectScope(
        allowed_effects=request.allowed_effects,
        write_paths=request.write_paths,
        repository_id=request.repository_id,
        tree_id=request.tree_id,
    )
    requirements = ValidationProofRequirements(
        validation_commands=_normalize_commands(request.validations),
        proof_obligations=_normalize_ids(
            request.proof_obligations, name="proof_obligation", limit=MAX_PROOFS
        ),
        acceptance_ids=_normalize_ids(
            request.acceptance_ids, name="acceptance", limit=MAX_PROOFS
        ),
        require_validation=request.require_validation,
        require_proof=request.require_proof,
    )

    # Deterministic hit still preserves validation/proof requirements.
    if not decision.preserves_validation_proof:
        raise DeltaTaskPacketIntegrityError(
            "deterministic resolution must preserve validation/proof requirements"
        )

    candidates = _candidate_items(request)
    resolved_ids = set(decision.resolved_item_ids)
    resolved_keys = _resolved_id_set(request) | resolved_ids

    resolved_items: list[DeltaItem] = []
    residual_items: list[DeltaItem] = []
    for item in candidates:
        kind = item.kind if isinstance(item.kind, DeltaItemKind) else DeltaItemKind.coerce(item.kind)
        markers = {
            item.item_id,
            item.digest,
            str(item.payload.get("id") or ""),
            str(item.payload.get("dependency_id") or ""),
            str(item.payload.get("obligation_id") or ""),
            str(item.payload.get("evidence_id") or ""),
            str(item.payload.get("symbol") or ""),
            f"{kind.value}:{item.digest}",
        }
        if markers & resolved_keys and kind not in {
            DeltaItemKind.VALIDATION,
            DeltaItemKind.PROOF,
            DeltaItemKind.EFFECT,
            DeltaItemKind.TASK,
        }:
            resolved_items.append(
                DeltaItem(
                    item_id=item.item_id,
                    kind=item.kind,
                    digest=item.digest,
                    summary=item.summary,
                    path=item.path,
                    payload=dict(item.payload),
                    included=True,
                    resolved_deterministically=True,
                    ordinal=item.ordinal,
                )
            )
            continue
        # On deterministic hit, residual is empty except requirements projection.
        if decision.is_hit and kind not in {
            DeltaItemKind.VALIDATION,
            DeltaItemKind.PROOF,
            DeltaItemKind.EFFECT,
            DeltaItemKind.TASK,
        }:
            continue
        residual_items.append(item)

    # Suppressed packets keep identity material but empty provider residual.
    if decision.is_suppressed:
        residual_for_budget = [
            item
            for item in residual_items
            if (
                item.kind
                if isinstance(item.kind, DeltaItemKind)
                else DeltaItemKind.coerce(item.kind)
            )
            in {
                DeltaItemKind.TASK,
                DeltaItemKind.VALIDATION,
                DeltaItemKind.PROOF,
                DeltaItemKind.EFFECT,
            }
        ]
    else:
        residual_for_budget = residual_items

    force_kinds = frozenset(
        {
            DeltaItemKind.TASK,
            DeltaItemKind.VALIDATION,
            DeltaItemKind.PROOF,
            DeltaItemKind.EFFECT,
        }
    )
    included, omitted, disposition, reasons = _apply_budgets(
        residual_for_budget,
        request.budget,
        force_kinds=force_kinds,
    )

    if decision.is_hit and not omitted:
        disposition = FrontierDisposition.DETERMINISTIC_RESOLVED

    if omitted and disposition is FrontierDisposition.BUDGET_OVERFLOW:
        completeness = Completeness.OVERFLOW
    elif omitted:
        completeness = Completeness.PARTIAL_WITH_FRONTIER
    elif decision.is_hit:
        completeness = Completeness.COMPLETE
    else:
        completeness = Completeness.COMPLETE

    frontier = PacketFrontier(
        disposition=disposition,
        omitted_item_ids=tuple(item.item_id for item in omitted),
        omitted_kinds=tuple(
            sorted(
                {
                    item.kind.value
                    if isinstance(item.kind, DeltaItemKind)
                    else str(item.kind)
                    for item in omitted
                }
            )
        ),
        expansion_handles=tuple(
            item.expansion_handle for item in omitted if item.expansion_handle
        ),
        reasons=tuple(dict.fromkeys(reasons)),
        has_more=bool(omitted),
    )

    failure_sig = request.failure_signature_id or compute_failure_signature_id(
        task_cid=request.task_cid,
        evidence_digest=request.evidence_digest,
        policy_id=request.policy_id,
    )

    packet = DeltaTaskPacket(
        packet_id="",
        task_cid=request.task_cid,
        repository_id=request.repository_id,
        tree_id=request.tree_id,
        context_cid=request.context_cid,
        plan_cid=request.plan_cid,
        policy_id=request.policy_id,
        policy_digest=request.policy_digest,
        schema_revision=request.schema_revision,
        evidence_digest=request.evidence_digest,
        effect_scope=effect_scope,
        requirements=requirements,
        unresolved_delta=tuple(included),
        resolved_delta=tuple(resolved_items),
        frontier=frontier,
        completeness=completeness,
        budget=request.budget,
        counterexample_digest=request.counterexample_digest(),
        failure_signature_id=failure_sig,
        decision=decision,
    )
    return packet


def admit_provider_packet(
    packet: DeltaTaskPacket,
) -> dict[str, Any]:
    """Admit a provider-facing packet or raise if dispatch is forbidden."""

    if packet.decision is not None and packet.decision.is_suppressed:
        raise DeltaTaskPacketSuppressedError(
            "unchanged failure cannot churn: replay circuit open",
            circuit_id=packet.decision.circuit_id,
        )
    if packet.decision is not None and packet.decision.is_hit:
        raise DeltaTaskPacketIntegrityError(
            "deterministic hit does not admit a provider packet",
            reason_code="deterministic_hit",
        )
    return packet.provider_packet()


def bind_provider_reply(
    packet: DeltaTaskPacket,
    reply: Mapping[str, Any],
    *,
    claimed_context_cid: str = "",
    claimed_effect_scope_digest: str = "",
    claimed_paths: Sequence[str] | None = None,
    claimed_effects: Sequence[str] | None = None,
) -> PacketReplyBinding:
    """Bind a provider reply to the exact packet context and effect scope.

    Fail closed when the reply claims a different context, effect scope,
    path outside the write ceiling, or effect outside the allowed set.
    """

    try:
        cleaned = _reject_or_redact_secrets(
            _strip_noise(dict(reply)), reject=True
        )
    except DeltaTaskPacketAuthorityError:
        return PacketReplyBinding(
            binding_id="",
            packet_id=packet.packet_id,
            context_cid=packet.context_cid,
            effect_scope_digest=packet.effect_scope.scope_digest,
            requirements_digest=packet.requirements.requirements_digest,
            reply_digest=_content_digest(_strip_noise(dict(reply))),
            accepted=False,
            reason="authority_claim_rejected",
        )
    except DeltaTaskPacketSecretError:
        return PacketReplyBinding(
            binding_id="",
            packet_id=packet.packet_id,
            context_cid=packet.context_cid,
            effect_scope_digest=packet.effect_scope.scope_digest,
            requirements_digest=packet.requirements.requirements_digest,
            reply_digest=_content_digest({"rejected": "secret_material"}),
            accepted=False,
            reason="secret_material_rejected",
        )
    reply_digest = _content_digest(cleaned)

    if claimed_context_cid and claimed_context_cid != packet.context_cid:
        return PacketReplyBinding(
            binding_id="",
            packet_id=packet.packet_id,
            context_cid=packet.context_cid,
            effect_scope_digest=packet.effect_scope.scope_digest,
            requirements_digest=packet.requirements.requirements_digest,
            reply_digest=reply_digest,
            accepted=False,
            reason="context_cid_mismatch",
        )

    if (
        claimed_effect_scope_digest
        and claimed_effect_scope_digest != packet.effect_scope.scope_digest
    ):
        return PacketReplyBinding(
            binding_id="",
            packet_id=packet.packet_id,
            context_cid=packet.context_cid,
            effect_scope_digest=packet.effect_scope.scope_digest,
            requirements_digest=packet.requirements.requirements_digest,
            reply_digest=reply_digest,
            accepted=False,
            reason="effect_scope_mismatch",
        )

    # Nested reply claims.
    nested_context = str(
        cleaned.get("context_cid") or cleaned.get("context_id") or ""
    ).strip()
    if nested_context and nested_context != packet.context_cid:
        return PacketReplyBinding(
            binding_id="",
            packet_id=packet.packet_id,
            context_cid=packet.context_cid,
            effect_scope_digest=packet.effect_scope.scope_digest,
            requirements_digest=packet.requirements.requirements_digest,
            reply_digest=reply_digest,
            accepted=False,
            reason="reply_context_escape",
        )

    nested_packet = str(cleaned.get("packet_id") or "").strip()
    if nested_packet and nested_packet != packet.packet_id:
        return PacketReplyBinding(
            binding_id="",
            packet_id=packet.packet_id,
            context_cid=packet.context_cid,
            effect_scope_digest=packet.effect_scope.scope_digest,
            requirements_digest=packet.requirements.requirements_digest,
            reply_digest=reply_digest,
            accepted=False,
            reason="reply_packet_mismatch",
        )

    paths = list(claimed_paths or ())
    if "write_paths" in cleaned and isinstance(cleaned["write_paths"], Sequence):
        paths.extend(str(item) for item in cleaned["write_paths"])
    if "path" in cleaned:
        paths.append(str(cleaned["path"]))
    for path in paths:
        try:
            if not packet.effect_scope.permits_path(path):
                return PacketReplyBinding(
                    binding_id="",
                    packet_id=packet.packet_id,
                    context_cid=packet.context_cid,
                    effect_scope_digest=packet.effect_scope.scope_digest,
                    requirements_digest=packet.requirements.requirements_digest,
                    reply_digest=reply_digest,
                    accepted=False,
                    reason="path_scope_escape",
                )
        except DeltaTaskPacketError:
            return PacketReplyBinding(
                binding_id="",
                packet_id=packet.packet_id,
                context_cid=packet.context_cid,
                effect_scope_digest=packet.effect_scope.scope_digest,
                requirements_digest=packet.requirements.requirements_digest,
                reply_digest=reply_digest,
                accepted=False,
                reason="path_scope_escape",
            )

    effects = list(claimed_effects or ())
    if "effects" in cleaned and isinstance(cleaned["effects"], Sequence):
        effects.extend(str(item) for item in cleaned["effects"])
    if "effect" in cleaned:
        effects.append(str(cleaned["effect"]))
    for effect in effects:
        try:
            if not packet.effect_scope.permits_effect(effect):
                return PacketReplyBinding(
                    binding_id="",
                    packet_id=packet.packet_id,
                    context_cid=packet.context_cid,
                    effect_scope_digest=packet.effect_scope.scope_digest,
                    requirements_digest=packet.requirements.requirements_digest,
                    reply_digest=reply_digest,
                    accepted=False,
                    reason="effect_scope_escape",
                )
        except DeltaTaskPacketError:
            return PacketReplyBinding(
                binding_id="",
                packet_id=packet.packet_id,
                context_cid=packet.context_cid,
                effect_scope_digest=packet.effect_scope.scope_digest,
                requirements_digest=packet.requirements.requirements_digest,
                reply_digest=reply_digest,
                accepted=False,
                reason="effect_scope_escape",
            )

    # Authority claims in the reply are never accepted as grants.
    for key in _PROVIDER_OMITTED_AUTHORITY_KEYS:
        if cleaned.get(key) is True:
            return PacketReplyBinding(
                binding_id="",
                packet_id=packet.packet_id,
                context_cid=packet.context_cid,
                effect_scope_digest=packet.effect_scope.scope_digest,
                requirements_digest=packet.requirements.requirements_digest,
                reply_digest=reply_digest,
                accepted=False,
                reason="authority_claim_rejected",
            )

    return PacketReplyBinding(
        binding_id="",
        packet_id=packet.packet_id,
        context_cid=packet.context_cid,
        effect_scope_digest=packet.effect_scope.scope_digest,
        requirements_digest=packet.requirements.requirements_digest,
        reply_digest=reply_digest,
        accepted=True,
        reason="bound",
    )


def record_unchanged_failure(
    request: DeltaTaskPacketRequest,
    *,
    circuit_store: ReplayCircuitStore,
    proposal_digest: str = "",
    now_ms: int | None = None,
) -> ReplayCircuit:
    """Record a failed proposal against unchanged evidence for circuit control.

    Circuit identity is intentionally evidence-bound (not proposal-bound) so
    repeated distinct proposals against the same frozen evidence still open the
    typed circuit. ``proposal_digest`` is accepted for callers/logs but does not
    diversify the suppression key.
    """

    del proposal_digest  # retained for API compatibility; not part of circuit key
    failure_sig = request.failure_signature_id or compute_failure_signature_id(
        task_cid=request.task_cid,
        evidence_digest=request.evidence_digest,
        policy_id=request.policy_id,
    )
    return circuit_store.record_failure(
        failure_signature_id=failure_sig,
        evidence_digest=request.evidence_digest,
        task_cid=request.task_cid,
        retry_budget=request.retry_budget,
        now_ms=now_ms,
        reason="unchanged_failure",
    )


class DeltaTaskPacketService:
    """In-process service composing cache, circuit, and packet admission."""

    INTERFACE: Final[str] = DELTA_TASK_PACKET_SERVICE_INTERFACE

    def __init__(
        self,
        *,
        decision_cache: DecisionCache | None = None,
        circuit_store: ReplayCircuitStore | None = None,
        clock_ms: Any = None,
    ) -> None:
        self._cache = decision_cache or DecisionCache()
        self._circuits = circuit_store or ReplayCircuitStore()
        self._clock_ms = clock_ms

    def _now(self) -> int:
        if self._clock_ms is None:
            return _now_ms()
        if callable(self._clock_ms):
            return int(self._clock_ms())
        return int(self._clock_ms)

    @property
    def decision_cache(self) -> DecisionCache:
        return self._cache

    @property
    def circuit_store(self) -> ReplayCircuitStore:
        return self._circuits

    def evaluate(
        self, request: DeltaTaskPacketRequest
    ) -> DeterministicFirstDecision:
        return evaluate_deterministic_first(
            request,
            decision_cache=self._cache,
            circuit_store=self._circuits,
            now_ms=self._now(),
        )

    def build(self, request: DeltaTaskPacketRequest) -> DeltaTaskPacket:
        decision = self.evaluate(request)
        return build_delta_task_packet(
            request,
            decision=decision,
            decision_cache=self._cache,
            circuit_store=self._circuits,
            now_ms=self._now(),
        )

    def admit(self, request: DeltaTaskPacketRequest) -> dict[str, Any]:
        packet = self.build(request)
        return admit_provider_packet(packet)

    def remember_resolution(
        self,
        request: DeltaTaskPacketRequest,
        *,
        resolved_item_ids: Sequence[str],
    ) -> None:
        requirements = ValidationProofRequirements(
            validation_commands=_normalize_commands(request.validations),
            proof_obligations=_normalize_ids(
                request.proof_obligations,
                name="proof_obligation",
                limit=MAX_PROOFS,
            ),
            acceptance_ids=_normalize_ids(
                request.acceptance_ids, name="acceptance", limit=MAX_PROOFS
            ),
            require_validation=request.require_validation,
            require_proof=request.require_proof,
        )
        self._cache.put(
            request.cache_key(),
            resolution={"resolved_item_ids": list(resolved_item_ids)},
            requirements_digest=requirements.requirements_digest,
        )

    def record_failure(
        self,
        request: DeltaTaskPacketRequest,
        *,
        proposal_digest: str = "",
    ) -> ReplayCircuit:
        return record_unchanged_failure(
            request,
            circuit_store=self._circuits,
            proposal_digest=proposal_digest,
            now_ms=self._now(),
        )

    def bind_reply(
        self,
        packet: DeltaTaskPacket,
        reply: Mapping[str, Any],
        **kwargs: Any,
    ) -> PacketReplyBinding:
        return bind_provider_reply(packet, reply, **kwargs)


__all__ = [
    "AUTHORITY_CLASS",
    "Completeness",
    "DEFAULT_ALLOWED_EFFECTS",
    "DEFAULT_POLICY_ID",
    "DEFAULT_RETRY_BUDGET",
    "DELTA_TASK_PACKET_INTERFACE",
    "DELTA_TASK_PACKET_SCHEMA",
    "DELTA_TASK_PACKET_SERVICE_INTERFACE",
    "DETERMINISTIC_FIRST_DECISION_INTERFACE",
    "DecisionCache",
    "DecisionSource",
    "DeltaItem",
    "DeltaItemKind",
    "DeltaTaskPacket",
    "DeltaTaskPacketAuthorityError",
    "DeltaTaskPacketBoundsError",
    "DeltaTaskPacketError",
    "DeltaTaskPacketIntegrityError",
    "DeltaTaskPacketOverflowError",
    "DeltaTaskPacketRequest",
    "DeltaTaskPacketScopeError",
    "DeltaTaskPacketSecretError",
    "DeltaTaskPacketService",
    "DeltaTaskPacketStaleError",
    "DeltaTaskPacketSuppressedError",
    "DeterministicAction",
    "DeterministicFirstDecision",
    "EffectScope",
    "FrontierDisposition",
    "PACKET_REPLY_BINDING_INTERFACE",
    "PRODUCER_ID",
    "PacketBudget",
    "PacketFrontier",
    "PacketReplyBinding",
    "PROVIDER_FACING_PACKET_SCHEMA",
    "REDACTION_MARKER",
    "ReplayCircuit",
    "ReplayCircuitStore",
    "UNTRUSTED_DATA_LABEL",
    "ValidationProofRequirements",
    "admit_provider_packet",
    "bind_provider_reply",
    "build_delta_task_packet",
    "compute_failure_signature_id",
    "evaluate_deterministic_first",
    "record_unchanged_failure",
]
