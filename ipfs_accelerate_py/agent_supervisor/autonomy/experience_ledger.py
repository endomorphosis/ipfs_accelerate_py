"""Bounded content-addressed ledger for compact public experience episodes.

The ledger persists only the closed :class:`ExperienceEpisode` field set plus
typed memory/index metadata.  It is an index over already content-addressed
identities and receipts, not a second artifact store, prompt archive, or
authority source.

Existing RuntimeCAS and artifact stores remain the physical durability
authorities.  This module talks to them only through an injected
:class:`ExperienceStoreAdapter` and never opens, mutates, or replaces those
backends.  Cold import performs no filesystem, network, or provider action.

Admission rules
---------------
* Unknown, secret, prompt, source-body, transcript, and private-reasoning
  fields are rejected before an episode is sealed.
* Context metrics stay compact integer units; nested bodies are refused.
* Evidence authority may be constrained but never raised.  Frequency, replay,
  lookup, compaction, and store round-trips cannot mint a stronger class.
* Invalidation withdraws only direct dependants.  Independent episodes and
  accepted history remain.  A withdrawn or dependency-invalidated episode
  cannot be revived as current authority.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, replace
from enum import Enum
from threading import RLock
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from ..proof.formal_verification_contracts import (
    CONTRACT_VERSION,
    canonical_json_bytes,
    content_identity,
)
from .contracts import (
    MAX_CANONICAL_RECORD_BYTES,
    MAX_IDENTIFIER_BYTES,
    MAX_INTEGER,
    MAX_MAPPING_ITEMS,
    MAX_NESTING_DEPTH,
    MAX_SEQUENCE_ITEMS,
    AuthorityClass,
    AutonomyContractError,
    ExperienceEpisode,
    MemoryClass,
    TerminalStatus,
)

EXPERIENCE_LEDGER_INTERFACE = "ExperienceLedger@1"
EXPERIENCE_EPISODE_INTERFACE = "ExperienceEpisode@1"
EXPERIENCE_LEDGER_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/experience-ledger@1"
)
EXPERIENCE_RECORD_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/experience-record@1"
)
EXPERIENCE_PROJECTION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/experience-projection@1"
)
EXPERIENCE_INDEX_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/experience-dependency-index@1"
)
EXPERIENCE_INVALIDATION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/experience-invalidation@1"
)
EXPERIENCE_SNAPSHOT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/experience-ledger-snapshot@1"
)

ALLOWED_COMPACT_EPISODE_FIELDS = frozenset(
    item.name for item in fields(ExperienceEpisode)
)
ALLOWED_EPISODE_ENVELOPE_FIELDS = frozenset(
    {
        "schema",
        "contract_version",
        "content_id",
        *ExperienceEpisode.IDENTITY_ALIASES,
        *ALLOWED_COMPACT_EPISODE_FIELDS,
    }
)
FORBIDDEN_LEDGER_FIELD_MARKERS = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "authorization",
        "auth_token",
        "bearer",
        "chain_of_thought",
        "chat_messages",
        "client_secret",
        "completion",
        "completions",
        "completion_body",
        "completion_text",
        "cookie",
        "credential",
        "credentials",
        "decoded_source",
        "decoded_text",
        "executable_code",
        "file_content",
        "hidden_reasoning",
        "hidden_witness",
        "messages",
        "model_output",
        "model_transcript",
        "passphrase",
        "passwd",
        "password",
        "private_key",
        "private_reasoning",
        "prompt",
        "prompts",
        "prompt_body",
        "prompt_text",
        "raw_completion",
        "raw_prompt",
        "raw_secret",
        "refresh_token",
        "repository_dump",
        "response_body",
        "response_text",
        "secret",
        "secrets",
        "session_token",
        "shell_command",
        "source_bodies",
        "source_body",
        "source_text",
        "token",
        "transcript",
        "transcripts",
    }
)

_AUTHORITY_RANK = {
    AuthorityClass.NONE: 0,
    AuthorityClass.ADVISORY: 1,
    AuthorityClass.DERIVED: 2,
    AuthorityClass.VERIFIED: 3,
    AuthorityClass.AUTHORITATIVE: 4,
    # Constraint, not an evidence upgrade.  It cannot outrank authoritative
    # evidence and cannot be used to mint a stronger stored class.
    AuthorityClass.OPERATOR_REQUIRED: 4,
}

_DEFAULT_TTL_MS = {
    MemoryClass.EPHEMERAL_ATTEMPT: 5 * 60 * 1000,
    MemoryClass.SHORT_LIVED_NEGATIVE: 5 * 60 * 1000,
    MemoryClass.TASK_EPISODE: 0,
    MemoryClass.REPOSITORY_PATTERN: 0,
    MemoryClass.CROSS_REPOSITORY_RULE: 0,
    MemoryClass.AUTHORITATIVE_CURRENT: 0,
    MemoryClass.WITHDRAWN: 0,
}

_COMPACTABLE_MEMORY = frozenset(
    {MemoryClass.EPHEMERAL_ATTEMPT, MemoryClass.SHORT_LIVED_NEGATIVE}
)

_RESULT_IDENTITY_FIELDS = (
    "evidence_ids",
    "validation_receipt_ids",
    "proof_receipt_ids",
    "merge_receipt_ids",
)
_DEPENDENCY_IDENTITY_FIELDS = _RESULT_IDENTITY_FIELDS + (
    "frozen_input_ids",
    "token_measurement_ids",
)

MAX_LEDGER_EPISODES = MAX_SEQUENCE_ITEMS
MAX_SCOPE_ITEMS = MAX_MAPPING_ITEMS
MAX_DEPENDENCY_INDEX_KEYS = 4_096
STORE_PROJECTION_PREFIX = "experience-projection/"
STORE_RECORD_PREFIX = "experience-record/"
STORE_INDEX_PREFIX = "experience-index/"
STORE_HEAD_KEY = "experience-ledger/head"

_TEXT_SECRET_PATTERNS = (
    re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(
        r"(?i)\b(api[_ -]?key|access[_ -]?token|auth[_ -]?token|"
        r"client[_ -]?secret|password|passphrase|secret)"
        r"(\s*[:=]\s*)[^\s,;]{4,}"
    ),
    re.compile(
        "-----"
        + "BEGIN "
        + r"(?:[A-Z0-9]+ )?"
        + "PRIVATE "
        + "KEY"
        + "-----"
        + ".*?"
        + "-----"
        + "END "
        + r"(?:[A-Z0-9]+ )?"
        + "PRIVATE "
        + "KEY"
        + "-----",
        re.DOTALL,
    ),
)


class ExperienceLedgerError(AutonomyContractError):
    """Raised when an experience ledger admission or replay rule fails."""


@runtime_checkable
class ExperienceStoreAdapter(Protocol):
    """Durability port over an existing CAS or artifact adapter.

    Implementations must treat ``key`` as an opaque content-addressed name
    and ``payload`` as already-canonical public bytes.  The ledger never
    sends raw prompts, source bodies, transcripts, or secret material.
    """

    def put(self, key: str, payload: bytes) -> None: ...

    def get(self, key: str) -> bytes | None: ...


class InMemoryExperienceStore:
    """Process-local adapter used by tests and by ledgers without a CAS."""

    def __init__(self) -> None:
        self._items: dict[str, bytes] = {}

    def put(self, key: str, payload: bytes) -> None:
        if not isinstance(key, str) or not key or any(char.isspace() for char in key):
            raise ExperienceLedgerError("store key must be a compact identifier")
        if not isinstance(payload, (bytes, bytearray)):
            raise ExperienceLedgerError("store payload must be canonical bytes")
        self._items[key] = bytes(payload)

    def get(self, key: str) -> bytes | None:
        return self._items.get(key)

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))

    def items(self) -> Mapping[str, bytes]:
        return MappingProxyType(dict(self._items))


def _enum(value: Any, enum_type: type[Enum], name: str) -> Any:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value))
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise ExperienceLedgerError(f"{name} must be one of: {allowed}") from exc


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ExperienceLedgerError(f"{name} must be a boolean")
    return value


def _int(value: Any, name: str, *, maximum: int = MAX_INTEGER) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > maximum:
        raise ExperienceLedgerError(f"{name} must be an integer between 0 and {maximum}")
    return value


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ExperienceLedgerError(f"{name} must be a string")
    result = unicodedata.normalize("NFC", value.strip())
    if (
        not result
        or len(result.encode("utf-8")) > MAX_IDENTIFIER_BYTES
        or any(char.isspace() for char in result)
        or "\x00" in result
    ):
        raise ExperienceLedgerError(f"{name} must be a compact bounded identifier")
    _reject_secret_text(result, name)
    return result


def _identifiers(values: Any, name: str, *, required: bool = False) -> tuple[str, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, str):
        raw = (values,)
    elif isinstance(values, Sequence) and not isinstance(values, (bytes, bytearray)):
        raw = values
    else:
        raise ExperienceLedgerError(f"{name} must be a sequence of identifiers")
    if len(raw) > MAX_SEQUENCE_ITEMS:
        raise ExperienceLedgerError(f"{name} contains too many items")
    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw:
        identifier = _identifier(item, name)
        if identifier not in seen:
            seen.add(identifier)
            normalized.append(identifier)
    if required and not normalized:
        raise ExperienceLedgerError(f"{name} must not be empty")
    return tuple(sorted(normalized))


def _normalize_field_name(key: Any) -> str:
    if not isinstance(key, str):
        raise ExperienceLedgerError("experience field names must be strings")
    return unicodedata.normalize("NFC", key.strip()).lower().replace("-", "_")


def field_is_forbidden(key: Any) -> bool:
    """Return whether a mapping key is a secret, prompt, source, or transcript."""

    normalized = _normalize_field_name(key)
    if not normalized:
        return False
    return any(
        normalized == marker or normalized.endswith("_" + marker)
        for marker in FORBIDDEN_LEDGER_FIELD_MARKERS
    )


def _reject_secret_text(value: str, name: str) -> None:
    if any(pattern.search(value) for pattern in _TEXT_SECRET_PATTERNS):
        raise ExperienceLedgerError(f"{name} contains forbidden secret material")


def _reject_forbidden_payload(value: Any, name: str, *, depth: int = 0) -> None:
    if depth > MAX_NESTING_DEPTH:
        raise ExperienceLedgerError(f"{name} exceeds maximum nesting")
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float):
        raise ExperienceLedgerError(f"{name} cannot contain floats")
    if isinstance(value, Enum):
        return
    if isinstance(value, str):
        _reject_secret_text(value, name)
        return
    if isinstance(value, Mapping):
        if len(value) > MAX_MAPPING_ITEMS:
            raise ExperienceLedgerError(f"{name} contains too many entries")
        for raw_key, raw_value in value.items():
            key_name = f"{name}.{raw_key}"
            if field_is_forbidden(raw_key):
                raise ExperienceLedgerError(
                    f"{name} contains forbidden private or executable data"
                )
            _reject_forbidden_payload(raw_value, key_name, depth=depth + 1)
        return
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        if len(value) > MAX_SEQUENCE_ITEMS:
            raise ExperienceLedgerError(f"{name} contains too many items")
        for index, item in enumerate(value):
            _reject_forbidden_payload(item, f"{name}[{index}]", depth=depth + 1)
        return
    if isinstance(value, ExperienceEpisode):
        _reject_forbidden_payload(value.to_dict(), name, depth=depth + 1)
        return
    raise ExperienceLedgerError(f"{name} contains unsupported value type {type(value).__name__}")


def _reject_unknown_episode_fields(payload: Mapping[str, Any]) -> None:
    extra = set(payload).difference(ALLOWED_EPISODE_ENVELOPE_FIELDS)
    if not extra:
        return
    if any(field_is_forbidden(key) for key in extra):
        raise ExperienceLedgerError("episode contains forbidden private or executable data")
    raise ExperienceLedgerError(
        "episode contains unsupported compact fields; rebuild its canonical payload"
    )


def authority_rank(value: AuthorityClass | str) -> int:
    authority = _enum(value, AuthorityClass, "authority_class")
    return _AUTHORITY_RANK[authority]


def min_authority(values: Sequence[AuthorityClass | str]) -> AuthorityClass:
    if not values:
        return AuthorityClass.NONE
    return min(
        (_enum(item, AuthorityClass, "authority_class") for item in values),
        key=authority_rank,
    )


def _cannot_upgrade(stored: AuthorityClass, claimed: AuthorityClass) -> None:
    if authority_rank(claimed) > authority_rank(stored):
        raise ExperienceLedgerError("experience ledger cannot upgrade evidence authority")


def compact_episode_payload(episode: ExperienceEpisode) -> dict[str, Any]:
    """Return the public episode projection with only allowed compact fields."""

    if not isinstance(episode, ExperienceEpisode):
        raise ExperienceLedgerError("episode must be an ExperienceEpisode")
    payload = episode.to_dict()
    _reject_forbidden_payload(payload, "episode")
    _reject_unknown_episode_fields(payload)
    return payload


def public_episode_projection(record: "ExperienceRecord") -> dict[str, Any]:
    """Return the durable public projection stored for one episode."""

    if not isinstance(record, ExperienceRecord):
        raise ExperienceLedgerError("projection requires an ExperienceRecord")
    payload = record.to_dict()
    _reject_forbidden_payload(payload, "projection")
    return payload


def _result_identity_ids(episode: ExperienceEpisode) -> tuple[str, ...]:
    collected: list[str] = []
    seen: set[str] = set()
    for name in _RESULT_IDENTITY_FIELDS:
        for item in getattr(episode, name):
            if item not in seen:
                seen.add(item)
                collected.append(item)
    return tuple(sorted(collected))


def _dependency_ids(
    episode: ExperienceEpisode,
    extra: Sequence[str] = (),
) -> tuple[str, ...]:
    collected: list[str] = []
    seen: set[str] = set()
    for name in _DEPENDENCY_IDENTITY_FIELDS:
        for item in getattr(episode, name):
            if item not in seen:
                seen.add(item)
                collected.append(item)
    for item in extra:
        identifier = _identifier(item, "invalidation_dependency_ids")
        if identifier not in seen:
            seen.add(identifier)
            collected.append(identifier)
    return tuple(sorted(collected))


def _coerce_episode(value: Any) -> ExperienceEpisode:
    if isinstance(value, ExperienceEpisode):
        _reject_forbidden_payload(value.to_dict(), "episode")
        _reject_unknown_episode_fields(value.to_dict())
        _reject_noncompact_metrics(value)
        return value
    if not isinstance(value, Mapping):
        raise ExperienceLedgerError("episode must be an ExperienceEpisode or mapping")
    _reject_forbidden_payload(value, "episode")
    _reject_unknown_episode_fields(value)
    try:
        episode = ExperienceEpisode.from_dict(value)
    except AutonomyContractError as exc:
        raise ExperienceLedgerError(str(exc)) from exc
    _reject_noncompact_metrics(episode)
    return episode


def _reject_noncompact_metrics(episode: ExperienceEpisode) -> None:
    metrics = episode.context_metrics
    if not isinstance(metrics, Mapping):
        raise ExperienceLedgerError("context_metrics must be a mapping of integer units")
    for key, raw in metrics.items():
        if field_is_forbidden(key):
            raise ExperienceLedgerError(
                "context_metrics contains forbidden private or executable data"
            )
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0 or raw > MAX_INTEGER:
            raise ExperienceLedgerError(
                "context_metrics must be compact non-negative integer units"
            )


def _evidence_authority_map(
    value: Any,
    episode: ExperienceEpisode,
) -> Mapping[str, AuthorityClass]:
    if value is None:
        raw: Mapping[str, Any] = {}
    elif isinstance(value, Mapping):
        raw = value
    else:
        raise ExperienceLedgerError("evidence_authority must be a mapping")
    if len(raw) > MAX_MAPPING_ITEMS:
        raise ExperienceLedgerError("evidence_authority contains too many entries")
    result_ids = set(_result_identity_ids(episode))
    normalized: dict[str, AuthorityClass] = {}
    for raw_key, raw_value in raw.items():
        evidence_id = _identifier(raw_key, "evidence_authority")
        if evidence_id not in result_ids:
            raise ExperienceLedgerError(
                "evidence_authority references an identity the episode does not cite"
            )
        normalized[evidence_id] = _enum(raw_value, AuthorityClass, "evidence_authority")
    for evidence_id in result_ids:
        normalized.setdefault(evidence_id, AuthorityClass.NONE)
    return MappingProxyType(dict(sorted(normalized.items())))


def _bound_authority(
    *,
    episode: ExperienceEpisode,
    evidence_authority: Mapping[str, AuthorityClass],
    claimed_authority: Any,
    memory_class: MemoryClass,
) -> AuthorityClass:
    ceiling = min_authority(tuple(evidence_authority.values()))
    if not evidence_authority:
        ceiling = AuthorityClass.NONE
    if claimed_authority is None:
        admitted = ceiling
    else:
        admitted = _enum(claimed_authority, AuthorityClass, "claimed_authority")
        _cannot_upgrade(ceiling, admitted)
    if admitted is AuthorityClass.AUTHORITATIVE:
        if not evidence_authority or any(
            item is not AuthorityClass.AUTHORITATIVE for item in evidence_authority.values()
        ):
            raise ExperienceLedgerError("experience ledger cannot upgrade evidence authority")
    if memory_class is MemoryClass.AUTHORITATIVE_CURRENT:
        if admitted is not AuthorityClass.AUTHORITATIVE:
            raise ExperienceLedgerError(
                "authoritative_current memory requires authoritative evidence"
            )
        if episode.terminal_status not in {TerminalStatus.SUCCEEDED, TerminalStatus.PENDING}:
            raise ExperienceLedgerError(
                "authoritative_current memory requires a current accepted or pending outcome"
            )
    return admitted


@dataclass(frozen=True)
class ExperienceRecord:
    """Immutable public ledger row: compact episode plus index metadata."""

    episode: ExperienceEpisode
    memory_class: MemoryClass
    authority_class: AuthorityClass
    evidence_authority: Mapping[str, AuthorityClass]
    invalidation_dependency_ids: tuple[str, ...]
    scope_ids: tuple[str, ...] = ()
    ttl_ms: int = 0
    expires_at_ms: int = 0
    recorded_at_ms: int = 0
    withdrawn: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.episode, ExperienceEpisode):
            raise ExperienceLedgerError("record episode must be an ExperienceEpisode")
        object.__setattr__(
            self, "memory_class", _enum(self.memory_class, MemoryClass, "memory_class")
        )
        object.__setattr__(
            self,
            "authority_class",
            _enum(self.authority_class, AuthorityClass, "authority_class"),
        )
        object.__setattr__(
            self,
            "evidence_authority",
            _evidence_authority_map(self.evidence_authority, self.episode),
        )
        object.__setattr__(
            self,
            "invalidation_dependency_ids",
            _identifiers(self.invalidation_dependency_ids, "invalidation_dependency_ids"),
        )
        object.__setattr__(self, "scope_ids", _identifiers(self.scope_ids, "scope_ids"))
        if len(self.scope_ids) > MAX_SCOPE_ITEMS:
            raise ExperienceLedgerError("scope_ids contains too many items")
        object.__setattr__(self, "ttl_ms", _int(self.ttl_ms, "ttl_ms"))
        object.__setattr__(self, "expires_at_ms", _int(self.expires_at_ms, "expires_at_ms"))
        object.__setattr__(self, "recorded_at_ms", _int(self.recorded_at_ms, "recorded_at_ms"))
        object.__setattr__(self, "withdrawn", _bool(self.withdrawn, "withdrawn"))
        if self.withdrawn and self.memory_class is not MemoryClass.WITHDRAWN:
            object.__setattr__(self, "memory_class", MemoryClass.WITHDRAWN)
        if self.memory_class is MemoryClass.WITHDRAWN and not self.withdrawn:
            object.__setattr__(self, "withdrawn", True)
        _cannot_upgrade(
            min_authority(tuple(self.evidence_authority.values()))
            if self.evidence_authority
            else AuthorityClass.NONE,
            self.authority_class,
        )
        payload = self.to_dict()
        _reject_forbidden_payload(payload, "record")
        encoded = canonical_json_bytes(payload)
        if len(encoded) > MAX_CANONICAL_RECORD_BYTES:
            raise ExperienceLedgerError("experience record exceeds its bounded canonical size")

    @property
    def episode_id(self) -> str:
        return self.episode.episode_id

    @property
    def record_id(self) -> str:
        return content_identity(self.to_dict())

    def is_current(self, now_ms: int = 0) -> bool:
        now = _int(now_ms, "now_ms")
        if self.withdrawn or self.memory_class is MemoryClass.WITHDRAWN:
            return False
        if self.expires_at_ms and now >= self.expires_at_ms:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        evidence = {
            key: value.value if isinstance(value, AuthorityClass) else str(value)
            for key, value in self.evidence_authority.items()
        }
        return {
            "schema": EXPERIENCE_RECORD_SCHEMA,
            "contract_version": CONTRACT_VERSION,
            "episode": compact_episode_payload(self.episode),
            "memory_class": self.memory_class.value,
            "authority_class": self.authority_class.value,
            "evidence_authority": dict(sorted(evidence.items())),
            "invalidation_dependency_ids": list(self.invalidation_dependency_ids),
            "scope_ids": list(self.scope_ids),
            "ttl_ms": self.ttl_ms,
            "expires_at_ms": self.expires_at_ms,
            "recorded_at_ms": self.recorded_at_ms,
            "withdrawn": self.withdrawn,
        }

    def to_public_projection(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload["schema"] = EXPERIENCE_PROJECTION_SCHEMA
        payload["episode_id"] = self.episode_id
        payload["record_id"] = self.record_id
        payload["content_id"] = self.record_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ExperienceRecord:
        if not isinstance(payload, Mapping):
            raise ExperienceLedgerError("experience record payload must be an object")
        _reject_forbidden_payload(payload, "record")
        schema = payload.get("schema")
        if schema not in (None, "", EXPERIENCE_RECORD_SCHEMA, EXPERIENCE_PROJECTION_SCHEMA):
            raise ExperienceLedgerError(
                f"unsupported experience record schema; use {EXPERIENCE_RECORD_SCHEMA}"
            )
        if payload.get("contract_version", CONTRACT_VERSION) != CONTRACT_VERSION:
            raise ExperienceLedgerError("unsupported experience record version")
        allowed = {
            "schema",
            "contract_version",
            "content_id",
            "episode_id",
            "record_id",
            "episode",
            "memory_class",
            "authority_class",
            "evidence_authority",
            "invalidation_dependency_ids",
            "scope_ids",
            "ttl_ms",
            "expires_at_ms",
            "recorded_at_ms",
            "withdrawn",
        }
        extra = set(payload).difference(allowed)
        if extra:
            if any(field_is_forbidden(key) for key in extra):
                raise ExperienceLedgerError(
                    "record contains forbidden private or executable data"
                )
            raise ExperienceLedgerError("experience record contains unsupported fields")
        episode_payload = payload.get("episode")
        episode = _coerce_episode(episode_payload)
        if payload.get("episode_id") not in (None, "", episode.episode_id):
            raise ExperienceLedgerError("episode identity does not match payload")
        record = cls(
            episode=episode,
            memory_class=payload.get("memory_class", MemoryClass.TASK_EPISODE),
            authority_class=payload.get("authority_class", AuthorityClass.NONE),
            evidence_authority=payload.get("evidence_authority") or {},
            invalidation_dependency_ids=payload.get("invalidation_dependency_ids") or (),
            scope_ids=payload.get("scope_ids") or (),
            ttl_ms=payload.get("ttl_ms", 0),
            expires_at_ms=payload.get("expires_at_ms", 0),
            recorded_at_ms=payload.get("recorded_at_ms", 0),
            withdrawn=payload.get("withdrawn", False),
        )
        if payload.get("record_id") not in (None, "", record.record_id):
            raise ExperienceLedgerError("experience record identity does not match payload")
        claimed_content = payload.get("content_id")
        if claimed_content not in (None, "", record.record_id, episode.episode_id):
            raise ExperienceLedgerError("experience record identity does not match payload")
        return record


@dataclass(frozen=True)
class ExperienceInvalidationReceipt:
    """Canonical receipt naming the direct dependants of one identity."""

    dependency_id: str
    invalidated_episode_ids: tuple[str, ...]
    retained_episode_ids: tuple[str, ...]
    recorded_at_ms: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "dependency_id", _identifier(self.dependency_id, "dependency_id"))
        object.__setattr__(
            self,
            "invalidated_episode_ids",
            _identifiers(self.invalidated_episode_ids, "invalidated_episode_ids"),
        )
        object.__setattr__(
            self,
            "retained_episode_ids",
            _identifiers(self.retained_episode_ids, "retained_episode_ids"),
        )
        overlap = set(self.invalidated_episode_ids).intersection(self.retained_episode_ids)
        if overlap:
            raise ExperienceLedgerError("invalidation cannot both withdraw and retain an episode")
        object.__setattr__(self, "recorded_at_ms", _int(self.recorded_at_ms, "recorded_at_ms"))

    @property
    def receipt_id(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": EXPERIENCE_INVALIDATION_SCHEMA,
            "contract_version": CONTRACT_VERSION,
            "dependency_id": self.dependency_id,
            "invalidated_episode_ids": list(self.invalidated_episode_ids),
            "retained_episode_ids": list(self.retained_episode_ids),
            "recorded_at_ms": self.recorded_at_ms,
        }


@dataclass(frozen=True)
class ExperienceLedgerSnapshot:
    """Canonical restart value for one experience ledger."""

    records: tuple[ExperienceRecord, ...]
    invalidated_dependency_ids: tuple[str, ...] = ()
    epoch: int = 0

    def __post_init__(self) -> None:
        raw = self.records
        if raw is None:
            items: Sequence[Any] = ()
        elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
            items = raw
        else:
            raise ExperienceLedgerError("snapshot records must be a sequence")
        records: list[ExperienceRecord] = []
        for item in items:
            if isinstance(item, ExperienceRecord):
                records.append(item)
            elif isinstance(item, Mapping):
                records.append(ExperienceRecord.from_dict(item))
            else:
                raise ExperienceLedgerError("snapshot records must contain ExperienceRecord values")
        object.__setattr__(self, "records", tuple(records))
        if len(self.records) > MAX_LEDGER_EPISODES:
            raise ExperienceLedgerError("experience ledger exceeds its bounded size")
        ids = [record.episode_id for record in self.records]
        if len(set(ids)) != len(ids):
            raise ExperienceLedgerError("experience snapshot contains duplicate episodes")
        object.__setattr__(
            self,
            "invalidated_dependency_ids",
            _identifiers(self.invalidated_dependency_ids, "invalidated_dependency_ids"),
        )
        object.__setattr__(self, "epoch", _int(self.epoch, "epoch"))
        encoded = canonical_json_bytes(self.to_dict())
        if len(encoded) > MAX_CANONICAL_RECORD_BYTES * 4:
            raise ExperienceLedgerError("experience snapshot exceeds its bounded canonical size")

    @property
    def snapshot_id(self) -> str:
        return content_identity(self.to_dict())

    ledger_id = snapshot_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": EXPERIENCE_SNAPSHOT_SCHEMA,
            "contract_version": CONTRACT_VERSION,
            "epoch": self.epoch,
            "invalidated_dependency_ids": list(self.invalidated_dependency_ids),
            "records": [record.to_dict() for record in self.records],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ExperienceLedgerSnapshot:
        if not isinstance(payload, Mapping):
            raise ExperienceLedgerError("experience snapshot payload must be an object")
        _reject_forbidden_payload(payload, "snapshot")
        if payload.get("schema") not in (None, "", EXPERIENCE_SNAPSHOT_SCHEMA):
            raise ExperienceLedgerError(
                f"unsupported experience snapshot schema; use {EXPERIENCE_SNAPSHOT_SCHEMA}"
            )
        if payload.get("contract_version", CONTRACT_VERSION) != CONTRACT_VERSION:
            raise ExperienceLedgerError("unsupported experience snapshot version")
        allowed = {
            "schema",
            "contract_version",
            "content_id",
            "snapshot_id",
            "ledger_id",
            "epoch",
            "invalidated_dependency_ids",
            "records",
        }
        extra = set(payload).difference(allowed)
        if extra:
            if any(field_is_forbidden(key) for key in extra):
                raise ExperienceLedgerError(
                    "snapshot contains forbidden private or executable data"
                )
            raise ExperienceLedgerError("experience snapshot contains unsupported fields")
        snapshot = cls(
            records=tuple(payload.get("records") or ()),
            invalidated_dependency_ids=payload.get("invalidated_dependency_ids") or (),
            epoch=payload.get("epoch", 0),
        )
        claimed = payload.get("content_id") or payload.get("snapshot_id") or payload.get("ledger_id")
        if claimed not in (None, "", snapshot.snapshot_id):
            raise ExperienceLedgerError("experience snapshot identity does not match payload")
        return snapshot


def _default_clock() -> int:
    return 0


class ExperienceLedger:
    """In-memory compact episode index with optional adapter persistence."""

    INTERFACE = EXPERIENCE_LEDGER_INTERFACE
    SCHEMA = EXPERIENCE_LEDGER_SCHEMA

    def __init__(
        self,
        store: ExperienceStoreAdapter | None = None,
        *,
        clock: Callable[[], int] | None = None,
        max_episodes: int = MAX_LEDGER_EPISODES,
        epoch: int = 0,
    ) -> None:
        if store is not None and not isinstance(store, ExperienceStoreAdapter):
            raise ExperienceLedgerError("store must implement ExperienceStoreAdapter")
        self._store = store
        self._clock = clock if clock is not None else _default_clock
        self._max_episodes = _int(max_episodes, "max_episodes", maximum=MAX_LEDGER_EPISODES)
        if self._max_episodes < 1:
            raise ExperienceLedgerError("max_episodes must be a positive integer")
        self._epoch = _int(epoch, "epoch")
        self._records: dict[str, ExperienceRecord] = {}
        self._dependencies: dict[str, set[str]] = {}
        self._invalidated: set[str] = set()
        self._lock = RLock()

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def store(self) -> ExperienceStoreAdapter | None:
        return self._store

    def _now_ms(self) -> int:
        value = self._clock()
        return _int(value, "clock")

    def _index_record(self, record: ExperienceRecord) -> None:
        self._records[record.episode_id] = record
        for dependency_id in record.invalidation_dependency_ids:
            self._dependencies.setdefault(dependency_id, set()).add(record.episode_id)
        if len(self._dependencies) > MAX_DEPENDENCY_INDEX_KEYS:
            raise ExperienceLedgerError("experience dependency index exceeds its bounded size")

    def _drop_from_index(self, record: ExperienceRecord) -> None:
        self._records.pop(record.episode_id, None)
        for dependency_id in record.invalidation_dependency_ids:
            holders = self._dependencies.get(dependency_id)
            if holders is None:
                continue
            holders.discard(record.episode_id)
            if not holders:
                self._dependencies.pop(dependency_id, None)

    def _persist(self, key: str, payload: Mapping[str, Any]) -> None:
        if self._store is None:
            return
        encoded = canonical_json_bytes(payload)
        if len(encoded) > MAX_CANONICAL_RECORD_BYTES:
            raise ExperienceLedgerError("persisted experience payload exceeds its bounded size")
        _reject_forbidden_payload(payload, "store")
        try:
            self._store.put(key, encoded)
        except ExperienceLedgerError:
            raise
        except Exception as exc:
            raise ExperienceLedgerError("experience store adapter rejected the compact payload") from exc

    def _persist_record(self, record: ExperienceRecord) -> None:
        projection = record.to_public_projection()
        self._persist(STORE_RECORD_PREFIX + record.record_id, record.to_dict())
        self._persist(STORE_PROJECTION_PREFIX + record.episode_id, projection)
        index_payload = {
            "schema": EXPERIENCE_INDEX_SCHEMA,
            "contract_version": CONTRACT_VERSION,
            "episode_id": record.episode_id,
            "invalidation_dependency_ids": list(record.invalidation_dependency_ids),
        }
        self._persist(STORE_INDEX_PREFIX + record.episode_id, index_payload)

    def _persist_head(self) -> None:
        if self._store is None:
            return
        snapshot = self._snapshot_unlocked()
        self._persist(
            STORE_HEAD_KEY,
            {
                "schema": EXPERIENCE_SNAPSHOT_SCHEMA,
                "contract_version": CONTRACT_VERSION,
                "snapshot_id": snapshot.snapshot_id,
                "epoch": snapshot.epoch,
                "episode_ids": [record.episode_id for record in snapshot.records],
                "record_ids": [record.record_id for record in snapshot.records],
                "invalidated_dependency_ids": list(snapshot.invalidated_dependency_ids),
            },
        )

    def _drop_expired_unlocked(self, now_ms: int) -> None:
        expired = [
            record
            for record in tuple(self._records.values())
            if record.memory_class in _COMPACTABLE_MEMORY and not record.is_current(now_ms)
        ]
        for record in expired:
            self._drop_from_index(record)

    def _admit_capacity(self, now_ms: int) -> None:
        if len(self._records) < self._max_episodes:
            return
        self._drop_expired_unlocked(now_ms)
        if len(self._records) >= self._max_episodes:
            raise ExperienceLedgerError("experience ledger exceeds its bounded size")

    @staticmethod
    def _admission_fingerprint(record: ExperienceRecord) -> tuple[Any, ...]:
        return (
            record.episode.episode_id,
            record.memory_class,
            record.authority_class,
            tuple(sorted((key, value) for key, value in record.evidence_authority.items())),
            record.invalidation_dependency_ids,
            record.scope_ids,
            record.ttl_ms,
        )

    def _conflict_or_replay(self, record: ExperienceRecord) -> ExperienceRecord | None:
        existing = self._records.get(record.episode_id)
        if existing is None:
            return None
        if existing.withdrawn or existing.memory_class is MemoryClass.WITHDRAWN:
            raise ExperienceLedgerError("cannot revive a withdrawn episode")
        if existing.record_id == record.record_id:
            return existing
        if self._admission_fingerprint(existing) == self._admission_fingerprint(record):
            return existing
        _cannot_upgrade(existing.authority_class, record.authority_class)
        raise ExperienceLedgerError("episode was replayed with a conflicting ledger admission")

    def record(
        self,
        episode: ExperienceEpisode | Mapping[str, Any],
        *,
        memory_class: MemoryClass | str = MemoryClass.TASK_EPISODE,
        evidence_authority: Mapping[str, Any] | None = None,
        claimed_authority: AuthorityClass | str | None = None,
        ttl_ms: int | None = None,
        scope_ids: Sequence[str] = (),
        recorded_at_ms: int | None = None,
        invalidation_dependency_ids: Sequence[str] = (),
    ) -> ExperienceRecord:
        """Admit one compact episode.  Replay of an identical row is idempotent."""

        sealed = _coerce_episode(episode)
        memory = _enum(memory_class, MemoryClass, "memory_class")
        if memory is MemoryClass.WITHDRAWN:
            raise ExperienceLedgerError("episodes cannot be admitted as withdrawn")
        extra_dependencies = _identifiers(
            invalidation_dependency_ids, "invalidation_dependency_ids"
        )
        dependencies = _dependency_ids(sealed, extra_dependencies)
        authority_map = _evidence_authority_map(evidence_authority, sealed)
        admitted_authority = _bound_authority(
            episode=sealed,
            evidence_authority=authority_map,
            claimed_authority=claimed_authority,
            memory_class=memory,
        )
        now_ms = self._now_ms()
        recorded = now_ms if recorded_at_ms is None else _int(recorded_at_ms, "recorded_at_ms")
        ttl = _DEFAULT_TTL_MS[memory] if ttl_ms is None else _int(ttl_ms, "ttl_ms")
        expires_at = recorded + ttl if ttl else 0
        record = ExperienceRecord(
            episode=sealed,
            memory_class=memory,
            authority_class=admitted_authority,
            evidence_authority=authority_map,
            invalidation_dependency_ids=dependencies,
            scope_ids=scope_ids,
            ttl_ms=ttl,
            expires_at_ms=expires_at,
            recorded_at_ms=recorded,
            withdrawn=False,
        )
        with self._lock:
            replayed = self._conflict_or_replay(record)
            if replayed is not None:
                return replayed
            blocked = set(record.invalidation_dependency_ids).intersection(self._invalidated)
            if blocked:
                raise ExperienceLedgerError("episode depends on invalidated evidence")
            self._admit_capacity(now_ms)
            self._persist_record(record)
            self._index_record(record)
            self._persist_head()
            return record

    def get(
        self,
        episode_id: str,
        *,
        include_withdrawn: bool = False,
        require_current: bool = False,
    ) -> ExperienceRecord | None:
        identifier = _identifier(episode_id, "episode_id")
        with self._lock:
            record = self._records.get(identifier)
            if record is None:
                return None
            now_ms = self._now_ms()
            current = record.is_current(now_ms)
            if require_current and not current:
                return None
            if record.withdrawn and not include_withdrawn:
                return None
            return record

    def projection(
        self,
        episode_id: str,
        *,
        include_withdrawn: bool = False,
    ) -> dict[str, Any] | None:
        record = self.get(episode_id, include_withdrawn=include_withdrawn)
        if record is None:
            return None
        return record.to_public_projection()

    def current(self) -> tuple[ExperienceRecord, ...]:
        now_ms = self._now_ms()
        with self._lock:
            return tuple(
                self._records[key]
                for key in sorted(self._records)
                if self._records[key].is_current(now_ms)
            )

    def episodes(self, *, include_withdrawn: bool = False) -> tuple[ExperienceEpisode, ...]:
        now_ms = self._now_ms()
        with self._lock:
            records = []
            for key in sorted(self._records):
                record = self._records[key]
                if record.withdrawn and not include_withdrawn:
                    continue
                if not include_withdrawn and not record.is_current(now_ms):
                    continue
                records.append(record.episode)
            return tuple(records)

    def by_dependency(
        self,
        dependency_id: str,
        *,
        include_withdrawn: bool = False,
    ) -> tuple[ExperienceRecord, ...]:
        identifier = _identifier(dependency_id, "dependency_id")
        now_ms = self._now_ms()
        with self._lock:
            holders = self._dependencies.get(identifier, set())
            records = []
            for episode_id in sorted(holders):
                record = self._records.get(episode_id)
                if record is None:
                    continue
                if record.withdrawn and not include_withdrawn:
                    continue
                if not include_withdrawn and not record.is_current(now_ms):
                    continue
                records.append(record)
            return tuple(records)

    def invalidate(
        self,
        dependency_id: str,
        *,
        recorded_at_ms: int | None = None,
    ) -> ExperienceInvalidationReceipt:
        """Withdraw only episodes that directly cite ``dependency_id``."""

        identifier = _identifier(dependency_id, "dependency_id")
        recorded = self._now_ms() if recorded_at_ms is None else _int(recorded_at_ms, "recorded_at_ms")
        with self._lock:
            self._invalidated.add(identifier)
            holders = tuple(sorted(self._dependencies.get(identifier, ())))
            invalidated: list[str] = []
            for episode_id in holders:
                record = self._records[episode_id]
                if record.withdrawn:
                    invalidated.append(episode_id)
                    continue
                withdrawn = replace(
                    record,
                    withdrawn=True,
                    memory_class=MemoryClass.WITHDRAWN,
                )
                self._persist_record(withdrawn)
                self._records[episode_id] = withdrawn
                invalidated.append(episode_id)
            retained = tuple(
                key for key in sorted(self._records) if key not in set(invalidated)
            )
            receipt = ExperienceInvalidationReceipt(
                dependency_id=identifier,
                invalidated_episode_ids=tuple(invalidated),
                retained_episode_ids=retained,
                recorded_at_ms=recorded,
            )
            self._persist(STORE_INDEX_PREFIX + "invalidation/" + identifier, receipt.to_dict())
            self._persist_head()
            return receipt

    def compact(self, *, now_ms: int | None = None) -> tuple[str, ...]:
        """Drop expired ephemeral/negative rows.  Accepted history is retained."""

        moment = self._now_ms() if now_ms is None else _int(now_ms, "now_ms")
        with self._lock:
            dropped: list[str] = []
            for record in tuple(self._records.values()):
                keep_counterexample = bool(record.episode.counterexample_ids)
                if record.memory_class in _COMPACTABLE_MEMORY and not record.is_current(moment):
                    if keep_counterexample:
                        continue
                    self._drop_from_index(record)
                    dropped.append(record.episode_id)
            self._persist_head()
            return tuple(sorted(dropped))

    def _snapshot_unlocked(self) -> ExperienceLedgerSnapshot:
        records = tuple(self._records[key] for key in sorted(self._records))
        return ExperienceLedgerSnapshot(
            records=records,
            invalidated_dependency_ids=tuple(sorted(self._invalidated)),
            epoch=self._epoch,
        )

    def snapshot(self) -> ExperienceLedgerSnapshot:
        with self._lock:
            return self._snapshot_unlocked()

    @classmethod
    def from_snapshot(
        cls,
        snapshot: ExperienceLedgerSnapshot | Mapping[str, Any],
        *,
        store: ExperienceStoreAdapter | None = None,
        clock: Callable[[], int] | None = None,
        max_episodes: int = MAX_LEDGER_EPISODES,
    ) -> ExperienceLedger:
        if isinstance(snapshot, ExperienceLedgerSnapshot):
            sealed = snapshot
        elif isinstance(snapshot, Mapping):
            sealed = ExperienceLedgerSnapshot.from_dict(snapshot)
        else:
            raise ExperienceLedgerError("snapshot must be an ExperienceLedgerSnapshot or mapping")
        ledger = cls(store=store, clock=clock, max_episodes=max_episodes, epoch=sealed.epoch)
        with ledger._lock:
            for record in sealed.records:
                if not isinstance(record, ExperienceRecord):
                    record = ExperienceRecord.from_dict(record)
                ledger._index_record(record)
            ledger._invalidated = set(sealed.invalidated_dependency_ids)
            rebuilt = ledger._snapshot_unlocked()
            if rebuilt.snapshot_id != sealed.snapshot_id:
                raise ExperienceLedgerError(
                    "restart snapshot is not a canonical projection of stored episodes"
                )
        return ledger


__all__ = [
    "ALLOWED_COMPACT_EPISODE_FIELDS",
    "ALLOWED_EPISODE_ENVELOPE_FIELDS",
    "EXPERIENCE_EPISODE_INTERFACE",
    "EXPERIENCE_INDEX_SCHEMA",
    "EXPERIENCE_INVALIDATION_SCHEMA",
    "EXPERIENCE_LEDGER_INTERFACE",
    "EXPERIENCE_LEDGER_SCHEMA",
    "EXPERIENCE_PROJECTION_SCHEMA",
    "EXPERIENCE_RECORD_SCHEMA",
    "EXPERIENCE_SNAPSHOT_SCHEMA",
    "FORBIDDEN_LEDGER_FIELD_MARKERS",
    "MAX_LEDGER_EPISODES",
    "ExperienceInvalidationReceipt",
    "ExperienceLedger",
    "ExperienceLedgerError",
    "ExperienceLedgerSnapshot",
    "ExperienceRecord",
    "ExperienceStoreAdapter",
    "InMemoryExperienceStore",
    "authority_rank",
    "compact_episode_payload",
    "field_is_forbidden",
    "min_authority",
    "public_episode_projection",
]
