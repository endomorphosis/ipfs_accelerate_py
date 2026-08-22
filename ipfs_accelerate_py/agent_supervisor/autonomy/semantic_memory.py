"""Bounded semantic memory over existing content-addressed artifacts.

``SemanticMemory@1`` is an index, not a second store.  Entries reference
already sealed identities.  Retention classes, TTL, dependency invalidation,
and compaction are closed.  Frequency may change retrieval rank but cannot
upgrade evidence authority.  Raw prompts, source bodies, and private
reasoning are refused.

Cold import performs no filesystem, network, or provider action.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from threading import RLock
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..proof.formal_verification_contracts import (
    CONTRACT_VERSION,
    canonical_json_bytes,
    content_identity,
)
from .contracts import (
    MAX_IDENTIFIER_BYTES,
    MAX_MAPPING_ITEMS,
    MAX_SEQUENCE_ITEMS,
    AuthorityClass,
    AutonomyContractError,
    MemoryClass,
)

SEMANTIC_MEMORY_INTERFACE: Final[str] = "SemanticMemory@1"
MEMORY_ENTRY_INTERFACE: Final[str] = "MemoryEntry@1"
SEMANTIC_MEMORY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/semantic-memory@1"
)
MEMORY_ENTRY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/memory-entry@1"
)
MEMORY_INVALIDATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/memory-invalidation@1"
)
MAX_MEMORY_ENTRIES: Final[int] = 4_096
MAX_INTEGER: Final[int] = (1 << 63) - 1
COMPACTION_KEEP: Final[frozenset[MemoryClass]] = frozenset(
    {
        MemoryClass.REPOSITORY_PATTERN,
        MemoryClass.CROSS_REPOSITORY_RULE,
        MemoryClass.AUTHORITATIVE_CURRENT,
        MemoryClass.TASK_EPISODE,
    }
)
FORBIDDEN_MEMORY_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "chain_of_thought",
        "decoded_source",
        "hidden_reasoning",
        "private_reasoning",
        "prompt",
        "raw_prompt",
        "source_body",
        "transcript",
    }
)


class SemanticMemoryError(AutonomyContractError):
    """Raised when a memory entry is malformed, private, or over bound."""


def _identifier(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not text or len(text.encode("utf-8")) > MAX_IDENTIFIER_BYTES:
        raise SemanticMemoryError(f"{name} must be a bounded identifier")
    return text


def _strings(value: Any, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        items = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        items = tuple(value)
    else:
        raise SemanticMemoryError(f"{name} must be a string sequence")
    if len(items) > MAX_SEQUENCE_ITEMS:
        raise SemanticMemoryError(f"{name} exceeds the sequence bound")
    return tuple(_identifier(item, name) for item in items)


def _int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SemanticMemoryError(f"{name} must be an integer")
    if value < minimum or value > MAX_INTEGER:
        raise SemanticMemoryError(f"{name} is out of bounds")
    return value


def _reject_forbidden(payload: Mapping[str, Any], noun: str) -> None:
    if len(payload) > MAX_MAPPING_ITEMS:
        raise SemanticMemoryError(f"{noun} contains too many fields")
    encoded = str(payload).lower()
    for marker in FORBIDDEN_MEMORY_MARKERS:
        if marker in encoded:
            raise SemanticMemoryError(
                f"{noun} contains raw prompt, source, or private reasoning"
            )


@dataclass(frozen=True)
class MemoryEntry:
    """One indexed, content-addressed semantic memory row."""

    INTERFACE: ClassVar[str] = MEMORY_ENTRY_INTERFACE
    SCHEMA: ClassVar[str] = MEMORY_ENTRY_SCHEMA

    artifact_id: str
    memory_class: MemoryClass
    evidence_class: AuthorityClass
    created_at_ms: int
    ttl_ms: int
    dependency_ids: tuple[str, ...] = ()
    scope_ids: tuple[str, ...] = ()
    frequency: int = 0
    retained_kind: str = "artifact"

    def __post_init__(self) -> None:
        payload = {
            "artifact_id": self.artifact_id,
            "memory_class": getattr(self.memory_class, "value", self.memory_class),
            "evidence_class": getattr(
                self.evidence_class, "value", self.evidence_class
            ),
            "created_at_ms": self.created_at_ms,
            "ttl_ms": self.ttl_ms,
            "dependency_ids": self.dependency_ids,
            "scope_ids": self.scope_ids,
            "frequency": self.frequency,
            "retained_kind": self.retained_kind,
        }
        _reject_forbidden(payload, "memory entry")
        object.__setattr__(self, "artifact_id", _identifier(self.artifact_id, "artifact_id"))
        memory_class = self.memory_class
        if not isinstance(memory_class, MemoryClass):
            memory_class = MemoryClass(str(memory_class))
        object.__setattr__(self, "memory_class", memory_class)
        evidence = self.evidence_class
        if not isinstance(evidence, AuthorityClass):
            evidence = AuthorityClass(str(evidence))
        object.__setattr__(self, "evidence_class", evidence)
        object.__setattr__(
            self, "created_at_ms", _int(self.created_at_ms, "created_at_ms")
        )
        object.__setattr__(self, "ttl_ms", _int(self.ttl_ms, "ttl_ms"))
        object.__setattr__(
            self, "dependency_ids", _strings(self.dependency_ids, "dependency_ids")
        )
        object.__setattr__(self, "scope_ids", _strings(self.scope_ids, "scope_ids"))
        object.__setattr__(self, "frequency", _int(self.frequency, "frequency"))
        kind = _identifier(self.retained_kind, "retained_kind")
        if kind not in {
            "artifact",
            "contract",
            "signature",
            "pattern",
            "rule",
            "capability",
            "dependency",
            "outcome",
            "answer",
            "counterexample",
        }:
            raise SemanticMemoryError("retained_kind is not a closed retention class")
        object.__setattr__(self, "retained_kind", kind)

    @property
    def entry_id(self) -> str:
        return content_identity(self._identity_payload())

    def expires_at_ms(self) -> int:
        return self.created_at_ms + self.ttl_ms

    def expired(self, now_ms: int) -> bool:
        return _int(now_ms, "now_ms") >= self.expires_at_ms()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "contract_version": CONTRACT_VERSION,
            "artifact_id": self.artifact_id,
            "memory_class": self.memory_class.value,
            "evidence_class": self.evidence_class.value,
            "created_at_ms": self.created_at_ms,
            "ttl_ms": self.ttl_ms,
            "dependency_ids": list(self.dependency_ids),
            "scope_ids": list(self.scope_ids),
            "frequency": self.frequency,
            "retained_kind": self.retained_kind,
        }

    def _identity_payload(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload.pop("frequency", None)
        return payload


@dataclass(frozen=True)
class MemoryInvalidationReceipt:
    SCHEMA: ClassVar[str] = MEMORY_INVALIDATION_SCHEMA
    artifact_id: str
    withdrawn_entry_ids: tuple[str, ...]
    remaining_entry_ids: tuple[str, ...]

    def to_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "schema": self.SCHEMA,
                "artifact_id": self.artifact_id,
                "withdrawn_entry_ids": list(self.withdrawn_entry_ids),
                "remaining_entry_ids": list(self.remaining_entry_ids),
            }
        )


@dataclass
class SemanticMemory:
    """Bounded in-process index over sealed artifact identities."""

    INTERFACE: ClassVar[str] = SEMANTIC_MEMORY_INTERFACE
    SCHEMA: ClassVar[str] = SEMANTIC_MEMORY_SCHEMA

    _lock: RLock = field(default_factory=RLock, init=False, repr=False)
    _entries: dict[str, MemoryEntry] = field(default_factory=dict, init=False)
    _withdrawn: set[str] = field(default_factory=set, init=False)

    def admit(self, entry: MemoryEntry) -> MemoryEntry:
        if not isinstance(entry, MemoryEntry):
            raise SemanticMemoryError("entry must be a MemoryEntry")
        if entry.memory_class is MemoryClass.WITHDRAWN:
            raise SemanticMemoryError("withdrawn entries cannot be admitted as current")
        with self._lock:
            if len(self._entries) >= MAX_MEMORY_ENTRIES:
                self._compact_locked(now_ms=entry.created_at_ms)
            if len(self._entries) >= MAX_MEMORY_ENTRIES:
                raise SemanticMemoryError("semantic memory is at the durable bound")
            if entry.entry_id in self._withdrawn:
                raise SemanticMemoryError("withdrawn memory cannot be revived")
            current = self._entries.get(entry.entry_id)
            if current is not None:
                return current
            self._entries[entry.entry_id] = entry
            return entry

    def observe(self, entry_id: str) -> MemoryEntry:
        """Increment frequency. Rank changes; authority does not."""

        identifier = _identifier(entry_id, "entry_id")
        with self._lock:
            current = self._entries.get(identifier)
            if current is None:
                raise SemanticMemoryError("memory entry is not current")
            previous_authority = current.evidence_class
            updated = MemoryEntry(
                artifact_id=current.artifact_id,
                memory_class=current.memory_class,
                evidence_class=current.evidence_class,
                created_at_ms=current.created_at_ms,
                ttl_ms=current.ttl_ms,
                dependency_ids=current.dependency_ids,
                scope_ids=current.scope_ids,
                frequency=current.frequency + 1,
                retained_kind=current.retained_kind,
            )
            if updated.evidence_class is not previous_authority:
                raise SemanticMemoryError("frequency cannot change evidence authority")
            if updated.entry_id != identifier:
                raise SemanticMemoryError("frequency cannot change memory identity")
            self._entries[identifier] = updated
            return updated

    def invalidate(self, artifact_id: str) -> MemoryInvalidationReceipt:
        target = _identifier(artifact_id, "artifact_id")
        with self._lock:
            withdrawn: list[str] = []
            remaining: list[str] = []
            for entry_id, entry in tuple(self._entries.items()):
                if entry.artifact_id == target or target in entry.dependency_ids:
                    self._withdrawn.add(entry_id)
                    del self._entries[entry_id]
                    withdrawn.append(entry_id)
                else:
                    remaining.append(entry_id)
            return MemoryInvalidationReceipt(
                artifact_id=target,
                withdrawn_entry_ids=tuple(withdrawn),
                remaining_entry_ids=tuple(remaining),
            )

    def compact(self, *, now_ms: int) -> tuple[MemoryEntry, ...]:
        with self._lock:
            return self._compact_locked(now_ms=now_ms)

    def retrieve(
        self,
        *,
        now_ms: int,
        scope_ids: Sequence[str] = (),
    ) -> tuple[MemoryEntry, ...]:
        allowed = set(_strings(scope_ids, "scope_ids")) if scope_ids else None
        with self._lock:
            current = []
            for entry in self._entries.values():
                if entry.expired(now_ms) and entry.memory_class in {
                    MemoryClass.EPHEMERAL_ATTEMPT,
                    MemoryClass.SHORT_LIVED_NEGATIVE,
                }:
                    continue
                if allowed is not None and allowed.isdisjoint(entry.scope_ids):
                    continue
                current.append(entry)
            return tuple(
                sorted(
                    current,
                    key=lambda item: (-item.frequency, item.created_at_ms, item.entry_id),
                )
            )

    def entries(self) -> tuple[MemoryEntry, ...]:
        with self._lock:
            return tuple(self._entries.values())

    def _compact_locked(self, *, now_ms: int) -> tuple[MemoryEntry, ...]:
        kept: dict[str, MemoryEntry] = {}
        for entry_id, entry in self._entries.items():
            ephemeral = entry.memory_class in {
                MemoryClass.EPHEMERAL_ATTEMPT,
                MemoryClass.SHORT_LIVED_NEGATIVE,
            }
            if ephemeral and entry.expired(now_ms):
                self._withdrawn.add(entry_id)
                continue
            if (
                not ephemeral
                and entry.memory_class not in COMPACTION_KEEP
                and entry.memory_class is not MemoryClass.AUTHORITATIVE_CURRENT
            ):
                if entry.expired(now_ms):
                    self._withdrawn.add(entry_id)
                    continue
            kept[entry_id] = entry
        self._entries = kept
        return tuple(kept.values())
