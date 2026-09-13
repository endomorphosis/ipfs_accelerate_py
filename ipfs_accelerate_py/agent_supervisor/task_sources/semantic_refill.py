"""Bounded semantic refill proposals with task semantic deduplication.

This module is the existing refill-path leaf named by DOEP-PLAN-V5.  Task
semantic deduplication (DOEP-054) is a binding of ``SemanticRefill``, not a second identity owner,
planner, queue, or event bus.  Canonical fingerprints come from
``canonical_task_identity``; display IDs, board namespaces, and cosmetic
rewrites cannot escape that identity.  Proposals never write DuckDB, never
mutate accepted history, and never authorize completion.  A worker or model
assertion cannot skip deduplication or treat an empty queue as done.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final, Iterable, Mapping, Sequence

from .task_identity import TASK_IDENTITY_SCHEMA, TaskIdentity, canonical_task_identity


SEMANTIC_REFILL_BINDING: Final[str] = "SemanticRefill@1"
SEMANTIC_REFILL_INTERFACE: Final[str] = SEMANTIC_REFILL_BINDING
SEMANTIC_REFILL_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-refill@1"
)
TASK_SEMANTIC_DEDUPLICATION_BINDING: Final[str] = "TaskSemanticDeduplication@1"
TASK_SEMANTIC_DEDUPLICATION_INTERFACE: Final[str] = (
    TASK_SEMANTIC_DEDUPLICATION_BINDING
)
TASK_SEMANTIC_DEDUPLICATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/task-semantic-deduplication@1"
)
AUTOMATIC_BOUNDED_TASK_REFILL_BINDING: Final[str] = "AutomaticBoundedTaskRefill@1"
TASK_SEMANTIC_DEDUPLICATION_CONSUMES: Final[tuple[str, ...]] = (
    TASK_IDENTITY_SCHEMA,
    AUTOMATIC_BOUNDED_TASK_REFILL_BINDING,
)
DEFAULT_MAX_BOUND: Final[int] = 4
IMMUTABLE_LIFECYCLES: Final[frozenset[str]] = frozenset(
    {
        "claimed",
        "running",
        "settling",
        "completed",
        "accepted",
    }
)
_IDLE_TRIGGERS: Final[frozenset[str]] = frozenset(
    {"", "idle", "idle_capacity", "idle-capacity"}
)


class RefillError(ValueError):
    """Raised when a refill proposal would violate bound or history fences."""


class SemanticDeduplicationError(RefillError):
    """Raised when refill candidates lack canonical semantic work metadata."""


class SemanticDeduplicationDisposition(str, Enum):
    ADMITTED = "admitted"
    DUPLICATE_EXISTING = "duplicate_existing"
    DUPLICATE_WAVE = "duplicate_wave"
    IMMUTABLE_HISTORY = "immutable_history"
    IDLE_CAPACITY = "idle_capacity_rejected"


def _compact(value: Any) -> str:
    return str(value or "").strip()


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    raise SemanticDeduplicationError("refill records must be mappings")


def _mapping_value(source: Mapping[str, Any], *keys: str) -> Any:
    normalized = {
        str(key).strip().casefold().replace("_", " "): value
        for key, value in source.items()
    }
    for key in keys:
        candidate = normalized.get(key.casefold().replace("_", " "))
        if candidate not in (None, "", [], ()):
            return candidate
    return ""


def _sequence(value: Any) -> tuple[Any, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, Mapping):
        return (value,)
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(item for item in value if item not in (None, ""))
    return (value,)


def _task_id(record: Mapping[str, Any]) -> str:
    return _compact(
        _mapping_value(record, "task id", "display task id", "id", "candidate id")
    )


def _lifecycle(record: Mapping[str, Any]) -> str:
    return _compact(
        _mapping_value(record, "lifecycle", "status", "state")
    ).casefold()


def _positive_bound(value: Any, default: int, name: str) -> int:
    if value in (None, ""):
        parsed = default
    else:
        parsed = int(value)
    if isinstance(value, bool) or parsed < 1:
        raise RefillError(f"{name} must be a positive integer")
    return parsed


def task_semantic_identity(task: Any, *, board_namespace: str = "") -> TaskIdentity:
    """Return the existing canonical identity; display IDs are provenance only."""

    try:
        return canonical_task_identity(task, board_namespace=board_namespace)
    except ValueError as exc:
        raise SemanticDeduplicationError(str(exc)) from exc


@dataclass(frozen=True)
class SemanticDeduplicationDecision:
    """One candidate's model-free semantic-dedup outcome."""

    disposition: SemanticDeduplicationDisposition
    candidate_id: str
    semantic_fingerprint: str
    canonical_task_key: str
    canonical_task_cid: str
    matched_task_id: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "disposition": self.disposition.value,
            "candidate_id": self.candidate_id,
            "semantic_fingerprint": self.semantic_fingerprint,
            "canonical_task_key": self.canonical_task_key,
            "canonical_task_cid": self.canonical_task_cid,
            "matched_task_id": self.matched_task_id,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class SemanticDeduplicationResult:
    """Receipt for one refill-wave semantic filter; never completion authority."""

    admitted: tuple[Mapping[str, Any], ...] = ()
    decisions: tuple[SemanticDeduplicationDecision, ...] = ()
    reason: str = ""
    schema: str = TASK_SEMANTIC_DEDUPLICATION_SCHEMA
    binding: str = TASK_SEMANTIC_DEDUPLICATION_BINDING
    carrier: str = "SemanticRefill"

    def __post_init__(self) -> None:
        object.__setattr__(self, "admitted", tuple(self.admitted))
        object.__setattr__(self, "decisions", tuple(self.decisions))
        object.__setattr__(
            self, "schema", self.schema or TASK_SEMANTIC_DEDUPLICATION_SCHEMA
        )
        if self.schema != TASK_SEMANTIC_DEDUPLICATION_SCHEMA:
            raise SemanticDeduplicationError(
                f"unsupported semantic deduplication schema: {self.schema}"
            )
        object.__setattr__(
            self, "binding", self.binding or TASK_SEMANTIC_DEDUPLICATION_BINDING
        )
        object.__setattr__(self, "carrier", self.carrier or "SemanticRefill")

    @property
    def admitted_task_ids(self) -> tuple[str, ...]:
        return tuple(
            decision.candidate_id
            for decision in self.decisions
            if decision.disposition is SemanticDeduplicationDisposition.ADMITTED
        )

    @property
    def suppressed(self) -> tuple[SemanticDeduplicationDecision, ...]:
        return tuple(
            decision
            for decision in self.decisions
            if decision.disposition is not SemanticDeduplicationDisposition.ADMITTED
        )

    @property
    def model_free(self) -> bool:
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "binding": self.binding,
            "interface": TASK_SEMANTIC_DEDUPLICATION_INTERFACE,
            "carrier": self.carrier,
            "consumes": list(TASK_SEMANTIC_DEDUPLICATION_CONSUMES),
            "admitted_task_ids": list(self.admitted_task_ids),
            "suppressed": [item.to_dict() for item in self.suppressed],
            "decisions": [item.to_dict() for item in self.decisions],
            "reason": self.reason,
            "model_free": True,
            "authorizes_append": False,
            "authorizes_completion": False,
            "database_write": False,
            "completion_authoritative": False,
            "worker_assertion_is_authority": False,
            "worker_completion_insufficient": True,
            "no_competing_subsystem_created": True,
            "empty_queue_is_completion": False,
            "identity_owner": TASK_IDENTITY_SCHEMA,
        }


def _index_existing(
    existing_tasks: Iterable[Any],
) -> dict[str, tuple[str, str]]:
    indexed: dict[str, tuple[str, str]] = {}
    for item in existing_tasks:
        record = _as_mapping(item)
        identity = task_semantic_identity(record)
        indexed[identity.semantic_fingerprint] = (
            _task_id(record) or identity.short_id,
            _lifecycle(record),
        )
    return indexed


def _idle_capacity_only(record: Mapping[str, Any]) -> bool:
    if not record.get("idle_capacity"):
        return False
    trigger = _compact(_mapping_value(record, "trigger")).casefold()
    evidence = _mapping_value(record, "evidence", "impacts", "plan delta")
    return trigger in _IDLE_TRIGGERS and evidence in (None, "", (), [])


def deduplicate_refill_tasks(
    candidates: Iterable[Any] = (),
    *,
    existing_tasks: Iterable[Any] = (),
    worker_assertion: bool = False,
    idle_capacity: bool = False,
    trigger: str = "",
    evidence: Any = None,
) -> SemanticDeduplicationResult:
    """Admit only novel semantic work for a refill wave.

    ``worker_assertion`` is recorded and ignored as authority.  First occurrence
    in candidate order wins; later cosmetic aliases and already-open or
    immutable history are suppressed.
    """

    del worker_assertion
    envelope = {
        "idle_capacity": idle_capacity,
        "trigger": trigger,
        "evidence": evidence,
    }
    if _idle_capacity_only(envelope):
        decisions: list[SemanticDeduplicationDecision] = []
        for item in _sequence(candidates):
            record = _as_mapping(item)
            identity = task_semantic_identity(record)
            decisions.append(
                SemanticDeduplicationDecision(
                    disposition=SemanticDeduplicationDisposition.IDLE_CAPACITY,
                    candidate_id=_task_id(record) or identity.short_id,
                    semantic_fingerprint=identity.semantic_fingerprint,
                    canonical_task_key=identity.canonical_task_key,
                    canonical_task_cid=identity.canonical_task_cid,
                    reason="idle_capacity_is_not_a_refill_trigger",
                )
            )
        return SemanticDeduplicationResult(
            decisions=tuple(decisions),
            reason="idle_capacity_is_not_a_refill_trigger",
        )

    known = _index_existing(existing_tasks)
    admitted: list[Mapping[str, Any]] = []
    decisions = []
    wave_ids: dict[str, str] = {}
    for item in _sequence(candidates):
        record = _as_mapping(item)
        identity = task_semantic_identity(record)
        candidate_id = _task_id(record) or identity.short_id
        fingerprint = identity.semantic_fingerprint
        existing_id, existing_lifecycle = known.get(fingerprint, ("", ""))
        if existing_lifecycle in IMMUTABLE_LIFECYCLES:
            decisions.append(
                SemanticDeduplicationDecision(
                    disposition=SemanticDeduplicationDisposition.IMMUTABLE_HISTORY,
                    candidate_id=candidate_id,
                    semantic_fingerprint=fingerprint,
                    canonical_task_key=identity.canonical_task_key,
                    canonical_task_cid=identity.canonical_task_cid,
                    matched_task_id=existing_id,
                    reason="immutable_claimed_running_settling_completed_accepted_history",
                )
            )
            continue
        if fingerprint in known:
            decisions.append(
                SemanticDeduplicationDecision(
                    disposition=SemanticDeduplicationDisposition.DUPLICATE_EXISTING,
                    candidate_id=candidate_id,
                    semantic_fingerprint=fingerprint,
                    canonical_task_key=identity.canonical_task_key,
                    canonical_task_cid=identity.canonical_task_cid,
                    matched_task_id=existing_id,
                    reason="canonical_semantic_identity_already_exists",
                )
            )
            continue
        prior_id = wave_ids.get(fingerprint)
        if prior_id is not None:
            decisions.append(
                SemanticDeduplicationDecision(
                    disposition=SemanticDeduplicationDisposition.DUPLICATE_WAVE,
                    candidate_id=candidate_id,
                    semantic_fingerprint=fingerprint,
                    canonical_task_key=identity.canonical_task_key,
                    canonical_task_cid=identity.canonical_task_cid,
                    matched_task_id=prior_id,
                    reason="duplicate_semantic_identity_in_refill_wave",
                )
            )
            continue
        wave_ids[fingerprint] = candidate_id
        admitted.append(record)
        decisions.append(
            SemanticDeduplicationDecision(
                disposition=SemanticDeduplicationDisposition.ADMITTED,
                candidate_id=candidate_id,
                semantic_fingerprint=fingerprint,
                canonical_task_key=identity.canonical_task_key,
                canonical_task_cid=identity.canonical_task_cid,
                reason="admitted",
            )
        )
    reason = "admitted"
    if not decisions:
        reason = "no_novel_semantic_work"
    elif not admitted:
        reason = decisions[0].reason
    return SemanticDeduplicationResult(
        admitted=tuple(admitted),
        decisions=tuple(decisions),
        reason=reason,
    )


def semantically_deduplicate_tasks(
    candidates: Iterable[Any] = (),
    *,
    existing_tasks: Iterable[Any] = (),
    worker_assertion: bool = False,
    idle_capacity: bool = False,
    trigger: str = "",
    evidence: Any = None,
) -> SemanticDeduplicationResult:
    """Binding entrypoint for TaskSemanticDeduplication@1."""

    return deduplicate_refill_tasks(
        candidates,
        existing_tasks=existing_tasks,
        worker_assertion=worker_assertion,
        idle_capacity=idle_capacity,
        trigger=trigger,
        evidence=evidence,
    )


def propose_refill(record: Mapping[str, Any] | None = None) -> MappingProxyType:
    """Return a proposal-only refill envelope with semantic deduplication applied."""

    payload = _as_mapping({} if record is None else record)
    if payload.get("rewrite_accepted"):
        raise RefillError("accepted history is immutable")
    bound = _positive_bound(payload.get("bound"), 1, "bound")
    max_bound = _positive_bound(payload.get("max_bound"), DEFAULT_MAX_BOUND, "max_bound")
    if bound > max_bound:
        raise RefillError("refill exceeds bound")
    candidates = _mapping_value(payload, "candidates", "refill tasks", "tasks")
    existing = _mapping_value(
        payload, "existing tasks", "existing", "open tasks", "history"
    )
    dedup = semantically_deduplicate_tasks(
        candidates or (),
        existing_tasks=existing or (),
        worker_assertion=bool(payload.get("worker_assertion")),
        idle_capacity=bool(payload.get("idle_capacity")),
        trigger=_compact(_mapping_value(payload, "trigger")),
        evidence=_mapping_value(payload, "evidence", "impacts", "plan delta"),
    )
    return MappingProxyType(
        {
            "proposal": True,
            "accepted": False,
            "bound": bound,
            "schema": SEMANTIC_REFILL_SCHEMA,
            "binding": SEMANTIC_REFILL_BINDING,
            "interface": SEMANTIC_REFILL_INTERFACE,
            "carrier": "SemanticRefill",
            "deduplication": dedup.to_dict(),
            "admitted_task_ids": list(dedup.admitted_task_ids),
            "reason": dedup.reason,
            "model_free": True,
            "authorizes_append": False,
            "authorizes_completion": False,
            "database_write": False,
            "completion_authoritative": False,
            "worker_assertion_is_authority": False,
            "worker_completion_insufficient": True,
            "no_competing_subsystem_created": True,
            "empty_queue_is_completion": False,
            "identity_owner": TASK_IDENTITY_SCHEMA,
        }
    )


class SemanticRefill:
    """Canonical refill-path carrier; not a second planner, queue, or event bus."""

    BINDING: ClassVar[str] = SEMANTIC_REFILL_BINDING
    INTERFACE: ClassVar[str] = SEMANTIC_REFILL_INTERFACE
    SCHEMA: ClassVar[str] = SEMANTIC_REFILL_SCHEMA
    DEDUPLICATION_BINDING: ClassVar[str] = TASK_SEMANTIC_DEDUPLICATION_BINDING
    DEDUPLICATION_INTERFACE: ClassVar[str] = TASK_SEMANTIC_DEDUPLICATION_INTERFACE
    DEDUPLICATION_SCHEMA: ClassVar[str] = TASK_SEMANTIC_DEDUPLICATION_SCHEMA

    def propose(self, record: Mapping[str, Any] | None = None) -> MappingProxyType:
        return propose_refill(record)

    def deduplicate(
        self,
        candidates: Iterable[Any] = (),
        *,
        existing_tasks: Iterable[Any] = (),
        worker_assertion: bool = False,
        idle_capacity: bool = False,
        trigger: str = "",
        evidence: Any = None,
    ) -> SemanticDeduplicationResult:
        return semantically_deduplicate_tasks(
            candidates,
            existing_tasks=existing_tasks,
            worker_assertion=worker_assertion,
            idle_capacity=idle_capacity,
            trigger=trigger,
            evidence=evidence,
        )
