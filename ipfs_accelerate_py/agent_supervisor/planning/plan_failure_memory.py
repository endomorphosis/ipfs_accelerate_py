"""Durable, typed branch-failure memory for bounded delta replanning.

The memory deliberately stores a small feature vector instead of provider
reasoning, prompts, logs, or exception text.  Every lookup is namespaced by the
exact repository tree, policy revision, execution environment, and planner
version.  This prevents a failure learned in one planning world from silently
poisoning another.

Two identities are kept separate:

* ``diagnostic_id`` identifies the typed failure and is stable when only its
  evidence revision changes; and
* ``event_id`` additionally binds the evidence revision and is therefore the
  semantic trigger for reopening a branch.

Delivery identifiers and timestamps are excluded from both identities.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Final, Iterable, Mapping

from ..proof.formal_verification_contracts import canonical_json, content_identity


PLAN_FAILURE_MEMORY_VERSION: Final[int] = 1
PLAN_FAILURE_MEMORY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/plan-failure-memory@1"
)
BRANCH_FAILURE_RECORD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/branch-failure-record@1"
)
DELTA_REPLAN_REQUIREMENT_ID: Final[str] = (
    "285414268422632231306428376746151397491"
)

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+@=-]{0,255}$")


class PlanFailureMemoryError(ValueError):
    """Raised when branch-failure state is malformed, stale, or unsafe."""


def _identifier(value: Any, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise PlanFailureMemoryError(f"{name} must be a string")
    result = value.strip()
    if not result and allow_empty:
        return ""
    if not result or not _IDENTIFIER.fullmatch(result):
        raise PlanFailureMemoryError(f"{name} must be a bounded typed identifier")
    return result


def _identifiers(
    values: Iterable[Any],
    name: str,
    *,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise PlanFailureMemoryError(f"{name} must be an array")
    result = tuple(
        sorted({_identifier(value, name) for value in values})
    )
    if not result and not allow_empty:
        raise PlanFailureMemoryError(f"{name} must not be empty")
    return result


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PlanFailureMemoryError(
            f"{name} must be an integer of at least {minimum}"
        )
    return value


class BranchFailureKind(str, Enum):
    """Closed failure taxonomy accepted by the learning boundary."""

    COUNTEREXAMPLE = "counterexample"
    FAILED_CONSTRAINT = "failed_constraint"
    CONSTRAINT_FAILURE = "failed_constraint"
    VALIDATION_SIGNATURE = "validation_signature"
    VALIDATION_FAILURE = "validation_signature"
    CAPABILITY_LOSS = "capability_loss"
    CONFLICT = "conflict"
    RESOURCE_INFEASIBILITY = "resource_infeasibility"
    RESOURCE_INFEASIBLE = "resource_infeasibility"


# Compatibility spelling useful at call sites which do not mention branches.
PlanFailureKind = BranchFailureKind


@dataclass(frozen=True)
class FailureMemoryScope:
    """Exact namespace within which historical features may influence search."""

    repository_tree_id: str
    policy_revision: str
    environment_id: str
    planner_version: str

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(
                self, name, _identifier(getattr(self, name), name)
            )

    @property
    def scope_id(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, str]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FailureMemoryScope":
        if not isinstance(payload, Mapping) or set(payload) != set(
            cls.__dataclass_fields__
        ):
            raise PlanFailureMemoryError("failure scope must use the closed schema")
        return cls(**dict(payload))


# A more explicit alias for consumers.
PlanFailureScope = FailureMemoryScope


@dataclass(frozen=True)
class TypedBranchFailure:
    """The complete learnable feature vector for one failed branch.

    There is intentionally no summary, message, traceback, prompt, transcript,
    or arbitrary metadata field.  ``failure_code`` and all bindings use a
    conservative identifier grammar and bounded length.
    """

    scope: FailureMemoryScope
    kind: BranchFailureKind
    failure_code: str
    branch_id: str
    step_ids: tuple[str, ...] = ()
    obligation_ids: tuple[str, ...] = ()
    alternative_ids: tuple[str, ...] = ()
    constraint_ids: tuple[str, ...] = ()
    validation_signature_ids: tuple[str, ...] = ()
    capability_ids: tuple[str, ...] = ()
    conflict_scope_ids: tuple[str, ...] = ()
    resource_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.scope, FailureMemoryScope):
            if not isinstance(self.scope, Mapping):
                raise PlanFailureMemoryError("scope must be FailureMemoryScope")
            object.__setattr__(
                self, "scope", FailureMemoryScope.from_dict(self.scope)
            )
        object.__setattr__(self, "kind", BranchFailureKind(self.kind))
        object.__setattr__(
            self, "failure_code", _identifier(self.failure_code, "failure_code")
        )
        object.__setattr__(
            self, "branch_id", _identifier(self.branch_id, "branch_id")
        )
        for name in (
            "step_ids",
            "obligation_ids",
            "alternative_ids",
            "constraint_ids",
            "validation_signature_ids",
            "capability_ids",
            "conflict_scope_ids",
            "resource_ids",
        ):
            object.__setattr__(
                self, name, _identifiers(getattr(self, name), name)
            )
        binding_fields = {
            BranchFailureKind.COUNTEREXAMPLE: (
                self.step_ids
                or self.obligation_ids
                or self.alternative_ids
                or self.conflict_scope_ids
            ),
            BranchFailureKind.FAILED_CONSTRAINT: self.constraint_ids,
            BranchFailureKind.VALIDATION_SIGNATURE: self.validation_signature_ids,
            BranchFailureKind.CAPABILITY_LOSS: self.capability_ids,
            BranchFailureKind.CONFLICT: self.conflict_scope_ids,
            BranchFailureKind.RESOURCE_INFEASIBILITY: self.resource_ids,
        }
        if not binding_fields[self.kind]:
            raise PlanFailureMemoryError(
                f"{self.kind.value} requires its typed failure binding"
            )

    @property
    def diagnostic_id(self) -> str:
        return content_identity(self.to_dict())

    @property
    def feature_id(self) -> str:
        return self.diagnostic_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope.to_dict(),
            "kind": self.kind.value,
            "failure_code": self.failure_code,
            "branch_id": self.branch_id,
            "step_ids": list(self.step_ids),
            "obligation_ids": list(self.obligation_ids),
            "alternative_ids": list(self.alternative_ids),
            "constraint_ids": list(self.constraint_ids),
            "validation_signature_ids": list(
                self.validation_signature_ids
            ),
            "capability_ids": list(self.capability_ids),
            "conflict_scope_ids": list(self.conflict_scope_ids),
            "resource_ids": list(self.resource_ids),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TypedBranchFailure":
        if not isinstance(payload, Mapping) or set(payload) != set(
            cls.__dataclass_fields__
        ):
            raise PlanFailureMemoryError(
                "typed branch failure must use the closed schema"
            )
        return cls(**dict(payload))


BranchFailureFeatures = TypedBranchFailure
FailureSignature = TypedBranchFailure


@dataclass(frozen=True)
class BranchFailureObservation:
    """One semantic failure observation.

    ``delivery_id`` can support transport auditing by the caller, but is
    deliberately omitted from :meth:`to_dict`, persistence, and semantic
    identity so redelivery noise cannot reopen a plan branch.
    """

    features: TypedBranchFailure
    evidence_id: str
    delivery_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.features, TypedBranchFailure):
            if not isinstance(self.features, Mapping):
                raise PlanFailureMemoryError(
                    "features must be TypedBranchFailure"
                )
            object.__setattr__(
                self, "features", TypedBranchFailure.from_dict(self.features)
            )
        object.__setattr__(
            self, "evidence_id", _identifier(self.evidence_id, "evidence_id")
        )
        object.__setattr__(
            self,
            "delivery_id",
            _identifier(self.delivery_id, "delivery_id", allow_empty=True),
        )

    @property
    def diagnostic_id(self) -> str:
        return self.features.diagnostic_id

    @property
    def event_id(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "features": self.features.to_dict(),
            "evidence_id": self.evidence_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BranchFailureObservation":
        if not isinstance(payload, Mapping) or set(payload) != {
            "features",
            "evidence_id",
        }:
            raise PlanFailureMemoryError(
                "branch failure observation must use the closed schema"
            )
        return cls(**dict(payload))


FailureObservation = BranchFailureObservation


@dataclass(frozen=True)
class FailureBackoffPolicy:
    """Finite retry and storage bounds for failure learning."""

    base_backoff_milliseconds: int = 1_000
    max_backoff_milliseconds: int = 300_000
    max_identical_failures: int = 8
    max_records: int = 4_096
    max_records_per_branch: int = 64

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(
                self,
                name,
                _integer(getattr(self, name), name, minimum=1),
            )
        if self.base_backoff_milliseconds > self.max_backoff_milliseconds:
            raise PlanFailureMemoryError(
                "base backoff cannot exceed maximum backoff"
            )
        if self.max_records_per_branch > self.max_records:
            raise PlanFailureMemoryError(
                "per-branch record bound cannot exceed total record bound"
            )

    def to_dict(self) -> dict[str, int]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FailureBackoffPolicy":
        if not isinstance(payload, Mapping) or set(payload) != set(
            cls.__dataclass_fields__
        ):
            raise PlanFailureMemoryError("backoff policy must use the closed schema")
        return cls(**dict(payload))


class FailureMemoryDisposition(str, Enum):
    NEW_FAILURE = "new_failure"
    CHANGED_EVIDENCE = "changed_evidence"
    UNCHANGED_BACKOFF = "unchanged_backoff"
    IDENTICAL_FAILURE_EXHAUSTED = "identical_failure_exhausted"
    MEMORY_BOUND_REACHED = "memory_bound_reached"


@dataclass(frozen=True)
class BranchFailureRecord:
    """Durable aggregate for one diagnostic in one exact scope."""

    features: TypedBranchFailure
    last_evidence_id: str
    occurrence_count: int
    identical_attempts: int
    first_observed_at_milliseconds: int
    last_observed_at_milliseconds: int

    def __post_init__(self) -> None:
        if not isinstance(self.features, TypedBranchFailure):
            if not isinstance(self.features, Mapping):
                raise PlanFailureMemoryError("record features are invalid")
            object.__setattr__(
                self, "features", TypedBranchFailure.from_dict(self.features)
            )
        object.__setattr__(
            self,
            "last_evidence_id",
            _identifier(self.last_evidence_id, "last_evidence_id"),
        )
        for name in (
            "occurrence_count",
            "first_observed_at_milliseconds",
            "last_observed_at_milliseconds",
        ):
            object.__setattr__(
                self, name, _integer(getattr(self, name), name, minimum=1)
            )
        if self.occurrence_count > 1_000_000:
            raise PlanFailureMemoryError(
                "occurrence_count exceeds the durable memory bound"
            )
        object.__setattr__(
            self,
            "identical_attempts",
            _integer(self.identical_attempts, "identical_attempts"),
        )
        if (
            self.last_observed_at_milliseconds
            < self.first_observed_at_milliseconds
        ):
            raise PlanFailureMemoryError(
                "failure record observation times are inconsistent"
            )

    @property
    def diagnostic_id(self) -> str:
        return self.features.diagnostic_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": BRANCH_FAILURE_RECORD_SCHEMA,
            "memory_version": PLAN_FAILURE_MEMORY_VERSION,
            "diagnostic_id": self.diagnostic_id,
            "features": self.features.to_dict(),
            "last_evidence_id": self.last_evidence_id,
            "occurrence_count": self.occurrence_count,
            "identical_attempts": self.identical_attempts,
            "first_observed_at_milliseconds": (
                self.first_observed_at_milliseconds
            ),
            "last_observed_at_milliseconds": self.last_observed_at_milliseconds,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BranchFailureRecord":
        expected = {
            "schema",
            "memory_version",
            "diagnostic_id",
            "features",
            "last_evidence_id",
            "occurrence_count",
            "identical_attempts",
            "first_observed_at_milliseconds",
            "last_observed_at_milliseconds",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise PlanFailureMemoryError(
                "branch failure record must use the closed schema"
            )
        if (
            payload.get("schema") != BRANCH_FAILURE_RECORD_SCHEMA
            or payload.get("memory_version") != PLAN_FAILURE_MEMORY_VERSION
        ):
            raise PlanFailureMemoryError(
                "branch failure record version is unsupported"
            )
        result = cls(
            features=payload.get("features") or {},
            last_evidence_id=payload.get("last_evidence_id", ""),
            occurrence_count=payload.get("occurrence_count", 0),
            identical_attempts=payload.get("identical_attempts", 0),
            first_observed_at_milliseconds=payload.get(
                "first_observed_at_milliseconds", 0
            ),
            last_observed_at_milliseconds=payload.get(
                "last_observed_at_milliseconds", 0
            ),
        )
        if payload.get("diagnostic_id") != result.diagnostic_id:
            raise PlanFailureMemoryError(
                "branch failure diagnostic identity does not match content"
            )
        return result


@dataclass(frozen=True)
class FailureMemoryDecision:
    disposition: FailureMemoryDisposition
    diagnostic_id: str
    event_id: str
    should_replan: bool
    diagnostic_reused: bool
    backoff_attempt: int
    backoff_milliseconds: int
    record: BranchFailureRecord | None

    @property
    def changed(self) -> bool:
        return self.disposition in {
            FailureMemoryDisposition.NEW_FAILURE,
            FailureMemoryDisposition.CHANGED_EVIDENCE,
        }

    @property
    def exhausted(self) -> bool:
        return (
            self.disposition
            is FailureMemoryDisposition.IDENTICAL_FAILURE_EXHAUSTED
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "disposition": self.disposition.value,
            "diagnostic_id": self.diagnostic_id,
            "event_id": self.event_id,
            "should_replan": self.should_replan,
            "diagnostic_reused": self.diagnostic_reused,
            "backoff_attempt": self.backoff_attempt,
            "backoff_milliseconds": self.backoff_milliseconds,
            "record": self.record.to_dict() if self.record else None,
        }


@dataclass(frozen=True)
class PlanFailureMemorySnapshot:
    policy: FailureBackoffPolicy
    records: tuple[BranchFailureRecord, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.policy, FailureBackoffPolicy):
            if not isinstance(self.policy, Mapping):
                raise PlanFailureMemoryError("snapshot policy is invalid")
            object.__setattr__(
                self, "policy", FailureBackoffPolicy.from_dict(self.policy)
            )
        records = tuple(
            item
            if isinstance(item, BranchFailureRecord)
            else BranchFailureRecord.from_dict(item)
            for item in self.records
        )
        ids = [item.diagnostic_id for item in records]
        if len(ids) != len(set(ids)):
            raise PlanFailureMemoryError(
                "failure memory contains duplicate diagnostics"
            )
        if len(records) > self.policy.max_records:
            raise PlanFailureMemoryError("failure memory exceeds its record bound")
        branch_counts: dict[tuple[str, str], int] = {}
        for record in records:
            key = (record.features.scope.scope_id, record.features.branch_id)
            branch_counts[key] = branch_counts.get(key, 0) + 1
        if any(
            count > self.policy.max_records_per_branch
            for count in branch_counts.values()
        ):
            raise PlanFailureMemoryError(
                "failure memory exceeds its per-branch bound"
            )
        object.__setattr__(
            self,
            "records",
            tuple(sorted(records, key=lambda item: item.diagnostic_id)),
        )

    @property
    def state_id(self) -> str:
        return content_identity(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema": PLAN_FAILURE_MEMORY_SCHEMA,
            "memory_version": PLAN_FAILURE_MEMORY_VERSION,
            "policy": self.policy.to_dict(),
            "records": [item.to_dict() for item in self.records],
        }
        if include_identity:
            payload["state_id"] = self.state_id
        return payload

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "PlanFailureMemorySnapshot":
        expected = {
            "schema",
            "memory_version",
            "state_id",
            "policy",
            "records",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise PlanFailureMemoryError(
                "failure memory snapshot must use the closed schema"
            )
        if (
            payload.get("schema") != PLAN_FAILURE_MEMORY_SCHEMA
            or payload.get("memory_version") != PLAN_FAILURE_MEMORY_VERSION
        ):
            raise PlanFailureMemoryError(
                "failure memory snapshot version is unsupported"
            )
        result = cls(
            policy=FailureBackoffPolicy.from_dict(payload.get("policy") or {}),
            records=tuple(
                BranchFailureRecord.from_dict(item)
                for item in payload.get("records") or ()
            ),
        )
        if payload.get("state_id") != result.state_id:
            raise PlanFailureMemoryError(
                "failure memory state identity does not match content"
            )
        return result


class PlanFailureMemory:
    """Bounded in-memory index with optional fail-closed durable persistence."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        policy: FailureBackoffPolicy | None = None,
    ) -> None:
        self.path = self._resolve_path(path)
        self.policy = policy or FailureBackoffPolicy()
        self._records: dict[str, BranchFailureRecord] = {}
        if self.path is not None and self.path.exists():
            snapshot = self._load_path(self.path)
            if policy is not None and snapshot.policy != policy:
                raise PlanFailureMemoryError(
                    "persisted failure-memory policy does not match requested policy"
                )
            self.policy = snapshot.policy
            self._records = {
                item.diagnostic_id: item for item in snapshot.records
            }

    @staticmethod
    def _resolve_path(path: str | Path | None) -> Path | None:
        if path is None:
            return None
        candidate = Path(path)
        if candidate.exists() and candidate.is_dir():
            candidate = candidate / "plan_failure_memory.json"
        elif not candidate.suffix:
            candidate = candidate / "plan_failure_memory.json"
        if candidate.is_symlink():
            raise PlanFailureMemoryError(
                "failure-memory state cannot be a symlink"
            )
        return candidate

    @property
    def records(self) -> tuple[BranchFailureRecord, ...]:
        return self.snapshot().records

    def snapshot(self) -> PlanFailureMemorySnapshot:
        return PlanFailureMemorySnapshot(
            policy=self.policy,
            records=tuple(self._records.values()),
        )

    def lookup(
        self,
        diagnostic_or_features: str | TypedBranchFailure,
        *,
        scope: FailureMemoryScope | None = None,
    ) -> BranchFailureRecord | None:
        if isinstance(diagnostic_or_features, TypedBranchFailure):
            if (
                scope is not None
                and diagnostic_or_features.scope != scope
            ):
                return None
            identity = diagnostic_or_features.diagnostic_id
        else:
            identity = _identifier(
                diagnostic_or_features, "diagnostic_id"
            )
        record = self._records.get(identity)
        if record is not None and scope is not None and record.features.scope != scope:
            return None
        return record

    def observe(
        self,
        observation: BranchFailureObservation | Mapping[str, Any],
        *,
        observed_at_milliseconds: int,
    ) -> FailureMemoryDecision:
        value = (
            observation
            if isinstance(observation, BranchFailureObservation)
            else BranchFailureObservation.from_dict(observation)
        )
        now = _integer(
            observed_at_milliseconds,
            "observed_at_milliseconds",
            minimum=1,
        )
        existing = self._records.get(value.diagnostic_id)
        if existing is None:
            same_branch = sum(
                item.features.scope == value.features.scope
                and item.features.branch_id == value.features.branch_id
                for item in self._records.values()
            )
            if (
                len(self._records) >= self.policy.max_records
                or same_branch >= self.policy.max_records_per_branch
            ):
                return FailureMemoryDecision(
                    disposition=FailureMemoryDisposition.MEMORY_BOUND_REACHED,
                    diagnostic_id=value.diagnostic_id,
                    event_id=value.event_id,
                    should_replan=False,
                    diagnostic_reused=False,
                    backoff_attempt=0,
                    backoff_milliseconds=0,
                    record=None,
                )
            record = BranchFailureRecord(
                features=value.features,
                last_evidence_id=value.evidence_id,
                occurrence_count=1,
                identical_attempts=0,
                first_observed_at_milliseconds=now,
                last_observed_at_milliseconds=now,
            )
            disposition = FailureMemoryDisposition.NEW_FAILURE
            should_replan = True
            backoff_attempt = 0
            backoff = 0
            reused = False
        elif existing.last_evidence_id != value.evidence_id:
            record = replace(
                existing,
                last_evidence_id=value.evidence_id,
                occurrence_count=min(
                    1_000_000, existing.occurrence_count + 1
                ),
                identical_attempts=0,
                last_observed_at_milliseconds=max(
                    now, existing.last_observed_at_milliseconds
                ),
            )
            disposition = FailureMemoryDisposition.CHANGED_EVIDENCE
            should_replan = True
            backoff_attempt = 0
            backoff = 0
            reused = True
        else:
            attempts = min(
                existing.identical_attempts + 1,
                self.policy.max_identical_failures,
            )
            record = replace(
                existing,
                occurrence_count=min(
                    1_000_000, existing.occurrence_count + 1
                ),
                identical_attempts=attempts,
                last_observed_at_milliseconds=max(
                    now, existing.last_observed_at_milliseconds
                ),
            )
            reused = True
            backoff_attempt = attempts
            if attempts >= self.policy.max_identical_failures:
                disposition = (
                    FailureMemoryDisposition.IDENTICAL_FAILURE_EXHAUSTED
                )
                should_replan = False
                backoff = 0
            else:
                disposition = FailureMemoryDisposition.UNCHANGED_BACKOFF
                should_replan = False
                backoff = min(
                    self.policy.max_backoff_milliseconds,
                    self.policy.base_backoff_milliseconds
                    * (2 ** min(attempts - 1, 30)),
                )
        self._records[value.diagnostic_id] = record
        self.persist()
        return FailureMemoryDecision(
            disposition=disposition,
            diagnostic_id=value.diagnostic_id,
            event_id=value.event_id,
            should_replan=should_replan,
            diagnostic_reused=reused,
            backoff_attempt=backoff_attempt,
            backoff_milliseconds=backoff,
            record=record,
        )

    record_failure = observe
    record = observe

    def historical_failure_millionths(
        self,
        *,
        scope: FailureMemoryScope,
        branch_id: str,
        feature_ids: Iterable[str] = (),
    ) -> int:
        """Return a bounded typed failure prior for one exact branch scope."""

        resolved_branch = _identifier(branch_id, "branch_id")
        selected_ids = _identifiers(feature_ids, "feature_ids")
        records = [
            item
            for item in self._records.values()
            if item.features.scope == scope
            and item.features.branch_id == resolved_branch
            and (
                not selected_ids
                or item.diagnostic_id in selected_ids
            )
        ]
        # Unique typed diagnostics matter more than redelivery count.  Each
        # diagnostic contributes at most 125k and the prior can never hard
        # prune a branch or exceed one million.
        return min(1_000_000, len(records) * 125_000)

    historical_failure_score = historical_failure_millionths

    def persist(self) -> Path | None:
        if self.path is None:
            return None
        path = self.path
        if path.exists() and path.is_symlink():
            raise PlanFailureMemoryError(
                "failure-memory state cannot be a symlink"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = (canonical_json(self.snapshot().to_dict()) + "\n").encode(
            "utf-8"
        )
        temporary_name = ""
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=path.parent,
                delete=False,
            ) as handle:
                temporary_name = handle.name
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        finally:
            if temporary_name:
                try:
                    Path(temporary_name).unlink(missing_ok=True)
                except OSError:
                    pass
        return path

    @classmethod
    def load(cls, path: str | Path) -> "PlanFailureMemory":
        return cls(path)

    @staticmethod
    def _load_path(path: Path) -> PlanFailureMemorySnapshot:
        if path.is_symlink() or not path.is_file():
            raise PlanFailureMemoryError(
                "failure-memory state is unavailable or unsafe"
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise PlanFailureMemoryError(
                "failure-memory state is unavailable or malformed"
            ) from exc
        return PlanFailureMemorySnapshot.from_dict(payload)


BranchFailureMemory = PlanFailureMemory


__all__ = [
    "BRANCH_FAILURE_RECORD_SCHEMA",
    "DELTA_REPLAN_REQUIREMENT_ID",
    "PLAN_FAILURE_MEMORY_SCHEMA",
    "PLAN_FAILURE_MEMORY_VERSION",
    "BranchFailureFeatures",
    "BranchFailureKind",
    "BranchFailureMemory",
    "BranchFailureObservation",
    "BranchFailureRecord",
    "FailureBackoffPolicy",
    "FailureMemoryDecision",
    "FailureMemoryDisposition",
    "FailureMemoryScope",
    "FailureObservation",
    "FailureSignature",
    "PlanFailureKind",
    "PlanFailureMemory",
    "PlanFailureMemoryError",
    "PlanFailureMemorySnapshot",
    "PlanFailureScope",
    "TypedBranchFailure",
]
