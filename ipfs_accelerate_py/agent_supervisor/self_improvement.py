"""Benchmark-driven, bounded self-improvement epoch contracts.

An empty task board is a scheduling fact, not evidence that the repository has
no useful successor work.  This module defines the stronger no-gap path: a
content-addressed epoch over a complete benchmark population, explicit healthy
analyzer results, an independent exhaustion quorum, stable objective
ownership, and unchanged objective/taskboard artifacts.

The pure evaluator never invents work.  The runtime boundary can pass
actionable observations through the existing bounded proposal/admission
pipeline and atomically project admitted goals into the supervisor backlog.
Only a fully healthy epoch can emit
:data:`HEALTHY_EXHAUSTION_REQUIREMENT_ID` and enter the durable
wait-for-trigger state.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .backlog_refinery import (
    DEFAULT_TASK_ID_PREFIX,
    effective_open_task_count,
    load_strategy,
    record_self_improvement_exhaustion,
    record_objective_backlog_findings,
    self_improvement_epoch_wait_active,
)
from .formal_verification_contracts import content_identity as _strict_content_identity
from .goal_completion import CompletionEvidence
from .objective_graph import (
    ObjectiveGenerationLimits,
    ObjectiveGoalMaterializationPolicy,
    ObjectiveWorkKind,
    ObjectiveWorkProposal,
    objective_heap_content_id,
    preview_objective_goal_materialization,
)
from .objective_tracker import (
    ObjectiveEvidenceProjection,
    ObjectiveMaterializationTransactionState,
    commit_objective_goal_materialization,
    objective_materialization_tree_identity,
    resolve_objective_evidence_projection,
)
from .scan_receipts import (
    ExhaustionBinding,
    ExhaustionQuorumResult,
    RefillScanResult,
    RepositoryTreeIdentity,
    ScanTerminalReason,
    build_scan_result,
    evaluate_exhaustion_quorum,
    scan_identity,
)


SUCCESSOR_REFILL_REQUIREMENT_ID = (
    "020061024173618462922348580596364003627"
)
"""Opaque ASI-G109 requirement: bounded novel successors are created once."""

EPOCH_IDEMPOTENCY_REQUIREMENT_ID = (
    "065313778069923158401871898168782520190"
)
"""Opaque ASI-G110 requirement: an identical epoch is idempotent."""

HEALTHY_EXHAUSTION_REQUIREMENT_ID = (
    "119294002389522221490347364495731444366"
)
"""Opaque ASI-G111 requirement: a healthy epoch creates no busywork."""

SELF_IMPROVEMENT_EPOCH_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.self_improvement_epoch.v1"
)
HEALTHY_EXHAUSTION_EVIDENCE_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.healthy_exhaustion_evidence.v1"
)
SUCCESSOR_REFILL_EVIDENCE_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.successor_refill_evidence.v1"
)
EPOCH_REPLAY_EVIDENCE_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.self_improvement_epoch_replay.v1"
)
BENCHMARK_OBSERVATION_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.self_improvement_benchmark.v1"
)
SELF_IMPROVEMENT_LEDGER_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.self_improvement_ledger.v1"
)
SELF_IMPROVEMENT_ANALYZER_VERSION = "self-improvement-benchmark-analyzer/v1"
SELF_IMPROVEMENT_EVIDENCE_PRODUCER_VERSION = (
    "healthy-exhaustion-evidence-producer/v1"
)
DEFAULT_SELF_IMPROVEMENT_GOAL_ID = "ASI-G111"
DEFAULT_SUCCESSOR_REFILL_GOAL_ID = "ASI-G109"
DEFAULT_EPOCH_IDEMPOTENCY_GOAL_ID = "ASI-G110"
DEFAULT_SELF_IMPROVEMENT_PARENT_GOAL_ID = "ASI-G080"
DEFAULT_BENCHMARK_DIMENSIONS = (
    "cache",
    "control",
    "efficiency",
    "planning",
    "safety",
    "throughput",
    "validation",
)
DEFAULT_MEANINGFUL_TRIGGERS = (
    "capability_snapshot_changed",
    "operator_objective_revision",
    "policy_changed",
    "regression_observed",
    "repository_tree_changed",
    "scheduled_observation_window",
    "stale_evidence_observed",
)


def content_identity(value: Any) -> str:
    """Content-address JSON, retaining finite benchmark measurements.

    The formal-proof helper deliberately rejects every float.  Benchmark
    result payloads may legitimately contain finite ratios and durations, so
    use that stricter CID when possible and a canonical sha256 identity for
    otherwise valid JSON measurements.
    """

    try:
        return _strict_content_identity(value)
    except ValueError:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return "sha256:" + sha256(encoded).hexdigest()


def _utc_datetime(value: datetime | str | None, *, field_name: str) -> datetime:
    if value is None:
        result = datetime.now(timezone.utc)
    elif isinstance(value, datetime):
        result = value
    else:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            result = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return result.astimezone(timezone.utc)


def _required_text(value: Any, field_name: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise ValueError(f"{field_name} is required")
    return result


def _string_tuple(value: Any, *, field_name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        values: Iterable[Any] = (value,)
    elif isinstance(value, Iterable) and not isinstance(
        value, (bytes, bytearray, Mapping)
    ):
        values = value
    else:
        raise TypeError(f"{field_name} must be a sequence of strings")
    result = tuple(
        dict.fromkeys(
            str(item).strip() for item in values if str(item).strip()
        )
    )
    return tuple(sorted(result))


def _artifact_digest(value: str) -> str:
    digest = _required_text(value, "artifact_digest")
    if not digest.startswith("sha256:") or len(digest) != 71:
        raise ValueError("artifact_digest must be a sha256: digest")
    try:
        int(digest[7:], 16)
    except ValueError as exc:
        raise ValueError("artifact_digest must be hexadecimal") from exc
    return digest.lower()


def _strict_keys(
    payload: Mapping[str, Any],
    allowed: set[str],
    *,
    record_name: str,
) -> None:
    unknown = sorted(str(key) for key in payload if str(key) not in allowed)
    if unknown:
        raise ValueError(
            f"{record_name} contains unknown fields: {', '.join(unknown)}"
        )


def _fsync_parent(path: Path) -> None:
    try:
        descriptor = os.open(path.parent, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_parent(path)
    finally:
        if temporary.exists():
            temporary.unlink()


class BenchmarkDisposition(str, Enum):
    """Classification of one required benchmark dimension."""

    HEALTHY = "healthy"
    REGRESSION = "regression"
    UNCOVERED = "uncovered"
    STALE = "stale"
    BOTTLENECK = "bottleneck"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    PARTIAL = "partial"

    @property
    def actionable(self) -> bool:
        return self is not BenchmarkDisposition.HEALTHY


@dataclass(frozen=True)
class SelfImprovementPolicy:
    """Closed benchmark population and exhaustion policy for one epoch."""

    required_dimensions: tuple[str, ...] = DEFAULT_BENCHMARK_DIMENSIONS
    required_independent_channels: int = 2
    next_triggers: tuple[str, ...] = DEFAULT_MEANINGFUL_TRIGGERS
    policy_name: str = "benchmark-driven-bounded-self-refill/v1"
    max_new_successor_goals: int = 3
    max_open_successor_goals: int = 48
    max_successor_depth: int = 3
    max_successor_breadth: int = 4
    successor_token_budget: int = 8192
    minimum_successor_confidence: float = 0.5
    minimum_successor_novelty: float = 0.5

    def __post_init__(self) -> None:
        dimensions = _string_tuple(
            self.required_dimensions, field_name="required_dimensions"
        )
        triggers = _string_tuple(self.next_triggers, field_name="next_triggers")
        if not dimensions:
            raise ValueError("required_dimensions must not be empty")
        if not triggers:
            raise ValueError("next_triggers must not be empty")
        if (
            isinstance(self.required_independent_channels, bool)
            or not isinstance(self.required_independent_channels, int)
            or self.required_independent_channels < 2
        ):
            raise ValueError(
                "required_independent_channels must be an integer of at least two"
            )
        for name in (
            "max_new_successor_goals",
            "max_open_successor_goals",
            "max_successor_depth",
            "max_successor_breadth",
            "successor_token_budget",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in (
            "minimum_successor_confidence",
            "minimum_successor_novelty",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "required_dimensions", dimensions)
        object.__setattr__(self, "next_triggers", triggers)
        object.__setattr__(
            self, "policy_name", _required_text(self.policy_name, "policy_name")
        )

    @property
    def policy_id(self) -> str:
        return content_identity(
            {
                "schema": (
                    "ipfs_accelerate_py.agent_supervisor."
                    "self_improvement_policy.v1"
                ),
                "policy_name": self.policy_name,
                "required_dimensions": self.required_dimensions,
                "required_independent_channels": (
                    self.required_independent_channels
                ),
                "next_triggers": self.next_triggers,
                "max_new_successor_goals": self.max_new_successor_goals,
                "max_open_successor_goals": self.max_open_successor_goals,
                "max_successor_depth": self.max_successor_depth,
                "max_successor_breadth": self.max_successor_breadth,
                "successor_token_budget": self.successor_token_budget,
                "minimum_successor_confidence": self.minimum_successor_confidence,
                "minimum_successor_novelty": self.minimum_successor_novelty,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": (
                "ipfs_accelerate_py.agent_supervisor."
                "self_improvement_policy.v1"
            ),
            "policy_name": self.policy_name,
            "policy_id": self.policy_id,
            "required_dimensions": list(self.required_dimensions),
            "required_independent_channels": (
                self.required_independent_channels
            ),
            "next_triggers": list(self.next_triggers),
            "max_new_successor_goals": self.max_new_successor_goals,
            "max_open_successor_goals": self.max_open_successor_goals,
            "max_successor_depth": self.max_successor_depth,
            "max_successor_breadth": self.max_successor_breadth,
            "successor_token_budget": self.successor_token_budget,
            "minimum_successor_confidence": self.minimum_successor_confidence,
            "minimum_successor_novelty": self.minimum_successor_novelty,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SelfImprovementPolicy":
        allowed = {
            "schema",
            "policy_name",
            "policy_id",
            "required_dimensions",
            "required_independent_channels",
            "next_triggers",
            "max_new_successor_goals",
            "max_open_successor_goals",
            "max_successor_depth",
            "max_successor_breadth",
            "successor_token_budget",
            "minimum_successor_confidence",
            "minimum_successor_novelty",
        }
        _strict_keys(payload, allowed, record_name="self-improvement policy")
        if (
            payload.get("schema")
            != (
                "ipfs_accelerate_py.agent_supervisor."
                "self_improvement_policy.v1"
            )
        ):
            raise ValueError("unsupported self-improvement policy schema")
        result = cls(
            required_dimensions=tuple(
                payload.get("required_dimensions") or ()
            ),
            required_independent_channels=payload.get(
                "required_independent_channels"
            ),
            next_triggers=tuple(payload.get("next_triggers") or ()),
            policy_name=str(payload.get("policy_name") or ""),
            max_new_successor_goals=payload.get("max_new_successor_goals", 3),
            max_open_successor_goals=payload.get("max_open_successor_goals", 48),
            max_successor_depth=payload.get("max_successor_depth", 3),
            max_successor_breadth=payload.get("max_successor_breadth", 4),
            successor_token_budget=payload.get("successor_token_budget", 8192),
            minimum_successor_confidence=payload.get(
                "minimum_successor_confidence", 0.5
            ),
            minimum_successor_novelty=payload.get(
                "minimum_successor_novelty", 0.5
            ),
        )
        if payload.get("policy_id") != result.policy_id:
            raise ValueError("self-improvement policy identity does not match")
        return result


@dataclass(frozen=True)
class SelfImprovementEpochBinding:
    """Every input whose meaningful change permits another epoch."""

    repository_id: str
    repository_tree: str
    objective_revision: str
    taskboard_revision: str
    policy_id: str
    capability_snapshot_id: str
    observation_window: str
    operator_revision: str = "operator-objective/v1"

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(
                self, name, _required_text(getattr(self, name), name)
            )

    @property
    def epoch_id(self) -> str:
        return content_identity(
            {
                "schema": SELF_IMPROVEMENT_EPOCH_SCHEMA,
                **asdict(self),
            }
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "schema": SELF_IMPROVEMENT_EPOCH_SCHEMA,
            **asdict(self),
            "epoch_id": self.epoch_id,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "SelfImprovementEpochBinding":
        allowed = {
            "schema",
            "epoch_id",
            *cls.__dataclass_fields__.keys(),
        }
        _strict_keys(payload, allowed, record_name="epoch binding")
        if payload.get("schema") != SELF_IMPROVEMENT_EPOCH_SCHEMA:
            raise ValueError("unsupported self-improvement epoch schema")
        result = cls(
            **{
                name: str(payload.get(name) or "")
                for name in cls.__dataclass_fields__
            }
        )
        if payload.get("epoch_id") != result.epoch_id:
            raise ValueError("self-improvement epoch identity does not match")
        return result


@dataclass(frozen=True)
class BenchmarkObservation:
    """Fresh typed result for one member of the benchmark population."""

    dimension: str
    evidence_channel: str
    producer_id: str
    repository_id: str
    repository_tree: str
    policy_id: str
    capability_snapshot_id: str
    command: str
    toolchain: str
    scope: tuple[str, ...]
    result: Mapping[str, Any]
    artifact_digest: str
    disposition: BenchmarkDisposition | str = BenchmarkDisposition.HEALTHY
    actionable_reasons: tuple[str, ...] = ()
    observed_at: datetime | str | None = None
    fresh_until: datetime | str | None = None
    complete: bool = True
    receipt_id: str = ""

    def __post_init__(self) -> None:
        for name in (
            "dimension",
            "evidence_channel",
            "producer_id",
            "repository_id",
            "repository_tree",
            "policy_id",
            "capability_snapshot_id",
            "command",
            "toolchain",
        ):
            object.__setattr__(
                self, name, _required_text(getattr(self, name), name)
            )
        try:
            disposition = (
                self.disposition
                if isinstance(self.disposition, BenchmarkDisposition)
                else BenchmarkDisposition(str(self.disposition))
            )
        except ValueError as exc:
            raise ValueError(
                f"unknown benchmark disposition: {self.disposition!r}"
            ) from exc
        scope = _string_tuple(self.scope, field_name="scope")
        if not scope:
            raise ValueError("scope must not be empty")
        if not isinstance(self.result, Mapping) or not self.result:
            raise ValueError("result must be a non-empty mapping")
        reasons = _string_tuple(
            self.actionable_reasons, field_name="actionable_reasons"
        )
        if disposition.actionable and not reasons:
            raise ValueError(
                "an actionable benchmark disposition requires a reason"
            )
        if not disposition.actionable and reasons:
            raise ValueError(
                "a healthy benchmark observation cannot be actionable"
            )
        observed = _utc_datetime(self.observed_at, field_name="observed_at")
        fresh_until = _utc_datetime(
            self.fresh_until, field_name="fresh_until"
        )
        if fresh_until < observed:
            raise ValueError("fresh_until must not precede observed_at")
        if not isinstance(self.complete, bool):
            raise TypeError("complete must be a boolean")
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "result", dict(self.result))
        object.__setattr__(self, "artifact_digest", _artifact_digest(self.artifact_digest))
        object.__setattr__(self, "actionable_reasons", reasons)
        object.__setattr__(self, "observed_at", observed)
        object.__setattr__(self, "fresh_until", fresh_until)
        expected = content_identity(self._identity_payload())
        supplied = str(self.receipt_id or "").strip()
        if supplied and supplied != expected:
            raise ValueError("benchmark observation receipt identity does not match")
        object.__setattr__(self, "receipt_id", expected)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": BENCHMARK_OBSERVATION_SCHEMA,
            "version": 1,
            "dimension": self.dimension,
            "evidence_channel": self.evidence_channel,
            "producer_id": self.producer_id,
            "repository_id": self.repository_id,
            "repository_tree": self.repository_tree,
            "policy_id": self.policy_id,
            "capability_snapshot_id": self.capability_snapshot_id,
            "command": self.command,
            "toolchain": self.toolchain,
            "scope": self.scope,
            "result": dict(self.result),
            "artifact_digest": self.artifact_digest,
            "disposition": self.disposition.value,
            "actionable_reasons": self.actionable_reasons,
            "observed_at": self.observed_at.isoformat(),
            "fresh_until": self.fresh_until.isoformat(),
            "complete": self.complete,
        }

    def healthy_at(self, now: datetime) -> bool:
        return bool(
            self.disposition is BenchmarkDisposition.HEALTHY
            and self.complete
            and self.observed_at <= now <= self.fresh_until
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._identity_payload(),
            "receipt_id": self.receipt_id,
            "producer_kind": "benchmark",
            "source_tier": "benchmark",
            "status": (
                "passed"
                if self.disposition is BenchmarkDisposition.HEALTHY
                else self.disposition.value
            ),
            "validation_passed": (
                self.disposition is BenchmarkDisposition.HEALTHY
            ),
            "coverage_complete": self.complete,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BenchmarkObservation":
        allowed = {
            "schema",
            "version",
            "receipt_id",
            "producer_kind",
            "source_tier",
            "status",
            "validation_passed",
            "coverage_complete",
            *cls.__dataclass_fields__.keys(),
        }
        _strict_keys(payload, allowed, record_name="benchmark observation")
        if (
            payload.get("schema") != BENCHMARK_OBSERVATION_SCHEMA
            or int(payload.get("version", 0)) != 1
        ):
            raise ValueError("unsupported benchmark observation schema")
        result = cls(
            dimension=str(payload.get("dimension") or ""),
            evidence_channel=str(payload.get("evidence_channel") or ""),
            producer_id=str(payload.get("producer_id") or ""),
            repository_id=str(payload.get("repository_id") or ""),
            repository_tree=str(payload.get("repository_tree") or ""),
            policy_id=str(payload.get("policy_id") or ""),
            capability_snapshot_id=str(
                payload.get("capability_snapshot_id") or ""
            ),
            command=str(payload.get("command") or ""),
            toolchain=str(payload.get("toolchain") or ""),
            scope=tuple(payload.get("scope") or ()),
            result=payload.get("result") or {},
            artifact_digest=str(payload.get("artifact_digest") or ""),
            disposition=str(payload.get("disposition") or ""),
            actionable_reasons=tuple(payload.get("actionable_reasons") or ()),
            observed_at=str(payload.get("observed_at") or ""),
            fresh_until=str(payload.get("fresh_until") or ""),
            complete=payload.get("complete"),
            receipt_id=str(payload.get("receipt_id") or ""),
        )
        projected = result.to_dict()
        for name in (
            "producer_kind",
            "source_tier",
            "status",
            "validation_passed",
            "coverage_complete",
        ):
            if payload.get(name) != projected[name]:
                raise ValueError(
                    f"benchmark observation {name} projection does not match"
                )
        return result


class SelfImprovementEpochStatus(str, Enum):
    """Terminal state of one identity-bound epoch evaluation."""

    HEALTHY_EXHAUSTED = "healthy_exhausted"
    ACTIONABLE = "actionable"
    SUCCESSORS_CREATED = "successors_created"
    INELIGIBLE = "ineligible"


@dataclass(frozen=True)
class HealthyExhaustionEvidence:
    """Content-addressed proof that one healthy epoch created no busywork."""

    binding: SelfImprovementEpochBinding
    goal_projection: ObjectiveEvidenceProjection
    policy: SelfImprovementPolicy
    observations: tuple[BenchmarkObservation, ...]
    exhaustion_quorum: ExhaustionQuorumResult
    objective_before_id: str
    objective_after_id: str
    taskboard_before_id: str
    taskboard_after_id: str
    observed_at: datetime | str
    next_triggers: tuple[str, ...]
    classified_gap_count: int = 0
    candidate_count: int = 0
    admitted_count: int = 0
    materialized_count: int = 0
    taskboard_write_count: int = 0
    requirement_id: str = HEALTHY_EXHAUSTION_REQUIREMENT_ID
    producer_version: str = SELF_IMPROVEMENT_EVIDENCE_PRODUCER_VERSION
    evidence_id: str = ""

    def __post_init__(self) -> None:
        binding = (
            self.binding
            if isinstance(self.binding, SelfImprovementEpochBinding)
            else SelfImprovementEpochBinding.from_dict(self.binding)
        )
        projection = self.goal_projection
        if not isinstance(projection, ObjectiveEvidenceProjection):
            projection = ObjectiveEvidenceProjection.from_dict(projection)
        policy = (
            self.policy
            if isinstance(self.policy, SelfImprovementPolicy)
            else SelfImprovementPolicy.from_dict(self.policy)
        )
        observations = tuple(
            item
            if isinstance(item, BenchmarkObservation)
            else BenchmarkObservation.from_dict(item)
            for item in self.observations
        )
        quorum = (
            self.exhaustion_quorum
            if isinstance(self.exhaustion_quorum, ExhaustionQuorumResult)
            else ExhaustionQuorumResult.from_dict(self.exhaustion_quorum)
        )
        observed_at = _utc_datetime(self.observed_at, field_name="observed_at")
        triggers = _string_tuple(self.next_triggers, field_name="next_triggers")
        requirement = _required_text(self.requirement_id, "requirement_id")
        if requirement != HEALTHY_EXHAUSTION_REQUIREMENT_ID:
            raise ValueError("healthy exhaustion evidence claims the wrong requirement")
        if projection.requirement_id != requirement:
            raise ValueError("goal projection does not own the healthy exhaustion requirement")
        if projection.objective_heap_id != binding.objective_revision:
            raise ValueError("goal projection does not match the epoch objective revision")
        if policy.policy_id != binding.policy_id:
            raise ValueError("policy does not match the epoch binding")
        channel_dimensions: dict[str, list[str]] = {}
        for item in observations:
            channel_dimensions.setdefault(item.evidence_channel, []).append(
                item.dimension
            )
        if (
            len(channel_dimensions) < policy.required_independent_channels
            or any(
                tuple(sorted(dimensions)) != policy.required_dimensions
                for dimensions in channel_dimensions.values()
            )
        ):
            raise ValueError(
                "each independent benchmark channel must cover the complete "
                "dimension population exactly once"
            )
        if any(not item.healthy_at(observed_at) for item in observations):
            raise ValueError("every benchmark observation must be fresh and healthy")
        if any(
            (
                item.repository_id != binding.repository_id
                or item.repository_tree != binding.repository_tree
                or item.policy_id != binding.policy_id
                or item.capability_snapshot_id
                != binding.capability_snapshot_id
            )
            for item in observations
        ):
            raise ValueError("benchmark observation binding does not match the epoch")
        exact_binding = quorum.binding
        if (
            exact_binding.repository_id != binding.repository_id
            or exact_binding.tree_id != binding.repository_tree
            or exact_binding.analyzer_version
            != SELF_IMPROVEMENT_ANALYZER_VERSION
            or exact_binding.configuration_revision != binding.policy_id
            or exact_binding.objective_revision != binding.objective_revision
        ):
            raise ValueError("exhaustion quorum binding does not match the epoch")
        if not quorum.satisfied:
            raise ValueError("healthy exhaustion requires a satisfied quorum")
        if quorum.required_members != policy.required_independent_channels:
            raise ValueError("quorum policy does not match the epoch policy")
        if self.objective_before_id != self.objective_after_id:
            raise ValueError("healthy exhaustion cannot mutate the objective heap")
        if self.taskboard_before_id != self.taskboard_after_id:
            raise ValueError("healthy exhaustion cannot mutate the taskboard")
        for name in (
            "classified_gap_count",
            "candidate_count",
            "admitted_count",
            "materialized_count",
            "taskboard_write_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != 0:
                raise ValueError(f"{name} must be zero for healthy exhaustion")
            object.__setattr__(self, name, 0)
        if triggers != policy.next_triggers:
            raise ValueError("next triggers do not match the epoch policy")
        object.__setattr__(self, "binding", binding)
        object.__setattr__(self, "goal_projection", projection)
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "exhaustion_quorum", quorum)
        object.__setattr__(self, "observed_at", observed_at)
        object.__setattr__(self, "next_triggers", triggers)
        object.__setattr__(
            self,
            "producer_version",
            _required_text(self.producer_version, "producer_version"),
        )
        expected = content_identity(self._identity_payload())
        supplied = str(self.evidence_id or "").strip()
        if supplied and supplied != expected:
            raise ValueError("healthy exhaustion evidence identity does not match")
        object.__setattr__(self, "evidence_id", expected)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": HEALTHY_EXHAUSTION_EVIDENCE_SCHEMA,
            "version": 1,
            "requirement_id": self.requirement_id,
            "producer_version": self.producer_version,
            "binding": self.binding.to_dict(),
            "goal_projection": self.goal_projection.to_dict(),
            "policy": self.policy.to_dict(),
            "observations": [item.to_dict() for item in self.observations],
            "exhaustion_quorum": self.exhaustion_quorum.to_dict(),
            "objective_before_id": self.objective_before_id,
            "objective_after_id": self.objective_after_id,
            "taskboard_before_id": self.taskboard_before_id,
            "taskboard_after_id": self.taskboard_after_id,
            "observed_at": self.observed_at.isoformat(),
            "next_triggers": self.next_triggers,
            "classified_gap_count": self.classified_gap_count,
            "candidate_count": self.candidate_count,
            "admitted_count": self.admitted_count,
            "materialized_count": self.materialized_count,
            "taskboard_write_count": self.taskboard_write_count,
            "wait_state": "waiting_for_meaningful_trigger",
        }

    @property
    def proved_requirement_ids(self) -> tuple[str, ...]:
        return (self.requirement_id,)

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._identity_payload(),
            "evidence_id": self.evidence_id,
            "witness_id": self.evidence_id,
            "receipt_id": self.evidence_id,
            "provenance_cid": self.evidence_id,
            "artifact_digest": self.evidence_id,
            "requirement_ids": [self.requirement_id],
            "proved_requirement_ids": [self.requirement_id],
            "producer_kind": "benchmark",
            "source_tier": "benchmark",
            "repository_id": self.binding.repository_id,
            "repository_tree": self.binding.repository_tree,
            "tree_id": self.binding.repository_tree,
            "policy_id": self.binding.policy_id,
            "status": "passed",
            "outcome": "healthy_exhausted",
            "validation_passed": True,
            "coverage_complete": True,
            "complete": True,
            "safe_for_completion_reasoning": True,
            "analyzer_health": {
                "status": "healthy",
                "completion_safe": True,
            },
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "HealthyExhaustionEvidence":
        allowed = {
            *cls.__dataclass_fields__.keys(),
            "schema",
            "version",
            "witness_id",
            "receipt_id",
            "provenance_cid",
            "artifact_digest",
            "requirement_ids",
            "proved_requirement_ids",
            "producer_kind",
            "source_tier",
            "repository_id",
            "repository_tree",
            "tree_id",
            "policy_id",
            "status",
            "outcome",
            "validation_passed",
            "coverage_complete",
            "complete",
            "safe_for_completion_reasoning",
            "analyzer_health",
            "wait_state",
        }
        _strict_keys(payload, allowed, record_name="healthy exhaustion evidence")
        if (
            payload.get("schema") != HEALTHY_EXHAUSTION_EVIDENCE_SCHEMA
            or int(payload.get("version", 0)) != 1
        ):
            raise ValueError("unsupported healthy exhaustion evidence schema")
        if payload.get("proved_requirement_ids") != [
            HEALTHY_EXHAUSTION_REQUIREMENT_ID
        ]:
            raise ValueError("healthy exhaustion proved-requirement projection is invalid")
        result = cls(
            binding=SelfImprovementEpochBinding.from_dict(
                payload.get("binding") or {}
            ),
            goal_projection=payload.get("goal_projection") or {},
            policy=payload.get("policy") or {},
            observations=tuple(payload.get("observations") or ()),
            exhaustion_quorum=payload.get("exhaustion_quorum") or {},
            objective_before_id=str(payload.get("objective_before_id") or ""),
            objective_after_id=str(payload.get("objective_after_id") or ""),
            taskboard_before_id=str(payload.get("taskboard_before_id") or ""),
            taskboard_after_id=str(payload.get("taskboard_after_id") or ""),
            observed_at=str(payload.get("observed_at") or ""),
            next_triggers=tuple(payload.get("next_triggers") or ()),
            classified_gap_count=payload.get("classified_gap_count"),
            candidate_count=payload.get("candidate_count"),
            admitted_count=payload.get("admitted_count"),
            materialized_count=payload.get("materialized_count"),
            taskboard_write_count=payload.get("taskboard_write_count"),
            requirement_id=str(payload.get("requirement_id") or ""),
            producer_version=str(payload.get("producer_version") or ""),
            evidence_id=str(payload.get("evidence_id") or ""),
        )
        for alias in (
            "witness_id",
            "receipt_id",
            "provenance_cid",
            "artifact_digest",
        ):
            if payload.get(alias) != result.evidence_id:
                raise ValueError(f"healthy exhaustion {alias} does not match")
        projected = result.to_dict()
        for name in (
            "requirement_ids",
            "producer_kind",
            "source_tier",
            "repository_id",
            "repository_tree",
            "tree_id",
            "policy_id",
            "status",
            "outcome",
            "validation_passed",
            "coverage_complete",
            "complete",
            "safe_for_completion_reasoning",
            "analyzer_health",
            "wait_state",
        ):
            if payload.get(name) != projected[name]:
                raise ValueError(
                    f"healthy exhaustion {name} projection does not match"
                )
        return result

    def completion_evidence(self) -> CompletionEvidence:
        """Project the operational witness into the canonical completion type."""

        return CompletionEvidence(
            acceptance_criterion=self.requirement_id,
            producing_task_or_scan=self.producer_version,
            producer_id=self.producer_version,
            producer_kind="scan",
            validation_receipt={
                "receipt_id": self.evidence_id,
                "status": "passed",
                "passed": True,
                "artifact_digest": self.evidence_id,
                "terminal_reason": "exhausted",
                "scan_mode": "drained_exhaustive",
                "safe_for_completion_reasoning": True,
            },
            repository_id=self.binding.repository_id,
            repository_tree=self.binding.repository_tree,
            observed_at=self.observed_at,
            fresh_until=min(item.fresh_until for item in self.observations),
            freshness="fresh",
            provenance_cid=self.evidence_id,
            validation_passed=True,
            metadata={
                "source_tier": "benchmark",
                "producer_kind": "benchmark",
                "requirement_id": self.requirement_id,
                "safe_for_completion_reasoning": True,
                "healthy_exhaustion_evidence": self.to_dict(),
            },
        )


@dataclass(frozen=True)
class SuccessorRefillEvidence:
    """Typed proof that one drained actionable epoch created bounded work."""

    binding: SelfImprovementEpochBinding
    goal_projection: ObjectiveEvidenceProjection
    policy: SelfImprovementPolicy
    observation_receipt_ids: tuple[str, ...]
    actionable_dimensions: tuple[str, ...]
    candidate_proposal_ids: tuple[str, ...]
    admitted_proposal_ids: tuple[str, ...]
    created_goal_ids: tuple[str, ...]
    created_task_ids: tuple[str, ...]
    transaction_id: str
    objective_before_id: str
    objective_after_id: str
    taskboard_before_id: str
    taskboard_after_id: str
    observed_at: datetime | str
    requirement_id: str = SUCCESSOR_REFILL_REQUIREMENT_ID
    producer_version: str = "bounded-successor-refill-evidence-producer/v1"
    evidence_id: str = ""

    def __post_init__(self) -> None:
        binding = (
            self.binding
            if isinstance(self.binding, SelfImprovementEpochBinding)
            else SelfImprovementEpochBinding.from_dict(self.binding)
        )
        projection = self.goal_projection
        if not isinstance(projection, ObjectiveEvidenceProjection):
            projection = ObjectiveEvidenceProjection.from_dict(projection)
        policy = (
            self.policy
            if isinstance(self.policy, SelfImprovementPolicy)
            else SelfImprovementPolicy.from_dict(self.policy)
        )
        observed = _utc_datetime(self.observed_at, field_name="observed_at")
        values: dict[str, tuple[str, ...]] = {}
        for name in (
            "observation_receipt_ids",
            "actionable_dimensions",
            "candidate_proposal_ids",
            "admitted_proposal_ids",
            "created_goal_ids",
            "created_task_ids",
        ):
            values[name] = _string_tuple(getattr(self, name), field_name=name)
        if self.requirement_id != SUCCESSOR_REFILL_REQUIREMENT_ID:
            raise ValueError("successor refill evidence claims the wrong requirement")
        if projection.requirement_id != self.requirement_id:
            raise ValueError("goal projection does not own the successor requirement")
        if projection.objective_heap_id != binding.objective_revision:
            raise ValueError("successor projection does not match the input heap")
        if policy.policy_id != binding.policy_id:
            raise ValueError("successor policy does not match the epoch")
        if not values["actionable_dimensions"] or not values["observation_receipt_ids"]:
            raise ValueError("successor evidence requires actionable observations")
        if not values["candidate_proposal_ids"]:
            raise ValueError("successor evidence requires candidates")
        if not values["admitted_proposal_ids"]:
            raise ValueError("successor evidence requires admitted proposals")
        if not set(values["admitted_proposal_ids"]).issubset(
            values["candidate_proposal_ids"]
        ):
            raise ValueError("every admitted proposal must be a candidate")
        if len(values["admitted_proposal_ids"]) != len(values["created_goal_ids"]):
            raise ValueError("every admitted proposal must create exactly one goal")
        if len(values["created_goal_ids"]) > policy.max_new_successor_goals:
            raise ValueError("created goals exceed the successor policy bound")
        if len(values["created_task_ids"]) != len(values["created_goal_ids"]):
            raise ValueError("every created goal must have one supervisor backlog task")
        if self.objective_before_id == self.objective_after_id:
            raise ValueError("successor evidence requires an objective heap change")
        if self.taskboard_before_id == self.taskboard_after_id:
            raise ValueError("successor evidence requires a taskboard change")
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "binding", binding)
        object.__setattr__(self, "goal_projection", projection)
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "observed_at", observed)
        object.__setattr__(
            self, "transaction_id", _required_text(self.transaction_id, "transaction_id")
        )
        object.__setattr__(
            self,
            "producer_version",
            _required_text(self.producer_version, "producer_version"),
        )
        expected = content_identity(self._identity_payload())
        supplied = str(self.evidence_id or "").strip()
        if supplied and supplied != expected:
            raise ValueError("successor refill evidence identity does not match")
        object.__setattr__(self, "evidence_id", expected)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": SUCCESSOR_REFILL_EVIDENCE_SCHEMA,
            "version": 1,
            "requirement_id": self.requirement_id,
            "producer_version": self.producer_version,
            "binding": self.binding.to_dict(),
            "goal_projection": self.goal_projection.to_dict(),
            "policy": self.policy.to_dict(),
            "observation_receipt_ids": self.observation_receipt_ids,
            "actionable_dimensions": self.actionable_dimensions,
            "candidate_proposal_ids": self.candidate_proposal_ids,
            "admitted_proposal_ids": self.admitted_proposal_ids,
            "created_goal_ids": self.created_goal_ids,
            "created_task_ids": self.created_task_ids,
            "transaction_id": self.transaction_id,
            "objective_before_id": self.objective_before_id,
            "objective_after_id": self.objective_after_id,
            "taskboard_before_id": self.taskboard_before_id,
            "taskboard_after_id": self.taskboard_after_id,
            "observed_at": self.observed_at.isoformat(),
        }

    @property
    def proved_requirement_ids(self) -> tuple[str, ...]:
        return (self.requirement_id,)

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._identity_payload(),
            "evidence_id": self.evidence_id,
            "receipt_id": self.evidence_id,
            "artifact_digest": self.evidence_id,
            "provenance_cid": self.evidence_id,
            "requirement_ids": [self.requirement_id],
            "proved_requirement_ids": [self.requirement_id],
            "producer_kind": "runtime",
            "source_tier": "runtime",
            "repository_id": self.binding.repository_id,
            "repository_tree": self.binding.repository_tree,
            "policy_id": self.binding.policy_id,
            "status": "passed",
            "validation_passed": True,
            "coverage_complete": True,
            "complete": True,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SuccessorRefillEvidence":
        allowed = {
            *cls.__dataclass_fields__.keys(),
            "schema",
            "version",
            "receipt_id",
            "artifact_digest",
            "provenance_cid",
            "requirement_ids",
            "proved_requirement_ids",
            "producer_kind",
            "source_tier",
            "repository_id",
            "repository_tree",
            "policy_id",
            "status",
            "validation_passed",
            "coverage_complete",
            "complete",
        }
        _strict_keys(payload, allowed, record_name="successor refill evidence")
        if (
            payload.get("schema") != SUCCESSOR_REFILL_EVIDENCE_SCHEMA
            or int(payload.get("version", 0)) != 1
        ):
            raise ValueError("unsupported successor refill evidence schema")
        result = cls(
            binding=SelfImprovementEpochBinding.from_dict(payload.get("binding") or {}),
            goal_projection=payload.get("goal_projection") or {},
            policy=payload.get("policy") or {},
            observation_receipt_ids=tuple(payload.get("observation_receipt_ids") or ()),
            actionable_dimensions=tuple(payload.get("actionable_dimensions") or ()),
            candidate_proposal_ids=tuple(payload.get("candidate_proposal_ids") or ()),
            admitted_proposal_ids=tuple(payload.get("admitted_proposal_ids") or ()),
            created_goal_ids=tuple(payload.get("created_goal_ids") or ()),
            created_task_ids=tuple(payload.get("created_task_ids") or ()),
            transaction_id=str(payload.get("transaction_id") or ""),
            objective_before_id=str(payload.get("objective_before_id") or ""),
            objective_after_id=str(payload.get("objective_after_id") or ""),
            taskboard_before_id=str(payload.get("taskboard_before_id") or ""),
            taskboard_after_id=str(payload.get("taskboard_after_id") or ""),
            observed_at=str(payload.get("observed_at") or ""),
            requirement_id=str(payload.get("requirement_id") or ""),
            producer_version=str(payload.get("producer_version") or ""),
            evidence_id=str(payload.get("evidence_id") or ""),
        )
        projected = result.to_dict()
        for name in (
            "receipt_id",
            "artifact_digest",
            "provenance_cid",
            "requirement_ids",
            "proved_requirement_ids",
            "producer_kind",
            "source_tier",
            "repository_id",
            "repository_tree",
            "policy_id",
            "status",
            "validation_passed",
            "coverage_complete",
            "complete",
        ):
            if payload.get(name) != projected[name]:
                raise ValueError(f"successor refill {name} projection does not match")
        return result

    def completion_evidence(self) -> CompletionEvidence:
        return CompletionEvidence(
            acceptance_criterion=self.requirement_id,
            producing_task_or_scan=self.producer_version,
            producer_id=self.producer_version,
            producer_kind="task",
            validation_receipt={
                "receipt_id": self.evidence_id,
                "status": "passed",
                "passed": True,
                "artifact_digest": self.evidence_id,
            },
            repository_id=self.binding.repository_id,
            repository_tree=self.binding.repository_tree,
            observed_at=self.observed_at,
            freshness="fresh",
            provenance_cid=self.evidence_id,
            validation_passed=True,
            metadata={
                "source_tier": "runtime",
                "requirement_id": self.requirement_id,
                "successor_refill_evidence": self.to_dict(),
            },
        )


@dataclass(frozen=True)
class SelfImprovementEpochReceipt:
    """Durable terminal account of one epoch, including non-evidentiary stops."""

    binding: SelfImprovementEpochBinding
    status: SelfImprovementEpochStatus | str
    observed_at: datetime | str
    observation_receipt_ids: tuple[str, ...] = ()
    blocker_codes: tuple[str, ...] = ()
    actionable_dimensions: tuple[str, ...] = ()
    evidence: HealthyExhaustionEvidence | None = None
    successor_evidence: SuccessorRefillEvidence | None = None
    created_goal_ids: tuple[str, ...] = ()
    created_task_ids: tuple[str, ...] = ()
    receipt_id: str = ""

    def __post_init__(self) -> None:
        binding = (
            self.binding
            if isinstance(self.binding, SelfImprovementEpochBinding)
            else SelfImprovementEpochBinding.from_dict(self.binding)
        )
        status = (
            self.status
            if isinstance(self.status, SelfImprovementEpochStatus)
            else SelfImprovementEpochStatus(str(self.status))
        )
        observed = _utc_datetime(self.observed_at, field_name="observed_at")
        observation_ids = _string_tuple(
            self.observation_receipt_ids,
            field_name="observation_receipt_ids",
        )
        blockers = _string_tuple(self.blocker_codes, field_name="blocker_codes")
        actionable = _string_tuple(
            self.actionable_dimensions, field_name="actionable_dimensions"
        )
        created = _string_tuple(self.created_goal_ids, field_name="created_goal_ids")
        evidence = self.evidence
        if evidence is not None and not isinstance(
            evidence, HealthyExhaustionEvidence
        ):
            evidence = HealthyExhaustionEvidence.from_dict(evidence)
        successor = self.successor_evidence
        if successor is not None and not isinstance(
            successor, SuccessorRefillEvidence
        ):
            successor = SuccessorRefillEvidence.from_dict(successor)
        created_tasks = _string_tuple(
            self.created_task_ids, field_name="created_task_ids"
        )
        if status is SelfImprovementEpochStatus.HEALTHY_EXHAUSTED:
            if (
                evidence is None
                or successor is not None
                or blockers
                or actionable
                or created
                or created_tasks
            ):
                raise ValueError("healthy exhausted epoch has inconsistent output")
            if evidence.binding.epoch_id != binding.epoch_id:
                raise ValueError("epoch evidence is bound to another epoch")
        elif status is SelfImprovementEpochStatus.SUCCESSORS_CREATED:
            if (
                successor is None
                or evidence is not None
                or blockers
                or not actionable
                or created != successor.created_goal_ids
                or created_tasks != successor.created_task_ids
            ):
                raise ValueError("successor-created epoch has inconsistent output")
            if successor.binding.epoch_id != binding.epoch_id:
                raise ValueError("successor evidence is bound to another epoch")
        elif evidence is not None or successor is not None or created or created_tasks:
            raise ValueError("non-producing epoch cannot carry output evidence")
        object.__setattr__(self, "binding", binding)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "observed_at", observed)
        object.__setattr__(self, "observation_receipt_ids", observation_ids)
        object.__setattr__(self, "blocker_codes", blockers)
        object.__setattr__(self, "actionable_dimensions", actionable)
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "successor_evidence", successor)
        object.__setattr__(self, "created_goal_ids", created)
        object.__setattr__(self, "created_task_ids", created_tasks)
        expected = content_identity(self._identity_payload())
        supplied = str(self.receipt_id or "").strip()
        if supplied and supplied != expected:
            raise ValueError("self-improvement epoch receipt identity does not match")
        object.__setattr__(self, "receipt_id", expected)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": (
                "ipfs_accelerate_py.agent_supervisor."
                "self_improvement_epoch_receipt.v1"
            ),
            "version": 1,
            "binding": self.binding.to_dict(),
            "status": self.status.value,
            "observed_at": self.observed_at.isoformat(),
            "observation_receipt_ids": self.observation_receipt_ids,
            "blocker_codes": self.blocker_codes,
            "actionable_dimensions": self.actionable_dimensions,
            "evidence": self.evidence.to_dict() if self.evidence else None,
            "successor_evidence": (
                self.successor_evidence.to_dict()
                if self.successor_evidence
                else None
            ),
            "created_goal_ids": self.created_goal_ids,
            "created_task_ids": self.created_task_ids,
        }

    @property
    def proved_requirement_ids(self) -> tuple[str, ...]:
        if self.evidence:
            return self.evidence.proved_requirement_ids
        if self.successor_evidence:
            return self.successor_evidence.proved_requirement_ids
        return ()

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._identity_payload(),
            "receipt_id": self.receipt_id,
            "epoch_id": self.binding.epoch_id,
            "proved_requirement_ids": list(self.proved_requirement_ids),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "SelfImprovementEpochReceipt":
        allowed = {
            "schema",
            "version",
            "binding",
            "status",
            "observed_at",
            "observation_receipt_ids",
            "blocker_codes",
            "actionable_dimensions",
            "evidence",
            "successor_evidence",
            "created_goal_ids",
            "created_task_ids",
            "receipt_id",
            "epoch_id",
            "proved_requirement_ids",
        }
        _strict_keys(payload, allowed, record_name="self-improvement epoch receipt")
        if (
            payload.get("schema")
            != (
                "ipfs_accelerate_py.agent_supervisor."
                "self_improvement_epoch_receipt.v1"
            )
            or int(payload.get("version", 0)) != 1
        ):
            raise ValueError("unsupported self-improvement epoch receipt schema")
        result = cls(
            binding=SelfImprovementEpochBinding.from_dict(
                payload.get("binding") or {}
            ),
            status=str(payload.get("status") or ""),
            observed_at=str(payload.get("observed_at") or ""),
            observation_receipt_ids=tuple(
                payload.get("observation_receipt_ids") or ()
            ),
            blocker_codes=tuple(payload.get("blocker_codes") or ()),
            actionable_dimensions=tuple(
                payload.get("actionable_dimensions") or ()
            ),
            evidence=(
                HealthyExhaustionEvidence.from_dict(payload["evidence"])
                if isinstance(payload.get("evidence"), Mapping)
                else None
            ),
            successor_evidence=(
                SuccessorRefillEvidence.from_dict(payload["successor_evidence"])
                if isinstance(payload.get("successor_evidence"), Mapping)
                else None
            ),
            created_goal_ids=tuple(payload.get("created_goal_ids") or ()),
            created_task_ids=tuple(payload.get("created_task_ids") or ()),
            receipt_id=str(payload.get("receipt_id") or ""),
        )
        if payload.get("epoch_id") != result.binding.epoch_id:
            raise ValueError("epoch receipt epoch_id does not match")
        if payload.get("proved_requirement_ids") != list(
            result.proved_requirement_ids
        ):
            raise ValueError("epoch receipt requirement projection does not match")
        return result


@dataclass(frozen=True)
class EpochReplayEvidence:
    """Typed proof that a persisted identical epoch performed zero work."""

    binding: SelfImprovementEpochBinding
    goal_projection: ObjectiveEvidenceProjection
    original_receipt_id: str
    objective_state_id: str
    taskboard_state_id: str
    replayed_at: datetime | str
    requirement_id: str = EPOCH_IDEMPOTENCY_REQUIREMENT_ID
    producer_version: str = "self-improvement-epoch-replay-evidence-producer/v1"
    provider_call_count: int = 0
    proposal_call_count: int = 0
    materialization_count: int = 0
    taskboard_write_count: int = 0
    evidence_id: str = ""

    def __post_init__(self) -> None:
        binding = (
            self.binding
            if isinstance(self.binding, SelfImprovementEpochBinding)
            else SelfImprovementEpochBinding.from_dict(self.binding)
        )
        projection = self.goal_projection
        if not isinstance(projection, ObjectiveEvidenceProjection):
            projection = ObjectiveEvidenceProjection.from_dict(projection)
        replayed = _utc_datetime(self.replayed_at, field_name="replayed_at")
        if self.requirement_id != EPOCH_IDEMPOTENCY_REQUIREMENT_ID:
            raise ValueError("replay evidence claims the wrong requirement")
        if projection.requirement_id != self.requirement_id:
            raise ValueError("goal projection does not own the replay requirement")
        if projection.objective_heap_id != binding.objective_revision:
            raise ValueError("replay projection does not match the current heap")
        for name in (
            "provider_call_count",
            "proposal_call_count",
            "materialization_count",
            "taskboard_write_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != 0:
                raise ValueError(f"{name} must be zero on exact replay")
        object.__setattr__(self, "binding", binding)
        object.__setattr__(self, "goal_projection", projection)
        object.__setattr__(self, "replayed_at", replayed)
        for name in (
            "original_receipt_id",
            "objective_state_id",
            "taskboard_state_id",
            "producer_version",
        ):
            object.__setattr__(
                self, name, _required_text(getattr(self, name), name)
            )
        expected = content_identity(self._identity_payload())
        supplied = str(self.evidence_id or "").strip()
        if supplied and supplied != expected:
            raise ValueError("epoch replay evidence identity does not match")
        object.__setattr__(self, "evidence_id", expected)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": EPOCH_REPLAY_EVIDENCE_SCHEMA,
            "version": 1,
            "requirement_id": self.requirement_id,
            "producer_version": self.producer_version,
            "binding": self.binding.to_dict(),
            "goal_projection": self.goal_projection.to_dict(),
            "original_receipt_id": self.original_receipt_id,
            "objective_state_id": self.objective_state_id,
            "taskboard_state_id": self.taskboard_state_id,
            "replayed_at": self.replayed_at.isoformat(),
            "provider_call_count": 0,
            "proposal_call_count": 0,
            "materialization_count": 0,
            "taskboard_write_count": 0,
        }

    @property
    def proved_requirement_ids(self) -> tuple[str, ...]:
        return (self.requirement_id,)

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._identity_payload(),
            "evidence_id": self.evidence_id,
            "receipt_id": self.evidence_id,
            "artifact_digest": self.evidence_id,
            "provenance_cid": self.evidence_id,
            "requirement_ids": [self.requirement_id],
            "proved_requirement_ids": [self.requirement_id],
            "producer_kind": "runtime",
            "source_tier": "runtime",
            "repository_id": self.binding.repository_id,
            "repository_tree": self.binding.repository_tree,
            "tree_id": self.binding.repository_tree,
            "policy_id": self.binding.policy_id,
            "status": "passed",
            "validation_passed": True,
            "coverage_complete": True,
            "complete": True,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EpochReplayEvidence":
        allowed = {
            *cls.__dataclass_fields__.keys(),
            "schema",
            "version",
            "receipt_id",
            "artifact_digest",
            "provenance_cid",
            "requirement_ids",
            "proved_requirement_ids",
            "producer_kind",
            "source_tier",
            "repository_id",
            "repository_tree",
            "tree_id",
            "policy_id",
            "status",
            "validation_passed",
            "coverage_complete",
            "complete",
        }
        _strict_keys(payload, allowed, record_name="epoch replay evidence")
        if (
            payload.get("schema") != EPOCH_REPLAY_EVIDENCE_SCHEMA
            or int(payload.get("version", 0)) != 1
        ):
            raise ValueError("unsupported epoch replay evidence schema")
        result = cls(
            binding=SelfImprovementEpochBinding.from_dict(
                payload.get("binding") or {}
            ),
            goal_projection=payload.get("goal_projection") or {},
            original_receipt_id=str(payload.get("original_receipt_id") or ""),
            objective_state_id=str(payload.get("objective_state_id") or ""),
            taskboard_state_id=str(payload.get("taskboard_state_id") or ""),
            replayed_at=str(payload.get("replayed_at") or ""),
            requirement_id=str(payload.get("requirement_id") or ""),
            producer_version=str(payload.get("producer_version") or ""),
            provider_call_count=int(payload.get("provider_call_count", -1)),
            proposal_call_count=int(payload.get("proposal_call_count", -1)),
            materialization_count=int(payload.get("materialization_count", -1)),
            taskboard_write_count=int(payload.get("taskboard_write_count", -1)),
            evidence_id=str(payload.get("evidence_id") or ""),
        )
        projected = result.to_dict()
        for name in allowed - set(cls.__dataclass_fields__):
            if name in {"schema", "version"}:
                continue
            if payload.get(name) != projected[name]:
                raise ValueError(f"epoch replay {name} projection does not match")
        return result

    def completion_evidence(self) -> CompletionEvidence:
        return CompletionEvidence(
            acceptance_criterion=self.requirement_id,
            producing_task_or_scan=self.producer_version,
            producer_id=self.producer_version,
            producer_kind="task",
            validation_receipt={
                "receipt_id": self.evidence_id,
                "status": "passed",
                "passed": True,
                "artifact_digest": self.evidence_id,
                "provider_call_count": 0,
                "proposal_call_count": 0,
                "materialization_count": 0,
                "taskboard_write_count": 0,
            },
            repository_id=self.binding.repository_id,
            repository_tree=self.binding.repository_tree,
            observed_at=self.replayed_at,
            freshness="fresh",
            provenance_cid=self.evidence_id,
            validation_passed=True,
            metadata={
                "source_tier": "runtime",
                "requirement_id": self.requirement_id,
                "epoch_replay_evidence": self.to_dict(),
            },
        )


@dataclass(frozen=True)
class SelfImprovementEpochRun:
    """Runtime wrapper distinguishing exact replay from first evaluation."""

    receipt: SelfImprovementEpochReceipt
    replayed: bool = False
    replay_evidence: EpochReplayEvidence | None = None

    @property
    def status(self) -> SelfImprovementEpochStatus:
        return self.receipt.status

    @property
    def evidence(self) -> HealthyExhaustionEvidence | None:
        return self.receipt.evidence

    @property
    def epoch_id(self) -> str:
        return self.receipt.binding.epoch_id

    @property
    def proved_requirement_ids(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (
                    *self.receipt.proved_requirement_ids,
                    *(
                        self.replay_evidence.proved_requirement_ids
                        if self.replay_evidence
                        else ()
                    ),
                )
            )
        )


def _observation_scan_receipts(
    observations: Sequence[BenchmarkObservation],
    *,
    binding: SelfImprovementEpochBinding,
    observed_at: datetime,
) -> tuple[RefillScanResult[Any], ...]:
    """Project healthy benchmark channels into canonical quorum receipts."""

    exact_binding = ExhaustionBinding(
        repository_id=binding.repository_id,
        tree_id=binding.repository_tree,
        analyzer_version=SELF_IMPROVEMENT_ANALYZER_VERSION,
        configuration_revision=binding.policy_id,
        objective_revision=binding.objective_revision,
    )
    by_channel: dict[str, list[BenchmarkObservation]] = {}
    for observation in observations:
        by_channel.setdefault(observation.evidence_channel, []).append(observation)
    receipts: list[RefillScanResult[Any]] = []
    for channel, members in sorted(by_channel.items()):
        started_at = min(item.observed_at for item in members)
        receipts.append(
            build_scan_result(
                ScanTerminalReason.EXHAUSTED,
                "drained_exhaustive",
                SELF_IMPROVEMENT_ANALYZER_VERSION,
                Path("."),
                started_at,
                finished_at=observed_at,
                safe_for_completion_reasoning=True,
                identity=RepositoryTreeIdentity(
                    binding.repository_id,
                    binding.repository_tree,
                ),
                metadata={
                    "analyzer_health": {
                        "status": "healthy",
                        "completion_safe": True,
                    },
                    "coverage_complete": True,
                    "exhaustive": True,
                    "evidence_channel": channel,
                    "configuration_revision": binding.policy_id,
                    "objective_revision": binding.objective_revision,
                    "exhaustion_binding": exact_binding.to_dict(),
                    "benchmark_receipt_ids": [
                        item.receipt_id for item in members
                    ],
                },
            )
        )
    return tuple(receipts)


def evaluate_self_improvement_epoch(
    *,
    binding: SelfImprovementEpochBinding,
    projection: ObjectiveEvidenceProjection,
    policy: SelfImprovementPolicy,
    observations: Iterable[BenchmarkObservation | Mapping[str, Any]],
    board_drained: bool,
    objective_before_id: str,
    objective_after_id: str,
    taskboard_before_id: str,
    taskboard_after_id: str,
    objective_written_during_epoch: bool = False,
    taskboard_written_during_epoch: bool = False,
    observed_at: datetime | str | None = None,
) -> SelfImprovementEpochReceipt:
    """Evaluate one epoch without mutating goals, tasks, or strategy state."""

    now = _utc_datetime(observed_at, field_name="observed_at")
    normalized: tuple[BenchmarkObservation, ...] = tuple(
        item
        if isinstance(item, BenchmarkObservation)
        else BenchmarkObservation.from_dict(item)
        for item in observations
    )
    blockers: list[str] = []
    if not board_drained:
        blockers.append("taskboard_not_drained")
    if projection.requirement_id != HEALTHY_EXHAUSTION_REQUIREMENT_ID:
        blockers.append("objective_requirement_owner_mismatch")
    if projection.objective_heap_id != binding.objective_revision:
        blockers.append("objective_revision_mismatch")
    if policy.policy_id != binding.policy_id:
        blockers.append("policy_mismatch")
    if objective_before_id != objective_after_id:
        blockers.append("objective_mutated_during_epoch")
    if objective_written_during_epoch:
        blockers.append("objective_written_during_epoch")
    if taskboard_before_id != taskboard_after_id:
        blockers.append("taskboard_mutated_during_epoch")
    if taskboard_written_during_epoch:
        blockers.append("taskboard_written_during_epoch")
    by_channel: dict[str, list[BenchmarkObservation]] = {}
    for observation in normalized:
        by_channel.setdefault(observation.evidence_channel, []).append(
            observation
        )
    if not by_channel or any(
        tuple(sorted(item.dimension for item in members))
        != policy.required_dimensions
        for members in by_channel.values()
    ):
        blockers.append("benchmark_population_incomplete")
    foreign = [
        item
        for item in normalized
        if (
            item.repository_id != binding.repository_id
            or item.repository_tree != binding.repository_tree
            or item.policy_id != binding.policy_id
            or item.capability_snapshot_id != binding.capability_snapshot_id
        )
    ]
    if foreign:
        blockers.append("benchmark_binding_mismatch")
    stale_or_partial = [
        item for item in normalized if not item.healthy_at(now) and not item.disposition.actionable
    ]
    if stale_or_partial:
        blockers.append("benchmark_not_fresh_and_complete")
    actionable = tuple(
        sorted(
            {
                item.dimension
                for item in normalized
                if item.disposition.actionable
            }
        )
    )
    observation_ids = tuple(item.receipt_id for item in normalized)
    # Binding, population, freshness, board-state, and artifact-integrity
    # blockers always outrank an actionable classification.  Otherwise a
    # foreign regression could become write authority merely by being severe.
    if blockers:
        return SelfImprovementEpochReceipt(
            binding=binding,
            status=SelfImprovementEpochStatus.INELIGIBLE,
            observed_at=now,
            observation_receipt_ids=observation_ids,
            blocker_codes=tuple(blockers),
            actionable_dimensions=actionable,
        )
    if actionable:
        return SelfImprovementEpochReceipt(
            binding=binding,
            status=SelfImprovementEpochStatus.ACTIONABLE,
            observed_at=now,
            observation_receipt_ids=observation_ids,
            blocker_codes=tuple(blockers),
            actionable_dimensions=actionable,
        )
    scan_receipts = _observation_scan_receipts(
        normalized, binding=binding, observed_at=now
    )
    exact_binding = ExhaustionBinding(
        repository_id=binding.repository_id,
        tree_id=binding.repository_tree,
        analyzer_version=SELF_IMPROVEMENT_ANALYZER_VERSION,
        configuration_revision=binding.policy_id,
        objective_revision=binding.objective_revision,
    )
    quorum = evaluate_exhaustion_quorum(
        scan_receipts,
        binding=exact_binding,
        required_members=policy.required_independent_channels,
    )
    if not quorum.satisfied:
        return SelfImprovementEpochReceipt(
            binding=binding,
            status=SelfImprovementEpochStatus.INELIGIBLE,
            observed_at=now,
            observation_receipt_ids=observation_ids,
            blocker_codes=("exhaustion_quorum_unsatisfied",),
        )
    evidence = HealthyExhaustionEvidence(
        binding=binding,
        goal_projection=projection,
        policy=policy,
        observations=normalized,
        exhaustion_quorum=quorum,
        objective_before_id=objective_before_id,
        objective_after_id=objective_after_id,
        taskboard_before_id=taskboard_before_id,
        taskboard_after_id=taskboard_after_id,
        observed_at=now,
        next_triggers=policy.next_triggers,
    )
    return SelfImprovementEpochReceipt(
        binding=binding,
        status=SelfImprovementEpochStatus.HEALTHY_EXHAUSTED,
        observed_at=now,
        observation_receipt_ids=observation_ids,
        evidence=evidence,
    )


def _artifact_content_id(data: bytes, *, kind: str) -> str:
    return content_identity(
        {
            "schema": (
                "ipfs_accelerate_py.agent_supervisor."
                f"self_improvement_{kind}.v1"
            ),
            "content": data.decode("utf-8", errors="surrogateescape"),
        }
    )


def _file_version(path: Path) -> tuple[int, int, int, int, int]:
    """Return metadata that changes for writes, including same-byte rewrites."""

    stat = path.stat()
    return (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )


def build_self_improvement_epoch_binding(
    *,
    repo_root: Path,
    objective_text: str,
    taskboard_bytes: bytes,
    policy: SelfImprovementPolicy,
    capability_snapshot_id: str,
    observation_window: str,
    operator_revision: str = "operator-objective/v1",
    objective_path: Path | None = None,
    materialization_journal_path: Path | None = None,
    control_paths: Sequence[Path] = (),
) -> SelfImprovementEpochBinding:
    """Build the exact trigger identity before any benchmark callback runs."""

    identity = (
        objective_materialization_tree_identity(
            repo_root,
            objective_path=objective_path,
            journal_path=(
                materialization_journal_path
                or objective_path.with_name(
                    f".{objective_path.name}.self-improvement.json"
                )
            ),
            control_paths=control_paths,
        )
        if objective_path is not None
        else scan_identity(repo_root)
    )
    return SelfImprovementEpochBinding(
        repository_id=identity.repository_id,
        repository_tree=identity.tree_id,
        objective_revision=objective_heap_content_id(objective_text),
        taskboard_revision=_artifact_content_id(taskboard_bytes, kind="taskboard"),
        policy_id=policy.policy_id,
        capability_snapshot_id=capability_snapshot_id,
        observation_window=observation_window,
        operator_revision=operator_revision,
    )


def _load_epoch_ledger(path: Path) -> dict[str, SelfImprovementEpochReceipt]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid self-improvement epoch ledger: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("self-improvement epoch ledger must contain an object")
    _strict_keys(
        payload,
        {"schema", "version", "epochs"},
        record_name="self-improvement epoch ledger",
    )
    if (
        payload.get("schema") != SELF_IMPROVEMENT_LEDGER_SCHEMA
        or int(payload.get("version", 0)) != 1
    ):
        raise ValueError("unsupported self-improvement epoch ledger schema")
    raw_epochs = payload.get("epochs")
    if not isinstance(raw_epochs, Mapping):
        raise ValueError("self-improvement epoch ledger epochs must be an object")
    result: dict[str, SelfImprovementEpochReceipt] = {}
    for epoch_id, raw in raw_epochs.items():
        if not isinstance(raw, Mapping):
            raise ValueError("self-improvement epoch ledger receipt must be an object")
        receipt = SelfImprovementEpochReceipt.from_dict(raw)
        if str(epoch_id) != receipt.binding.epoch_id:
            raise ValueError("self-improvement ledger epoch key does not match")
        result[str(epoch_id)] = receipt
    return result


def _persist_epoch_ledger(
    path: Path, receipts: Mapping[str, SelfImprovementEpochReceipt]
) -> None:
    _atomic_write_json(
        path,
        {
            "schema": SELF_IMPROVEMENT_LEDGER_SCHEMA,
            "version": 1,
            "epochs": {
                key: receipts[key].to_dict() for key in sorted(receipts)
            },
        },
    )


def _project_wait_state(
    strategy_path: Path,
    evidence: HealthyExhaustionEvidence,
) -> None:
    strategy = load_strategy(strategy_path)
    if self_improvement_epoch_wait_active(
        strategy,
        epoch_id=evidence.binding.epoch_id,
        evidence_id=evidence.evidence_id,
        requirement_id=evidence.requirement_id,
        next_triggers=evidence.next_triggers,
    ):
        return
    record_self_improvement_exhaustion(
        strategy_path,
        epoch_id=evidence.binding.epoch_id,
        evidence_id=evidence.evidence_id,
        requirement_id=evidence.requirement_id,
        quorum=evidence.exhaustion_quorum.to_dict(),
        next_triggers=evidence.next_triggers,
        recorded_at=evidence.observed_at.isoformat(),
    )


def materialize_self_improvement_successors(
    *,
    receipt: SelfImprovementEpochReceipt,
    proposals: Iterable[ObjectiveWorkProposal | Mapping[str, Any]],
    repo_root: Path,
    objective_path: Path,
    todo_path: Path,
    materialization_journal_path: Path,
    discovery_dir: Path,
    bundle_dir: Path,
    strategy_path: Path,
    policy: SelfImprovementPolicy,
    state_path: Path | None = None,
    task_prefix: str = DEFAULT_TASK_ID_PREFIX,
    control_paths: Sequence[Path] = (),
    expected_goal_id: str = DEFAULT_SUCCESSOR_REFILL_GOAL_ID,
    expected_parent_goal_id: str = DEFAULT_SELF_IMPROVEMENT_PARENT_GOAL_ID,
    observed_at: datetime | str | None = None,
) -> SelfImprovementEpochReceipt:
    """Admit, commit, and backlog-project one actionable successor batch.

    The objective preview and transaction journal provide the CAS/restart
    boundary.  Evidence is emitted only after the forced objective refill has
    created exactly one supervisor task for every committed goal.
    """

    if receipt.status is not SelfImprovementEpochStatus.ACTIONABLE:
        raise ValueError("only an actionable epoch can create successor goals")
    if receipt.blocker_codes or not receipt.actionable_dimensions:
        raise ValueError("blocked or unclassified epochs cannot create successors")
    objective_before = objective_path.read_bytes()
    taskboard_before = todo_path.read_bytes()
    objective_text = objective_before.decode("utf-8")
    if objective_heap_content_id(objective_text) != receipt.binding.objective_revision:
        raise ValueError("objective heap changed after actionable classification")
    if (
        _artifact_content_id(taskboard_before, kind="taskboard")
        != receipt.binding.taskboard_revision
    ):
        raise ValueError("taskboard changed after actionable classification")
    projection = resolve_objective_evidence_projection(
        objective_text,
        requirement_id=SUCCESSOR_REFILL_REQUIREMENT_ID,
        expected_goal_id=expected_goal_id,
        expected_parent_goal_id=expected_parent_goal_id,
    )
    normalized = tuple(
        item
        if isinstance(item, ObjectiveWorkProposal)
        else ObjectiveWorkProposal.from_dict(item)
        for item in proposals
    )
    if not normalized:
        raise ValueError("actionable epoch proposal provider returned no candidates")
    low_quality = [
        item.canonical_id
        for item in normalized
        if (
            item.confidence < policy.minimum_successor_confidence
            or item.novelty < policy.minimum_successor_novelty
            or item.kind not in {ObjectiveWorkKind.GOAL, ObjectiveWorkKind.SUBGOAL}
        )
    ]
    if low_quality:
        raise ValueError(
            "successor candidates failed quality, novelty, or kind policy: "
            + ", ".join(sorted(low_quality))
        )
    limits = ObjectiveGenerationLimits(
        max_depth=policy.max_successor_depth,
        max_breadth_per_parent=policy.max_successor_breadth,
        max_new_work=policy.max_new_successor_goals,
        max_open_work=policy.max_open_successor_goals,
        token_budget=policy.successor_token_budget,
    )
    preview = preview_objective_goal_materialization(
        objective_text,
        normalized,
        policy=ObjectiveGoalMaterializationPolicy(
            limits=limits,
            expected_heap_content_id=receipt.binding.objective_revision,
            lifecycle_owner="self_improvement_epoch",
            atomic=False,
        ),
    )
    if not preview.ready or not preview.materialized:
        reasons = [
            *preview.fatal_reasons,
            *(item.reason for item in preview.rejected),
        ]
        raise ValueError(
            "no bounded novel successor proposal was admissible"
            + (": " + ", ".join(sorted(set(reasons))) if reasons else "")
        )
    transaction = commit_objective_goal_materialization(
        repo_root=repo_root,
        objective_path=objective_path,
        journal_path=materialization_journal_path,
        preview=preview,
        expected_repository_tree_id=receipt.binding.repository_tree,
        control_paths=(
            todo_path,
            strategy_path,
            discovery_dir,
            bundle_dir,
            *control_paths,
        ),
    )
    if transaction.state is not ObjectiveMaterializationTransactionState.COMMITTED:
        raise RuntimeError(
            "successor objective transaction did not commit: "
            + ", ".join(transaction.reason_codes)
        )
    admitted_proposal_ids = transaction.admitted_proposal_ids
    goal_ids = tuple(item.goal.goal_id for item in preview.materialized)
    refill = record_objective_backlog_findings(
        repo_root=repo_root,
        objective_path=objective_path,
        todo_path=todo_path,
        discovery_dir=discovery_dir,
        bundle_dir=bundle_dir,
        strategy_path=strategy_path,
        state_path=state_path,
        task_prefix=task_prefix,
        min_open_tasks=max(1, len(goal_ids)),
        max_findings=max(1, len(goal_ids)),
        cooldown_seconds=0,
        force=True,
        force_goal_ids=goal_ids,
        persist_ast_dataset=False,
        write_todo_vector_index=False,
    )
    task_by_goal = {
        str(item.get("goal_id") or ""): str(item.get("follow_up_task_id") or "")
        for item in refill.items
        if isinstance(item, Mapping)
    }
    if set(task_by_goal) != set(goal_ids) or any(not value for value in task_by_goal.values()):
        raise RuntimeError(
            "successor objective commit was not projected exactly into the "
            f"supervisor backlog: expected={sorted(goal_ids)!r}, "
            f"projected={sorted(task_by_goal)!r}, "
            f"terminal_reason={refill.terminal_reason.value!r}, "
            f"error={refill.error!r}"
        )
    objective_after = objective_path.read_bytes()
    taskboard_after = todo_path.read_bytes()
    evidence = SuccessorRefillEvidence(
        binding=receipt.binding,
        goal_projection=projection,
        policy=policy,
        observation_receipt_ids=receipt.observation_receipt_ids,
        actionable_dimensions=receipt.actionable_dimensions,
        candidate_proposal_ids=tuple(item.canonical_id for item in normalized),
        admitted_proposal_ids=admitted_proposal_ids,
        created_goal_ids=goal_ids,
        created_task_ids=tuple(task_by_goal[goal_id] for goal_id in goal_ids),
        transaction_id=transaction.transaction_id,
        objective_before_id=_artifact_content_id(
            objective_before, kind="objective"
        ),
        objective_after_id=_artifact_content_id(
            objective_after, kind="objective"
        ),
        taskboard_before_id=_artifact_content_id(
            taskboard_before, kind="taskboard"
        ),
        taskboard_after_id=_artifact_content_id(
            taskboard_after, kind="taskboard"
        ),
        observed_at=observed_at or receipt.observed_at,
    )
    return SelfImprovementEpochReceipt(
        binding=receipt.binding,
        status=SelfImprovementEpochStatus.SUCCESSORS_CREATED,
        observed_at=observed_at or receipt.observed_at,
        observation_receipt_ids=receipt.observation_receipt_ids,
        actionable_dimensions=receipt.actionable_dimensions,
        successor_evidence=evidence,
        created_goal_ids=evidence.created_goal_ids,
        created_task_ids=evidence.created_task_ids,
    )


def run_self_improvement_epoch(
    *,
    repo_root: Path,
    objective_path: Path,
    todo_path: Path,
    ledger_path: Path,
    strategy_path: Path,
    observation_provider: Callable[
        [SelfImprovementEpochBinding],
        Iterable[BenchmarkObservation | Mapping[str, Any]],
    ],
    capability_snapshot_id: str,
    observation_window: str,
    proposal_provider: Callable[
        [SelfImprovementEpochBinding, tuple[BenchmarkObservation, ...]],
        Iterable[ObjectiveWorkProposal | Mapping[str, Any]],
    ]
    | None = None,
    materialization_journal_path: Path | None = None,
    discovery_dir: Path | None = None,
    bundle_dir: Path | None = None,
    policy: SelfImprovementPolicy | None = None,
    state_path: Path | None = None,
    task_prefix: str = DEFAULT_TASK_ID_PREFIX,
    operator_revision: str = "operator-objective/v1",
    expected_goal_id: str = DEFAULT_SELF_IMPROVEMENT_GOAL_ID,
    expected_successor_goal_id: str = DEFAULT_SUCCESSOR_REFILL_GOAL_ID,
    expected_idempotency_goal_id: str = DEFAULT_EPOCH_IDEMPOTENCY_GOAL_ID,
    expected_parent_goal_id: str = DEFAULT_SELF_IMPROVEMENT_PARENT_GOAL_ID,
    observed_at: datetime | str | None = None,
) -> SelfImprovementEpochRun:
    """Run or exactly replay one benchmark-driven self-refill epoch.

    The identity and ledger replay check happen before either provider is
    called.  Benchmark callbacks are read-only.  When an explicit proposal
    provider and materialization paths are supplied, a blocker-free actionable
    epoch is admitted through the bounded objective transaction and then
    projected into the supervisor backlog.
    """

    if not callable(observation_provider):
        raise TypeError("observation_provider must be callable")
    active_policy = policy or SelfImprovementPolicy()
    journal_path = materialization_journal_path or ledger_path.with_name(
        "self-improvement-objective-materialization.json"
    )
    resolved_discovery_dir = discovery_dir or repo_root / "data" / "agent_supervisor" / "discovery"
    resolved_bundle_dir = bundle_dir or repo_root / "data" / "agent_supervisor" / "objective_bundles"
    objective_before = objective_path.read_bytes()
    taskboard_before = todo_path.read_bytes()
    objective_version_before = _file_version(objective_path)
    taskboard_version_before = _file_version(todo_path)
    objective_text = objective_before.decode("utf-8")
    projection = resolve_objective_evidence_projection(
        objective_text,
        requirement_id=HEALTHY_EXHAUSTION_REQUIREMENT_ID,
        expected_parent_goal_id=expected_parent_goal_id,
        expected_goal_id=expected_goal_id,
    )
    binding = build_self_improvement_epoch_binding(
        repo_root=repo_root,
        objective_text=objective_text,
        taskboard_bytes=taskboard_before,
        policy=active_policy,
        capability_snapshot_id=capability_snapshot_id,
        observation_window=observation_window,
        operator_revision=operator_revision,
        objective_path=objective_path,
        materialization_journal_path=journal_path,
        control_paths=(
            todo_path,
            ledger_path,
            strategy_path,
            resolved_discovery_dir,
            resolved_bundle_dir,
        ),
    )
    receipts = _load_epoch_ledger(ledger_path)

    def replay_evidence_for(
        original: SelfImprovementEpochReceipt,
    ) -> EpochReplayEvidence | None:
        try:
            projection = resolve_objective_evidence_projection(
                objective_text,
                requirement_id=EPOCH_IDEMPOTENCY_REQUIREMENT_ID,
                expected_parent_goal_id=expected_parent_goal_id,
                expected_goal_id=expected_idempotency_goal_id,
            )
        except ValueError:
            # Legacy heaps predating ASI-G110 may replay operationally but
            # cannot claim its objective requirement.
            return None
        return EpochReplayEvidence(
            binding=binding,
            goal_projection=projection,
            original_receipt_id=original.receipt_id,
            objective_state_id=_artifact_content_id(
                objective_before, kind="objective"
            ),
            taskboard_state_id=_artifact_content_id(
                taskboard_before, kind="taskboard"
            ),
            replayed_at=observed_at,
        )

    existing = receipts.get(binding.epoch_id)
    if existing is not None:
        if existing.evidence is not None:
            _project_wait_state(strategy_path, existing.evidence)
        return SelfImprovementEpochRun(
            existing,
            replayed=True,
            replay_evidence=replay_evidence_for(existing),
        )
    current_objective_state = _artifact_content_id(
        objective_before, kind="objective"
    )
    current_taskboard_state = _artifact_content_id(
        taskboard_before, kind="taskboard"
    )
    for prior in receipts.values():
        successor = prior.successor_evidence
        if successor is None:
            continue
        same_external_binding = (
            prior.binding.repository_id == binding.repository_id
            and prior.binding.repository_tree == binding.repository_tree
            and prior.binding.policy_id == binding.policy_id
            and prior.binding.capability_snapshot_id
            == binding.capability_snapshot_id
            and prior.binding.observation_window == binding.observation_window
            and prior.binding.operator_revision == binding.operator_revision
        )
        if (
            same_external_binding
            and successor.objective_after_id == current_objective_state
            and successor.taskboard_after_id == current_taskboard_state
        ):
            return SelfImprovementEpochRun(
                prior,
                replayed=True,
                replay_evidence=replay_evidence_for(prior),
            )

    board_drained = (
        effective_open_task_count(
            taskboard_before.decode("utf-8"),
            state_path=state_path,
            task_prefix=task_prefix,
        )
        == 0
    )
    observations = tuple(observation_provider(binding))
    objective_after = objective_path.read_bytes()
    taskboard_after = todo_path.read_bytes()
    objective_version_after = _file_version(objective_path)
    taskboard_version_after = _file_version(todo_path)
    now = _utc_datetime(observed_at, field_name="observed_at")
    receipt = evaluate_self_improvement_epoch(
        binding=binding,
        projection=projection,
        policy=active_policy,
        observations=observations,
        board_drained=board_drained,
        objective_before_id=_artifact_content_id(
            objective_before, kind="objective"
        ),
        objective_after_id=_artifact_content_id(
            objective_after, kind="objective"
        ),
        taskboard_before_id=_artifact_content_id(
            taskboard_before, kind="taskboard"
        ),
        taskboard_after_id=_artifact_content_id(
            taskboard_after, kind="taskboard"
        ),
        objective_written_during_epoch=(
            objective_version_before != objective_version_after
        ),
        taskboard_written_during_epoch=(
            taskboard_version_before != taskboard_version_after
        ),
        observed_at=now,
    )
    if (
        receipt.status is SelfImprovementEpochStatus.ACTIONABLE
        and proposal_provider is not None
    ):
        proposals = tuple(proposal_provider(binding, observations))
        receipt = materialize_self_improvement_successors(
            receipt=receipt,
            proposals=proposals,
            repo_root=repo_root,
            objective_path=objective_path,
            todo_path=todo_path,
            materialization_journal_path=journal_path,
            discovery_dir=resolved_discovery_dir,
            bundle_dir=resolved_bundle_dir,
            strategy_path=strategy_path,
            policy=active_policy,
            state_path=state_path,
            task_prefix=task_prefix,
            control_paths=(ledger_path,),
            expected_goal_id=expected_successor_goal_id,
            expected_parent_goal_id=expected_parent_goal_id,
            observed_at=now,
        )
    receipts[binding.epoch_id] = receipt
    _persist_epoch_ledger(ledger_path, receipts)
    if receipt.evidence is not None:
        _project_wait_state(strategy_path, receipt.evidence)
    return SelfImprovementEpochRun(receipt, replayed=False)


__all__ = [
    "BENCHMARK_OBSERVATION_SCHEMA",
    "DEFAULT_BENCHMARK_DIMENSIONS",
    "DEFAULT_EPOCH_IDEMPOTENCY_GOAL_ID",
    "DEFAULT_MEANINGFUL_TRIGGERS",
    "DEFAULT_SELF_IMPROVEMENT_GOAL_ID",
    "DEFAULT_SELF_IMPROVEMENT_PARENT_GOAL_ID",
    "DEFAULT_SUCCESSOR_REFILL_GOAL_ID",
    "EPOCH_IDEMPOTENCY_REQUIREMENT_ID",
    "EPOCH_REPLAY_EVIDENCE_SCHEMA",
    "HEALTHY_EXHAUSTION_EVIDENCE_SCHEMA",
    "HEALTHY_EXHAUSTION_REQUIREMENT_ID",
    "SELF_IMPROVEMENT_ANALYZER_VERSION",
    "SUCCESSOR_REFILL_EVIDENCE_SCHEMA",
    "SUCCESSOR_REFILL_REQUIREMENT_ID",
    "BenchmarkDisposition",
    "BenchmarkObservation",
    "EpochReplayEvidence",
    "HealthyExhaustionEvidence",
    "SelfImprovementEpochBinding",
    "SelfImprovementEpochReceipt",
    "SelfImprovementEpochRun",
    "SelfImprovementEpochStatus",
    "SelfImprovementPolicy",
    "SuccessorRefillEvidence",
    "build_self_improvement_epoch_binding",
    "evaluate_self_improvement_epoch",
    "materialize_self_improvement_successors",
    "run_self_improvement_epoch",
]
