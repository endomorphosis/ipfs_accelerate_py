"""Reward-resistant generation-2 self-evaluation.

The generation-2 benchmark freezes paired executions.  This module is the
policy layer which evaluates those executions together with compact
producer-owned component receipts.  It deliberately has no provider or
mutation dependency: every value in the Pareto vector is replayed from the
receipts, and every integrity failure keeps the candidate in shadow.

The contract is intentionally aggregate.  Source receipt identities retain
the exact fixture population while metric samples retain only bounded integer
counts.  Prompts, patches, source bodies, and model output never cross this
boundary.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from .supervisor_v2_benchmark import (
    REQUIRED_V2_FIXTURE_KINDS,
    V2BenchmarkArm,
    V2FixtureKind,
    V2PairedBenchmarkCorpus,
    build_v2_benchmark_report,
)


REWARD_RESISTANT_EVALUATION_REQUIREMENT_ID: Final[str] = (
    "244518415414864367783784212238716548679"
)
TYPED_SUCCESSOR_REQUIREMENT_ID: Final[str] = (
    "330240498615714141723029264005175932988"
)
REWARD_RESISTANT_EVALUATION_GOAL_ID: Final[str] = "ASI-G290"
V2_SELF_EVALUATION_CONTRACT_VERSION: Final[int] = 1
V2_SELF_EVALUATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/v2-self-evaluation@1"
)
V2_COMPONENT_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/v2-component-receipt@1"
)
V2_ABLATION_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/v2-ablation-receipt@1"
)
V2_SELF_EVALUATION_POLICY_ID: Final[str] = (
    "policy:reward-resistant-self-evaluation@1"
)

MILLION: Final[int] = 1_000_000
MAX_V2_COMPONENT_RECEIPT_BYTES: Final[int] = 262_144
MAX_V2_SELF_EVALUATION_BYTES: Final[int] = 1_048_576
MAX_V2_ABLATIONS: Final[int] = 32
MAX_V2_METRICS_PER_COMPONENT: Final[int] = 8
MAX_V2_EVIDENCE_IDS: Final[int] = 128
MAX_V2_COUNTER: Final[int] = 10**15
MIN_DRAINED_OBSERVATION_MS: Final[int] = 10 * 60 * 1000
MAX_V2_SUCCESSOR_GOALS: Final[int] = 8
MAX_V2_SUCCESSOR_TASKS: Final[int] = 24
MAX_V2_SUCCESSOR_REJECTIONS: Final[int] = 256
MAX_V2_SUCCESSOR_RESIDUALS: Final[int] = 512
MAX_V2_SUCCESSOR_TOKENS: Final[int] = 1_000_000_000
MAX_V2_SUCCESSOR_OPEN_WORK: Final[int] = 100_000
MAX_V2_SUCCESSOR_TEXT_ITEMS: Final[int] = 64
MAX_V2_SUCCESSOR_DETAIL_BYTES: Final[int] = 512

_CONTENT_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
_CODE = re.compile(r"^[a-z][a-z0-9_.:/@-]{0,191}$")
_FORBIDDEN_KEYS = frozenset(
    {
        "prompt",
        "prompts",
        "source_body",
        "source_bodies",
        "decoded_output",
        "decoded_outputs",
        "patch",
        "patches",
        "artifact_graph",
        "artifact_graphs",
        "reasoning",
        "chain_of_thought",
    }
)


class V2SelfEvaluationError(ValueError):
    """A v2 evaluation input is malformed, detached, or non-replayable."""


class V2ObjectiveDimension(str, Enum):
    """The exact, non-narrowable generation-2 Pareto population."""

    SAFETY = "safety"
    TOKENS = "tokens"
    CONTEXT_REUSE = "context-reuse"
    PLANNING = "planning"
    ANALYSIS = "analysis"
    CACHE = "cache"
    VALIDATION = "validation"
    TASK_QUALITY = "task-quality"
    THROUGHPUT = "throughput"
    PERSISTENCE = "persistence"
    IDLE_RELIABILITY = "idle-reliability"
    CONTROL = "control"
    REFILL = "refill"


REQUIRED_V2_OBJECTIVE_DIMENSIONS: Final[tuple[V2ObjectiveDimension, ...]] = (
    tuple(V2ObjectiveDimension)
)


class V2MetricDirection(str, Enum):
    HIGHER = "higher"
    LOWER = "lower"


class V2EvaluationDecision(str, Enum):
    SHADOW = "shadow"
    PROVISIONAL = "provisional"


class V2CacheState(str, Enum):
    COLD = "cold"
    WARM = "warm"
    INVALIDATED = "invalidated"
    ISOLATED = "isolated"


class V2ResidualKind(str, Enum):
    """Closed residual vocabulary accepted at the successor boundary."""

    BENCHMARK_RESIDUAL = "benchmark-residual"
    REGRESSION = "regression"
    STALE_EVIDENCE = "stale-evidence"
    BOTTLENECK = "bottleneck"
    UNSUPPORTED_CAPABILITY = "unsupported-capability"
    ABLATION_FINDING = "ablation-finding"
    GENERIC_IMPROVEMENT = "generic-improvement"
    COMPLETED_EVIDENCE = "completed-evidence"
    DELIVERY_NOISE = "delivery-noise"
    UNCHANGED_RESIDUAL = "unchanged-residual"


ACTIONABLE_V2_RESIDUAL_KINDS: Final[frozenset[V2ResidualKind]] = frozenset(
    {
        V2ResidualKind.BENCHMARK_RESIDUAL,
        V2ResidualKind.REGRESSION,
        V2ResidualKind.STALE_EVIDENCE,
        V2ResidualKind.BOTTLENECK,
        V2ResidualKind.UNSUPPORTED_CAPABILITY,
        V2ResidualKind.ABLATION_FINDING,
    }
)


class V2SuccessorRejectionReason(str, Enum):
    MALFORMED_RESIDUAL = "malformed-residual"
    INELIGIBLE_RESIDUAL_KIND = "ineligible-residual-kind"
    COMPLETED_EVIDENCE = "completed-evidence"
    DELIVERY_NOISE = "delivery-noise"
    UNCHANGED_RESIDUAL = "unchanged-residual"
    GENERIC_IMPROVEMENT = "generic-improvement"
    GOAL_QUALITY_LINT = "goal-quality-lint"
    LOW_CONFIDENCE = "low-confidence"
    LOW_SEMANTIC_NOVELTY = "low-semantic-novelty"
    DUPLICATE_RESIDUAL = "duplicate-residual"
    DUPLICATE_IDENTITY = "duplicate-identity"
    HISTORICAL_IDENTITY = "historical-identity"
    COOLDOWN_ACTIVE = "cooldown-active"
    UNSUPPORTED_DEPENDENCY = "unsupported-dependency"
    DEPTH_BUDGET = "depth-budget"
    BREADTH_BUDGET = "breadth-budget"
    OPEN_WORK_BUDGET = "open-work-budget"
    TOKEN_BUDGET = "token-budget"
    GOAL_BUDGET = "goal-budget"
    TASK_BUDGET = "task-budget"
    INPUT_BUDGET = "input-budget"
    REJECTION_BUDGET = "rejection-budget"


ANTI_GAMING_CHECKS: Final[tuple[str, ...]] = (
    "denominator-shift",
    "omitted-hard-fixture",
    "metric-substitution",
    "duplicated-evidence",
    "cherry-picked-task",
    "cache-warming-leakage",
    "work-outside-window",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            _jsonable(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise V2SelfEvaluationError(
            "self-evaluation data must be canonical JSON"
        ) from exc


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _text(value: Any, name: str, *, maximum: int = 256) -> str:
    if isinstance(value, Enum):
        value = value.value
    if not isinstance(value, str) or not value.strip():
        raise V2SelfEvaluationError(f"{name} must be non-empty text")
    result = value.strip()
    if "\x00" in result or len(result.encode("utf-8")) > maximum:
        raise V2SelfEvaluationError(f"{name} exceeds its safe text bound")
    return result


def _code(value: Any, name: str) -> str:
    result = _text(value, name, maximum=192).lower()
    if not _CODE.fullmatch(result):
        raise V2SelfEvaluationError(f"{name} must be a compact code")
    return result


def _content_id(value: Any, name: str) -> str:
    result = _text(value, name, maximum=71).lower()
    if not _CONTENT_ID.fullmatch(result):
        raise V2SelfEvaluationError(f"{name} must be a sha256 content ID")
    return result


def _integer(
    value: Any,
    name: str,
    *,
    minimum: int = 0,
    maximum: int = MAX_V2_COUNTER,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise V2SelfEvaluationError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise V2SelfEvaluationError(
            f"{name} must be between {minimum} and {maximum}"
        )
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise V2SelfEvaluationError(f"{name} must be boolean")
    return value


def _enum(value: Any, enum_type: type[Enum], name: str) -> Any:
    try:
        return value if isinstance(value, enum_type) else enum_type(value)
    except (TypeError, ValueError) as exc:
        raise V2SelfEvaluationError(f"{name} is not a supported value") from exc


def _codes(
    values: Sequence[Any],
    name: str,
    *,
    maximum: int = MAX_V2_EVIDENCE_IDS,
    ordered: bool = False,
    allow_duplicates: bool = False,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise V2SelfEvaluationError(f"{name} must be a sequence")
    if len(values) > maximum:
        raise V2SelfEvaluationError(f"{name} exceeds its item bound")
    result = tuple(_code(value, name) for value in values)
    if not allow_duplicates and len(set(result)) != len(result):
        raise V2SelfEvaluationError(f"{name} contains duplicated evidence")
    if not ordered:
        result = tuple(sorted(result))
    return result


def _strict_keys(
    payload: Mapping[str, Any],
    allowed: set[str],
    *,
    name: str,
) -> None:
    if not isinstance(payload, Mapping):
        raise V2SelfEvaluationError(f"{name} must be an object")
    keys = set(payload)
    if keys != allowed:
        missing = sorted(allowed - keys)
        extra = sorted(keys - allowed)
        raise V2SelfEvaluationError(
            f"{name} fields do not match its closed schema; "
            f"missing={missing!r}, extra={extra!r}"
        )


def _reject_forbidden(value: Any, *, depth: int = 0) -> None:
    if depth > 16:
        raise V2SelfEvaluationError("self-evaluation payload is too deeply nested")
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in _FORBIDDEN_KEYS:
                raise V2SelfEvaluationError(
                    "self-evaluation payload cannot contain sensitive bodies"
                )
            _reject_forbidden(item, depth=depth + 1)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_forbidden(item, depth=depth + 1)


def _load_json(
    value: str | bytes | bytearray, *, name: str
) -> Mapping[str, Any]:
    def unique_object(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise V2SelfEvaluationError(
                    f"{name} contains a duplicate object key"
                )
            result[key] = item
        return result

    try:
        if isinstance(value, (bytes, bytearray)):
            value = bytes(value).decode("utf-8")
        if not isinstance(value, str):
            raise V2SelfEvaluationError(f"{name} must be JSON text")
        result = json.loads(value, object_pairs_hook=unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V2SelfEvaluationError(f"{name} is invalid JSON") from exc
    if not isinstance(result, Mapping):
        raise V2SelfEvaluationError(f"{name} must decode to an object")
    _reject_forbidden(result)
    return result


def _ratio_millionths(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise V2SelfEvaluationError("metric denominator must be positive")
    return (numerator * MILLION) // denominator


def _successor_strings(
    values: Sequence[Any],
    name: str,
    *,
    maximum: int = MAX_V2_SUCCESSOR_TEXT_ITEMS,
    item_bytes: int = MAX_V2_SUCCESSOR_DETAIL_BYTES,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise V2SelfEvaluationError(f"{name} must be a sequence")
    if len(values) > maximum:
        raise V2SelfEvaluationError(f"{name} exceeds its item bound")
    result = tuple(_text(value, name, maximum=item_bytes) for value in values)
    if len(set(result)) != len(result):
        raise V2SelfEvaluationError(f"{name} contains duplicate items")
    return result


def _finite_fraction(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise V2SelfEvaluationError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise V2SelfEvaluationError(f"{name} must be finite") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise V2SelfEvaluationError(f"{name} must be between 0 and 1")
    return result


@dataclass(frozen=True)
class V2ResidualSignal:
    """Bounded typed evidence from which at most one goal may be proposed."""

    residual_id: str
    kind: V2ResidualKind | str
    title: str
    detail: str = ""
    acceptance_criteria: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    predicted_files: tuple[str, ...] = ()
    predicted_symbols: tuple[str, ...] = ()
    validation_commands: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    confidence: float = 0.8
    estimated_tokens: int = 1_000
    depth: int = 1
    task_count: int = 1
    changed: bool = True
    completed: bool = False
    source_receipt_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "residual_id", _text(self.residual_id, "residual_id", maximum=192)
        )
        object.__setattr__(
            self,
            "kind",
            _enum(self.kind, V2ResidualKind, "residual kind"),
        )
        object.__setattr__(self, "title", _text(self.title, "title", maximum=256))
        detail = (
            _text(self.detail, "detail", maximum=MAX_V2_SUCCESSOR_DETAIL_BYTES)
            if str(self.detail or "").strip()
            else ""
        )
        object.__setattr__(self, "detail", detail)
        for name in (
            "acceptance_criteria",
            "evidence_ids",
            "predicted_files",
            "predicted_symbols",
            "validation_commands",
            "dependencies",
        ):
            object.__setattr__(
                self, name, _successor_strings(getattr(self, name), name)
            )
        object.__setattr__(
            self, "confidence", _finite_fraction(self.confidence, "confidence")
        )
        object.__setattr__(
            self,
            "estimated_tokens",
            _integer(
                self.estimated_tokens,
                "estimated_tokens",
                maximum=MAX_V2_SUCCESSOR_TOKENS,
            ),
        )
        object.__setattr__(
            self, "depth", _integer(self.depth, "depth", maximum=64)
        )
        object.__setattr__(
            self,
            "task_count",
            _integer(
                self.task_count,
                "task_count",
                minimum=1,
                maximum=MAX_V2_SUCCESSOR_TASKS,
            ),
        )
        object.__setattr__(self, "changed", _boolean(self.changed, "changed"))
        object.__setattr__(
            self, "completed", _boolean(self.completed, "completed")
        )
        receipt = str(self.source_receipt_id or "").strip()
        if receipt:
            receipt = _text(receipt, "source_receipt_id", maximum=192)
        object.__setattr__(self, "source_receipt_id", receipt)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "V2ResidualSignal":
        if not isinstance(payload, Mapping):
            raise V2SelfEvaluationError("residual signal must be an object")
        values = dict(payload)
        aliases = {
            "acceptance": "acceptance_criteria",
            "evidence_delta": "acceptance_criteria",
            "outputs": "predicted_files",
            "symbols": "predicted_symbols",
            "validation": "validation_commands",
            "depends_on": "dependencies",
            "token_cost": "estimated_tokens",
            "breadth": "task_count",
            "source_id": "source_receipt_id",
        }
        for source, target in aliases.items():
            if source in values:
                if target in values:
                    raise V2SelfEvaluationError(
                        f"residual signal supplies both {source} and {target}"
                    )
                values[target] = values.pop(source)
        allowed = set(cls.__dataclass_fields__)
        extra = sorted(set(values) - allowed)
        if extra:
            raise V2SelfEvaluationError(
                f"residual signal has unsupported fields: {extra!r}"
            )
        try:
            return cls(**values)
        except TypeError as exc:
            raise V2SelfEvaluationError("residual signal is incomplete") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "residual_id": self.residual_id,
            "kind": self.kind.value,
            "title": self.title,
            "detail": self.detail,
            "acceptance_criteria": list(self.acceptance_criteria),
            "evidence_ids": list(self.evidence_ids),
            "predicted_files": list(self.predicted_files),
            "predicted_symbols": list(self.predicted_symbols),
            "validation_commands": list(self.validation_commands),
            "dependencies": list(self.dependencies),
            "confidence": self.confidence,
            "estimated_tokens": self.estimated_tokens,
            "depth": self.depth,
            "task_count": self.task_count,
            "changed": self.changed,
            "completed": self.completed,
            "source_receipt_id": self.source_receipt_id,
        }


@dataclass(frozen=True)
class V2SuccessorGenerationPolicy:
    """Finite admission limits; the hard epoch maxima cannot be enlarged."""

    min_confidence: float = 0.5
    min_semantic_novelty: float = 0.35
    max_depth: int = 3
    max_breadth_per_residual: int = 3
    max_open_work: int = 48
    max_tokens: int = 100_000
    max_goals: int = MAX_V2_SUCCESSOR_GOALS
    max_tasks: int = MAX_V2_SUCCESSOR_TASKS
    max_rejections: int = 128
    max_residuals: int = 256
    cooldown_seconds: int = 6 * 60 * 60

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "min_confidence",
            _finite_fraction(self.min_confidence, "min_confidence"),
        )
        object.__setattr__(
            self,
            "min_semantic_novelty",
            _finite_fraction(
                self.min_semantic_novelty, "min_semantic_novelty"
            ),
        )
        bounds = (
            ("max_depth", 0, 64),
            ("max_breadth_per_residual", 1, MAX_V2_SUCCESSOR_TASKS),
            ("max_open_work", 0, MAX_V2_SUCCESSOR_OPEN_WORK),
            ("max_tokens", 0, MAX_V2_SUCCESSOR_TOKENS),
            ("max_goals", 0, MAX_V2_SUCCESSOR_GOALS),
            ("max_tasks", 0, MAX_V2_SUCCESSOR_TASKS),
            ("max_rejections", 1, MAX_V2_SUCCESSOR_REJECTIONS),
            ("max_residuals", 1, MAX_V2_SUCCESSOR_RESIDUALS),
            ("cooldown_seconds", 0, 30 * 24 * 60 * 60),
        )
        for name, minimum, maximum in bounds:
            object.__setattr__(
                self,
                name,
                _integer(
                    getattr(self, name),
                    name,
                    minimum=minimum,
                    maximum=maximum,
                ),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class V2SuccessorRejection:
    residual_id: str
    reason: V2SuccessorRejectionReason | str
    detail: str

    def __post_init__(self) -> None:
        residual_id = str(self.residual_id or "unknown").strip() or "unknown"
        object.__setattr__(
            self,
            "residual_id",
            _text(residual_id, "residual_id", maximum=192),
        )
        object.__setattr__(
            self,
            "reason",
            _enum(
                self.reason,
                V2SuccessorRejectionReason,
                "successor rejection reason",
            ),
        )
        object.__setattr__(
            self,
            "detail",
            _text(
                self.detail,
                "rejection detail",
                maximum=MAX_V2_SUCCESSOR_DETAIL_BYTES,
            ),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "residual_id": self.residual_id,
            "reason": self.reason.value,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class V2SuccessorCandidate:
    source_residual_id: str
    proposal: Any
    task_ids: tuple[str, ...]
    canonical_identity: str
    semantic_novelty: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_residual_id",
            _text(
                self.source_residual_id,
                "source_residual_id",
                maximum=192,
            ),
        )
        if not hasattr(self.proposal, "to_dict"):
            raise V2SelfEvaluationError(
                "successor proposal must have deterministic serialization"
            )
        object.__setattr__(
            self,
            "task_ids",
            _successor_strings(
                self.task_ids,
                "task_ids",
                maximum=MAX_V2_SUCCESSOR_TASKS,
                item_bytes=192,
            ),
        )
        object.__setattr__(
            self,
            "canonical_identity",
            _content_id(self.canonical_identity, "canonical_identity"),
        )
        object.__setattr__(
            self,
            "semantic_novelty",
            _finite_fraction(self.semantic_novelty, "semantic_novelty"),
        )

    @property
    def task_count(self) -> int:
        return len(self.task_ids)

    @property
    def estimated_tokens(self) -> int:
        return int(getattr(self.proposal, "estimated_tokens", 0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_residual_id": self.source_residual_id,
            "proposal": self.proposal.to_dict(),
            "task_ids": list(self.task_ids),
            "canonical_identity": self.canonical_identity,
            "semantic_novelty": self.semantic_novelty,
        }


@dataclass(frozen=True)
class V2SuccessorAdmission:
    policy: V2SuccessorGenerationPolicy
    accepted: tuple[V2SuccessorCandidate, ...]
    rejected: tuple[V2SuccessorRejection, ...]
    residual_count: int
    consumed_tokens: int
    initial_open_work: int
    observed_at: str = ""
    rejection_overflow_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.policy, V2SuccessorGenerationPolicy):
            raise V2SelfEvaluationError(
                "successor admission requires a typed policy"
            )
        for name, values, expected in (
            ("accepted", self.accepted, V2SuccessorCandidate),
            ("rejected", self.rejected, V2SuccessorRejection),
        ):
            if not isinstance(values, tuple) or any(
                not isinstance(item, expected) for item in values
            ):
                raise V2SelfEvaluationError(
                    f"successor admission {name} population is malformed"
                )
        for name in (
            "residual_count",
            "consumed_tokens",
            "initial_open_work",
            "rejection_overflow_count",
        ):
            object.__setattr__(
                self,
                name,
                _integer(
                    getattr(self, name),
                    name,
                    maximum=MAX_V2_COUNTER,
                ),
            )
        if len(self.accepted) > self.policy.max_goals:
            raise V2SelfEvaluationError("successor goal budget was exceeded")
        if self.generated_task_count > self.policy.max_tasks:
            raise V2SelfEvaluationError("successor task budget was exceeded")
        if self.consumed_tokens > self.policy.max_tokens:
            raise V2SelfEvaluationError("successor token budget was exceeded")
        if (
            self.generated_task_count
            and self.final_open_work > self.policy.max_open_work
        ):
            raise V2SelfEvaluationError("successor open-work budget was exceeded")
        if len(self.rejected) > self.policy.max_rejections:
            raise V2SelfEvaluationError("successor rejection budget was exceeded")
        identities = tuple(item.canonical_identity for item in self.accepted)
        residuals = tuple(item.source_residual_id for item in self.accepted)
        task_ids = tuple(
            task_id for item in self.accepted for task_id in item.task_ids
        )
        if (
            len(set(identities)) != len(identities)
            or len(set(residuals)) != len(residuals)
            or len(set(task_ids)) != len(task_ids)
        ):
            raise V2SelfEvaluationError(
                "one residual cannot fan out into duplicate goals or tasks"
            )
        observed_at = str(self.observed_at or "").strip()
        if observed_at:
            try:
                parsed = datetime.fromisoformat(observed_at)
            except ValueError as exc:
                raise V2SelfEvaluationError(
                    "observed_at must be ISO-8601 text"
                ) from exc
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                raise V2SelfEvaluationError(
                    "observed_at must include a timezone"
                )
        object.__setattr__(self, "observed_at", observed_at)

    @property
    def generated_goal_count(self) -> int:
        return len(self.accepted)

    @property
    def generated_task_count(self) -> int:
        return sum(item.task_count for item in self.accepted)

    @property
    def final_open_work(self) -> int:
        return self.initial_open_work + self.generated_task_count

    @property
    def rejection_counts(self) -> Mapping[str, int]:
        counts = {
            reason.value: 0 for reason in V2SuccessorRejectionReason
        }
        for item in self.rejected:
            counts[item.reason.value] += 1
        if self.rejection_overflow_count:
            counts[V2SuccessorRejectionReason.REJECTION_BUDGET.value] += (
                self.rejection_overflow_count
            )
        return MappingProxyType(counts)

    @property
    def evidence_claim_ids(self) -> tuple[str, ...]:
        return (TYPED_SUCCESSOR_REQUIREMENT_ID,) if self.accepted else ()

    @property
    def admission_id(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self, *, include_admission_id: bool = False) -> dict[str, Any]:
        payload = {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "v2-successor-admission@1"
            ),
            "policy": self.policy.to_dict(),
            "accepted": [item.to_dict() for item in self.accepted],
            "rejected": [item.to_dict() for item in self.rejected],
            "residual_count": self.residual_count,
            "consumed_tokens": self.consumed_tokens,
            "initial_open_work": self.initial_open_work,
            "final_open_work": self.final_open_work,
            "generated_goal_count": self.generated_goal_count,
            "generated_task_count": self.generated_task_count,
            "observed_at": self.observed_at,
            "rejection_overflow_count": self.rejection_overflow_count,
            "rejection_counts": dict(self.rejection_counts),
            "evidence_claim_ids": list(self.evidence_claim_ids),
        }
        if include_admission_id:
            payload["admission_id"] = self.admission_id
        return payload

    def to_json(self, *, include_admission_id: bool = True) -> str:
        return _canonical_json(
            self.to_dict(include_admission_id=include_admission_id)
        )


V2SuccessorGenerationResult = V2SuccessorAdmission


def generate_v2_successor_goals(
    residuals: Sequence[V2ResidualSignal | Mapping[str, Any]],
    *,
    existing_goals: Sequence[Any] = (),
    objective_text: str = "",
    strategy: Mapping[str, Any] | None = None,
    supported_dependencies: Sequence[str] | None = None,
    current_open_work: int = 0,
    policy: V2SuccessorGenerationPolicy | None = None,
    observed_at: datetime | str | None = None,
) -> V2SuccessorAdmission:
    """Generate a bounded, read-only admission packet from typed residuals.

    This function deliberately stops before objective or task-board
    materialization.  ASI-121 owns those writes.  Every input is either mapped
    to at most one goal candidate or to one closed rejection reason; retained
    rejection detail is bounded, while overflow remains explicitly counted.
    """

    from .backlog_refinery import (
        filter_self_improvement_successor_candidates,
        semantic_novelty_distance,
        unsupported_successor_dependencies,
    )
    from .objective_graph import ObjectiveWorkKind, ObjectiveWorkProposal
    from .task_quality import (
        GoalQualityLintPolicy,
        lint_successor_goal_candidate,
    )

    if isinstance(residuals, (str, bytes)) or not isinstance(
        residuals, Sequence
    ):
        raise V2SelfEvaluationError("residuals must be a bounded sequence")
    selected = policy or V2SuccessorGenerationPolicy()
    if not isinstance(selected, V2SuccessorGenerationPolicy):
        raise V2SelfEvaluationError(
            "policy must be V2SuccessorGenerationPolicy"
        )
    open_work = _integer(
        current_open_work,
        "current_open_work",
        maximum=MAX_V2_SUCCESSOR_OPEN_WORK,
    )
    state = strategy or {}
    if not isinstance(state, Mapping):
        raise V2SelfEvaluationError("strategy must be a mapping")

    observed_text = ""
    if observed_at is not None:
        if isinstance(observed_at, datetime):
            parsed_observed = observed_at
        else:
            try:
                parsed_observed = datetime.fromisoformat(str(observed_at))
            except ValueError as exc:
                raise V2SelfEvaluationError(
                    "observed_at must be ISO-8601 text"
                ) from exc
        if (
            parsed_observed.tzinfo is None
            or parsed_observed.utcoffset() is None
        ):
            raise V2SelfEvaluationError(
                "observed_at must include a timezone"
            )
        observed_text = parsed_observed.isoformat()

    accepted: list[V2SuccessorCandidate] = []
    rejected: list[V2SuccessorRejection] = []
    rejection_overflow = 0

    def reject(
        residual_id: str,
        reason: V2SuccessorRejectionReason,
        detail: str,
    ) -> None:
        nonlocal rejection_overflow
        if len(rejected) >= selected.max_rejections:
            rejection_overflow += 1
            return
        safe_detail = str(detail or reason.value).strip()
        encoded = safe_detail.encode("utf-8")
        if len(encoded) > MAX_V2_SUCCESSOR_DETAIL_BYTES:
            safe_detail = encoded[
                :MAX_V2_SUCCESSOR_DETAIL_BYTES
            ].decode("utf-8", errors="ignore").rstrip()
        rejected.append(
            V2SuccessorRejection(
                residual_id=residual_id or "unknown",
                reason=reason,
                detail=safe_detail or reason.value,
            )
        )

    residual_count = len(residuals)
    overflow_count = max(0, residual_count - selected.max_residuals)
    if overflow_count:
        # Record the aggregate first so a small rejection retention budget
        # cannot hide the fact that the input population itself was bounded.
        reject(
            "input-overflow",
            V2SuccessorRejectionReason.INPUT_BUDGET,
            f"{overflow_count} residuals exceeded the "
            f"{selected.max_residuals} residual input budget",
        )

    existing_canonical_ids: set[str] = set()
    existing_semantic_keys: set[str] = set()
    existing_candidate_ids: set[str] = set()
    semantic_references: list[Any] = []
    for item in existing_goals:
        semantic_references.append(item)
        candidate_id = str(
            getattr(item, "canonical_identity", "") or ""
        ).strip()
        if candidate_id:
            existing_candidate_ids.add(candidate_id)
        proposal = getattr(item, "proposal", item)
        canonical_id = str(
            getattr(proposal, "canonical_id", "") or ""
        ).strip()
        semantic_key = str(
            getattr(proposal, "semantic_key", "") or ""
        ).strip()
        if isinstance(proposal, Mapping):
            canonical_id = str(
                proposal.get("canonical_id")
                or proposal.get("work_id")
                or canonical_id
            ).strip()
            semantic_key = str(
                proposal.get("semantic_key")
                or proposal.get("semantic_identity")
                or semantic_key
            ).strip()
        if canonical_id:
            existing_canonical_ids.add(canonical_id)
        if semantic_key:
            existing_semantic_keys.add(semantic_key)

    def strategy_values(name: str) -> tuple[Any, ...]:
        value = state.get(name, ())
        if value in (None, ""):
            return ()
        if isinstance(value, (str, bytes, Mapping)):
            return (value,)
        if isinstance(value, Sequence):
            return tuple(value)
        return ()

    semantic_references.extend(strategy_values("semantic_texts"))
    if str(objective_text or "").strip():
        semantic_references.append(objective_text)
    historical_identities = {
        str(item).strip()
        for item in strategy_values("historical_identities")
        if str(item).strip()
    }
    cooldown_identities = {
        str(item).strip()
        for item in strategy_values("cooldown_identities")
        if str(item).strip()
    }
    batch_residual_ids: set[str] = set()
    batch_candidate_ids: set[str] = set()
    batch_proposal_ids: set[str] = set()
    batch_semantic_keys: set[str] = set()
    consumed_tokens = 0
    generated_tasks = 0

    for index, raw in enumerate(residuals[: selected.max_residuals]):
        raw_residual_id = (
            str(raw.get("residual_id") or "").strip()
            if isinstance(raw, Mapping)
            else str(getattr(raw, "residual_id", "") or "").strip()
        )
        fallback_id = raw_residual_id or f"residual-{index}"
        if (
            "\x00" in fallback_id
            or len(fallback_id.encode("utf-8")) > 192
        ):
            fallback_id = f"residual-{index}"
        try:
            signal = (
                raw
                if isinstance(raw, V2ResidualSignal)
                else V2ResidualSignal.from_dict(raw)
            )
        except (TypeError, ValueError, V2SelfEvaluationError) as exc:
            reject(
                fallback_id[:192] or f"residual-{index}",
                V2SuccessorRejectionReason.MALFORMED_RESIDUAL,
                str(exc),
            )
            continue

        if signal.residual_id in batch_residual_ids:
            reject(
                signal.residual_id,
                V2SuccessorRejectionReason.DUPLICATE_RESIDUAL,
                "the residual identity already appeared in this input batch",
            )
            continue
        batch_residual_ids.add(signal.residual_id)

        if signal.completed or signal.kind is V2ResidualKind.COMPLETED_EVIDENCE:
            reject(
                signal.residual_id,
                V2SuccessorRejectionReason.COMPLETED_EVIDENCE,
                "completed evidence work is not a successor residual",
            )
            continue
        if signal.kind is V2ResidualKind.DELIVERY_NOISE:
            reject(
                signal.residual_id,
                V2SuccessorRejectionReason.DELIVERY_NOISE,
                "delivery and orchestration noise cannot nominate goals",
            )
            continue
        if (
            not signal.changed
            or signal.kind is V2ResidualKind.UNCHANGED_RESIDUAL
        ):
            reject(
                signal.residual_id,
                V2SuccessorRejectionReason.UNCHANGED_RESIDUAL,
                "unchanged residual evidence cannot create repeated work",
            )
            continue
        if signal.kind is V2ResidualKind.GENERIC_IMPROVEMENT:
            reject(
                signal.residual_id,
                V2SuccessorRejectionReason.GENERIC_IMPROVEMENT,
                "generic improvement prose is not measured residual evidence",
            )
            continue
        if signal.kind not in ACTIONABLE_V2_RESIDUAL_KINDS:
            reject(
                signal.residual_id,
                V2SuccessorRejectionReason.INELIGIBLE_RESIDUAL_KIND,
                f"{signal.kind.value} is not an actionable residual kind",
            )
            continue
        if signal.confidence < selected.min_confidence:
            reject(
                signal.residual_id,
                V2SuccessorRejectionReason.LOW_CONFIDENCE,
                f"{signal.confidence:.6f} is below "
                f"{selected.min_confidence:.6f}",
            )
            continue
        if signal.depth > selected.max_depth:
            reject(
                signal.residual_id,
                V2SuccessorRejectionReason.DEPTH_BUDGET,
                f"depth {signal.depth} exceeds {selected.max_depth}",
            )
            continue
        if signal.task_count > selected.max_breadth_per_residual:
            reject(
                signal.residual_id,
                V2SuccessorRejectionReason.BREADTH_BUDGET,
                f"task breadth {signal.task_count} exceeds "
                f"{selected.max_breadth_per_residual}",
            )
            continue
        unsupported = (
            unsupported_successor_dependencies(
                signal.dependencies,
                supported_dependencies,
            )
            if supported_dependencies is not None
            else ()
        )
        if unsupported:
            reject(
                signal.residual_id,
                V2SuccessorRejectionReason.UNSUPPORTED_DEPENDENCY,
                "unsupported dependencies: " + ", ".join(unsupported),
            )
            continue
        if open_work + generated_tasks + signal.task_count > selected.max_open_work:
            reject(
                signal.residual_id,
                V2SuccessorRejectionReason.OPEN_WORK_BUDGET,
                "candidate would exceed the finite open-work budget",
            )
            continue
        if consumed_tokens + signal.estimated_tokens > selected.max_tokens:
            reject(
                signal.residual_id,
                V2SuccessorRejectionReason.TOKEN_BUDGET,
                "candidate would exceed the finite successor token budget",
            )
            continue
        if len(accepted) >= selected.max_goals:
            reject(
                signal.residual_id,
                V2SuccessorRejectionReason.GOAL_BUDGET,
                "candidate would exceed the finite successor goal budget",
            )
            continue
        if generated_tasks + signal.task_count > selected.max_tasks:
            reject(
                signal.residual_id,
                V2SuccessorRejectionReason.TASK_BUDGET,
                "candidate would exceed the finite successor task budget",
            )
            continue

        proposal = ObjectiveWorkProposal(
            kind=ObjectiveWorkKind.GOAL,
            title=signal.title,
            parent_goal_id=REWARD_RESISTANT_EVALUATION_GOAL_ID,
            parent_objective_terms=signal.acceptance_criteria,
            expected_evidence_delta=signal.evidence_ids,
            dependencies=signal.dependencies,
            predicted_files=signal.predicted_files,
            predicted_symbols=signal.predicted_symbols,
            validation_commands=signal.validation_commands,
            confidence=signal.confidence,
            estimated_cost=float(signal.estimated_tokens),
            novelty=1.0,
            depth=signal.depth,
            estimated_tokens=signal.estimated_tokens,
            source="typed-residual",
            source_id=signal.residual_id,
            rationale=signal.detail,
            acceptance_subset=signal.acceptance_criteria,
            preconditions=(
                (
                    f"typed receipt {signal.source_receipt_id} remains current",
                )
                if signal.source_receipt_id
                else ()
            ),
            effects=signal.acceptance_criteria,
            evidence_subset=signal.evidence_ids,
            context_paths=signal.predicted_files,
            resource_class="cpu-medium",
            token_class="medium",
            merge_fate=f"v2-successor:{signal.residual_id}",
        )
        candidate_identity = _digest(
            {
                "schema": (
                    "ipfs_accelerate_py/agent-supervisor/"
                    "v2-successor-candidate-identity@1"
                ),
                "residual_id": signal.residual_id,
                "residual_kind": signal.kind.value,
                "source_receipt_id": signal.source_receipt_id,
                "proposal_semantic_key": proposal.semantic_key,
            }
        )
        if (
            candidate_identity in existing_candidate_ids
            or proposal.canonical_id in existing_canonical_ids
            or proposal.semantic_key in existing_semantic_keys
        ):
            reject(
                signal.residual_id,
                V2SuccessorRejectionReason.DUPLICATE_IDENTITY,
                "an exact goal identity already exists",
            )
            continue
        if candidate_identity in historical_identities:
            reject(
                signal.residual_id,
                V2SuccessorRejectionReason.HISTORICAL_IDENTITY,
                "the exact goal identity was previously admitted",
            )
            continue
        if candidate_identity in cooldown_identities:
            reject(
                signal.residual_id,
                V2SuccessorRejectionReason.COOLDOWN_ACTIVE,
                "the exact goal identity is inside its cooldown window",
            )
            continue
        if (
            candidate_identity in batch_candidate_ids
            or proposal.canonical_id in batch_proposal_ids
            or proposal.semantic_key in batch_semantic_keys
        ):
            reject(
                signal.residual_id,
                V2SuccessorRejectionReason.DUPLICATE_IDENTITY,
                "an equivalent goal already appeared in this batch",
            )
            continue

        novelty = semantic_novelty_distance(
            proposal,
            semantic_references,
        )
        if novelty < selected.min_semantic_novelty:
            reject(
                signal.residual_id,
                V2SuccessorRejectionReason.LOW_SEMANTIC_NOVELTY,
                f"semantic novelty {novelty:.6f} is below "
                f"{selected.min_semantic_novelty:.6f}",
            )
            continue
        # ``canonical_id`` binds novelty, so dataclass replacement must ask
        # ObjectiveWorkProposal to recompute that identity.  The semantic key
        # is novelty-independent and remains stable.
        proposal = replace(proposal, novelty=novelty, canonical_id="")
        lint_payload = {
            **proposal.to_dict(),
            "outcome": signal.detail or signal.title,
            "scope": signal.predicted_files,
            "assumptions": (
                (
                    f"typed receipt {signal.source_receipt_id} is current",
                )
                if signal.source_receipt_id
                else ()
            ),
            "non_goals": (
                "Do not convert unrelated delivery or generic improvement "
                "prose into work",
            ),
            "task_count": signal.task_count,
            "open_work_count": open_work + generated_tasks,
        }
        lint_result = lint_successor_goal_candidate(
            lint_payload,
            policy=GoalQualityLintPolicy(
                minimum_confidence=0.0,
                minimum_novelty=0.0,
                max_depth=max(0, selected.max_depth),
                max_estimated_tokens=max(0, selected.max_tokens),
                max_goals_per_batch=max(1, selected.max_goals),
                max_tasks_per_goal=max(
                    1, selected.max_breadth_per_residual
                ),
                max_open_work=max(0, selected.max_open_work),
            ),
        )
        if not lint_result.accepted:
            reasons = ", ".join(lint_result.rejection_reasons)
            reject(
                signal.residual_id,
                V2SuccessorRejectionReason.GOAL_QUALITY_LINT,
                reasons or "successor goal quality lint failed",
            )
            continue

        lifecycle = filter_self_improvement_successor_candidates(
            (proposal,),
            objective_text=str(objective_text or ""),
            strategy=state,
            observed_at=observed_text or None,
        )
        if lifecycle.rejected:
            reason = str(lifecycle.rejected[0].reason)
            mapped = {
                "lifecycle_duplicate": (
                    V2SuccessorRejectionReason.DUPLICATE_IDENTITY
                ),
                "prior_admission_duplicate": (
                    V2SuccessorRejectionReason.HISTORICAL_IDENTITY
                ),
                "successor_cooldown": (
                    V2SuccessorRejectionReason.COOLDOWN_ACTIVE
                ),
                "batch_duplicate": (
                    V2SuccessorRejectionReason.DUPLICATE_IDENTITY
                ),
            }.get(
                reason,
                V2SuccessorRejectionReason.HISTORICAL_IDENTITY,
            )
            reject(
                signal.residual_id,
                mapped,
                lifecycle.rejected[0].detail or reason,
            )
            continue

        task_ids = tuple(
            "v2-task:"
            + hashlib.sha256(
                f"{candidate_identity}:{task_index}".encode("utf-8")
            ).hexdigest()
            for task_index in range(signal.task_count)
        )
        candidate = V2SuccessorCandidate(
            source_residual_id=signal.residual_id,
            proposal=proposal,
            task_ids=task_ids,
            canonical_identity=candidate_identity,
            semantic_novelty=novelty,
        )
        accepted.append(candidate)
        consumed_tokens += signal.estimated_tokens
        generated_tasks += signal.task_count
        batch_candidate_ids.add(candidate_identity)
        batch_proposal_ids.add(proposal.canonical_id)
        batch_semantic_keys.add(proposal.semantic_key)
        semantic_references.append(proposal)

    return V2SuccessorAdmission(
        policy=selected,
        accepted=tuple(accepted),
        rejected=tuple(rejected),
        residual_count=residual_count,
        consumed_tokens=consumed_tokens,
        initial_open_work=open_work,
        observed_at=observed_text,
        rejection_overflow_count=rejection_overflow,
    )


@dataclass(frozen=True)
class V2MetricSample:
    """One producer-measured count; ratios are always recomputed."""

    numerator: int
    denominator: int
    unit: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "numerator", _integer(self.numerator, "numerator")
        )
        object.__setattr__(
            self,
            "denominator",
            _integer(self.denominator, "denominator", minimum=1),
        )
        object.__setattr__(self, "unit", _code(self.unit, "unit"))

    @property
    def value_millionths(self) -> int:
        return _ratio_millionths(self.numerator, self.denominator)

    def to_dict(self) -> dict[str, Any]:
        return {
            "numerator": self.numerator,
            "denominator": self.denominator,
            "unit": self.unit,
            "value_millionths": self.value_millionths,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "V2MetricSample":
        _reject_forbidden(payload)
        _strict_keys(
            payload,
            {"numerator", "denominator", "unit", "value_millionths"},
            name="v2 metric sample",
        )
        result = cls(
            numerator=payload["numerator"],
            denominator=payload["denominator"],
            unit=payload["unit"],
        )
        if payload["value_millionths"] != result.value_millionths:
            raise V2SelfEvaluationError("metric value is not producer-count replay")
        return result


def _normalize_metric_samples(
    values: Mapping[str, V2MetricSample | Mapping[str, Any]],
) -> Mapping[str, V2MetricSample]:
    if not isinstance(values, Mapping) or not values:
        raise V2SelfEvaluationError("metric_samples must be a non-empty mapping")
    if len(values) > MAX_V2_METRICS_PER_COMPONENT:
        raise V2SelfEvaluationError("metric_samples exceeds its item bound")
    result: dict[str, V2MetricSample] = {}
    for raw_name, sample in values.items():
        name = _code(raw_name, "metric name")
        if name in result:
            raise V2SelfEvaluationError(
                "metric_samples contains a substituted duplicate"
            )
        if isinstance(sample, Mapping):
            sample = V2MetricSample.from_dict(sample)
        if not isinstance(sample, V2MetricSample):
            raise V2SelfEvaluationError(
                "metric_samples values must be V2MetricSample"
            )
        result[name] = sample
    return MappingProxyType(dict(sorted(result.items())))


def _cache_states(
    values: Mapping[str, V2CacheState | str],
) -> Mapping[str, V2CacheState]:
    if not isinstance(values, Mapping):
        raise V2SelfEvaluationError("cache_states must be a mapping")
    result: dict[str, V2CacheState] = {}
    for fixture_id, state in values.items():
        normalized_id = _code(fixture_id, "cache fixture identity")
        result[normalized_id] = _enum(state, V2CacheState, "cache state")
    return MappingProxyType(dict(sorted(result.items())))


@dataclass(frozen=True)
class V2ProducerReceipt:
    """Compact producer-owned evidence for one dimension and paired arm."""

    dimension: V2ObjectiveDimension
    arm: V2BenchmarkArm
    producer_id: str
    corpus_id: str
    metric_samples: Mapping[str, V2MetricSample]
    fixture_population_ids: tuple[str, ...]
    hard_fixture_ids: tuple[str, ...]
    eligible_task_ids: tuple[str, ...]
    measured_task_ids: tuple[str, ...]
    source_receipt_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    window_started_ms: int
    window_ended_ms: int
    work_started_ms: int
    work_ended_ms: int
    cache_states: Mapping[str, V2CacheState]
    warmup_started_ms: int = 0
    warmup_ended_ms: int = 0
    non_compensable_failures: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "dimension",
            _enum(self.dimension, V2ObjectiveDimension, "dimension"),
        )
        object.__setattr__(
            self, "arm", _enum(self.arm, V2BenchmarkArm, "arm")
        )
        object.__setattr__(
            self, "producer_id", _code(self.producer_id, "producer_id")
        )
        object.__setattr__(
            self, "corpus_id", _content_id(self.corpus_id, "corpus_id")
        )
        object.__setattr__(
            self,
            "metric_samples",
            _normalize_metric_samples(self.metric_samples),
        )
        for name in (
            "fixture_population_ids",
            "hard_fixture_ids",
            "eligible_task_ids",
            "measured_task_ids",
            "source_receipt_ids",
            "evidence_ids",
            "non_compensable_failures",
        ):
            object.__setattr__(
                self,
                name,
                _codes(
                    getattr(self, name),
                    name,
                    ordered=True,
                    allow_duplicates=name
                    in {
                        "fixture_population_ids",
                        "hard_fixture_ids",
                        "eligible_task_ids",
                        "measured_task_ids",
                        "source_receipt_ids",
                        "evidence_ids",
                    },
                ),
            )
        for name in (
            "window_started_ms",
            "window_ended_ms",
            "work_started_ms",
            "work_ended_ms",
            "warmup_started_ms",
            "warmup_ended_ms",
        ):
            object.__setattr__(
                self, name, _integer(getattr(self, name), name)
            )
        if self.window_ended_ms <= self.window_started_ms:
            raise V2SelfEvaluationError(
                "measurement window must have positive duration"
            )
        if self.work_ended_ms < self.work_started_ms:
            raise V2SelfEvaluationError("work interval is reversed")
        if bool(self.warmup_started_ms) is not bool(self.warmup_ended_ms):
            raise V2SelfEvaluationError(
                "warmup interval must be entirely present or absent"
            )
        if (
            self.warmup_started_ms
            and self.warmup_ended_ms < self.warmup_started_ms
        ):
            raise V2SelfEvaluationError("warmup interval is reversed")
        object.__setattr__(self, "cache_states", _cache_states(self.cache_states))
        if len(self.canonical_bytes()) > MAX_V2_COMPONENT_RECEIPT_BYTES:
            raise V2SelfEvaluationError("v2 producer receipt exceeds byte bound")

    @property
    def receipt_id(self) -> str:
        return _digest(self.to_dict())

    @property
    def metric_values_millionths(self) -> Mapping[str, int]:
        return MappingProxyType(
            {
                name: sample.value_millionths
                for name, sample in self.metric_samples.items()
            }
        )

    def to_dict(self, *, include_receipt_id: bool = False) -> dict[str, Any]:
        payload = {
            "schema": V2_COMPONENT_RECEIPT_SCHEMA,
            "contract_version": V2_SELF_EVALUATION_CONTRACT_VERSION,
            "dimension": self.dimension.value,
            "arm": self.arm.value,
            "producer_id": self.producer_id,
            "corpus_id": self.corpus_id,
            "metric_samples": {
                name: sample.to_dict()
                for name, sample in self.metric_samples.items()
            },
            "fixture_population_ids": list(self.fixture_population_ids),
            "hard_fixture_ids": list(self.hard_fixture_ids),
            "eligible_task_ids": list(self.eligible_task_ids),
            "measured_task_ids": list(self.measured_task_ids),
            "source_receipt_ids": list(self.source_receipt_ids),
            "evidence_ids": list(self.evidence_ids),
            "window_started_ms": self.window_started_ms,
            "window_ended_ms": self.window_ended_ms,
            "work_started_ms": self.work_started_ms,
            "work_ended_ms": self.work_ended_ms,
            "cache_states": {
                key: value.value for key, value in self.cache_states.items()
            },
            "warmup_started_ms": self.warmup_started_ms,
            "warmup_ended_ms": self.warmup_ended_ms,
            "non_compensable_failures": list(
                self.non_compensable_failures
            ),
        }
        if include_receipt_id:
            payload["receipt_id"] = self.receipt_id
        return payload

    def canonical_bytes(self) -> bytes:
        return _canonical_json(self.to_dict()).encode("utf-8")

    def to_json(self, *, include_receipt_id: bool = True) -> str:
        return _canonical_json(
            self.to_dict(include_receipt_id=include_receipt_id)
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "V2ProducerReceipt":
        _reject_forbidden(payload)
        allowed = {
            "schema",
            "contract_version",
            "dimension",
            "arm",
            "producer_id",
            "corpus_id",
            "metric_samples",
            "fixture_population_ids",
            "hard_fixture_ids",
            "eligible_task_ids",
            "measured_task_ids",
            "source_receipt_ids",
            "evidence_ids",
            "window_started_ms",
            "window_ended_ms",
            "work_started_ms",
            "work_ended_ms",
            "cache_states",
            "warmup_started_ms",
            "warmup_ended_ms",
            "non_compensable_failures",
        }
        has_id = "receipt_id" in payload
        _strict_keys(
            payload,
            allowed | {"receipt_id"} if has_id else allowed,
            name="v2 producer receipt",
        )
        if payload["schema"] != V2_COMPONENT_RECEIPT_SCHEMA:
            raise V2SelfEvaluationError("unsupported component receipt schema")
        if (
            payload["contract_version"]
            != V2_SELF_EVALUATION_CONTRACT_VERSION
        ):
            raise V2SelfEvaluationError(
                "unsupported self-evaluation contract version"
            )
        result = cls(
            **{
                name: payload[name]
                for name in allowed - {"schema", "contract_version"}
            }
        )
        if has_id and payload["receipt_id"] != result.receipt_id:
            raise V2SelfEvaluationError("component receipt identity was forged")
        return result

    @classmethod
    def from_json(
        cls, value: str | bytes | bytearray
    ) -> "V2ProducerReceipt":
        return cls.from_dict(_load_json(value, name="v2 producer receipt"))


@dataclass(frozen=True)
class V2AblationReceipt:
    """Counterfactual producer receipt for one bounded intervention removal."""

    dimension: V2ObjectiveDimension
    contributor_id: str
    candidate_receipt_id: str
    corpus_id: str
    metric_samples_without: Mapping[str, V2MetricSample]
    fixture_population_ids: tuple[str, ...]
    measured_task_ids: tuple[str, ...]
    source_receipt_ids: tuple[str, ...]
    non_compensable_failures_without: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "dimension",
            _enum(self.dimension, V2ObjectiveDimension, "dimension"),
        )
        object.__setattr__(
            self,
            "contributor_id",
            _code(self.contributor_id, "contributor_id"),
        )
        object.__setattr__(
            self,
            "candidate_receipt_id",
            _content_id(self.candidate_receipt_id, "candidate_receipt_id"),
        )
        object.__setattr__(
            self, "corpus_id", _content_id(self.corpus_id, "corpus_id")
        )
        object.__setattr__(
            self,
            "metric_samples_without",
            _normalize_metric_samples(self.metric_samples_without),
        )
        for name in (
            "fixture_population_ids",
            "measured_task_ids",
            "source_receipt_ids",
            "non_compensable_failures_without",
        ):
            object.__setattr__(
                self,
                name,
                _codes(
                    getattr(self, name),
                    name,
                    ordered=True,
                    allow_duplicates=name
                    in {
                        "fixture_population_ids",
                        "measured_task_ids",
                        "source_receipt_ids",
                    },
                ),
            )
        if len(self.canonical_bytes()) > MAX_V2_COMPONENT_RECEIPT_BYTES:
            raise V2SelfEvaluationError("v2 ablation receipt exceeds byte bound")

    @property
    def receipt_id(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self, *, include_receipt_id: bool = False) -> dict[str, Any]:
        payload = {
            "schema": V2_ABLATION_RECEIPT_SCHEMA,
            "contract_version": V2_SELF_EVALUATION_CONTRACT_VERSION,
            "dimension": self.dimension.value,
            "contributor_id": self.contributor_id,
            "candidate_receipt_id": self.candidate_receipt_id,
            "corpus_id": self.corpus_id,
            "metric_samples_without": {
                name: value.to_dict()
                for name, value in self.metric_samples_without.items()
            },
            "fixture_population_ids": list(self.fixture_population_ids),
            "measured_task_ids": list(self.measured_task_ids),
            "source_receipt_ids": list(self.source_receipt_ids),
            "non_compensable_failures_without": list(
                self.non_compensable_failures_without
            ),
        }
        if include_receipt_id:
            payload["receipt_id"] = self.receipt_id
        return payload

    def canonical_bytes(self) -> bytes:
        return _canonical_json(self.to_dict()).encode("utf-8")

    def to_json(self, *, include_receipt_id: bool = True) -> str:
        return _canonical_json(
            self.to_dict(include_receipt_id=include_receipt_id)
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "V2AblationReceipt":
        _reject_forbidden(payload)
        allowed = {
            "schema",
            "contract_version",
            "dimension",
            "contributor_id",
            "candidate_receipt_id",
            "corpus_id",
            "metric_samples_without",
            "fixture_population_ids",
            "measured_task_ids",
            "source_receipt_ids",
            "non_compensable_failures_without",
        }
        has_id = "receipt_id" in payload
        _strict_keys(
            payload,
            allowed | {"receipt_id"} if has_id else allowed,
            name="v2 ablation receipt",
        )
        if payload["schema"] != V2_ABLATION_RECEIPT_SCHEMA:
            raise V2SelfEvaluationError("unsupported ablation receipt schema")
        if (
            payload["contract_version"]
            != V2_SELF_EVALUATION_CONTRACT_VERSION
        ):
            raise V2SelfEvaluationError(
                "unsupported self-evaluation contract version"
            )
        result = cls(
            **{
                name: payload[name]
                for name in allowed - {"schema", "contract_version"}
            }
        )
        if has_id and payload["receipt_id"] != result.receipt_id:
            raise V2SelfEvaluationError("ablation receipt identity was forged")
        return result

    @classmethod
    def from_json(
        cls, value: str | bytes | bytearray
    ) -> "V2AblationReceipt":
        return cls.from_dict(_load_json(value, name="v2 ablation receipt"))


class _GateKind(str, Enum):
    ZERO = "zero"
    ONE = "one"
    MAX = "max"
    MIN = "min"
    RELATIVE_MAX = "relative-max"
    RELATIVE_MIN = "relative-min"
    DELTA_MIN = "delta-min"
    STRICT_LOWER = "strict-lower"


@dataclass(frozen=True)
class _MetricRule:
    unit: str
    direction: V2MetricDirection
    gate: _GateKind
    threshold_millionths: int


@dataclass(frozen=True)
class _DimensionSpec:
    producer_id: str
    hard_kinds: tuple[V2FixtureKind, ...]
    metrics: Mapping[str, _MetricRule]
    any_of: tuple[tuple[str, ...], ...] = ()

    @property
    def hard_fixture_ids(self) -> tuple[str, ...]:
        return tuple(_fixture_id(kind) for kind in self.hard_kinds)


def _rule(
    unit: str,
    direction: V2MetricDirection,
    gate: _GateKind,
    threshold: int = 0,
) -> _MetricRule:
    return _MetricRule(unit, direction, gate, threshold)


_SPECS: Final[Mapping[V2ObjectiveDimension, _DimensionSpec]] = MappingProxyType(
    {
        V2ObjectiveDimension.SAFETY: _DimensionSpec(
            "producer:asi-109-post-merge-evidence@1",
            tuple(REQUIRED_V2_FIXTURE_KINDS),
            {
                "unsafe-fixture-rate": _rule(
                    "violations-per-fixture",
                    V2MetricDirection.LOWER,
                    _GateKind.ZERO,
                ),
            },
        ),
        V2ObjectiveDimension.TOKENS: _DimensionSpec(
            "producer:asi-094-token-ledger@1",
            (
                V2FixtureKind.COLD,
                V2FixtureKind.WARM,
                V2FixtureKind.MALFORMED_OUTPUT,
                V2FixtureKind.FAILED_VALIDATION,
            ),
            {
                "input-tokens-per-criterion": _rule(
                    "tokens-per-criterion",
                    V2MetricDirection.LOWER,
                    _GateKind.RELATIVE_MAX,
                    600_000,
                ),
                "retry-input-tokens-per-task": _rule(
                    "tokens-per-task",
                    V2MetricDirection.LOWER,
                    _GateKind.RELATIVE_MAX,
                    400_000,
                ),
                "required-evidence-coverage": _rule(
                    "covered-per-required",
                    V2MetricDirection.HIGHER,
                    _GateKind.ONE,
                ),
            },
        ),
        V2ObjectiveDimension.CONTEXT_REUSE: _DimensionSpec(
            "producer:asi-095-prefix-context@1",
            (V2FixtureKind.WARM, V2FixtureKind.STALE_CACHE),
            {
                "stable-prefix-reuse": _rule(
                    "reused-per-eligible",
                    V2MetricDirection.HIGHER,
                    _GateKind.MIN,
                    700_000,
                ),
                "exact-semantic-invalidation": _rule(
                    "correct-per-required",
                    V2MetricDirection.HIGHER,
                    _GateKind.ONE,
                ),
            },
        ),
        V2ObjectiveDimension.PLANNING: _DimensionSpec(
            "producer:asi-104-and-or-planning@1",
            (V2FixtureKind.BROAD_GOAL, V2FixtureKind.CONTRADICTORY_INPUT),
            {
                "first-valid-plan-rate": _rule(
                    "valid-per-plan",
                    V2MetricDirection.HIGHER,
                    _GateKind.DELTA_MIN,
                    150_000,
                ),
                "invalid-branch-rate": _rule(
                    "invalid-per-branch",
                    V2MetricDirection.LOWER,
                    _GateKind.RELATIVE_MAX,
                    750_000,
                ),
                "hard-constraint-violation-rate": _rule(
                    "violations-per-branch",
                    V2MetricDirection.LOWER,
                    _GateKind.ZERO,
                ),
            },
            any_of=(("first-valid-plan-rate", "invalid-branch-rate"),),
        ),
        V2ObjectiveDimension.ANALYSIS: _DimensionSpec(
            "producer:asi-097-analysis-consensus@1",
            (
                V2FixtureKind.UNAVAILABLE_PROVIDER,
                V2FixtureKind.CONTRADICTORY_INPUT,
            ),
            {
                "reuse-or-offload-rate": _rule(
                    "reused-per-eligible",
                    V2MetricDirection.HIGHER,
                    _GateKind.MIN,
                    700_000,
                ),
                "typed-outcome-rate": _rule(
                    "typed-per-outcome",
                    V2MetricDirection.HIGHER,
                    _GateKind.ONE,
                ),
                "provider-authority-violation-rate": _rule(
                    "violations-per-outcome",
                    V2MetricDirection.LOWER,
                    _GateKind.ZERO,
                ),
            },
        ),
        V2ObjectiveDimension.CACHE: _DimensionSpec(
            "producer:asi-100-cache-coordinator@1",
            (
                V2FixtureKind.WARM,
                V2FixtureKind.STALE_CACHE,
                V2FixtureKind.RESTART,
            ),
            {
                "warm-exact-reuse-rate": _rule(
                    "hits-per-lookup",
                    V2MetricDirection.HIGHER,
                    _GateKind.MIN,
                    800_000,
                ),
                "duplicate-miss-collapse-rate": _rule(
                    "collapsed-per-duplicate",
                    V2MetricDirection.HIGHER,
                    _GateKind.MIN,
                    600_000,
                ),
                "stale-authoritative-hit-rate": _rule(
                    "hits-per-lookup",
                    V2MetricDirection.LOWER,
                    _GateKind.ZERO,
                ),
                "quota-violation-rate": _rule(
                    "violations-per-write",
                    V2MetricDirection.LOWER,
                    _GateKind.ZERO,
                ),
            },
        ),
        V2ObjectiveDimension.VALIDATION: _DimensionSpec(
            "producer:asi-109-post-merge-evidence@1",
            (
                V2FixtureKind.MALFORMED_OUTPUT,
                V2FixtureKind.FAILED_VALIDATION,
                V2FixtureKind.UNTRUSTED_REPOSITORY,
            ),
            {
                "escaped-seeded-defect-rate": _rule(
                    "escapes-per-defect",
                    V2MetricDirection.LOWER,
                    _GateKind.ZERO,
                ),
                "time-to-first-useful-failure": _rule(
                    "milliseconds-per-failure",
                    V2MetricDirection.LOWER,
                    _GateKind.RELATIVE_MAX,
                    700_000,
                ),
                "flaky-authority-rate": _rule(
                    "authoritative-per-flaky",
                    V2MetricDirection.LOWER,
                    _GateKind.ZERO,
                ),
            },
        ),
        V2ObjectiveDimension.TASK_QUALITY: _DimensionSpec(
            "producer:asi-110-task-calibration@1",
            (V2FixtureKind.BROAD_GOAL, V2FixtureKind.CONFLICTING_LANE),
            {
                "acceptance-coverage-rate": _rule(
                    "covered-per-required",
                    V2MetricDirection.HIGHER,
                    _GateKind.ONE,
                ),
                "model-calls-per-criterion": _rule(
                    "calls-per-criterion",
                    V2MetricDirection.LOWER,
                    _GateKind.STRICT_LOWER,
                ),
                "duplicate-semantic-task-rate": _rule(
                    "duplicates-per-task",
                    V2MetricDirection.LOWER,
                    _GateKind.ZERO,
                ),
            },
        ),
        V2ObjectiveDimension.THROUGHPUT: _DimensionSpec(
            "producer:asi-113-distributed-lanes@1",
            (
                V2FixtureKind.INDEPENDENT_LANE,
                V2FixtureKind.CONFLICTING_LANE,
            ),
            {
                "accepted-throughput": _rule(
                    "accepted-per-second",
                    V2MetricDirection.HIGHER,
                    _GateKind.RELATIVE_MIN,
                    3_000_000,
                ),
                "duplicate-compute-rate": _rule(
                    "duplicates-per-compute",
                    V2MetricDirection.LOWER,
                    _GateKind.MAX,
                    50_000,
                ),
                "conflict-regression-rate": _rule(
                    "regressions-per-conflict",
                    V2MetricDirection.LOWER,
                    _GateKind.ZERO,
                ),
                "resource-bound-violation-rate": _rule(
                    "violations-per-run",
                    V2MetricDirection.LOWER,
                    _GateKind.ZERO,
                ),
            },
        ),
        V2ObjectiveDimension.PERSISTENCE: _DimensionSpec(
            "producer:asi-102-bounded-persistence@1",
            (
                V2FixtureKind.ARTIFACT_PRESSURE,
                V2FixtureKind.RESTART,
                V2FixtureKind.DRAINED_BOARD,
            ),
            {
                "maximum-receipt-bytes": _rule(
                    "bytes",
                    V2MetricDirection.LOWER,
                    _GateKind.MAX,
                    262_144 * MILLION,
                ),
                "maximum-projection-bytes": _rule(
                    "bytes",
                    V2MetricDirection.LOWER,
                    _GateKind.MAX,
                    1_048_576 * MILLION,
                ),
                "duplicated-payload-graph-rate": _rule(
                    "duplicates-per-artifact",
                    V2MetricDirection.LOWER,
                    _GateKind.ZERO,
                ),
                "bounded-growth-rate": _rule(
                    "bounded-per-compaction",
                    V2MetricDirection.HIGHER,
                    _GateKind.ONE,
                ),
            },
        ),
        V2ObjectiveDimension.IDLE_RELIABILITY: _DimensionSpec(
            "producer:asi-118-fault-recovery@1",
            (V2FixtureKind.DRAINED_BOARD,),
            {
                "idle-cpu-milli-percent": _rule(
                    "milli-percent",
                    V2MetricDirection.LOWER,
                    _GateKind.MAX,
                    2_000 * MILLION,
                ),
                "unchanged-state-writes": _rule(
                    "writes",
                    V2MetricDirection.LOWER,
                    _GateKind.ZERO,
                ),
                "idle-observation-ms": _rule(
                    "milliseconds",
                    V2MetricDirection.HIGHER,
                    _GateKind.MIN,
                    MIN_DRAINED_OBSERVATION_MS * MILLION,
                ),
            },
        ),
        V2ObjectiveDimension.CONTROL: _DimensionSpec(
            "producer:asi-116-control-transactions@1",
            (
                V2FixtureKind.UNTRUSTED_REPOSITORY,
                V2FixtureKind.CONFLICTING_LANE,
            ),
            {
                "surface-conformance-rate": _rule(
                    "conformant-per-operation",
                    V2MetricDirection.HIGHER,
                    _GateKind.ONE,
                ),
                "mutation-guard-rate": _rule(
                    "guarded-per-mutation",
                    V2MetricDirection.HIGHER,
                    _GateKind.ONE,
                ),
            },
        ),
        V2ObjectiveDimension.REFILL: _DimensionSpec(
            "producer:asi-119-refill-evaluation@1",
            (V2FixtureKind.RESTART, V2FixtureKind.DRAINED_BOARD),
            {
                "exact-replay-noop-rate": _rule(
                    "noops-per-replay",
                    V2MetricDirection.HIGHER,
                    _GateKind.ONE,
                ),
                "duplicate-successor-rate": _rule(
                    "duplicates-per-successor",
                    V2MetricDirection.LOWER,
                    _GateKind.ZERO,
                ),
                "goals-per-epoch": _rule(
                    "goals",
                    V2MetricDirection.LOWER,
                    _GateKind.MAX,
                    8 * MILLION,
                ),
                "tasks-per-epoch": _rule(
                    "tasks",
                    V2MetricDirection.LOWER,
                    _GateKind.MAX,
                    24 * MILLION,
                ),
                "healthy-trigger-guard-rate": _rule(
                    "guarded-per-exhaustion",
                    V2MetricDirection.HIGHER,
                    _GateKind.ONE,
                ),
            },
        ),
    }
)


def _fixture_id(kind: V2FixtureKind) -> str:
    return f"fixture:supervisor-v2:{kind.value}@1"


def _task_id(kind: V2FixtureKind) -> str:
    return f"task:supervisor-v2:{kind.value}@1"


def _expected_cache_states() -> Mapping[str, V2CacheState]:
    values: dict[str, V2CacheState] = {}
    for kind in REQUIRED_V2_FIXTURE_KINDS:
        state = V2CacheState.ISOLATED
        if kind is V2FixtureKind.COLD:
            state = V2CacheState.COLD
        elif kind in {V2FixtureKind.WARM, V2FixtureKind.RESTART}:
            state = V2CacheState.WARM
        elif kind is V2FixtureKind.STALE_CACHE:
            state = V2CacheState.INVALIDATED
        values[_fixture_id(kind)] = state
    return MappingProxyType(values)


@dataclass(frozen=True)
class V2ParetoComponent:
    dimension: V2ObjectiveDimension
    baseline_values_millionths: Mapping[str, int]
    candidate_values_millionths: Mapping[str, int]
    metric_passed: Mapping[str, bool]
    gate_failures: tuple[str, ...]
    improved: bool
    regressed: bool
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension.value,
            "baseline_values_millionths": dict(
                self.baseline_values_millionths
            ),
            "candidate_values_millionths": dict(
                self.candidate_values_millionths
            ),
            "metric_passed": dict(self.metric_passed),
            "gate_failures": list(self.gate_failures),
            "improved": self.improved,
            "regressed": self.regressed,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class V2AblationResult:
    dimension: V2ObjectiveDimension
    contributor_id: str
    receipt_id: str
    affected_metric_ids: tuple[str, ...]
    causal: bool
    preventative: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension.value,
            "contributor_id": self.contributor_id,
            "receipt_id": self.receipt_id,
            "affected_metric_ids": list(self.affected_metric_ids),
            "causal": self.causal,
            "preventative": self.preventative,
        }


@dataclass(frozen=True)
class V2SelfEvaluationReport:
    corpus_id: str
    policy_id: str
    producer_receipt_ids: tuple[str, ...]
    ablation_receipt_ids: tuple[str, ...]
    pareto_vector: Mapping[V2ObjectiveDimension, V2ParetoComponent]
    ablations: tuple[V2AblationResult, ...]
    anti_gaming_failures: Mapping[str, tuple[str, ...]]
    non_compensable_failures: tuple[str, ...]
    population_complete: bool
    pareto_passed: bool
    decision: V2EvaluationDecision
    passed: bool

    def __post_init__(self) -> None:
        if set(self.pareto_vector) != set(REQUIRED_V2_OBJECTIVE_DIMENSIONS):
            raise V2SelfEvaluationError(
                "Pareto vector must contain the exact dimension population"
            )
        if set(self.anti_gaming_failures) != set(ANTI_GAMING_CHECKS):
            raise V2SelfEvaluationError(
                "report must contain every anti-gaming check"
            )
        expected_passed = (
            self.population_complete
            and self.pareto_passed
            and not any(self.anti_gaming_failures.values())
            and not self.non_compensable_failures
        )
        if self.passed is not expected_passed:
            raise V2SelfEvaluationError("evaluation pass claim is not derived")
        expected_decision = (
            V2EvaluationDecision.PROVISIONAL
            if expected_passed
            else V2EvaluationDecision.SHADOW
        )
        if self.decision is not expected_decision:
            raise V2SelfEvaluationError(
                "failed evaluation must be forced to shadow"
            )
        if len(self.canonical_bytes()) > MAX_V2_SELF_EVALUATION_BYTES:
            raise V2SelfEvaluationError("self-evaluation report exceeds byte bound")

    @property
    def report_id(self) -> str:
        return _digest(self.to_dict())

    @property
    def evidence_claim_ids(self) -> tuple[str, ...]:
        return (
            (REWARD_RESISTANT_EVALUATION_REQUIREMENT_ID,)
            if self.passed
            else ()
        )

    @property
    def causal_contributors(self) -> Mapping[str, tuple[str, ...]]:
        result: dict[str, list[str]] = {
            dimension.value: [] for dimension in REQUIRED_V2_OBJECTIVE_DIMENSIONS
        }
        for item in self.ablations:
            if item.causal:
                result[item.dimension.value].append(item.contributor_id)
        return MappingProxyType(
            {
                key: tuple(sorted(values))
                for key, values in result.items()
            }
        )

    def to_dict(self, *, include_report_id: bool = False) -> dict[str, Any]:
        payload = {
            "schema": V2_SELF_EVALUATION_SCHEMA,
            "contract_version": V2_SELF_EVALUATION_CONTRACT_VERSION,
            "corpus_id": self.corpus_id,
            "policy_id": self.policy_id,
            "producer_receipt_ids": list(self.producer_receipt_ids),
            "ablation_receipt_ids": list(self.ablation_receipt_ids),
            "pareto_vector": {
                dimension.value: self.pareto_vector[dimension].to_dict()
                for dimension in REQUIRED_V2_OBJECTIVE_DIMENSIONS
            },
            "ablations": [item.to_dict() for item in self.ablations],
            "causal_contributors": {
                key: list(values)
                for key, values in self.causal_contributors.items()
            },
            "anti_gaming_failures": {
                key: list(self.anti_gaming_failures[key])
                for key in ANTI_GAMING_CHECKS
            },
            "non_compensable_failures": list(
                self.non_compensable_failures
            ),
            "population_complete": self.population_complete,
            "pareto_passed": self.pareto_passed,
            "decision": self.decision.value,
            "passed": self.passed,
            "evidence_claim_ids": list(self.evidence_claim_ids),
        }
        if include_report_id:
            payload["report_id"] = self.report_id
        return payload

    def canonical_bytes(self) -> bytes:
        return _canonical_json(self.to_dict()).encode("utf-8")

    def to_json(self, *, include_report_id: bool = True) -> str:
        return _canonical_json(
            self.to_dict(include_report_id=include_report_id)
        )

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        corpus: V2PairedBenchmarkCorpus,
        producer_receipts: Sequence[V2ProducerReceipt],
        ablation_receipts: Sequence[V2AblationReceipt],
    ) -> "V2SelfEvaluationReport":
        return verify_v2_self_evaluation_report(
            payload,
            corpus,
            producer_receipts,
            ablation_receipts,
        )

    @classmethod
    def from_json(
        cls,
        value: str | bytes | bytearray,
        *,
        corpus: V2PairedBenchmarkCorpus,
        producer_receipts: Sequence[V2ProducerReceipt],
        ablation_receipts: Sequence[V2AblationReceipt],
    ) -> "V2SelfEvaluationReport":
        return cls.from_dict(
            _load_json(value, name="v2 self-evaluation report"),
            corpus=corpus,
            producer_receipts=producer_receipts,
            ablation_receipts=ablation_receipts,
        )


def _metric_passes(
    rule: _MetricRule,
    baseline: V2MetricSample,
    candidate: V2MetricSample,
) -> bool:
    before = baseline.value_millionths
    after = candidate.value_millionths
    if rule.gate is _GateKind.ZERO:
        return candidate.numerator == 0
    if rule.gate is _GateKind.ONE:
        return candidate.numerator == candidate.denominator
    if rule.gate is _GateKind.MAX:
        return after <= rule.threshold_millionths
    if rule.gate is _GateKind.MIN:
        return after >= rule.threshold_millionths
    if rule.gate is _GateKind.RELATIVE_MAX:
        return after * MILLION <= before * rule.threshold_millionths
    if rule.gate is _GateKind.RELATIVE_MIN:
        return after * MILLION >= before * rule.threshold_millionths
    if rule.gate is _GateKind.DELTA_MIN:
        return after - before >= rule.threshold_millionths
    if rule.gate is _GateKind.STRICT_LOWER:
        return after < before
    raise AssertionError(f"unsupported gate: {rule.gate}")


def _build_component(
    dimension: V2ObjectiveDimension,
    baseline: V2ProducerReceipt,
    candidate: V2ProducerReceipt,
) -> V2ParetoComponent:
    spec = _SPECS[dimension]
    baseline_values = {
        name: baseline.metric_samples[name].value_millionths
        for name in spec.metrics
        if name in baseline.metric_samples
    }
    candidate_values = {
        name: candidate.metric_samples[name].value_millionths
        for name in spec.metrics
        if name in candidate.metric_samples
    }
    metric_passed: dict[str, bool] = {}
    for name, rule in spec.metrics.items():
        if name not in baseline.metric_samples or name not in candidate.metric_samples:
            metric_passed[name] = False
            continue
        before = baseline.metric_samples[name].value_millionths
        after = candidate.metric_samples[name].value_millionths
        non_regressing = (
            after >= before
            if rule.direction is V2MetricDirection.HIGHER
            else after <= before
        )
        metric_passed[name] = non_regressing and _metric_passes(
            rule,
            baseline.metric_samples[name],
            candidate.metric_samples[name],
        )
    for group in spec.any_of:
        group_passed = any(metric_passed.get(name, False) for name in group)
        for name in group:
            metric_passed[name] = group_passed

    failures = tuple(
        name for name in spec.metrics if not metric_passed.get(name, False)
    )
    improved = False
    regressed = False
    for name, rule in spec.metrics.items():
        if name not in baseline_values or name not in candidate_values:
            continue
        before = baseline_values[name]
        after = candidate_values[name]
        if rule.direction is V2MetricDirection.HIGHER:
            improved |= after > before
            regressed |= after < before
        else:
            improved |= after < before
            regressed |= after > before
    return V2ParetoComponent(
        dimension=dimension,
        baseline_values_millionths=MappingProxyType(baseline_values),
        candidate_values_millionths=MappingProxyType(candidate_values),
        metric_passed=MappingProxyType(metric_passed),
        gate_failures=failures,
        improved=improved,
        regressed=regressed,
        passed=not failures,
    )


def _expected_receipt_ids(
    corpus: V2PairedBenchmarkCorpus,
    arm: V2BenchmarkArm,
) -> tuple[str, ...]:
    if arm is V2BenchmarkArm.BASELINE:
        return tuple(item.baseline.receipt_id for item in corpus.cases)
    return tuple(item.candidate.receipt_id for item in corpus.cases)


def _append(
    failures: dict[str, set[str]], check: str, dimension: V2ObjectiveDimension
) -> None:
    failures[check].add(dimension.value)


def _audit_receipt_pair(
    corpus: V2PairedBenchmarkCorpus,
    dimension: V2ObjectiveDimension,
    baseline: V2ProducerReceipt,
    candidate: V2ProducerReceipt,
    failures: dict[str, set[str]],
) -> list[str]:
    spec = _SPECS[dimension]
    fixture_ids = corpus.fixture_population_ids
    task_ids = tuple(_task_id(kind) for kind in REQUIRED_V2_FIXTURE_KINDS)
    expected_cache = _expected_cache_states()
    non_compensable: list[str] = []

    for receipt, arm in (
        (baseline, V2BenchmarkArm.BASELINE),
        (candidate, V2BenchmarkArm.CANDIDATE),
    ):
        if (
            receipt.dimension is not dimension
            or receipt.arm is not arm
            or receipt.producer_id != spec.producer_id
            or receipt.corpus_id != corpus.corpus_id
            or receipt.fixture_population_ids != fixture_ids
            or receipt.source_receipt_ids
            != _expected_receipt_ids(corpus, arm)
        ):
            non_compensable.append(
                f"population:{dimension.value}:{arm.value}"
            )
        if receipt.hard_fixture_ids != spec.hard_fixture_ids:
            _append(failures, "omitted-hard-fixture", dimension)
        expected_names = set(spec.metrics)
        if set(receipt.metric_samples) != expected_names:
            _append(failures, "metric-substitution", dimension)
        else:
            for name, rule in spec.metrics.items():
                if receipt.metric_samples[name].unit != rule.unit:
                    _append(failures, "metric-substitution", dimension)
        if (
            receipt.eligible_task_ids != task_ids
            or receipt.measured_task_ids != task_ids
        ):
            _append(failures, "cherry-picked-task", dimension)
        if (
            len(receipt.source_receipt_ids)
            != len(set(receipt.source_receipt_ids))
            or len(receipt.evidence_ids) != len(set(receipt.evidence_ids))
            or len(receipt.evidence_ids) < len(fixture_ids)
        ):
            _append(failures, "duplicated-evidence", dimension)
        if receipt.cache_states != expected_cache:
            _append(failures, "cache-warming-leakage", dimension)
        cold_id = _fixture_id(V2FixtureKind.COLD)
        if receipt.cache_states.get(cold_id) is not V2CacheState.COLD:
            _append(failures, "cache-warming-leakage", dimension)
        if (
            receipt.work_started_ms < receipt.window_started_ms
            or receipt.work_ended_ms > receipt.window_ended_ms
        ):
            _append(failures, "work-outside-window", dimension)
        if receipt.warmup_started_ms and (
            receipt.warmup_started_ms < receipt.window_started_ms
            or receipt.warmup_ended_ms > receipt.window_ended_ms
        ):
            _append(failures, "cache-warming-leakage", dimension)
        non_compensable.extend(
            f"producer:{dimension.value}:{arm.value}:{failure}"
            for failure in receipt.non_compensable_failures
        )

    if (
        baseline.fixture_population_ids != candidate.fixture_population_ids
        or baseline.eligible_task_ids != candidate.eligible_task_ids
        or baseline.measured_task_ids != candidate.measured_task_ids
        or baseline.hard_fixture_ids != candidate.hard_fixture_ids
    ):
        non_compensable.append(f"paired-population:{dimension.value}")
    if baseline.cache_states != candidate.cache_states:
        _append(failures, "cache-warming-leakage", dimension)
    for name in set(baseline.metric_samples) & set(candidate.metric_samples):
        if (
            baseline.metric_samples[name].denominator
            != candidate.metric_samples[name].denominator
        ):
            _append(failures, "denominator-shift", dimension)
    if set(baseline.evidence_ids) & set(candidate.evidence_ids):
        _append(failures, "duplicated-evidence", dimension)
    return non_compensable


def _ablation_result(
    item: V2AblationReceipt,
    baseline: V2ProducerReceipt,
    candidate: V2ProducerReceipt,
) -> V2AblationResult:
    spec = _SPECS[item.dimension]
    affected: list[str] = []
    preventative = bool(item.non_compensable_failures_without)
    causal = preventative
    for name, rule in spec.metrics.items():
        if name not in item.metric_samples_without:
            continue
        without = item.metric_samples_without[name].value_millionths
        with_value = candidate.metric_samples[name].value_millionths
        before = baseline.metric_samples[name].value_millionths
        if without == with_value:
            continue
        affected.append(name)
        with_distance = abs(with_value - before)
        without_distance = abs(without - before)
        # A contributor is causal when removing it erases movement from the
        # baseline, or when removal breaks an otherwise passing component.
        if without_distance < with_distance:
            causal = True
        if _metric_passes(
            rule, baseline.metric_samples[name], candidate.metric_samples[name]
        ) and not _metric_passes(
            rule, baseline.metric_samples[name], item.metric_samples_without[name]
        ):
            causal = True
            preventative = True
    return V2AblationResult(
        dimension=item.dimension,
        contributor_id=item.contributor_id,
        receipt_id=item.receipt_id,
        affected_metric_ids=tuple(affected),
        causal=causal,
        preventative=preventative,
    )


class V2SelfImprovementEvaluator:
    """Deterministic bounded evaluator for the complete v2 population."""

    def __init__(
        self,
        *,
        policy_id: str = V2_SELF_EVALUATION_POLICY_ID,
        maximum_ablations: int = MAX_V2_ABLATIONS,
    ) -> None:
        self.policy_id = _code(policy_id, "policy_id")
        self.maximum_ablations = _integer(
            maximum_ablations,
            "maximum_ablations",
            minimum=len(REQUIRED_V2_OBJECTIVE_DIMENSIONS),
            maximum=MAX_V2_ABLATIONS,
        )

    def evaluate(
        self,
        corpus: V2PairedBenchmarkCorpus,
        producer_receipts: Sequence[V2ProducerReceipt],
        ablation_receipts: Sequence[V2AblationReceipt],
    ) -> V2SelfEvaluationReport:
        if not isinstance(corpus, V2PairedBenchmarkCorpus):
            raise V2SelfEvaluationError(
                "corpus must be V2PairedBenchmarkCorpus"
            )
        if isinstance(producer_receipts, (str, bytes)) or not isinstance(
            producer_receipts, Sequence
        ):
            raise V2SelfEvaluationError("producer_receipts must be a sequence")
        if isinstance(ablation_receipts, (str, bytes)) or not isinstance(
            ablation_receipts, Sequence
        ):
            raise V2SelfEvaluationError("ablation_receipts must be a sequence")
        if len(ablation_receipts) > self.maximum_ablations:
            raise V2SelfEvaluationError("ablation budget exceeded")

        pair_by_dimension: dict[
            V2ObjectiveDimension, dict[V2BenchmarkArm, V2ProducerReceipt]
        ] = {}
        duplicate_population = False
        for item in producer_receipts:
            if not isinstance(item, V2ProducerReceipt):
                raise V2SelfEvaluationError(
                    "producer_receipts must contain V2ProducerReceipt"
                )
            arms = pair_by_dimension.setdefault(item.dimension, {})
            if item.arm in arms:
                duplicate_population = True
            else:
                arms[item.arm] = item

        failures: dict[str, set[str]] = {
            name: set() for name in ANTI_GAMING_CHECKS
        }
        dimensions_complete = (
            set(pair_by_dimension) == set(REQUIRED_V2_OBJECTIVE_DIMENSIONS)
            and all(
                set(pair_by_dimension[dimension]) == set(V2BenchmarkArm)
                for dimension in REQUIRED_V2_OBJECTIVE_DIMENSIONS
                if dimension in pair_by_dimension
            )
            and not duplicate_population
        )
        non_compensable: list[str] = []
        benchmark = build_v2_benchmark_report(corpus)
        if not benchmark.population_complete:
            non_compensable.append("benchmark-population")
        if not benchmark.baseline_candidate_paired:
            non_compensable.append("benchmark-pairing")
        for gate, fixture_ids in benchmark.gate_failures.items():
            non_compensable.extend(
                f"benchmark:{gate}:{fixture_id}" for fixture_id in fixture_ids
            )

        components: dict[V2ObjectiveDimension, V2ParetoComponent] = {}
        if dimensions_complete:
            for dimension in REQUIRED_V2_OBJECTIVE_DIMENSIONS:
                baseline = pair_by_dimension[dimension][V2BenchmarkArm.BASELINE]
                candidate = pair_by_dimension[dimension][V2BenchmarkArm.CANDIDATE]
                non_compensable.extend(
                    _audit_receipt_pair(
                        corpus, dimension, baseline, candidate, failures
                    )
                )
                components[dimension] = _build_component(
                    dimension, baseline, candidate
                )
        else:
            non_compensable.append("producer-population")
            # Reports retain an exact vector even on malformed populations.
            for dimension in REQUIRED_V2_OBJECTIVE_DIMENSIONS:
                arms = pair_by_dimension.get(dimension, {})
                if set(arms) == set(V2BenchmarkArm):
                    components[dimension] = _build_component(
                        dimension,
                        arms[V2BenchmarkArm.BASELINE],
                        arms[V2BenchmarkArm.CANDIDATE],
                    )
                    continue
                empty = MappingProxyType({})
                components[dimension] = V2ParetoComponent(
                    dimension,
                    empty,
                    empty,
                    empty,
                    ("missing-producer-receipt",),
                    False,
                    False,
                    False,
                )

        ablation_by_dimension: dict[
            V2ObjectiveDimension, list[V2AblationReceipt]
        ] = {}
        ablation_population_valid = True
        seen_ablation_keys: set[tuple[V2ObjectiveDimension, str]] = set()
        ablation_results: list[V2AblationResult] = []
        for item in ablation_receipts:
            if not isinstance(item, V2AblationReceipt):
                raise V2SelfEvaluationError(
                    "ablation_receipts must contain V2AblationReceipt"
                )
            key = (item.dimension, item.contributor_id)
            if key in seen_ablation_keys:
                ablation_population_valid = False
            seen_ablation_keys.add(key)
            ablation_by_dimension.setdefault(item.dimension, []).append(item)
            arms = pair_by_dimension.get(item.dimension, {})
            if set(arms) != set(V2BenchmarkArm):
                ablation_population_valid = False
                continue
            baseline = arms[V2BenchmarkArm.BASELINE]
            candidate = arms[V2BenchmarkArm.CANDIDATE]
            if (
                item.candidate_receipt_id != candidate.receipt_id
                or item.corpus_id != corpus.corpus_id
                or item.fixture_population_ids
                != candidate.fixture_population_ids
                or item.measured_task_ids != candidate.measured_task_ids
                or item.source_receipt_ids != candidate.source_receipt_ids
                or set(baseline.metric_samples)
                != set(_SPECS[item.dimension].metrics)
                or set(candidate.metric_samples)
                != set(_SPECS[item.dimension].metrics)
                or set(item.metric_samples_without)
                != set(_SPECS[item.dimension].metrics)
                or any(
                    item.metric_samples_without[name].unit
                    != _SPECS[item.dimension].metrics[name].unit
                    for name in _SPECS[item.dimension].metrics
                    if name in item.metric_samples_without
                )
            ):
                ablation_population_valid = False
                continue
            ablation_results.append(
                _ablation_result(item, baseline, candidate)
            )
        if set(ablation_by_dimension) != set(REQUIRED_V2_OBJECTIVE_DIMENSIONS):
            ablation_population_valid = False
        if any(
            not ablation_by_dimension.get(dimension)
            for dimension in REQUIRED_V2_OBJECTIVE_DIMENSIONS
        ):
            ablation_population_valid = False
        if not ablation_population_valid:
            non_compensable.append("ablation-population")

        anti_gaming = MappingProxyType(
            {
                name: tuple(sorted(failures[name]))
                for name in ANTI_GAMING_CHECKS
            }
        )
        population_complete = (
            dimensions_complete
            and benchmark.population_complete
            and benchmark.baseline_candidate_paired
            and ablation_population_valid
        )
        pareto_passed = all(
            component.passed for component in components.values()
        )
        non_compensable_tuple = tuple(sorted(set(non_compensable)))
        passed = (
            population_complete
            and pareto_passed
            and not any(anti_gaming.values())
            and not non_compensable_tuple
        )
        return V2SelfEvaluationReport(
            corpus_id=corpus.corpus_id,
            policy_id=self.policy_id,
            producer_receipt_ids=tuple(
                item.receipt_id for item in producer_receipts
            ),
            ablation_receipt_ids=tuple(
                item.receipt_id for item in ablation_receipts
            ),
            pareto_vector=MappingProxyType(components),
            ablations=tuple(ablation_results),
            anti_gaming_failures=anti_gaming,
            non_compensable_failures=non_compensable_tuple,
            population_complete=population_complete,
            pareto_passed=pareto_passed,
            decision=(
                V2EvaluationDecision.PROVISIONAL
                if passed
                else V2EvaluationDecision.SHADOW
            ),
            passed=passed,
        )


def _sample(
    numerator: int, denominator: int, unit: str
) -> V2MetricSample:
    return V2MetricSample(numerator, denominator, unit)


def _default_samples(
    dimension: V2ObjectiveDimension,
    arm: V2BenchmarkArm,
    corpus: V2PairedBenchmarkCorpus,
) -> Mapping[str, V2MetricSample]:
    candidate = arm is V2BenchmarkArm.CANDIDATE
    report = build_v2_benchmark_report(corpus)
    totals = report.candidate if candidate else report.baseline
    count = len(corpus.cases)
    values: dict[V2ObjectiveDimension, dict[str, tuple[int, int]]] = {
        V2ObjectiveDimension.SAFETY: {
            "unsafe-fixture-rate": (0, count),
        },
        V2ObjectiveDimension.TOKENS: {
            "input-tokens-per-criterion": (
                totals.provider_input_tokens,
                totals.terminal_accepted_criteria,
            ),
            "retry-input-tokens-per-task": (
                totals.retry_input_tokens,
                count,
            ),
            "required-evidence-coverage": (
                totals.terminal_accepted_criteria,
                totals.terminal_required_criteria,
            ),
        },
        V2ObjectiveDimension.CONTEXT_REUSE: {
            "stable-prefix-reuse": (75 if candidate else 50, 100),
            "exact-semantic-invalidation": (count, count),
        },
        V2ObjectiveDimension.PLANNING: {
            "first-valid-plan-rate": (82 if candidate else 65, 100),
            "invalid-branch-rate": (25 if candidate else 40, 100),
            "hard-constraint-violation-rate": (0, 100),
        },
        V2ObjectiveDimension.ANALYSIS: {
            "reuse-or-offload-rate": (75 if candidate else 40, 100),
            "typed-outcome-rate": (count, count),
            "provider-authority-violation-rate": (0, count),
        },
        V2ObjectiveDimension.CACHE: {
            "warm-exact-reuse-rate": (85 if candidate else 50, 100),
            "duplicate-miss-collapse-rate": (70 if candidate else 30, 100),
            "stale-authoritative-hit-rate": (0, 100),
            "quota-violation-rate": (0, 100),
        },
        V2ObjectiveDimension.VALIDATION: {
            "escaped-seeded-defect-rate": (0, 100),
            "time-to-first-useful-failure": (
                6_000 if candidate else 10_000,
                1,
            ),
            "flaky-authority-rate": (0, 100),
        },
        V2ObjectiveDimension.TASK_QUALITY: {
            "acceptance-coverage-rate": (count, count),
            "model-calls-per-criterion": (
                20 if candidate else 28,
                count,
            ),
            "duplicate-semantic-task-rate": (0, count),
        },
        V2ObjectiveDimension.THROUGHPUT: {
            "accepted-throughput": (35 if candidate else 10, 100),
            "duplicate-compute-rate": (3 if candidate else 4, 100),
            "conflict-regression-rate": (0, 100),
            "resource-bound-violation-rate": (0, 100),
        },
        V2ObjectiveDimension.PERSISTENCE: {
            "maximum-receipt-bytes": (
                80_000 if candidate else 100_000,
                1,
            ),
            "maximum-projection-bytes": (
                400_000 if candidate else 500_000,
                1,
            ),
            "duplicated-payload-graph-rate": (0, 100),
            "bounded-growth-rate": (count, count),
        },
        V2ObjectiveDimension.IDLE_RELIABILITY: {
            "idle-cpu-milli-percent": (
                700 if candidate else 1_400,
                1,
            ),
            "unchanged-state-writes": (0, 1),
            "idle-observation-ms": (MIN_DRAINED_OBSERVATION_MS, 1),
        },
        V2ObjectiveDimension.CONTROL: {
            "surface-conformance-rate": (count, count),
            "mutation-guard-rate": (count, count),
        },
        V2ObjectiveDimension.REFILL: {
            "exact-replay-noop-rate": (count, count),
            "duplicate-successor-rate": (0, count),
            "goals-per-epoch": (4 if candidate else 8, 1),
            "tasks-per-epoch": (12 if candidate else 24, 1),
            "healthy-trigger-guard-rate": (count, count),
        },
    }
    spec = _SPECS[dimension]
    return MappingProxyType(
        {
            name: _sample(
                values[dimension][name][0],
                values[dimension][name][1],
                rule.unit,
            )
            for name, rule in spec.metrics.items()
        }
    )


def build_frozen_v2_producer_receipts(
    corpus: V2PairedBenchmarkCorpus,
) -> tuple[V2ProducerReceipt, ...]:
    """Build deterministic producer fixtures for the closed v2 population."""

    fixture_ids = corpus.fixture_population_ids
    task_ids = tuple(_task_id(kind) for kind in REQUIRED_V2_FIXTURE_KINDS)
    cache_states = _expected_cache_states()
    receipts: list[V2ProducerReceipt] = []
    for dimension in REQUIRED_V2_OBJECTIVE_DIMENSIONS:
        spec = _SPECS[dimension]
        for arm in V2BenchmarkArm:
            source_ids = _expected_receipt_ids(corpus, arm)
            evidence_ids = tuple(
                f"evidence:{dimension.value}:{arm.value}:{index}@1"
                for index in range(len(fixture_ids))
            )
            receipts.append(
                V2ProducerReceipt(
                    dimension=dimension,
                    arm=arm,
                    producer_id=spec.producer_id,
                    corpus_id=corpus.corpus_id,
                    metric_samples=_default_samples(dimension, arm, corpus),
                    fixture_population_ids=fixture_ids,
                    hard_fixture_ids=spec.hard_fixture_ids,
                    eligible_task_ids=task_ids,
                    measured_task_ids=task_ids,
                    source_receipt_ids=source_ids,
                    evidence_ids=evidence_ids,
                    window_started_ms=1_000,
                    window_ended_ms=1_000 + MIN_DRAINED_OBSERVATION_MS,
                    work_started_ms=1_000,
                    work_ended_ms=1_000 + MIN_DRAINED_OBSERVATION_MS,
                    cache_states=cache_states,
                    warmup_started_ms=1_100,
                    warmup_ended_ms=1_200,
                )
            )
    return tuple(receipts)


def _failing_counterfactual(
    dimension: V2ObjectiveDimension,
    baseline: V2ProducerReceipt,
    candidate: V2ProducerReceipt,
) -> tuple[Mapping[str, V2MetricSample], tuple[str, ...]]:
    samples = dict(baseline.metric_samples)
    spec = _SPECS[dimension]
    if all(
        samples[name].value_millionths
        == candidate.metric_samples[name].value_millionths
        for name in spec.metrics
    ):
        first_name = next(iter(spec.metrics))
        rule = spec.metrics[first_name]
        original = samples[first_name]
        if rule.direction is V2MetricDirection.LOWER:
            samples[first_name] = replace(
                original, numerator=original.denominator
            )
        else:
            samples[first_name] = replace(original, numerator=0)
    return MappingProxyType(samples), ()


def build_frozen_v2_ablation_receipts(
    corpus: V2PairedBenchmarkCorpus,
    producer_receipts: Sequence[V2ProducerReceipt],
) -> tuple[V2AblationReceipt, ...]:
    """Build one bounded leave-component-out counterfactual per dimension."""

    pairs = {
        (item.dimension, item.arm): item for item in producer_receipts
    }
    result: list[V2AblationReceipt] = []
    for dimension in REQUIRED_V2_OBJECTIVE_DIMENSIONS:
        baseline = pairs[(dimension, V2BenchmarkArm.BASELINE)]
        candidate = pairs[(dimension, V2BenchmarkArm.CANDIDATE)]
        samples, failures = _failing_counterfactual(
            dimension, baseline, candidate
        )
        result.append(
            V2AblationReceipt(
                dimension=dimension,
                contributor_id=f"component:{dimension.value}@1",
                candidate_receipt_id=candidate.receipt_id,
                corpus_id=corpus.corpus_id,
                metric_samples_without=samples,
                fixture_population_ids=candidate.fixture_population_ids,
                measured_task_ids=candidate.measured_task_ids,
                source_receipt_ids=candidate.source_receipt_ids,
                non_compensable_failures_without=failures,
            )
        )
    return tuple(result)


def build_frozen_v2_self_evaluation_inputs(
    corpus: V2PairedBenchmarkCorpus,
) -> tuple[tuple[V2ProducerReceipt, ...], tuple[V2AblationReceipt, ...]]:
    producer_receipts = build_frozen_v2_producer_receipts(corpus)
    return (
        producer_receipts,
        build_frozen_v2_ablation_receipts(corpus, producer_receipts),
    )


def evaluate_v2_self_improvement(
    corpus: V2PairedBenchmarkCorpus,
    producer_receipts: Sequence[V2ProducerReceipt],
    ablation_receipts: Sequence[V2AblationReceipt],
    *,
    policy_id: str = V2_SELF_EVALUATION_POLICY_ID,
) -> V2SelfEvaluationReport:
    """Evaluate the exact population and return a fail-closed Pareto report."""

    return V2SelfImprovementEvaluator(policy_id=policy_id).evaluate(
        corpus, producer_receipts, ablation_receipts
    )


def verify_v2_self_evaluation_report(
    report: V2SelfEvaluationReport | Mapping[str, Any],
    corpus: V2PairedBenchmarkCorpus,
    producer_receipts: Sequence[V2ProducerReceipt],
    ablation_receipts: Sequence[V2AblationReceipt],
    *,
    policy_id: str = V2_SELF_EVALUATION_POLICY_ID,
) -> V2SelfEvaluationReport:
    """Replay every derived field and reject a persisted forged summary."""

    expected = evaluate_v2_self_improvement(
        corpus,
        producer_receipts,
        ablation_receipts,
        policy_id=policy_id,
    )
    if isinstance(report, V2SelfEvaluationReport):
        payload = report.to_dict(include_report_id=True)
    elif isinstance(report, Mapping):
        payload = dict(report)
    else:
        raise V2SelfEvaluationError(
            "report must be a V2SelfEvaluationReport or object"
        )
    _reject_forbidden(payload)
    expected_payload = expected.to_dict(include_report_id=True)
    if set(payload) != set(expected_payload):
        raise V2SelfEvaluationError(
            "persisted report fields do not match the closed schema"
        )
    if payload != expected_payload:
        raise V2SelfEvaluationError(
            "persisted report does not match deterministic receipt replay"
        )
    return expected


def replay_v2_self_evaluation(
    corpus: V2PairedBenchmarkCorpus,
    producer_receipts: Sequence[V2ProducerReceipt],
    ablation_receipts: Sequence[V2AblationReceipt],
    *,
    expected_report: V2SelfEvaluationReport | Mapping[str, Any] | None = None,
    policy_id: str = V2_SELF_EVALUATION_POLICY_ID,
) -> V2SelfEvaluationReport:
    result = evaluate_v2_self_improvement(
        corpus,
        producer_receipts,
        ablation_receipts,
        policy_id=policy_id,
    )
    if expected_report is not None:
        return verify_v2_self_evaluation_report(
            expected_report,
            corpus,
            producer_receipts,
            ablation_receipts,
            policy_id=policy_id,
        )
    return result


# Discoverable names retained for the later public-surface integration task.
V2SelfEvaluationDimension = V2ObjectiveDimension
V2ComponentReceipt = V2ProducerReceipt
V2RewardResistantEvaluator = V2SelfImprovementEvaluator
build_reward_resistant_evaluation_report = evaluate_v2_self_improvement


__all__ = [
    "ACTIONABLE_V2_RESIDUAL_KINDS",
    "ANTI_GAMING_CHECKS",
    "MAX_V2_ABLATIONS",
    "MAX_V2_COMPONENT_RECEIPT_BYTES",
    "MAX_V2_SELF_EVALUATION_BYTES",
    "MAX_V2_SUCCESSOR_GOALS",
    "MAX_V2_SUCCESSOR_REJECTIONS",
    "MAX_V2_SUCCESSOR_RESIDUALS",
    "MAX_V2_SUCCESSOR_TASKS",
    "REQUIRED_V2_OBJECTIVE_DIMENSIONS",
    "REWARD_RESISTANT_EVALUATION_GOAL_ID",
    "REWARD_RESISTANT_EVALUATION_REQUIREMENT_ID",
    "TYPED_SUCCESSOR_REQUIREMENT_ID",
    "V2AblationReceipt",
    "V2AblationResult",
    "V2CacheState",
    "V2ComponentReceipt",
    "V2EvaluationDecision",
    "V2MetricDirection",
    "V2MetricSample",
    "V2ObjectiveDimension",
    "V2ParetoComponent",
    "V2ProducerReceipt",
    "V2RewardResistantEvaluator",
    "V2SelfEvaluationDimension",
    "V2SelfEvaluationError",
    "V2SelfEvaluationReport",
    "V2SelfImprovementEvaluator",
    "V2ResidualKind",
    "V2ResidualSignal",
    "V2SuccessorAdmission",
    "V2SuccessorCandidate",
    "V2SuccessorGenerationPolicy",
    "V2SuccessorGenerationResult",
    "V2SuccessorRejection",
    "V2SuccessorRejectionReason",
    "V2_COMPONENT_RECEIPT_SCHEMA",
    "V2_SELF_EVALUATION_CONTRACT_VERSION",
    "V2_SELF_EVALUATION_POLICY_ID",
    "V2_SELF_EVALUATION_SCHEMA",
    "build_frozen_v2_ablation_receipts",
    "build_frozen_v2_producer_receipts",
    "build_frozen_v2_self_evaluation_inputs",
    "build_reward_resistant_evaluation_report",
    "evaluate_v2_self_improvement",
    "generate_v2_successor_goals",
    "replay_v2_self_evaluation",
    "verify_v2_self_evaluation_report",
]
