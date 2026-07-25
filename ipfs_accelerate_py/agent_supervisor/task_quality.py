"""Deterministic quality, sizing, and semantic admission for generated tasks.

The task generator has two distinct responsibilities:

* describe a complete, executable unit of work; and
* keep the open board useful by rejecting duplicates and resizing work before
  it consumes scheduler capacity.

This module implements that boundary without depending on a model provider or
on Markdown display IDs.  Every decision is reproducible from canonical task
content, policy, history, and the current open-work count.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Any, Final

from .task_identity import (
    canonical_content_cid,
    canonical_json_bytes,
    canonical_task_identity,
)


TASK_QUALITY_SCHEMA = "ipfs_accelerate_py/agent-supervisor/task-quality@1"
TASK_SEMANTIC_IDENTITY_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/task-semantic-identity@1"
)
TASK_QUALITY_EVALUATOR_VERSION = "task-quality/v1"
TASK_SPLIT_REFILL_REQUIREMENT_ID: Final = (
    "127990245919649912156052660092678945998"
)
TASK_SPLIT_REFILL_EVIDENCE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/task-split-refill-evidence@1"
)
_TASK_SPLIT_REFILL_EVIDENCE_SEAL: Final = object()

RESOURCE_CLASSES = frozenset(
    {
        "cpu-small",
        "cpu-medium",
        "cpu-large",
        "cpu-proof-solver",
        "gpu-small",
        "gpu-medium",
        "gpu-large",
        "io-small",
        "io-medium",
        "network",
    }
)
TOKEN_CLASSES = frozenset({"tiny", "small", "medium", "large", "xlarge"})
TOKEN_CLASS_LIMITS = {
    "tiny": 1_024,
    "small": 4_096,
    "medium": 16_384,
    "large": 32_768,
    "xlarge": 65_536,
}


def _normalized_text(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


def _normalized_display_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _normalized_path(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return re.sub(r"/+", "/", text).rstrip("/")


def _strings(value: Any, *, paths: bool = False) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        raw: Iterable[Any] = re.split(r"[,;\n]+", value)
    elif isinstance(value, Mapping):
        raw = value.keys()
    elif isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        raw = value
    else:
        raw = (value,)
    normalize = _normalized_path if paths else _normalized_display_text
    values = {normalize(item) for item in raw if normalize(item)}
    return tuple(sorted(values, key=lambda item: (item.casefold(), item)))


def _mapping_value(payload: Mapping[str, Any], *names: str, default: Any = "") -> Any:
    normalized = {
        str(key).strip().casefold().replace("_", " "): value
        for key, value in payload.items()
    }
    for name in names:
        value = normalized.get(name.casefold().replace("_", " "))
        if value not in (None, ""):
            return value
    return default


def _non_negative_int(value: Any, name: str) -> int:
    if value in (None, ""):
        return 0
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return parsed


def _finite_ratio(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number between zero and one") from exc
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{name} must be a number between zero and one")
    return parsed


def _evidence_hash_material(value: Any) -> Any:
    """Project finite floats to stable strings for canonical receipt hashing."""

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("task-quality evidence cannot contain non-finite floats")
        return {"$finite_float": format(value, ".17g")}
    if isinstance(value, Mapping):
        return {
            str(key): _evidence_hash_material(child)
            for key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_evidence_hash_material(child) for child in value]
    return value


def _task_quality_evidence_bytes(value: Any) -> bytes:
    return canonical_json_bytes(_evidence_hash_material(value))


def _task_quality_evidence_cid(value: Any) -> str:
    return canonical_content_cid(_evidence_hash_material(value))


def _semantic_material(value: "TaskCandidate | Mapping[str, Any]") -> dict[str, Any]:
    candidate = (
        value
        if isinstance(value, TaskCandidate)
        else TaskCandidate.from_mapping(value, validate_identity=False)
    )
    material: dict[str, Any] = {
        "schema": TASK_SEMANTIC_IDENTITY_SCHEMA,
        "goal_id": _normalized_text(candidate.goal_id),
        "acceptance": sorted(_normalized_text(item) for item in candidate.acceptance),
        "preconditions": sorted(_normalized_text(item) for item in candidate.preconditions),
        "effects": sorted(_normalized_text(item) for item in candidate.effects),
        "evidence_subset": sorted(
            _normalized_text(item) for item in candidate.evidence_subset
        ),
        "outputs": sorted(path.casefold() for path in candidate.outputs),
        "predicted_symbols": sorted(
            _normalized_text(item) for item in candidate.predicted_symbols
        ),
        "validation_commands": sorted(
            _normalized_display_text(item) for item in candidate.validation_commands
        ),
        "merge_fate": _normalized_text(candidate.merge_fate),
    }
    # Context changes execution cost, but not the work's purpose.  It is used
    # as identity material only when there is no concrete output/symbol surface.
    if not material["outputs"] and not material["predicted_symbols"]:
        material["context_paths"] = sorted(
            path.casefold() for path in candidate.context_paths
        )
    if not any(
        material[name]
        for name in ("acceptance", "effects", "evidence_subset", "outputs", "predicted_symbols")
    ):
        material["title"] = _normalized_text(candidate.title)
    return material


def canonical_semantic_identity(value: "TaskCandidate | Mapping[str, Any]") -> str:
    """Return a display-ID-independent identity for one semantic unit of work."""

    digest = hashlib.sha256(canonical_json_bytes(_semantic_material(value))).hexdigest()
    return f"task-quality/v1/{digest}"


@dataclass(frozen=True)
class TaskCandidate:
    """A candidate task at the generated-task admission boundary.

    Fields which make a task executable intentionally default to empty.  This
    lets :func:`score_task_candidate` return explicit rejection reasons instead
    of failing while decoding an incomplete model response.
    """

    title: str = ""
    goal_id: str = ""
    acceptance: tuple[str, ...] = ()
    acceptance_criteria: tuple[str, ...] = ()
    preconditions: tuple[str, ...] = ()
    effects: tuple[str, ...] = ()
    evidence_subset: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    validation_commands: tuple[str, ...] = ()
    context_paths: tuple[str, ...] = ()
    context_keys: tuple[str, ...] = ()
    predicted_paths: tuple[str, ...] = ()
    predicted_symbols: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    resources: tuple[str, ...] = ()
    resource_class: str = ""
    token_class: str = ""
    merge_fate: str = ""
    priority: str = ""
    track: str = ""
    estimated_context_tokens: int = 0
    estimated_validation_seconds: int = 0
    estimated_tokens: int = 0
    historical_duplicate_similarity: float = 0.0
    historical_failure_similarity: float = 0.0
    source_id: str = ""
    semantic_identity: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        normalized_acceptance = _strings(
            self.acceptance or self.acceptance_criteria
        )
        normalized_context = _strings(
            self.context_paths or self.context_keys,
            paths=True,
        )
        object.__setattr__(self, "acceptance", normalized_acceptance)
        object.__setattr__(self, "acceptance_criteria", normalized_acceptance)
        object.__setattr__(self, "context_paths", normalized_context)
        object.__setattr__(self, "context_keys", normalized_context)
        for name in (
            "title",
            "goal_id",
            "resource_class",
            "token_class",
            "merge_fate",
            "priority",
            "track",
            "source_id",
        ):
            object.__setattr__(self, name, _normalized_display_text(getattr(self, name)))
        for name in (
            "preconditions",
            "effects",
            "evidence_subset",
            "validation_commands",
            "predicted_symbols",
            "dependencies",
            "conflicts",
            "resources",
        ):
            object.__setattr__(self, name, _strings(getattr(self, name)))
        for name in ("outputs", "predicted_paths"):
            object.__setattr__(self, name, _strings(getattr(self, name), paths=True))
        for name in (
            "estimated_context_tokens",
            "estimated_validation_seconds",
            "estimated_tokens",
        ):
            object.__setattr__(
                self,
                name,
                _non_negative_int(getattr(self, name), name),
            )
        for name in (
            "historical_duplicate_similarity",
            "historical_failure_similarity",
        ):
            object.__setattr__(self, name, _finite_ratio(getattr(self, name), name))
        metadata = dict(self.metadata) if isinstance(self.metadata, Mapping) else {}
        object.__setattr__(self, "metadata", metadata)
        expected = canonical_semantic_identity(self)
        supplied = str(self.semantic_identity or "").strip()
        if supplied and supplied != expected:
            raise ValueError(
                "semantic_identity does not match canonical semantic task content"
            )
        object.__setattr__(self, "semantic_identity", expected)

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        validate_identity: bool = True,
    ) -> "TaskCandidate":
        if not isinstance(payload, Mapping):
            raise TypeError("task candidates must be mappings")
        candidate = cls(
            title=_mapping_value(payload, "title", "summary"),
            goal_id=_mapping_value(
                payload,
                "goal id",
                "parent goal id",
                "parent objective id",
                "goal",
            ),
            acceptance=_strings(
                _mapping_value(
                    payload,
                    "acceptance",
                    "acceptance subset",
                    "acceptance criteria",
                )
            ),
            preconditions=_strings(
                _mapping_value(payload, "preconditions", "required preconditions")
            ),
            effects=_strings(
                _mapping_value(payload, "effects", "expected effects")
            ),
            evidence_subset=_strings(
                _mapping_value(
                    payload,
                    "evidence subset",
                    "expected evidence delta",
                    "missing evidence",
                    "evidence",
                )
            ),
            outputs=_strings(
                _mapping_value(payload, "outputs", "files"),
                paths=True,
            ),
            validation_commands=_strings(
                _mapping_value(payload, "validation commands", "validation")
            ),
            context_paths=_strings(
                _mapping_value(
                    payload,
                    "context paths",
                    "context keys",
                    "context files",
                    "context",
                ),
                paths=True,
            ),
            predicted_paths=_strings(
                _mapping_value(payload, "predicted paths", "predicted files"),
                paths=True,
            ),
            predicted_symbols=_strings(
                _mapping_value(payload, "predicted symbols", "ast symbols", "symbols")
            ),
            dependencies=_strings(
                _mapping_value(payload, "dependencies", "depends on")
            ),
            conflicts=_strings(
                _mapping_value(payload, "conflicts", "conflict keys")
            ),
            resources=_strings(
                _mapping_value(payload, "resources", "required resources")
            ),
            resource_class=_mapping_value(payload, "resource class"),
            token_class=_mapping_value(payload, "token class", "token budget class"),
            merge_fate=_mapping_value(payload, "merge fate", "merge family", "merge key"),
            priority=_mapping_value(payload, "priority"),
            track=_mapping_value(payload, "track"),
            estimated_context_tokens=_mapping_value(
                payload, "estimated context tokens", "context tokens", default=0
            ),
            estimated_validation_seconds=_mapping_value(
                payload,
                "estimated validation seconds",
                "validation seconds",
                "validation cost",
                default=0,
            ),
            estimated_tokens=_mapping_value(
                payload, "estimated tokens", "token cost", default=0
            ),
            historical_duplicate_similarity=_mapping_value(
                payload,
                "historical duplicate similarity",
                "duplicate similarity",
                default=0.0,
            ),
            historical_failure_similarity=_mapping_value(
                payload,
                "historical failure similarity",
                "failure similarity",
                default=0.0,
            ),
            source_id=_mapping_value(
                payload, "source id", "task id", "finding id", "proposal id"
            ),
            semantic_identity=(
                _mapping_value(
                    payload,
                    "canonical semantic identity",
                )
                if validate_identity
                else ""
            ),
            metadata=payload.get("metadata", {}),
        )
        if validate_identity:
            supplied_key = str(
                _mapping_value(payload, "canonical task key") or ""
            ).strip()
            supplied_cid = str(
                _mapping_value(payload, "canonical task cid", "task cid") or ""
            ).strip()
            if supplied_key and supplied_key != candidate.canonical_task_key:
                raise ValueError(
                    "canonical_task_key does not match canonical semantic task content"
                )
            if supplied_cid and supplied_cid != candidate.canonical_task_cid:
                raise ValueError(
                    "canonical_task_cid does not match canonical semantic task content"
                )
        return candidate

    from_dict = from_mapping

    @property
    def predicted_path_breadth(self) -> int:
        return len(set(self.outputs) | set(self.predicted_paths))

    @property
    def predicted_symbol_breadth(self) -> int:
        return len(self.predicted_symbols)

    @property
    def canonical_task_key(self) -> str:
        """Return the repository-wide canonical task key for this candidate."""

        return canonical_task_identity(
            {"dedupe_key": self.semantic_identity}
        ).canonical_task_key

    @property
    def canonical_task_cid(self) -> str:
        """Return the repository-wide canonical task CID for this candidate."""

        return canonical_task_identity(
            {"dedupe_key": self.semantic_identity}
        ).canonical_task_cid

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for name in (
            "acceptance",
            "acceptance_criteria",
            "preconditions",
            "effects",
            "evidence_subset",
            "outputs",
            "validation_commands",
            "context_paths",
            "context_keys",
            "predicted_paths",
            "predicted_symbols",
            "dependencies",
            "conflicts",
            "resources",
        ):
            result[name] = list(result[name])
        result["canonical_semantic_identity"] = self.semantic_identity
        result["canonical_task_key"] = self.canonical_task_key
        result["canonical_task_cid"] = self.canonical_task_cid
        result["task_cid"] = self.canonical_task_cid
        result["predicted_path_breadth"] = self.predicted_path_breadth
        result["predicted_symbol_breadth"] = self.predicted_symbol_breadth
        return result


@dataclass(frozen=True)
class HistoricalTask:
    """A normalized historical work item used for similarity penalties."""

    candidate: TaskCandidate
    outcome: str = "accepted"
    failure_reason: str = ""

    @classmethod
    def from_value(cls, value: Any, *, default_outcome: str) -> "HistoricalTask":
        if isinstance(value, cls):
            return value
        if isinstance(value, TaskCandidate):
            return cls(value, default_outcome)
        if not isinstance(value, Mapping):
            raise TypeError("historical task records must be candidates or mappings")
        task_payload = value.get("candidate") or value.get("task") or value
        return cls(
            candidate=(
                task_payload
                if isinstance(task_payload, TaskCandidate)
                else TaskCandidate.from_mapping(task_payload)
            ),
            outcome=_normalized_display_text(value.get("outcome") or default_outcome).casefold(),
            failure_reason=_normalized_display_text(
                value.get("failure_reason") or value.get("reason") or ""
            ),
        )


@dataclass(frozen=True)
class TaskQualityPolicy:
    """Admission thresholds and deterministic sizing limits."""

    min_quality_score: float = 0.55
    duplicate_similarity_threshold: float = 0.82
    failure_similarity_threshold: float = 0.90
    max_predicted_paths: int = 8
    max_predicted_symbols: int = 24
    max_acceptance_criteria: int = 12
    max_effects: int = 12
    max_evidence_items: int = 16
    max_context_paths: int = 16
    max_context_tokens: int = 24_000
    max_validation_seconds: int = 1_800
    max_estimated_tokens: int = 32_768
    max_dependencies: int = 16
    max_conflicts: int = 8
    tiny_max_paths: int = 2
    tiny_max_symbols: int = 2
    tiny_max_tokens: int = 1_024
    max_open_work: int = 48
    max_new_work: int = 12
    max_split_parts: int = 32
    split_over_broad: bool = True
    coalesce_tiny: bool = True

    def __post_init__(self) -> None:
        for name in (
            "min_quality_score",
            "duplicate_similarity_threshold",
            "failure_similarity_threshold",
        ):
            object.__setattr__(self, name, _finite_ratio(getattr(self, name), name))
        for name in (
            "max_predicted_paths",
            "max_predicted_symbols",
            "max_acceptance_criteria",
            "max_effects",
            "max_evidence_items",
            "max_context_paths",
            "max_context_tokens",
            "max_validation_seconds",
            "max_estimated_tokens",
            "max_dependencies",
            "max_conflicts",
            "tiny_max_paths",
            "tiny_max_symbols",
            "tiny_max_tokens",
            "max_open_work",
            "max_new_work",
            "max_split_parts",
        ):
            value = _non_negative_int(getattr(self, name), name)
            object.__setattr__(self, name, value)
        if (
            self.max_predicted_paths == 0
            or self.max_predicted_symbols == 0
            or self.max_acceptance_criteria == 0
            or self.max_effects == 0
            or self.max_evidence_items == 0
            or self.max_split_parts == 0
        ):
            raise ValueError("task breadth limits must be positive")
        if self.max_new_work > self.max_open_work:
            object.__setattr__(self, "max_new_work", self.max_open_work)

    @property
    def policy_id(self) -> str:
        """Return the content identity of every sizing and admission threshold."""

        digest = hashlib.sha256(
            _task_quality_evidence_bytes(asdict(self))
        ).hexdigest()
        return f"task-quality-policy/v1/{digest}"


@dataclass(frozen=True)
class TaskQualityScore:
    """The complete auditable score for one task candidate."""

    acceptance_coverage: float
    coherent_effects: float
    breadth_fit: float
    context_cost: float
    validation_cost: float
    dependency_cost: float
    conflict_cost: float
    resource_fit: float
    historical_novelty: float
    historical_failure_safety: float
    duplicate_similarity: float
    failure_similarity: float
    total: float
    rationale: tuple[str, ...] = ()
    rejection_reasons: tuple["TaskRejection", ...] = ()

    @property
    def score(self) -> float:
        return self.total

    @property
    def total_score(self) -> float:
        return self.total

    @property
    def accepted(self) -> bool:
        return not self.rejection_reasons

    @property
    def dimensions(self) -> dict[str, float]:
        return {
            "acceptance_coverage": self.acceptance_coverage,
            "effect_coherence": self.coherent_effects,
            "scope": self.breadth_fit,
            "context_cost": self.context_cost,
            "validation_cost": self.validation_cost,
            "dependency_quality": self.dependency_cost,
            "conflict_cost": self.conflict_cost,
            "resource_fit": self.resource_fit,
            "duplicate_novelty": self.historical_novelty,
            "failure_risk": self.historical_failure_safety,
        }

    @property
    def total_millionths(self) -> int:
        return int(round(self.total * 1_000_000))

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["rationale"] = list(self.rationale)
        result["rejection_reasons"] = [
            item.to_dict() for item in self.rejection_reasons
        ]
        result["accepted"] = self.accepted
        result["dimensions"] = self.dimensions
        result["total_millionths"] = self.total_millionths
        return result


@dataclass(frozen=True)
class TaskRejection:
    """A machine-readable reason a candidate did not consume board capacity."""

    reason: str
    semantic_identity: str
    detail: str = ""

    @property
    def code(self) -> str:
        return self.reason

    @property
    def message(self) -> str:
        return self.detail

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


class TaskAdmissionStatus(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    SPLIT = "split"
    COALESCED = "coalesced"


@dataclass(frozen=True)
class TaskAdmissionDecision:
    candidate: TaskCandidate
    score: TaskQualityScore
    status: TaskAdmissionStatus
    rejections: tuple[TaskRejection, ...] = ()
    source_identities: tuple[str, ...] = ()

    @property
    def accepted(self) -> bool:
        return self.status in {
            TaskAdmissionStatus.ACCEPTED,
            TaskAdmissionStatus.SPLIT,
            TaskAdmissionStatus.COALESCED,
        }

    @property
    def rejection_reasons(self) -> tuple[str, ...]:
        return tuple(item.reason for item in self.rejections)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "score": self.score.to_dict(),
            "status": self.status.value,
            "accepted": self.accepted,
            "rejections": [item.to_dict() for item in self.rejections],
            "rejection_reasons": list(self.rejection_reasons),
            "source_identities": list(self.source_identities),
        }


@dataclass(frozen=True)
class TaskAdmissionResult:
    """Stable output of one split/coalesce/dedupe/admission cycle."""

    decisions: tuple[TaskAdmissionDecision, ...]
    initial_open_work: int
    max_open_work: int
    candidate_count: int

    @property
    def accepted(self) -> tuple[TaskCandidate, ...]:
        return tuple(item.candidate for item in self.decisions if item.accepted)

    @property
    def rejected(self) -> tuple[TaskAdmissionDecision, ...]:
        return tuple(item for item in self.decisions if not item.accepted)

    @property
    def final_open_work(self) -> int:
        return self.initial_open_work + len(self.accepted)

    @property
    def final_open_work_count(self) -> int:
        return self.final_open_work

    @property
    def bounded(self) -> bool:
        return self.final_open_work <= self.max_open_work

    def to_dict(self) -> dict[str, Any]:
        rejection_counts: dict[str, int] = {}
        for decision in self.rejected:
            for reason in decision.rejection_reasons:
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
        return {
            "schema": TASK_QUALITY_SCHEMA,
            "evaluator_version": TASK_QUALITY_EVALUATOR_VERSION,
            "decisions": [item.to_dict() for item in self.decisions],
            "accepted": [item.to_dict() for item in self.accepted],
            "rejected": [item.to_dict() for item in self.rejected],
            "rejection_counts": dict(sorted(rejection_counts.items())),
            "candidate_count": self.candidate_count,
            "initial_open_work": self.initial_open_work,
            "final_open_work": self.final_open_work,
            "max_open_work": self.max_open_work,
            "bounded": self.bounded,
        }


def _semantic_tokens(candidate: TaskCandidate) -> set[str]:
    values = [
        candidate.goal_id,
        *candidate.acceptance,
        *candidate.effects,
        *candidate.evidence_subset,
        *candidate.outputs,
        *candidate.predicted_symbols,
    ]
    return set(re.findall(r"[a-z0-9]+", " ".join(values).casefold()))


def task_semantic_similarity(left: TaskCandidate, right: TaskCandidate) -> float:
    """Return a conservative semantic Jaccard similarity in ``[0, 1]``."""

    if left.semantic_identity == right.semantic_identity:
        return 1.0
    if (
        left.goal_id
        and right.goal_id
        and _normalized_text(left.goal_id) != _normalized_text(right.goal_id)
    ):
        return 0.0
    left_tokens = _semantic_tokens(left)
    right_tokens = _semantic_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    token_similarity = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
    left_outputs, right_outputs = set(left.outputs), set(right.outputs)
    output_similarity = (
        len(left_outputs & right_outputs) / len(left_outputs | right_outputs)
        if left_outputs and right_outputs
        else 0.0
    )
    # Requiring both semantic and output overlap avoids treating tasks that
    # merely share a broad goal as duplicates.
    return 0.75 * token_similarity + 0.25 * output_similarity


def _history(
    values: Iterable[Any],
    *,
    default_outcome: str,
) -> tuple[HistoricalTask, ...]:
    return tuple(
        HistoricalTask.from_value(value, default_outcome=default_outcome)
        for value in values
    )


def _bounded_cost(value: int, limit: int) -> float:
    if value <= 0:
        return 1.0
    if limit <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - value / (limit * 1.25)))


def _acceptance_coverage(candidate: TaskCandidate) -> float:
    if not candidate.acceptance:
        return 0.0
    dimensions = [
        bool(candidate.evidence_subset),
        bool(candidate.validation_commands),
        bool(candidate.effects),
        bool(candidate.outputs or candidate.predicted_paths),
    ]
    return 0.5 + 0.5 * sum(dimensions) / len(dimensions)


def _effect_coherence(candidate: TaskCandidate) -> float:
    if not candidate.effects:
        return 0.0
    effect_tokens = set(
        re.findall(r"[a-z0-9]+", " ".join(candidate.effects).casefold())
    )
    surface_tokens = set(
        re.findall(
            r"[a-z0-9]+",
            " ".join(
                [
                    *candidate.acceptance,
                    *candidate.outputs,
                    *candidate.predicted_paths,
                    *candidate.predicted_symbols,
                ]
            ).casefold(),
        )
    )
    if not surface_tokens:
        return 0.5
    return 0.5 + 0.5 * min(1.0, len(effect_tokens & surface_tokens) / 3)


def _breadth_fit(candidate: TaskCandidate, policy: TaskQualityPolicy) -> float:
    paths = candidate.predicted_path_breadth
    symbols = candidate.predicted_symbol_breadth
    path_fit = 1.0 if paths else 0.25
    symbol_fit = 1.0
    if paths > policy.max_predicted_paths:
        path_fit = max(0.0, policy.max_predicted_paths / paths)
    if symbols > policy.max_predicted_symbols:
        symbol_fit = max(0.0, policy.max_predicted_symbols / symbols)
    return (path_fit + symbol_fit) / 2


def _resource_fit(candidate: TaskCandidate) -> float:
    if not candidate.resource_class or not candidate.token_class:
        return 0.0
    if (
        candidate.resource_class.casefold() not in RESOURCE_CLASSES
        or candidate.token_class.casefold() not in TOKEN_CLASSES
    ):
        return 0.0
    token_limit = TOKEN_CLASS_LIMITS[candidate.token_class.casefold()]
    if candidate.estimated_tokens and candidate.estimated_tokens > token_limit:
        return max(0.0, token_limit / candidate.estimated_tokens)
    return 1.0


def score_task_candidate(
    candidate: TaskCandidate | Mapping[str, Any],
    *,
    policy: TaskQualityPolicy | None = None,
    historical_tasks: Iterable[Any] = (),
    historical_failures: Iterable[Any] = (),
) -> TaskQualityScore:
    """Score all acceptance dimensions without mutating scheduler state."""

    item = (
        candidate if isinstance(candidate, TaskCandidate) else TaskCandidate.from_mapping(candidate)
    )
    selected = policy or TaskQualityPolicy()
    tasks = _history(historical_tasks, default_outcome="accepted")
    failures = _history(historical_failures, default_outcome="failed")
    duplicate_similarity = max(
        item.historical_duplicate_similarity,
        max(
            (task_semantic_similarity(item, prior.candidate) for prior in tasks),
            default=0.0,
        ),
    )
    failure_similarity = max(
        item.historical_failure_similarity,
        max(
            (task_semantic_similarity(item, prior.candidate) for prior in failures),
            default=0.0,
        ),
    )
    dimensions = {
        "acceptance_coverage": _acceptance_coverage(item),
        "coherent_effects": _effect_coherence(item),
        "breadth_fit": _breadth_fit(item, selected),
        "context_cost": (
            _bounded_cost(len(item.context_paths), selected.max_context_paths)
            + _bounded_cost(item.estimated_context_tokens, selected.max_context_tokens)
        )
        / 2,
        "validation_cost": _bounded_cost(
            item.estimated_validation_seconds,
            selected.max_validation_seconds,
        ),
        "dependency_cost": _bounded_cost(
            len(item.dependencies),
            selected.max_dependencies,
        ),
        "conflict_cost": _bounded_cost(len(item.conflicts), selected.max_conflicts),
        "resource_fit": _resource_fit(item),
        "historical_novelty": 1.0 - duplicate_similarity,
        "historical_failure_safety": 1.0 - failure_similarity,
    }
    weights = {
        "acceptance_coverage": 0.16,
        "coherent_effects": 0.14,
        "breadth_fit": 0.12,
        "context_cost": 0.09,
        "validation_cost": 0.09,
        "dependency_cost": 0.08,
        "conflict_cost": 0.08,
        "resource_fit": 0.08,
        "historical_novelty": 0.10,
        "historical_failure_safety": 0.06,
    }
    total = sum(dimensions[name] * weights[name] for name in weights)
    rationale = tuple(
        f"{name}={dimensions[name]:.6f} weight={weights[name]:.6f}"
        for name in weights
    )
    raw_score = TaskQualityScore(
        **dimensions,
        duplicate_similarity=duplicate_similarity,
        failure_similarity=failure_similarity,
        total=max(0.0, min(1.0, total)),
        rationale=rationale,
    )
    return replace(
        raw_score,
        rejection_reasons=_task_rejections(item, raw_score, selected),
    )


def _task_rejections(
    candidate: TaskCandidate,
    score: TaskQualityScore,
    policy: TaskQualityPolicy,
) -> tuple[TaskRejection, ...]:
    identity = candidate.semantic_identity
    reasons: list[TaskRejection] = []

    def reject(reason: str, detail: str) -> None:
        reasons.append(TaskRejection(reason, identity, detail))

    required = (
        ("missing_goal_id", candidate.goal_id, "goal_id is required"),
        ("missing_acceptance", candidate.acceptance, "acceptance coverage is required"),
        ("missing_preconditions", candidate.preconditions, "preconditions are required"),
        ("missing_effects", candidate.effects, "effects are required"),
        (
            "missing_evidence_subset",
            candidate.evidence_subset,
            "an evidence subset is required",
        ),
        (
            "missing_outputs",
            candidate.outputs or candidate.predicted_paths,
            "at least one output or predicted path is required",
        ),
        (
            "missing_validation",
            candidate.validation_commands,
            "validation commands are required",
        ),
        (
            "missing_resource_class",
            candidate.resource_class,
            "resource_class is required",
        ),
        ("missing_token_class", candidate.token_class, "token_class is required"),
        ("missing_merge_fate", candidate.merge_fate, "merge_fate is required"),
    )
    for reason, value, detail in required:
        if not value:
            reject(reason, detail)
    if (
        candidate.resource_class
        and candidate.resource_class.casefold() not in RESOURCE_CLASSES
    ):
        reject(
            "invalid_resource_class",
            f"unsupported resource class {candidate.resource_class!r}",
        )
    if candidate.token_class and candidate.token_class.casefold() not in TOKEN_CLASSES:
        reject("invalid_token_class", f"unsupported token class {candidate.token_class!r}")
    if candidate.predicted_path_breadth > policy.max_predicted_paths:
        reject(
            "predicted_path_breadth",
            f"{candidate.predicted_path_breadth} paths exceed {policy.max_predicted_paths}",
        )
    if candidate.predicted_symbol_breadth > policy.max_predicted_symbols:
        reject(
            "predicted_symbol_breadth",
            f"{candidate.predicted_symbol_breadth} symbols exceed {policy.max_predicted_symbols}",
        )
    if len(candidate.acceptance) > policy.max_acceptance_criteria:
        reject(
            "acceptance_breadth",
            f"{len(candidate.acceptance)} criteria exceed {policy.max_acceptance_criteria}",
        )
    if len(candidate.effects) > policy.max_effects:
        reject(
            "effect_breadth",
            f"{len(candidate.effects)} effects exceed {policy.max_effects}",
        )
    if len(candidate.evidence_subset) > policy.max_evidence_items:
        reject(
            "evidence_breadth",
            f"{len(candidate.evidence_subset)} evidence items exceed {policy.max_evidence_items}",
        )
    if len(candidate.context_paths) > policy.max_context_paths:
        reject(
            "context_path_cost",
            f"{len(candidate.context_paths)} context paths exceed {policy.max_context_paths}",
        )
    if candidate.estimated_context_tokens > policy.max_context_tokens:
        reject(
            "context_token_cost",
            f"{candidate.estimated_context_tokens} exceeds {policy.max_context_tokens}",
        )
    if candidate.estimated_validation_seconds > policy.max_validation_seconds:
        reject(
            "validation_cost",
            f"{candidate.estimated_validation_seconds}s exceeds {policy.max_validation_seconds}s",
        )
    if candidate.estimated_tokens > policy.max_estimated_tokens:
        reject(
            "task_token_cost",
            f"{candidate.estimated_tokens} exceeds {policy.max_estimated_tokens}",
        )
    if len(candidate.dependencies) > policy.max_dependencies:
        reject(
            "dependency_cost",
            f"{len(candidate.dependencies)} dependencies exceed {policy.max_dependencies}",
        )
    if len(candidate.conflicts) > policy.max_conflicts:
        reject(
            "conflict_cost",
            f"{len(candidate.conflicts)} conflicts exceed {policy.max_conflicts}",
        )
    if score.duplicate_similarity >= policy.duplicate_similarity_threshold:
        reject(
            "historical_duplicate",
            f"similarity {score.duplicate_similarity:.6f} reaches "
            f"{policy.duplicate_similarity_threshold:.6f}",
        )
    if score.failure_similarity >= policy.failure_similarity_threshold:
        reject(
            "historical_failure",
            f"similarity {score.failure_similarity:.6f} reaches "
            f"{policy.failure_similarity_threshold:.6f}",
        )
    if score.total < policy.min_quality_score:
        reject(
            "quality_below_threshold",
            f"score {score.total:.6f} is below {policy.min_quality_score:.6f}",
        )
    return tuple(reasons)


def admit_task_candidate(
    candidate: TaskCandidate | Mapping[str, Any],
    *,
    policy: TaskQualityPolicy | None = None,
    historical_tasks: Iterable[Any] = (),
    historical_failures: Iterable[Any] = (),
) -> TaskAdmissionDecision:
    item = (
        candidate if isinstance(candidate, TaskCandidate) else TaskCandidate.from_mapping(candidate)
    )
    selected = policy or TaskQualityPolicy()
    score = score_task_candidate(
        item,
        policy=selected,
        historical_tasks=historical_tasks,
        historical_failures=historical_failures,
    )
    rejections = _task_rejections(item, score, selected)
    return TaskAdmissionDecision(
        candidate=item,
        score=score,
        status=(
            TaskAdmissionStatus.REJECTED
            if rejections
            else TaskAdmissionStatus.ACCEPTED
        ),
        rejections=rejections,
        source_identities=(item.semantic_identity,),
    )


def is_over_broad(
    candidate: TaskCandidate,
    policy: TaskQualityPolicy | None = None,
) -> bool:
    selected = policy or TaskQualityPolicy()
    return (
        candidate.predicted_path_breadth > selected.max_predicted_paths
        or candidate.predicted_symbol_breadth > selected.max_predicted_symbols
        or len(candidate.acceptance) > selected.max_acceptance_criteria
        or len(candidate.effects) > selected.max_effects
        or len(candidate.evidence_subset) > selected.max_evidence_items
        or len(candidate.context_paths) > selected.max_context_paths
        or candidate.estimated_context_tokens > selected.max_context_tokens
        or candidate.estimated_tokens > selected.max_estimated_tokens
    )


def is_tiny(candidate: TaskCandidate, policy: TaskQualityPolicy | None = None) -> bool:
    selected = policy or TaskQualityPolicy()
    return (
        candidate.predicted_path_breadth <= selected.tiny_max_paths
        and candidate.predicted_symbol_breadth <= selected.tiny_max_symbols
        and candidate.estimated_tokens <= selected.tiny_max_tokens
    )


def split_task_candidate(
    candidate: TaskCandidate | Mapping[str, Any],
    policy: TaskQualityPolicy | None = None,
    *,
    max_paths: int | None = None,
    max_symbols: int | None = None,
) -> tuple[TaskCandidate, ...]:
    """Split over-broad work deterministically while preserving dependencies.

    Every child retains the acceptance/proof contract, validation, goal, merge
    fate, and external prerequisites.  Disjoint children remain independent;
    their source identity is carried by the admission decision instead of
    inventing dependency edges which would collapse critical-path width.
    """

    item = (
        candidate if isinstance(candidate, TaskCandidate) else TaskCandidate.from_mapping(candidate)
    )
    selected = policy or TaskQualityPolicy()
    path_limit = max(1, int(max_paths or selected.max_predicted_paths))
    symbol_limit = max(1, int(max_symbols or selected.max_predicted_symbols))
    paths = tuple(sorted(set(item.outputs) | set(item.predicted_paths)))
    symbols = item.predicted_symbols
    required_parts = max(
        1,
        math.ceil(len(paths) / path_limit),
        math.ceil(len(symbols) / symbol_limit),
        math.ceil(len(item.acceptance) / selected.max_acceptance_criteria),
        math.ceil(len(item.effects) / selected.max_effects),
        math.ceil(len(item.evidence_subset) / selected.max_evidence_items),
        (
            math.ceil(len(item.context_paths) / selected.max_context_paths)
            if selected.max_context_paths
            else 1
        ),
        (
            math.ceil(item.estimated_context_tokens / selected.max_context_tokens)
            if selected.max_context_tokens
            else 1
        ),
        (
            math.ceil(item.estimated_tokens / selected.max_estimated_tokens)
            if selected.max_estimated_tokens
            else 1
        ),
    )
    part_count = min(required_parts, selected.max_split_parts)
    if part_count == 1:
        return (item,)

    path_chunks = [paths[index::part_count] for index in range(part_count)]
    symbol_chunks = [symbols[index::part_count] for index in range(part_count)]
    acceptance_chunks = [
        item.acceptance[index::part_count] for index in range(part_count)
    ]
    effect_chunks = [item.effects[index::part_count] for index in range(part_count)]
    evidence_chunks = [
        item.evidence_subset[index::part_count] for index in range(part_count)
    ]
    context_chunks: list[list[str]] = [[] for _ in range(part_count)]
    path_part = {
        path: index
        for index, chunk in enumerate(path_chunks)
        for path in chunk
    }
    residual_context_index = 0
    for path in item.context_paths:
        target = path_part.get(path)
        if target is None:
            target = residual_context_index % part_count
            residual_context_index += 1
        context_chunks[target].append(path)
    children: list[TaskCandidate] = []
    for index in range(part_count):
        child_paths = tuple(path_chunks[index])
        outputs = tuple(path for path in item.outputs if path in child_paths)
        if not outputs and child_paths:
            outputs = child_paths
        # A task may legitimately need context which is not itself an output.
        # Partition residual context while keeping exact output context with
        # its owner; copying whole directories into every child would leave
        # each split over-broad and collapse independent execution width.
        contexts = _strings(context_chunks[index], paths=True)
        child = replace(
            item,
            title=f"{item.title} [{index + 1}/{part_count}]",
            outputs=outputs,
            predicted_paths=child_paths,
            predicted_symbols=tuple(symbol_chunks[index]),
            acceptance=tuple(acceptance_chunks[index] or item.acceptance),
            effects=tuple(effect_chunks[index] or item.effects),
            evidence_subset=tuple(evidence_chunks[index] or item.evidence_subset),
            dependencies=item.dependencies,
            context_paths=contexts,
            estimated_context_tokens=math.ceil(
                item.estimated_context_tokens / part_count
            ),
            estimated_tokens=math.ceil(item.estimated_tokens / part_count),
            source_id=f"{item.source_id}:split:{index + 1}" if item.source_id else "",
            semantic_identity="",
        )
        children.append(child)
    return tuple(children)


def can_coalesce_tasks(
    left: TaskCandidate | Mapping[str, Any],
    right: TaskCandidate | Mapping[str, Any],
    *,
    policy: TaskQualityPolicy | None = None,
) -> bool:
    """Return whether tiny tasks share every required merge-fate boundary."""

    first = left if isinstance(left, TaskCandidate) else TaskCandidate.from_mapping(left)
    second = right if isinstance(right, TaskCandidate) else TaskCandidate.from_mapping(right)
    selected = policy or TaskQualityPolicy()
    return (
        first.semantic_identity != second.semantic_identity
        and is_tiny(first, selected)
        and is_tiny(second, selected)
        and _normalized_text(first.goal_id) == _normalized_text(second.goal_id)
        and first.context_paths == second.context_paths
        and first.outputs == second.outputs
        and first.validation_commands == second.validation_commands
        and _normalized_text(first.merge_fate) == _normalized_text(second.merge_fate)
        and first.resource_class.casefold() == second.resource_class.casefold()
        and first.token_class.casefold() == second.token_class.casefold()
    )


def coalesce_task_candidates(
    candidates: Iterable[TaskCandidate | Mapping[str, Any]],
    *,
    policy: TaskQualityPolicy | None = None,
) -> TaskCandidate:
    """Coalesce compatible tiny candidates into one task or fail closed."""

    items = tuple(sorted((
        item if isinstance(item, TaskCandidate) else TaskCandidate.from_mapping(item)
        for item in candidates
    ), key=lambda item: item.semantic_identity))
    if not items:
        raise ValueError("at least one task candidate is required")
    selected = policy or TaskQualityPolicy()
    anchor = items[0]
    if any(not can_coalesce_tasks(anchor, item, policy=selected) for item in items[1:]):
        raise ValueError(
            "tiny candidates may coalesce only with shared goal, context, outputs, "
            "validation, resource/token class, and merge fate"
        )

    def union(name: str) -> tuple[str, ...]:
        return _strings(item for value in items for item in getattr(value, name))

    return replace(
        anchor,
        title="; ".join(item.title for item in items if item.title),
        acceptance=union("acceptance"),
        preconditions=union("preconditions"),
        effects=union("effects"),
        evidence_subset=union("evidence_subset"),
        predicted_paths=_strings(
            (
                path
                for item in items
                for path in item.predicted_paths
            ),
            paths=True,
        ),
        predicted_symbols=union("predicted_symbols"),
        dependencies=union("dependencies"),
        conflicts=union("conflicts"),
        resources=union("resources"),
        estimated_context_tokens=max(item.estimated_context_tokens for item in items),
        estimated_validation_seconds=max(
            item.estimated_validation_seconds for item in items
        ),
        estimated_tokens=sum(item.estimated_tokens for item in items),
        source_id=",".join(item.source_id for item in items if item.source_id),
        semantic_identity="",
    )


def _coalesce_tiny_groups(
    candidates: Sequence[TaskCandidate],
    policy: TaskQualityPolicy,
) -> tuple[tuple[TaskCandidate, tuple[str, ...]], ...]:
    remaining = list(sorted(candidates, key=lambda item: item.semantic_identity))
    result: list[tuple[TaskCandidate, tuple[str, ...]]] = []
    while remaining:
        anchor = remaining.pop(0)
        compatible = [
            item for item in remaining if can_coalesce_tasks(anchor, item, policy=policy)
        ]
        if not compatible:
            result.append((anchor, (anchor.semantic_identity,)))
            continue
        group = [anchor, *compatible]
        compatible_ids = {item.semantic_identity for item in compatible}
        remaining = [
            item for item in remaining if item.semantic_identity not in compatible_ids
        ]
        merged = coalesce_task_candidates(group, policy=policy)
        result.append((merged, tuple(item.semantic_identity for item in group)))
    return tuple(result)


def refine_task_candidates(
    candidates: Iterable[TaskCandidate | Mapping[str, Any]],
    *,
    policy: TaskQualityPolicy | None = None,
    existing_tasks: Iterable[Any] = (),
    existing: Iterable[Any] | None = None,
    historical_tasks: Iterable[Any] = (),
    historical_failures: Iterable[Any] = (),
    current_open_work: int = 0,
    open_work_count: int | None = None,
    max_open_work: int | None = None,
) -> TaskAdmissionResult:
    """Resize, semantically deduplicate, score, and pressure-bound candidates."""

    selected = policy or TaskQualityPolicy()
    provided_existing = tuple(existing_tasks)
    if existing is not None:
        if provided_existing:
            raise ValueError("pass either existing_tasks or existing, not both")
        provided_existing = tuple(existing)
    if open_work_count is not None:
        if current_open_work:
            raise ValueError(
                "pass either current_open_work or open_work_count, not both"
            )
        current_open_work = open_work_count
    open_count = _non_negative_int(current_open_work, "current_open_work")
    open_limit = (
        selected.max_open_work
        if max_open_work is None
        else _non_negative_int(max_open_work, "max_open_work")
    )
    raw = tuple(
        item if isinstance(item, TaskCandidate) else TaskCandidate.from_mapping(item)
        for item in candidates
    )
    historical = _history(historical_tasks, default_outcome="accepted")
    existing_history = _history(provided_existing, default_outcome="open")
    failures = _history(historical_failures, default_outcome="failed")
    known = tuple(item.candidate for item in (*existing_history, *historical))

    resized: list[TaskCandidate] = []
    source_map: dict[str, tuple[str, ...]] = {}
    for candidate in sorted(raw, key=lambda item: item.semantic_identity):
        pieces = (
            split_task_candidate(candidate, policy=selected)
            if selected.split_over_broad and is_over_broad(candidate, selected)
            else (candidate,)
        )
        for piece in pieces:
            resized.append(piece)
            source_map[piece.semantic_identity] = (candidate.semantic_identity,)

    grouped = (
        _coalesce_tiny_groups(resized, selected)
        if selected.coalesce_tiny
        else tuple((item, (item.semantic_identity,)) for item in resized)
    )
    candidates_with_sources = sorted(grouped, key=lambda pair: pair[0].semantic_identity)
    decisions: list[TaskAdmissionDecision] = []
    observed_identities = {item.semantic_identity for item in known}
    observed_candidates = list(known)
    capacity = min(selected.max_new_work, max(0, open_limit - open_count))
    accepted_count = 0

    for candidate, coalesced_sources in candidates_with_sources:
        source_identities = tuple(
            source
            for identity in coalesced_sources
            for source in source_map.get(identity, (identity,))
        )
        score = score_task_candidate(
            candidate,
            policy=selected,
            historical_tasks=observed_candidates,
            historical_failures=(item.candidate for item in failures),
        )
        rejections = list(_task_rejections(candidate, score, selected))
        if candidate.semantic_identity in observed_identities:
            rejections.append(
                TaskRejection(
                    "duplicate_semantic_identity",
                    candidate.semantic_identity,
                    "canonical semantic identity already exists",
                )
            )
        if not rejections and accepted_count >= capacity:
            rejections.append(
                TaskRejection(
                    "open_work_limit",
                    candidate.semantic_identity,
                    f"cycle has {capacity} available open-work slots",
                )
            )
        if rejections:
            decisions.append(
                TaskAdmissionDecision(
                    candidate=candidate,
                    score=score,
                    status=TaskAdmissionStatus.REJECTED,
                    rejections=tuple(rejections),
                    source_identities=source_identities,
                )
            )
            observed_identities.add(candidate.semantic_identity)
            observed_candidates.append(candidate)
            continue
        status = TaskAdmissionStatus.ACCEPTED
        if len(source_identities) > 1:
            status = TaskAdmissionStatus.COALESCED
        elif source_identities and source_identities[0] != candidate.semantic_identity:
            status = TaskAdmissionStatus.SPLIT
        decisions.append(
            TaskAdmissionDecision(
                candidate=candidate,
                score=score,
                status=status,
                source_identities=source_identities,
            )
        )
        accepted_count += 1
        observed_identities.add(candidate.semantic_identity)
        observed_candidates.append(candidate)

    return TaskAdmissionResult(
        decisions=tuple(decisions),
        initial_open_work=open_count,
        max_open_work=open_limit,
        candidate_count=len(raw),
    )


def _task_split_refill_material(
    *,
    source_candidate: TaskCandidate,
    policy: TaskQualityPolicy,
    initial_open_work: int,
    first_admission: TaskAdmissionResult,
    refill_admission: TaskAdmissionResult,
    repository_tree: str,
) -> dict[str, Any]:
    return {
        "schema": TASK_SPLIT_REFILL_EVIDENCE_SCHEMA,
        "requirement_id": TASK_SPLIT_REFILL_REQUIREMENT_ID,
        "repository_tree": str(repository_tree),
        "policy_id": policy.policy_id,
        "policy": asdict(policy),
        "source_candidate": source_candidate.to_dict(),
        "initial_open_work": initial_open_work,
        "first_admission": first_admission.to_dict(),
        "refill_admission": refill_admission.to_dict(),
    }


def _task_split_refill_qualifies(material: Mapping[str, Any]) -> bool:
    """Independently reproduce the complete split-then-refill obligation."""

    try:
        source_payload = material.get("source_candidate")
        policy_payload = material.get("policy")
        first_payload = material.get("first_admission")
        refill_payload = material.get("refill_admission")
        if not all(
            isinstance(value, Mapping)
            for value in (
                source_payload,
                policy_payload,
                first_payload,
                refill_payload,
            )
        ):
            return False
        repository_tree = str(material.get("repository_tree") or "").strip()
        if not repository_tree:
            return False
        source = TaskCandidate.from_dict(source_payload)
        policy = TaskQualityPolicy(**dict(policy_payload))
        if str(material.get("policy_id") or "") != policy.policy_id:
            return False
        initial_open_work = _non_negative_int(
            material.get("initial_open_work"), "initial_open_work"
        )
        if not is_over_broad(source, policy):
            return False
        expected_children = split_task_candidate(source, policy=policy)
        if len(expected_children) < 2 or any(
            is_over_broad(child, policy) for child in expected_children
        ):
            return False
        expected_semantic_ids = tuple(
            sorted(child.semantic_identity for child in expected_children)
        )
        expected_task_cids = tuple(
            sorted(child.canonical_task_cid for child in expected_children)
        )
        if (
            len(set(expected_semantic_ids)) != len(expected_semantic_ids)
            or len(set(expected_task_cids)) != len(expected_task_cids)
        ):
            return False

        first = refine_task_candidates(
            (source,),
            policy=policy,
            current_open_work=initial_open_work,
        )
        refill = refine_task_candidates(
            (source,),
            policy=policy,
            existing_tasks=first.accepted,
            current_open_work=first.final_open_work,
        )
        if first.to_dict() != dict(first_payload):
            return False
        if refill.to_dict() != dict(refill_payload):
            return False

        admitted_semantic_ids = tuple(
            sorted(candidate.semantic_identity for candidate in first.accepted)
        )
        admitted_task_cids = tuple(
            sorted(candidate.canonical_task_cid for candidate in first.accepted)
        )
        if (
            admitted_semantic_ids != expected_semantic_ids
            or admitted_task_cids != expected_task_cids
            or any(
                decision.status is not TaskAdmissionStatus.SPLIT
                or decision.source_identities != (source.semantic_identity,)
                for decision in first.decisions
            )
        ):
            return False
        if first.final_open_work != initial_open_work + len(expected_children):
            return False

        # Splitting must cover the complete work surface and retain external
        # prerequisites without creating sibling dependencies.
        if {
            value for child in expected_children for value in child.acceptance
        } != set(source.acceptance):
            return False
        if {
            value for child in expected_children for value in child.effects
        } != set(source.effects):
            return False
        if {
            value for child in expected_children for value in child.evidence_subset
        } != set(source.evidence_subset):
            return False
        if {
            value for child in expected_children for value in child.predicted_paths
        } != set(source.outputs) | set(source.predicted_paths):
            return False
        if {
            value for child in expected_children for value in child.predicted_symbols
        } != set(source.predicted_symbols):
            return False
        if {
            value for child in expected_children for value in child.context_paths
        } != set(source.context_paths):
            return False
        if any(
            child.goal_id != source.goal_id
            or child.preconditions != source.preconditions
            or child.dependencies != source.dependencies
            or child.conflicts != source.conflicts
            or child.resources != source.resources
            or child.resource_class != source.resource_class
            or child.token_class != source.token_class
            or child.validation_commands != source.validation_commands
            or child.merge_fate != source.merge_fate
            for child in expected_children
        ):
            return False

        replay_semantic_ids = tuple(
            sorted(decision.candidate.semantic_identity for decision in refill.decisions)
        )
        duplicate_codes = {
            "duplicate_semantic_identity",
            "historical_duplicate",
        }
        return (
            not refill.accepted
            and replay_semantic_ids == expected_semantic_ids
            and refill.initial_open_work == first.final_open_work
            and refill.final_open_work == first.final_open_work
            and refill.bounded
            and all(
                duplicate_codes.intersection(decision.rejection_reasons)
                for decision in refill.decisions
            )
        )
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True)
class TaskSplitRefillEvidence:
    """Content-addressed proof that broad work cannot duplicate on refill."""

    repository_tree: str
    policy_id: str
    policy: Mapping[str, Any]
    source_candidate: Mapping[str, Any]
    initial_open_work: int
    first_admission: Mapping[str, Any]
    refill_admission: Mapping[str, Any]
    evidence_id: str
    integrity_digest: str
    _producer_seal: Any = field(default=None, compare=False, repr=False)

    @classmethod
    def create(
        cls,
        source_candidate: TaskCandidate | Mapping[str, Any],
        *,
        policy: TaskQualityPolicy | None = None,
        initial_open_work: int = 0,
        repository_tree: str = "in-memory",
    ) -> "TaskSplitRefillEvidence":
        """Execute both admission cycles and bind their exact canonical result."""

        source = (
            source_candidate
            if isinstance(source_candidate, TaskCandidate)
            else TaskCandidate.from_mapping(source_candidate)
        )
        selected = policy or TaskQualityPolicy()
        open_work = _non_negative_int(initial_open_work, "initial_open_work")
        first = refine_task_candidates(
            (source,),
            policy=selected,
            current_open_work=open_work,
        )
        refill = refine_task_candidates(
            (source,),
            policy=selected,
            existing_tasks=first.accepted,
            current_open_work=first.final_open_work,
        )
        material = _task_split_refill_material(
            source_candidate=source,
            policy=selected,
            initial_open_work=open_work,
            first_admission=first,
            refill_admission=refill,
            repository_tree=repository_tree,
        )
        digest = hashlib.sha256(_task_quality_evidence_bytes(material)).hexdigest()
        return cls(
            repository_tree=str(repository_tree),
            policy_id=selected.policy_id,
            policy=dict(material["policy"]),
            source_candidate=dict(material["source_candidate"]),
            initial_open_work=open_work,
            first_admission=dict(material["first_admission"]),
            refill_admission=dict(material["refill_admission"]),
            evidence_id=_task_quality_evidence_cid(material),
            integrity_digest=digest,
            _producer_seal=_TASK_SPLIT_REFILL_EVIDENCE_SEAL,
        )

    def _material(self) -> dict[str, Any]:
        return {
            "schema": TASK_SPLIT_REFILL_EVIDENCE_SCHEMA,
            "requirement_id": TASK_SPLIT_REFILL_REQUIREMENT_ID,
            "repository_tree": self.repository_tree,
            "policy_id": self.policy_id,
            "policy": dict(self.policy),
            "source_candidate": dict(self.source_candidate),
            "initial_open_work": self.initial_open_work,
            "first_admission": dict(self.first_admission),
            "refill_admission": dict(self.refill_admission),
        }

    def verify_integrity(self) -> bool:
        material = self._material()
        return (
            self.integrity_digest
            == hashlib.sha256(_task_quality_evidence_bytes(material)).hexdigest()
            and self.evidence_id == _task_quality_evidence_cid(material)
        )

    @property
    def proved_requirement_ids(self) -> tuple[str, ...]:
        if (
            self._producer_seal is _TASK_SPLIT_REFILL_EVIDENCE_SEAL
            and self.verify_integrity()
            and _task_split_refill_qualifies(self._material())
        ):
            return (TASK_SPLIT_REFILL_REQUIREMENT_ID,)
        return ()

    def to_dict(self) -> dict[str, Any]:
        material = self._material()
        qualifies = bool(self.proved_requirement_ids)
        material.update(
            {
                "evidence_id": self.evidence_id,
                "integrity_digest": self.integrity_digest,
                "proved_requirement_ids": list(self.proved_requirement_ids),
                "status": "passed" if qualifies else "diagnostic",
                "complete": qualifies,
                "coverage_complete": qualifies,
                "source_tier": "validation",
            }
        )
        return material

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TaskSplitRefillEvidence":
        """Restore integrity-checked diagnostics without producer authority."""

        evidence = cls(
            repository_tree=str(payload.get("repository_tree") or ""),
            policy_id=str(payload.get("policy_id") or ""),
            policy=dict(payload.get("policy") or {}),
            source_candidate=dict(payload.get("source_candidate") or {}),
            initial_open_work=_non_negative_int(
                payload.get("initial_open_work"), "initial_open_work"
            ),
            first_admission=dict(payload.get("first_admission") or {}),
            refill_admission=dict(payload.get("refill_admission") or {}),
            evidence_id=str(payload.get("evidence_id") or ""),
            integrity_digest=str(payload.get("integrity_digest") or ""),
            _producer_seal=None,
        )
        if not evidence.verify_integrity():
            raise ValueError("task split/refill evidence digest mismatch")
        return evidence


def prove_task_split_refill(
    source_candidate: TaskCandidate | Mapping[str, Any],
    *,
    policy: TaskQualityPolicy | None = None,
    initial_open_work: int = 0,
    repository_tree: str = "in-memory",
) -> TaskSplitRefillEvidence:
    """Produce the authoritative two-cycle split/refill validation receipt."""

    return TaskSplitRefillEvidence.create(
        source_candidate,
        policy=policy,
        initial_open_work=initial_open_work,
        repository_tree=repository_tree,
    )


# Compatibility spellings used by existing proposal evaluators and early ASI
# design notes.
TaskQualityResult = TaskAdmissionResult
TaskRefinementResult = TaskAdmissionResult
TaskQualityRejection = TaskRejection
canonical_task_semantic_identity = canonical_semantic_identity
coalesce_tasks = coalesce_task_candidates
split_over_broad_candidate = split_task_candidate
evaluate_task_candidates = refine_task_candidates


__all__ = [
    "RESOURCE_CLASSES",
    "TASK_SPLIT_REFILL_EVIDENCE_SCHEMA",
    "TASK_SPLIT_REFILL_REQUIREMENT_ID",
    "TASK_QUALITY_EVALUATOR_VERSION",
    "TASK_QUALITY_SCHEMA",
    "TASK_SEMANTIC_IDENTITY_SCHEMA",
    "TOKEN_CLASSES",
    "TOKEN_CLASS_LIMITS",
    "HistoricalTask",
    "TaskAdmissionDecision",
    "TaskAdmissionResult",
    "TaskAdmissionStatus",
    "TaskCandidate",
    "TaskQualityPolicy",
    "TaskQualityRejection",
    "TaskQualityResult",
    "TaskQualityScore",
    "TaskRefinementResult",
    "TaskRejection",
    "TaskSplitRefillEvidence",
    "admit_task_candidate",
    "can_coalesce_tasks",
    "canonical_semantic_identity",
    "canonical_task_semantic_identity",
    "coalesce_task_candidates",
    "coalesce_tasks",
    "evaluate_task_candidates",
    "is_over_broad",
    "is_tiny",
    "prove_task_split_refill",
    "refine_task_candidates",
    "score_task_candidate",
    "split_over_broad_candidate",
    "split_task_candidate",
    "task_semantic_similarity",
]
