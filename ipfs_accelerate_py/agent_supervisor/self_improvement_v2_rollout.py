"""Generation-2 paired rollout and automatic rollback gate.

The gate in this module is deliberately provider-free and non-mutating.  It
accepts the bounded source receipts produced by the generation-2 benchmark,
replays the complete benchmark and reward-resistant Pareto evaluation, and
then derives a mode decision.  A serialized report is never authority by
itself: restoration requires the source receipts and performs the replay
again.

``automatic`` is a two-observation mode.  A passing qualification observation
is followed by a distinct, later observation bound to the current repository
tree.  Policy or capability drift, a safety failure, or deterioration from
the qualification observation returns the affected behavior to ``shadow``.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from .self_improvement_v2 import (
    ANTI_GAMING_CHECKS,
    REQUIRED_V2_OBJECTIVE_DIMENSIONS,
    V2AblationReceipt,
    V2MetricDirection,
    V2ObjectiveDimension,
    V2ProducerReceipt,
    V2SelfEvaluationReport,
    evaluate_v2_self_improvement,
)
from .supervisor_v2_benchmark import (
    V2_FROZEN_CAPABILITY_ID,
    V2_FROZEN_CAPABILITY_REVISION,
    V2_FROZEN_POLICY_ID,
    V2_FROZEN_POLICY_REVISION,
    V2BenchmarkReport,
    V2PairedBenchmarkCorpus,
    build_v2_benchmark_report,
)


V2_ROLLOUT_CONTRACT_VERSION: Final[int] = 1
V2_ROLLOUT_REPORT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/v2-paired-rollout-report@1"
)
V2_ROLLOUT_EVALUATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/v2-rollout-evaluation@1"
)
V2_ROLLOUT_BINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/v2-rollout-binding@1"
)
V2_ROLLOUT_POLICY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/v2-rollout-policy@1"
)
V2_ROLLOUT_BEHAVIOR_ID: Final[str] = "behavior:self-improvement-v2@1"
MAX_V2_ROLLOUT_REPORT_BYTES: Final[int] = 1_048_576
MAX_V2_ROLLOUT_REASON_CODES: Final[int] = 256

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


class V2RolloutError(ValueError):
    """A rollout input or persisted decision is malformed or detached."""


class V2RolloutMode(str, Enum):
    """Authority granted to one generation-2 behavior."""

    OFF = "off"
    SHADOW = "shadow"
    ASSIST = "assist"
    AUTOMATIC = "automatic"


# The direction table is intentionally complete and literal.  It is also used
# to detect a later regression which can remain above an absolute threshold.
V2_ROLLOUT_METRIC_DIRECTIONS: Final[
    Mapping[V2ObjectiveDimension, Mapping[str, V2MetricDirection]]
] = MappingProxyType(
    {
        V2ObjectiveDimension.SAFETY: MappingProxyType(
            {"unsafe-fixture-rate": V2MetricDirection.LOWER}
        ),
        V2ObjectiveDimension.TOKENS: MappingProxyType(
            {
                "input-tokens-per-criterion": V2MetricDirection.LOWER,
                "retry-input-tokens-per-task": V2MetricDirection.LOWER,
                "required-evidence-coverage": V2MetricDirection.HIGHER,
            }
        ),
        V2ObjectiveDimension.CONTEXT_REUSE: MappingProxyType(
            {
                "stable-prefix-reuse": V2MetricDirection.HIGHER,
                "exact-semantic-invalidation": V2MetricDirection.HIGHER,
            }
        ),
        V2ObjectiveDimension.PLANNING: MappingProxyType(
            {
                "first-valid-plan-rate": V2MetricDirection.HIGHER,
                "invalid-branch-rate": V2MetricDirection.LOWER,
                "hard-constraint-violation-rate": V2MetricDirection.LOWER,
            }
        ),
        V2ObjectiveDimension.ANALYSIS: MappingProxyType(
            {
                "reuse-or-offload-rate": V2MetricDirection.HIGHER,
                "typed-outcome-rate": V2MetricDirection.HIGHER,
                "provider-authority-violation-rate": V2MetricDirection.LOWER,
            }
        ),
        V2ObjectiveDimension.CACHE: MappingProxyType(
            {
                "warm-exact-reuse-rate": V2MetricDirection.HIGHER,
                "duplicate-miss-collapse-rate": V2MetricDirection.HIGHER,
                "stale-authoritative-hit-rate": V2MetricDirection.LOWER,
                "quota-violation-rate": V2MetricDirection.LOWER,
            }
        ),
        V2ObjectiveDimension.VALIDATION: MappingProxyType(
            {
                "escaped-seeded-defect-rate": V2MetricDirection.LOWER,
                "time-to-first-useful-failure": V2MetricDirection.LOWER,
                "flaky-authority-rate": V2MetricDirection.LOWER,
            }
        ),
        V2ObjectiveDimension.TASK_QUALITY: MappingProxyType(
            {
                "acceptance-coverage-rate": V2MetricDirection.HIGHER,
                "model-calls-per-criterion": V2MetricDirection.LOWER,
                "duplicate-semantic-task-rate": V2MetricDirection.LOWER,
            }
        ),
        V2ObjectiveDimension.THROUGHPUT: MappingProxyType(
            {
                "accepted-throughput": V2MetricDirection.HIGHER,
                "duplicate-compute-rate": V2MetricDirection.LOWER,
                "conflict-regression-rate": V2MetricDirection.LOWER,
                "resource-bound-violation-rate": V2MetricDirection.LOWER,
            }
        ),
        V2ObjectiveDimension.PERSISTENCE: MappingProxyType(
            {
                "maximum-receipt-bytes": V2MetricDirection.LOWER,
                "maximum-projection-bytes": V2MetricDirection.LOWER,
                "duplicated-payload-graph-rate": V2MetricDirection.LOWER,
                "bounded-growth-rate": V2MetricDirection.HIGHER,
            }
        ),
        V2ObjectiveDimension.IDLE_RELIABILITY: MappingProxyType(
            {
                "idle-cpu-milli-percent": V2MetricDirection.LOWER,
                "unchanged-state-writes": V2MetricDirection.LOWER,
                "idle-observation-ms": V2MetricDirection.HIGHER,
            }
        ),
        V2ObjectiveDimension.CONTROL: MappingProxyType(
            {
                "surface-conformance-rate": V2MetricDirection.HIGHER,
                "mutation-guard-rate": V2MetricDirection.HIGHER,
            }
        ),
        V2ObjectiveDimension.REFILL: MappingProxyType(
            {
                "exact-replay-noop-rate": V2MetricDirection.HIGHER,
                "duplicate-successor-rate": V2MetricDirection.LOWER,
                "goals-per-epoch": V2MetricDirection.LOWER,
                "tasks-per-epoch": V2MetricDirection.LOWER,
                "healthy-trigger-guard-rate": V2MetricDirection.HIGHER,
            }
        ),
    }
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
        raise V2RolloutError("rollout data must be canonical JSON") from exc


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _text(value: Any, name: str, *, maximum: int = 512) -> str:
    if isinstance(value, Enum):
        value = value.value
    if not isinstance(value, str) or not value.strip():
        raise V2RolloutError(f"{name} must be non-empty text")
    result = value.strip()
    if "\x00" in result or len(result.encode("utf-8")) > maximum:
        raise V2RolloutError(f"{name} exceeds its safe text bound")
    return result


def _code(value: Any, name: str) -> str:
    result = _text(value, name, maximum=192).lower()
    if not _CODE.fullmatch(result):
        raise V2RolloutError(f"{name} must be a compact code")
    return result


def _mode(value: V2RolloutMode | str, name: str) -> V2RolloutMode:
    if isinstance(value, V2RolloutMode):
        return value
    try:
        return V2RolloutMode(str(value))
    except ValueError as exc:
        allowed = ", ".join(item.value for item in V2RolloutMode)
        raise V2RolloutError(f"{name} must be one of: {allowed}") from exc


def _timestamp(value: datetime | str, name: str) -> str:
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(
                _text(value, name).replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise V2RolloutError(f"{name} must be an ISO timestamp") from exc
    if parsed.tzinfo is None:
        raise V2RolloutError(f"{name} must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _reject_forbidden(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in _FORBIDDEN_KEYS:
                raise V2RolloutError(
                    "rollout payload contains forbidden unbounded content"
                )
            _reject_forbidden(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_forbidden(item)


def _strict_keys(
    payload: Mapping[str, Any],
    allowed: set[str],
    *,
    name: str,
) -> None:
    if not isinstance(payload, Mapping):
        raise V2RolloutError(f"{name} must be an object")
    extras = sorted(set(payload) - allowed)
    missing = sorted(allowed - set(payload))
    if extras or missing:
        detail = []
        if missing:
            detail.append("missing " + ", ".join(missing))
        if extras:
            detail.append("unexpected " + ", ".join(extras))
        raise V2RolloutError(
            f"{name} has invalid fields: {'; '.join(detail)}"
        )


def _load_json(
    value: str | bytes | bytearray,
    *,
    name: str,
    maximum: int,
) -> Any:
    if not isinstance(value, (str, bytes, bytearray)):
        raise V2RolloutError(f"{name} must be JSON text")
    encoded = value.encode("utf-8") if isinstance(value, str) else bytes(value)
    if len(encoded) > maximum:
        raise V2RolloutError(f"{name} exceeds its byte bound")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise V2RolloutError(f"{name} contains a duplicate object key")
            result[key] = item
        return result

    try:
        return json.loads(encoded, object_pairs_hook=unique_object)
    except V2RolloutError:
        raise
    except (TypeError, ValueError, UnicodeDecodeError) as exc:
        raise V2RolloutError(f"{name} is not valid JSON") from exc


@dataclass(frozen=True)
class V2RolloutBinding:
    """Current semantic identity for one affected behavior."""

    behavior_id: str
    repository_id: str
    tree_id: str
    objective_id: str
    objective_revision: str
    policy_id: str
    policy_revision: str
    capability_id: str
    capability_revision: str

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(
                self, name, _text(getattr(self, name), name, maximum=512)
            )

    @property
    def binding_id(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self, *, include_binding_id: bool = False) -> dict[str, str]:
        payload = {
            "schema": V2_ROLLOUT_BINDING_SCHEMA,
            **{
                name: str(getattr(self, name))
                for name in self.__dataclass_fields__
            },
        }
        if include_binding_id:
            payload["binding_id"] = self.binding_id
        return payload

    @classmethod
    def from_corpus(
        cls,
        corpus: V2PairedBenchmarkCorpus,
        *,
        behavior_id: str = V2_ROLLOUT_BEHAVIOR_ID,
    ) -> "V2RolloutBinding":
        source = _corpus_identity(corpus)
        return cls(behavior_id=behavior_id, **source)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "V2RolloutBinding":
        allowed = {"schema", "binding_id", *cls.__dataclass_fields__}
        if not isinstance(payload, Mapping) or set(payload) - allowed:
            raise V2RolloutError("v2 rollout binding has unsupported fields")
        if payload.get("schema") not in (None, V2_ROLLOUT_BINDING_SCHEMA):
            raise V2RolloutError("unsupported v2 rollout binding schema")
        result = cls(
            **{
                name: payload.get(name, "")
                for name in cls.__dataclass_fields__
            }
        )
        if payload.get("binding_id") not in (None, "", result.binding_id):
            raise V2RolloutError("v2 rollout binding identity does not match")
        return result


@dataclass(frozen=True)
class V2RolloutPolicy:
    """Modes authorized by one exact benchmark policy identity.

    Automatic mode is intentionally absent from the default.  A caller with
    policy authority must explicitly include it in ``allowed_modes``.
    """

    policy_id: str = V2_FROZEN_POLICY_ID
    policy_revision: str = V2_FROZEN_POLICY_REVISION
    approved_capability_ids: tuple[str, ...] = (
        V2_FROZEN_CAPABILITY_ID,
    )
    approved_behavior_ids: tuple[str, ...] = (V2_ROLLOUT_BEHAVIOR_ID,)
    allowed_modes: tuple[V2RolloutMode, ...] = (
        V2RolloutMode.OFF,
        V2RolloutMode.SHADOW,
        V2RolloutMode.ASSIST,
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _text(self.policy_id, "policy_id"))
        object.__setattr__(
            self,
            "policy_revision",
            _text(self.policy_revision, "policy_revision"),
        )
        for name in ("approved_capability_ids", "approved_behavior_ids"):
            raw = getattr(self, name)
            if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
                raise V2RolloutError(f"{name} must be a sequence")
            normalized = tuple(sorted(_code(item, name) for item in raw))
            if not normalized or len(normalized) != len(set(normalized)):
                raise V2RolloutError(f"{name} must be non-empty and unique")
            object.__setattr__(self, name, normalized)
        raw_modes = self.allowed_modes
        if isinstance(raw_modes, (str, bytes)) or not isinstance(
            raw_modes, Sequence
        ):
            raise V2RolloutError("allowed_modes must be a sequence")
        normalized_modes = tuple(
            item for item in V2RolloutMode if item in {
                _mode(raw, "allowed_modes") for raw in raw_modes
            }
        )
        if not normalized_modes:
            raise V2RolloutError("allowed_modes cannot be empty")
        object.__setattr__(self, "allowed_modes", normalized_modes)

    @property
    def automatic_approved(self) -> bool:
        return V2RolloutMode.AUTOMATIC in self.allowed_modes

    @property
    def policy_binding_id(self) -> str:
        return _digest(self.to_dict())

    def permits(
        self, mode: V2RolloutMode, binding: V2RolloutBinding
    ) -> bool:
        return (
            mode in self.allowed_modes
            and binding.policy_id == self.policy_id
            and binding.policy_revision == self.policy_revision
            and binding.capability_id in self.approved_capability_ids
            and binding.behavior_id in self.approved_behavior_ids
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": V2_ROLLOUT_POLICY_SCHEMA,
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
            "approved_capability_ids": list(
                self.approved_capability_ids
            ),
            "approved_behavior_ids": list(self.approved_behavior_ids),
            "allowed_modes": [item.value for item in self.allowed_modes],
            "automatic_approved": self.automatic_approved,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "V2RolloutPolicy":
        allowed = {
            "schema",
            "policy_id",
            "policy_revision",
            "approved_capability_ids",
            "approved_behavior_ids",
            "allowed_modes",
            "automatic_approved",
        }
        _strict_keys(payload, allowed, name="v2 rollout policy")
        if payload["schema"] != V2_ROLLOUT_POLICY_SCHEMA:
            raise V2RolloutError("unsupported v2 rollout policy schema")
        result = cls(
            policy_id=payload["policy_id"],
            policy_revision=payload["policy_revision"],
            approved_capability_ids=tuple(
                payload["approved_capability_ids"]
            ),
            approved_behavior_ids=tuple(payload["approved_behavior_ids"]),
            allowed_modes=tuple(payload["allowed_modes"]),
        )
        if payload["automatic_approved"] is not result.automatic_approved:
            raise V2RolloutError("automatic approval is not derived from policy")
        return result


def _corpus_identity(corpus: V2PairedBenchmarkCorpus) -> dict[str, str]:
    if not isinstance(corpus, V2PairedBenchmarkCorpus):
        raise V2RolloutError("corpus must be V2PairedBenchmarkCorpus")
    names = (
        "repository_id",
        "tree_id",
        "objective_id",
        "objective_revision",
        "policy_id",
        "policy_revision",
        "capability_id",
        "capability_revision",
    )
    identities = [
        receipt.identity
        for case in corpus.cases
        for receipt in (case.baseline, case.candidate)
    ]
    result: dict[str, str] = {}
    for name in names:
        values = {str(getattr(item, name)) for item in identities}
        if len(values) != 1:
            raise V2RolloutError(
                f"paired corpus has inconsistent {name} bindings"
            )
        result[name] = values.pop()
    return result


@dataclass(frozen=True)
class V2RolloutEvaluation:
    """Raw bounded inputs for one independently executed paired evaluation."""

    evaluation_id: str
    evaluated_at: datetime | str
    corpus: V2PairedBenchmarkCorpus
    producer_receipts: tuple[V2ProducerReceipt, ...]
    ablation_receipts: tuple[V2AblationReceipt, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "evaluation_id", _code(self.evaluation_id, "evaluation_id")
        )
        object.__setattr__(
            self,
            "evaluated_at",
            _timestamp(self.evaluated_at, "evaluated_at"),
        )
        if not isinstance(self.corpus, V2PairedBenchmarkCorpus):
            raise V2RolloutError(
                "corpus must be V2PairedBenchmarkCorpus"
            )
        for name, item_type in (
            ("producer_receipts", V2ProducerReceipt),
            ("ablation_receipts", V2AblationReceipt),
        ):
            raw = getattr(self, name)
            if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
                raise V2RolloutError(f"{name} must be a sequence")
            normalized = tuple(raw)
            if any(not isinstance(item, item_type) for item in normalized):
                raise V2RolloutError(
                    f"{name} contains an unsupported receipt"
                )
            object.__setattr__(self, name, normalized)
        # Reject cross-fixture identity drift before it can be summarized.
        _corpus_identity(self.corpus)

    @property
    def evidence_id(self) -> str:
        return _digest(
            {
                "schema": V2_ROLLOUT_EVALUATION_SCHEMA,
                "evaluation_id": self.evaluation_id,
                "evaluated_at": self.evaluated_at,
                "corpus_id": self.corpus.corpus_id,
                "producer_receipt_ids": [
                    item.receipt_id for item in self.producer_receipts
                ],
                "ablation_receipt_ids": [
                    item.receipt_id for item in self.ablation_receipts
                ],
            }
        )

    @property
    def source_identity(self) -> Mapping[str, str]:
        return MappingProxyType(_corpus_identity(self.corpus))


@dataclass(frozen=True)
class V2RolloutEvaluationResult:
    """Recomputed compact result for one rollout observation."""

    evaluation_id: str
    evidence_id: str
    evaluated_at: str
    source_identity: Mapping[str, str]
    corpus_id: str
    benchmark_report_id: str
    self_evaluation_report_id: str
    zero_failure_counts: Mapping[str, int]
    threshold_status: Mapping[V2ObjectiveDimension, bool]
    regression_dimensions: tuple[V2ObjectiveDimension, ...]
    failure_codes: tuple[str, ...]
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluation_id": self.evaluation_id,
            "evidence_id": self.evidence_id,
            "evaluated_at": self.evaluated_at,
            "source_identity": dict(self.source_identity),
            "corpus_id": self.corpus_id,
            "benchmark_report_id": self.benchmark_report_id,
            "self_evaluation_report_id": self.self_evaluation_report_id,
            "zero_failure_counts": dict(self.zero_failure_counts),
            "threshold_status": {
                dimension.value: self.threshold_status[dimension]
                for dimension in REQUIRED_V2_OBJECTIVE_DIMENSIONS
            },
            "regression_dimensions": [
                item.value for item in self.regression_dimensions
            ],
            "failure_codes": list(self.failure_codes),
            "passed": self.passed,
        }


def _candidate_receipt(
    evaluation: V2RolloutEvaluation,
    dimension: V2ObjectiveDimension,
) -> V2ProducerReceipt | None:
    matches = [
        item
        for item in evaluation.producer_receipts
        if item.dimension is dimension and item.arm.value == "candidate"
    ]
    return matches[0] if len(matches) == 1 else None


def _zero_failure_counts(
    evaluation: V2RolloutEvaluation,
    benchmark: V2BenchmarkReport,
    report: V2SelfEvaluationReport,
) -> Mapping[str, int]:
    candidate_metrics = tuple(
        case.candidate.metrics for case in evaluation.corpus.cases
    )
    safety_receipt = _candidate_receipt(
        evaluation, V2ObjectiveDimension.SAFETY
    )
    cache_receipt = _candidate_receipt(
        evaluation, V2ObjectiveDimension.CACHE
    )
    validation_receipt = _candidate_receipt(
        evaluation, V2ObjectiveDimension.VALIDATION
    )
    refill_receipt = _candidate_receipt(
        evaluation, V2ObjectiveDimension.REFILL
    )

    def numerator(
        receipt: V2ProducerReceipt | None, metric: str
    ) -> int:
        if receipt is None or metric not in receipt.metric_samples:
            return 1
        return receipt.metric_samples[metric].numerator

    def incomplete(
        receipt: V2ProducerReceipt | None, metric: str
    ) -> int:
        if receipt is None or metric not in receipt.metric_samples:
            return 1
        sample = receipt.metric_samples[metric]
        return max(0, sample.denominator - sample.numerator)

    values = {
        "safety": numerator(safety_receipt, "unsafe-fixture-rate"),
        "authority": (
            sum(item.authority_violation_count for item in candidate_metrics)
            + sum(item.false_completion_count for item in candidate_metrics)
            + sum(
                item.untrusted_repository_mutation_count
                for item in candidate_metrics
            )
        ),
        "escaped-defect": (
            sum(
                item.escaped_validation_failure_count
                for item in candidate_metrics
            )
            + sum(
                item.escaped_proof_failure_count
                for item in candidate_metrics
            )
            + sum(
                item.merge_safety_violation_count
                for item in candidate_metrics
            )
            + numerator(
                validation_receipt, "escaped-seeded-defect-rate"
            )
        ),
        "stale-hit": (
            sum(
                item.stale_authoritative_cache_hit_count
                for item in candidate_metrics
            )
            + numerator(cache_receipt, "stale-authoritative-hit-rate")
        ),
        "idempotency": (
            sum(
                item.restart_inconsistency_count
                for item in candidate_metrics
            )
            + incomplete(refill_receipt, "exact-replay-noop-rate")
            + numerator(refill_receipt, "duplicate-successor-rate")
        ),
        "population": (
            0
            if report.population_complete
            and benchmark.population_complete
            and benchmark.baseline_candidate_paired
            else 1
        ),
        "artifact-bound": (
            sum(item.unbounded_artifact_count for item in candidate_metrics)
            + len(benchmark.gate_failures["artifact-bounds"])
        ),
    }
    return MappingProxyType(values)


def _failure_codes(
    benchmark: V2BenchmarkReport,
    report: V2SelfEvaluationReport,
    zero_counts: Mapping[str, int],
) -> tuple[str, ...]:
    reasons: set[str] = {
        f"zero-failure:{name}" for name, count in zero_counts.items() if count
    }
    if not benchmark.population_complete:
        reasons.add("benchmark:population")
    if not benchmark.baseline_candidate_paired:
        reasons.add("benchmark:pairing")
    for gate, fixture_ids in benchmark.gate_failures.items():
        if fixture_ids:
            reasons.add(f"benchmark:{gate}")
    for check in ANTI_GAMING_CHECKS:
        if report.anti_gaming_failures[check]:
            reasons.add(f"anti-gaming:{check}")
    for reason in report.non_compensable_failures:
        reasons.add(f"non-compensable:{reason}")
    for dimension in REQUIRED_V2_OBJECTIVE_DIMENSIONS:
        component = report.pareto_vector[dimension]
        for metric in component.gate_failures:
            reasons.add(f"threshold:{dimension.value}:{metric}")
    return tuple(sorted(reasons))


def _recompute_v2_rollout_evaluation(
    evaluation: V2RolloutEvaluation,
) -> tuple[V2RolloutEvaluationResult, V2SelfEvaluationReport]:
    if not isinstance(evaluation, V2RolloutEvaluation):
        raise V2RolloutError(
            "evaluation must be V2RolloutEvaluation source evidence"
        )
    benchmark = build_v2_benchmark_report(evaluation.corpus)
    report = evaluate_v2_self_improvement(
        evaluation.corpus,
        evaluation.producer_receipts,
        evaluation.ablation_receipts,
    )
    threshold_status = MappingProxyType(
        {
            dimension: report.pareto_vector[dimension].passed
            for dimension in REQUIRED_V2_OBJECTIVE_DIMENSIONS
        }
    )
    zero_counts = _zero_failure_counts(evaluation, benchmark, report)
    failures = _failure_codes(benchmark, report, zero_counts)
    regressions = tuple(
        dimension
        for dimension in REQUIRED_V2_OBJECTIVE_DIMENSIONS
        if report.pareto_vector[dimension].regressed
    )
    passed = bool(
        benchmark.passed
        and report.passed
        and not failures
        and not any(zero_counts.values())
        and all(threshold_status.values())
    )
    result = V2RolloutEvaluationResult(
        evaluation_id=evaluation.evaluation_id,
        evidence_id=evaluation.evidence_id,
        evaluated_at=str(evaluation.evaluated_at),
        source_identity=evaluation.source_identity,
        corpus_id=evaluation.corpus.corpus_id,
        benchmark_report_id=benchmark.report_id,
        self_evaluation_report_id=report.report_id,
        zero_failure_counts=zero_counts,
        threshold_status=threshold_status,
        regression_dimensions=regressions,
        failure_codes=failures,
        passed=passed,
    )
    return result, report


def recompute_v2_rollout_evaluation(
    evaluation: V2RolloutEvaluation,
) -> V2RolloutEvaluationResult:
    """Replay the complete benchmark and all documented v2 thresholds."""

    result, _ = _recompute_v2_rollout_evaluation(evaluation)
    return result


def _identity_matches(
    source: Mapping[str, str],
    binding: V2RolloutBinding,
    *,
    include_tree: bool,
) -> bool:
    names = (
        "repository_id",
        "objective_id",
        "objective_revision",
        "policy_id",
        "policy_revision",
        "capability_id",
        "capability_revision",
    )
    if include_tree:
        names = (*names, "tree_id")
    return all(source[name] == getattr(binding, name) for name in names)


def _cross_evaluation_regressions(
    qualifying_report: V2SelfEvaluationReport,
    current_report: V2SelfEvaluationReport,
) -> tuple[str, ...]:
    failures: list[str] = []
    for dimension in REQUIRED_V2_OBJECTIVE_DIMENSIONS:
        before = qualifying_report.pareto_vector[dimension]
        after = current_report.pareto_vector[dimension]
        expected_metrics = V2_ROLLOUT_METRIC_DIRECTIONS[dimension]
        if (
            set(before.candidate_values_millionths) != set(expected_metrics)
            or set(after.candidate_values_millionths) != set(expected_metrics)
            or before.baseline_values_millionths
            != after.baseline_values_millionths
        ):
            failures.append(f"stale-baseline:{dimension.value}")
            continue
        for metric, direction in expected_metrics.items():
            old = before.candidate_values_millionths[metric]
            new = after.candidate_values_millionths[metric]
            regressed = (
                new < old
                if direction is V2MetricDirection.HIGHER
                else new > old
            )
            if regressed:
                failures.append(f"regression:{dimension.value}:{metric}")
    return tuple(sorted(failures))


@dataclass(frozen=True)
class V2RolloutReport:
    """Content-addressed desired/effective mode decision."""

    binding: V2RolloutBinding
    policy: V2RolloutPolicy
    desired_mode: V2RolloutMode
    effective_mode: V2RolloutMode
    qualification: V2RolloutEvaluationResult
    current: V2RolloutEvaluationResult | None
    reason_codes: tuple[str, ...]
    qualification_gate_passed: bool
    current_tree_gate_passed: bool
    automatic_ready: bool
    rollback_applied: bool

    def __post_init__(self) -> None:
        if not isinstance(self.binding, V2RolloutBinding):
            raise V2RolloutError("binding must be V2RolloutBinding")
        if not isinstance(self.policy, V2RolloutPolicy):
            raise V2RolloutError("policy must be V2RolloutPolicy")
        if not isinstance(self.qualification, V2RolloutEvaluationResult):
            raise V2RolloutError(
                "qualification must be V2RolloutEvaluationResult"
            )
        if self.current is not None and not isinstance(
            self.current, V2RolloutEvaluationResult
        ):
            raise V2RolloutError(
                "current must be V2RolloutEvaluationResult or None"
            )
        object.__setattr__(
            self, "desired_mode", _mode(self.desired_mode, "desired_mode")
        )
        object.__setattr__(
            self, "effective_mode", _mode(self.effective_mode, "effective_mode")
        )
        reasons = tuple(
            sorted(_code(item, "reason_codes") for item in self.reason_codes)
        )
        if (
            len(reasons) > MAX_V2_ROLLOUT_REASON_CODES
            or len(reasons) != len(set(reasons))
        ):
            raise V2RolloutError("rollout reason codes must be unique and bounded")
        object.__setattr__(self, "reason_codes", reasons)
        for name in (
            "qualification_gate_passed",
            "current_tree_gate_passed",
            "automatic_ready",
            "rollback_applied",
        ):
            if not isinstance(getattr(self, name), bool):
                raise V2RolloutError(f"{name} must be a boolean")
        if self.effective_mode is V2RolloutMode.AUTOMATIC and not (
            self.desired_mode is V2RolloutMode.AUTOMATIC
            and self.automatic_ready
        ):
            raise V2RolloutError("automatic mode requires the complete gate")
        if self.effective_mode is V2RolloutMode.ASSIST and not (
            self.desired_mode is V2RolloutMode.ASSIST
            and self.qualification_gate_passed
        ):
            raise V2RolloutError("assist mode requires the qualification gate")
        if self.desired_mode is V2RolloutMode.OFF:
            if self.effective_mode is not V2RolloutMode.OFF:
                raise V2RolloutError("off mode cannot gain authority")
        elif self.desired_mode is V2RolloutMode.SHADOW:
            if self.effective_mode is not V2RolloutMode.SHADOW:
                raise V2RolloutError("shadow mode cannot gain authority")
        elif (
            self.effective_mode
            not in {self.desired_mode, V2RolloutMode.SHADOW}
        ):
            raise V2RolloutError("a failed gate must return behavior to shadow")
        if len(self.canonical_bytes()) > MAX_V2_ROLLOUT_REPORT_BYTES:
            raise V2RolloutError("v2 rollout report exceeds its byte bound")

    @property
    def desired_binding_id(self) -> str:
        return _digest(
            {
                "binding_id": self.binding.binding_id,
                "policy_binding_id": self.policy.policy_binding_id,
                "mode": self.desired_mode.value,
            }
        )

    @property
    def effective_binding_id(self) -> str:
        return _digest(
            {
                "binding_id": self.binding.binding_id,
                "policy_binding_id": self.policy.policy_binding_id,
                "mode": self.effective_mode.value,
            }
        )

    @property
    def report_id(self) -> str:
        return _digest(self.to_dict())

    @property
    def promotion_allowed(self) -> bool:
        return self.effective_mode in {
            V2RolloutMode.ASSIST,
            V2RolloutMode.AUTOMATIC,
        }

    @property
    def gate_passed(self) -> bool:
        return (
            self.automatic_ready
            if self.desired_mode is V2RolloutMode.AUTOMATIC
            else self.qualification_gate_passed
        )

    def to_dict(self, *, include_report_id: bool = False) -> dict[str, Any]:
        payload = {
            "schema": V2_ROLLOUT_REPORT_SCHEMA,
            "contract_version": V2_ROLLOUT_CONTRACT_VERSION,
            "binding": self.binding.to_dict(include_binding_id=True),
            "policy": self.policy.to_dict(),
            "desired_mode": self.desired_mode.value,
            "effective_mode": self.effective_mode.value,
            "desired_binding_id": self.desired_binding_id,
            "effective_binding_id": self.effective_binding_id,
            "qualification": self.qualification.to_dict(),
            "current": self.current.to_dict() if self.current else None,
            "reason_codes": list(self.reason_codes),
            "qualification_gate_passed": self.qualification_gate_passed,
            "current_tree_gate_passed": self.current_tree_gate_passed,
            "automatic_ready": self.automatic_ready,
            "rollback_applied": self.rollback_applied,
            "gate_passed": self.gate_passed,
            "promotion_allowed": self.promotion_allowed,
            "affected_behavior_ids": [self.binding.behavior_id],
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
        qualification: V2RolloutEvaluation,
        current: V2RolloutEvaluation | None = None,
    ) -> "V2RolloutReport":
        _reject_forbidden(payload)
        if not isinstance(payload, Mapping):
            raise V2RolloutError("v2 rollout report must be an object")
        binding = V2RolloutBinding.from_dict(payload.get("binding", {}))
        policy = V2RolloutPolicy.from_dict(payload.get("policy", {}))
        desired = payload.get("desired_mode", "")
        expected = evaluate_v2_self_improvement_rollout(
            qualification,
            binding=binding,
            desired_mode=desired,
            policy=policy,
            current_evaluation=current,
        )
        actual = dict(payload)
        claimed_id = actual.pop("report_id", expected.report_id)
        if actual != expected.to_dict():
            raise V2RolloutError(
                "persisted rollout report does not match source replay"
            )
        if claimed_id != expected.report_id:
            raise V2RolloutError("v2 rollout report identity does not match")
        return expected

    @classmethod
    def from_json(
        cls,
        value: str | bytes | bytearray,
        *,
        qualification: V2RolloutEvaluation,
        current: V2RolloutEvaluation | None = None,
    ) -> "V2RolloutReport":
        payload = _load_json(
            value,
            name="v2 rollout report",
            maximum=MAX_V2_ROLLOUT_REPORT_BYTES,
        )
        return cls.from_dict(
            payload, qualification=qualification, current=current
        )


def evaluate_v2_self_improvement_rollout(
    qualification: V2RolloutEvaluation,
    *,
    binding: V2RolloutBinding | Mapping[str, Any] | None = None,
    desired_mode: V2RolloutMode | str = V2RolloutMode.SHADOW,
    policy: V2RolloutPolicy | Mapping[str, Any] | None = None,
    current_evaluation: V2RolloutEvaluation | None = None,
) -> V2RolloutReport:
    """Recompute source evidence and derive a fail-closed rollout mode."""

    if not isinstance(qualification, V2RolloutEvaluation):
        raise V2RolloutError(
            "qualification must be a V2RolloutEvaluation"
        )
    desired = _mode(desired_mode, "desired_mode")
    if binding is None:
        normalized_binding = V2RolloutBinding.from_corpus(
            qualification.corpus
        )
    elif isinstance(binding, V2RolloutBinding):
        normalized_binding = binding
    else:
        normalized_binding = V2RolloutBinding.from_dict(binding)
    if policy is None:
        normalized_policy = V2RolloutPolicy(
            policy_id=normalized_binding.policy_id,
            policy_revision=normalized_binding.policy_revision,
            approved_capability_ids=(
                normalized_binding.capability_id,
            ),
            approved_behavior_ids=(normalized_binding.behavior_id,),
        )
    elif isinstance(policy, V2RolloutPolicy):
        normalized_policy = policy
    else:
        normalized_policy = V2RolloutPolicy.from_dict(policy)
    if current_evaluation is not None and not isinstance(
        current_evaluation, V2RolloutEvaluation
    ):
        raise V2RolloutError(
            "current_evaluation must be V2RolloutEvaluation"
        )

    qualifying_result, qualifying_self_report = (
        _recompute_v2_rollout_evaluation(qualification)
    )
    current_result = None
    current_self_report = None
    if current_evaluation is not None:
        current_result, current_self_report = (
            _recompute_v2_rollout_evaluation(current_evaluation)
        )
    reasons: set[str] = set()
    reasons.update(
        f"qualification:{item}"
        for item in qualifying_result.failure_codes
    )

    qualification_identity_matches = _identity_matches(
        qualifying_result.source_identity,
        normalized_binding,
        include_tree=desired is not V2RolloutMode.AUTOMATIC,
    )
    if not qualification_identity_matches:
        reasons.add("stale-binding:qualification")
    policy_permits = normalized_policy.permits(
        desired, normalized_binding
    )
    if desired is not V2RolloutMode.OFF and not policy_permits:
        reasons.add(f"policy-mode-not-approved:{desired.value}")

    qualification_gate_passed = bool(
        qualifying_result.passed
        and qualification_identity_matches
        and (
            desired is V2RolloutMode.OFF
            or desired is V2RolloutMode.SHADOW
            or policy_permits
        )
    )

    current_tree_gate_passed = False
    cross_regressions: tuple[str, ...] = ()
    if current_result is not None:
        reasons.update(
            f"current:{item}" for item in current_result.failure_codes
        )
        current_identity_matches = _identity_matches(
            current_result.source_identity,
            normalized_binding,
            include_tree=True,
        )
        if not current_identity_matches:
            reasons.add("stale-binding:current")
        distinct = (
            qualification.evidence_id != current_evaluation.evidence_id
            and qualification.evaluation_id
            != current_evaluation.evaluation_id
        )
        if not distinct:
            reasons.add("current-evaluation-not-separate")
        later = _datetime(str(current_evaluation.evaluated_at)) > _datetime(
            str(qualification.evaluated_at)
        )
        if not later:
            reasons.add("current-evaluation-not-later")
        assert current_self_report is not None
        cross_regressions = _cross_evaluation_regressions(
            qualifying_self_report, current_self_report
        )
        reasons.update(cross_regressions)
        current_tree_gate_passed = bool(
            current_result.passed
            and current_identity_matches
            and distinct
            and later
            and not cross_regressions
        )
    elif desired is V2RolloutMode.AUTOMATIC:
        reasons.add("current-tree-evaluation-required")

    automatic_ready = bool(
        desired is V2RolloutMode.AUTOMATIC
        and qualification_gate_passed
        and current_tree_gate_passed
        and normalized_policy.automatic_approved
        and policy_permits
    )
    if desired is V2RolloutMode.OFF:
        effective = V2RolloutMode.OFF
    elif desired is V2RolloutMode.SHADOW:
        effective = V2RolloutMode.SHADOW
    elif desired is V2RolloutMode.ASSIST and qualification_gate_passed:
        # If a caller supplies a monitoring observation, it is authoritative
        # for regression detection even though assist does not require one.
        effective = (
            V2RolloutMode.ASSIST
            if current_evaluation is None or current_tree_gate_passed
            else V2RolloutMode.SHADOW
        )
    elif automatic_ready:
        effective = V2RolloutMode.AUTOMATIC
    else:
        effective = V2RolloutMode.SHADOW

    rollback_reasons = any(
        reason.startswith(("stale-binding:", "regression:", "current:"))
        for reason in reasons
    )
    rollback_applied = bool(
        desired in {V2RolloutMode.ASSIST, V2RolloutMode.AUTOMATIC}
        and effective is V2RolloutMode.SHADOW
        and rollback_reasons
    )
    return V2RolloutReport(
        binding=normalized_binding,
        policy=normalized_policy,
        desired_mode=desired,
        effective_mode=effective,
        qualification=qualifying_result,
        current=current_result,
        reason_codes=tuple(sorted(reasons)),
        qualification_gate_passed=qualification_gate_passed,
        current_tree_gate_passed=current_tree_gate_passed,
        automatic_ready=automatic_ready,
        rollback_applied=rollback_applied,
    )


def verify_v2_rollout_report(
    report: V2RolloutReport | Mapping[str, Any],
    qualification: V2RolloutEvaluation,
    *,
    current_evaluation: V2RolloutEvaluation | None = None,
) -> V2RolloutReport:
    """Reject a persisted decision unless source replay reproduces it."""

    payload = (
        report.to_dict(include_report_id=True)
        if isinstance(report, V2RolloutReport)
        else report
    )
    if not isinstance(payload, Mapping):
        raise V2RolloutError("report must be a V2RolloutReport or object")
    return V2RolloutReport.from_dict(
        payload,
        qualification=qualification,
        current=current_evaluation,
    )


def replay_v2_rollout(
    qualification: V2RolloutEvaluation,
    *,
    binding: V2RolloutBinding | Mapping[str, Any] | None = None,
    desired_mode: V2RolloutMode | str = V2RolloutMode.SHADOW,
    policy: V2RolloutPolicy | Mapping[str, Any] | None = None,
    current_evaluation: V2RolloutEvaluation | None = None,
    expected_report: V2RolloutReport | Mapping[str, Any] | None = None,
) -> V2RolloutReport:
    """Recompute a decision, optionally verifying a persisted report."""

    result = evaluate_v2_self_improvement_rollout(
        qualification,
        binding=binding,
        desired_mode=desired_mode,
        policy=policy,
        current_evaluation=current_evaluation,
    )
    if expected_report is not None:
        return verify_v2_rollout_report(
            expected_report,
            qualification,
            current_evaluation=current_evaluation,
        )
    return result


# Compact aliases retained for the later public-surface integration task.
SelfImprovementV2RolloutMode = V2RolloutMode
SelfImprovementRolloutMode = V2RolloutMode
Generation2RolloutMode = V2RolloutMode
Generation2RolloutReport = V2RolloutReport
PairedV2RolloutPolicy = V2RolloutPolicy
PairedV2RolloutReport = V2RolloutReport
evaluate_generation2_rollout = evaluate_v2_self_improvement_rollout
evaluate_paired_v2_self_improvement_rollout = (
    evaluate_v2_self_improvement_rollout
)
evaluate_v2_rollout = evaluate_v2_self_improvement_rollout


__all__ = [
    "Generation2RolloutMode",
    "Generation2RolloutReport",
    "MAX_V2_ROLLOUT_REASON_CODES",
    "MAX_V2_ROLLOUT_REPORT_BYTES",
    "PairedV2RolloutPolicy",
    "PairedV2RolloutReport",
    "SelfImprovementRolloutMode",
    "SelfImprovementV2RolloutMode",
    "V2RolloutBinding",
    "V2RolloutError",
    "V2RolloutEvaluation",
    "V2RolloutEvaluationResult",
    "V2RolloutMode",
    "V2RolloutPolicy",
    "V2RolloutReport",
    "V2_ROLLOUT_BEHAVIOR_ID",
    "V2_ROLLOUT_BINDING_SCHEMA",
    "V2_ROLLOUT_CONTRACT_VERSION",
    "V2_ROLLOUT_EVALUATION_SCHEMA",
    "V2_ROLLOUT_METRIC_DIRECTIONS",
    "V2_ROLLOUT_POLICY_SCHEMA",
    "V2_ROLLOUT_REPORT_SCHEMA",
    "evaluate_generation2_rollout",
    "evaluate_paired_v2_self_improvement_rollout",
    "evaluate_v2_rollout",
    "evaluate_v2_self_improvement_rollout",
    "recompute_v2_rollout_evaluation",
    "replay_v2_rollout",
    "verify_v2_rollout_report",
]
