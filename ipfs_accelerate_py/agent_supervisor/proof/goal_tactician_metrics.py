"""GoalTacticianMetrics@1 — receipt-derived tactician quality and observability.

Conflict policy (FVT-G063 / FVT-033): this module owns the goal-tactician
benchmark metrics surface.  It does **not** invent synthetic performance
distributions.  Every rate is recomputed from additive counters taken from
cohort run receipts.  Unstable timing ratios and tool availability never
become correctness gates.

Acceptance invariants:

* metrics come from actual cohort receipts (not sampled/synthetic distributions);
* hard correctness, privacy, and authority gates are 100 percent or fail closed;
* wall-clock / resource timings are observational unless an explicit calibration
  receipt is present;
* cache hits must preserve authority level and exact identity bindings; and
* progress projections expose unresolved holes, witnesses, critical path,
  budgets, and next actions without private material.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final, Iterator

from .formal_verification_contracts import ContractValidationError


# ---------------------------------------------------------------------------
# Interface / schema constants
# ---------------------------------------------------------------------------

GOAL_TACTICIAN_METRICS_INTERFACE: Final = "GoalTacticianMetrics@1"
GOAL_TACTICIAN_BENCHMARK_INTERFACE: Final = "GoalTacticianBenchmark@1"
GOAL_TACTICIAN_METRICS_VERSION: Final = "1.0.0"
GOAL_TACTICIAN_BENCHMARK_VERSION: Final = "1.0.0"

GOAL_TACTICIAN_METRICS_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/goal-tactician-metrics@1"
)
GOAL_TACTICIAN_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/goal-tactician-run-receipt@1"
)
GOAL_TACTICIAN_PROGRESS_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/goal-tactician-progress@1"
)
GOAL_TACTICIAN_BENCHMARK_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/goal-tactician-benchmark@1"
)
GOAL_TACTICIAN_BENCHMARK_REPORT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/goal-tactician-benchmark-report@1"
)

BASIS_POINTS: Final = 10_000
MAX_RECEIPTS: Final = 4_096
MAX_TEXT_BYTES: Final = 512
MAX_ID_BYTES: Final = 256
MAX_NEXT_ACTIONS: Final = 64
MAX_HOLES: Final = 1_024
MAX_WITNESSES: Final = 1_024
MAX_CRITICAL_PATH: Final = 512
MAX_PROVIDER_IDS: Final = 64
MAX_COUNTER: Final = 10**15
MAX_DURATION_MS: Final = 31 * 24 * 60 * 60 * 1000

# Hard gates: these must be perfect (100%) for the cohort to pass correctness.
HARD_GATE_NAMES: Final = (
    "correctness",
    "privacy",
    "authority",
)

# Observational only unless calibrated — never correctness gates by default.
OBSERVATIONAL_METRIC_NAMES: Final = (
    "wall_time_ms",
    "cpu_time_ms",
    "memory_peak_bytes",
    "solver_latency_ms",
    "kernel_latency_ms",
    "model_latency_ms",
    "cache_latency_ms",
    "cancellation_latency_ms",
)

# Public-projection deny-list (aligned with proof_metrics / formal contracts).
_PRIVATE_KEY_PARTS: Final = (
    "transcript",
    "witness_body",
    "proof_term",
    "proof_body",
    "raw_proof",
    "private_premise",
    "source_excerpt",
    "hidden_witness",
    "private_witness",
)
_PRIVATE_KEYS: Final = frozenset(
    {
        "stdout",
        "stderr",
        "prompt",
        "response",
        "completion",
        "statement",
        "proof_log",
        "model_output",
        "raw_output",
        "hidden",
        "secret",
        "api_key",
        "access_token",
        "refresh_token",
        "password",
        "private_key",
        "credential",
    }
)

_ASSURANCE_RANK: Final = {
    "unverified": 0,
    "candidate": 1,
    "solver_checked": 2,
    "kernel_verified": 3,
    "attested": 4,
}


class GoalTacticianMetricsError(ContractValidationError):
    """Raised when goal-tactician metrics inputs or reports are unsafe."""


class EvidenceClass(str, Enum):
    """Whether a receipt is from a fixture cohort or live tools."""

    FIXTURE = "fixture"
    LIVE = "live"
    CALIBRATED = "calibrated"


class CacheOutcome(str, Enum):
    HIT = "hit"
    MISS = "miss"
    REJECTED = "rejected"
    BYPASS = "bypass"


class GateStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"
    OBSERVATIONAL = "observational"


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise GoalTacticianMetricsError(
            "goal tactician metrics require canonical JSON values"
        ) from exc


def _content_id(value: Any, *, prefix: str = "sha256:") -> str:
    digest = hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()
    return f"{prefix}{digest}" if prefix.endswith(":") else f"{prefix}{digest}"


def _text(value: Any, name: str, *, maximum: int = MAX_TEXT_BYTES) -> str:
    if isinstance(value, Enum):
        value = value.value
    if not isinstance(value, str) or not value.strip():
        raise GoalTacticianMetricsError(f"{name} must be non-empty text")
    result = value.strip()
    if "\x00" in result:
        raise GoalTacticianMetricsError(f"{name} contains a NUL byte")
    if len(result.encode("utf-8")) > maximum:
        raise GoalTacticianMetricsError(
            f"{name} exceeds its {maximum}-byte bound"
        )
    return result


def _optional_text(
    value: Any, name: str, *, maximum: int = MAX_TEXT_BYTES
) -> str:
    if value in (None, ""):
        return ""
    return _text(value, name, maximum=maximum)


def _nonnegative_int(value: Any, name: str, *, maximum: int = MAX_COUNTER) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise GoalTacticianMetricsError(f"{name} must be a non-negative integer")
    if value < 0 or value > maximum:
        raise GoalTacticianMetricsError(
            f"{name} must be in [0, {maximum}]"
        )
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise GoalTacticianMetricsError(f"{name} must be a boolean")
    return value


def _rate_bps(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        return 0
    return (numerator * BASIS_POINTS) // denominator


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(min(1.0, numerator / denominator), 6)


def _is_private_key(name: str) -> bool:
    """Return whether a JSON key names private proving material.

    Boolean *policy* fields such as ``contains_private_witnesses`` are public
    projections and must not be treated as private payloads.
    """

    normalized = name.strip().lower().replace("-", "_")
    if normalized.startswith("contains_"):
        return False
    if normalized in _PRIVATE_KEYS:
        return True
    if normalized in _PRIVATE_KEY_PARTS:
        return True
    # Exact suffix matches for nested payload smuggling (e.g. raw_proof_body).
    for part in _PRIVATE_KEY_PARTS:
        if normalized == part or normalized.endswith(f"_{part}"):
            return True
    return False


def _reject_private_material(value: Any, *, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            if _is_private_key(key_text):
                raise GoalTacticianMetricsError(
                    f"private material forbidden at {path}.{key_text}"
                )
            _reject_private_material(item, path=f"{path}.{key_text}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_private_material(item, path=f"{path}[{index}]")


def _assurance_rank(value: Any) -> int:
    if value in (None, ""):
        return 0
    text = str(value).strip().lower().replace("-", "_")
    if text not in _ASSURANCE_RANK:
        raise GoalTacticianMetricsError(f"unsupported assurance level: {value!r}")
    return _ASSURANCE_RANK[text]


def _check_schema(payload: Mapping[str, Any], expected: str, artifact: str) -> None:
    if not isinstance(payload, Mapping):
        raise GoalTacticianMetricsError(f"{artifact} must be an object")
    if payload.get("schema") != expected:
        raise GoalTacticianMetricsError(f"unsupported {artifact} schema")


# ---------------------------------------------------------------------------
# Receipt and progress contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GoalTacticianRunReceipt:
    """Public projection of one tactician run used for cohort metrics.

    Receipts are measurement inputs, not proof authority.  They must not carry
    private witnesses, proof bodies, prompts, or provider transcripts.
    """

    receipt_id: str
    run_id: str
    goal_id: str
    repository_tree_id: str
    policy_id: str
    provider_id: str
    evidence_class: EvidenceClass

    # Formalization
    formalization_attempted: bool
    formalization_succeeded: bool
    formalization_required: bool

    # Proof-gap quality (additive TP/FP/FN style counts)
    proof_gap_true_positive: int
    proof_gap_false_positive: int
    proof_gap_false_negative: int
    proof_gap_true_negative: int

    # Plan solvability
    plan_steps_total: int
    plan_steps_solvable: int
    plan_admitted: bool

    # Authority
    claimed_assurance: str
    authoritative_assurance: str
    authority_boundary_violation: bool
    false_completion: bool
    privacy_violation: bool

    # Counterexample lifecycle
    counterexample_count: int
    counterexample_replayable_count: int
    counterexample_reduced_count: int
    counterexample_explained_count: int

    # Multi-provider agreement
    providers_queried: tuple[str, ...]
    providers_agreeing: tuple[str, ...]

    # Resources / cancellation (observational by default)
    wall_time_ms: int
    cpu_time_ms: int
    memory_peak_bytes: int
    cancelled: bool
    cancellation_honored: bool
    calibration_receipt_id: str

    # Cache
    cache_outcome: CacheOutcome
    cache_key: str
    cache_authority_preserved: bool
    cache_identity_preserved: bool

    # Progress snapshot fields (public identifiers only)
    unresolved_hole_ids: tuple[str, ...]
    witness_ids: tuple[str, ...]
    critical_path_step_ids: tuple[str, ...]
    budget_cpu_ms_remaining: int
    budget_memory_bytes_remaining: int
    budget_token_remaining: int
    next_actions: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "receipt_id", _text(self.receipt_id, "receipt_id", maximum=MAX_ID_BYTES)
        )
        object.__setattr__(
            self, "run_id", _text(self.run_id, "run_id", maximum=MAX_ID_BYTES)
        )
        object.__setattr__(
            self, "goal_id", _text(self.goal_id, "goal_id", maximum=MAX_ID_BYTES)
        )
        object.__setattr__(
            self,
            "repository_tree_id",
            _text(self.repository_tree_id, "repository_tree_id", maximum=MAX_ID_BYTES),
        )
        object.__setattr__(
            self, "policy_id", _text(self.policy_id, "policy_id", maximum=MAX_ID_BYTES)
        )
        object.__setattr__(
            self,
            "provider_id",
            _text(self.provider_id, "provider_id", maximum=MAX_ID_BYTES),
        )

        try:
            evidence = (
                self.evidence_class
                if isinstance(self.evidence_class, EvidenceClass)
                else EvidenceClass(str(self.evidence_class))
            )
        except ValueError as exc:
            raise GoalTacticianMetricsError(
                "unsupported evidence_class"
            ) from exc
        object.__setattr__(self, "evidence_class", evidence)

        for name in (
            "formalization_attempted",
            "formalization_succeeded",
            "formalization_required",
            "plan_admitted",
            "authority_boundary_violation",
            "false_completion",
            "privacy_violation",
            "cancelled",
            "cancellation_honored",
            "cache_authority_preserved",
            "cache_identity_preserved",
        ):
            object.__setattr__(self, name, _boolean(getattr(self, name), name))

        for name in (
            "proof_gap_true_positive",
            "proof_gap_false_positive",
            "proof_gap_false_negative",
            "proof_gap_true_negative",
            "plan_steps_total",
            "plan_steps_solvable",
            "counterexample_count",
            "counterexample_replayable_count",
            "counterexample_reduced_count",
            "counterexample_explained_count",
            "wall_time_ms",
            "cpu_time_ms",
            "memory_peak_bytes",
            "budget_cpu_ms_remaining",
            "budget_memory_bytes_remaining",
            "budget_token_remaining",
        ):
            maximum = MAX_DURATION_MS if name.endswith("_ms") else MAX_COUNTER
            if name == "memory_peak_bytes" or name == "budget_memory_bytes_remaining":
                maximum = MAX_COUNTER
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name, maximum=maximum),
            )

        if self.plan_steps_solvable > self.plan_steps_total:
            raise GoalTacticianMetricsError(
                "plan_steps_solvable cannot exceed plan_steps_total"
            )
        for reduced_name, total_name in (
            ("counterexample_replayable_count", "counterexample_count"),
            ("counterexample_reduced_count", "counterexample_count"),
            ("counterexample_explained_count", "counterexample_count"),
        ):
            if getattr(self, reduced_name) > getattr(self, total_name):
                raise GoalTacticianMetricsError(
                    f"{reduced_name} cannot exceed {total_name}"
                )

        object.__setattr__(
            self,
            "claimed_assurance",
            _text(self.claimed_assurance, "claimed_assurance", maximum=64).lower(),
        )
        object.__setattr__(
            self,
            "authoritative_assurance",
            _text(
                self.authoritative_assurance,
                "authoritative_assurance",
                maximum=64,
            ).lower(),
        )
        _assurance_rank(self.claimed_assurance)
        _assurance_rank(self.authoritative_assurance)

        # Authority cannot be upgraded by a claim above independent assurance.
        if _assurance_rank(self.claimed_assurance) > _assurance_rank(
            self.authoritative_assurance
        ):
            # Not automatically a violation flag — callers set the flag — but
            # we refuse inconsistent receipts that claim authority they lack
            # without declaring the violation.
            if not self.authority_boundary_violation:
                raise GoalTacticianMetricsError(
                    "claimed_assurance exceeds authoritative_assurance without "
                    "authority_boundary_violation=True"
                )

        try:
            cache_outcome = (
                self.cache_outcome
                if isinstance(self.cache_outcome, CacheOutcome)
                else CacheOutcome(str(self.cache_outcome))
            )
        except ValueError as exc:
            raise GoalTacticianMetricsError("unsupported cache_outcome") from exc
        object.__setattr__(self, "cache_outcome", cache_outcome)
        object.__setattr__(
            self,
            "cache_key",
            _optional_text(self.cache_key, "cache_key", maximum=MAX_ID_BYTES),
        )
        if cache_outcome is CacheOutcome.HIT and not self.cache_key:
            raise GoalTacticianMetricsError("cache hits require a cache_key")
        if cache_outcome is CacheOutcome.HIT and (
            not self.cache_authority_preserved or not self.cache_identity_preserved
        ):
            # Hits that drop authority or identity are rejected measurements.
            raise GoalTacticianMetricsError(
                "cache hits must preserve authority and exact identity"
            )

        object.__setattr__(
            self,
            "calibration_receipt_id",
            _optional_text(
                self.calibration_receipt_id,
                "calibration_receipt_id",
                maximum=MAX_ID_BYTES,
            ),
        )
        if (
            self.evidence_class is EvidenceClass.CALIBRATED
            and not self.calibration_receipt_id
        ):
            raise GoalTacticianMetricsError(
                "calibrated receipts require calibration_receipt_id"
            )

        def _id_tuple(
            values: Sequence[str] | tuple[str, ...],
            name: str,
            *,
            maximum: int,
        ) -> tuple[str, ...]:
            if not isinstance(values, (list, tuple)):
                raise GoalTacticianMetricsError(f"{name} must be a sequence")
            if len(values) > maximum:
                raise GoalTacticianMetricsError(f"{name} exceeds bound {maximum}")
            cleaned: list[str] = []
            seen: set[str] = set()
            for item in values:
                text = _text(item, name, maximum=MAX_ID_BYTES)
                if text not in seen:
                    seen.add(text)
                    cleaned.append(text)
            return tuple(cleaned)

        object.__setattr__(
            self,
            "providers_queried",
            _id_tuple(self.providers_queried, "providers_queried", maximum=MAX_PROVIDER_IDS),
        )
        object.__setattr__(
            self,
            "providers_agreeing",
            _id_tuple(
                self.providers_agreeing, "providers_agreeing", maximum=MAX_PROVIDER_IDS
            ),
        )
        agreeing = set(self.providers_agreeing)
        queried = set(self.providers_queried)
        if agreeing - queried:
            raise GoalTacticianMetricsError(
                "providers_agreeing must be a subset of providers_queried"
            )

        object.__setattr__(
            self,
            "unresolved_hole_ids",
            _id_tuple(
                self.unresolved_hole_ids, "unresolved_hole_ids", maximum=MAX_HOLES
            ),
        )
        object.__setattr__(
            self,
            "witness_ids",
            _id_tuple(self.witness_ids, "witness_ids", maximum=MAX_WITNESSES),
        )
        object.__setattr__(
            self,
            "critical_path_step_ids",
            _id_tuple(
                self.critical_path_step_ids,
                "critical_path_step_ids",
                maximum=MAX_CRITICAL_PATH,
            ),
        )
        object.__setattr__(
            self,
            "next_actions",
            _id_tuple(self.next_actions, "next_actions", maximum=MAX_NEXT_ACTIONS),
        )

        if self.cancelled and not self.cancellation_honored:
            # Allowed as a measurement of a bug, but must not pass gates.
            pass

        payload = self.to_dict()
        _reject_private_material(payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": GOAL_TACTICIAN_RECEIPT_SCHEMA,
            "receipt_id": self.receipt_id,
            "run_id": self.run_id,
            "goal_id": self.goal_id,
            "repository_tree_id": self.repository_tree_id,
            "policy_id": self.policy_id,
            "provider_id": self.provider_id,
            "evidence_class": self.evidence_class.value,
            "formalization_attempted": self.formalization_attempted,
            "formalization_succeeded": self.formalization_succeeded,
            "formalization_required": self.formalization_required,
            "proof_gap_true_positive": self.proof_gap_true_positive,
            "proof_gap_false_positive": self.proof_gap_false_positive,
            "proof_gap_false_negative": self.proof_gap_false_negative,
            "proof_gap_true_negative": self.proof_gap_true_negative,
            "plan_steps_total": self.plan_steps_total,
            "plan_steps_solvable": self.plan_steps_solvable,
            "plan_admitted": self.plan_admitted,
            "claimed_assurance": self.claimed_assurance,
            "authoritative_assurance": self.authoritative_assurance,
            "authority_boundary_violation": self.authority_boundary_violation,
            "false_completion": self.false_completion,
            "privacy_violation": self.privacy_violation,
            "counterexample_count": self.counterexample_count,
            "counterexample_replayable_count": self.counterexample_replayable_count,
            "counterexample_reduced_count": self.counterexample_reduced_count,
            "counterexample_explained_count": self.counterexample_explained_count,
            "providers_queried": list(self.providers_queried),
            "providers_agreeing": list(self.providers_agreeing),
            "wall_time_ms": self.wall_time_ms,
            "cpu_time_ms": self.cpu_time_ms,
            "memory_peak_bytes": self.memory_peak_bytes,
            "cancelled": self.cancelled,
            "cancellation_honored": self.cancellation_honored,
            "calibration_receipt_id": self.calibration_receipt_id,
            "cache_outcome": self.cache_outcome.value,
            "cache_key": self.cache_key,
            "cache_authority_preserved": self.cache_authority_preserved,
            "cache_identity_preserved": self.cache_identity_preserved,
            "unresolved_hole_ids": list(self.unresolved_hole_ids),
            "witness_ids": list(self.witness_ids),
            "critical_path_step_ids": list(self.critical_path_step_ids),
            "budget_cpu_ms_remaining": self.budget_cpu_ms_remaining,
            "budget_memory_bytes_remaining": self.budget_memory_bytes_remaining,
            "budget_token_remaining": self.budget_token_remaining,
            "next_actions": list(self.next_actions),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GoalTacticianRunReceipt":
        if not isinstance(payload, Mapping):
            raise GoalTacticianMetricsError("run receipt must be an object")
        schema = payload.get("schema")
        if schema not in (None, "", GOAL_TACTICIAN_RECEIPT_SCHEMA):
            raise GoalTacticianMetricsError("unsupported run receipt schema")
        _reject_private_material(payload)
        return cls(
            receipt_id=payload.get("receipt_id", ""),
            run_id=payload.get("run_id", ""),
            goal_id=payload.get("goal_id", ""),
            repository_tree_id=payload.get("repository_tree_id", ""),
            policy_id=payload.get("policy_id", ""),
            provider_id=payload.get("provider_id", ""),
            evidence_class=payload.get("evidence_class", EvidenceClass.FIXTURE.value),
            formalization_attempted=bool(payload.get("formalization_attempted", False)),
            formalization_succeeded=bool(payload.get("formalization_succeeded", False)),
            formalization_required=bool(payload.get("formalization_required", False)),
            proof_gap_true_positive=int(payload.get("proof_gap_true_positive", 0) or 0),
            proof_gap_false_positive=int(payload.get("proof_gap_false_positive", 0) or 0),
            proof_gap_false_negative=int(payload.get("proof_gap_false_negative", 0) or 0),
            proof_gap_true_negative=int(payload.get("proof_gap_true_negative", 0) or 0),
            plan_steps_total=int(payload.get("plan_steps_total", 0) or 0),
            plan_steps_solvable=int(payload.get("plan_steps_solvable", 0) or 0),
            plan_admitted=bool(payload.get("plan_admitted", False)),
            claimed_assurance=str(payload.get("claimed_assurance", "unverified")),
            authoritative_assurance=str(
                payload.get("authoritative_assurance", "unverified")
            ),
            authority_boundary_violation=bool(
                payload.get("authority_boundary_violation", False)
            ),
            false_completion=bool(payload.get("false_completion", False)),
            privacy_violation=bool(payload.get("privacy_violation", False)),
            counterexample_count=int(payload.get("counterexample_count", 0) or 0),
            counterexample_replayable_count=int(
                payload.get("counterexample_replayable_count", 0) or 0
            ),
            counterexample_reduced_count=int(
                payload.get("counterexample_reduced_count", 0) or 0
            ),
            counterexample_explained_count=int(
                payload.get("counterexample_explained_count", 0) or 0
            ),
            providers_queried=tuple(payload.get("providers_queried") or ()),
            providers_agreeing=tuple(payload.get("providers_agreeing") or ()),
            wall_time_ms=int(payload.get("wall_time_ms", 0) or 0),
            cpu_time_ms=int(payload.get("cpu_time_ms", 0) or 0),
            memory_peak_bytes=int(payload.get("memory_peak_bytes", 0) or 0),
            cancelled=bool(payload.get("cancelled", False)),
            cancellation_honored=bool(payload.get("cancellation_honored", True)),
            calibration_receipt_id=str(payload.get("calibration_receipt_id") or ""),
            cache_outcome=payload.get("cache_outcome", CacheOutcome.MISS.value),
            cache_key=str(payload.get("cache_key") or ""),
            cache_authority_preserved=bool(
                payload.get("cache_authority_preserved", True)
            ),
            cache_identity_preserved=bool(
                payload.get("cache_identity_preserved", True)
            ),
            unresolved_hole_ids=tuple(payload.get("unresolved_hole_ids") or ()),
            witness_ids=tuple(payload.get("witness_ids") or ()),
            critical_path_step_ids=tuple(
                payload.get("critical_path_step_ids") or ()
            ),
            budget_cpu_ms_remaining=int(
                payload.get("budget_cpu_ms_remaining", 0) or 0
            ),
            budget_memory_bytes_remaining=int(
                payload.get("budget_memory_bytes_remaining", 0) or 0
            ),
            budget_token_remaining=int(payload.get("budget_token_remaining", 0) or 0),
            next_actions=tuple(payload.get("next_actions") or ()),
        )


@dataclass(frozen=True)
class GoalTacticianProgress:
    """Supervisor-facing progress projection (public identifiers only)."""

    unresolved_hole_ids: tuple[str, ...]
    witness_ids: tuple[str, ...]
    critical_path_step_ids: tuple[str, ...]
    budget_cpu_ms_remaining: int
    budget_memory_bytes_remaining: int
    budget_token_remaining: int
    next_actions: tuple[str, ...]
    cancelled_run_count: int
    open_plan_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": GOAL_TACTICIAN_PROGRESS_SCHEMA,
            "unresolved_hole_ids": list(self.unresolved_hole_ids),
            "witness_ids": list(self.witness_ids),
            "critical_path_step_ids": list(self.critical_path_step_ids),
            "budgets": {
                "cpu_ms_remaining": self.budget_cpu_ms_remaining,
                "memory_bytes_remaining": self.budget_memory_bytes_remaining,
                "token_remaining": self.budget_token_remaining,
            },
            "next_actions": list(self.next_actions),
            "cancelled_run_count": self.cancelled_run_count,
            "open_plan_count": self.open_plan_count,
            "unresolved_hole_count": len(self.unresolved_hole_ids),
            "witness_count": len(self.witness_ids),
            "critical_path_length": len(self.critical_path_step_ids),
        }


# ---------------------------------------------------------------------------
# Aggregated metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GoalTacticianMetrics:
    """Additive cohort metrics with independently recomputed rates.

    Interface: ``GoalTacticianMetrics@1``.
    """

    receipt_count: int
    evidence_classes: tuple[str, ...]

    formalization_required_count: int
    formalization_attempted_count: int
    formalization_succeeded_count: int

    proof_gap_true_positive: int
    proof_gap_false_positive: int
    proof_gap_false_negative: int
    proof_gap_true_negative: int

    plan_steps_total: int
    plan_steps_solvable: int
    plan_admitted_count: int

    authority_boundary_violation_count: int
    false_completion_count: int
    privacy_violation_count: int
    authoritative_kernel_or_above_count: int

    counterexample_count: int
    counterexample_replayable_count: int
    counterexample_reduced_count: int
    counterexample_explained_count: int

    provider_query_pairs: int
    provider_agreement_pairs: int

    wall_time_total_ms: int
    wall_time_mean_ms: int
    cpu_time_total_ms: int
    memory_peak_bytes_max: int
    cancelled_count: int
    cancellation_honored_count: int
    calibrated_timing: bool

    cache_lookup_count: int
    cache_hit_count: int
    cache_miss_count: int
    cache_rejection_count: int
    cache_hit_authority_preserved_count: int
    cache_hit_identity_preserved_count: int

    hard_gate_correctness_bps: int
    hard_gate_privacy_bps: int
    hard_gate_authority_bps: int
    hard_gates_passed: bool

    progress: GoalTacticianProgress
    source: str = "cohort_receipts"
    synthetic_distributions: bool = False

    def __post_init__(self) -> None:
        if self.synthetic_distributions:
            raise GoalTacticianMetricsError(
                "synthetic distributions are forbidden; metrics must derive "
                "from cohort receipts"
            )
        if self.source != "cohort_receipts":
            raise GoalTacticianMetricsError(
                "metrics source must be cohort_receipts"
            )

    @property
    def formalization_success_bps(self) -> int:
        return _rate_bps(
            self.formalization_succeeded_count, self.formalization_required_count
        )

    @property
    def proof_gap_precision_bps(self) -> int:
        return _rate_bps(
            self.proof_gap_true_positive,
            self.proof_gap_true_positive + self.proof_gap_false_positive,
        )

    @property
    def proof_gap_recall_bps(self) -> int:
        return _rate_bps(
            self.proof_gap_true_positive,
            self.proof_gap_true_positive + self.proof_gap_false_negative,
        )

    @property
    def plan_solvability_bps(self) -> int:
        return _rate_bps(self.plan_steps_solvable, self.plan_steps_total)

    @property
    def counterexample_replay_bps(self) -> int:
        return _rate_bps(
            self.counterexample_replayable_count, self.counterexample_count
        )

    @property
    def counterexample_reduction_bps(self) -> int:
        return _rate_bps(
            self.counterexample_reduced_count, self.counterexample_count
        )

    @property
    def counterexample_explanation_bps(self) -> int:
        return _rate_bps(
            self.counterexample_explained_count, self.counterexample_count
        )

    @property
    def provider_agreement_bps(self) -> int:
        return _rate_bps(self.provider_agreement_pairs, self.provider_query_pairs)

    @property
    def cache_hit_rate_bps(self) -> int:
        return _rate_bps(self.cache_hit_count, self.cache_lookup_count)

    @property
    def cancellation_honor_bps(self) -> int:
        return _rate_bps(self.cancellation_honored_count, self.cancelled_count)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": GOAL_TACTICIAN_METRICS_SCHEMA,
            "interface": GOAL_TACTICIAN_METRICS_INTERFACE,
            "version": GOAL_TACTICIAN_METRICS_VERSION,
            "source": self.source,
            "synthetic_distributions": self.synthetic_distributions,
            "receipt_count": self.receipt_count,
            "evidence_classes": list(self.evidence_classes),
            "formalization": {
                "required_count": self.formalization_required_count,
                "attempted_count": self.formalization_attempted_count,
                "succeeded_count": self.formalization_succeeded_count,
                "success_bps": self.formalization_success_bps,
                "success_rate": _ratio(
                    self.formalization_succeeded_count,
                    self.formalization_required_count,
                ),
            },
            "proof_gap": {
                "true_positive": self.proof_gap_true_positive,
                "false_positive": self.proof_gap_false_positive,
                "false_negative": self.proof_gap_false_negative,
                "true_negative": self.proof_gap_true_negative,
                "precision_bps": self.proof_gap_precision_bps,
                "recall_bps": self.proof_gap_recall_bps,
                "precision": _ratio(
                    self.proof_gap_true_positive,
                    self.proof_gap_true_positive + self.proof_gap_false_positive,
                ),
                "recall": _ratio(
                    self.proof_gap_true_positive,
                    self.proof_gap_true_positive + self.proof_gap_false_negative,
                ),
            },
            "plan_solvability": {
                "steps_total": self.plan_steps_total,
                "steps_solvable": self.plan_steps_solvable,
                "admitted_count": self.plan_admitted_count,
                "solvability_bps": self.plan_solvability_bps,
                "solvability_rate": _ratio(
                    self.plan_steps_solvable, self.plan_steps_total
                ),
            },
            "proof_authority": {
                "authority_boundary_violation_count": (
                    self.authority_boundary_violation_count
                ),
                "false_completion_count": self.false_completion_count,
                "privacy_violation_count": self.privacy_violation_count,
                "authoritative_kernel_or_above_count": (
                    self.authoritative_kernel_or_above_count
                ),
            },
            "counterexamples": {
                "count": self.counterexample_count,
                "replayable_count": self.counterexample_replayable_count,
                "reduced_count": self.counterexample_reduced_count,
                "explained_count": self.counterexample_explained_count,
                "replay_bps": self.counterexample_replay_bps,
                "reduction_bps": self.counterexample_reduction_bps,
                "explanation_bps": self.counterexample_explanation_bps,
            },
            "provider_agreement": {
                "query_pairs": self.provider_query_pairs,
                "agreement_pairs": self.provider_agreement_pairs,
                "agreement_bps": self.provider_agreement_bps,
                "agreement_rate": _ratio(
                    self.provider_agreement_pairs, self.provider_query_pairs
                ),
            },
            "resources": {
                "wall_time_total_ms": self.wall_time_total_ms,
                "wall_time_mean_ms": self.wall_time_mean_ms,
                "cpu_time_total_ms": self.cpu_time_total_ms,
                "memory_peak_bytes_max": self.memory_peak_bytes_max,
                "cancelled_count": self.cancelled_count,
                "cancellation_honored_count": self.cancellation_honored_count,
                "cancellation_honor_bps": self.cancellation_honor_bps,
                "calibrated_timing": self.calibrated_timing,
                "timing_role": (
                    "calibrated_gate_eligible"
                    if self.calibrated_timing
                    else "observational"
                ),
                "observational_fields": list(OBSERVATIONAL_METRIC_NAMES),
            },
            "cache": {
                "lookup_count": self.cache_lookup_count,
                "hit_count": self.cache_hit_count,
                "miss_count": self.cache_miss_count,
                "rejection_count": self.cache_rejection_count,
                "hit_rate_bps": self.cache_hit_rate_bps,
                "hit_authority_preserved_count": (
                    self.cache_hit_authority_preserved_count
                ),
                "hit_identity_preserved_count": (
                    self.cache_hit_identity_preserved_count
                ),
                "hits_preserve_authority_and_identity": (
                    self.cache_hit_count == 0
                    or (
                        self.cache_hit_authority_preserved_count == self.cache_hit_count
                        and self.cache_hit_identity_preserved_count
                        == self.cache_hit_count
                    )
                ),
            },
            "hard_gates": {
                "correctness_bps": self.hard_gate_correctness_bps,
                "privacy_bps": self.hard_gate_privacy_bps,
                "authority_bps": self.hard_gate_authority_bps,
                "required_bps": BASIS_POINTS,
                "passed": self.hard_gates_passed,
                "gate_names": list(HARD_GATE_NAMES),
            },
            "progress": self.progress.to_dict(),
        }
        _reject_private_material(payload)
        return payload


def derive_goal_tactician_metrics(
    receipts: Sequence[GoalTacticianRunReceipt | Mapping[str, Any]],
) -> GoalTacticianMetrics:
    """Derive metrics from actual cohort receipts (never synthetic)."""

    if not isinstance(receipts, Sequence) or isinstance(receipts, (str, bytes)):
        raise GoalTacticianMetricsError("receipts must be a sequence")
    if not receipts:
        raise GoalTacticianMetricsError("at least one cohort receipt is required")
    if len(receipts) > MAX_RECEIPTS:
        raise GoalTacticianMetricsError(f"receipt cohort exceeds {MAX_RECEIPTS}")

    typed: list[GoalTacticianRunReceipt] = []
    for item in receipts:
        if isinstance(item, GoalTacticianRunReceipt):
            typed.append(item)
        elif isinstance(item, Mapping):
            typed.append(GoalTacticianRunReceipt.from_dict(item))
        else:
            raise GoalTacticianMetricsError(
                "each receipt must be GoalTacticianRunReceipt or a mapping"
            )

    receipt_ids = [item.receipt_id for item in typed]
    if len(set(receipt_ids)) != len(receipt_ids):
        raise GoalTacticianMetricsError("receipt_id values must be unique in a cohort")

    formalization_required = sum(1 for r in typed if r.formalization_required)
    formalization_attempted = sum(1 for r in typed if r.formalization_attempted)
    formalization_succeeded = sum(
        1 for r in typed if r.formalization_required and r.formalization_succeeded
    )

    tp = sum(r.proof_gap_true_positive for r in typed)
    fp = sum(r.proof_gap_false_positive for r in typed)
    fn = sum(r.proof_gap_false_negative for r in typed)
    tn = sum(r.proof_gap_true_negative for r in typed)

    plan_total = sum(r.plan_steps_total for r in typed)
    plan_solvable = sum(r.plan_steps_solvable for r in typed)
    plan_admitted = sum(1 for r in typed if r.plan_admitted)

    authority_violations = sum(1 for r in typed if r.authority_boundary_violation)
    false_completions = sum(1 for r in typed if r.false_completion)
    privacy_violations = sum(1 for r in typed if r.privacy_violation)
    kernel_or_above = sum(
        1
        for r in typed
        if _assurance_rank(r.authoritative_assurance)
        >= _ASSURANCE_RANK["kernel_verified"]
    )

    cx_total = sum(r.counterexample_count for r in typed)
    cx_replay = sum(r.counterexample_replayable_count for r in typed)
    cx_reduced = sum(r.counterexample_reduced_count for r in typed)
    cx_explained = sum(r.counterexample_explained_count for r in typed)

    provider_query_pairs = 0
    provider_agreement_pairs = 0
    for r in typed:
        # One multi-provider query contributes |queried| choose-style pairs as
        # |queried| "slots" with |agreeing| successes — simple additive form.
        queried = len(r.providers_queried)
        agreeing = len(r.providers_agreeing)
        if queried:
            provider_query_pairs += queried
            provider_agreement_pairs += agreeing

    wall_total = sum(r.wall_time_ms for r in typed)
    cpu_total = sum(r.cpu_time_ms for r in typed)
    memory_max = max(r.memory_peak_bytes for r in typed)
    cancelled = sum(1 for r in typed if r.cancelled)
    cancellation_honored = sum(
        1 for r in typed if r.cancelled and r.cancellation_honored
    )
    calibrated = all(
        r.evidence_class is EvidenceClass.CALIBRATED
        and bool(r.calibration_receipt_id)
        for r in typed
    ) and bool(typed)

    cache_hits = sum(1 for r in typed if r.cache_outcome is CacheOutcome.HIT)
    cache_misses = sum(1 for r in typed if r.cache_outcome is CacheOutcome.MISS)
    cache_rejections = sum(
        1 for r in typed if r.cache_outcome is CacheOutcome.REJECTED
    )
    # Lookups include hits, misses, rejections (not pure bypass).
    cache_lookups = cache_hits + cache_misses + cache_rejections
    hit_authority = sum(
        1
        for r in typed
        if r.cache_outcome is CacheOutcome.HIT and r.cache_authority_preserved
    )
    hit_identity = sum(
        1
        for r in typed
        if r.cache_outcome is CacheOutcome.HIT and r.cache_identity_preserved
    )

    # Hard gates: 100% or fail.  Correctness = no false completions.
    # Privacy = no privacy violations.  Authority = no boundary violations.
    correctness_bps = BASIS_POINTS if false_completions == 0 else 0
    privacy_bps = BASIS_POINTS if privacy_violations == 0 else 0
    authority_bps = BASIS_POINTS if authority_violations == 0 else 0
    # Cache hits already enforced at receipt construction; double-check aggregate.
    if cache_hits and (
        hit_authority != cache_hits or hit_identity != cache_hits
    ):
        authority_bps = 0
    hard_passed = (
        correctness_bps == BASIS_POINTS
        and privacy_bps == BASIS_POINTS
        and authority_bps == BASIS_POINTS
    )

    # Progress: union of open holes / witnesses / latest critical path & budgets.
    hole_ids: list[str] = []
    hole_seen: set[str] = set()
    witness_ids: list[str] = []
    witness_seen: set[str] = set()
    next_actions: list[str] = []
    action_seen: set[str] = set()
    critical_path: tuple[str, ...] = ()
    # Prefer the longest critical path as the cohort-visible bottleneck.
    for r in typed:
        for hole in r.unresolved_hole_ids:
            if hole not in hole_seen:
                hole_seen.add(hole)
                hole_ids.append(hole)
        for witness in r.witness_ids:
            if witness not in witness_seen:
                witness_seen.add(witness)
                witness_ids.append(witness)
        for action in r.next_actions:
            if action not in action_seen:
                action_seen.add(action)
                next_actions.append(action)
        if len(r.critical_path_step_ids) >= len(critical_path):
            critical_path = r.critical_path_step_ids

    # Remaining budgets: min remaining across open plans (most constrained).
    open_plans = [r for r in typed if r.unresolved_hole_ids or not r.plan_admitted]
    if open_plans:
        budget_cpu = min(r.budget_cpu_ms_remaining for r in open_plans)
        budget_mem = min(r.budget_memory_bytes_remaining for r in open_plans)
        budget_tok = min(r.budget_token_remaining for r in open_plans)
    else:
        budget_cpu = min(r.budget_cpu_ms_remaining for r in typed)
        budget_mem = min(r.budget_memory_bytes_remaining for r in typed)
        budget_tok = min(r.budget_token_remaining for r in typed)

    progress = GoalTacticianProgress(
        unresolved_hole_ids=tuple(hole_ids[:MAX_HOLES]),
        witness_ids=tuple(witness_ids[:MAX_WITNESSES]),
        critical_path_step_ids=critical_path[:MAX_CRITICAL_PATH],
        budget_cpu_ms_remaining=budget_cpu,
        budget_memory_bytes_remaining=budget_mem,
        budget_token_remaining=budget_tok,
        next_actions=tuple(next_actions[:MAX_NEXT_ACTIONS]),
        cancelled_run_count=cancelled,
        open_plan_count=len(open_plans),
    )

    evidence_classes = tuple(sorted({r.evidence_class.value for r in typed}))
    mean_wall = wall_total // len(typed)

    return GoalTacticianMetrics(
        receipt_count=len(typed),
        evidence_classes=evidence_classes,
        formalization_required_count=formalization_required,
        formalization_attempted_count=formalization_attempted,
        formalization_succeeded_count=formalization_succeeded,
        proof_gap_true_positive=tp,
        proof_gap_false_positive=fp,
        proof_gap_false_negative=fn,
        proof_gap_true_negative=tn,
        plan_steps_total=plan_total,
        plan_steps_solvable=plan_solvable,
        plan_admitted_count=plan_admitted,
        authority_boundary_violation_count=authority_violations,
        false_completion_count=false_completions,
        privacy_violation_count=privacy_violations,
        authoritative_kernel_or_above_count=kernel_or_above,
        counterexample_count=cx_total,
        counterexample_replayable_count=cx_replay,
        counterexample_reduced_count=cx_reduced,
        counterexample_explained_count=cx_explained,
        provider_query_pairs=provider_query_pairs,
        provider_agreement_pairs=provider_agreement_pairs,
        wall_time_total_ms=wall_total,
        wall_time_mean_ms=mean_wall,
        cpu_time_total_ms=cpu_total,
        memory_peak_bytes_max=memory_max,
        cancelled_count=cancelled,
        cancellation_honored_count=cancellation_honored,
        calibrated_timing=calibrated,
        cache_lookup_count=cache_lookups,
        cache_hit_count=cache_hits,
        cache_miss_count=cache_misses,
        cache_rejection_count=cache_rejections,
        cache_hit_authority_preserved_count=hit_authority,
        cache_hit_identity_preserved_count=hit_identity,
        hard_gate_correctness_bps=correctness_bps,
        hard_gate_privacy_bps=privacy_bps,
        hard_gate_authority_bps=authority_bps,
        hard_gates_passed=hard_passed,
        progress=progress,
        source="cohort_receipts",
        synthetic_distributions=False,
    )


# ---------------------------------------------------------------------------
# Benchmark report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GoalTacticianBenchmarkReport(Mapping[str, Any]):
    """Immutable GoalTacticianBenchmark@1 report wrapper."""

    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        copied = json.loads(_canonical_json(dict(self.payload)))
        if copied.get("schema") != GOAL_TACTICIAN_BENCHMARK_REPORT_SCHEMA:
            raise GoalTacticianMetricsError("unsupported benchmark report schema")
        if copied.get("interface") != GOAL_TACTICIAN_BENCHMARK_INTERFACE:
            raise GoalTacticianMetricsError("unsupported benchmark report interface")
        if copied.get("metrics_interface") != GOAL_TACTICIAN_METRICS_INTERFACE:
            raise GoalTacticianMetricsError("metrics interface mismatch")
        if copied.get("synthetic_distributions") is not False:
            raise GoalTacticianMetricsError(
                "benchmark report cannot use synthetic distributions"
            )
        if copied.get("source") != "cohort_receipts":
            raise GoalTacticianMetricsError(
                "benchmark report source must be cohort_receipts"
            )
        metrics = copied.get("metrics")
        if not isinstance(metrics, Mapping):
            raise GoalTacticianMetricsError("benchmark report metrics missing")
        hard = metrics.get("hard_gates")
        if not isinstance(hard, Mapping):
            raise GoalTacticianMetricsError("hard_gates missing from metrics")
        for name in HARD_GATE_NAMES:
            key = f"{name}_bps"
            if hard.get(key) != BASIS_POINTS and hard.get("passed") is True:
                raise GoalTacticianMetricsError(
                    f"hard gate {name} cannot pass below 100 percent"
                )
        resources = metrics.get("resources")
        if not isinstance(resources, Mapping):
            raise GoalTacticianMetricsError("resources missing from metrics")
        if resources.get("calibrated_timing") is not True:
            if resources.get("timing_role") != "observational":
                raise GoalTacticianMetricsError(
                    "uncalibrated timing must remain observational"
                )
        cache = metrics.get("cache")
        if not isinstance(cache, Mapping):
            raise GoalTacticianMetricsError("cache section missing")
        if cache.get("hits_preserve_authority_and_identity") is not True:
            raise GoalTacticianMetricsError(
                "cache hits must preserve authority and exact identity"
            )
        progress = metrics.get("progress")
        if not isinstance(progress, Mapping):
            raise GoalTacticianMetricsError("progress section missing")
        for required in (
            "unresolved_hole_ids",
            "witness_ids",
            "critical_path_step_ids",
            "budgets",
            "next_actions",
        ):
            if required not in progress:
                raise GoalTacticianMetricsError(
                    f"progress must expose {required}"
                )
        _reject_private_material(copied)
        if not copied.get("report_id"):
            identity = {k: v for k, v in copied.items() if k != "report_id"}
            copied["report_id"] = _content_id(identity, prefix="goal-tactician-bench-")
        object.__setattr__(self, "payload", copied)

    def to_dict(self) -> dict[str, Any]:
        return json.loads(_canonical_json(self.payload))

    def __getitem__(self, key: str) -> Any:
        return self.payload[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.payload)

    def __len__(self) -> int:
        return len(self.payload)

    @property
    def hard_gates_passed(self) -> bool:
        metrics = self.payload.get("metrics") or {}
        hard = metrics.get("hard_gates") or {}
        return bool(hard.get("passed"))


def build_goal_tactician_benchmark_report(
    receipts: Sequence[GoalTacticianRunReceipt | Mapping[str, Any]],
    *,
    goal_id: str = "FVT-G063",
    task_id: str = "FVT-033",
    cohort_id: str = "formal-verification-tactician/real-tool-quality",
    generated_at: str = "2026-07-30T20:00:00Z",
    notes: str = "",
) -> GoalTacticianBenchmarkReport:
    """Build a GoalTacticianBenchmark@1 report from cohort receipts."""

    metrics = derive_goal_tactician_metrics(receipts)
    metrics_dict = metrics.to_dict()
    typed_receipts = [
        item if isinstance(item, GoalTacticianRunReceipt) else GoalTacticianRunReceipt.from_dict(item)
        for item in receipts
    ]
    receipt_ids = [r.receipt_id for r in typed_receipts]
    payload = {
        "schema": GOAL_TACTICIAN_BENCHMARK_REPORT_SCHEMA,
        "interface": GOAL_TACTICIAN_BENCHMARK_INTERFACE,
        "metrics_interface": GOAL_TACTICIAN_METRICS_INTERFACE,
        "version": GOAL_TACTICIAN_BENCHMARK_VERSION,
        "goal_id": _text(goal_id, "goal_id", maximum=MAX_ID_BYTES),
        "task_id": _text(task_id, "task_id", maximum=MAX_ID_BYTES),
        "cohort_id": _text(cohort_id, "cohort_id", maximum=MAX_TEXT_BYTES),
        "generated_at": _text(generated_at, "generated_at", maximum=64),
        "source": "cohort_receipts",
        "synthetic_distributions": False,
        "conflict_policy": (
            "Own benchmark, report, and tactician metrics; do not turn "
            "unstable timing ratios or tool availability into correctness gates."
        ),
        "acceptance": {
            "metrics_from_cohort_receipts": True,
            "hard_correctness_privacy_authority_100_percent": metrics.hard_gates_passed,
            "timing_observational_unless_calibrated": (
                metrics.calibrated_timing is False
                or metrics_dict["resources"]["timing_role"]
                == "calibrated_gate_eligible"
            ),
            "cache_hits_preserve_authority_and_identity": metrics_dict["cache"][
                "hits_preserve_authority_and_identity"
            ],
            "progress_exposes_holes_witnesses_critical_path_budgets_next_actions": all(
                key in metrics_dict["progress"]
                for key in (
                    "unresolved_hole_ids",
                    "witness_ids",
                    "critical_path_step_ids",
                    "budgets",
                    "next_actions",
                )
            ),
        },
        "receipt_ids": receipt_ids,
        "receipt_count": len(receipt_ids),
        "metrics": metrics_dict,
        "gates": {
            "hard": {
                name: {
                    "required_bps": BASIS_POINTS,
                    "actual_bps": getattr(metrics, f"hard_gate_{name}_bps"),
                    "status": (
                        GateStatus.PASS.value
                        if getattr(metrics, f"hard_gate_{name}_bps") == BASIS_POINTS
                        else GateStatus.FAIL.value
                    ),
                }
                for name in HARD_GATE_NAMES
            },
            "timing": {
                "status": (
                    GateStatus.OBSERVATIONAL.value
                    if not metrics.calibrated_timing
                    else GateStatus.PASS.value
                ),
                "calibrated": metrics.calibrated_timing,
                "fields": list(OBSERVATIONAL_METRIC_NAMES),
            },
            "tool_availability": {
                "status": GateStatus.NOT_APPLICABLE.value,
                "reason": (
                    "tool availability is not a correctness gate for this "
                    "benchmark surface"
                ),
            },
        },
        "notes": _optional_text(notes, "notes", maximum=2_048),
        "bounded": True,
        "contains_prompts": False,
        "contains_proof_transcripts": False,
        "contains_private_witnesses": False,
    }
    return GoalTacticianBenchmarkReport(payload)


def architecture_benchmark_document(
    report: GoalTacticianBenchmarkReport | Mapping[str, Any],
    *,
    program: str = "formal-verification-tactician/readiness",
) -> dict[str, Any]:
    """Project a benchmark report into the architecture JSON document shape."""

    body = report.to_dict() if isinstance(report, GoalTacticianBenchmarkReport) else dict(report)
    if not isinstance(body, Mapping):
        raise GoalTacticianMetricsError("architecture document requires a report object")
    document = {
        "schema_version": "formal-verification-tactician-benchmark/v1",
        "schema": GOAL_TACTICIAN_BENCHMARK_SCHEMA,
        "interface": GOAL_TACTICIAN_BENCHMARK_INTERFACE,
        "metrics_interface": GOAL_TACTICIAN_METRICS_INTERFACE,
        "goal_id": body.get("goal_id", "FVT-G063"),
        "task_id": body.get("task_id", "FVT-033"),
        "program": program,
        "description": (
            "Receipt-derived quality, resource, cache, and observability "
            "benchmark for the formal verification goal tactician. Metrics "
            "are additive counts and recomputed rates from cohort run "
            "receipts; hard correctness/privacy/authority gates require 100 "
            "percent; timing is observational unless calibrated; cache hits "
            "preserve authority and exact identity; progress exposes "
            "unresolved holes, witnesses, critical path, budgets, and next "
            "actions."
        ),
        "generated_at": body.get("generated_at"),
        "cohort_id": body.get("cohort_id"),
        "source": body.get("source"),
        "synthetic_distributions": False,
        "conflict_policy": body.get("conflict_policy"),
        "acceptance": body.get("acceptance"),
        "metric_dimensions": [
            "formalization",
            "proof_gap_recall_precision",
            "plan_solvability",
            "proof_authority",
            "counterexample_replay_reduction_explanation",
            "provider_agreement",
            "resources",
            "cancellation",
            "cache_correctness",
            "supervisor_progress",
        ],
        "hard_gates": HARD_GATE_NAMES,
        "observational_fields": list(OBSERVATIONAL_METRIC_NAMES),
        "report": body,
    }
    _reject_private_material(document)
    return document


def fixture_cohort_receipts() -> tuple[GoalTacticianRunReceipt, ...]:
    """Deterministic fixture cohort used by the architecture benchmark document.

    These are **fixture receipts** (not live tool outcomes).  They still count
    as actual cohort receipts for metrics derivation — they are not synthetic
    distributions or random samples.
    """

    common = {
        "repository_tree_id": "sha256:fvt063-fixture-tree-0000000000000000000000000000000000000000000000000000",
        "policy_id": "policy:formal-verification-tactician@1",
        "evidence_class": EvidenceClass.FIXTURE,
        "formalization_required": True,
        "formalization_attempted": True,
        "formalization_succeeded": True,
        "false_completion": False,
        "privacy_violation": False,
        "authority_boundary_violation": False,
        "calibration_receipt_id": "",
        "cancellation_honored": True,
    }

    cold = GoalTacticianRunReceipt(
        receipt_id="receipt:fvt063:fixture:cold",
        run_id="run:fvt063:cold",
        goal_id="FVT-G063",
        provider_id="provider:z3@1",
        proof_gap_true_positive=4,
        proof_gap_false_positive=0,
        proof_gap_false_negative=1,
        proof_gap_true_negative=3,
        plan_steps_total=6,
        plan_steps_solvable=5,
        plan_admitted=True,
        claimed_assurance="kernel_verified",
        authoritative_assurance="kernel_verified",
        counterexample_count=2,
        counterexample_replayable_count=2,
        counterexample_reduced_count=2,
        counterexample_explained_count=1,
        providers_queried=("provider:z3@1", "provider:cvc5@1"),
        providers_agreeing=("provider:z3@1", "provider:cvc5@1"),
        wall_time_ms=1_200,
        cpu_time_ms=900,
        memory_peak_bytes=64 * 1024 * 1024,
        cancelled=False,
        cache_outcome=CacheOutcome.MISS,
        cache_key="",
        cache_authority_preserved=True,
        cache_identity_preserved=True,
        unresolved_hole_ids=("hole:invariant-strengthening",),
        witness_ids=("witness:cx-1", "witness:cx-2"),
        critical_path_step_ids=("step:formalize", "step:plan", "step:prove"),
        budget_cpu_ms_remaining=8_000,
        budget_memory_bytes_remaining=512 * 1024 * 1024,
        budget_token_remaining=12_000,
        next_actions=("reduce-counterexample", "rank-proof-plan"),
        **common,
    )

    warm = GoalTacticianRunReceipt(
        receipt_id="receipt:fvt063:fixture:warm",
        run_id="run:fvt063:warm",
        goal_id="FVT-G063",
        provider_id="provider:z3@1",
        proof_gap_true_positive=5,
        proof_gap_false_positive=0,
        proof_gap_false_negative=0,
        proof_gap_true_negative=3,
        plan_steps_total=6,
        plan_steps_solvable=6,
        plan_admitted=True,
        claimed_assurance="kernel_verified",
        authoritative_assurance="kernel_verified",
        counterexample_count=1,
        counterexample_replayable_count=1,
        counterexample_reduced_count=1,
        counterexample_explained_count=1,
        providers_queried=("provider:z3@1", "provider:cvc5@1"),
        providers_agreeing=("provider:z3@1", "provider:cvc5@1"),
        wall_time_ms=450,
        cpu_time_ms=300,
        memory_peak_bytes=48 * 1024 * 1024,
        cancelled=False,
        cache_outcome=CacheOutcome.HIT,
        cache_key="cache:fvt063:warm:exact-identity",
        cache_authority_preserved=True,
        cache_identity_preserved=True,
        unresolved_hole_ids=(),
        witness_ids=("witness:cx-1",),
        critical_path_step_ids=("step:formalize", "step:cache-hit"),
        budget_cpu_ms_remaining=10_000,
        budget_memory_bytes_remaining=600 * 1024 * 1024,
        budget_token_remaining=14_000,
        next_actions=("emit-completion-receipt",),
        **common,
    )

    cancelled = GoalTacticianRunReceipt(
        receipt_id="receipt:fvt063:fixture:cancelled",
        run_id="run:fvt063:cancelled",
        goal_id="FVT-G063",
        provider_id="provider:cvc5@1",
        proof_gap_true_positive=1,
        proof_gap_false_positive=0,
        proof_gap_false_negative=0,
        proof_gap_true_negative=1,
        plan_steps_total=4,
        plan_steps_solvable=2,
        plan_admitted=False,
        claimed_assurance="solver_checked",
        authoritative_assurance="solver_checked",
        counterexample_count=0,
        counterexample_replayable_count=0,
        counterexample_reduced_count=0,
        counterexample_explained_count=0,
        providers_queried=("provider:cvc5@1",),
        providers_agreeing=("provider:cvc5@1",),
        wall_time_ms=80,
        cpu_time_ms=40,
        memory_peak_bytes=16 * 1024 * 1024,
        cancelled=True,
        cache_outcome=CacheOutcome.BYPASS,
        cache_key="",
        cache_authority_preserved=True,
        cache_identity_preserved=True,
        unresolved_hole_ids=("hole:budget-exhausted",),
        witness_ids=(),
        critical_path_step_ids=("step:formalize", "step:cancel"),
        budget_cpu_ms_remaining=0,
        budget_memory_bytes_remaining=16 * 1024 * 1024,
        budget_token_remaining=100,
        next_actions=("resume-from-checkpoint", "raise-cpu-budget"),
        **common,
    )

    return (cold, warm, cancelled)


__all__ = [
    "BASIS_POINTS",
    "CacheOutcome",
    "EvidenceClass",
    "GOAL_TACTICIAN_BENCHMARK_INTERFACE",
    "GOAL_TACTICIAN_BENCHMARK_REPORT_SCHEMA",
    "GOAL_TACTICIAN_BENCHMARK_SCHEMA",
    "GOAL_TACTICIAN_METRICS_INTERFACE",
    "GOAL_TACTICIAN_METRICS_SCHEMA",
    "GOAL_TACTICIAN_PROGRESS_SCHEMA",
    "GOAL_TACTICIAN_RECEIPT_SCHEMA",
    "GateStatus",
    "HARD_GATE_NAMES",
    "OBSERVATIONAL_METRIC_NAMES",
    "GoalTacticianBenchmarkReport",
    "GoalTacticianMetrics",
    "GoalTacticianMetricsError",
    "GoalTacticianProgress",
    "GoalTacticianRunReceipt",
    "architecture_benchmark_document",
    "build_goal_tactician_benchmark_report",
    "derive_goal_tactician_metrics",
    "fixture_cohort_receipts",
]
