"""Fail-closed staged rollout controls for proof-backed pytest reuse.

The pytest configuration remains ``off`` unless an operator applies a
successful, fresh promotion decision.  This module does not mutate
configuration or provider state; it produces deterministic evidence and
decisions that an operator-controlled integration can apply.

Promotion is deliberately one stage at a time::

    off -> shadow -> read -> opt_in_readwrite -> eligible_default

Every promotion is bound to the current repository tree and reviewed policy.
Read-capable stages additionally require clean forced-rerun observations.
Safety observations are evaluated independently of promotion evidence so a
rollback cannot be prevented by a stale or malformed benchmark receipt.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Final, Optional

from .config import ProofReuseConfig, ProofReuseMode


PROOF_REUSE_ROLLOUT_VERSION: Final = 1
PROOF_REUSE_CONFIG_INTERFACE: Final = "ProofReuseConfig@1"
PROOF_REUSE_METRICS_INTERFACE: Final = "ProofReuseMetrics@1"
BENCHMARK_RECEIPT_INTERFACE: Final = "BenchmarkReceipt@1"
PROOF_REUSE_BENCHMARK_INTERFACE: Final = "ProofReuseBenchmark@1"
PROOF_REUSE_ROLLOUT_DECISION_INTERFACE: Final = "ProofReuseRolloutDecision@1"
PROOF_REUSE_ROLLOUT_EVIDENCE_SCHEMA: Final = (
    "ipfs_accelerate_py/testing/proof-reuse-rollout-evidence@1"
)
PROOF_REUSE_ROLLOUT_DECISION_SCHEMA: Final = (
    "ipfs_accelerate_py/testing/proof-reuse-rollout-decision@1"
)
PROOF_REUSE_ROLLBACK_DECISION_SCHEMA: Final = (
    "ipfs_accelerate_py/testing/proof-reuse-rollback-decision@1"
)
PROOF_REUSE_FORCED_RERUN_SCHEMA: Final = (
    "ipfs_accelerate_py/testing/proof-reuse-forced-rerun@1"
)
PROOF_REUSE_ROLLOUT_REQUIREMENT_ID: Final = "ptr/rollout-decision@1"

MAX_TEXT_BYTES: Final = 512
MAX_EVIDENCE_BYTES: Final = 512 * 1024
MAX_COUNTER: Final = (1 << 63) - 1
MAX_SAMPLE_RATE_BPS: Final = 10_000


class ProofReuseRolloutError(ValueError):
    """Rollout policy, evidence, or observation is malformed."""


class ProofReuseRolloutStage(str, Enum):
    """Closed, ordered rollout vocabulary."""

    OFF = "off"
    SHADOW = "shadow"
    READ = "read"
    OPT_IN_READWRITE = "opt_in_readwrite"
    ELIGIBLE_DEFAULT = "eligible_default"

    @property
    def rank(self) -> int:
        return _STAGE_ORDER.index(self)


_STAGE_ORDER: Final = tuple(ProofReuseRolloutStage)


class RolloutDisposition(str, Enum):
    PROMOTE = "promote"
    HOLD = "hold"


class ForcedRerunOutcome(str, Enum):
    """Outcome vocabulary shared by predictions and actual executions."""

    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"
    SKIP = "skip"


class RollbackTrigger(str, Enum):
    FALSE_SKIP = "false_skip"
    AUTHORITY_CONTRADICTION = "authority_contradiction"
    CORRUPTION_SPIKE = "corruption_spike"
    STALE_KEY = "stale_key"
    UNEXPLAINED_MISMATCH = "unexplained_mismatch"


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _plain(value.to_dict())
    return value


def _canonical_bytes(value: Any) -> bytes:
    try:
        result = json.dumps(
            _plain(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProofReuseRolloutError(
            "rollout data must be canonical JSON"
        ) from exc
    if len(result) > MAX_EVIDENCE_BYTES:
        raise ProofReuseRolloutError("rollout data exceeds its byte bound")
    return result


def _content_id(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _text(value: Any, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ProofReuseRolloutError(f"{name} must be text")
    if value != value.strip() or "\x00" in value:
        raise ProofReuseRolloutError(f"{name} must be canonical safe text")
    if not allow_empty and not value:
        raise ProofReuseRolloutError(f"{name} must not be empty")
    if len(value.encode("utf-8")) > MAX_TEXT_BYTES:
        raise ProofReuseRolloutError(f"{name} exceeds its byte bound")
    return value


def _optional_text(value: Any, name: str) -> str:
    return _text(value, name, allow_empty=True)


def _counter(value: Any, name: str, *, optional: bool = False) -> Optional[int]:
    if optional and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProofReuseRolloutError(f"{name} must be a non-negative integer")
    if value < 0 or value > MAX_COUNTER:
        raise ProofReuseRolloutError(f"{name} is out of bounds")
    return value


def _optional_bool(value: Any, name: str) -> Optional[bool]:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ProofReuseRolloutError(f"{name} must be a boolean")
    return value


def _timestamp(value: datetime | str, name: str) -> str:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ProofReuseRolloutError(f"{name} is invalid") from exc
    else:
        raise ProofReuseRolloutError(f"{name} must be a timestamp")
    if parsed.tzinfo is None:
        raise ProofReuseRolloutError(f"{name} must include a timezone")
    return (
        parsed.astimezone(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _datetime(value: datetime | str, name: str) -> datetime:
    return datetime.fromisoformat(_timestamp(value, name).replace("Z", "+00:00"))


def _stage(value: Any, name: str = "stage") -> ProofReuseRolloutStage:
    if isinstance(value, ProofReuseRolloutStage):
        return value
    try:
        return ProofReuseRolloutStage(str(getattr(value, "value", value)))
    except (TypeError, ValueError) as exc:
        raise ProofReuseRolloutError(f"{name} is invalid") from exc


def _outcome(value: Any, name: str) -> ForcedRerunOutcome:
    if isinstance(value, ForcedRerunOutcome):
        return value
    if isinstance(value, bool):
        return ForcedRerunOutcome.PASS if value else ForcedRerunOutcome.FAIL
    try:
        return ForcedRerunOutcome(str(getattr(value, "value", value)))
    except (TypeError, ValueError) as exc:
        raise ProofReuseRolloutError(f"{name} is invalid") from exc


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        raise ProofReuseRolloutError(f"{name} must be a mapping")
    # Round-trip through canonical JSON to detach mutable caller-owned values.
    return json.loads(_canonical_bytes(value).decode("utf-8"))


@dataclass(frozen=True)
class ForcedRerunObservation:
    """Privacy-safe comparison of one predicted and actual test outcome."""

    sample_id: str
    predicted_outcome: ForcedRerunOutcome | str
    actual_outcome: ForcedRerunOutcome | str
    mismatch_explained: bool = False
    explanation_code: str = ""
    authority_contradiction: bool = False

    def __post_init__(self) -> None:
        sample_id = _text(self.sample_id, "sample_id")
        if not sample_id.startswith("sha256:") or len(sample_id) != 71:
            raise ProofReuseRolloutError("sample_id must be a sha256 identity")
        try:
            int(sample_id[7:], 16)
        except ValueError as exc:
            raise ProofReuseRolloutError(
                "sample_id must be a sha256 identity"
            ) from exc
        object.__setattr__(self, "sample_id", sample_id)
        object.__setattr__(
            self,
            "predicted_outcome",
            _outcome(self.predicted_outcome, "predicted_outcome"),
        )
        object.__setattr__(
            self, "actual_outcome", _outcome(self.actual_outcome, "actual_outcome")
        )
        if not isinstance(self.mismatch_explained, bool):
            raise ProofReuseRolloutError("mismatch_explained must be a boolean")
        if not isinstance(self.authority_contradiction, bool):
            raise ProofReuseRolloutError(
                "authority_contradiction must be a boolean"
            )
        code = _optional_text(self.explanation_code, "explanation_code")
        if self.mismatch_explained and not code:
            raise ProofReuseRolloutError(
                "an explained mismatch requires an explanation code"
            )
        if not self.mismatch and (self.mismatch_explained or code):
            raise ProofReuseRolloutError(
                "matching outcomes cannot carry a mismatch explanation"
            )
        object.__setattr__(self, "explanation_code", code)

    @property
    def mismatch(self) -> bool:
        return self.predicted_outcome is not self.actual_outcome

    @property
    def false_skip(self) -> bool:
        """A predicted reusable pass that did not actually pass."""

        return (
            self.predicted_outcome is ForcedRerunOutcome.PASS
            and self.actual_outcome is not ForcedRerunOutcome.PASS
        )

    @property
    def unexplained_mismatch(self) -> bool:
        return self.mismatch and not self.mismatch_explained

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROOF_REUSE_FORCED_RERUN_SCHEMA,
            "sample_id": self.sample_id,
            "predicted_outcome": self.predicted_outcome.value,
            "actual_outcome": self.actual_outcome.value,
            "mismatch": self.mismatch,
            "false_skip": self.false_skip,
            "unexplained_mismatch": self.unexplained_mismatch,
            "mismatch_explained": self.mismatch_explained,
            "explanation_code": self.explanation_code,
            "authority_contradiction": self.authority_contradiction,
        }


@dataclass(frozen=True)
class ForcedRerunSummary:
    """Aggregate-only sampling evidence suitable for rollout telemetry."""

    selected: int = 0
    completed: int = 0
    matched: int = 0
    false_skips: int = 0
    explained_mismatches: int = 0
    unexplained_mismatches: int = 0
    authority_contradictions: int = 0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(
                self, name, _counter(getattr(self, name), name)
            )
        if self.completed > self.selected:
            raise ProofReuseRolloutError("completed samples exceed selected samples")
        classified = (
            self.matched
            + self.explained_mismatches
            + self.unexplained_mismatches
        )
        if classified != self.completed:
            raise ProofReuseRolloutError(
                "completed samples must have exactly one comparison result"
            )
        if self.false_skips > (
            self.explained_mismatches + self.unexplained_mismatches
        ):
            raise ProofReuseRolloutError(
                "false skips cannot exceed total mismatches"
            )
        if self.authority_contradictions > self.completed:
            raise ProofReuseRolloutError(
                "authority contradictions cannot exceed completed samples"
            )

    @property
    def clean(self) -> bool:
        return (
            self.selected == self.completed
            and self.false_skips == 0
            and self.unexplained_mismatches == 0
            and self.authority_contradictions == 0
        )

    @classmethod
    def from_observations(
        cls,
        observations: Iterable[ForcedRerunObservation],
        *,
        selected: Optional[int] = None,
    ) -> "ForcedRerunSummary":
        items = tuple(observations)
        if any(not isinstance(item, ForcedRerunObservation) for item in items):
            raise ProofReuseRolloutError(
                "forced-rerun observations have the wrong type"
            )
        chosen = len(items) if selected is None else _counter(selected, "selected")
        if chosen < len(items):
            raise ProofReuseRolloutError("selected samples are fewer than observations")
        return cls(
            selected=chosen,
            completed=len(items),
            matched=sum(not item.mismatch for item in items),
            false_skips=sum(item.false_skip for item in items),
            explained_mismatches=sum(
                item.mismatch and item.mismatch_explained for item in items
            ),
            unexplained_mismatches=sum(
                item.unexplained_mismatch for item in items
            ),
            authority_contradictions=sum(
                item.authority_contradiction for item in items
            ),
        )

    def to_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ForcedRerunSummary":
        if not isinstance(value, Mapping) or set(value) != set(cls.__dataclass_fields__):
            raise ProofReuseRolloutError("forced-rerun summary fields are invalid")
        return cls(**dict(value))


@dataclass(frozen=True)
class ForcedRerunSampler:
    """Deterministic, stable sampler for verified reuse candidates.

    The caller's execution identity is never included in observations or
    telemetry.  A keyed digest makes the selection stable across processes and
    xdist workers without relying on Python's randomized ``hash()``.
    """

    sample_rate_bps: int
    seed: str

    def __post_init__(self) -> None:
        rate = _counter(self.sample_rate_bps, "sample_rate_bps")
        if rate > MAX_SAMPLE_RATE_BPS:
            raise ProofReuseRolloutError("sample_rate_bps cannot exceed 10000")
        object.__setattr__(self, "sample_rate_bps", rate)
        object.__setattr__(self, "seed", _text(self.seed, "seed"))

    def sample_id(self, execution_identity: str) -> str:
        identity = _text(execution_identity, "execution_identity")
        return _content_id(
            {
                "domain": "proof-reuse-forced-rerun@1",
                "seed": self.seed,
                "execution_identity": identity,
            }
        )

    def should_sample(self, execution_identity: str) -> bool:
        sample_id = self.sample_id(execution_identity)
        bucket = int(sample_id[7:23], 16) % 10_000
        return bucket < self.sample_rate_bps

    # Operator-facing terminology used in the runbook.
    should_force_rerun = should_sample

    def compare(
        self,
        execution_identity: str,
        predicted_outcome: ForcedRerunOutcome | str | bool,
        actual_outcome: ForcedRerunOutcome | str | bool,
        *,
        mismatch_explained: bool = False,
        explanation_code: str = "",
        authority_contradiction: bool = False,
    ) -> ForcedRerunObservation:
        """Compare a selected candidate's prediction with real execution."""

        if not self.should_sample(execution_identity):
            raise ProofReuseRolloutError(
                "cannot compare an identity that was not selected"
            )
        return ForcedRerunObservation(
            sample_id=self.sample_id(execution_identity),
            predicted_outcome=predicted_outcome,
            actual_outcome=actual_outcome,
            mismatch_explained=mismatch_explained,
            explanation_code=explanation_code,
            authority_contradiction=authority_contradiction,
        )

    compare_predicted_actual = compare


@dataclass(frozen=True)
class ProofReusePromotionEvidence:
    """Fresh, exact evidence packet for one adjacent promotion."""

    observed_at: datetime | str
    repository_id: str
    tree_id: str
    policy_id: str
    policy_revision: str
    current_stage: ProofReuseRolloutStage | str
    target_stage: ProofReuseRolloutStage | str
    benchmark_receipt: Any = None
    metrics_snapshot: Any = None
    forced_reruns: ForcedRerunSummary | Mapping[str, Any] | None = None
    mutation_false_skips: Optional[int] = None
    degradation_false_skips: Optional[int] = None
    authority_contradictions: Optional[int] = None
    corruption_spike: Optional[bool] = None
    stale_keys: Optional[int] = None
    key_health_ok: Optional[bool] = None
    revocation_health_ok: Optional[bool] = None
    operator_approval_id: str = ""
    controlled_issuer: Optional[bool] = None
    current_tree_gate_passed: Optional[bool] = None
    all_repositories_passed: Optional[bool] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "observed_at", _timestamp(self.observed_at, "observed_at")
        )
        for name in ("repository_id", "tree_id", "policy_id", "policy_revision"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self, "current_stage", _stage(self.current_stage, "current_stage")
        )
        object.__setattr__(
            self, "target_stage", _stage(self.target_stage, "target_stage")
        )
        if self.benchmark_receipt is not None:
            object.__setattr__(
                self,
                "benchmark_receipt",
                _mapping(self.benchmark_receipt, "benchmark_receipt"),
            )
        if self.metrics_snapshot is not None:
            object.__setattr__(
                self,
                "metrics_snapshot",
                _mapping(self.metrics_snapshot, "metrics_snapshot"),
            )
        reruns = self.forced_reruns
        if isinstance(reruns, Mapping):
            reruns = ForcedRerunSummary.from_dict(reruns)
        if reruns is not None and not isinstance(reruns, ForcedRerunSummary):
            raise ProofReuseRolloutError("forced_reruns has the wrong type")
        object.__setattr__(self, "forced_reruns", reruns)
        for name in (
            "mutation_false_skips",
            "degradation_false_skips",
            "authority_contradictions",
            "stale_keys",
        ):
            object.__setattr__(
                self,
                name,
                _counter(getattr(self, name), name, optional=True),
            )
        for name in (
            "corruption_spike",
            "key_health_ok",
            "revocation_health_ok",
            "controlled_issuer",
            "current_tree_gate_passed",
            "all_repositories_passed",
        ):
            object.__setattr__(
                self, name, _optional_bool(getattr(self, name), name)
            )
        object.__setattr__(
            self,
            "operator_approval_id",
            _optional_text(self.operator_approval_id, "operator_approval_id"),
        )

    @property
    def evidence_id(self) -> str:
        return _content_id(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROOF_REUSE_ROLLOUT_EVIDENCE_SCHEMA,
            "version": PROOF_REUSE_ROLLOUT_VERSION,
            "requirement_id": PROOF_REUSE_ROLLOUT_REQUIREMENT_ID,
            "observed_at": self.observed_at,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
            "current_stage": self.current_stage.value,
            "target_stage": self.target_stage.value,
            "benchmark_receipt": _plain(self.benchmark_receipt),
            "metrics_snapshot": _plain(self.metrics_snapshot),
            "forced_reruns": (
                self.forced_reruns.to_dict()
                if self.forced_reruns is not None
                else None
            ),
            "mutation_false_skips": self.mutation_false_skips,
            "degradation_false_skips": self.degradation_false_skips,
            "authority_contradictions": self.authority_contradictions,
            "corruption_spike": self.corruption_spike,
            "stale_keys": self.stale_keys,
            "key_health_ok": self.key_health_ok,
            "revocation_health_ok": self.revocation_health_ok,
            "operator_approval_id": self.operator_approval_id,
            "controlled_issuer": self.controlled_issuer,
            "current_tree_gate_passed": self.current_tree_gate_passed,
            "all_repositories_passed": self.all_repositories_passed,
        }


@dataclass(frozen=True)
class RolloutGate:
    name: str
    passed: bool
    detail: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, "gate name"))
        if not isinstance(self.passed, bool):
            raise ProofReuseRolloutError("gate passed must be a boolean")
        object.__setattr__(self, "detail", _mapping(self.detail, "gate detail"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": _plain(self.detail),
        }


@dataclass(frozen=True)
class ProofReuseRolloutDecision:
    """Immutable promotion decision; evidence only, never authority itself."""

    current_stage: ProofReuseRolloutStage
    requested_stage: ProofReuseRolloutStage
    effective_stage: ProofReuseRolloutStage
    disposition: RolloutDisposition
    gates: tuple[RolloutGate, ...]
    evidence_id: str
    policy_id: str
    policy_revision: str

    @property
    def promoted(self) -> bool:
        return self.disposition is RolloutDisposition.PROMOTE

    @property
    def reason_codes(self) -> tuple[str, ...]:
        return tuple(gate.name for gate in self.gates if not gate.passed)

    @property
    def decision_id(self) -> str:
        return _content_id(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROOF_REUSE_ROLLOUT_DECISION_SCHEMA,
            "interface": PROOF_REUSE_ROLLOUT_DECISION_INTERFACE,
            "version": PROOF_REUSE_ROLLOUT_VERSION,
            "requirement_id": PROOF_REUSE_ROLLOUT_REQUIREMENT_ID,
            "current_stage": self.current_stage.value,
            "requested_stage": self.requested_stage.value,
            "effective_stage": self.effective_stage.value,
            "disposition": self.disposition.value,
            "gates": [gate.to_dict() for gate in self.gates],
            "reason_codes": list(self.reason_codes),
            "evidence_id": self.evidence_id,
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
        }

    def to_json(self) -> str:
        return _canonical_bytes(self.to_dict()).decode("utf-8")


@dataclass(frozen=True)
class ProofReuseSafetySignals:
    """Current safety counters; missing values are intentionally invalid."""

    false_skips: Optional[int] = None
    authority_contradictions: Optional[int] = None
    corruption_spike: Optional[bool] = None
    stale_keys: Optional[int] = None
    unexplained_mismatches: Optional[int] = None

    def __post_init__(self) -> None:
        for name in (
            "false_skips",
            "authority_contradictions",
            "stale_keys",
            "unexplained_mismatches",
        ):
            object.__setattr__(
                self,
                name,
                _counter(getattr(self, name), name, optional=True),
            )
        object.__setattr__(
            self,
            "corruption_spike",
            _optional_bool(self.corruption_spike, "corruption_spike"),
        )

    @classmethod
    def from_forced_reruns(
        cls,
        summary: ForcedRerunSummary,
        *,
        corruption_spike: Optional[bool] = None,
        stale_keys: Optional[int] = None,
    ) -> "ProofReuseSafetySignals":
        if not isinstance(summary, ForcedRerunSummary):
            raise ProofReuseRolloutError("summary has the wrong type")
        return cls(
            false_skips=summary.false_skips,
            authority_contradictions=summary.authority_contradictions,
            corruption_spike=corruption_spike,
            stale_keys=stale_keys,
            unexplained_mismatches=summary.unexplained_mismatches,
        )

    @property
    def complete(self) -> bool:
        return all(
            getattr(self, name) is not None for name in self.__dataclass_fields__
        )

    @property
    def triggers(self) -> tuple[RollbackTrigger, ...]:
        result = []
        if self.false_skips:
            result.append(RollbackTrigger.FALSE_SKIP)
        if self.authority_contradictions:
            result.append(RollbackTrigger.AUTHORITY_CONTRADICTION)
        if self.corruption_spike:
            result.append(RollbackTrigger.CORRUPTION_SPIKE)
        if self.stale_keys:
            result.append(RollbackTrigger.STALE_KEY)
        if self.unexplained_mismatches:
            result.append(RollbackTrigger.UNEXPLAINED_MISMATCH)
        if not self.complete and RollbackTrigger.UNEXPLAINED_MISMATCH not in result:
            # Missing monitoring evidence cannot justify remaining in an active
            # mode. Treat it as an unexplained safety mismatch without masking
            # any severe signal that was present in the partial observation.
            result.append(RollbackTrigger.UNEXPLAINED_MISMATCH)
        return tuple(result)


@dataclass(frozen=True)
class ProofReuseRollbackDecision:
    """Automatic rollback result for current safety signals."""

    current_stage: ProofReuseRolloutStage
    effective_stage: ProofReuseRolloutStage
    triggers: tuple[RollbackTrigger, ...]
    automatic: bool = True

    @classmethod
    def evaluate(
        cls,
        current_stage: ProofReuseRolloutStage | str,
        signals: ProofReuseSafetySignals,
    ) -> "ProofReuseRollbackDecision":
        stage = _stage(current_stage, "current_stage")
        if not isinstance(signals, ProofReuseSafetySignals):
            raise ProofReuseRolloutError("signals have the wrong type")
        triggers = signals.triggers
        severe = {
            RollbackTrigger.FALSE_SKIP,
            RollbackTrigger.AUTHORITY_CONTRADICTION,
            RollbackTrigger.STALE_KEY,
        }
        if not triggers:
            target = stage
        elif stage is ProofReuseRolloutStage.OFF:
            target = ProofReuseRolloutStage.OFF
        elif severe.intersection(triggers):
            target = ProofReuseRolloutStage.OFF
        elif stage is ProofReuseRolloutStage.SHADOW:
            target = ProofReuseRolloutStage.OFF
        else:
            target = ProofReuseRolloutStage.SHADOW
        return cls(
            current_stage=stage,
            effective_stage=target,
            triggers=triggers,
        )

    @property
    def triggered(self) -> bool:
        return bool(self.triggers)

    @property
    def rolled_back(self) -> bool:
        return self.effective_stage.rank < self.current_stage.rank

    @property
    def decision_id(self) -> str:
        return _content_id(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROOF_REUSE_ROLLBACK_DECISION_SCHEMA,
            "interface": PROOF_REUSE_ROLLOUT_DECISION_INTERFACE,
            "version": PROOF_REUSE_ROLLOUT_VERSION,
            "requirement_id": PROOF_REUSE_ROLLOUT_REQUIREMENT_ID,
            "current_stage": self.current_stage.value,
            "effective_stage": self.effective_stage.value,
            "triggers": [trigger.value for trigger in self.triggers],
            "automatic": self.automatic,
            "triggered": self.triggered,
            "rolled_back": self.rolled_back,
        }


@dataclass(frozen=True)
class ProofReuseRolloutPolicy:
    """Reviewed promotion, sampling, and rollback policy.

    The default policy approves only ``off`` and ``shadow``.  Read authority
    and eligible-default operation therefore require an explicitly reviewed
    policy as well as a fresh evidence packet.
    """

    policy_id: str = "proof-reuse-rollout-v1"
    policy_revision: str = "1"
    approved_stages: tuple[ProofReuseRolloutStage | str, ...] = (
        ProofReuseRolloutStage.OFF,
        ProofReuseRolloutStage.SHADOW,
    )
    max_evidence_age_seconds: int = 24 * 60 * 60
    max_future_skew_seconds: int = 60
    min_forced_reruns: int = 1
    forced_rerun_sample_bps: int = 100
    allow_eligible_default: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _text(self.policy_id, "policy_id"))
        object.__setattr__(
            self,
            "policy_revision",
            _text(self.policy_revision, "policy_revision"),
        )
        stages = tuple(
            sorted({_stage(item) for item in self.approved_stages}, key=lambda x: x.rank)
        )
        if ProofReuseRolloutStage.OFF not in stages:
            raise ProofReuseRolloutError("approved stages must include off")
        object.__setattr__(self, "approved_stages", stages)
        for name in (
            "max_evidence_age_seconds",
            "max_future_skew_seconds",
            "min_forced_reruns",
            "forced_rerun_sample_bps",
        ):
            value = _counter(getattr(self, name), name)
            object.__setattr__(self, name, value)
        if self.max_evidence_age_seconds <= 0:
            raise ProofReuseRolloutError(
                "max_evidence_age_seconds must be positive"
            )
        if self.forced_rerun_sample_bps > MAX_SAMPLE_RATE_BPS:
            raise ProofReuseRolloutError(
                "forced_rerun_sample_bps cannot exceed 10000"
            )
        if not isinstance(self.allow_eligible_default, bool):
            raise ProofReuseRolloutError(
                "allow_eligible_default must be a boolean"
            )

    @property
    def default_stage(self) -> ProofReuseRolloutStage:
        """Unpromoted policy is always off, regardless of approved stages."""

        return ProofReuseRolloutStage.OFF

    @property
    def policy_binding_id(self) -> str:
        return _content_id(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface": PROOF_REUSE_CONFIG_INTERFACE,
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
            "approved_stages": [stage.value for stage in self.approved_stages],
            "max_evidence_age_seconds": self.max_evidence_age_seconds,
            "max_future_skew_seconds": self.max_future_skew_seconds,
            "min_forced_reruns": self.min_forced_reruns,
            "forced_rerun_sample_bps": self.forced_rerun_sample_bps,
            "allow_eligible_default": self.allow_eligible_default,
            "default_stage": self.default_stage.value,
        }

    def sampler(self, *, seed: str) -> ForcedRerunSampler:
        return ForcedRerunSampler(
            sample_rate_bps=self.forced_rerun_sample_bps,
            seed=seed,
        )

    def mode_for(
        self,
        stage: ProofReuseRolloutStage | str,
        *,
        eligible: bool = False,
        readwrite_opt_in: bool = False,
    ) -> ProofReuseMode:
        """Narrow a promoted stage to a concrete plugin mode.

        Ineligible tests remain off.  Eligible-default grants read authority,
        never write authority; writes always retain the controlled opt-in.
        """

        selected = _stage(stage)
        if not isinstance(eligible, bool) or not isinstance(readwrite_opt_in, bool):
            raise ProofReuseRolloutError("eligibility flags must be booleans")
        if selected is ProofReuseRolloutStage.OFF:
            return ProofReuseMode.OFF
        if selected is ProofReuseRolloutStage.SHADOW:
            return ProofReuseMode.SHADOW
        if not eligible:
            return ProofReuseMode.OFF
        if (
            selected.rank >= ProofReuseRolloutStage.OPT_IN_READWRITE.rank
            and readwrite_opt_in
        ):
            return ProofReuseMode.READWRITE
        return ProofReuseMode.READ

    def config_for(
        self,
        stage: ProofReuseRolloutStage | str,
        *,
        eligible: bool = False,
        readwrite_opt_in: bool = False,
    ) -> ProofReuseConfig:
        return ProofReuseConfig(
            mode=self.mode_for(
                stage,
                eligible=eligible,
                readwrite_opt_in=readwrite_opt_in,
            ),
            source=f"rollout:{_stage(stage).value}",
        )

    def evaluate_promotion(
        self,
        evidence: ProofReusePromotionEvidence,
        *,
        current_stage: ProofReuseRolloutStage | str | None = None,
        target_stage: ProofReuseRolloutStage | str | None = None,
        current_repository_id: str | None = None,
        current_tree_id: str | None = None,
        now: datetime | str | None = None,
    ) -> ProofReuseRolloutDecision:
        """Evaluate every non-waivable gate for one adjacent promotion."""

        if not isinstance(evidence, ProofReusePromotionEvidence):
            raise ProofReuseRolloutError("evidence has the wrong type")
        current = (
            evidence.current_stage
            if current_stage is None
            else _stage(current_stage, "current_stage")
        )
        target = (
            evidence.target_stage
            if target_stage is None
            else _stage(target_stage, "target_stage")
        )
        now_dt = (
            datetime.now(timezone.utc)
            if now is None
            else _datetime(now, "now")
        )
        observed = _datetime(evidence.observed_at, "observed_at")
        age_seconds = (now_dt - observed).total_seconds()
        expected_repository = (
            ""
            if current_repository_id is None
            else _text(current_repository_id, "current_repository_id")
        )
        expected_tree = (
            ""
            if current_tree_id is None
            else _text(current_tree_id, "current_tree_id")
        )

        benchmark = evidence.benchmark_receipt
        benchmark_ok = (
            isinstance(benchmark, Mapping)
            and benchmark.get("interface") == BENCHMARK_RECEIPT_INTERFACE
            and benchmark.get("benchmark_interface")
            == PROOF_REUSE_BENCHMARK_INTERFACE
            and benchmark.get("metrics_interface") == PROOF_REUSE_METRICS_INTERFACE
            and benchmark.get("passed") is True
            and benchmark.get("false_admissions") == 0
            and isinstance(benchmark.get("gates"), list)
            and bool(benchmark.get("gates"))
            and all(
                isinstance(gate, Mapping) and gate.get("passed") is True
                for gate in benchmark.get("gates", ())
            )
        )
        metrics = evidence.metrics_snapshot
        metrics_ok = (
            isinstance(metrics, Mapping)
            and metrics.get("interface") == PROOF_REUSE_METRICS_INTERFACE
            and isinstance(metrics.get("counts"), Mapping)
        )
        reruns_required = target.rank >= ProofReuseRolloutStage.READ.rank
        reruns = evidence.forced_reruns
        reruns_ok = (
            not reruns_required
            or (
                isinstance(reruns, ForcedRerunSummary)
                and reruns.completed >= self.min_forced_reruns
                and reruns.clean
            )
        )
        write_required = (
            target.rank >= ProofReuseRolloutStage.OPT_IN_READWRITE.rank
        )
        default_required = target is ProofReuseRolloutStage.ELIGIBLE_DEFAULT

        gates = (
            RolloutGate(
                "adjacent_stage",
                target.rank == current.rank + 1
                and evidence.current_stage is current
                and evidence.target_stage is target,
                {"current": current.value, "target": target.value},
            ),
            RolloutGate(
                "policy_approved",
                target in self.approved_stages,
                {"target": target.value},
            ),
            RolloutGate(
                "evidence_fresh",
                (
                    age_seconds >= -self.max_future_skew_seconds
                    and age_seconds <= self.max_evidence_age_seconds
                ),
                {
                    "age_seconds": round(age_seconds, 6),
                    "max_age_seconds": self.max_evidence_age_seconds,
                    "max_future_skew_seconds": self.max_future_skew_seconds,
                },
            ),
            RolloutGate(
                "policy_binding_current",
                (
                    evidence.policy_id == self.policy_id
                    and evidence.policy_revision == self.policy_revision
                ),
                {
                    "policy_id": evidence.policy_id,
                    "policy_revision": evidence.policy_revision,
                },
            ),
            RolloutGate(
                "deployment_binding_current",
                (
                    bool(expected_repository)
                    and bool(expected_tree)
                    and evidence.repository_id == expected_repository
                    and evidence.tree_id == expected_tree
                ),
                {
                    "repository_matches": (
                        bool(expected_repository)
                        and evidence.repository_id == expected_repository
                    ),
                    "tree_matches": (
                        bool(expected_tree) and evidence.tree_id == expected_tree
                    ),
                },
            ),
            RolloutGate(
                "operator_approved",
                bool(evidence.operator_approval_id),
                {"approval_present": bool(evidence.operator_approval_id)},
            ),
            RolloutGate("benchmark_passed", benchmark_ok),
            RolloutGate("metrics_interface_current", metrics_ok),
            RolloutGate(
                "mutation_degradation_clean",
                (
                    evidence.mutation_false_skips == 0
                    and evidence.degradation_false_skips == 0
                ),
                {
                    "mutation_false_skips": evidence.mutation_false_skips,
                    "degradation_false_skips": evidence.degradation_false_skips,
                },
            ),
            RolloutGate(
                "forced_reruns_clean",
                reruns_ok,
                {
                    "required": reruns_required,
                    "minimum": self.min_forced_reruns,
                    "completed": reruns.completed if reruns else None,
                },
            ),
            RolloutGate(
                "authority_consistent",
                evidence.authority_contradictions == 0
                and (reruns is None or reruns.authority_contradictions == 0),
                {
                    "authority_contradictions": evidence.authority_contradictions
                },
            ),
            RolloutGate(
                "corruption_stable",
                evidence.corruption_spike is False,
                {"corruption_spike": evidence.corruption_spike},
            ),
            RolloutGate(
                "key_revocation_healthy",
                (
                    evidence.stale_keys == 0
                    and evidence.key_health_ok is True
                    and evidence.revocation_health_ok is True
                ),
                {
                    "stale_keys": evidence.stale_keys,
                    "key_health_ok": evidence.key_health_ok,
                    "revocation_health_ok": evidence.revocation_health_ok,
                },
            ),
            RolloutGate(
                "controlled_issuer",
                not write_required or evidence.controlled_issuer is True,
                {
                    "required": write_required,
                    "controlled_issuer": evidence.controlled_issuer,
                },
            ),
            RolloutGate(
                "eligible_default_current_tree",
                (
                    not default_required
                    or (
                        self.allow_eligible_default
                        and evidence.current_tree_gate_passed is True
                        and evidence.all_repositories_passed is True
                    )
                ),
                {
                    "required": default_required,
                    "policy_allows": self.allow_eligible_default,
                    "current_tree_gate_passed": evidence.current_tree_gate_passed,
                    "all_repositories_passed": evidence.all_repositories_passed,
                },
            ),
        )
        promoted = all(gate.passed for gate in gates)
        return ProofReuseRolloutDecision(
            current_stage=current,
            requested_stage=target,
            effective_stage=target if promoted else current,
            disposition=(
                RolloutDisposition.PROMOTE
                if promoted
                else RolloutDisposition.HOLD
            ),
            gates=gates,
            evidence_id=evidence.evidence_id,
            policy_id=self.policy_id,
            policy_revision=self.policy_revision,
        )

    def evaluate_rollback(
        self,
        current_stage: ProofReuseRolloutStage | str,
        signals: ProofReuseSafetySignals,
    ) -> ProofReuseRollbackDecision:
        return ProofReuseRollbackDecision.evaluate(current_stage, signals)


# Concise alias for callers that already operate in the proof-reuse namespace.
RolloutStage = ProofReuseRolloutStage


__all__ = [
    "BENCHMARK_RECEIPT_INTERFACE",
    "MAX_SAMPLE_RATE_BPS",
    "PROOF_REUSE_BENCHMARK_INTERFACE",
    "PROOF_REUSE_CONFIG_INTERFACE",
    "PROOF_REUSE_FORCED_RERUN_SCHEMA",
    "PROOF_REUSE_METRICS_INTERFACE",
    "PROOF_REUSE_ROLLBACK_DECISION_SCHEMA",
    "PROOF_REUSE_ROLLOUT_DECISION_INTERFACE",
    "PROOF_REUSE_ROLLOUT_DECISION_SCHEMA",
    "PROOF_REUSE_ROLLOUT_EVIDENCE_SCHEMA",
    "PROOF_REUSE_ROLLOUT_REQUIREMENT_ID",
    "PROOF_REUSE_ROLLOUT_VERSION",
    "ForcedRerunObservation",
    "ForcedRerunOutcome",
    "ForcedRerunSampler",
    "ForcedRerunSummary",
    "ProofReusePromotionEvidence",
    "ProofReuseRollbackDecision",
    "ProofReuseRolloutDecision",
    "ProofReuseRolloutError",
    "ProofReuseRolloutPolicy",
    "ProofReuseRolloutStage",
    "ProofReuseSafetySignals",
    "RollbackTrigger",
    "RolloutDisposition",
    "RolloutGate",
    "RolloutStage",
]
