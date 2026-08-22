# ruff: noqa: UP042 - the package retains Python 3.8 compatibility
"""Offline held-out and counterfactual evaluation for shadow route policies.

``RoutePolicyEvaluation@1`` scores a frozen :class:`RoutePolicyCandidate`
against logged comparison and propensity evidence.  Evaluation is emit-only:
it never mutates production policy, never authorizes promotion, and never
fabricates an improvement claim when comparison or propensity evidence is
missing.

Missing comparison evidence or missing/non-positive logged propensity returns
exactly ``insufficient_counterfactual_evidence``.  That result is
promotion-ineligible.  APMC-019 remains the sole promotion CAS owner.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from types import MappingProxyType
from typing import Any, Final

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
    AutonomyContractError,
    MetaAction,
    PolicyObservation,
    RoutePolicyCandidate,
    TerminalStatus,
)

ROUTE_POLICY_EVALUATION_INTERFACE: Final[str] = "RoutePolicyEvaluation@1"
ROUTE_POLICY_EVALUATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/route-policy-evaluation@1"
)
LOGGED_DECISION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/logged-decision@1"
)
COMPARISON_EVIDENCE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/comparison-evidence@1"
)
ROUTE_POLICY_EVALUATION_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/route-policy-evaluation-result@1"
)

INSUFFICIENT_COUNTERFACTUAL_EVIDENCE: Final[str] = "insufficient_counterfactual_evidence"
BASIS_POINTS: Final[int] = 10_000
MAX_EVALUATION_ITEMS: Final[int] = MAX_SEQUENCE_ITEMS

PROTECTED_POLICY_AXES: Final[tuple[str, ...]] = (
    "authority",
    "confirmation",
    "privacy",
    "proof",
    "provider",
    "validation",
)

_FORBIDDEN_FIELD_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "authorization",
        "chain_of_thought",
        "chat_messages",
        "completion",
        "cookie",
        "credential",
        "decoded_source",
        "executable_code",
        "hidden_reasoning",
        "messages",
        "model_output",
        "model_transcript",
        "password",
        "private_key",
        "private_reasoning",
        "prompt",
        "raw_prompt",
        "refresh_token",
        "secret",
        "shell_command",
        "source_body",
        "transcript",
    }
)


class PolicyEvaluationError(AutonomyContractError):
    """Raised when offline evaluation inputs are malformed or unsafe."""


class EvaluationPartition(str, Enum):
    """Closed corpus partitions.  Training rows cannot score promotion."""

    TRAINING = "training"
    HELD_OUT = "held_out"


class PairingKind(str, Enum):
    """Temperature of one frozen paired comparison."""

    COLD = "cold"
    WARM = "warm"


class EvaluationDisposition(str, Enum):
    """Closed evaluation outcome.  Insufficiency is an ordinary result."""

    EVALUATED = "evaluated"
    INSUFFICIENT_COUNTERFACTUAL_EVIDENCE = "insufficient_counterfactual_evidence"
    HOLDOUT_TRAINING_OVERLAP = "holdout_training_overlap"
    SAFETY_FLOOR_FAILED = "safety_floor_failed"
    QUALITY_FLOOR_FAILED = "quality_floor_failed"


def _enum(value: Any, enum_type: type[Enum], name: str) -> Any:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value))
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise PolicyEvaluationError(f"{name} must be one of: {allowed}") from exc


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise PolicyEvaluationError(f"{name} must be a boolean")
    return value


def _int(
    value: Any,
    name: str,
    *,
    minimum: int = 0,
    maximum: int = MAX_INTEGER,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum or value > maximum:
        raise PolicyEvaluationError(
            f"{name} must be an integer between {minimum} and {maximum}"
        )
    return value


def _signed_int(
    value: Any,
    name: str,
    *,
    maximum: int = MAX_INTEGER,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or abs(value) > maximum:
        raise PolicyEvaluationError(f"{name} must be a bounded integer")
    return value


def _identifier(value: Any, name: str, *, required: bool = True) -> str:
    if value is None:
        result = ""
    elif isinstance(value, str):
        result = value.strip()
    else:
        raise PolicyEvaluationError(f"{name} must be a string")
    if required and not result:
        raise PolicyEvaluationError(f"{name} is required")
    if result and (
        len(result.encode("utf-8")) > MAX_IDENTIFIER_BYTES
        or any(char.isspace() for char in result)
        or "\x00" in result
    ):
        raise PolicyEvaluationError(f"{name} must be a compact bounded identifier")
    return result


def _identifiers(values: Any, name: str, *, required: bool = False) -> tuple[str, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, str):
        raw = (values,)
    elif isinstance(values, Sequence) and not isinstance(values, (bytes, bytearray)):
        raw = values
    else:
        raise PolicyEvaluationError(f"{name} must be a sequence of identifiers")
    if len(raw) > MAX_SEQUENCE_ITEMS:
        raise PolicyEvaluationError(f"{name} contains too many items")
    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw:
        identifier = _identifier(item, name)
        if identifier not in seen:
            seen.add(identifier)
            normalized.append(identifier)
    if required and not normalized:
        raise PolicyEvaluationError(f"{name} must not be empty")
    return tuple(normalized)


def _normalize_field_name(key: Any) -> str:
    if not isinstance(key, str):
        raise PolicyEvaluationError("evaluation field names must be strings")
    return key.strip().lower().replace("-", "_")


def field_is_forbidden(key: Any) -> bool:
    """Return whether a mapping key is secret, executable, or self-authorizing."""

    normalized = _normalize_field_name(key)
    if not normalized:
        return False
    return any(
        normalized == marker or normalized.endswith("_" + marker)
        for marker in _FORBIDDEN_FIELD_MARKERS
    )


def _reject_forbidden_payload(value: Any, name: str, *, depth: int = 0) -> None:
    if depth > MAX_NESTING_DEPTH:
        raise PolicyEvaluationError(f"{name} exceeds maximum nesting")
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float):
        raise PolicyEvaluationError(f"{name} cannot contain floats")
    if isinstance(value, (Enum, Fraction)):
        return
    if isinstance(value, str):
        return
    if isinstance(value, Mapping):
        if len(value) > MAX_MAPPING_ITEMS:
            raise PolicyEvaluationError(f"{name} contains too many entries")
        for raw_key, raw_value in value.items():
            if field_is_forbidden(raw_key):
                raise PolicyEvaluationError(
                    f"{name} contains forbidden private or executable data"
                )
            _reject_forbidden_payload(raw_value, f"{name}.{raw_key}", depth=depth + 1)
        return
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        if len(value) > MAX_SEQUENCE_ITEMS:
            raise PolicyEvaluationError(f"{name} contains too many items")
        for index, item in enumerate(value):
            _reject_forbidden_payload(item, f"{name}[{index}]", depth=depth + 1)
        return
    raise PolicyEvaluationError(f"{name} contains unsupported value type {type(value).__name__}")


def _unchanged_policy_axes() -> Mapping[str, bool]:
    return MappingProxyType({axis: False for axis in PROTECTED_POLICY_AXES})


def _sequence(value: Any, name: str) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise PolicyEvaluationError(f"{name} must be a sequence")
    if not isinstance(value, Sequence):
        raise PolicyEvaluationError(f"{name} must be a sequence")
    if len(value) > MAX_EVALUATION_ITEMS:
        raise PolicyEvaluationError(f"{name} contains too many items")
    return tuple(value)


def _candidate_from(value: RoutePolicyCandidate | Mapping[str, Any]) -> RoutePolicyCandidate:
    if isinstance(value, RoutePolicyCandidate):
        candidate = value
    elif isinstance(value, Mapping):
        _reject_forbidden_payload(value, "candidate")
        candidate = RoutePolicyCandidate.from_dict(value)
    else:
        raise PolicyEvaluationError("candidate must be a RoutePolicyCandidate")
    if not candidate.shadow_only:
        raise PolicyEvaluationError(
            "route-policy candidates are shadow-only until external promotion"
        )
    if candidate.external_authorization_id:
        raise PolicyEvaluationError("evaluation cannot authorize or mutate production policy")
    return candidate


def _observation_from(
    value: PolicyObservation | Mapping[str, Any] | None,
) -> PolicyObservation | None:
    if value is None:
        return None
    if isinstance(value, PolicyObservation):
        return value
    if isinstance(value, Mapping):
        _reject_forbidden_payload(value, "observation")
        return PolicyObservation.from_dict(value)
    raise PolicyEvaluationError("observation must be a PolicyObservation")


def _mean_bp(values: Sequence[int]) -> int:
    if not values:
        return 0
    total = sum((Fraction(item) for item in values), Fraction(0))
    return int(round(total / len(values)))


def _propensity_missing(value: int | None) -> bool:
    return value is None or value <= 0


@dataclass(frozen=True)
class LoggedDecision:
    """One logged action used for held-out or counterfactual evaluation."""

    decision_id: str
    episode_id: str
    selected_action: MetaAction
    policy_id: str
    frozen_input_ids: tuple[str, ...]
    feature_ids: tuple[str, ...]
    terminal_status: TerminalStatus
    partition: EvaluationPartition = EvaluationPartition.HELD_OUT
    pairing_kind: PairingKind = PairingKind.COLD
    propensity_bp: int | None = None
    accepted_criterion_ids: tuple[str, ...] = ()
    evidence_gain_bp: int = 0
    cost_micros: int = 0
    latency_ms: int = 0
    safety_violation: bool = False
    observation: PolicyObservation | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision_id", _identifier(self.decision_id, "decision_id"))
        object.__setattr__(self, "episode_id", _identifier(self.episode_id, "episode_id"))
        object.__setattr__(
            self, "selected_action", _enum(self.selected_action, MetaAction, "selected_action")
        )
        object.__setattr__(self, "policy_id", _identifier(self.policy_id, "policy_id"))
        object.__setattr__(
            self,
            "frozen_input_ids",
            _identifiers(self.frozen_input_ids, "frozen_input_ids", required=True),
        )
        object.__setattr__(
            self, "feature_ids", _identifiers(self.feature_ids, "feature_ids", required=True)
        )
        object.__setattr__(
            self, "terminal_status", _enum(self.terminal_status, TerminalStatus, "terminal_status")
        )
        object.__setattr__(
            self, "partition", _enum(self.partition, EvaluationPartition, "partition")
        )
        object.__setattr__(
            self, "pairing_kind", _enum(self.pairing_kind, PairingKind, "pairing_kind")
        )
        if self.propensity_bp is None:
            object.__setattr__(self, "propensity_bp", None)
        else:
            object.__setattr__(
                self,
                "propensity_bp",
                _int(self.propensity_bp, "propensity_bp", maximum=BASIS_POINTS),
            )
        object.__setattr__(
            self,
            "accepted_criterion_ids",
            _identifiers(self.accepted_criterion_ids, "accepted_criterion_ids"),
        )
        object.__setattr__(
            self, "evidence_gain_bp", _int(self.evidence_gain_bp, "evidence_gain_bp", maximum=BASIS_POINTS)
        )
        object.__setattr__(self, "cost_micros", _int(self.cost_micros, "cost_micros"))
        object.__setattr__(self, "latency_ms", _int(self.latency_ms, "latency_ms"))
        object.__setattr__(
            self, "safety_violation", _bool(self.safety_violation, "safety_violation")
        )
        observation = _observation_from(self.observation)
        object.__setattr__(self, "observation", observation)
        if observation is not None:
            if observation.observation_id != self.decision_id:
                raise PolicyEvaluationError("logged decision_id must match observation_id")
            if observation.episode_id != self.episode_id:
                raise PolicyEvaluationError("logged episode_id must match the observation")
            if observation.selected_action is not self.selected_action:
                raise PolicyEvaluationError("logged action must match the observation")
            if observation.route_policy_id != self.policy_id:
                raise PolicyEvaluationError("logged policy_id must match the observation")
            if observation.action_propensity_bp != self.propensity_bp:
                raise PolicyEvaluationError(
                    "logged propensity must match the observation when one is bound"
                )

    @classmethod
    def from_observation(
        cls,
        observation: PolicyObservation | Mapping[str, Any],
        *,
        frozen_input_ids: Sequence[str],
        partition: EvaluationPartition | str = EvaluationPartition.HELD_OUT,
        pairing_kind: PairingKind | str = PairingKind.COLD,
        propensity_bp: int | None = None,
    ) -> LoggedDecision:
        bound = _observation_from(observation)
        if bound is None:
            raise PolicyEvaluationError("observation is required")
        return cls(
            decision_id=bound.observation_id,
            episode_id=bound.episode_id,
            selected_action=bound.selected_action,
            policy_id=bound.route_policy_id,
            frozen_input_ids=frozen_input_ids,
            feature_ids=bound.feature_ids,
            terminal_status=bound.terminal_status,
            partition=partition,  # type: ignore[arg-type]
            pairing_kind=pairing_kind,  # type: ignore[arg-type]
            propensity_bp=bound.action_propensity_bp if propensity_bp is None else propensity_bp,
            accepted_criterion_ids=bound.accepted_criterion_ids,
            evidence_gain_bp=bound.evidence_gain_bp,
            cost_micros=bound.cost_micros,
            latency_ms=bound.latency_ms,
            safety_violation=bound.safety_violation,
            observation=bound,
        )

    @property
    def content_id(self) -> str:
        payload = dict(self.to_dict())
        payload.pop("content_id", None)
        return content_identity(payload)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": LOGGED_DECISION_SCHEMA,
            "contract_version": CONTRACT_VERSION,
            "decision_id": self.decision_id,
            "episode_id": self.episode_id,
            "selected_action": self.selected_action.value,
            "policy_id": self.policy_id,
            "frozen_input_ids": list(self.frozen_input_ids),
            "feature_ids": list(self.feature_ids),
            "terminal_status": self.terminal_status.value,
            "partition": self.partition.value,
            "pairing_kind": self.pairing_kind.value,
            "propensity_bp": self.propensity_bp,
            "accepted_criterion_ids": list(self.accepted_criterion_ids),
            "evidence_gain_bp": self.evidence_gain_bp,
            "cost_micros": self.cost_micros,
            "latency_ms": self.latency_ms,
            "safety_violation": self.safety_violation,
            "observation": None if self.observation is None else self.observation.to_dict(),
        }
        payload["content_id"] = content_identity(
            {key: value for key, value in payload.items() if key != "content_id"}
        )
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | LoggedDecision) -> LoggedDecision:
        if isinstance(payload, LoggedDecision):
            return payload
        if not isinstance(payload, Mapping):
            raise PolicyEvaluationError("logged decision must be an object")
        _reject_forbidden_payload(payload, "logged_decision")
        extra = set(payload).difference(
            {
                "schema",
                "contract_version",
                "content_id",
                "decision_id",
                "episode_id",
                "selected_action",
                "policy_id",
                "frozen_input_ids",
                "feature_ids",
                "terminal_status",
                "partition",
                "pairing_kind",
                "propensity_bp",
                "accepted_criterion_ids",
                "evidence_gain_bp",
                "cost_micros",
                "latency_ms",
                "safety_violation",
                "observation",
            }
        )
        if extra:
            raise PolicyEvaluationError("logged decision contains unsupported fields")
        item = cls(
            decision_id=payload.get("decision_id") or "",
            episode_id=payload.get("episode_id") or "",
            selected_action=payload.get("selected_action"),  # type: ignore[arg-type]
            policy_id=payload.get("policy_id") or "",
            frozen_input_ids=payload.get("frozen_input_ids") or (),
            feature_ids=payload.get("feature_ids") or (),
            terminal_status=payload.get("terminal_status"),  # type: ignore[arg-type]
            partition=payload.get("partition", EvaluationPartition.HELD_OUT),  # type: ignore[arg-type]
            pairing_kind=payload.get("pairing_kind", PairingKind.COLD),  # type: ignore[arg-type]
            propensity_bp=payload.get("propensity_bp"),
            accepted_criterion_ids=payload.get("accepted_criterion_ids") or (),
            evidence_gain_bp=payload.get("evidence_gain_bp", 0),
            cost_micros=payload.get("cost_micros", 0),
            latency_ms=payload.get("latency_ms", 0),
            safety_violation=payload.get("safety_violation", False),
            observation=payload.get("observation"),
        )
        claimed = payload.get("content_id")
        if claimed not in (None, "", item.content_id):
            raise PolicyEvaluationError("logged decision content identity does not match payload")
        return item


def _logged_decision_from(value: LoggedDecision | PolicyObservation | Mapping[str, Any]) -> LoggedDecision:
    if isinstance(value, LoggedDecision):
        return value
    if isinstance(value, PolicyObservation):
        raise PolicyEvaluationError(
            "PolicyObservation requires frozen_input_ids; wrap it with LoggedDecision.from_observation"
        )
    if isinstance(value, Mapping):
        if "frozen_input_ids" in value or "decision_id" in value:
            return LoggedDecision.from_dict(value)
        observation = value.get("observation")
        if observation is not None:
            extras = {
                key: item
                for key, item in value.items()
                if key
                in {
                    "frozen_input_ids",
                    "partition",
                    "pairing_kind",
                    "propensity_bp",
                }
            }
            return LoggedDecision.from_observation(observation, **extras)
    raise PolicyEvaluationError("logged decision must be a LoggedDecision")


def _reward_bp(decision: LoggedDecision) -> int:
    if decision.safety_violation:
        return -BASIS_POINTS
    if decision.terminal_status is TerminalStatus.PENDING:
        raise PolicyEvaluationError("pending decisions cannot score an offline evaluation")
    if decision.terminal_status is TerminalStatus.SUCCEEDED:
        return max(1, decision.evidence_gain_bp)
    return -max(1, BASIS_POINTS - decision.evidence_gain_bp)


@dataclass(frozen=True)
class ComparisonEvidence:
    """Paired baseline versus candidate outcomes on identical frozen inputs."""

    pair_id: str
    baseline: LoggedDecision
    candidate: LoggedDecision

    def __post_init__(self) -> None:
        object.__setattr__(self, "pair_id", _identifier(self.pair_id, "pair_id"))
        object.__setattr__(self, "baseline", _logged_decision_from(self.baseline))
        object.__setattr__(self, "candidate", _logged_decision_from(self.candidate))
        if self.baseline.frozen_input_ids != self.candidate.frozen_input_ids:
            raise PolicyEvaluationError(
                "comparison evidence must bind identical frozen input identities"
            )
        if self.baseline.pairing_kind is not self.candidate.pairing_kind:
            raise PolicyEvaluationError(
                "comparison evidence must bind the same cold/warm pairing kind"
            )
        if self.baseline.partition is EvaluationPartition.TRAINING:
            raise PolicyEvaluationError("comparison evidence cannot score training partitions")
        if self.candidate.partition is EvaluationPartition.TRAINING:
            raise PolicyEvaluationError("comparison evidence cannot score training partitions")
        if self.baseline.policy_id == self.candidate.policy_id:
            raise PolicyEvaluationError(
                "comparison evidence requires distinct baseline and candidate policies"
            )

    @property
    def frozen_input_ids(self) -> tuple[str, ...]:
        return self.baseline.frozen_input_ids

    @property
    def pairing_kind(self) -> PairingKind:
        return self.baseline.pairing_kind

    @property
    def content_id(self) -> str:
        payload = dict(self.to_dict())
        payload.pop("content_id", None)
        return content_identity(payload)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": COMPARISON_EVIDENCE_SCHEMA,
            "contract_version": CONTRACT_VERSION,
            "pair_id": self.pair_id,
            "baseline": self.baseline.to_dict(),
            "candidate": self.candidate.to_dict(),
            "frozen_input_ids": list(self.frozen_input_ids),
            "pairing_kind": self.pairing_kind.value,
        }
        payload["content_id"] = content_identity(
            {key: value for key, value in payload.items() if key != "content_id"}
        )
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | ComparisonEvidence) -> ComparisonEvidence:
        if isinstance(payload, ComparisonEvidence):
            return payload
        if not isinstance(payload, Mapping):
            raise PolicyEvaluationError("comparison evidence must be an object")
        _reject_forbidden_payload(payload, "comparison")
        extra = set(payload).difference(
            {
                "schema",
                "contract_version",
                "content_id",
                "pair_id",
                "baseline",
                "candidate",
                "frozen_input_ids",
                "pairing_kind",
            }
        )
        if extra:
            raise PolicyEvaluationError("comparison evidence contains unsupported fields")
        item = cls(
            pair_id=payload.get("pair_id") or "",
            baseline=payload.get("baseline"),  # type: ignore[arg-type]
            candidate=payload.get("candidate"),  # type: ignore[arg-type]
        )
        claimed = payload.get("content_id")
        if claimed not in (None, "", item.content_id):
            raise PolicyEvaluationError("comparison content identity does not match payload")
        return item


def _comparison_from(value: ComparisonEvidence | Mapping[str, Any]) -> ComparisonEvidence:
    if isinstance(value, ComparisonEvidence):
        return value
    if isinstance(value, Mapping):
        return ComparisonEvidence.from_dict(value)
    raise PolicyEvaluationError("comparison must be ComparisonEvidence")


def _valid_comparisons(values: Sequence[Any]) -> tuple[ComparisonEvidence, ...]:
    pairs: list[ComparisonEvidence] = []
    seen: set[str] = set()
    for item in _sequence(values, "comparisons"):
        pair = _comparison_from(item)
        if pair.pair_id in seen:
            raise PolicyEvaluationError("comparisons contain duplicate pair_id values")
        seen.add(pair.pair_id)
        pairs.append(pair)
    return tuple(pairs)


def _logged_decisions(values: Sequence[Any]) -> tuple[LoggedDecision, ...]:
    decisions: list[LoggedDecision] = []
    seen: set[str] = set()
    for item in _sequence(values, "observations"):
        decision = _logged_decision_from(item)
        if decision.decision_id in seen:
            raise PolicyEvaluationError("observations contain duplicate decision_id values")
        seen.add(decision.decision_id)
        decisions.append(decision)
    return tuple(decisions)


def _cold_warm_paired(pairs: Sequence[ComparisonEvidence]) -> bool:
    by_inputs: dict[tuple[str, ...], set[PairingKind]] = {}
    for pair in pairs:
        by_inputs.setdefault(pair.frozen_input_ids, set()).add(pair.pairing_kind)
    if not by_inputs:
        return False
    required = {PairingKind.COLD, PairingKind.WARM}
    return all(kinds == required for kinds in by_inputs.values())


@dataclass(frozen=True)
class RoutePolicyEvaluationResult:
    """Content-addressed offline evaluation receipt.  It is never a promotion."""

    disposition: EvaluationDisposition
    reason_codes: tuple[str, ...]
    candidate_id: str
    policy_version: str
    parent_policy_id: str
    evaluation_id: str
    held_out_decision_ids: tuple[str, ...]
    comparison_pair_ids: tuple[str, ...]
    blocker_codes: tuple[str, ...] = ()
    promotion_eligible: bool = False
    improvement_claimed: bool = False
    shadow_only: bool = True
    live_routing_effect: bool = False
    production_exploration: bool = False
    production_policy_mutated: bool = False
    affects_production_acceptance: bool = False
    holdout_separated: bool = False
    propensity_supported: bool = False
    comparison_supported: bool = False
    version_bound: bool = False
    cold_warm_paired: bool = False
    safety_floor_passed: bool = False
    quality_floor_passed: bool = False
    baseline_value_bp: int = 0
    candidate_value_bp: int = 0
    paired_delta_bp: int = 0
    ips_value_bp: int = 0
    accepted_criterion_delta: int = 0
    cost_delta_micros: int = 0
    latency_delta_ms: int = 0
    safety_violation_count: int = 0
    policy_axis_changes: Mapping[str, bool] = field(default_factory=_unchanged_policy_axes)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, EvaluationDisposition, "disposition"),
        )
        object.__setattr__(
            self,
            "reason_codes",
            _identifiers(self.reason_codes, "reason_codes", required=True),
        )
        for name in ("candidate_id", "policy_version", "parent_policy_id", "evaluation_id"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(
            self,
            "held_out_decision_ids",
            _identifiers(self.held_out_decision_ids, "held_out_decision_ids"),
        )
        object.__setattr__(
            self,
            "comparison_pair_ids",
            _identifiers(self.comparison_pair_ids, "comparison_pair_ids"),
        )
        object.__setattr__(
            self, "blocker_codes", _identifiers(self.blocker_codes, "blocker_codes")
        )
        for name in (
            "promotion_eligible",
            "improvement_claimed",
            "shadow_only",
            "live_routing_effect",
            "production_exploration",
            "production_policy_mutated",
            "affects_production_acceptance",
            "holdout_separated",
            "propensity_supported",
            "comparison_supported",
            "version_bound",
            "cold_warm_paired",
            "safety_floor_passed",
            "quality_floor_passed",
        ):
            object.__setattr__(self, name, _bool(getattr(self, name), name))
        object.__setattr__(
            self, "baseline_value_bp", _signed_int(self.baseline_value_bp, "baseline_value_bp")
        )
        object.__setattr__(
            self, "candidate_value_bp", _signed_int(self.candidate_value_bp, "candidate_value_bp")
        )
        object.__setattr__(
            self, "paired_delta_bp", _signed_int(self.paired_delta_bp, "paired_delta_bp")
        )
        object.__setattr__(self, "ips_value_bp", _signed_int(self.ips_value_bp, "ips_value_bp"))
        object.__setattr__(
            self,
            "accepted_criterion_delta",
            _signed_int(self.accepted_criterion_delta, "accepted_criterion_delta"),
        )
        object.__setattr__(
            self, "cost_delta_micros", _signed_int(self.cost_delta_micros, "cost_delta_micros")
        )
        object.__setattr__(
            self, "latency_delta_ms", _signed_int(self.latency_delta_ms, "latency_delta_ms")
        )
        object.__setattr__(
            self, "safety_violation_count", _int(self.safety_violation_count, "safety_violation_count")
        )
        if not self.shadow_only:
            raise PolicyEvaluationError(
                "route-policy evaluation remains shadow-only until external promotion"
            )
        if (
            self.live_routing_effect
            or self.production_exploration
            or self.production_policy_mutated
            or self.affects_production_acceptance
        ):
            raise PolicyEvaluationError("evaluation cannot mutate production policy")
        axes = _unchanged_policy_axes()
        supplied = self.policy_axis_changes or axes
        if not isinstance(supplied, Mapping):
            raise PolicyEvaluationError("policy_axis_changes must be a mapping")
        normalized = {axis: False for axis in PROTECTED_POLICY_AXES}
        for key, value in supplied.items():
            axis = _identifier(key, "policy_axis_changes")
            if axis not in normalized:
                raise PolicyEvaluationError("policy_axis_changes names an unknown policy axis")
            if _bool(value, "policy_axis_changes"):
                raise PolicyEvaluationError(
                    "evaluation cannot change provider, authority, privacy, "
                    "validation, proof, or confirmation policy"
                )
            normalized[axis] = False
        object.__setattr__(self, "policy_axis_changes", MappingProxyType(normalized))
        if self.disposition is EvaluationDisposition.INSUFFICIENT_COUNTERFACTUAL_EVIDENCE:
            if self.reason_codes != (INSUFFICIENT_COUNTERFACTUAL_EVIDENCE,):
                raise PolicyEvaluationError(
                    "missing comparison or propensity evidence must return exactly "
                    f"{INSUFFICIENT_COUNTERFACTUAL_EVIDENCE}"
                )
            if self.blocker_codes != (INSUFFICIENT_COUNTERFACTUAL_EVIDENCE,):
                raise PolicyEvaluationError(
                    "missing comparison or propensity evidence must block with exactly "
                    f"{INSUFFICIENT_COUNTERFACTUAL_EVIDENCE}"
                )
            if self.improvement_claimed or self.promotion_eligible:
                raise PolicyEvaluationError("insufficient evidence cannot fabricate an improvement claim")
            if self.paired_delta_bp != 0 or self.ips_value_bp != 0:
                raise PolicyEvaluationError("insufficient evidence cannot report a counterfactual delta")
        if self.improvement_claimed and not (
            self.comparison_supported and self.propensity_supported and self.safety_floor_passed
        ):
            raise PolicyEvaluationError("improvement claims require comparison, propensity, and safety evidence")
        if self.promotion_eligible:
            if self.disposition is not EvaluationDisposition.EVALUATED:
                raise PolicyEvaluationError("only a complete evaluation can be promotion-eligible")
            if self.blocker_codes or not self.improvement_claimed:
                raise PolicyEvaluationError("promotion-eligible results cannot carry blockers")
            if not (
                self.holdout_separated
                and self.version_bound
                and self.quality_floor_passed
                and self.safety_floor_passed
            ):
                raise PolicyEvaluationError("promotion eligibility requires every evaluation floor")
        encoded = canonical_json_bytes(self.to_dict())
        if len(encoded) > MAX_CANONICAL_RECORD_BYTES:
            raise PolicyEvaluationError("evaluation result exceeds its bounded size")

    @property
    def result_id(self) -> str:
        payload = dict(self.to_dict())
        payload.pop("result_id", None)
        return content_identity(payload)

    @property
    def evaluation_code(self) -> str:
        return self.disposition.value

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": ROUTE_POLICY_EVALUATION_RESULT_SCHEMA,
            "contract_version": CONTRACT_VERSION,
            "interface": ROUTE_POLICY_EVALUATION_INTERFACE,
            "disposition": self.disposition.value,
            "reason_codes": list(self.reason_codes),
            "candidate_id": self.candidate_id,
            "policy_version": self.policy_version,
            "parent_policy_id": self.parent_policy_id,
            "evaluation_id": self.evaluation_id,
            "held_out_decision_ids": list(self.held_out_decision_ids),
            "comparison_pair_ids": list(self.comparison_pair_ids),
            "blocker_codes": list(self.blocker_codes),
            "promotion_eligible": self.promotion_eligible,
            "improvement_claimed": self.improvement_claimed,
            "shadow_only": True,
            "live_routing_effect": False,
            "production_exploration": False,
            "production_policy_mutated": False,
            "affects_production_acceptance": False,
            "holdout_separated": self.holdout_separated,
            "propensity_supported": self.propensity_supported,
            "comparison_supported": self.comparison_supported,
            "version_bound": self.version_bound,
            "cold_warm_paired": self.cold_warm_paired,
            "safety_floor_passed": self.safety_floor_passed,
            "quality_floor_passed": self.quality_floor_passed,
            "baseline_value_bp": self.baseline_value_bp,
            "candidate_value_bp": self.candidate_value_bp,
            "paired_delta_bp": self.paired_delta_bp,
            "ips_value_bp": self.ips_value_bp,
            "accepted_criterion_delta": self.accepted_criterion_delta,
            "cost_delta_micros": self.cost_delta_micros,
            "latency_delta_ms": self.latency_delta_ms,
            "safety_violation_count": self.safety_violation_count,
            "policy_axis_changes": dict(self.policy_axis_changes),
        }
        payload["result_id"] = content_identity(
            {key: value for key, value in payload.items() if key != "result_id"}
        )
        return payload


def _ips_value_bp(pairs: Sequence[ComparisonEvidence]) -> int:
    if not pairs:
        return 0
    total = Fraction(0)
    for pair in pairs:
        propensity = pair.baseline.propensity_bp
        if _propensity_missing(propensity):
            return 0
        indicator = 1 if pair.candidate.selected_action is pair.baseline.selected_action else 0
        weight = Fraction(BASIS_POINTS, propensity)
        total += indicator * weight * _reward_bp(pair.baseline)
    return int(round(total / len(pairs)))


class RoutePolicyEvaluation:
    """Offline held-out and counterfactual evaluator.  Promotion stays external."""

    INTERFACE = ROUTE_POLICY_EVALUATION_INTERFACE
    SCHEMA = ROUTE_POLICY_EVALUATION_SCHEMA

    def __init__(self) -> None:
        self._shadow_only = True

    @property
    def shadow_only(self) -> bool:
        return True

    @property
    def live_routing_effect(self) -> bool:
        return False

    @property
    def production_exploration(self) -> bool:
        return False

    @property
    def production_policy_mutated(self) -> bool:
        return False

    @property
    def affects_production_acceptance(self) -> bool:
        return False

    @property
    def policy_axis_changes(self) -> Mapping[str, bool]:
        return _unchanged_policy_axes()

    def evaluate(
        self,
        candidate: RoutePolicyCandidate | Mapping[str, Any],
        *,
        comparisons: Sequence[ComparisonEvidence | Mapping[str, Any]] = (),
        observations: Sequence[LoggedDecision | Mapping[str, Any]] = (),
        evaluation_id: str = "",
        production: bool = False,
        explore_production: bool = False,
        live: bool = False,
        mutate_production_policy: bool = False,
    ) -> RoutePolicyEvaluationResult:
        """Evaluate a shadow candidate.  Missing evidence is not an improvement."""

        if production or explore_production or live or mutate_production_policy:
            raise PolicyEvaluationError("evaluation cannot mutate production policy")
        bound = _candidate_from(candidate)
        pairs = _valid_comparisons(comparisons)
        extra = _logged_decisions(observations)
        evaluation_key = _identifier(
            evaluation_id or (bound.held_out_evaluation_ids[0] if bound.held_out_evaluation_ids else ""),
            "evaluation_id",
        )
        version_bound = evaluation_key in bound.held_out_evaluation_ids
        if pairs:
            version_bound = version_bound and all(
                pair.candidate.policy_id == bound.candidate_id
                and pair.baseline.policy_id == bound.parent_policy_id
                for pair in pairs
            )
        held_out = tuple(
            item
            for item in extra
            if item.partition is EvaluationPartition.HELD_OUT
        )
        held_out_ids = tuple(item.decision_id for item in held_out)
        pair_ids = tuple(item.pair_id for item in pairs)
        pair_decision_ids = tuple(
            decision.decision_id
            for pair in pairs
            for decision in (pair.baseline, pair.candidate)
        )
        training_ids = set(bound.training_observation_ids)
        overlap_ids = training_ids.intersection(held_out_ids).union(
            training_ids.intersection(pair_decision_ids)
        )
        holdout_separated = not overlap_ids
        comparison_supported = bool(pairs) and version_bound
        propensity_supported = bool(pairs) and all(
            not _propensity_missing(pair.baseline.propensity_bp)
            and not _propensity_missing(pair.candidate.propensity_bp)
            for pair in pairs
        )
        common = {
            "candidate_id": bound.candidate_id,
            "policy_version": bound.policy_version,
            "parent_policy_id": bound.parent_policy_id,
            "evaluation_id": evaluation_key,
            "held_out_decision_ids": held_out_ids or pair_decision_ids,
            "comparison_pair_ids": pair_ids,
            "holdout_separated": holdout_separated,
            "propensity_supported": propensity_supported,
            "comparison_supported": comparison_supported,
            "version_bound": version_bound,
        }
        if not comparison_supported or not propensity_supported:
            return RoutePolicyEvaluationResult(
                disposition=EvaluationDisposition.INSUFFICIENT_COUNTERFACTUAL_EVIDENCE,
                reason_codes=(INSUFFICIENT_COUNTERFACTUAL_EVIDENCE,),
                blocker_codes=(INSUFFICIENT_COUNTERFACTUAL_EVIDENCE,),
                **common,
            )
        if not holdout_separated:
            return RoutePolicyEvaluationResult(
                disposition=EvaluationDisposition.HOLDOUT_TRAINING_OVERLAP,
                reason_codes=("holdout_training_overlap",),
                blocker_codes=("holdout_training_overlap",),
                **common,
            )
        baseline_rewards = tuple(_reward_bp(pair.baseline) for pair in pairs)
        candidate_rewards = tuple(_reward_bp(pair.candidate) for pair in pairs)
        baseline_value = _mean_bp(baseline_rewards)
        candidate_value = _mean_bp(candidate_rewards)
        paired_delta = candidate_value - baseline_value
        ips_value = _ips_value_bp(pairs)
        safety_count = sum(1 for pair in pairs if pair.candidate.safety_violation)
        safety_floor = safety_count == 0
        baseline_accepted = sum(len(pair.baseline.accepted_criterion_ids) for pair in pairs)
        candidate_accepted = sum(len(pair.candidate.accepted_criterion_ids) for pair in pairs)
        baseline_success = sum(
            1 for pair in pairs if pair.baseline.terminal_status is TerminalStatus.SUCCEEDED
        )
        candidate_success = sum(
            1 for pair in pairs if pair.candidate.terminal_status is TerminalStatus.SUCCEEDED
        )
        quality_floor = candidate_accepted >= baseline_accepted and candidate_success >= baseline_success
        cold_warm = _cold_warm_paired(pairs)
        cost_delta = sum(pair.candidate.cost_micros - pair.baseline.cost_micros for pair in pairs)
        latency_delta = sum(pair.candidate.latency_ms - pair.baseline.latency_ms for pair in pairs)
        accepted_delta = candidate_accepted - baseline_accepted
        cost_improved = cold_warm and cost_delta < 0 and latency_delta <= 0
        quality_improved = accepted_delta > 0 or candidate_success > baseline_success or paired_delta > 0
        improvement = bool(
            safety_floor
            and quality_floor
            and (quality_improved or cost_improved)
            and (paired_delta > 0 or cost_improved)
        )
        if not safety_floor:
            return RoutePolicyEvaluationResult(
                disposition=EvaluationDisposition.SAFETY_FLOOR_FAILED,
                reason_codes=("safety_floor_failed",),
                blocker_codes=("safety_floor_failed",),
                cold_warm_paired=cold_warm,
                safety_floor_passed=False,
                quality_floor_passed=quality_floor,
                baseline_value_bp=baseline_value,
                candidate_value_bp=candidate_value,
                paired_delta_bp=paired_delta,
                ips_value_bp=ips_value,
                accepted_criterion_delta=accepted_delta,
                cost_delta_micros=cost_delta if cold_warm else 0,
                latency_delta_ms=latency_delta if cold_warm else 0,
                safety_violation_count=safety_count,
                **common,
            )
        if not quality_floor:
            return RoutePolicyEvaluationResult(
                disposition=EvaluationDisposition.QUALITY_FLOOR_FAILED,
                reason_codes=("quality_floor_failed",),
                blocker_codes=("quality_floor_failed",),
                cold_warm_paired=cold_warm,
                safety_floor_passed=True,
                quality_floor_passed=False,
                baseline_value_bp=baseline_value,
                candidate_value_bp=candidate_value,
                paired_delta_bp=paired_delta,
                ips_value_bp=ips_value,
                accepted_criterion_delta=accepted_delta,
                cost_delta_micros=cost_delta if cold_warm else 0,
                latency_delta_ms=latency_delta if cold_warm else 0,
                safety_violation_count=0,
                **common,
            )
        reasons = [
            "held_out_separated",
            "propensity_supported",
            "comparison_supported",
            "version_bound",
            "safety_floor_passed",
            "quality_floor_passed",
        ]
        if cold_warm:
            reasons.append("cold_warm_paired")
        reasons.append("improvement_supported" if improvement else "no_improvement_claim")
        return RoutePolicyEvaluationResult(
            disposition=EvaluationDisposition.EVALUATED,
            reason_codes=tuple(reasons),
            blocker_codes=() if improvement else ("no_improvement_claim",),
            promotion_eligible=improvement,
            improvement_claimed=improvement,
            cold_warm_paired=cold_warm,
            safety_floor_passed=True,
            quality_floor_passed=True,
            baseline_value_bp=baseline_value,
            candidate_value_bp=candidate_value,
            paired_delta_bp=paired_delta,
            ips_value_bp=ips_value,
            accepted_criterion_delta=accepted_delta,
            cost_delta_micros=cost_delta if cold_warm else 0,
            latency_delta_ms=latency_delta if cold_warm else 0,
            safety_violation_count=0,
            **common,
        )

    def promote(self, authorization_id: str = "") -> None:
        del authorization_id
        raise PolicyEvaluationError("evaluation cannot authorize or mutate production policy")

    def apply_production_policy(self) -> None:
        raise PolicyEvaluationError("evaluation cannot mutate production policy")

    def apply_live_route(self) -> None:
        raise PolicyEvaluationError("evaluation cannot mutate production policy")


def route_policy_evaluation() -> RoutePolicyEvaluation:
    """Construct the offline RoutePolicyEvaluation@1 evaluator."""

    return RoutePolicyEvaluation()


def evaluate_route_policy(
    candidate: RoutePolicyCandidate | Mapping[str, Any],
    **kwargs: Any,
) -> RoutePolicyEvaluationResult:
    """Evaluate a shadow route-policy candidate from logged evidence."""

    return RoutePolicyEvaluation().evaluate(candidate, **kwargs)


__all__ = [
    "BASIS_POINTS",
    "COMPARISON_EVIDENCE_SCHEMA",
    "INSUFFICIENT_COUNTERFACTUAL_EVIDENCE",
    "LOGGED_DECISION_SCHEMA",
    "PROTECTED_POLICY_AXES",
    "ROUTE_POLICY_EVALUATION_INTERFACE",
    "ROUTE_POLICY_EVALUATION_RESULT_SCHEMA",
    "ROUTE_POLICY_EVALUATION_SCHEMA",
    "ComparisonEvidence",
    "EvaluationDisposition",
    "EvaluationPartition",
    "LoggedDecision",
    "PairingKind",
    "PolicyEvaluationError",
    "RoutePolicyEvaluation",
    "RoutePolicyEvaluationResult",
    "evaluate_route_policy",
    "field_is_forbidden",
    "route_policy_evaluation",
]
