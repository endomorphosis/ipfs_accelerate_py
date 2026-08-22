# ruff: noqa: UP042 - the package retains Python 3.8 compatibility
"""Shadow-only constrained route learner for already admitted actions.

``ShadowRoutePolicy@1`` scores policy-admitted :class:`MetaAction` values with
deterministic integer or rational linear features.  A linear-UCB bonus may be
added during shadow evaluation.  The learner never production-explores, never
expands the closed action set, never raises authority, privacy, provider,
validation, proof, or confirmation policy, and never produces a live routing
effect.  Existing live route authority remains outside this module.

Promotion is external.  Candidates stay ``shadow_only`` and cannot carry an
authorization identity.  Exact versioned rollback restores a prior candidate
and its snapshotted shadow counts; accepted history is not rewritten.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from math import isqrt
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
    AuthorityClass,
    AutonomyContractError,
    MetaAction,
    PolicyObservation,
    PrivacyClass,
    ResolutionAction,
    ResolutionCandidate,
    RoutePolicyCandidate,
    TerminalStatus,
)

SHADOW_ROUTE_POLICY_INTERFACE: Final[str] = "ShadowRoutePolicy@1"
ROUTE_POLICY_CANDIDATE_INTERFACE: Final[str] = "RoutePolicyCandidate@1"
SHADOW_ROUTE_POLICY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/shadow-route-policy@1"
)
LINEAR_UCB_STATE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/linear-ucb-state@1"
)
SHADOW_ROUTE_SCORE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/shadow-route-score@1"
)
SHADOW_ROUTE_SELECTION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/shadow-route-selection@1"
)
ROUTE_POLICY_SNAPSHOT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/route-policy-snapshot@1"
)

BASIS_POINTS: Final[int] = 10_000
DEFAULT_RIDGE_LAMBDA: Final[int] = 1
DEFAULT_UCB_ALPHA_BP: Final[int] = BASIS_POINTS
MAX_ROUTE_FEATURES: Final[int] = 32
MAX_POLICY_VERSIONS: Final[int] = MAX_SEQUENCE_ITEMS

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

_FORBIDDEN_FEATURE_NAMES: Final[frozenset[str]] = frozenset(
    {
        "authorization",
        "external_authorization_id",
        "live_route",
        "live_routing_effect",
        "production_explore",
        "production_exploration",
        "raise_authority",
        "shadow_only",
    }
)

_REMOTE_MODEL_ACTIONS: Final[frozenset[MetaAction]] = frozenset(
    {
        MetaAction.CALL_REMOTE_STANDARD_MODEL,
        MetaAction.CALL_REMOTE_STRONG_MODEL,
    }
)


class ShadowRoutePolicyError(AutonomyContractError):
    """Raised when a shadow route-policy input or transition is unsafe."""


class SelectionMode(str, Enum):
    """Closed scoring modes.  Linear UCB is shadow evaluation only."""

    LINEAR_SCORE = "linear_score"
    LINEAR_UCB = "linear_ucb"


class SelectionDisposition(str, Enum):
    """Closed outcome of one shadow ranking."""

    SELECTED = "selected"
    ABSTAINED = "abstained"


def _enum(value: Any, enum_type: type[Enum], name: str) -> Any:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value))
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise ShadowRoutePolicyError(f"{name} must be one of: {allowed}") from exc


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ShadowRoutePolicyError(f"{name} must be a boolean")
    return value


def _int(
    value: Any,
    name: str,
    *,
    minimum: int = 0,
    maximum: int = MAX_INTEGER,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum or value > maximum:
        raise ShadowRoutePolicyError(
            f"{name} must be an integer between {minimum} and {maximum}"
        )
    return value


def _identifier(value: Any, name: str, *, required: bool = True) -> str:
    if value is None:
        result = ""
    elif isinstance(value, str):
        result = value.strip()
    else:
        raise ShadowRoutePolicyError(f"{name} must be a string")
    if required and not result:
        raise ShadowRoutePolicyError(f"{name} is required")
    if result and (
        len(result.encode("utf-8")) > MAX_IDENTIFIER_BYTES
        or any(char.isspace() for char in result)
        or "\x00" in result
    ):
        raise ShadowRoutePolicyError(f"{name} must be a compact bounded identifier")
    return result


def _identifiers(values: Any, name: str, *, required: bool = False) -> tuple[str, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, str):
        raw = (values,)
    elif isinstance(values, Sequence) and not isinstance(values, (bytes, bytearray)):
        raw = values
    else:
        raise ShadowRoutePolicyError(f"{name} must be a sequence of identifiers")
    if len(raw) > MAX_SEQUENCE_ITEMS:
        raise ShadowRoutePolicyError(f"{name} contains too many items")
    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw:
        identifier = _identifier(item, name)
        if identifier not in seen:
            seen.add(identifier)
            normalized.append(identifier)
    if required and not normalized:
        raise ShadowRoutePolicyError(f"{name} must not be empty")
    return tuple(normalized)


def _normalize_field_name(key: Any) -> str:
    if not isinstance(key, str):
        raise ShadowRoutePolicyError("route-policy field names must be strings")
    return key.strip().lower().replace("-", "_")


def field_is_forbidden(key: Any) -> bool:
    """Return whether a mapping key is secret, executable, or self-authorizing."""

    normalized = _normalize_field_name(key)
    if not normalized:
        return False
    if normalized in _FORBIDDEN_FEATURE_NAMES:
        return True
    return any(
        normalized == marker or normalized.endswith("_" + marker)
        for marker in _FORBIDDEN_FIELD_MARKERS
    )


def _reject_forbidden_payload(value: Any, name: str, *, depth: int = 0) -> None:
    if depth > MAX_NESTING_DEPTH:
        raise ShadowRoutePolicyError(f"{name} exceeds maximum nesting")
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float):
        raise ShadowRoutePolicyError(f"{name} cannot contain floats")
    if isinstance(value, (Enum, Fraction)):
        return
    if isinstance(value, str):
        return
    if isinstance(value, Mapping):
        if len(value) > MAX_MAPPING_ITEMS:
            raise ShadowRoutePolicyError(f"{name} contains too many entries")
        for raw_key, raw_value in value.items():
            if field_is_forbidden(raw_key):
                raise ShadowRoutePolicyError(
                    f"{name} contains forbidden private or executable data"
                )
            _reject_forbidden_payload(raw_value, f"{name}.{raw_key}", depth=depth + 1)
        return
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        if len(value) > MAX_SEQUENCE_ITEMS:
            raise ShadowRoutePolicyError(f"{name} contains too many items")
        for index, item in enumerate(value):
            _reject_forbidden_payload(item, f"{name}[{index}]", depth=depth + 1)
        return
    raise ShadowRoutePolicyError(f"{name} contains unsupported value type {type(value).__name__}")


def _fraction_from_payload(value: Any, name: str) -> Fraction:
    if isinstance(value, bool) or value is None:
        raise ShadowRoutePolicyError(f"{name} must be an integer or rational")
    if isinstance(value, float):
        raise ShadowRoutePolicyError(f"{name} cannot contain floats")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        if abs(value) > MAX_INTEGER:
            raise ShadowRoutePolicyError(f"{name} integer is out of range")
        return Fraction(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) != 2:
            raise ShadowRoutePolicyError(f"{name} rational must be a numerator/denominator pair")
        numerator, denominator = value
        if isinstance(numerator, bool) or isinstance(denominator, bool):
            raise ShadowRoutePolicyError(f"{name} must be an integer or rational")
        if (
            not isinstance(numerator, int)
            or not isinstance(denominator, int)
            or denominator == 0
            or abs(numerator) > MAX_INTEGER
            or abs(denominator) > MAX_INTEGER
        ):
            raise ShadowRoutePolicyError(f"{name} rational is malformed or unbounded")
        return Fraction(numerator, denominator)
    if isinstance(value, Mapping):
        extra = set(value).difference({"numerator", "denominator"})
        if extra:
            raise ShadowRoutePolicyError(f"{name} contains unsupported rational fields")
        return _fraction_from_payload(
            (value.get("numerator"), value.get("denominator")),
            name,
        )
    raise ShadowRoutePolicyError(f"{name} must be an integer or rational")


def _fraction_payload(value: Fraction) -> dict[str, int]:
    reduced = Fraction(value.numerator, value.denominator)
    if abs(reduced.numerator) > MAX_INTEGER or abs(reduced.denominator) > MAX_INTEGER:
        raise ShadowRoutePolicyError("rational value exceeds its integer bound")
    return {"numerator": reduced.numerator, "denominator": reduced.denominator}


def _feature_value(value: Any, name: str) -> Fraction:
    return _fraction_from_payload(value, name)


def _feature_vector(
    values: Mapping[str, Any] | None,
    feature_names: Sequence[str],
    *,
    name: str,
) -> tuple[Fraction, ...]:
    if values is None:
        payload: Mapping[str, Any] = {}
    elif isinstance(values, Mapping):
        payload = values
    else:
        raise ShadowRoutePolicyError(f"{name} must be a mapping of feature values")
    _reject_forbidden_payload(payload, name)
    extra = set(payload).difference(feature_names)
    if extra:
        raise ShadowRoutePolicyError(f"{name} references undeclared feature names")
    for key in payload:
        if field_is_forbidden(key) or key in _FORBIDDEN_FEATURE_NAMES:
            raise ShadowRoutePolicyError(f"{name} contains a forbidden feature name")
    return tuple(_feature_value(payload.get(item, 0), f"{name}.{item}") for item in feature_names)


def _authority_rank(authority: AuthorityClass) -> int:
    return tuple(AuthorityClass).index(authority)


def _privacy_rank(privacy: PrivacyClass) -> int:
    return tuple(PrivacyClass).index(privacy)


def _coerce_action(value: Any) -> MetaAction:
    if isinstance(value, ResolutionCandidate):
        return value.resolution_action.action
    if isinstance(value, ResolutionAction):
        return value.action
    return _enum(value, MetaAction, "action")


def _action_admission_reason(
    value: Any,
    *,
    allowed: frozenset[MetaAction],
    admitted: frozenset[MetaAction] | None,
    authority_ceiling: AuthorityClass | None,
    privacy_ceiling: PrivacyClass | None,
) -> str:
    action = _coerce_action(value)
    if action not in allowed:
        if isinstance(value, ResolutionCandidate) and not value.admissible:
            return "candidate_not_admissible"
        if isinstance(value, (ResolutionCandidate, ResolutionAction)):
            authority = (
                value.resolution_action.authority_class
                if isinstance(value, ResolutionCandidate)
                else value.authority_class
            )
            if authority_ceiling is not None and _authority_rank(authority) > _authority_rank(
                authority_ceiling
            ):
                return "authority_increase_denied"
        return "action_not_policy_admitted"
    if admitted is not None and action not in admitted:
        return "action_not_currently_admitted"
    if isinstance(value, ResolutionCandidate) and not value.admissible:
        return "candidate_not_admissible"
    if isinstance(value, (ResolutionCandidate, ResolutionAction)):
        bound = value.resolution_action if isinstance(value, ResolutionCandidate) else value
        if authority_ceiling is not None and _authority_rank(bound.authority_class) > _authority_rank(
            authority_ceiling
        ):
            return "authority_increase_denied"
        if privacy_ceiling is not None and _privacy_rank(bound.privacy_class) > _privacy_rank(
            privacy_ceiling
        ):
            return "privacy_policy_increase_denied"
        if bound.action in _REMOTE_MODEL_ACTIONS and bound.privacy_class in {
            PrivacyClass.LOCAL_ONLY,
            PrivacyClass.FORBIDDEN_EXTERNAL,
        }:
            return "privacy_class_forbids_remote_route"
    return ""


def _solve(
    matrix: Sequence[Sequence[Fraction]],
    vector: Sequence[Fraction],
) -> tuple[Fraction, ...]:
    size = len(vector)
    if size == 0:
        return ()
    if len(matrix) != size or any(len(row) != size for row in matrix):
        raise ShadowRoutePolicyError("gram matrix shape is inconsistent")
    tableau = [list(row) + [vector[index]] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot_row = max(range(column, size), key=lambda row: abs(tableau[row][column]))
        if tableau[pivot_row][column] == 0:
            raise ShadowRoutePolicyError("route-policy gram matrix is singular")
        tableau[column], tableau[pivot_row] = tableau[pivot_row], tableau[column]
        pivot = tableau[column][column]
        tableau[column] = [item / pivot for item in tableau[column]]
        for row in range(size):
            if row == column:
                continue
            factor = tableau[row][column]
            tableau[row] = [
                tableau[row][index] - factor * tableau[column][index]
                for index in range(size + 1)
            ]
    return tuple(tableau[index][size] for index in range(size))


def _dot(left: Sequence[Fraction], right: Sequence[Fraction]) -> Fraction:
    return sum((left[index] * right[index] for index in range(len(left))), Fraction(0))


def _identity(size: int, ridge: int) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(Fraction(ridge if row == column else 0) for column in range(size))
        for row in range(size)
    )


def _sqrt_bp(value: Fraction) -> int:
    if value < 0:
        raise ShadowRoutePolicyError("ucb variance cannot be negative")
    if value == 0:
        return 0
    scaled = (value.numerator * BASIS_POINTS * BASIS_POINTS) // value.denominator
    if scaled > MAX_INTEGER:
        raise ShadowRoutePolicyError("ucb variance exceeds its integer bound")
    return isqrt(max(0, scaled))


def _fraction_bp(value: Fraction) -> int:
    scaled = round(value * BASIS_POINTS)
    if abs(scaled) > MAX_INTEGER:
        raise ShadowRoutePolicyError("linear score exceeds its integer bound")
    return scaled


def _reward(observation: PolicyObservation) -> int:
    if observation.safety_violation:
        return -BASIS_POINTS
    if observation.terminal_status is TerminalStatus.PENDING:
        raise ShadowRoutePolicyError("pending observations cannot update or fit a route policy")
    if observation.terminal_status is TerminalStatus.SUCCEEDED:
        return max(1, observation.evidence_gain_bp)
    return -max(1, BASIS_POINTS - observation.evidence_gain_bp)


def _unchanged_policy_axes() -> Mapping[str, bool]:
    return MappingProxyType({axis: False for axis in PROTECTED_POLICY_AXES})


def _candidate_from(value: RoutePolicyCandidate | Mapping[str, Any]) -> RoutePolicyCandidate:
    if isinstance(value, RoutePolicyCandidate):
        candidate = value
    elif isinstance(value, Mapping):
        candidate = RoutePolicyCandidate.from_dict(value)
    else:
        raise ShadowRoutePolicyError("candidate must be a RoutePolicyCandidate")
    if not candidate.shadow_only:
        raise ShadowRoutePolicyError(
            "route-policy candidates are shadow-only until external promotion"
        )
    if candidate.external_authorization_id:
        raise ShadowRoutePolicyError("a candidate policy cannot authorize its own promotion")
    if len(candidate.feature_names) > MAX_ROUTE_FEATURES:
        raise ShadowRoutePolicyError("route-policy feature set exceeds its bound")
    for name in candidate.feature_names:
        if field_is_forbidden(name):
            raise ShadowRoutePolicyError("route-policy feature names include a forbidden field")
    return candidate


@dataclass(frozen=True)
class LinearUcbActionState:
    """Exact per-action ridge statistics used only in shadow UCB scoring."""

    gram: tuple[tuple[Fraction, ...], ...]
    bias: tuple[Fraction, ...]
    observation_count: int = 0

    def __post_init__(self) -> None:
        size = len(self.bias)
        gram = tuple(tuple(Fraction(item) for item in row) for row in self.gram)
        if size > MAX_ROUTE_FEATURES or len(gram) != size or any(len(row) != size for row in gram):
            raise ShadowRoutePolicyError("linear UCB action state shape is inconsistent")
        object.__setattr__(self, "gram", gram)
        object.__setattr__(
            self, "bias", tuple(Fraction(item) for item in self.bias)
        )
        object.__setattr__(
            self,
            "observation_count",
            _int(self.observation_count, "observation_count"),
        )

    def updated(self, features: Sequence[Fraction], reward: int) -> LinearUcbActionState:
        size = len(self.bias)
        if len(features) != size:
            raise ShadowRoutePolicyError("ucb feature vector does not match gram dimension")
        gram = [
            [
                self.gram[row][column] + features[row] * features[column]
                for column in range(size)
            ]
            for row in range(size)
        ]
        bias = [self.bias[index] + reward * features[index] for index in range(size)]
        for row in gram:
            for value in row:
                if abs(value.numerator) > MAX_INTEGER or abs(value.denominator) > MAX_INTEGER:
                    raise ShadowRoutePolicyError("ucb gram exceeds its integer bound")
        for value in bias:
            if abs(value.numerator) > MAX_INTEGER or abs(value.denominator) > MAX_INTEGER:
                raise ShadowRoutePolicyError("ucb bias exceeds its integer bound")
        return LinearUcbActionState(
            gram=tuple(tuple(row) for row in gram),
            bias=tuple(bias),
            observation_count=self.observation_count + 1,
        )

    def theta(self) -> tuple[Fraction, ...]:
        return _solve(self.gram, self.bias)

    def variance(self, features: Sequence[Fraction]) -> Fraction:
        inverse_product = _solve(self.gram, features)
        variance = _dot(features, inverse_product)
        if variance < 0:
            return Fraction(0)
        return variance

    def to_dict(self) -> dict[str, Any]:
        return {
            "gram": [[_fraction_payload(item) for item in row] for row in self.gram],
            "bias": [_fraction_payload(item) for item in self.bias],
            "observation_count": self.observation_count,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any] | LinearUcbActionState
    ) -> LinearUcbActionState:
        if isinstance(payload, LinearUcbActionState):
            return payload
        if not isinstance(payload, Mapping):
            raise ShadowRoutePolicyError("linear UCB action state must be an object")
        gram_payload = payload.get("gram") or ()
        bias_payload = payload.get("bias") or ()
        gram = tuple(
            tuple(_fraction_from_payload(item, "gram") for item in row)
            for row in gram_payload
        )
        bias = tuple(_fraction_from_payload(item, "bias") for item in bias_payload)
        return cls(
            gram=gram,
            bias=bias,
            observation_count=payload.get("observation_count", 0),
        )


@dataclass(frozen=True)
class LinearUcbState:
    """Disjoint linear-UCB statistics for the closed admitted action set."""

    feature_names: tuple[str, ...]
    actions: Mapping[str, LinearUcbActionState]
    ridge_lambda: int = DEFAULT_RIDGE_LAMBDA
    alpha_bp: int = DEFAULT_UCB_ALPHA_BP

    def __post_init__(self) -> None:
        names = _identifiers(self.feature_names, "feature_names", required=True)
        if len(names) > MAX_ROUTE_FEATURES:
            raise ShadowRoutePolicyError("route-policy feature set exceeds its bound")
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "ridge_lambda", _int(self.ridge_lambda, "ridge_lambda", minimum=1))
        object.__setattr__(self, "alpha_bp", _int(self.alpha_bp, "alpha_bp", maximum=BASIS_POINTS))
        if not isinstance(self.actions, Mapping):
            raise ShadowRoutePolicyError("ucb actions must be a mapping")
        if len(self.actions) > MAX_SEQUENCE_ITEMS:
            raise ShadowRoutePolicyError("ucb actions exceed the closed action bound")
        normalized: dict[str, LinearUcbActionState] = {}
        for raw_key, raw_state in self.actions.items():
            action = _enum(raw_key, MetaAction, "ucb_action")
            state = LinearUcbActionState.from_dict(raw_state)
            if len(state.bias) != len(names):
                raise ShadowRoutePolicyError("ucb action dimension does not match feature_names")
            normalized[action.value] = state
        object.__setattr__(self, "actions", MappingProxyType(dict(sorted(normalized.items()))))

    @classmethod
    def initial(
        cls,
        candidate: RoutePolicyCandidate,
        *,
        ridge_lambda: int = DEFAULT_RIDGE_LAMBDA,
        alpha_bp: int = DEFAULT_UCB_ALPHA_BP,
    ) -> LinearUcbState:
        size = len(candidate.feature_names)
        blank = LinearUcbActionState(
            gram=_identity(size, ridge_lambda),
            bias=tuple(Fraction(0) for _ in range(size)),
            observation_count=0,
        )
        return cls(
            feature_names=candidate.feature_names,
            ridge_lambda=ridge_lambda,
            alpha_bp=alpha_bp,
            actions={action.value: blank for action in candidate.allowed_actions},
        )

    def for_action(self, action: MetaAction) -> LinearUcbActionState:
        try:
            return self.actions[action.value]
        except KeyError as exc:
            raise ShadowRoutePolicyError("ucb state is missing a policy-admitted action") from exc

    def updated(
        self, action: MetaAction, features: Sequence[Fraction], reward: int
    ) -> LinearUcbState:
        if action.value not in self.actions:
            raise ShadowRoutePolicyError("cannot observe an action outside the closed set")
        replaced = dict(self.actions)
        replaced[action.value] = self.actions[action.value].updated(features, reward)
        return LinearUcbState(
            feature_names=self.feature_names,
            actions=replaced,
            ridge_lambda=self.ridge_lambda,
            alpha_bp=self.alpha_bp,
        )

    @property
    def state_id(self) -> str:
        payload = dict(self.to_dict())
        payload.pop("state_id", None)
        return content_identity(payload)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": LINEAR_UCB_STATE_SCHEMA,
            "contract_version": CONTRACT_VERSION,
            "feature_names": list(self.feature_names),
            "ridge_lambda": self.ridge_lambda,
            "alpha_bp": self.alpha_bp,
            "actions": {
                name: self.actions[name].to_dict() for name in sorted(self.actions)
            },
        }
        payload["state_id"] = content_identity(
            {key: value for key, value in payload.items() if key != "state_id"}
        )
        encoded = canonical_json_bytes(payload)
        if len(encoded) > MAX_CANONICAL_RECORD_BYTES:
            raise ShadowRoutePolicyError("linear UCB state exceeds its bounded size")
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | LinearUcbState) -> LinearUcbState:
        if isinstance(payload, LinearUcbState):
            return payload
        if not isinstance(payload, Mapping):
            raise ShadowRoutePolicyError("linear UCB state must be an object")
        _reject_forbidden_payload(payload, "ucb_state")
        extra = set(payload).difference(
            {
                "schema",
                "contract_version",
                "content_id",
                "state_id",
                "feature_names",
                "ridge_lambda",
                "alpha_bp",
                "actions",
            }
        )
        if extra:
            raise ShadowRoutePolicyError("linear UCB state contains unsupported fields")
        return cls(
            feature_names=payload.get("feature_names") or (),
            actions=payload.get("actions") or {},
            ridge_lambda=payload.get("ridge_lambda", DEFAULT_RIDGE_LAMBDA),
            alpha_bp=payload.get("alpha_bp", DEFAULT_UCB_ALPHA_BP),
        )


@dataclass(frozen=True)
class ShadowRouteScore:
    """Deterministic integer-unit score for one already admitted action."""

    action: MetaAction
    linear_score: Fraction
    ucb_variance: Fraction
    ucb_bonus_bp: int
    total_bp: int
    feature_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]
    admissible: bool
    observation_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", _enum(self.action, MetaAction, "action"))
        object.__setattr__(self, "linear_score", Fraction(self.linear_score))
        object.__setattr__(self, "ucb_variance", Fraction(self.ucb_variance))
        object.__setattr__(self, "ucb_bonus_bp", _int(self.ucb_bonus_bp, "ucb_bonus_bp"))
        object.__setattr__(
            self,
            "total_bp",
            _int(self.total_bp, "total_bp", minimum=-MAX_INTEGER),
        )
        object.__setattr__(
            self, "feature_ids", _identifiers(self.feature_ids, "feature_ids", required=True)
        )
        object.__setattr__(
            self,
            "reason_codes",
            _identifiers(self.reason_codes, "reason_codes", required=True),
        )
        object.__setattr__(self, "admissible", _bool(self.admissible, "admissible"))
        object.__setattr__(
            self, "observation_count", _int(self.observation_count, "observation_count")
        )

    @property
    def score_id(self) -> str:
        payload = dict(self.to_dict())
        payload.pop("score_id", None)
        return content_identity(payload)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": SHADOW_ROUTE_SCORE_SCHEMA,
            "contract_version": CONTRACT_VERSION,
            "action": self.action.value,
            "linear_score": _fraction_payload(self.linear_score),
            "ucb_variance": _fraction_payload(self.ucb_variance),
            "ucb_bonus_bp": self.ucb_bonus_bp,
            "total_bp": self.total_bp,
            "feature_ids": list(self.feature_ids),
            "reason_codes": list(self.reason_codes),
            "admissible": self.admissible,
            "observation_count": self.observation_count,
        }
        payload["score_id"] = content_identity(
            {key: value for key, value in payload.items() if key != "score_id"}
        )
        return payload


@dataclass(frozen=True)
class ShadowRouteSelection:
    """Shadow ranking result.  It is never a live routing permit."""

    scores: tuple[ShadowRouteScore, ...]
    candidate_id: str
    policy_version: str
    mode: SelectionMode
    disposition: SelectionDisposition
    reason_codes: tuple[str, ...]
    selected_action: MetaAction | None = None
    observation: PolicyObservation | None = None
    shadow_only: bool = True
    live_routing_effect: bool = False
    production_exploration: bool = False
    affects_production_acceptance: bool = False
    policy_axis_changes: Mapping[str, bool] = field(default_factory=_unchanged_policy_axes)

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", _enum(self.mode, SelectionMode, "mode"))
        object.__setattr__(
            self, "disposition", _enum(self.disposition, SelectionDisposition, "disposition")
        )
        object.__setattr__(self, "candidate_id", _identifier(self.candidate_id, "candidate_id"))
        object.__setattr__(
            self, "policy_version", _identifier(self.policy_version, "policy_version")
        )
        object.__setattr__(
            self,
            "reason_codes",
            _identifiers(self.reason_codes, "reason_codes", required=True),
        )
        scores = tuple(self.scores)
        if len(scores) > MAX_SEQUENCE_ITEMS:
            raise ShadowRoutePolicyError("shadow selection contains too many scores")
        if any(not isinstance(item, ShadowRouteScore) for item in scores):
            raise ShadowRoutePolicyError("shadow scores must be ShadowRouteScore values")
        object.__setattr__(self, "scores", scores)
        selected = (
            None
            if self.selected_action is None
            else _enum(self.selected_action, MetaAction, "selected_action")
        )
        object.__setattr__(self, "selected_action", selected)
        if self.observation is not None and not isinstance(self.observation, PolicyObservation):
            raise ShadowRoutePolicyError("observation must be a PolicyObservation")
        for name in (
            "shadow_only",
            "live_routing_effect",
            "production_exploration",
            "affects_production_acceptance",
        ):
            object.__setattr__(self, name, _bool(getattr(self, name), name))
        if not self.shadow_only:
            raise ShadowRoutePolicyError(
                "route-policy candidates are shadow-only until external promotion"
            )
        if self.live_routing_effect or self.production_exploration:
            raise ShadowRoutePolicyError(
                "shadow route policy produces no live routing effect before external promotion"
            )
        if self.affects_production_acceptance:
            raise ShadowRoutePolicyError("shadow route policy cannot affect production acceptance")
        axes = _unchanged_policy_axes()
        supplied = self.policy_axis_changes or axes
        if not isinstance(supplied, Mapping):
            raise ShadowRoutePolicyError("policy_axis_changes must be a mapping")
        normalized = {axis: False for axis in PROTECTED_POLICY_AXES}
        for key, value in supplied.items():
            axis = _identifier(key, "policy_axis_changes")
            if axis not in normalized:
                raise ShadowRoutePolicyError("policy_axis_changes names an unknown policy axis")
            if _bool(value, "policy_axis_changes"):
                raise ShadowRoutePolicyError(
                    "shadow route policy cannot change provider, authority, privacy, "
                    "validation, proof, or confirmation policy"
                )
            normalized[axis] = False
        object.__setattr__(self, "policy_axis_changes", MappingProxyType(normalized))
        if self.disposition is SelectionDisposition.SELECTED:
            if selected is None or self.observation is None:
                raise ShadowRoutePolicyError("selected shadow rankings require an observation")
            if self.observation.selected_action is not selected:
                raise ShadowRoutePolicyError("observation action does not match the selection")
            if not any(item.action is selected and item.admissible for item in scores):
                raise ShadowRoutePolicyError("learner can choose only already policy-admitted actions")
        elif selected is not None or self.observation is not None:
            raise ShadowRoutePolicyError("abstained rankings cannot bind a selected action")
        encoded = canonical_json_bytes(self.to_dict())
        if len(encoded) > MAX_CANONICAL_RECORD_BYTES:
            raise ShadowRoutePolicyError("shadow selection exceeds its bounded size")

    @property
    def selection_id(self) -> str:
        payload = dict(self.to_dict())
        payload.pop("selection_id", None)
        return content_identity(payload)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": SHADOW_ROUTE_SELECTION_SCHEMA,
            "contract_version": CONTRACT_VERSION,
            "scores": [item.to_dict() for item in self.scores],
            "candidate_id": self.candidate_id,
            "policy_version": self.policy_version,
            "mode": self.mode.value,
            "disposition": self.disposition.value,
            "reason_codes": list(self.reason_codes),
            "selected_action": None if self.selected_action is None else self.selected_action.value,
            "observation": None if self.observation is None else self.observation.to_dict(),
            "shadow_only": True,
            "live_routing_effect": False,
            "production_exploration": False,
            "affects_production_acceptance": False,
            "policy_axis_changes": dict(self.policy_axis_changes),
        }
        payload["selection_id"] = content_identity(
            {key: value for key, value in payload.items() if key != "selection_id"}
        )
        return payload


@dataclass(frozen=True)
class RoutePolicySnapshot:
    """Exact versioned rollback vector for one shadow candidate."""

    candidate: RoutePolicyCandidate
    ucb_state: LinearUcbState

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate", _candidate_from(self.candidate))
        object.__setattr__(self, "ucb_state", LinearUcbState.from_dict(self.ucb_state))
        if tuple(self.ucb_state.feature_names) != tuple(self.candidate.feature_names):
            raise ShadowRoutePolicyError("rollback snapshot feature names do not match the candidate")
        allowed = {action.value for action in self.candidate.allowed_actions}
        if set(self.ucb_state.actions) != allowed:
            raise ShadowRoutePolicyError("rollback snapshot action set does not match the candidate")

    @property
    def snapshot_id(self) -> str:
        payload = dict(self.to_dict())
        payload.pop("snapshot_id", None)
        return content_identity(payload)

    @property
    def candidate_id(self) -> str:
        return self.candidate.candidate_id

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": ROUTE_POLICY_SNAPSHOT_SCHEMA,
            "contract_version": CONTRACT_VERSION,
            "candidate": self.candidate.to_dict(),
            "ucb_state": self.ucb_state.to_dict(),
        }
        payload["snapshot_id"] = content_identity(
            {key: value for key, value in payload.items() if key != "snapshot_id"}
        )
        return payload


def _ridge_weights(
    feature_names: Sequence[str],
    rows: Sequence[tuple[tuple[Fraction, ...], int]],
    *,
    ridge_lambda: int,
) -> dict[str, int]:
    size = len(feature_names)
    gram = [list(row) for row in _identity(size, ridge_lambda)]
    bias = [Fraction(0) for _ in range(size)]
    for features, reward in rows:
        if len(features) != size:
            raise ShadowRoutePolicyError("training feature vector does not match feature_names")
        for row in range(size):
            bias[row] += features[row] * reward
            for column in range(size):
                gram[row][column] += features[row] * features[column]
    theta = _solve(gram, bias)
    weights = {name: round(theta[index]) for index, name in enumerate(feature_names)}
    for value in weights.values():
        if abs(value) > MAX_INTEGER:
            raise ShadowRoutePolicyError("fitted weights exceed the integer bound")
    return weights


class ShadowRoutePolicy:
    """Constrained shadow learner over an already admitted closed action set."""

    INTERFACE = SHADOW_ROUTE_POLICY_INTERFACE
    CANDIDATE_INTERFACE = ROUTE_POLICY_CANDIDATE_INTERFACE
    SCHEMA = SHADOW_ROUTE_POLICY_SCHEMA

    def __init__(
        self,
        candidate: RoutePolicyCandidate | Mapping[str, Any],
        *,
        ucb_state: LinearUcbState | Mapping[str, Any] | None = None,
        snapshots: Sequence[RoutePolicySnapshot | Mapping[str, Any]] = (),
        ridge_lambda: int = DEFAULT_RIDGE_LAMBDA,
        alpha_bp: int = DEFAULT_UCB_ALPHA_BP,
    ) -> None:
        bound = _candidate_from(candidate)
        if ucb_state is None:
            state = LinearUcbState.initial(
                bound, ridge_lambda=ridge_lambda, alpha_bp=alpha_bp
            )
        else:
            state = LinearUcbState.from_dict(ucb_state)
        if tuple(state.feature_names) != tuple(bound.feature_names):
            raise ShadowRoutePolicyError("ucb feature names must match the candidate")
        allowed = {action.value for action in bound.allowed_actions}
        if set(state.actions) != allowed:
            raise ShadowRoutePolicyError("ucb action set must equal the candidate closed set")
        if len(snapshots) > MAX_POLICY_VERSIONS:
            raise ShadowRoutePolicyError("policy lineage exceeds its bound")
        bound_snapshots: list[RoutePolicySnapshot] = []
        for item in snapshots:
            if isinstance(item, RoutePolicySnapshot):
                bound_snapshots.append(item)
            elif isinstance(item, Mapping):
                bound_snapshots.append(
                    RoutePolicySnapshot(
                        candidate=_candidate_from(item["candidate"]),
                        ucb_state=LinearUcbState.from_dict(item["ucb_state"]),
                    )
                )
            else:
                raise ShadowRoutePolicyError("snapshots must be RoutePolicySnapshot values")
        bound_snapshots_tuple = tuple(bound_snapshots)
        self._candidate = bound
        self._ucb_state = state
        self._snapshots = bound_snapshots_tuple

    @property
    def candidate(self) -> RoutePolicyCandidate:
        return self._candidate

    @property
    def ucb_state(self) -> LinearUcbState:
        return self._ucb_state

    @property
    def snapshots(self) -> tuple[RoutePolicySnapshot, ...]:
        return self._snapshots

    @property
    def lineage_ids(self) -> tuple[str, ...]:
        return tuple(item.candidate_id for item in self._snapshots) + (
            self._candidate.candidate_id,
        )

    @property
    def rollback_vectors(self) -> tuple[Mapping[str, str], ...]:
        current = self._candidate
        vectors = [
            MappingProxyType(
                {
                    "from_candidate_id": current.candidate_id,
                    "to_candidate_id": item.candidate_id,
                    "policy_version": item.candidate.policy_version,
                    "parent_policy_id": item.candidate.parent_policy_id,
                }
            )
            for item in reversed(self._snapshots)
        ]
        return tuple(vectors)

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
    def affects_production_acceptance(self) -> bool:
        return False

    @property
    def policy_axis_changes(self) -> Mapping[str, bool]:
        return _unchanged_policy_axes()

    def allowed_actions(self) -> tuple[MetaAction, ...]:
        return self._candidate.allowed_actions

    def _linear_score(self, features: Sequence[Fraction]) -> Fraction:
        weights = self._candidate.integer_weights
        names = self._candidate.feature_names
        total = Fraction(0)
        for index, name in enumerate(names):
            total += Fraction(weights.get(name, 0)) * features[index]
        return total

    def _score_action(
        self,
        action: MetaAction,
        features: Sequence[Fraction],
        *,
        mode: SelectionMode,
        reason: str,
    ) -> ShadowRouteScore:
        linear = self._linear_score(features)
        reasons = [reason] if reason else []
        admissible = not reason
        variance = Fraction(0)
        bonus_bp = 0
        count = 0
        if admissible:
            state = self._ucb_state.for_action(action)
            count = state.observation_count
            reasons.append("closed_action_set")
            reasons.append("linear_score")
            if mode is SelectionMode.LINEAR_UCB:
                variance = state.variance(features)
                # Constrained linear UCB: bound the bonus independently of raw
                # feature magnitude so token/cost scale cannot mint exploration
                # of an unadmitted or merely large-magnitude action.
                norm_sq = _dot(features, features)
                unit_variance = variance / norm_sq if norm_sq != 0 else variance
                bonus_bp = (self._ucb_state.alpha_bp * _sqrt_bp(unit_variance)) // BASIS_POINTS
                reasons.append("linear_ucb")
            reasons.append("shadow_only_enforced")
            reasons.append("production_exploration_denied")
            reasons.append("live_routing_effect_absent")
        total = _fraction_bp(linear) + (bonus_bp if admissible else 0)
        return ShadowRouteScore(
            action=action,
            linear_score=linear,
            ucb_variance=variance,
            ucb_bonus_bp=bonus_bp if admissible else 0,
            total_bp=total if admissible else _fraction_bp(linear),
            feature_ids=self._candidate.feature_names,
            reason_codes=tuple(reasons) or ("action_not_policy_admitted",),
            admissible=admissible,
            observation_count=count,
        )

    def score(
        self,
        features_by_action: Mapping[Any, Mapping[str, Any]],
        *,
        admitted_actions: Sequence[Any] | None = None,
        mode: SelectionMode | str = SelectionMode.LINEAR_SCORE,
        authority_ceiling: AuthorityClass | str | None = None,
        privacy_ceiling: PrivacyClass | str | None = None,
        production: bool = False,
        explore_production: bool = False,
    ) -> tuple[ShadowRouteScore, ...]:
        """Score only already admitted actions.  Unadmitted actions are ineligible."""

        if production or explore_production:
            raise ShadowRoutePolicyError("no production exploration is allowed")
        bound_mode = _enum(mode, SelectionMode, "mode")
        if not isinstance(features_by_action, Mapping):
            raise ShadowRoutePolicyError("features_by_action must be a mapping")
        if len(features_by_action) > MAX_SEQUENCE_ITEMS:
            raise ShadowRoutePolicyError("features_by_action contains too many actions")
        allowed = frozenset(self._candidate.allowed_actions)
        admitted: frozenset[MetaAction] | None
        if admitted_actions is None:
            admitted = None
        else:
            if isinstance(admitted_actions, (str, bytes, bytearray, Mapping)):
                raise ShadowRoutePolicyError("admitted_actions must be a sequence")
            admitted = frozenset(_coerce_action(item) for item in admitted_actions)
        ceiling = (
            None
            if authority_ceiling is None
            else _enum(authority_ceiling, AuthorityClass, "authority_ceiling")
        )
        privacy = (
            None
            if privacy_ceiling is None
            else _enum(privacy_ceiling, PrivacyClass, "privacy_ceiling")
        )
        scores: list[ShadowRouteScore] = []
        seen: set[MetaAction] = set()
        for raw_action, raw_features in features_by_action.items():
            action = _coerce_action(raw_action)
            if action in seen:
                raise ShadowRoutePolicyError("features_by_action contains duplicate actions")
            seen.add(action)
            reason = _action_admission_reason(
                raw_action,
                allowed=allowed,
                admitted=admitted,
                authority_ceiling=ceiling,
                privacy_ceiling=privacy,
            )
            if reason == "" and action not in allowed:
                reason = "action_not_policy_admitted"
            features = _feature_vector(
                raw_features, self._candidate.feature_names, name=f"features.{action.value}"
            )
            scores.append(
                self._score_action(action, features, mode=bound_mode, reason=reason)
            )
        if admitted_actions is not None:
            for raw_action in admitted_actions:
                action = _coerce_action(raw_action)
                if action in seen:
                    continue
                reason = _action_admission_reason(
                    raw_action,
                    allowed=allowed,
                    admitted=admitted,
                    authority_ceiling=ceiling,
                    privacy_ceiling=privacy,
                ) or "missing_action_features"
                zeros = tuple(Fraction(0) for _ in self._candidate.feature_names)
                scores.append(
                    self._score_action(action, zeros, mode=bound_mode, reason=reason)
                )
        return tuple(sorted(scores, key=lambda item: item.action.value))

    def select(
        self,
        features_by_action: Mapping[Any, Mapping[str, Any]],
        *,
        episode_id: str,
        admitted_actions: Sequence[Any] | None = None,
        mode: SelectionMode | str = SelectionMode.LINEAR_SCORE,
        terminal_status: TerminalStatus | str = TerminalStatus.PENDING,
        accepted_criterion_ids: Sequence[str] = (),
        evidence_gain_bp: int = 0,
        cost_micros: int = 0,
        latency_ms: int = 0,
        safety_violation: bool = False,
        authority_ceiling: AuthorityClass | str | None = None,
        privacy_ceiling: PrivacyClass | str | None = None,
        production: bool = False,
        explore_production: bool = False,
        live: bool = False,
    ) -> ShadowRouteSelection:
        """Rank admitted actions in shadow.  Never a live routing effect."""

        if production or explore_production or live:
            raise ShadowRoutePolicyError(
                "shadow route policy produces no live routing effect before external promotion"
            )
        bound_mode = _enum(mode, SelectionMode, "mode")
        scores = self.score(
            features_by_action,
            admitted_actions=admitted_actions,
            mode=bound_mode,
            authority_ceiling=authority_ceiling,
            privacy_ceiling=privacy_ceiling,
        )
        eligible = tuple(item for item in scores if item.admissible)
        reasons = [
            "shadow_only_enforced",
            "closed_action_set",
            "production_exploration_denied",
            "live_routing_effect_absent",
            bound_mode.value,
        ]
        if not eligible:
            return ShadowRouteSelection(
                scores=scores,
                candidate_id=self._candidate.candidate_id,
                policy_version=self._candidate.policy_version,
                mode=bound_mode,
                disposition=SelectionDisposition.ABSTAINED,
                reason_codes=tuple(reasons + ["no_policy_admitted_action"]),
            )
        allowed_order = {action: index for index, action in enumerate(self._candidate.allowed_actions)}
        ranked = sorted(
            eligible,
            key=lambda item: (
                -item.total_bp,
                allowed_order.get(item.action, len(allowed_order)),
                item.action.value,
            ),
        )
        chosen = ranked[0]
        if len(ranked) > 1 and ranked[1].total_bp == chosen.total_bp:
            reasons.append("tie_broken_by_allowed_action_order")
        observation = PolicyObservation(
            episode_id=_identifier(episode_id, "episode_id"),
            route_policy_id=self._candidate.candidate_id,
            selected_action=chosen.action,
            selection_reason_codes=tuple(reasons),
            feature_ids=self._candidate.feature_names,
            terminal_status=_enum(terminal_status, TerminalStatus, "terminal_status"),
            action_propensity_bp=BASIS_POINTS,
            accepted_criterion_ids=accepted_criterion_ids,
            evidence_gain_bp=evidence_gain_bp,
            cost_micros=cost_micros,
            latency_ms=latency_ms,
            safety_violation=safety_violation,
        )
        return ShadowRouteSelection(
            scores=scores,
            candidate_id=self._candidate.candidate_id,
            policy_version=self._candidate.policy_version,
            mode=bound_mode,
            disposition=SelectionDisposition.SELECTED,
            reason_codes=observation.selection_reason_codes,
            selected_action=chosen.action,
            observation=observation,
        )

    def observe(
        self,
        observation: PolicyObservation | Mapping[str, Any],
        features: Mapping[str, Any],
    ) -> ShadowRoutePolicy:
        """Update shadow UCB counts.  This does not change live routing."""

        bound = (
            observation
            if isinstance(observation, PolicyObservation)
            else PolicyObservation.from_dict(observation)
        )
        if bound.route_policy_id not in {self._candidate.candidate_id, self._candidate.parent_policy_id}:
            raise ShadowRoutePolicyError("observation is not bound to this route-policy version")
        if bound.selected_action not in self._candidate.allowed_actions:
            raise ShadowRoutePolicyError("learner can choose only already policy-admitted actions")
        vector = _feature_vector(
            features, self._candidate.feature_names, name="observation_features"
        )
        updated = self._ucb_state.updated(bound.selected_action, vector, _reward(bound))
        return ShadowRoutePolicy(
            self._candidate, ucb_state=updated, snapshots=self._snapshots
        )

    def propose(
        self,
        observations: Sequence[PolicyObservation | Mapping[str, Any]],
        *,
        features_by_observation: Mapping[str, Mapping[str, Any]],
        held_out_evaluation_ids: Sequence[str],
        safety_gate_receipt_ids: Sequence[str],
        policy_version: str,
        selection_reason: str = "linear_score",
        allowed_actions: Sequence[MetaAction | str] | None = None,
    ) -> RoutePolicyCandidate:
        """Fit integer weights.  The closed action set cannot grow."""

        if not isinstance(observations, Sequence) or isinstance(
            observations, (str, bytes, bytearray)
        ):
            raise ShadowRoutePolicyError("observations must be a sequence")
        if len(observations) > MAX_SEQUENCE_ITEMS:
            raise ShadowRoutePolicyError("observations contain too many items")
        if not isinstance(features_by_observation, Mapping):
            raise ShadowRoutePolicyError("features_by_observation must be a mapping")
        bound_observations = tuple(
            item if isinstance(item, PolicyObservation) else PolicyObservation.from_dict(item)
            for item in observations
        )
        parent_allowed = frozenset(self._candidate.allowed_actions)
        if allowed_actions is None:
            next_allowed = self._candidate.allowed_actions
        else:
            next_allowed = tuple(_coerce_action(item) for item in allowed_actions)
            if len(next_allowed) != len(set(next_allowed)):
                raise ShadowRoutePolicyError("proposed allowed_actions contain duplicates")
            extra = [item for item in next_allowed if item not in parent_allowed]
            if extra:
                raise ShadowRoutePolicyError(
                    "learner can choose only already policy-admitted actions"
                )
            if not next_allowed:
                raise ShadowRoutePolicyError("proposed allowed_actions must not be empty")
        rows: list[tuple[tuple[Fraction, ...], int]] = []
        training_ids: list[str] = []
        for item in bound_observations:
            if item.selected_action not in parent_allowed:
                raise ShadowRoutePolicyError(
                    "learner can choose only already policy-admitted actions"
                )
            if item.safety_violation and item.terminal_status is TerminalStatus.SUCCEEDED:
                raise ShadowRoutePolicyError("a safety violation cannot be fitted as success")
            if item.observation_id not in features_by_observation:
                raise ShadowRoutePolicyError("training observation is missing features")
            vector = _feature_vector(
                features_by_observation[item.observation_id],
                self._candidate.feature_names,
                name=f"features.{item.observation_id}",
            )
            rows.append((vector, _reward(item)))
            training_ids.append(item.observation_id)
        if not rows:
            raise ShadowRoutePolicyError("fitting requires completed training observations")
        weights = _ridge_weights(
            self._candidate.feature_names, rows, ridge_lambda=self._ucb_state.ridge_lambda
        )
        return RoutePolicyCandidate(
            parent_policy_id=self._candidate.candidate_id,
            policy_version=_identifier(policy_version, "policy_version"),
            allowed_actions=next_allowed,
            feature_names=self._candidate.feature_names,
            integer_weights=weights,
            training_observation_ids=tuple(training_ids),
            held_out_evaluation_ids=held_out_evaluation_ids,
            safety_gate_receipt_ids=safety_gate_receipt_ids,
            selection_reason=_identifier(selection_reason, "selection_reason"),
            shadow_only=True,
            external_authorization_id="",
        )

    def fit(
        self,
        observations: Sequence[PolicyObservation | Mapping[str, Any]],
        *,
        features_by_observation: Mapping[str, Mapping[str, Any]],
        held_out_evaluation_ids: Sequence[str],
        safety_gate_receipt_ids: Sequence[str],
        policy_version: str,
        selection_reason: str = "linear_score",
        allowed_actions: Sequence[MetaAction | str] | None = None,
    ) -> ShadowRoutePolicy:
        """Commit a fitted shadow candidate.  Live routing is unchanged."""

        proposed = self.propose(
            observations,
            features_by_observation=features_by_observation,
            held_out_evaluation_ids=held_out_evaluation_ids,
            safety_gate_receipt_ids=safety_gate_receipt_ids,
            policy_version=policy_version,
            selection_reason=selection_reason,
            allowed_actions=allowed_actions,
        )
        return self.commit(proposed)

    def commit(self, candidate: RoutePolicyCandidate | Mapping[str, Any]) -> ShadowRoutePolicy:
        """Install a new shadow candidate and snapshot the previous version."""

        bound = _candidate_from(candidate)
        parent_allowed = frozenset(self._candidate.allowed_actions)
        if any(action not in parent_allowed for action in bound.allowed_actions):
            raise ShadowRoutePolicyError(
                "learner can choose only already policy-admitted actions"
            )
        if bound.feature_names != self._candidate.feature_names:
            raise ShadowRoutePolicyError("fitted candidates cannot change the declared feature set")
        snapshot = RoutePolicySnapshot(candidate=self._candidate, ucb_state=self._ucb_state)
        snapshots = self._snapshots + (snapshot,)
        if len(snapshots) > MAX_POLICY_VERSIONS:
            raise ShadowRoutePolicyError("policy lineage exceeds its bound")
        return ShadowRoutePolicy(
            bound,
            ucb_state=LinearUcbState.initial(
                bound,
                ridge_lambda=self._ucb_state.ridge_lambda,
                alpha_bp=self._ucb_state.alpha_bp,
            ),
            snapshots=snapshots,
        )

    def rollback(self, candidate_id: str | None = None) -> ShadowRoutePolicy:
        """Restore an exact prior candidate.  History is not rewritten."""

        if not self._snapshots:
            raise ShadowRoutePolicyError("no versioned route-policy candidate is available to roll back")
        if candidate_id is None:
            snapshot = self._snapshots[-1]
            remaining = self._snapshots[:-1]
        else:
            target = _identifier(candidate_id, "candidate_id")
            index = None
            for position, item in enumerate(self._snapshots):
                if item.candidate_id == target:
                    index = position
            if index is None:
                if target == self._candidate.candidate_id:
                    return ShadowRoutePolicy(
                        self._candidate,
                        ucb_state=self._ucb_state,
                        snapshots=self._snapshots,
                    )
                raise ShadowRoutePolicyError(
                    "rollback target is not an exact prior versioned candidate"
                )
            snapshot = self._snapshots[index]
            remaining = self._snapshots[:index]
        return ShadowRoutePolicy(
            snapshot.candidate, ucb_state=snapshot.ucb_state, snapshots=remaining
        )

    def promote(self, authorization_id: str = "") -> None:
        del authorization_id
        raise ShadowRoutePolicyError("a candidate policy cannot authorize its own promotion")

    def apply_live_route(self) -> None:
        raise ShadowRoutePolicyError(
            "shadow route policy produces no live routing effect before external promotion"
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": SHADOW_ROUTE_POLICY_SCHEMA,
            "contract_version": CONTRACT_VERSION,
            "interface": self.INTERFACE,
            "candidate": self._candidate.to_dict(),
            "ucb_state": self._ucb_state.to_dict(),
            "snapshots": [item.to_dict() for item in self._snapshots],
            "shadow_only": True,
            "live_routing_effect": False,
            "production_exploration": False,
            "affects_production_acceptance": False,
            "policy_axis_changes": dict(self.policy_axis_changes),
        }
        encoded = canonical_json_bytes(payload)
        if len(encoded) > MAX_CANONICAL_RECORD_BYTES:
            raise ShadowRoutePolicyError("shadow route policy exceeds its bounded size")
        return payload


def shadow_route_policy(
    candidate: RoutePolicyCandidate | Mapping[str, Any],
    **kwargs: Any,
) -> ShadowRoutePolicy:
    """Construct a shadow-only constrained route policy."""

    return ShadowRoutePolicy(candidate, **kwargs)


__all__ = [
    "BASIS_POINTS",
    "LINEAR_UCB_STATE_SCHEMA",
    "PROTECTED_POLICY_AXES",
    "ROUTE_POLICY_CANDIDATE_INTERFACE",
    "SHADOW_ROUTE_POLICY_INTERFACE",
    "SHADOW_ROUTE_POLICY_SCHEMA",
    "LinearUcbActionState",
    "LinearUcbState",
    "RoutePolicySnapshot",
    "SelectionDisposition",
    "SelectionMode",
    "ShadowRoutePolicy",
    "ShadowRoutePolicyError",
    "ShadowRouteScore",
    "ShadowRouteSelection",
    "field_is_forbidden",
    "shadow_route_policy",
]
