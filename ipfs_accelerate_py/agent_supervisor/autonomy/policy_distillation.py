# ruff: noqa: UP042 - the package retains Python 3.8 compatibility
"""CEGIS-style distillation of bounded declarative decision rules.

``PolicyDistiller@1`` converts independently validated, repeatedly observed
decision classes into :class:`DistilledDecisionRule` values.  It emits only
the closed declarative ``when`` DSL, never Python, shell, or model-generated
executable policy.  Successful candidates stay shadow-only: promotion is
owned by APMC-019 and cannot be performed here.

Admission rules
---------------
* Feature keys are the closed distilled ``when`` vocabulary.
* A class must have enough independent validated examples, one stable
  output action, and a non-empty common feature conjunction.
* Emitted rules include every stable common equality, so they are not
  broader than the evidenced class.  Counterexamples can only add further
  conjuncts.
* Out-of-domain inputs, including adversarial feature mutations, take the
  explicit fallback action and never the distilled action.
* Missing validation or causal attribution is insufficiency, not a rule.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
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
    CausalAttribution,
    DistillationCandidate,
    DistillationStatus,
    DistilledDecisionRule,
    ExperienceEpisode,
    MetaAction,
    TerminalStatus,
)

POLICY_DISTILLER_INTERFACE: Final[str] = "PolicyDistiller@1"
DISTILLED_DECISION_RULE_INTERFACE: Final[str] = "DistilledDecisionRule@1"
POLICY_DISTILLER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/policy-distiller@1"
)
DISTILLATION_EXAMPLE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/distillation-example@1"
)
DISTILLATION_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/distillation-result@1"
)
DISTILLATION_GATE_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/distillation-gate-receipt@1"
)
RULE_APPLICATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/rule-application@1"
)
ADVERSARIAL_MUTATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/distillation-adversarial-mutation@1"
)

MIN_INDEPENDENT_DEVELOPMENT_EXAMPLES: Final[int] = 3
MIN_HELD_OUT_EXAMPLES: Final[int] = 1
MAX_CEGIS_ROUNDS: Final[int] = 8
MAX_AAE_MUTATIONS: Final[int] = 32
DEFAULT_RULE_VERSION: Final[str] = "shadow-v1"
NARROWED_RULE_VERSION: Final[str] = "shadow-narrowed-v1"

DECLARATIVE_WHEN_KEYS: Final[frozenset[str]] = frozenset(
    {
        "context_confidence",
        "failure_signature",
        "language",
        "proof_requirements",
        "provider_health",
        "repository_family",
        "required_capabilities",
        "risk_class",
        "task_class",
        "token_budget",
    }
)
SCOPE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "decision_class",
        "language",
        "repository_family",
        "risk_class",
        "task_class",
    }
)
_FORBIDDEN_FIELD_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "ast",
        "authorization",
        "bytecode",
        "callable",
        "chain_of_thought",
        "chat_messages",
        "code",
        "compile",
        "completion",
        "cookie",
        "credential",
        "decoded_source",
        "eval",
        "exec",
        "executable_code",
        "function",
        "hidden_reasoning",
        "import",
        "lambda",
        "messages",
        "model_output",
        "model_transcript",
        "module",
        "password",
        "private_key",
        "private_reasoning",
        "prompt",
        "python",
        "python_code",
        "raw_prompt",
        "refresh_token",
        "script",
        "secret",
        "shell_command",
        "source_body",
        "source_text",
        "transcript",
        "__import__",
    }
)
_EXECUTABLE_VALUE_MARKERS: Final[tuple[str, ...]] = (
    "import ",
    "from ",
    "def ",
    "class ",
    "lambda ",
    "exec(",
    "eval(",
    "compile(",
    "__import__",
    "os.system",
    "subprocess",
    "#!/",
)
_EXECUTABLE_VALUE_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "__import__",
        "compile",
        "eval",
        "exec",
        "lambda",
        "os.system",
        "subprocess",
    }
)
_PROPOSAL_EXECUTABLE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "ast",
        "body",
        "bytecode",
        "callable",
        "code",
        "executable",
        "executable_code",
        "function",
        "lambda",
        "module",
        "python",
        "python_code",
        "script",
        "shell_command",
        "source",
        "source_body",
    }
)


class PolicyDistillationError(ValueError):
    """Raised when distillation inputs themselves violate the closed contract."""


class DistillationPartition(str, Enum):
    """Closed example-split vocabulary."""

    DEVELOPMENT = "development"
    HELD_OUT = "held_out"
    COUNTEREXAMPLE = "counterexample"


class DistillationDisposition(str, Enum):
    """Closed outcome of one distillation attempt.  Never includes promotion."""

    SHADOW_CANDIDATE = "shadow_candidate"
    REJECTED = "rejected"
    DEVELOPMENT_FAILED = "development_failed"
    HELD_OUT_FAILED = "held_out_failed"
    ROLLED_BACK = "rolled_back"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class DistillationGate(str, Enum):
    """Closed gate vocabulary run before a shadow candidate may be emitted."""

    DSL = "dsl"
    DEVELOPMENT = "development"
    COUNTEREXAMPLE = "counterexample"
    HELD_OUT = "held_out"
    AAE = "aae"
    SHADOW = "shadow"
    PROMOTION_FORBIDDEN = "promotion_forbidden"


def _enum(value: Any, enum_type: type[Enum], name: str) -> Any:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value))
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise PolicyDistillationError(f"{name} must be one of: {allowed}") from exc


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise PolicyDistillationError(f"{name} must be a boolean")
    return value


def _identifier(value: Any, name: str, *, required: bool = True) -> str:
    if value is None:
        result = ""
    elif isinstance(value, str):
        result = value.strip()
    else:
        raise PolicyDistillationError(f"{name} must be a string")
    if not result:
        if required:
            raise PolicyDistillationError(f"{name} must be a compact bounded identifier")
        return ""
    if (
        len(result.encode("utf-8")) > MAX_IDENTIFIER_BYTES
        or any(char.isspace() for char in result)
        or "\x00" in result
    ):
        raise PolicyDistillationError(f"{name} must be a compact bounded identifier")
    return result


def _identifiers(values: Any, name: str, *, required: bool = False) -> tuple[str, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, str):
        raw = (values,)
    elif isinstance(values, Sequence) and not isinstance(values, (bytes, bytearray)):
        raw = values
    else:
        raise PolicyDistillationError(f"{name} must be a sequence of identifiers")
    if len(raw) > MAX_SEQUENCE_ITEMS:
        raise PolicyDistillationError(f"{name} contains too many items")
    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw:
        identifier = _identifier(item, name)
        if identifier not in seen:
            seen.add(identifier)
            normalized.append(identifier)
    if required and not normalized:
        raise PolicyDistillationError(f"{name} must not be empty")
    return tuple(sorted(normalized))


def _normalize_field_name(key: Any) -> str:
    if not isinstance(key, str):
        raise PolicyDistillationError("distillation field names must be strings")
    return key.strip().lower().replace("-", "_")


def field_is_forbidden(key: Any) -> bool:
    """Return whether a mapping key is secret, prompt, source, or executable policy."""

    normalized = _normalize_field_name(key)
    if not normalized:
        return False
    return any(
        normalized == marker or normalized.endswith("_" + marker)
        for marker in _FORBIDDEN_FIELD_MARKERS
    )


def _text_contains_executable(value: str) -> bool:
    lowered = value.lower()
    if lowered in _EXECUTABLE_VALUE_TOKENS:
        return True
    return any(marker in lowered for marker in _EXECUTABLE_VALUE_MARKERS)


def _reject_forbidden_payload(value: Any, name: str, *, depth: int = 0) -> None:
    if depth > MAX_NESTING_DEPTH:
        raise PolicyDistillationError(f"{name} exceeds maximum nesting")
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float):
        raise PolicyDistillationError(f"{name} cannot contain floats")
    if isinstance(value, str):
        if _text_contains_executable(value):
            raise PolicyDistillationError(f"{name} contains executable policy")
        return
    if isinstance(value, Mapping):
        if len(value) > MAX_MAPPING_ITEMS:
            raise PolicyDistillationError(f"{name} contains too many entries")
        for raw_key, nested in value.items():
            if field_is_forbidden(raw_key):
                raise PolicyDistillationError(
                    f"{name} contains forbidden private or executable data"
                )
            _reject_forbidden_payload(nested, name, depth=depth + 1)
        return
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        if len(value) > MAX_SEQUENCE_ITEMS:
            raise PolicyDistillationError(f"{name} contains too many items")
        for item in value:
            _reject_forbidden_payload(item, name, depth=depth + 1)
        return
    raise PolicyDistillationError(f"{name} contains unsupported value type {type(value).__name__}")


def _freeze_when_value(value: Any, name: str) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        if abs(value) > MAX_INTEGER:
            raise PolicyDistillationError(f"{name} integer is out of range")
        if value < 0:
            raise PolicyDistillationError(f"{name} must be a non-negative integer")
        return value
    if isinstance(value, str):
        if _text_contains_executable(value):
            raise PolicyDistillationError(f"{name} contains executable policy")
        return _identifier(value, name)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return _identifiers(value, name, required=True)
    raise PolicyDistillationError(
        f"{name} must be an identifier, identifier list, boolean, or non-negative integer"
    )


def _freeze_when(value: Any, name: str = "when") -> Mapping[str, Any]:
    if value is None:
        raw: Mapping[str, Any] = {}
    elif isinstance(value, Mapping):
        raw = value
    else:
        raise PolicyDistillationError(f"{name} must be a mapping")
    if len(raw) > MAX_MAPPING_ITEMS:
        raise PolicyDistillationError(f"{name} contains too many entries")
    _reject_forbidden_payload(raw, name)
    result: dict[str, Any] = {}
    for raw_key, raw_value in raw.items():
        key = _identifier(raw_key, name)
        if key not in DECLARATIVE_WHEN_KEYS:
            raise PolicyDistillationError(f"{name} uses unsupported declarative conditions")
        result[key] = _freeze_when_value(raw_value, name)
    return MappingProxyType(dict(sorted(result.items())))


def _freeze_scope(value: Any, decision_class: str) -> Mapping[str, Any]:
    if value is None:
        raw: Mapping[str, Any] = {}
    elif isinstance(value, Mapping):
        raw = value
    else:
        raise PolicyDistillationError("scope must be a mapping")
    _reject_forbidden_payload(raw, "scope")
    result: dict[str, Any] = {"decision_class": _identifier(decision_class, "decision_class")}
    for raw_key, raw_value in raw.items():
        key = _identifier(raw_key, "scope")
        if key not in SCOPE_KEYS:
            raise PolicyDistillationError("scope uses unsupported declarative conditions")
        if key == "decision_class":
            declared = _identifier(raw_value, "scope")
            if declared != result["decision_class"]:
                raise PolicyDistillationError("scope decision_class does not match the example class")
            continue
        result[key] = _freeze_when_value(raw_value, "scope")
    return MappingProxyType(dict(sorted(result.items())))


def _values_equal(left: Any, right: Any) -> bool:
    return left == right


def conditions_match(when: Mapping[str, Any], features: Mapping[str, Any]) -> bool:
    """Return whether every declarative condition is present and equal."""

    if not when:
        return False
    for key, expected in when.items():
        if key not in features:
            return False
        if not _values_equal(expected, features[key]):
            return False
    return True


def common_feature_conjunction(
    feature_maps: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Return equalities shared by every mapping; empty when none are stable."""

    if not feature_maps:
        return MappingProxyType({})
    shared = dict(feature_maps[0])
    for features in feature_maps[1:]:
        for key in list(shared):
            if key not in features or not _values_equal(shared[key], features[key]):
                shared.pop(key, None)
    return MappingProxyType(dict(sorted(shared.items())))


def rule_is_narrower_than_evidence(
    when: Mapping[str, Any],
    positive_features: Sequence[Mapping[str, Any]],
    negative_features: Sequence[Mapping[str, Any]] = (),
) -> bool:
    """Return whether ``when`` is no broader than the evidenced class.

    The rule must include every stable common equality, still cover every
    positive example, and must not fire on any negative example.
    """

    common = common_feature_conjunction(positive_features)
    if not common or not when:
        return False
    for key, value in common.items():
        if key not in when or not _values_equal(when[key], value):
            return False
    if not all(conditions_match(when, features) for features in positive_features):
        return False
    if any(conditions_match(when, features) for features in negative_features):
        return False
    return True


def _default_fallback(action: MetaAction) -> MetaAction:
    if action is MetaAction.CALL_LOCAL_SMALL_MODEL:
        return MetaAction.RUN_LOCAL_STATIC_ANALYSIS
    return MetaAction.CALL_LOCAL_SMALL_MODEL


def _coerce_episode(value: ExperienceEpisode | Mapping[str, Any] | None) -> ExperienceEpisode:
    if isinstance(value, ExperienceEpisode):
        return value
    if isinstance(value, Mapping):
        _reject_forbidden_payload(value, "episode")
        try:
            return ExperienceEpisode.from_dict(value)
        except AutonomyContractError as exc:
            raise PolicyDistillationError(str(exc)) from exc
    raise PolicyDistillationError("episode must be an ExperienceEpisode or mapping")


def _coerce_attribution(
    value: CausalAttribution | Mapping[str, Any] | None,
) -> CausalAttribution:
    if isinstance(value, CausalAttribution):
        return value
    if isinstance(value, Mapping):
        _reject_forbidden_payload(value, "attribution")
        try:
            return CausalAttribution.from_dict(value)
        except AutonomyContractError as exc:
            raise PolicyDistillationError(str(exc)) from exc
    raise PolicyDistillationError("attribution must be a CausalAttribution or mapping")


def _coerce_action(value: Any, name: str = "action") -> MetaAction:
    return _enum(value, MetaAction, name)


@dataclass(frozen=True)
class DistillationExample:
    """One typed, identity-only distillation example.  No source or prompts."""

    example_id: str
    decision_class: str
    features: Mapping[str, Any]
    action: MetaAction
    episode: ExperienceEpisode
    attribution: CausalAttribution
    partition: DistillationPartition
    independence_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "example_id", _identifier(self.example_id, "example_id"))
        object.__setattr__(
            self, "decision_class", _identifier(self.decision_class, "decision_class")
        )
        object.__setattr__(self, "features", _freeze_when(self.features, "features"))
        object.__setattr__(self, "action", _coerce_action(self.action, "action"))
        episode = self.episode if isinstance(self.episode, ExperienceEpisode) else _coerce_episode(
            self.episode
        )
        attribution = (
            self.attribution
            if isinstance(self.attribution, CausalAttribution)
            else _coerce_attribution(self.attribution)
        )
        object.__setattr__(self, "episode", episode)
        object.__setattr__(self, "attribution", attribution)
        object.__setattr__(
            self, "partition", _enum(self.partition, DistillationPartition, "partition")
        )
        independence = _identifier(self.independence_id, "independence_id", required=False)
        if not independence:
            independence = episode.frozen_input_ids[0] if episode.frozen_input_ids else episode.episode_id
        object.__setattr__(self, "independence_id", independence)
        if episode.selected_action is not self.action:
            raise PolicyDistillationError("example action must match the episode selected_action")
        if episode.episode_id not in attribution.episode_ids:
            raise PolicyDistillationError("causal attribution does not cover the example episode")
        encoded = canonical_json_bytes(self.to_dict())
        if len(encoded) > MAX_CANONICAL_RECORD_BYTES:
            raise PolicyDistillationError("distillation example exceeds its bounded size")

    @property
    def episode_id(self) -> str:
        return self.episode.episode_id

    @property
    def validation_receipt_ids(self) -> tuple[str, ...]:
        return self.episode.validation_receipt_ids

    @property
    def is_validated_positive(self) -> bool:
        return (
            self.episode.terminal_status is TerminalStatus.SUCCEEDED
            and bool(self.episode.validation_receipt_ids)
            and bool(self.episode.accepted_criterion_ids)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DISTILLATION_EXAMPLE_SCHEMA,
            "contract_version": CONTRACT_VERSION,
            "example_id": self.example_id,
            "decision_class": self.decision_class,
            "features": dict(self.features),
            "action": self.action.value,
            "episode": self.episode.to_dict(),
            "attribution": self.attribution.to_dict(),
            "partition": self.partition.value,
            "independence_id": self.independence_id,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any] | DistillationExample
    ) -> DistillationExample:
        if isinstance(payload, DistillationExample):
            return payload
        if not isinstance(payload, Mapping):
            raise PolicyDistillationError("distillation example must be an object")
        _reject_forbidden_payload(payload, "example")
        return cls(
            example_id=payload.get("example_id") or payload.get("id") or "",  # type: ignore[arg-type]
            decision_class=payload.get("decision_class") or "",  # type: ignore[arg-type]
            features=payload.get("features") or {},
            action=payload.get("action"),  # type: ignore[arg-type]
            episode=payload.get("episode"),  # type: ignore[arg-type]
            attribution=payload.get("attribution"),  # type: ignore[arg-type]
            partition=payload.get("partition") or DistillationPartition.DEVELOPMENT,  # type: ignore[arg-type]
            independence_id=payload.get("independence_id") or "",
        )


@dataclass(frozen=True)
class AdversarialMutation:
    """One AAE feature mutant.  Out-of-domain by default."""

    mutation_id: str
    features: Mapping[str, Any]
    expected_in_domain: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "mutation_id", _identifier(self.mutation_id, "mutation_id"))
        object.__setattr__(self, "features", _freeze_when(self.features, "features"))
        object.__setattr__(
            self, "expected_in_domain", _bool(self.expected_in_domain, "expected_in_domain")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": ADVERSARIAL_MUTATION_SCHEMA,
            "contract_version": CONTRACT_VERSION,
            "mutation_id": self.mutation_id,
            "features": dict(self.features),
            "expected_in_domain": self.expected_in_domain,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any] | AdversarialMutation
    ) -> AdversarialMutation:
        if isinstance(payload, AdversarialMutation):
            return payload
        if not isinstance(payload, Mapping):
            raise PolicyDistillationError("adversarial mutation must be an object")
        _reject_forbidden_payload(payload, "mutation")
        return cls(
            mutation_id=payload.get("mutation_id") or payload.get("id") or "",  # type: ignore[arg-type]
            features=payload.get("features") or {},
            expected_in_domain=payload.get("expected_in_domain", False),
        )


@dataclass(frozen=True)
class DistillationGateReceipt:
    """Content-addressed pass/fail record for one distillation gate."""

    gate: DistillationGate
    passed: bool
    reason_codes: tuple[str, ...]
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "gate", _enum(self.gate, DistillationGate, "gate"))
        object.__setattr__(self, "passed", _bool(self.passed, "passed"))
        object.__setattr__(
            self, "reason_codes", _identifiers(self.reason_codes, "reason_codes", required=True)
        )
        object.__setattr__(
            self, "evidence_ids", _identifiers(self.evidence_ids, "evidence_ids")
        )

    @property
    def receipt_id(self) -> str:
        payload = dict(self.to_dict())
        payload.pop("receipt_id", None)
        return content_identity(payload)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": DISTILLATION_GATE_RECEIPT_SCHEMA,
            "contract_version": CONTRACT_VERSION,
            "gate": self.gate.value,
            "passed": self.passed,
            "reason_codes": list(self.reason_codes),
            "evidence_ids": list(self.evidence_ids),
        }
        payload["receipt_id"] = content_identity(
            {key: value for key, value in payload.items() if key != "receipt_id"}
        )
        return payload


@dataclass(frozen=True)
class RuleApplication:
    """Result of applying a distilled rule, including out-of-domain fallback."""

    selected_action: MetaAction
    in_domain: bool
    used_fallback: bool
    reason_codes: tuple[str, ...]
    rule_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "selected_action", _coerce_action(self.selected_action, "selected_action")
        )
        object.__setattr__(self, "in_domain", _bool(self.in_domain, "in_domain"))
        object.__setattr__(self, "used_fallback", _bool(self.used_fallback, "used_fallback"))
        object.__setattr__(
            self, "reason_codes", _identifiers(self.reason_codes, "reason_codes", required=True)
        )
        object.__setattr__(self, "rule_id", _identifier(self.rule_id, "rule_id", required=False))
        if self.in_domain == self.used_fallback:
            raise PolicyDistillationError("in-domain application cannot also use the fallback")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": RULE_APPLICATION_SCHEMA,
            "contract_version": CONTRACT_VERSION,
            "selected_action": self.selected_action.value,
            "in_domain": self.in_domain,
            "used_fallback": self.used_fallback,
            "reason_codes": list(self.reason_codes),
            "rule_id": self.rule_id,
        }


@dataclass(frozen=True)
class DistillationResult:
    """Typed distillation outcome.  Absence of a rule is an ordinary result."""

    disposition: DistillationDisposition
    reason_codes: tuple[str, ...]
    decision_class: str
    candidate: DistillationCandidate | None = None
    rule: DistilledDecisionRule | None = None
    gate_receipts: tuple[DistillationGateReceipt, ...] = ()
    rolled_back_rule_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, DistillationDisposition, "disposition"),
        )
        object.__setattr__(
            self, "reason_codes", _identifiers(self.reason_codes, "reason_codes", required=True)
        )
        object.__setattr__(
            self, "decision_class", _identifier(self.decision_class, "decision_class")
        )
        if self.candidate is not None and not isinstance(self.candidate, DistillationCandidate):
            raise PolicyDistillationError("candidate must be a DistillationCandidate")
        if self.rule is not None and not isinstance(self.rule, DistilledDecisionRule):
            raise PolicyDistillationError("rule must be a DistilledDecisionRule")
        if isinstance(self.gate_receipts, DistillationGateReceipt):
            receipts: tuple[DistillationGateReceipt, ...] = (self.gate_receipts,)
        else:
            receipts = tuple(self.gate_receipts)
        if len(receipts) > MAX_SEQUENCE_ITEMS:
            raise PolicyDistillationError("too many gate receipts")
        if any(not isinstance(item, DistillationGateReceipt) for item in receipts):
            raise PolicyDistillationError("gate receipts must be DistillationGateReceipt values")
        object.__setattr__(self, "gate_receipts", receipts)
        object.__setattr__(
            self,
            "rolled_back_rule_id",
            _identifier(self.rolled_back_rule_id, "rolled_back_rule_id", required=False),
        )
        if self.rule is not None:
            if not self.rule.shadow_only:
                raise PolicyDistillationError("distilled rules cannot leave shadow mode")
            if self.rule.authorized_promotion_id:
                raise PolicyDistillationError("a distilled rule cannot authorize its own promotion")
        if self.candidate is not None and self.candidate.status is DistillationStatus.PROMOTED:
            raise PolicyDistillationError("the distiller cannot emit a promoted candidate")
        if (
            self.disposition is DistillationDisposition.SHADOW_CANDIDATE
            and (self.rule is None or self.candidate is None)
        ):
            raise PolicyDistillationError("shadow candidates require a rule and candidate record")

    @property
    def result_id(self) -> str:
        payload = dict(self.to_dict())
        payload.pop("result_id", None)
        return content_identity(payload)

    @property
    def passed(self) -> bool:
        return self.disposition is DistillationDisposition.SHADOW_CANDIDATE

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": DISTILLATION_RESULT_SCHEMA,
            "contract_version": CONTRACT_VERSION,
            "disposition": self.disposition.value,
            "reason_codes": list(self.reason_codes),
            "decision_class": self.decision_class,
            "candidate": None if self.candidate is None else self.candidate.to_dict(),
            "rule": None if self.rule is None else self.rule.to_dict(),
            "gate_receipts": [item.to_dict() for item in self.gate_receipts],
            "rolled_back_rule_id": self.rolled_back_rule_id,
            "shadow_only": True,
            "self_promoted": False,
        }
        payload["result_id"] = content_identity(
            {key: value for key, value in payload.items() if key != "result_id"}
        )
        return payload


@dataclass(frozen=True)
class StableDecisionClass:
    """A decision class that currently meets the independent-example bar."""

    decision_class: str
    action: MetaAction
    independent_example_count: int
    common_features: Mapping[str, Any]
    example_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_class": self.decision_class,
            "action": self.action.value,
            "independent_example_count": self.independent_example_count,
            "common_features": dict(self.common_features),
            "example_ids": list(self.example_ids),
        }


def _gate(
    gate: DistillationGate,
    passed: bool,
    *reason_codes: str,
    evidence_ids: Sequence[str] = (),
) -> DistillationGateReceipt:
    return DistillationGateReceipt(
        gate=gate,
        passed=passed,
        reason_codes=reason_codes,
        evidence_ids=tuple(evidence_ids),
    )


def _result(
    disposition: DistillationDisposition,
    decision_class: str,
    *reason_codes: str,
    candidate: DistillationCandidate | None = None,
    rule: DistilledDecisionRule | None = None,
    gate_receipts: DistillationGateReceipt | Sequence[DistillationGateReceipt] = (),
    rolled_back_rule_id: str = "",
) -> DistillationResult:
    if isinstance(gate_receipts, DistillationGateReceipt):
        receipts: tuple[DistillationGateReceipt, ...] = (gate_receipts,)
    else:
        receipts = tuple(gate_receipts)
    return DistillationResult(
        disposition=disposition,
        reason_codes=reason_codes,
        decision_class=decision_class,
        candidate=candidate,
        rule=rule,
        gate_receipts=receipts,
        rolled_back_rule_id=rolled_back_rule_id,
    )


def _proposal_looks_executable(payload: Mapping[str, Any]) -> bool:
    kind = payload.get("kind") or payload.get("type") or payload.get("language")
    if isinstance(kind, str) and _normalize_field_name(kind) in {
        "python",
        "py",
        "executable",
        "code",
        "shell",
        "bash",
        "lambda",
    }:
        return True
    return any(field_is_forbidden(key) or _normalize_field_name(key) in _PROPOSAL_EXECUTABLE_KEYS for key in payload)


def _mutate_value(key: str, value: Any) -> Any:
    if isinstance(value, bool):
        return not value
    if isinstance(value, int) and not isinstance(value, bool):
        return value + 1
    if isinstance(value, tuple):
        return tuple(sorted({"aae-mutant-" + key, *value}))
    return "aae-mutant-" + key


def generate_aae_mutations(when: Mapping[str, Any]) -> tuple[AdversarialMutation, ...]:
    """Flip each constrained feature so the rule must fall back out of domain."""

    mutants: list[AdversarialMutation] = []
    for index, (key, value) in enumerate(sorted(when.items())):
        if len(mutants) >= MAX_AAE_MUTATIONS:
            break
        mutated = dict(when)
        mutated[key] = _mutate_value(key, value)
        mutants.append(
            AdversarialMutation(
                mutation_id=f"aae-{index}-{key}",
                features=mutated,
                expected_in_domain=False,
            )
        )
    return tuple(mutants)


def _distinguishing_additions(
    positives: Sequence[DistillationExample],
    counterexample: DistillationExample,
    current: Mapping[str, Any],
) -> Mapping[str, Any]:
    common = common_feature_conjunction(tuple(item.features for item in positives))
    additions: dict[str, Any] = {}
    for key, value in common.items():
        if key in current:
            continue
        observed = counterexample.features.get(key)
        if observed is None or not _values_equal(observed, value):
            additions[key] = value
    return MappingProxyType(dict(sorted(additions.items())))


def _narrow_when(
    initial: Mapping[str, Any],
    positives: Sequence[DistillationExample],
    counterexamples: Sequence[DistillationExample],
) -> tuple[Mapping[str, Any] | None, tuple[str, ...]]:
    common = common_feature_conjunction(tuple(item.features for item in positives))
    if not common:
        return None, ("unstable_features", "empty_common_conjunction")
    when = dict(common)
    if initial:
        for key, value in initial.items():
            if key not in common:
                return None, ("ungrounded_condition", key)
            if not _values_equal(value, common[key]):
                return None, ("proposal_conflicts_with_evidence", key)
        # Keep every common equality even if the proposal omitted some.
        when = dict(common)
    reasons: list[str] = []
    if initial and set(initial) != set(common):
        reasons.append("counterexample_narrowed")
        reasons.append("proposal_was_broader_than_evidence")
    for round_index in range(MAX_CEGIS_ROUNDS + 1):
        matching_negatives = tuple(
            item
            for item in counterexamples
            if conditions_match(when, item.features) and item.action is not positives[0].action
        )
        if not matching_negatives:
            frozen = MappingProxyType(dict(sorted(when.items())))
            extra = ("cegis_fixed_point",) if round_index else ()
            return frozen, tuple(dict.fromkeys((*reasons, *extra)))
        if round_index == MAX_CEGIS_ROUNDS:
            return None, ("counterexample_not_narrowable", "cegis_round_exhausted")
        added = False
        for negative in matching_negatives:
            extra = _distinguishing_additions(positives, negative, when)
            if not extra:
                return None, ("counterexample_not_distinguishable", negative.example_id)
            when.update(extra)
            added = True
            reasons.append("counterexample_narrowed")
        if not added:
            return None, ("counterexample_not_narrowable",)
    return None, ("counterexample_not_narrowable",)


class PolicyDistiller:
    """Detect stable decision classes and emit shadow-only declarative rules."""

    INTERFACE = POLICY_DISTILLER_INTERFACE
    RULE_INTERFACE = DISTILLED_DECISION_RULE_INTERFACE
    SCHEMA = POLICY_DISTILLER_SCHEMA

    def detect_stable_classes(
        self,
        examples: Sequence[DistillationExample | Mapping[str, Any]],
    ) -> tuple[StableDecisionClass, ...]:
        bound = self._bind_examples(examples)
        reports: list[StableDecisionClass] = []
        for decision_class in sorted({item.decision_class for item in bound}):
            development = tuple(
                item
                for item in bound
                if item.decision_class == decision_class
                and item.partition is DistillationPartition.DEVELOPMENT
            )
            report = self._class_stability(decision_class, development)
            if report is not None:
                reports.append(report)
        return tuple(reports)

    def distill(
        self,
        examples: Sequence[DistillationExample | Mapping[str, Any]],
        *,
        decision_class: str | None = None,
        proposed_rule: Mapping[str, Any] | DistilledDecisionRule | None = None,
        fallback: MetaAction | str | None = None,
        current_shadow_rule: DistilledDecisionRule | None = None,
        aae_mutations: Sequence[AdversarialMutation | Mapping[str, Any]] = (),
    ) -> DistillationResult:
        """Run development / CEGIS / held-out / AAE / shadow gates.

        A passing result is always ``shadow_candidate``.  This method cannot
        promote, authorize promotion, or emit executable policy.
        """

        bound = self._bind_examples(examples)
        if not bound:
            raise PolicyDistillationError("examples must not be empty")
        class_id = (
            _identifier(decision_class, "decision_class")
            if decision_class
            else bound[0].decision_class
        )
        if any(item.decision_class != class_id for item in bound):
            raise PolicyDistillationError("distill accepts one decision_class per call")
        rollback_id = ""
        if current_shadow_rule is not None:
            self._reject_self_promotion(current_shadow_rule)
            rollback_id = current_shadow_rule.rule_id

        proposal_gate, proposed_when, proposed_action, proposed_fallback, proposed_scope = (
            self._admit_proposal(proposed_rule, class_id)
        )
        if not proposal_gate.passed:
            return self._failed(
                DistillationDisposition.REJECTED,
                class_id,
                proposal_gate.reason_codes,
                (proposal_gate,),
                rollback_id,
                bound,
            )

        development = tuple(
            item for item in bound if item.partition is DistillationPartition.DEVELOPMENT
        )
        held_out = tuple(item for item in bound if item.partition is DistillationPartition.HELD_OUT)
        counterexamples = tuple(
            item for item in bound if item.partition is DistillationPartition.COUNTEREXAMPLE
        )
        precheck = self._preconditions(class_id, development, held_out)
        if precheck is not None:
            return self._failed(
                precheck[0],
                class_id,
                precheck[1],
                (proposal_gate, *precheck[2]),
                rollback_id,
                bound,
            )

        stability = self._class_stability(class_id, development)
        if stability is None:
            reasons = self._instability_reasons(development)
            gate = _gate(DistillationGate.DEVELOPMENT, False, *reasons)
            disposition = (
                DistillationDisposition.INSUFFICIENT_EVIDENCE
                if "independent_example_threshold" in reasons
                or "unstable_features" in reasons
                else DistillationDisposition.DEVELOPMENT_FAILED
            )
            return self._failed(disposition, class_id, reasons, (proposal_gate, gate), rollback_id, bound)

        if proposed_action is not None and proposed_action is not stability.action:
            gate = _gate(
                DistillationGate.DEVELOPMENT,
                False,
                "proposal_action_conflicts_with_evidence",
            )
            return self._failed(
                DistillationDisposition.DEVELOPMENT_FAILED,
                class_id,
                gate.reason_codes,
                (proposal_gate, gate),
                rollback_id,
                bound,
            )

        narrowed, narrow_reasons = _narrow_when(proposed_when, development, counterexamples)
        if narrowed is None:
            gate = _gate(DistillationGate.COUNTEREXAMPLE, False, *narrow_reasons)
            return self._failed(
                DistillationDisposition.REJECTED,
                class_id,
                narrow_reasons,
                (proposal_gate, gate),
                rollback_id,
                bound,
            )
        if not rule_is_narrower_than_evidence(
            narrowed,
            tuple(item.features for item in development),
            tuple(item.features for item in counterexamples if item.action is not stability.action),
        ):
            gate = _gate(
                DistillationGate.COUNTEREXAMPLE,
                False,
                "rule_broader_than_evidence",
                *narrow_reasons,
            )
            return self._failed(
                DistillationDisposition.REJECTED,
                class_id,
                gate.reason_codes,
                (proposal_gate, gate),
                rollback_id,
                bound,
            )
        counterexample_gate = _gate(
            DistillationGate.COUNTEREXAMPLE,
            True,
            *(narrow_reasons or ("no_matching_counterexample",)),
            evidence_ids=tuple(item.example_id for item in counterexamples),
        )

        development_fail = tuple(
            item for item in development if not conditions_match(narrowed, item.features)
        )
        if development_fail:
            gate = _gate(
                DistillationGate.DEVELOPMENT,
                False,
                "development_coverage_failed",
                evidence_ids=tuple(item.example_id for item in development_fail),
            )
            return self._failed(
                DistillationDisposition.DEVELOPMENT_FAILED,
                class_id,
                gate.reason_codes,
                (proposal_gate, counterexample_gate, gate),
                rollback_id,
                bound,
            )
        mixed = tuple(item for item in development if item.action is not stability.action)
        if mixed:
            gate = _gate(
                DistillationGate.DEVELOPMENT,
                False,
                "unstable_output",
                evidence_ids=tuple(item.example_id for item in mixed),
            )
            return self._failed(
                DistillationDisposition.DEVELOPMENT_FAILED,
                class_id,
                gate.reason_codes,
                (proposal_gate, counterexample_gate, gate),
                rollback_id,
                bound,
            )
        development_gate = _gate(
            DistillationGate.DEVELOPMENT,
            True,
            "development_covered",
            "stable_output",
            evidence_ids=tuple(item.example_id for item in development),
        )

        held_conflicts = tuple(
            item
            for item in held_out
            if conditions_match(narrowed, item.features) and item.action is not stability.action
        )
        held_uncovered = tuple(
            item
            for item in held_out
            if item.action is stability.action and not conditions_match(narrowed, item.features)
        )
        if held_conflicts or held_uncovered:
            gate = _gate(
                DistillationGate.HELD_OUT,
                False,
                "held_out_failed",
                *(("held_out_action_conflict",) if held_conflicts else ()),
                *(("held_out_uncovered",) if held_uncovered else ()),
                evidence_ids=tuple(item.example_id for item in (*held_conflicts, *held_uncovered)),
            )
            rule = self._build_rule(
                class_id,
                narrowed,
                stability.action,
                proposed_fallback if proposed_fallback is not None else _coerce_optional_fallback(fallback, stability.action),
                proposed_scope,
                development,
                held_out,
                counterexamples,
                version=NARROWED_RULE_VERSION if "counterexample_narrowed" in narrow_reasons else DEFAULT_RULE_VERSION,
            )
            candidate = self._candidate(
                class_id,
                development,
                held_out,
                counterexamples,
                rule,
                DistillationStatus.HELD_OUT_FAILED,
            )
            return _result(
                DistillationDisposition.HELD_OUT_FAILED,
                class_id,
                *gate.reason_codes,
                candidate=candidate,
                rule=rule,
                gate_receipts=(proposal_gate, counterexample_gate, development_gate, gate),
                rolled_back_rule_id=rollback_id,
            )
        held_out_gate = _gate(
            DistillationGate.HELD_OUT,
            True,
            "held_out_passed",
            evidence_ids=tuple(item.example_id for item in held_out),
        )

        bound_fallback = (
            proposed_fallback
            if proposed_fallback is not None
            else _coerce_optional_fallback(fallback, stability.action)
        )
        if bound_fallback is stability.action:
            gate = _gate(DistillationGate.DSL, False, "fallback_must_differ_from_action")
            return self._failed(
                DistillationDisposition.REJECTED,
                class_id,
                gate.reason_codes,
                (proposal_gate, counterexample_gate, development_gate, held_out_gate, gate),
                rollback_id,
                bound,
            )

        scope = dict(proposed_scope)
        for key in ("repository_family", "language", "risk_class", "task_class"):
            if key in narrowed and key not in scope:
                scope[key] = narrowed[key]
        frozen_scope = _freeze_scope(scope, class_id)

        rule = self._build_rule(
            class_id,
            narrowed,
            stability.action,
            bound_fallback,
            frozen_scope,
            development,
            held_out,
            counterexamples,
            version=NARROWED_RULE_VERSION if "counterexample_narrowed" in narrow_reasons else DEFAULT_RULE_VERSION,
        )
        application_ok, aae_reasons, aae_ids = self._aae_gate(rule, aae_mutations)
        if not application_ok:
            gate = _gate(DistillationGate.AAE, False, *aae_reasons, evidence_ids=aae_ids)
            candidate = self._candidate(
                class_id,
                development,
                held_out,
                counterexamples,
                rule,
                DistillationStatus.REJECTED,
            )
            return _result(
                DistillationDisposition.REJECTED,
                class_id,
                *gate.reason_codes,
                candidate=candidate,
                rule=rule,
                gate_receipts=(
                    proposal_gate,
                    counterexample_gate,
                    development_gate,
                    held_out_gate,
                    gate,
                ),
                rolled_back_rule_id=rollback_id,
            )
        aae_gate = _gate(DistillationGate.AAE, True, *aae_reasons, evidence_ids=aae_ids)
        shadow_gate = _gate(
            DistillationGate.SHADOW,
            True,
            "shadow_only",
            "cannot_self_promote",
        )
        promotion_gate = _gate(
            DistillationGate.PROMOTION_FORBIDDEN,
            True,
            "promotion_owned_by_external_authority",
        )
        candidate = self._candidate(
            class_id,
            development,
            held_out,
            counterexamples,
            rule,
            DistillationStatus.SHADOW,
        )
        return _result(
            DistillationDisposition.SHADOW_CANDIDATE,
            class_id,
            "shadow_only",
            "narrower_than_evidence",
            "out_of_domain_fallback",
            "cannot_self_promote",
            candidate=candidate,
            rule=rule,
            gate_receipts=(
                proposal_gate,
                counterexample_gate,
                development_gate,
                held_out_gate,
                aae_gate,
                shadow_gate,
                promotion_gate,
            ),
            rolled_back_rule_id=rollback_id,
        )

    def apply(
        self,
        rule: DistilledDecisionRule | Mapping[str, Any],
        features: Mapping[str, Any],
    ) -> RuleApplication:
        """Apply a distilled rule, using fallback when features are out of domain."""

        bound_rule = self._bind_rule(rule)
        bound_features = _freeze_when(features, "features")
        in_domain = conditions_match(bound_rule.when, bound_features) and _scope_matches(
            bound_rule.scope, bound_features
        )
        if in_domain:
            return RuleApplication(
                selected_action=bound_rule.action,
                in_domain=True,
                used_fallback=False,
                reason_codes=("in_domain",),
                rule_id=bound_rule.rule_id,
            )
        return RuleApplication(
            selected_action=bound_rule.fallback,
            in_domain=False,
            used_fallback=True,
            reason_codes=("out_of_domain", "fallback"),
            rule_id=bound_rule.rule_id,
        )

    def rollback(
        self,
        rule: DistilledDecisionRule | Mapping[str, Any],
        *,
        reason_codes: Sequence[str] = ("rollback_requested",),
        development_example_ids: Sequence[str] = (),
        held_out_example_ids: Sequence[str] = (),
        decision_class: str = "rolled-back-class",
    ) -> DistillationResult:
        """Retain the rule identity as history and mark the candidate rolled back."""

        bound = self._bind_rule(rule)
        class_id = _identifier(
            bound.scope.get("decision_class") or decision_class, "decision_class"
        )
        development_ids = _identifiers(
            development_example_ids or bound.source_episode_ids,
            "development_example_ids",
            required=True,
        )
        held_ids = _identifiers(
            held_out_example_ids or bound.held_out_evaluation_ids,
            "held_out_example_ids",
            required=True,
        )
        candidate = DistillationCandidate(
            decision_class=class_id,
            episode_ids=bound.source_episode_ids,
            input_feature_names=tuple(sorted(bound.when)),
            output_actions=(bound.action,),
            development_example_ids=development_ids,
            held_out_example_ids=held_ids,
            proposed_rule_id=bound.rule_id,
            status=DistillationStatus.ROLLED_BACK,
            counterexample_ids=bound.counterexample_ids,
        )
        return _result(
            DistillationDisposition.ROLLED_BACK,
            class_id,
            *tuple(reason_codes) or ("rollback_requested",),
            "history_retained",
            "cannot_reactivate",
            candidate=candidate,
            rule=bound,
            gate_receipts=(
                _gate(DistillationGate.SHADOW, False, "rolled_back"),
                _gate(
                    DistillationGate.PROMOTION_FORBIDDEN,
                    True,
                    "cannot_self_promote",
                    "cannot_reactivate",
                ),
            ),
            rolled_back_rule_id=bound.rule_id,
        )

    def promote(
        self,
        rule: DistilledDecisionRule | Mapping[str, Any] | None = None,
        authorization_id: str = "",
    ) -> DistillationResult:
        """Refuse promotion.  APMC-019 owns expected-old CAS promotion."""

        del authorization_id
        class_id = "unpromoted"
        if rule is not None:
            bound = self._bind_rule(rule)
            class_id = _identifier(
                bound.scope.get("decision_class") or "unpromoted", "decision_class"
            )
        return _result(
            DistillationDisposition.REJECTED,
            class_id,
            "cannot_self_promote",
            "promotion_owned_by_external_authority",
            gate_receipts=(
                _gate(
                    DistillationGate.PROMOTION_FORBIDDEN,
                    False,
                    "cannot_self_promote",
                    "promotion_owned_by_external_authority",
                ),
            ),
        )

    def _bind_examples(
        self, examples: Sequence[DistillationExample | Mapping[str, Any]]
    ) -> tuple[DistillationExample, ...]:
        if examples is None:
            raw: Sequence[Any] = ()
        elif isinstance(examples, Sequence) and not isinstance(examples, (str, bytes, bytearray)):
            raw = examples
        else:
            raise PolicyDistillationError("examples must be a sequence")
        if len(raw) > MAX_SEQUENCE_ITEMS:
            raise PolicyDistillationError("examples contains too many items")
        bound = tuple(DistillationExample.from_dict(item) for item in raw)
        seen: set[str] = set()
        for item in bound:
            if item.example_id in seen:
                raise PolicyDistillationError("example_id values must be unique")
            seen.add(item.example_id)
        return bound

    def _bind_rule(self, rule: DistilledDecisionRule | Mapping[str, Any]) -> DistilledDecisionRule:
        if isinstance(rule, DistilledDecisionRule):
            bound = rule
        elif isinstance(rule, Mapping):
            _reject_forbidden_payload(rule, "rule")
            try:
                bound = DistilledDecisionRule.from_dict(rule)
            except AutonomyContractError as exc:
                raise PolicyDistillationError(str(exc)) from exc
        else:
            raise PolicyDistillationError("rule must be a DistilledDecisionRule or mapping")
        self._reject_self_promotion(bound)
        if not bound.when or not set(bound.when).issubset(DECLARATIVE_WHEN_KEYS):
            raise PolicyDistillationError("distilled rule uses unsupported declarative conditions")
        if bound.action is bound.fallback:
            raise PolicyDistillationError("fallback must differ from the distilled action")
        return bound

    def _reject_self_promotion(self, rule: DistilledDecisionRule) -> None:
        if not rule.shadow_only or rule.authorized_promotion_id:
            raise PolicyDistillationError("a distilled rule cannot authorize its own promotion")

    def _admit_proposal(
        self,
        proposed_rule: Mapping[str, Any] | DistilledDecisionRule | None,
        decision_class: str,
    ) -> tuple[
        DistillationGateReceipt,
        Mapping[str, Any],
        MetaAction | None,
        MetaAction | None,
        Mapping[str, Any],
    ]:
        if proposed_rule is None:
            return (
                _gate(DistillationGate.DSL, True, "no_proposal", "declarative_dsl_only"),
                MappingProxyType({}),
                None,
                None,
                MappingProxyType({"decision_class": decision_class}),
            )
        if isinstance(proposed_rule, DistilledDecisionRule):
            try:
                self._reject_self_promotion(proposed_rule)
            except PolicyDistillationError:
                return (
                    _gate(
                        DistillationGate.PROMOTION_FORBIDDEN,
                        False,
                        "cannot_self_promote",
                        "proposal_attempted_promotion",
                    ),
                    MappingProxyType({}),
                    None,
                    None,
                    MappingProxyType({}),
                )
            try:
                when = _freeze_when(proposed_rule.when, "when")
                scope = _freeze_scope(proposed_rule.scope, decision_class)
            except PolicyDistillationError:
                return (
                    _gate(DistillationGate.DSL, False, "dsl_rejected", "executable_policy_rejected"),
                    MappingProxyType({}),
                    None,
                    None,
                    MappingProxyType({}),
                )
            return (
                _gate(DistillationGate.DSL, True, "declarative_dsl_only"),
                when,
                proposed_rule.action,
                proposed_rule.fallback,
                scope,
            )
        if not isinstance(proposed_rule, Mapping):
            raise PolicyDistillationError("proposed_rule must be a mapping or DistilledDecisionRule")
        try:
            _reject_forbidden_payload(proposed_rule, "proposed_rule")
        except PolicyDistillationError:
            return (
                _gate(DistillationGate.DSL, False, "dsl_rejected", "executable_policy_rejected"),
                MappingProxyType({}),
                None,
                None,
                MappingProxyType({}),
            )
        if _proposal_looks_executable(proposed_rule):
            return (
                _gate(DistillationGate.DSL, False, "dsl_rejected", "executable_policy_rejected"),
                MappingProxyType({}),
                None,
                None,
                MappingProxyType({}),
            )
        raw_when = proposed_rule.get("when", proposed_rule.get("conditions"))
        if raw_when is None and "action" not in proposed_rule:
            raw_when = {key: value for key, value in proposed_rule.items() if key in DECLARATIVE_WHEN_KEYS}
        try:
            when = _freeze_when(raw_when or {}, "when")
        except PolicyDistillationError:
            return (
                _gate(DistillationGate.DSL, False, "dsl_rejected", "unsupported_declarative_conditions"),
                MappingProxyType({}),
                None,
                None,
                MappingProxyType({}),
            )
        action = proposed_rule.get("action")
        bound_action = None if action is None else _coerce_action(action, "action")
        raw_fallback = proposed_rule.get("fallback")
        bound_fallback = (
            None if raw_fallback is None else _coerce_action(raw_fallback, "fallback")
        )
        try:
            scope = _freeze_scope(proposed_rule.get("scope") or {}, decision_class)
        except PolicyDistillationError:
            return (
                _gate(DistillationGate.DSL, False, "dsl_rejected", "unsupported_scope"),
                MappingProxyType({}),
                None,
                None,
                MappingProxyType({}),
            )
        return (
            _gate(DistillationGate.DSL, True, "declarative_dsl_only"),
            when,
            bound_action,
            bound_fallback,
            scope,
        )

    def _preconditions(
        self,
        decision_class: str,
        development: Sequence[DistillationExample],
        held_out: Sequence[DistillationExample],
    ) -> tuple[DistillationDisposition, tuple[str, ...], tuple[DistillationGateReceipt, ...]] | None:
        del decision_class
        if len(development) < MIN_INDEPENDENT_DEVELOPMENT_EXAMPLES:
            reasons = ("independent_example_threshold", "insufficient_development_examples")
            return (
                DistillationDisposition.INSUFFICIENT_EVIDENCE,
                reasons,
                (_gate(DistillationGate.DEVELOPMENT, False, *reasons),),
            )
        independent = {item.independence_id for item in development}
        if len(independent) < MIN_INDEPENDENT_DEVELOPMENT_EXAMPLES:
            reasons = ("independent_example_threshold",)
            return (
                DistillationDisposition.INSUFFICIENT_EVIDENCE,
                reasons,
                (_gate(DistillationGate.DEVELOPMENT, False, *reasons),),
            )
        if len(held_out) < MIN_HELD_OUT_EXAMPLES:
            reasons = ("missing_held_out_partition",)
            return (
                DistillationDisposition.INSUFFICIENT_EVIDENCE,
                reasons,
                (_gate(DistillationGate.HELD_OUT, False, *reasons),),
            )
        overlap = {item.example_id for item in development} & {item.example_id for item in held_out}
        independence_overlap = {item.independence_id for item in development} & {
            item.independence_id for item in held_out
        }
        if overlap or independence_overlap:
            reasons = ("held_out_partition_overlap",)
            return (
                DistillationDisposition.HELD_OUT_FAILED,
                reasons,
                (_gate(DistillationGate.HELD_OUT, False, *reasons),),
            )
        missing_validation = tuple(
            item.example_id for item in (*development, *held_out) if not item.is_validated_positive
        )
        if missing_validation:
            reasons = ("missing_validation_receipt",)
            return (
                DistillationDisposition.INSUFFICIENT_EVIDENCE,
                reasons,
                (
                    _gate(
                        DistillationGate.DEVELOPMENT,
                        False,
                        *reasons,
                        evidence_ids=missing_validation,
                    ),
                ),
            )
        return None

    def _class_stability(
        self,
        decision_class: str,
        development: Sequence[DistillationExample],
    ) -> StableDecisionClass | None:
        if len({item.independence_id for item in development}) < MIN_INDEPENDENT_DEVELOPMENT_EXAMPLES:
            return None
        actions = {item.action for item in development}
        if len(actions) != 1:
            return None
        common = common_feature_conjunction(tuple(item.features for item in development))
        if not common:
            return None
        action = next(iter(actions))
        return StableDecisionClass(
            decision_class=decision_class,
            action=action,
            independent_example_count=len({item.independence_id for item in development}),
            common_features=common,
            example_ids=tuple(item.example_id for item in development),
        )

    def _instability_reasons(self, development: Sequence[DistillationExample]) -> tuple[str, ...]:
        if len({item.independence_id for item in development}) < MIN_INDEPENDENT_DEVELOPMENT_EXAMPLES:
            return ("independent_example_threshold",)
        if len({item.action for item in development}) != 1:
            return ("unstable_output",)
        if not common_feature_conjunction(tuple(item.features for item in development)):
            return ("unstable_features", "empty_common_conjunction")
        return ("unstable_decision_class",)

    def _build_rule(
        self,
        decision_class: str,
        when: Mapping[str, Any],
        action: MetaAction,
        fallback: MetaAction,
        scope: Mapping[str, Any],
        development: Sequence[DistillationExample],
        held_out: Sequence[DistillationExample],
        counterexamples: Sequence[DistillationExample],
        *,
        version: str,
    ) -> DistilledDecisionRule:
        validation_ids = _identifiers(
            tuple(
                receipt
                for item in (*development, *held_out)
                for receipt in item.validation_receipt_ids
            ),
            "required_validation_ids",
            required=True,
        )
        try:
            rule = DistilledDecisionRule(
                version=version,
                when=dict(when),
                action=action,
                required_validation_ids=validation_ids,
                fallback=fallback,
                scope=dict(_freeze_scope(scope, decision_class)),
                source_episode_ids=tuple(item.episode_id for item in development),
                held_out_evaluation_ids=tuple(item.example_id for item in held_out),
                counterexample_ids=tuple(item.example_id for item in counterexamples),
                shadow_only=True,
                authorized_promotion_id="",
            )
        except AutonomyContractError as exc:
            raise PolicyDistillationError(str(exc)) from exc
        if not rule.shadow_only or rule.authorized_promotion_id:
            raise PolicyDistillationError("a distilled rule cannot authorize its own promotion")
        if rule.action is rule.fallback:
            raise PolicyDistillationError("fallback must differ from the distilled action")
        return rule

    def _candidate(
        self,
        decision_class: str,
        development: Sequence[DistillationExample],
        held_out: Sequence[DistillationExample],
        counterexamples: Sequence[DistillationExample],
        rule: DistilledDecisionRule,
        status: DistillationStatus,
    ) -> DistillationCandidate:
        if status is DistillationStatus.PROMOTED:
            raise PolicyDistillationError("the distiller cannot emit a promoted candidate")
        return DistillationCandidate(
            decision_class=decision_class,
            episode_ids=tuple(item.episode_id for item in (*development, *held_out, *counterexamples)),
            input_feature_names=tuple(sorted(rule.when)),
            output_actions=(rule.action,),
            development_example_ids=tuple(item.example_id for item in development),
            held_out_example_ids=tuple(item.example_id for item in held_out),
            proposed_rule_id=rule.rule_id,
            status=status,
            counterexample_ids=tuple(item.example_id for item in counterexamples),
        )

    def _aae_gate(
        self,
        rule: DistilledDecisionRule,
        supplied: Sequence[AdversarialMutation | Mapping[str, Any]],
    ) -> tuple[bool, tuple[str, ...], tuple[str, ...]]:
        generated = generate_aae_mutations(rule.when)
        extra = tuple(AdversarialMutation.from_dict(item) for item in supplied)
        if len(extra) > MAX_AAE_MUTATIONS:
            raise PolicyDistillationError("too many adversarial mutations")
        mutants = generated + extra
        failed: list[str] = []
        for mutant in mutants:
            application = self.apply(rule, mutant.features)
            if mutant.expected_in_domain:
                if not application.in_domain or application.selected_action is not rule.action:
                    failed.append(mutant.mutation_id)
            elif application.in_domain or application.selected_action is rule.action:
                failed.append(mutant.mutation_id)
        if failed:
            return False, ("aae_gate_failed", "mutation_remained_in_domain"), tuple(sorted(failed))
        return True, ("aae_mutations_out_of_domain", "fallback_retained"), tuple(
            item.mutation_id for item in mutants
        )

    def _failed(
        self,
        disposition: DistillationDisposition,
        decision_class: str,
        reasons: Sequence[str],
        gates: Sequence[DistillationGateReceipt],
        rollback_id: str,
        examples: Sequence[DistillationExample],
    ) -> DistillationResult:
        del examples
        if rollback_id and disposition is not DistillationDisposition.SHADOW_CANDIDATE:
            extra = ("rolled_back",) if "rolled_back" not in reasons else ()
            disposition = (
                DistillationDisposition.ROLLED_BACK
                if disposition
                in {
                    DistillationDisposition.REJECTED,
                    DistillationDisposition.DEVELOPMENT_FAILED,
                    DistillationDisposition.HELD_OUT_FAILED,
                    DistillationDisposition.INSUFFICIENT_EVIDENCE,
                }
                else disposition
            )
            reasons = tuple(dict.fromkeys((*reasons, *extra)))
        return _result(
            disposition,
            decision_class,
            *reasons,
            gate_receipts=gates,
            rolled_back_rule_id=rollback_id,
        )


def _scope_matches(scope: Mapping[str, Any], features: Mapping[str, Any]) -> bool:
    for key, expected in scope.items():
        if key == "decision_class":
            continue
        if key in features and not _values_equal(expected, features[key]):
            return False
    return True


def _coerce_optional_fallback(value: MetaAction | str | None, action: MetaAction) -> MetaAction:
    if value is None:
        return _default_fallback(action)
    return _coerce_action(value, "fallback")


__all__ = [
    "ADVERSARIAL_MUTATION_SCHEMA",
    "DECLARATIVE_WHEN_KEYS",
    "DEFAULT_RULE_VERSION",
    "DISTILLED_DECISION_RULE_INTERFACE",
    "DISTILLATION_EXAMPLE_SCHEMA",
    "DISTILLATION_GATE_RECEIPT_SCHEMA",
    "DISTILLATION_RESULT_SCHEMA",
    "MIN_HELD_OUT_EXAMPLES",
    "MIN_INDEPENDENT_DEVELOPMENT_EXAMPLES",
    "NARROWED_RULE_VERSION",
    "POLICY_DISTILLER_INTERFACE",
    "POLICY_DISTILLER_SCHEMA",
    "RULE_APPLICATION_SCHEMA",
    "AdversarialMutation",
    "DistillationDisposition",
    "DistillationExample",
    "DistillationGate",
    "DistillationGateReceipt",
    "DistillationPartition",
    "DistillationResult",
    "PolicyDistillationError",
    "PolicyDistiller",
    "RuleApplication",
    "StableDecisionClass",
    "common_feature_conjunction",
    "conditions_match",
    "field_is_forbidden",
    "generate_aae_mutations",
    "rule_is_narrower_than_evidence",
]
