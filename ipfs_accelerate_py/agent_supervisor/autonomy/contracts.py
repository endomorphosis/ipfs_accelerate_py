# ruff: noqa: UP042 - the package retains Python 3.8 compatibility
"""Immutable public contracts for bounded supervisor autonomy.

This module is deliberately dependency-light.  It reuses the supervisor's
canonical DAG-JSON/content-identity implementation and defines only the typed
bodies needed by the autonomous meta-controller.  Provider accounting,
artifact persistence, leases, execution admission, and receipt envelopes stay
owned by their existing subsystems and are referenced here by content ID.

Open-ended values are bounded, deeply frozen, and screened for raw prompts,
source bodies, transcripts, executable policy, and secret material.  Floats
are rejected so every accepted value has one canonical integer-unit encoding.
"""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, ClassVar, TypeVar

from ..proof.formal_verification_contracts import (
    CONTRACT_VERSION,
    CanonicalContract,
    ContractValidationError,
    EvidenceFreshness,
    canonical_json_bytes,
)

AUTONOMOUS_META_CONTROLLER_PROGRAM_ID = "agent-supervisor-autonomous-meta-controller-v1"
AUTONOMOUS_META_CONTROLLER_ROOT_OBJECTIVE_ID = "APMC-G000"
AUTONOMOUS_META_CONTROLLER_TASK_PREFIX = "APMC-"

_SCHEMA_PREFIX = "ipfs_accelerate_py/agent-supervisor/autonomy"
MAX_CANONICAL_RECORD_BYTES = 262_144
MAX_TEXT_BYTES = 8_192
MAX_IDENTIFIER_BYTES = 512
MAX_SEQUENCE_ITEMS = 1_024
MAX_MAPPING_ITEMS = 256
MAX_NESTING_DEPTH = 8
MAX_INTEGER = (1 << 63) - 1

_FORBIDDEN_OPEN_FIELD_MARKERS = frozenset(
    {
        "access_token",
        "api_key",
        "authorization",
        "chain_of_thought",
        "cookie",
        "credential",
        "decoded_source",
        "executable_code",
        "hidden_reasoning",
        "model_transcript",
        "password",
        "private_key",
        "prompt",
        "raw_prompt",
        "refresh_token",
        "secret",
        "shell_command",
        "source_body",
        "transcript",
    }
)


class AutonomyContractError(ContractValidationError):
    """Raised when an autonomy contract is malformed or unsafe."""


class AutonomyLevel(str, Enum):
    """Closed autonomy vocabulary; intentionally has no unrestricted level."""

    OBSERVE_ONLY = "observe_only"
    RECOMMEND = "recommend"
    DRY_RUN = "dry_run"
    EXECUTE_REVERSIBLE = "execute_reversible"
    EXECUTE_BOUNDED_MUTATION = "execute_bounded_mutation"
    SELF_REPAIR_ISOLATED = "self_repair_isolated"

    @property
    def rank(self) -> int:
        return tuple(AutonomyLevel).index(self)


class RiskClass(str, Enum):
    R0_PURE = "R0_PURE"
    R1_READ_ONLY = "R1_READ_ONLY"
    R2_REVERSIBLE_LOCAL = "R2_REVERSIBLE_LOCAL"
    R3_BOUNDED_REPOSITORY_MUTATION = "R3_BOUNDED_REPOSITORY_MUTATION"
    R4_SECURITY_OR_PROTOCOL_SENSITIVE = "R4_SECURITY_OR_PROTOCOL_SENSITIVE"
    R5_IRREVERSIBLE_EXTERNAL_OR_LEGAL = "R5_IRREVERSIBLE_EXTERNAL_OR_LEGAL"

    @property
    def rank(self) -> int:
        return tuple(RiskClass).index(self)


class DecisionQuestionType(str, Enum):
    WHICH_FILES_ARE_AFFECTED = "which_files_are_affected"
    WHICH_CONTRACT_CHANGED = "which_contract_changed"
    WHICH_TEST_IS_REQUIRED = "which_test_is_required"
    WHICH_PROOF_OBLIGATION_APPLIES = "which_proof_obligation_applies"
    WHETHER_CACHE_IS_REUSABLE = "whether_cache_is_reusable"
    WHETHER_CAPABILITY_IS_LIVE = "whether_capability_is_live"
    WHETHER_CONTEXT_IS_SUFFICIENT = "whether_context_is_sufficient"
    WHETHER_PATCH_IS_SEMANTICALLY_NONEMPTY = "whether_patch_is_semantically_nonempty"
    WHETHER_FAILURE_IS_FLAKY = "whether_failure_is_flaky"
    WHETHER_REPLAN_IS_REQUIRED = "whether_replan_is_required"
    WHETHER_HUMAN_CHOICE_IS_IRREDUCIBLE = "whether_human_choice_is_irreducible"


class QuestionDisposition(str, Enum):
    UNRESOLVED = "unresolved"
    RESOLVED = "resolved"
    INVALIDATED = "invalidated"
    BLOCKED = "blocked"


class MetaAction(str, Enum):
    NO_OP = "NO_OP"
    READ_CACHED_RECEIPT = "READ_CACHED_RECEIPT"
    RUN_LOCAL_STATIC_ANALYSIS = "RUN_LOCAL_STATIC_ANALYSIS"
    RUN_INCREMENTAL_INDEX_QUERY = "RUN_INCREMENTAL_INDEX_QUERY"
    RUN_GRAPH_RETRIEVAL = "RUN_GRAPH_RETRIEVAL"
    EXPAND_CONTEXT_REFERENCE = "EXPAND_CONTEXT_REFERENCE"
    RUN_SCHEMA_VALIDATION = "RUN_SCHEMA_VALIDATION"
    RUN_TYPE_CHECK = "RUN_TYPE_CHECK"
    RUN_SELECTED_TEST = "RUN_SELECTED_TEST"
    RUN_FULL_VALIDATION = "RUN_FULL_VALIDATION"
    RUN_SMT_OR_PROVER = "RUN_SMT_OR_PROVER"
    CALL_LOCAL_SMALL_MODEL = "CALL_LOCAL_SMALL_MODEL"
    CALL_REMOTE_STANDARD_MODEL = "CALL_REMOTE_STANDARD_MODEL"
    CALL_REMOTE_STRONG_MODEL = "CALL_REMOTE_STRONG_MODEL"
    REQUEST_HUMAN_DECISION = "REQUEST_HUMAN_DECISION"
    GENERATE_BOUNDED_REPAIR = "GENERATE_BOUNDED_REPAIR"
    REPLAN_AFFECTED_SUFFIX = "REPLAN_AFFECTED_SUFFIX"
    QUARANTINE_TASK = "QUARANTINE_TASK"


class ResolutionEvidenceKind(str, Enum):
    NONE = "none"
    CACHED_RECEIPT = "cached_receipt"
    STATIC_ANALYSIS = "static_analysis"
    INDEX_QUERY = "index_query"
    GRAPH_RETRIEVAL = "graph_retrieval"
    CONTEXT_REFERENCE = "context_reference"
    SCHEMA_VALIDATION = "schema_validation"
    TYPE_CHECK = "type_check"
    TEST_RESULT = "test_result"
    VALIDATION_RESULT = "validation_result"
    PROOF_RESULT = "proof_result"
    MODEL_ADVICE = "model_advice"
    HUMAN_DECISION = "human_decision"
    REPAIR_RECEIPT = "repair_receipt"
    REPLAN_RECEIPT = "replan_receipt"
    QUARANTINE_RECEIPT = "quarantine_receipt"


class PrivacyClass(str, Enum):
    PUBLIC = "public"
    REPOSITORY_PRIVATE = "repository_private"
    SENSITIVE = "sensitive"
    LOCAL_ONLY = "local_only"
    FORBIDDEN_EXTERNAL = "forbidden_external"


class AuthorityClass(str, Enum):
    NONE = "none"
    ADVISORY = "advisory"
    DERIVED = "derived"
    VERIFIED = "verified"
    AUTHORITATIVE = "authoritative"
    OPERATOR_REQUIRED = "operator_required"


class CancellationBehavior(str, Enum):
    IMMEDIATE = "immediate"
    COOPERATIVE = "cooperative"
    NOT_CANCELLABLE = "not_cancellable"


class BudgetReservationStatus(str, Enum):
    RESERVED = "reserved"
    RECONCILED = "reconciled"
    RELEASED = "released"
    CANCELLED = "cancelled"


class BudgetPurpose(str, Enum):
    PLANNING = "planning"
    ANALYSIS = "analysis"
    MODEL = "model"
    PROOF = "proof"
    VALIDATION = "validation"
    HUMAN = "human"
    REPAIR = "repair"
    CONTEXT = "context"


class BudgetDimension(str, Enum):
    TOTAL_MODEL_CALLS = "total_model_calls"
    STRONG_MODEL_CALLS = "strong_model_calls"
    INPUT_TOKENS = "input_tokens"
    OUTPUT_TOKENS = "output_tokens"
    PROVIDER_SPEND_MICROS = "provider_spend_micros"
    PROOF_TIME_MS = "proof_time_ms"
    VALIDATION_TIME_MS = "validation_time_ms"
    HUMAN_QUESTIONS = "human_questions"
    REPAIR_ROUNDS = "repair_rounds"
    PLAN_BRANCHES = "plan_branches"
    CONTEXT_EXPANSIONS = "context_expansions"
    WALL_TIME_MS = "wall_time_ms"


class BudgetExhaustionReason(str, Enum):
    CAPACITY_EXHAUSTED = "capacity_exhausted"
    VALIDATION_RESERVE_PROTECTED = "validation_reserve_protected"
    PROOF_RESERVE_PROTECTED = "proof_reserve_protected"
    LEDGER_TERMINAL = "ledger_terminal"


class TerminalStatus(str, Enum):
    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BLOCKED = "blocked"
    UNAVAILABLE = "unavailable"
    EXHAUSTED = "exhausted"
    CANCELLED = "cancelled"
    NON_PROMOTED = "non_promoted"


class MetaDecisionDisposition(str, Enum):
    SELECTED = "selected"
    NO_OP = "no_op"
    ESCALATE = "escalate"
    QUARANTINE = "quarantine"
    BLOCKED = "blocked"


class AttributionCause(str, Enum):
    CONTEXT_OMISSION = "context_omission"
    MODEL_CAPABILITY_FAILURE = "model_capability_failure"
    PROVIDER_FAILURE = "provider_failure"
    BAD_TASK_DECOMPOSITION = "bad_task_decomposition"
    BAD_PLAN_BRANCH = "bad_plan_branch"
    STALE_EVIDENCE = "stale_evidence"
    INCORRECT_CACHE_REUSE = "incorrect_cache_reuse"
    VALIDATION_SELECTION_FAILURE = "validation_selection_failure"
    PROOF_SELECTION_FAILURE = "proof_selection_failure"
    MERGE_CONFLICT = "merge_conflict"
    ENVIRONMENT_FAILURE = "environment_failure"
    HUMAN_POLICY_BLOCKER = "human_policy_blocker"


class RepairTier(str, Enum):
    DETERMINISTIC = "deterministic"
    TEMPLATE_CONSTRAINED = "template_constrained"
    MODEL_ASSISTED_BOUNDED = "model_assisted_bounded"


class DistillationStatus(str, Enum):
    CANDIDATE = "candidate"
    DEVELOPMENT_FAILED = "development_failed"
    HELD_OUT_FAILED = "held_out_failed"
    SHADOW = "shadow"
    PROMOTED = "promoted"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


class PromotionStatus(str, Enum):
    PROMOTED = "promoted"
    NON_PROMOTED = "non_promoted"
    ROLLED_BACK = "rolled_back"


class MemoryClass(str, Enum):
    EPHEMERAL_ATTEMPT = "ephemeral_attempt"
    SHORT_LIVED_NEGATIVE = "short_lived_negative"
    TASK_EPISODE = "task_episode"
    REPOSITORY_PATTERN = "repository_pattern"
    CROSS_REPOSITORY_RULE = "cross_repository_rule"
    AUTHORITATIVE_CURRENT = "authoritative_current"
    WITHDRAWN = "withdrawn"


_EnumT = TypeVar("_EnumT", bound=Enum)


def _enum(value: Any, enum_type: type[_EnumT], name: str) -> _EnumT:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value))
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise AutonomyContractError(f"{name} must be one of: {allowed}") from exc


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise AutonomyContractError(f"{name} must be a boolean")
    return value


def _int(value: Any, name: str, *, maximum: int = MAX_INTEGER) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > maximum:
        raise AutonomyContractError(f"{name} must be an integer between 0 and {maximum}")
    return value


def _bp(value: Any, name: str) -> int:
    return _int(value, name, maximum=10_000)


def _text(value: Any, name: str, *, required: bool = False, maximum: int = MAX_TEXT_BYTES) -> str:
    if value is None:
        result = ""
    elif isinstance(value, str):
        result = unicodedata.normalize("NFC", value.strip())
    else:
        raise AutonomyContractError(f"{name} must be a string")
    if required and not result:
        raise AutonomyContractError(f"{name} is required")
    if len(result.encode("utf-8")) > maximum:
        raise AutonomyContractError(f"{name} exceeds its bounded size")
    if any(ord(char) < 32 and char not in "\t\n\r" for char in result):
        raise AutonomyContractError(f"{name} contains control characters")
    return result


def _id(value: Any, name: str, *, required: bool = True) -> str:
    result = _text(value, name, required=required, maximum=MAX_IDENTIFIER_BYTES)
    if result and any(char.isspace() for char in result):
        raise AutonomyContractError(f"{name} must be a compact identifier")
    return result


def _strings(
    value: Any,
    name: str,
    *,
    required: bool = False,
    identifiers: bool = False,
    preserve_order: bool = False,
    maximum: int = MAX_SEQUENCE_ITEMS,
) -> tuple[str, ...]:
    if value is None:
        raw: Sequence[Any] = ()
    elif isinstance(value, str):
        raw = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        raw = value
    else:
        raise AutonomyContractError(f"{name} must be a sequence of strings")
    if len(raw) > maximum:
        raise AutonomyContractError(f"{name} contains too many items")
    normalized: list[str] = []
    for item in raw:
        result = _id(item, name) if identifiers else _text(item, name, required=True)
        if result not in normalized:
            normalized.append(result)
    if required and not normalized:
        raise AutonomyContractError(f"{name} must not be empty")
    return tuple(normalized if preserve_order else sorted(normalized))


def _path(value: Any, name: str) -> str:
    result = _text(value, name, required=True, maximum=1_024)
    if "\\" in result or "\x00" in result:
        raise AutonomyContractError(f"{name} must be a repository-relative POSIX path")
    parsed = PurePosixPath(result)
    if parsed.is_absolute() or ".." in parsed.parts or result in {".", ""}:
        raise AutonomyContractError(f"{name} must be a repository-relative POSIX path")
    return parsed.as_posix()


def _paths(value: Any, name: str, *, required: bool = False) -> tuple[str, ...]:
    raw = _strings(value, name, required=required, preserve_order=True)
    normalized: list[str] = []
    for item in raw:
        item = _path(item, name)
        if item not in normalized:
            normalized.append(item)
    return tuple(sorted(normalized))


def _has_forbidden_key(key: str) -> bool:
    normalized = key.strip().lower().replace("-", "_")
    return any(
        normalized == marker or normalized.endswith("_" + marker)
        for marker in _FORBIDDEN_OPEN_FIELD_MARKERS
    )


def _freeze_json(value: Any, name: str, *, depth: int = 0) -> Any:
    if depth > MAX_NESTING_DEPTH:
        raise AutonomyContractError(f"{name} exceeds maximum nesting")
    if value is None or isinstance(value, (str, bool, int)):
        if isinstance(value, str):
            return _text(value, name)
        if isinstance(value, int) and not isinstance(value, bool):
            if abs(value) > MAX_INTEGER:
                raise AutonomyContractError(f"{name} integer is out of range")
        return value
    if isinstance(value, float):
        raise AutonomyContractError(f"{name} cannot contain floats")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, CanonicalContract):
        return value
    if isinstance(value, Mapping):
        if len(value) > MAX_MAPPING_ITEMS:
            raise AutonomyContractError(f"{name} contains too many entries")
        result: dict[str, Any] = {}
        for raw_key in sorted(value):
            if not isinstance(raw_key, str):
                raise AutonomyContractError(f"{name} keys must be strings")
            key = _text(raw_key, name, required=True, maximum=256)
            if _has_forbidden_key(key):
                raise AutonomyContractError(f"{name} contains forbidden private or executable data")
            result[key] = _freeze_json(value[raw_key], name, depth=depth + 1)
        return MappingProxyType(result)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) > MAX_SEQUENCE_ITEMS:
            raise AutonomyContractError(f"{name} contains too many items")
        return tuple(_freeze_json(item, name, depth=depth + 1) for item in value)
    raise AutonomyContractError(f"{name} contains unsupported value type {type(value).__name__}")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise AutonomyContractError(f"{name} must be a mapping")
    result = _freeze_json(value, name)
    assert isinstance(result, Mapping)
    return result


def _mapping_of_strings(
    value: Any,
    name: str,
    *,
    value_enum: type[_EnumT] | None = None,
) -> Mapping[str, Any]:
    mapping = _mapping(value, name)
    result: dict[str, Any] = {}
    for key, raw in mapping.items():
        result[_id(key, name)] = _enum(raw, value_enum, name) if value_enum else _text(raw, name)
    return MappingProxyType(dict(sorted(result.items())))


def _contract(value: Any, contract_type: type[Any], name: str) -> Any:
    if isinstance(value, contract_type):
        return value
    if isinstance(value, Mapping):
        return contract_type.from_dict(value)
    raise AutonomyContractError(f"{name} must be a {contract_type.__name__} or mapping")


def _contracts(value: Any, contract_type: type[Any], name: str) -> tuple[Any, ...]:
    if value is None:
        raw: Sequence[Any] = ()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        raw = value
    else:
        raise AutonomyContractError(f"{name} must be a sequence")
    if len(raw) > MAX_SEQUENCE_ITEMS:
        raise AutonomyContractError(f"{name} contains too many items")
    return tuple(_contract(item, contract_type, name) for item in raw)


def _enum_tuple(
    value: Any, enum_type: type[_EnumT], name: str, *, required: bool = False
) -> tuple[_EnumT, ...]:
    if value is None:
        raw: Sequence[Any] = ()
    elif isinstance(value, (str, Enum)):
        raw = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        raw = value
    else:
        raise AutonomyContractError(f"{name} must be a sequence")
    if len(raw) > MAX_SEQUENCE_ITEMS:
        raise AutonomyContractError(f"{name} contains too many items")
    result: list[_EnumT] = []
    for item in raw:
        normalized = _enum(item, enum_type, name)
        if normalized not in result:
            result.append(normalized)
    if required and not result:
        raise AutonomyContractError(f"{name} must not be empty")
    return tuple(result)


class _AutonomyContract(CanonicalContract):
    """Strict, bounded, content-addressed contract mixin."""

    IDENTITY_ALIASES: ClassVar[tuple[str, ...]] = ()

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            **{item.name: getattr(self, item.name) for item in fields(self)},
        }

    def _seal(self) -> None:
        if len(canonical_json_bytes(self.to_dict())) > MAX_CANONICAL_RECORD_BYTES:
            raise AutonomyContractError("autonomy contract exceeds its bounded canonical size")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Any:
        if not isinstance(payload, Mapping):
            raise AutonomyContractError("contract payload must be an object")
        supplied_schema = payload.get("schema")
        if supplied_schema not in (None, "", cls.SCHEMA):
            raise AutonomyContractError(f"unsupported contract schema; use {cls.SCHEMA}")
        if payload.get("contract_version", CONTRACT_VERSION) != CONTRACT_VERSION:
            raise AutonomyContractError("unsupported autonomy contract version")
        names = {item.name for item in fields(cls)}
        allowed = names | {"schema", "contract_version", "content_id", *cls.IDENTITY_ALIASES}
        if set(payload).difference(allowed):
            raise AutonomyContractError(
                f"{cls.__name__} contains unsupported fields; rebuild its canonical payload"
            )
        kwargs = {name: payload[name] for name in names if name in payload}
        try:
            result = cls(**kwargs)
        except TypeError as exc:
            raise AutonomyContractError(f"{cls.__name__} is missing required fields") from exc
        claims = [
            payload.get(name) for name in ("content_id", *cls.IDENTITY_ALIASES) if payload.get(name)
        ]
        if any(claim != result.content_id for claim in claims):
            raise AutonomyContractError("autonomy content identity does not match payload")
        return result

    @classmethod
    def from_json(cls, payload: str) -> Any:
        duplicates: set[str] = set()

        def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    duplicates.add(key)
                result[key] = value
            return result

        try:
            decoded = json.loads(payload, object_pairs_hook=pairs_hook)
        except (TypeError, json.JSONDecodeError) as exc:
            raise AutonomyContractError("contract JSON is malformed") from exc
        if duplicates:
            raise AutonomyContractError("contract JSON contains duplicate fields")
        return cls.from_dict(decoded)


def _schema(name: str) -> str:
    return f"{_SCHEMA_PREFIX}/{name}@1"


def _initial_risk_levels() -> Mapping[str, AutonomyLevel]:
    return {
        RiskClass.R0_PURE.value: AutonomyLevel.EXECUTE_REVERSIBLE,
        RiskClass.R1_READ_ONLY.value: AutonomyLevel.EXECUTE_REVERSIBLE,
        RiskClass.R2_REVERSIBLE_LOCAL.value: AutonomyLevel.EXECUTE_REVERSIBLE,
        RiskClass.R3_BOUNDED_REPOSITORY_MUTATION.value: AutonomyLevel.SELF_REPAIR_ISOLATED,
        RiskClass.R4_SECURITY_OR_PROTOCOL_SENSITIVE.value: AutonomyLevel.DRY_RUN,
        RiskClass.R5_IRREVERSIBLE_EXTERNAL_OR_LEGAL.value: AutonomyLevel.RECOMMEND,
    }


@dataclass(frozen=True)
class AutonomyPolicy(_AutonomyContract):
    SCHEMA: ClassVar[str] = _schema("policy")
    IDENTITY_ALIASES: ClassVar[tuple[str, ...]] = ("policy_id",)

    policy_revision: str
    authority_id: str
    human_escalation_policy_id: str
    default_level: AutonomyLevel = AutonomyLevel.OBSERVE_ONLY
    max_level_by_risk: Mapping[str, AutonomyLevel] = field(default_factory=_initial_risk_levels)
    required_validation_by_risk: Mapping[str, Any] = field(default_factory=dict)
    autonomous_merge_enabled: bool = False
    expires_at_ms: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_revision", _id(self.policy_revision, "policy_revision"))
        object.__setattr__(self, "authority_id", _id(self.authority_id, "authority_id"))
        object.__setattr__(
            self,
            "human_escalation_policy_id",
            _id(self.human_escalation_policy_id, "human_escalation_policy_id"),
        )
        object.__setattr__(
            self, "default_level", _enum(self.default_level, AutonomyLevel, "default_level")
        )
        levels = _mapping_of_strings(
            self.max_level_by_risk, "max_level_by_risk", value_enum=AutonomyLevel
        )
        expected = {risk.value for risk in RiskClass}
        if set(levels) != expected:
            raise AutonomyContractError("max_level_by_risk must define every closed risk class")
        if (
            levels[RiskClass.R5_IRREVERSIBLE_EXTERNAL_OR_LEGAL.value].rank
            > AutonomyLevel.RECOMMEND.rank
        ):
            raise AutonomyContractError("R5 always requires a human decision")
        object.__setattr__(self, "max_level_by_risk", levels)
        object.__setattr__(
            self,
            "required_validation_by_risk",
            _mapping(self.required_validation_by_risk, "required_validation_by_risk"),
        )
        object.__setattr__(
            self,
            "autonomous_merge_enabled",
            _bool(self.autonomous_merge_enabled, "autonomous_merge_enabled"),
        )
        object.__setattr__(self, "expires_at_ms", _int(self.expires_at_ms, "expires_at_ms"))
        self._seal()

    @property
    def policy_id(self) -> str:
        return self.content_id

    def allows(self, level: AutonomyLevel, risk: RiskClass) -> bool:
        requested = _enum(level, AutonomyLevel, "level")
        assessed = _enum(risk, RiskClass, "risk")
        maximum = _enum(self.max_level_by_risk[assessed.value], AutonomyLevel, "maximum")
        # The active/default policy level is a second independent ceiling.
        # Risk defaults may narrow it but can never be used by an operation to
        # raise its own currently admitted autonomy.
        return requested.rank <= min(self.default_level.rank, maximum.rank)


@dataclass(frozen=True)
class RiskAssessment(_AutonomyContract):
    SCHEMA: ClassVar[str] = _schema("risk-assessment")
    IDENTITY_ALIASES: ClassVar[tuple[str, ...]] = ("assessment_id", "risk_assessment_id")

    risk_class: RiskClass
    reversible: bool
    blast_radius_paths: tuple[str, ...] = ()
    blast_radius_symbols: tuple[str, ...] = ()
    security_sensitive: bool = False
    protocol_sensitive: bool = False
    irreversible_external_effect: bool = False
    legal_or_financial_effect: bool = False
    evidence_ids: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "risk_class", _enum(self.risk_class, RiskClass, "risk_class"))
        for name in (
            "reversible",
            "security_sensitive",
            "protocol_sensitive",
            "irreversible_external_effect",
            "legal_or_financial_effect",
        ):
            object.__setattr__(self, name, _bool(getattr(self, name), name))
        object.__setattr__(
            self, "blast_radius_paths", _paths(self.blast_radius_paths, "blast_radius_paths")
        )
        object.__setattr__(
            self,
            "blast_radius_symbols",
            _strings(self.blast_radius_symbols, "blast_radius_symbols"),
        )
        object.__setattr__(
            self, "evidence_ids", _strings(self.evidence_ids, "evidence_ids", identifiers=True)
        )
        object.__setattr__(
            self, "reason_codes", _strings(self.reason_codes, "reason_codes", identifiers=True)
        )
        if (
            self.security_sensitive or self.protocol_sensitive
        ) and self.risk_class.rank < RiskClass.R4_SECURITY_OR_PROTOCOL_SENSITIVE.rank:
            raise AutonomyContractError("security or protocol sensitivity requires risk R4 or R5")
        if (
            self.irreversible_external_effect or self.legal_or_financial_effect
        ) and self.risk_class is not RiskClass.R5_IRREVERSIBLE_EXTERNAL_OR_LEGAL:
            raise AutonomyContractError("irreversible external or legal effects require risk R5")
        if self.risk_class is RiskClass.R5_IRREVERSIBLE_EXTERNAL_OR_LEGAL and self.reversible:
            raise AutonomyContractError("R5 effects cannot be declared reversible")
        self._seal()

    @property
    def risk_assessment_id(self) -> str:
        return self.content_id

    assessment_id = risk_assessment_id


@dataclass(frozen=True)
class CognitiveBudget(_AutonomyContract):
    SCHEMA: ClassVar[str] = _schema("cognitive-budget")
    IDENTITY_ALIASES: ClassVar[tuple[str, ...]] = ("budget_id",)

    max_total_model_calls: int
    max_strong_model_calls: int
    max_input_tokens: int
    max_output_tokens: int
    max_provider_spend_micros: int
    max_proof_time_ms: int
    max_validation_time_ms: int
    max_human_questions: int
    max_repair_rounds: int
    max_plan_branches: int
    max_context_expansions: int
    max_wall_time_ms: int
    validation_reserve_ms: int = 0
    proof_reserve_ms: int = 0

    def __post_init__(self) -> None:
        for name in _BUDGET_AMOUNT_FIELDS:
            object.__setattr__(self, name, _int(getattr(self, name), name))
        object.__setattr__(
            self, "validation_reserve_ms", _int(self.validation_reserve_ms, "validation_reserve_ms")
        )
        object.__setattr__(
            self, "proof_reserve_ms", _int(self.proof_reserve_ms, "proof_reserve_ms")
        )
        if self.max_strong_model_calls > self.max_total_model_calls:
            raise AutonomyContractError("strong-model calls cannot exceed total model calls")
        if self.validation_reserve_ms > self.max_validation_time_ms:
            raise AutonomyContractError("validation reserve cannot exceed validation budget")
        if self.proof_reserve_ms > self.max_proof_time_ms:
            raise AutonomyContractError("proof reserve cannot exceed proof budget")
        self._seal()

    @property
    def budget_id(self) -> str:
        return self.content_id


_BUDGET_AMOUNT_FIELDS = (
    "max_total_model_calls",
    "max_strong_model_calls",
    "max_input_tokens",
    "max_output_tokens",
    "max_provider_spend_micros",
    "max_proof_time_ms",
    "max_validation_time_ms",
    "max_human_questions",
    "max_repair_rounds",
    "max_plan_branches",
    "max_context_expansions",
    "max_wall_time_ms",
)


@dataclass(frozen=True)
class AutonomyEnvelope(_AutonomyContract):
    SCHEMA: ClassVar[str] = _schema("envelope")
    IDENTITY_ALIASES: ClassVar[tuple[str, ...]] = ("envelope_id",)

    repository_id: str
    tree_id: str
    objective_id: str
    objective_revision: str
    task_id: str
    acceptance_criterion_ids: tuple[str, ...]
    risk_assessment: RiskAssessment
    autonomy_level: AutonomyLevel
    cognitive_budget: CognitiveBudget
    allowed_paths: tuple[str, ...]
    allowed_symbols: tuple[str, ...]
    required_test_ids: tuple[str, ...]
    required_proof_ids: tuple[str, ...]
    authority_id: str
    policy_id: str
    provider_usage_envelope_id: str
    resource_budget_id: str
    human_escalation_policy_id: str
    expiry_ms: int
    reversible: bool = False
    blast_radius: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "repository_id",
            "tree_id",
            "objective_id",
            "objective_revision",
            "task_id",
            "authority_id",
            "policy_id",
            "provider_usage_envelope_id",
            "resource_budget_id",
            "human_escalation_policy_id",
        ):
            object.__setattr__(self, name, _id(getattr(self, name), name))
        object.__setattr__(
            self,
            "acceptance_criterion_ids",
            _strings(
                self.acceptance_criterion_ids,
                "acceptance_criterion_ids",
                identifiers=True,
                required=True,
            ),
        )
        object.__setattr__(
            self,
            "risk_assessment",
            _contract(self.risk_assessment, RiskAssessment, "risk_assessment"),
        )
        object.__setattr__(
            self, "autonomy_level", _enum(self.autonomy_level, AutonomyLevel, "autonomy_level")
        )
        object.__setattr__(
            self,
            "cognitive_budget",
            _contract(self.cognitive_budget, CognitiveBudget, "cognitive_budget"),
        )
        object.__setattr__(self, "allowed_paths", _paths(self.allowed_paths, "allowed_paths"))
        object.__setattr__(
            self, "allowed_symbols", _strings(self.allowed_symbols, "allowed_symbols")
        )
        object.__setattr__(
            self,
            "required_test_ids",
            _strings(self.required_test_ids, "required_test_ids", identifiers=True),
        )
        object.__setattr__(
            self,
            "required_proof_ids",
            _strings(self.required_proof_ids, "required_proof_ids", identifiers=True),
        )
        object.__setattr__(self, "expiry_ms", _int(self.expiry_ms, "expiry_ms"))
        object.__setattr__(self, "reversible", _bool(self.reversible, "reversible"))
        object.__setattr__(self, "blast_radius", _mapping(self.blast_radius, "blast_radius"))
        if (
            self.risk_assessment.risk_class is RiskClass.R5_IRREVERSIBLE_EXTERNAL_OR_LEGAL
            and self.autonomy_level.rank > AutonomyLevel.RECOMMEND.rank
        ):
            raise AutonomyContractError("R5 envelopes cannot authorize execution")
        if (
            self.risk_assessment.risk_class is RiskClass.R4_SECURITY_OR_PROTOCOL_SENSITIVE
            and self.autonomy_level.rank > AutonomyLevel.DRY_RUN.rank
        ):
            raise AutonomyContractError(
                "R4 execution requires a separately admitted operator policy"
            )
        if self.reversible != self.risk_assessment.reversible:
            raise AutonomyContractError("envelope reversibility must match its risk assessment")
        self._seal()

    @property
    def envelope_id(self) -> str:
        return self.content_id


@dataclass(frozen=True)
class DecisionQuestion(_AutonomyContract):
    SCHEMA: ClassVar[str] = _schema("decision-question")
    IDENTITY_ALIASES: ClassVar[tuple[str, ...]] = ("question_id",)

    objective_id: str
    acceptance_criterion_ids: tuple[str, ...]
    question_type: DecisionQuestionType
    current_alternatives: tuple[str, ...]
    required_evidence_ids: tuple[str, ...]
    known_evidence_ids: tuple[str, ...]
    contradictory_evidence_ids: tuple[str, ...]
    residual_uncertainty_bp: int
    decision_deadline_ms: int
    risk_if_incorrect: RiskClass
    risk_if_left_unresolved: RiskClass
    possible_resolution_action_ids: tuple[str, ...]
    dependency_question_ids: tuple[str, ...]
    terminal_decision_rule: str
    mandatory: bool = True
    disposition: QuestionDisposition = QuestionDisposition.UNRESOLVED
    terminal_answer: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "objective_id", _id(self.objective_id, "objective_id"))
        object.__setattr__(
            self,
            "acceptance_criterion_ids",
            _strings(
                self.acceptance_criterion_ids,
                "acceptance_criterion_ids",
                identifiers=True,
                required=True,
            ),
        )
        object.__setattr__(
            self, "question_type", _enum(self.question_type, DecisionQuestionType, "question_type")
        )
        object.__setattr__(
            self,
            "current_alternatives",
            _strings(
                self.current_alternatives,
                "current_alternatives",
                required=True,
                preserve_order=True,
            ),
        )
        for name in (
            "required_evidence_ids",
            "known_evidence_ids",
            "contradictory_evidence_ids",
            "possible_resolution_action_ids",
            "dependency_question_ids",
        ):
            object.__setattr__(self, name, _strings(getattr(self, name), name, identifiers=True))
        object.__setattr__(
            self,
            "residual_uncertainty_bp",
            _bp(self.residual_uncertainty_bp, "residual_uncertainty_bp"),
        )
        object.__setattr__(
            self, "decision_deadline_ms", _int(self.decision_deadline_ms, "decision_deadline_ms")
        )
        object.__setattr__(
            self, "risk_if_incorrect", _enum(self.risk_if_incorrect, RiskClass, "risk_if_incorrect")
        )
        object.__setattr__(
            self,
            "risk_if_left_unresolved",
            _enum(self.risk_if_left_unresolved, RiskClass, "risk_if_left_unresolved"),
        )
        object.__setattr__(
            self,
            "terminal_decision_rule",
            _text(self.terminal_decision_rule, "terminal_decision_rule", required=True),
        )
        object.__setattr__(self, "mandatory", _bool(self.mandatory, "mandatory"))
        object.__setattr__(
            self, "disposition", _enum(self.disposition, QuestionDisposition, "disposition")
        )
        object.__setattr__(self, "terminal_answer", _text(self.terminal_answer, "terminal_answer"))
        if set(self.known_evidence_ids).intersection(self.contradictory_evidence_ids):
            raise AutonomyContractError("known and contradictory evidence must be disjoint")
        if self.disposition is QuestionDisposition.RESOLVED and not self.terminal_answer:
            raise AutonomyContractError("a resolved question requires a terminal answer")
        if self.disposition is not QuestionDisposition.RESOLVED and self.terminal_answer:
            raise AutonomyContractError("only a resolved question may carry a terminal answer")
        self._seal()

    @property
    def question_id(self) -> str:
        return self.content_id


@dataclass(frozen=True)
class DecisionGraph(_AutonomyContract):
    SCHEMA: ClassVar[str] = _schema("decision-graph")
    IDENTITY_ALIASES: ClassVar[tuple[str, ...]] = ("graph_id",)

    repository_id: str
    tree_id: str
    objective_id: str
    objective_revision: str
    graph_revision: int
    questions: tuple[DecisionQuestion, ...]
    evidence_dependencies: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("repository_id", "tree_id", "objective_id", "objective_revision"):
            object.__setattr__(self, name, _id(getattr(self, name), name))
        object.__setattr__(self, "graph_revision", _int(self.graph_revision, "graph_revision"))
        object.__setattr__(
            self, "questions", _contracts(self.questions, DecisionQuestion, "questions")
        )
        dependencies = _mapping(self.evidence_dependencies, "evidence_dependencies")
        normalized_dependencies: dict[str, tuple[str, ...]] = {}
        question_ids = {question.question_id for question in self.questions}
        if len(question_ids) != len(self.questions):
            raise AutonomyContractError("decision graph contains duplicate questions")
        for key, value in dependencies.items():
            question_id = _id(key, "evidence_dependencies")
            if question_id not in question_ids:
                raise AutonomyContractError("evidence dependency references an unknown question")
            normalized_dependencies[question_id] = _strings(
                value, "evidence_dependencies", identifiers=True
            )
        for question in self.questions:
            if question.objective_id != self.objective_id:
                raise AutonomyContractError("question objective does not match graph objective")
            if question.question_id in question.dependency_question_ids:
                raise AutonomyContractError("a question cannot depend on itself")
            if not set(question.dependency_question_ids).issubset(question_ids):
                raise AutonomyContractError("question dependency references an unknown question")
        visiting: set[str] = set()
        visited: set[str] = set()
        by_id = {question.question_id: question for question in self.questions}

        def visit(question_id: str) -> None:
            if question_id in visiting:
                raise AutonomyContractError("decision graph dependencies must be acyclic")
            if question_id in visited:
                return
            visiting.add(question_id)
            for dependency in by_id[question_id].dependency_question_ids:
                visit(dependency)
            visiting.remove(question_id)
            visited.add(question_id)

        for question_id in sorted(question_ids):
            visit(question_id)
        object.__setattr__(
            self,
            "evidence_dependencies",
            MappingProxyType(dict(sorted(normalized_dependencies.items()))),
        )
        self._seal()

    @property
    def graph_id(self) -> str:
        return self.content_id


@dataclass(frozen=True)
class BeliefFact(_AutonomyContract):
    SCHEMA: ClassVar[str] = _schema("belief-fact")
    IDENTITY_ALIASES: ClassVar[tuple[str, ...]] = ("fact_id",)

    subject_question_id: str
    predicate: str
    value: Any
    evidence_ids: tuple[str, ...]
    authority_class: AuthorityClass
    freshness: EvidenceFreshness
    confidence_bp: int
    observed_tree_id: str
    expires_at_ms: int = 0
    contradicts_fact_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "subject_question_id", _id(self.subject_question_id, "subject_question_id")
        )
        object.__setattr__(self, "predicate", _id(self.predicate, "predicate"))
        object.__setattr__(self, "value", _freeze_json(self.value, "value"))
        object.__setattr__(
            self,
            "evidence_ids",
            _strings(self.evidence_ids, "evidence_ids", identifiers=True, required=True),
        )
        object.__setattr__(
            self, "authority_class", _enum(self.authority_class, AuthorityClass, "authority_class")
        )
        object.__setattr__(self, "freshness", _enum(self.freshness, EvidenceFreshness, "freshness"))
        object.__setattr__(self, "confidence_bp", _bp(self.confidence_bp, "confidence_bp"))
        object.__setattr__(self, "observed_tree_id", _id(self.observed_tree_id, "observed_tree_id"))
        object.__setattr__(self, "expires_at_ms", _int(self.expires_at_ms, "expires_at_ms"))
        object.__setattr__(
            self,
            "contradicts_fact_ids",
            _strings(self.contradicts_fact_ids, "contradicts_fact_ids", identifiers=True),
        )
        if (
            self.authority_class is AuthorityClass.AUTHORITATIVE
            and self.freshness is not EvidenceFreshness.CURRENT
        ):
            raise AutonomyContractError("authoritative belief facts must have current evidence")
        self._seal()

    @property
    def fact_id(self) -> str:
        return self.content_id


@dataclass(frozen=True)
class BeliefState(_AutonomyContract):
    SCHEMA: ClassVar[str] = _schema("belief-state")
    IDENTITY_ALIASES: ClassVar[tuple[str, ...]] = ("belief_state_id",)

    objective_id: str
    objective_revision: str
    current_tree_id: str
    revision: int
    facts: tuple[BeliefFact, ...]

    def __post_init__(self) -> None:
        for name in ("objective_id", "objective_revision", "current_tree_id"):
            object.__setattr__(self, name, _id(getattr(self, name), name))
        object.__setattr__(self, "revision", _int(self.revision, "revision"))
        object.__setattr__(self, "facts", _contracts(self.facts, BeliefFact, "facts"))
        if len({fact.fact_id for fact in self.facts}) != len(self.facts):
            raise AutonomyContractError("belief state contains duplicate facts")
        if any(
            fact.authority_class is AuthorityClass.AUTHORITATIVE
            and fact.observed_tree_id != self.current_tree_id
            for fact in self.facts
        ):
            raise AutonomyContractError(
                "authoritative belief facts must bind the belief state's current tree"
            )
        self._seal()

    @property
    def belief_state_id(self) -> str:
        return self.content_id


@dataclass(frozen=True)
class ResolutionAction(_AutonomyContract):
    SCHEMA: ClassVar[str] = _schema("resolution-action")
    IDENTITY_ALIASES: ClassVar[tuple[str, ...]] = ("action_id",)

    action: MetaAction
    precondition_ids: tuple[str, ...]
    expected_evidence_kind: ResolutionEvidenceKind
    expected_uncertainty_reduction_bp: int
    token_cost: int
    latency_cost_ms: int
    provider_cost_micros: int
    resource_cost_units: int
    invalidation_cost_units: int
    privacy_cost_units: int
    privacy_class: PrivacyClass
    risk_class: RiskClass
    cancellation_behavior: CancellationBehavior
    cacheable: bool
    authority_class: AuthorityClass
    can_change_decision: bool = True
    accepted_as_authority: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", _enum(self.action, MetaAction, "action"))
        object.__setattr__(
            self,
            "precondition_ids",
            _strings(self.precondition_ids, "precondition_ids", identifiers=True),
        )
        object.__setattr__(
            self,
            "expected_evidence_kind",
            _enum(self.expected_evidence_kind, ResolutionEvidenceKind, "expected_evidence_kind"),
        )
        object.__setattr__(
            self,
            "expected_uncertainty_reduction_bp",
            _bp(self.expected_uncertainty_reduction_bp, "expected_uncertainty_reduction_bp"),
        )
        for name in (
            "token_cost",
            "latency_cost_ms",
            "provider_cost_micros",
            "resource_cost_units",
            "invalidation_cost_units",
            "privacy_cost_units",
        ):
            object.__setattr__(self, name, _int(getattr(self, name), name))
        object.__setattr__(
            self, "privacy_class", _enum(self.privacy_class, PrivacyClass, "privacy_class")
        )
        object.__setattr__(self, "risk_class", _enum(self.risk_class, RiskClass, "risk_class"))
        object.__setattr__(
            self,
            "cancellation_behavior",
            _enum(self.cancellation_behavior, CancellationBehavior, "cancellation_behavior"),
        )
        object.__setattr__(self, "cacheable", _bool(self.cacheable, "cacheable"))
        object.__setattr__(
            self, "authority_class", _enum(self.authority_class, AuthorityClass, "authority_class")
        )
        object.__setattr__(
            self, "can_change_decision", _bool(self.can_change_decision, "can_change_decision")
        )
        object.__setattr__(
            self,
            "accepted_as_authority",
            _bool(self.accepted_as_authority, "accepted_as_authority"),
        )
        if self.action in {
            MetaAction.CALL_REMOTE_STANDARD_MODEL,
            MetaAction.CALL_REMOTE_STRONG_MODEL,
        } and self.privacy_class in {PrivacyClass.LOCAL_ONLY, PrivacyClass.FORBIDDEN_EXTERNAL}:
            raise AutonomyContractError("privacy policy forbids this remote model action")
        if self.accepted_as_authority and self.authority_class in {
            AuthorityClass.NONE,
            AuthorityClass.ADVISORY,
        }:
            raise AutonomyContractError("advisory results cannot be accepted as authority")
        self._seal()

    @property
    def action_id(self) -> str:
        return self.content_id


@dataclass(frozen=True)
class ResolutionCandidate(_AutonomyContract):
    SCHEMA: ClassVar[str] = _schema("resolution-candidate")
    IDENTITY_ALIASES: ClassVar[tuple[str, ...]] = ("candidate_id",)

    question_id: str
    resolution_action: ResolutionAction
    expected_decision_value: int
    admissible: bool
    policy_id: str
    reason_codes: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "question_id", _id(self.question_id, "question_id"))
        object.__setattr__(
            self,
            "resolution_action",
            _contract(self.resolution_action, ResolutionAction, "resolution_action"),
        )
        object.__setattr__(
            self,
            "expected_decision_value",
            _int(self.expected_decision_value, "expected_decision_value"),
        )
        object.__setattr__(self, "admissible", _bool(self.admissible, "admissible"))
        object.__setattr__(self, "policy_id", _id(self.policy_id, "policy_id"))
        object.__setattr__(
            self, "reason_codes", _strings(self.reason_codes, "reason_codes", identifiers=True)
        )
        object.__setattr__(
            self, "evidence_ids", _strings(self.evidence_ids, "evidence_ids", identifiers=True)
        )
        self._seal()

    @property
    def candidate_id(self) -> str:
        return self.content_id


@dataclass(frozen=True)
class MetaDecision(_AutonomyContract):
    SCHEMA: ClassVar[str] = _schema("meta-decision")
    IDENTITY_ALIASES: ClassVar[tuple[str, ...]] = ("decision_id",)

    question_id: str
    selected_candidate_id: str
    selected_action: MetaAction
    considered_candidate_ids: tuple[str, ...]
    rejected_candidate_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    reservation_id: str
    policy_id: str
    disposition: MetaDecisionDisposition
    reason_codes: tuple[str, ...] = ()
    authorizes_mutation: bool = False

    def __post_init__(self) -> None:
        for name in ("question_id", "selected_candidate_id", "reservation_id", "policy_id"):
            object.__setattr__(
                self,
                name,
                _id(
                    getattr(self, name),
                    name,
                    required=name not in {"selected_candidate_id", "reservation_id"},
                ),
            )
        object.__setattr__(
            self, "selected_action", _enum(self.selected_action, MetaAction, "selected_action")
        )
        for name in (
            "considered_candidate_ids",
            "rejected_candidate_ids",
            "evidence_ids",
            "reason_codes",
        ):
            object.__setattr__(self, name, _strings(getattr(self, name), name, identifiers=True))
        object.__setattr__(
            self, "disposition", _enum(self.disposition, MetaDecisionDisposition, "disposition")
        )
        object.__setattr__(
            self, "authorizes_mutation", _bool(self.authorizes_mutation, "authorizes_mutation")
        )
        if self.authorizes_mutation:
            raise AutonomyContractError("a meta-decision receipt cannot itself authorize mutation")
        if self.disposition is MetaDecisionDisposition.SELECTED and not self.selected_candidate_id:
            raise AutonomyContractError("a selected decision requires a candidate")
        self._seal()

    @property
    def decision_id(self) -> str:
        return self.content_id


@dataclass(frozen=True)
class BudgetExhaustion(_AutonomyContract):
    """Non-authorizing receipt for a rejected cognitive reservation."""

    SCHEMA: ClassVar[str] = _schema("budget-exhaustion")
    IDENTITY_ALIASES: ClassVar[tuple[str, ...]] = ("exhaustion_id",)

    budget_id: str
    ledger_id: str
    idempotency_key: str
    request_fingerprint: str
    question_id: str
    action_id: str
    purpose: BudgetPurpose
    dimension: BudgetDimension
    reason: BudgetExhaustionReason
    requested: int
    available: int
    protected_reserve: int = 0

    def __post_init__(self) -> None:
        for name in (
            "budget_id",
            "ledger_id",
            "idempotency_key",
            "request_fingerprint",
            "question_id",
            "action_id",
        ):
            object.__setattr__(self, name, _id(getattr(self, name), name))
        object.__setattr__(self, "purpose", _enum(self.purpose, BudgetPurpose, "purpose"))
        object.__setattr__(self, "dimension", _enum(self.dimension, BudgetDimension, "dimension"))
        object.__setattr__(
            self,
            "reason",
            _enum(self.reason, BudgetExhaustionReason, "reason"),
        )
        for name in ("requested", "available", "protected_reserve"):
            object.__setattr__(self, name, _int(getattr(self, name), name))
        self._seal()

    @property
    def exhaustion_id(self) -> str:
        return self.content_id

    @property
    def terminal_status(self) -> TerminalStatus:
        return TerminalStatus.EXHAUSTED


@dataclass(frozen=True)
class BudgetReservation(_AutonomyContract):
    SCHEMA: ClassVar[str] = _schema("budget-reservation")
    IDENTITY_ALIASES: ClassVar[tuple[str, ...]] = ("reservation_id",)

    budget_id: str
    idempotency_key: str
    question_id: str
    action_id: str
    purpose: BudgetPurpose
    status: BudgetReservationStatus
    max_total_model_calls: int = 0
    max_strong_model_calls: int = 0
    max_input_tokens: int = 0
    max_output_tokens: int = 0
    max_provider_spend_micros: int = 0
    max_proof_time_ms: int = 0
    max_validation_time_ms: int = 0
    max_human_questions: int = 0
    max_repair_rounds: int = 0
    max_plan_branches: int = 0
    max_context_expansions: int = 0
    max_wall_time_ms: int = 0
    actual_total_model_calls: int = 0
    actual_strong_model_calls: int = 0
    actual_input_tokens: int = 0
    actual_output_tokens: int = 0
    actual_provider_spend_micros: int = 0
    actual_proof_time_ms: int = 0
    actual_validation_time_ms: int = 0
    actual_human_questions: int = 0
    actual_repair_rounds: int = 0
    actual_plan_branches: int = 0
    actual_context_expansions: int = 0
    actual_wall_time_ms: int = 0
    provider_usage_receipt_ids: tuple[str, ...] = ()
    token_measurement_ids: tuple[str, ...] = ()
    expires_at_ms: int = 0

    def __post_init__(self) -> None:
        for name in ("budget_id", "idempotency_key", "question_id", "action_id"):
            object.__setattr__(self, name, _id(getattr(self, name), name))
        object.__setattr__(self, "purpose", _enum(self.purpose, BudgetPurpose, "purpose"))
        object.__setattr__(self, "status", _enum(self.status, BudgetReservationStatus, "status"))
        for name in _BUDGET_AMOUNT_FIELDS:
            object.__setattr__(self, name, _int(getattr(self, name), name))
        actual_names = tuple("actual_" + name[4:] for name in _BUDGET_AMOUNT_FIELDS)
        for name in actual_names:
            object.__setattr__(self, name, _int(getattr(self, name), name))
        object.__setattr__(
            self,
            "provider_usage_receipt_ids",
            _strings(
                self.provider_usage_receipt_ids,
                "provider_usage_receipt_ids",
                identifiers=True,
            ),
        )
        object.__setattr__(
            self,
            "token_measurement_ids",
            _strings(
                self.token_measurement_ids,
                "token_measurement_ids",
                identifiers=True,
            ),
        )
        object.__setattr__(self, "expires_at_ms", _int(self.expires_at_ms, "expires_at_ms"))
        if self.max_strong_model_calls > self.max_total_model_calls:
            raise AutonomyContractError(
                "strong-model reservation cannot exceed total model reservation"
            )
        if self.actual_strong_model_calls > self.actual_total_model_calls:
            raise AutonomyContractError(
                "actual strong-model calls cannot exceed actual total model calls"
            )
        actual_nonzero = any(getattr(self, name) for name in actual_names)
        if self.status is BudgetReservationStatus.RECONCILED:
            if (
                self.actual_input_tokens or self.actual_output_tokens
            ) and not self.token_measurement_ids:
                raise AutonomyContractError(
                    "reconciled token use requires token-measurement authority"
                )
            if (
                self.actual_total_model_calls or self.actual_provider_spend_micros
            ) and not self.provider_usage_receipt_ids:
                raise AutonomyContractError(
                    "reconciled provider use requires provider-usage authority"
                )
        elif actual_nonzero or self.provider_usage_receipt_ids or self.token_measurement_ids:
            raise AutonomyContractError("only a reconciled reservation may contain actual usage")
        self._seal()

    @property
    def reservation_id(self) -> str:
        return self.content_id


@dataclass(frozen=True)
class BudgetLedger(_AutonomyContract):
    SCHEMA: ClassVar[str] = _schema("budget-ledger")
    IDENTITY_ALIASES: ClassVar[tuple[str, ...]] = ("ledger_id",)

    budget: CognitiveBudget
    epoch: int
    reservations: tuple[BudgetReservation, ...] = ()
    exhaustions: tuple[BudgetExhaustion, ...] = ()
    committed_total_model_calls: int = 0
    committed_strong_model_calls: int = 0
    committed_input_tokens: int = 0
    committed_output_tokens: int = 0
    committed_provider_spend_micros: int = 0
    committed_proof_time_ms: int = 0
    committed_validation_time_ms: int = 0
    committed_human_questions: int = 0
    committed_repair_rounds: int = 0
    committed_plan_branches: int = 0
    committed_context_expansions: int = 0
    committed_wall_time_ms: int = 0
    provider_usage_receipt_ids: tuple[str, ...] = ()
    token_measurement_ids: tuple[str, ...] = ()
    status: TerminalStatus = TerminalStatus.PENDING

    def __post_init__(self) -> None:
        object.__setattr__(self, "budget", _contract(self.budget, CognitiveBudget, "budget"))
        object.__setattr__(self, "epoch", _int(self.epoch, "epoch"))
        object.__setattr__(
            self, "reservations", _contracts(self.reservations, BudgetReservation, "reservations")
        )
        object.__setattr__(
            self,
            "exhaustions",
            _contracts(self.exhaustions, BudgetExhaustion, "exhaustions"),
        )
        if any(item.budget_id != self.budget.budget_id for item in self.reservations):
            raise AutonomyContractError("reservation is bound to a different budget")
        if any(item.budget_id != self.budget.budget_id for item in self.exhaustions):
            raise AutonomyContractError("exhaustion is bound to a different budget")
        if len({item.idempotency_key for item in self.reservations}) != len(self.reservations):
            raise AutonomyContractError("budget ledger contains duplicate idempotency keys")
        exhaustion_keys = {item.idempotency_key for item in self.exhaustions}
        if len(exhaustion_keys) != len(self.exhaustions):
            raise AutonomyContractError("budget ledger contains duplicate exhaustion keys")
        if exhaustion_keys.intersection(item.idempotency_key for item in self.reservations):
            raise AutonomyContractError(
                "budget idempotency key has conflicting reservation outcomes"
            )
        committed_names = tuple("committed_" + name[4:] for name in _BUDGET_AMOUNT_FIELDS)
        for name in committed_names:
            object.__setattr__(self, name, _int(getattr(self, name), name))
        object.__setattr__(
            self,
            "provider_usage_receipt_ids",
            _strings(
                self.provider_usage_receipt_ids, "provider_usage_receipt_ids", identifiers=True
            ),
        )
        object.__setattr__(
            self,
            "token_measurement_ids",
            _strings(self.token_measurement_ids, "token_measurement_ids", identifiers=True),
        )
        object.__setattr__(self, "status", _enum(self.status, TerminalStatus, "status"))
        if self.committed_strong_model_calls > self.committed_total_model_calls:
            raise AutonomyContractError(
                "committed strong-model calls cannot exceed total model calls"
            )
        if (
            self.committed_input_tokens or self.committed_output_tokens
        ) and not self.token_measurement_ids:
            raise AutonomyContractError("attributed token measurements are required for token use")
        reconciled = [
            item for item in self.reservations if item.status is BudgetReservationStatus.RECONCILED
        ]
        for maximum_name in _BUDGET_AMOUNT_FIELDS:
            suffix = maximum_name[4:]
            attributed = sum(getattr(item, "actual_" + suffix) for item in reconciled)
            if attributed != getattr(self, "committed_" + suffix):
                raise AutonomyContractError(
                    "committed budget use must be exactly attributed to reservations"
                )
        attributed_provider_receipts = {
            receipt_id for item in reconciled for receipt_id in item.provider_usage_receipt_ids
        }
        if attributed_provider_receipts != set(self.provider_usage_receipt_ids):
            raise AutonomyContractError(
                "provider usage receipts must be exactly attributed to reservations"
            )
        attributed_token_measurements = {
            measurement_id for item in reconciled for measurement_id in item.token_measurement_ids
        }
        if attributed_token_measurements != set(self.token_measurement_ids):
            raise AutonomyContractError(
                "token measurements must be exactly attributed to reservations"
            )
        active = [
            item for item in self.reservations if item.status is BudgetReservationStatus.RESERVED
        ]
        observed_overrun = any(
            getattr(item, "actual_" + maximum_name[4:]) > getattr(item, maximum_name)
            for item in reconciled
            for maximum_name in _BUDGET_AMOUNT_FIELDS
        )
        for maximum_name in _BUDGET_AMOUNT_FIELDS:
            suffix = maximum_name[4:]
            committed = getattr(self, "committed_" + suffix)
            reserved = sum(getattr(item, maximum_name) for item in active)
            if committed + reserved > getattr(self.budget, maximum_name):
                observed_overrun = True
                if self.status is not TerminalStatus.EXHAUSTED:
                    raise AutonomyContractError(f"budget ledger exceeds {maximum_name}")
        # A restart value cannot spend a protected reserve under a different
        # purpose.  Observed overruns remain representable, but only as a
        # terminal ledger that cannot admit further work.
        for maximum_name, reserve_name, permitted_purpose in (
            ("max_validation_time_ms", "validation_reserve_ms", BudgetPurpose.VALIDATION),
            ("max_proof_time_ms", "proof_reserve_ms", BudgetPurpose.PROOF),
        ):
            protected = getattr(self.budget, reserve_name)
            if not protected:
                continue
            other_purpose_use = sum(
                (
                    getattr(item, "actual_" + maximum_name[4:])
                    if item.status is BudgetReservationStatus.RECONCILED
                    else getattr(item, maximum_name)
                )
                for item in self.reservations
                if item.status
                in {BudgetReservationStatus.RECONCILED, BudgetReservationStatus.RESERVED}
                and item.purpose is not permitted_purpose
            )
            if other_purpose_use > getattr(self.budget, maximum_name) - protected:
                observed_overrun = True
                if self.status is not TerminalStatus.EXHAUSTED:
                    raise AutonomyContractError(f"budget ledger consumes protected {reserve_name}")
        if observed_overrun and self.status is not TerminalStatus.EXHAUSTED:
            raise AutonomyContractError("budget ledger contains an observed reservation overrun")
        if self.status is TerminalStatus.EXHAUSTED and not observed_overrun:
            raise AutonomyContractError(
                "an exhausted budget ledger must identify an observed overrun"
            )
        self._seal()

    @property
    def ledger_id(self) -> str:
        return self.content_id


@dataclass(frozen=True)
class ExperienceEpisode(_AutonomyContract):
    """Compact outcome record; it intentionally contains identities, not bodies."""

    SCHEMA: ClassVar[str] = _schema("experience-episode")
    IDENTITY_ALIASES: ClassVar[tuple[str, ...]] = ("episode_id",)

    frozen_input_ids: tuple[str, ...]
    question_feature_ids: tuple[str, ...]
    selected_action: MetaAction
    selection_policy_id: str
    selection_policy_version: str
    terminal_status: TerminalStatus
    provider_id: str = ""
    model_id: str = ""
    context_metrics: Mapping[str, Any] = field(default_factory=dict)
    token_measurement_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    accepted_criterion_ids: tuple[str, ...] = ()
    validation_receipt_ids: tuple[str, ...] = ()
    proof_receipt_ids: tuple[str, ...] = ()
    merge_receipt_ids: tuple[str, ...] = ()
    human_intervention_ids: tuple[str, ...] = ()
    failure_signature: str = ""
    repair_signature: str = ""
    counterexample_ids: tuple[str, ...] = ()
    cost_micros: int = 0
    latency_ms: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "frozen_input_ids",
            _strings(self.frozen_input_ids, "frozen_input_ids", identifiers=True, required=True),
        )
        object.__setattr__(
            self,
            "question_feature_ids",
            _strings(
                self.question_feature_ids, "question_feature_ids", identifiers=True, required=True
            ),
        )
        object.__setattr__(
            self, "selected_action", _enum(self.selected_action, MetaAction, "selected_action")
        )
        for name in ("selection_policy_id", "selection_policy_version"):
            object.__setattr__(self, name, _id(getattr(self, name), name))
        object.__setattr__(
            self, "terminal_status", _enum(self.terminal_status, TerminalStatus, "terminal_status")
        )
        object.__setattr__(
            self, "provider_id", _id(self.provider_id, "provider_id", required=False)
        )
        object.__setattr__(self, "model_id", _id(self.model_id, "model_id", required=False))
        object.__setattr__(
            self, "context_metrics", _mapping(self.context_metrics, "context_metrics")
        )
        for name in (
            "token_measurement_ids",
            "evidence_ids",
            "accepted_criterion_ids",
            "validation_receipt_ids",
            "proof_receipt_ids",
            "merge_receipt_ids",
            "human_intervention_ids",
            "counterexample_ids",
        ):
            object.__setattr__(self, name, _strings(getattr(self, name), name, identifiers=True))
        object.__setattr__(
            self,
            "failure_signature",
            _id(self.failure_signature, "failure_signature", required=False),
        )
        object.__setattr__(
            self, "repair_signature", _id(self.repair_signature, "repair_signature", required=False)
        )
        object.__setattr__(self, "cost_micros", _int(self.cost_micros, "cost_micros"))
        object.__setattr__(self, "latency_ms", _int(self.latency_ms, "latency_ms"))
        self._seal()

    @property
    def episode_id(self) -> str:
        return self.content_id


@dataclass(frozen=True)
class CausalAttribution(_AutonomyContract):
    SCHEMA: ClassVar[str] = _schema("causal-attribution")
    IDENTITY_ALIASES: ClassVar[tuple[str, ...]] = ("attribution_id",)

    episode_ids: tuple[str, ...]
    primary_cause: AttributionCause
    evidence_ids: tuple[str, ...]
    confidence_bp: int
    secondary_causes: tuple[AttributionCause, ...] = ()
    controlled_ablation_ids: tuple[str, ...] = ()
    counterexample_ids: tuple[str, ...] = ()
    shadow_only: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "episode_ids",
            _strings(self.episode_ids, "episode_ids", identifiers=True, required=True),
        )
        object.__setattr__(
            self, "primary_cause", _enum(self.primary_cause, AttributionCause, "primary_cause")
        )
        object.__setattr__(
            self,
            "evidence_ids",
            _strings(self.evidence_ids, "evidence_ids", identifiers=True, required=True),
        )
        object.__setattr__(self, "confidence_bp", _bp(self.confidence_bp, "confidence_bp"))
        object.__setattr__(
            self,
            "secondary_causes",
            _enum_tuple(self.secondary_causes, AttributionCause, "secondary_causes"),
        )
        object.__setattr__(
            self,
            "controlled_ablation_ids",
            _strings(self.controlled_ablation_ids, "controlled_ablation_ids", identifiers=True),
        )
        object.__setattr__(
            self,
            "counterexample_ids",
            _strings(self.counterexample_ids, "counterexample_ids", identifiers=True),
        )
        object.__setattr__(self, "shadow_only", _bool(self.shadow_only, "shadow_only"))
        self._seal()

    @property
    def attribution_id(self) -> str:
        return self.content_id


@dataclass(frozen=True)
class PolicyObservation(_AutonomyContract):
    SCHEMA: ClassVar[str] = _schema("policy-observation")
    IDENTITY_ALIASES: ClassVar[tuple[str, ...]] = ("observation_id",)

    episode_id: str
    route_policy_id: str
    selected_action: MetaAction
    selection_reason_codes: tuple[str, ...]
    feature_ids: tuple[str, ...]
    terminal_status: TerminalStatus
    action_propensity_bp: int = 10_000
    accepted_criterion_ids: tuple[str, ...] = ()
    evidence_gain_bp: int = 0
    cost_micros: int = 0
    latency_ms: int = 0
    safety_violation: bool = False

    def __post_init__(self) -> None:
        for name in ("episode_id", "route_policy_id"):
            object.__setattr__(self, name, _id(getattr(self, name), name))
        object.__setattr__(
            self, "selected_action", _enum(self.selected_action, MetaAction, "selected_action")
        )
        object.__setattr__(
            self,
            "selection_reason_codes",
            _strings(
                self.selection_reason_codes,
                "selection_reason_codes",
                identifiers=True,
                required=True,
            ),
        )
        object.__setattr__(
            self,
            "feature_ids",
            _strings(self.feature_ids, "feature_ids", identifiers=True, required=True),
        )
        object.__setattr__(
            self, "terminal_status", _enum(self.terminal_status, TerminalStatus, "terminal_status")
        )
        object.__setattr__(
            self, "action_propensity_bp", _bp(self.action_propensity_bp, "action_propensity_bp")
        )
        if self.action_propensity_bp == 0:
            raise AutonomyContractError("logged action propensity must be positive")
        object.__setattr__(
            self,
            "accepted_criterion_ids",
            _strings(self.accepted_criterion_ids, "accepted_criterion_ids", identifiers=True),
        )
        object.__setattr__(self, "evidence_gain_bp", _bp(self.evidence_gain_bp, "evidence_gain_bp"))
        object.__setattr__(self, "cost_micros", _int(self.cost_micros, "cost_micros"))
        object.__setattr__(self, "latency_ms", _int(self.latency_ms, "latency_ms"))
        object.__setattr__(
            self, "safety_violation", _bool(self.safety_violation, "safety_violation")
        )
        self._seal()

    @property
    def observation_id(self) -> str:
        return self.content_id


@dataclass(frozen=True)
class RoutePolicyCandidate(_AutonomyContract):
    SCHEMA: ClassVar[str] = _schema("route-policy-candidate")
    IDENTITY_ALIASES: ClassVar[tuple[str, ...]] = ("candidate_id", "route_policy_candidate_id")

    parent_policy_id: str
    policy_version: str
    allowed_actions: tuple[MetaAction, ...]
    feature_names: tuple[str, ...]
    integer_weights: Mapping[str, Any]
    training_observation_ids: tuple[str, ...]
    held_out_evaluation_ids: tuple[str, ...]
    safety_gate_receipt_ids: tuple[str, ...]
    selection_reason: str
    shadow_only: bool = True
    external_authorization_id: str = ""

    def __post_init__(self) -> None:
        for name in ("parent_policy_id", "policy_version"):
            object.__setattr__(self, name, _id(getattr(self, name), name))
        object.__setattr__(
            self,
            "allowed_actions",
            _enum_tuple(self.allowed_actions, MetaAction, "allowed_actions", required=True),
        )
        object.__setattr__(
            self,
            "feature_names",
            _strings(self.feature_names, "feature_names", identifiers=True, required=True),
        )
        weights = _mapping(self.integer_weights, "integer_weights")
        normalized_weights: dict[str, int] = {}
        for name, value in weights.items():
            name = _id(name, "integer_weights")
            if name not in self.feature_names:
                raise AutonomyContractError("route-policy weight references an undeclared feature")
            if isinstance(value, bool) or not isinstance(value, int) or abs(value) > MAX_INTEGER:
                raise AutonomyContractError("route-policy weights must be bounded integers")
            normalized_weights[name] = value
        object.__setattr__(
            self, "integer_weights", MappingProxyType(dict(sorted(normalized_weights.items())))
        )
        for name in (
            "training_observation_ids",
            "held_out_evaluation_ids",
            "safety_gate_receipt_ids",
        ):
            object.__setattr__(
                self, name, _strings(getattr(self, name), name, identifiers=True, required=True)
            )
        object.__setattr__(self, "selection_reason", _id(self.selection_reason, "selection_reason"))
        object.__setattr__(self, "shadow_only", _bool(self.shadow_only, "shadow_only"))
        object.__setattr__(
            self,
            "external_authorization_id",
            _id(self.external_authorization_id, "external_authorization_id", required=False),
        )
        if not self.shadow_only:
            raise AutonomyContractError(
                "route-policy candidates are shadow-only until external promotion"
            )
        if self.external_authorization_id:
            raise AutonomyContractError("a candidate policy cannot authorize its own promotion")
        self._seal()

    @property
    def route_policy_candidate_id(self) -> str:
        return self.content_id

    candidate_id = route_policy_candidate_id


@dataclass(frozen=True)
class DistillationCandidate(_AutonomyContract):
    SCHEMA: ClassVar[str] = _schema("distillation-candidate")
    IDENTITY_ALIASES: ClassVar[tuple[str, ...]] = ("candidate_id", "distillation_candidate_id")

    decision_class: str
    episode_ids: tuple[str, ...]
    input_feature_names: tuple[str, ...]
    output_actions: tuple[MetaAction, ...]
    development_example_ids: tuple[str, ...]
    held_out_example_ids: tuple[str, ...]
    proposed_rule_id: str
    status: DistillationStatus = DistillationStatus.CANDIDATE
    counterexample_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision_class", _id(self.decision_class, "decision_class"))
        for name in (
            "episode_ids",
            "input_feature_names",
            "development_example_ids",
            "held_out_example_ids",
        ):
            object.__setattr__(
                self, name, _strings(getattr(self, name), name, identifiers=True, required=True)
            )
        object.__setattr__(
            self,
            "output_actions",
            _enum_tuple(self.output_actions, MetaAction, "output_actions", required=True),
        )
        object.__setattr__(self, "proposed_rule_id", _id(self.proposed_rule_id, "proposed_rule_id"))
        object.__setattr__(self, "status", _enum(self.status, DistillationStatus, "status"))
        object.__setattr__(
            self,
            "counterexample_ids",
            _strings(self.counterexample_ids, "counterexample_ids", identifiers=True),
        )
        self._seal()

    @property
    def distillation_candidate_id(self) -> str:
        return self.content_id

    candidate_id = distillation_candidate_id


_DISTILLED_WHEN_KEYS = frozenset(
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


@dataclass(frozen=True)
class DistilledDecisionRule(_AutonomyContract):
    SCHEMA: ClassVar[str] = _schema("distilled-decision-rule")
    IDENTITY_ALIASES: ClassVar[tuple[str, ...]] = ("rule_id",)

    version: str
    when: Mapping[str, Any]
    action: MetaAction
    required_validation_ids: tuple[str, ...]
    fallback: MetaAction
    scope: Mapping[str, Any]
    source_episode_ids: tuple[str, ...]
    held_out_evaluation_ids: tuple[str, ...]
    counterexample_ids: tuple[str, ...] = ()
    shadow_only: bool = True
    authorized_promotion_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "version", _id(self.version, "version"))
        when = _mapping(self.when, "when")
        if not when or not set(when).issubset(_DISTILLED_WHEN_KEYS):
            raise AutonomyContractError("distilled rule uses unsupported declarative conditions")
        object.__setattr__(self, "when", when)
        object.__setattr__(self, "action", _enum(self.action, MetaAction, "action"))
        object.__setattr__(
            self,
            "required_validation_ids",
            _strings(
                self.required_validation_ids,
                "required_validation_ids",
                identifiers=True,
                required=True,
            ),
        )
        object.__setattr__(self, "fallback", _enum(self.fallback, MetaAction, "fallback"))
        object.__setattr__(self, "scope", _mapping(self.scope, "scope"))
        object.__setattr__(
            self,
            "source_episode_ids",
            _strings(
                self.source_episode_ids, "source_episode_ids", identifiers=True, required=True
            ),
        )
        object.__setattr__(
            self,
            "held_out_evaluation_ids",
            _strings(
                self.held_out_evaluation_ids,
                "held_out_evaluation_ids",
                identifiers=True,
                required=True,
            ),
        )
        object.__setattr__(
            self,
            "counterexample_ids",
            _strings(self.counterexample_ids, "counterexample_ids", identifiers=True),
        )
        object.__setattr__(self, "shadow_only", _bool(self.shadow_only, "shadow_only"))
        object.__setattr__(
            self,
            "authorized_promotion_id",
            _id(self.authorized_promotion_id, "authorized_promotion_id", required=False),
        )
        if not self.shadow_only and not self.authorized_promotion_id:
            raise AutonomyContractError(
                "active distilled rules require external promotion authority"
            )
        self._seal()

    @property
    def rule_id(self) -> str:
        return self.content_id


@dataclass(frozen=True)
class SupervisorSkill(_AutonomyContract):
    SCHEMA: ClassVar[str] = _schema("supervisor-skill")
    IDENTITY_ALIASES: ClassVar[tuple[str, ...]] = ("skill_id",)

    version: str
    precondition_ids: tuple[str, ...]
    input_schema_id: str
    effect_class: str
    steps: tuple[MetaAction, ...]
    postcondition_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    rollback_action_ids: tuple[str, ...]
    fallback: MetaAction
    scope_paths: tuple[str, ...]
    scope_symbols: tuple[str, ...]
    risk_class: RiskClass

    def __post_init__(self) -> None:
        object.__setattr__(self, "version", _id(self.version, "version"))
        object.__setattr__(
            self,
            "precondition_ids",
            _strings(self.precondition_ids, "precondition_ids", identifiers=True, required=True),
        )
        object.__setattr__(self, "input_schema_id", _id(self.input_schema_id, "input_schema_id"))
        object.__setattr__(self, "effect_class", _id(self.effect_class, "effect_class"))
        object.__setattr__(
            self, "steps", _enum_tuple(self.steps, MetaAction, "steps", required=True)
        )
        for name in ("postcondition_ids", "validation_ids", "rollback_action_ids"):
            object.__setattr__(
                self, name, _strings(getattr(self, name), name, identifiers=True, required=True)
            )
        object.__setattr__(self, "fallback", _enum(self.fallback, MetaAction, "fallback"))
        object.__setattr__(self, "scope_paths", _paths(self.scope_paths, "scope_paths"))
        object.__setattr__(self, "scope_symbols", _strings(self.scope_symbols, "scope_symbols"))
        object.__setattr__(self, "risk_class", _enum(self.risk_class, RiskClass, "risk_class"))
        self._seal()

    @property
    def skill_id(self) -> str:
        return self.content_id


@dataclass(frozen=True)
class HumanEscalationPacket(_AutonomyContract):
    SCHEMA: ClassVar[str] = _schema("human-escalation-packet")
    IDENTITY_ALIASES: ClassVar[tuple[str, ...]] = ("packet_id",)

    objective_id: str
    blocked_criterion_ids: tuple[str, ...]
    question: str
    options: tuple[str, ...]
    recommended_option: str
    predicted_consequences: Mapping[str, Any]
    cost_and_risk: Mapping[str, Any]
    evidence_ids: tuple[str, ...]
    continuation_by_option: Mapping[str, Any]
    expires_at_ms: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "objective_id", _id(self.objective_id, "objective_id"))
        object.__setattr__(
            self,
            "blocked_criterion_ids",
            _strings(
                self.blocked_criterion_ids, "blocked_criterion_ids", identifiers=True, required=True
            ),
        )
        object.__setattr__(
            self, "question", _text(self.question, "question", required=True, maximum=2_048)
        )
        object.__setattr__(
            self,
            "options",
            _strings(self.options, "options", required=True, preserve_order=True, maximum=4),
        )
        if len(self.options) < 2:
            raise AutonomyContractError("human escalation requires 2 to 4 bounded options")
        object.__setattr__(
            self,
            "recommended_option",
            _text(self.recommended_option, "recommended_option", required=True),
        )
        if self.recommended_option not in self.options:
            raise AutonomyContractError("recommended option must be one of the bounded options")
        for name in ("predicted_consequences", "cost_and_risk", "continuation_by_option"):
            value = _mapping(getattr(self, name), name)
            if set(value) != set(self.options):
                raise AutonomyContractError(f"{name} must cover every bounded option exactly")
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "evidence_ids",
            _strings(self.evidence_ids, "evidence_ids", identifiers=True, required=True),
        )
        object.__setattr__(self, "expires_at_ms", _int(self.expires_at_ms, "expires_at_ms"))
        self._seal()

    @property
    def packet_id(self) -> str:
        return self.content_id


@dataclass(frozen=True)
class AutonomousRepairPlan(_AutonomyContract):
    SCHEMA: ClassVar[str] = _schema("autonomous-repair-plan")
    IDENTITY_ALIASES: ClassVar[tuple[str, ...]] = ("plan_id",)

    objective_id: str
    task_id: str
    repair_tier: RepairTier
    predicted_files: tuple[str, ...]
    predicted_symbols: tuple[str, ...]
    patch_envelope_id: str
    context_reference_ids: tuple[str, ...]
    required_test_ids: tuple[str, ...]
    required_proof_ids: tuple[str, ...]
    worktree_id: str
    allowed_paths: tuple[str, ...]
    forbidden_symbols: tuple[str, ...]
    rollback_plan_id: str
    risk_class: RiskClass
    max_changed_files: int
    max_changed_lines: int

    def __post_init__(self) -> None:
        for name in (
            "objective_id",
            "task_id",
            "patch_envelope_id",
            "worktree_id",
            "rollback_plan_id",
        ):
            object.__setattr__(self, name, _id(getattr(self, name), name))
        object.__setattr__(self, "repair_tier", _enum(self.repair_tier, RepairTier, "repair_tier"))
        object.__setattr__(
            self, "predicted_files", _paths(self.predicted_files, "predicted_files", required=True)
        )
        object.__setattr__(
            self,
            "predicted_symbols",
            _strings(self.predicted_symbols, "predicted_symbols", required=True),
        )
        for name in ("context_reference_ids", "required_test_ids", "required_proof_ids"):
            object.__setattr__(
                self,
                name,
                _strings(
                    getattr(self, name),
                    name,
                    identifiers=True,
                    required=name == "context_reference_ids",
                ),
            )
        object.__setattr__(
            self, "allowed_paths", _paths(self.allowed_paths, "allowed_paths", required=True)
        )
        object.__setattr__(
            self, "forbidden_symbols", _strings(self.forbidden_symbols, "forbidden_symbols")
        )
        object.__setattr__(self, "risk_class", _enum(self.risk_class, RiskClass, "risk_class"))
        object.__setattr__(
            self,
            "max_changed_files",
            _int(self.max_changed_files, "max_changed_files", maximum=10_000),
        )
        object.__setattr__(
            self,
            "max_changed_lines",
            _int(self.max_changed_lines, "max_changed_lines", maximum=1_000_000),
        )
        if self.max_changed_files == 0 or self.max_changed_lines == 0:
            raise AutonomyContractError("repair plan patch bounds must be positive")
        if len(self.predicted_files) > self.max_changed_files:
            raise AutonomyContractError("predicted repair files exceed the patch envelope")
        allowed = set(self.allowed_paths)
        if not all(
            any(path == prefix or path.startswith(prefix.rstrip("/") + "/") for prefix in allowed)
            for path in self.predicted_files
        ):
            raise AutonomyContractError("predicted repair file escapes allowed paths")
        if self.repair_tier is RepairTier.MODEL_ASSISTED_BOUNDED and not self.worktree_id:
            raise AutonomyContractError("model-assisted repair requires an isolated worktree")
        self._seal()

    @property
    def plan_id(self) -> str:
        return self.content_id


@dataclass(frozen=True)
class AutonomousRepairReceipt(_AutonomyContract):
    SCHEMA: ClassVar[str] = _schema("autonomous-repair-receipt")
    IDENTITY_ALIASES: ClassVar[tuple[str, ...]] = ("receipt_id",)

    plan_id: str
    envelope_id: str
    terminal_status: TerminalStatus
    changed_paths: tuple[str, ...]
    validation_receipt_ids: tuple[str, ...]
    proof_receipt_ids: tuple[str, ...]
    adversarial_assurance_receipt_ids: tuple[str, ...]
    rollback_receipt_id: str = ""
    failure_signature: str = ""
    diagnostic_receipt_id: str = ""
    authorizes_merge: bool = False

    def __post_init__(self) -> None:
        for name in ("plan_id", "envelope_id"):
            object.__setattr__(self, name, _id(getattr(self, name), name))
        object.__setattr__(
            self, "terminal_status", _enum(self.terminal_status, TerminalStatus, "terminal_status")
        )
        object.__setattr__(self, "changed_paths", _paths(self.changed_paths, "changed_paths"))
        for name in (
            "validation_receipt_ids",
            "proof_receipt_ids",
            "adversarial_assurance_receipt_ids",
        ):
            object.__setattr__(self, name, _strings(getattr(self, name), name, identifiers=True))
        object.__setattr__(
            self,
            "rollback_receipt_id",
            _id(self.rollback_receipt_id, "rollback_receipt_id", required=False),
        )
        object.__setattr__(
            self,
            "failure_signature",
            _id(self.failure_signature, "failure_signature", required=False),
        )
        object.__setattr__(
            self,
            "diagnostic_receipt_id",
            _id(self.diagnostic_receipt_id, "diagnostic_receipt_id", required=False),
        )
        object.__setattr__(
            self, "authorizes_merge", _bool(self.authorizes_merge, "authorizes_merge")
        )
        if self.authorizes_merge:
            raise AutonomyContractError(
                "repair receipts are evidence and cannot independently authorize merge"
            )
        if self.terminal_status is TerminalStatus.SUCCEEDED and not self.validation_receipt_ids:
            raise AutonomyContractError("successful repair requires current validation evidence")
        self._seal()

    @property
    def receipt_id(self) -> str:
        return self.content_id


_REQUIRED_SAFETY_GATES = (
    "false_completions",
    "unauthorized_mutations",
    "simulated_as_live",
    "stale_authoritative_cache_hits",
    "confirmation_replays",
    "path_or_scope_escapes",
    "hidden_validation_reductions",
    "escaped_critical_seeded_defects",
    "self_authorized_policy_promotions",
)


def _safety_results(value: Any, name: str) -> Mapping[str, bool]:
    raw = _mapping(value, name)
    if set(raw) != set(_REQUIRED_SAFETY_GATES):
        raise AutonomyContractError(f"{name} must report every non-compensable safety gate")
    result: dict[str, bool] = {}
    for key, item in raw.items():
        result[key] = _bool(item, name)
    return MappingProxyType(dict(sorted(result.items())))


@dataclass(frozen=True)
class AutonomyRunReceipt(_AutonomyContract):
    SCHEMA: ClassVar[str] = _schema("run-receipt")
    IDENTITY_ALIASES: ClassVar[tuple[str, ...]] = ("receipt_id", "run_receipt_id")

    envelope_id: str
    policy_id: str
    graph_id: str
    budget_ledger_id: str
    terminal_status: TerminalStatus
    safety_gate_results: Mapping[str, bool]
    meta_decision_ids: tuple[str, ...] = ()
    action_receipt_ids: tuple[str, ...] = ()
    accepted_criterion_ids: tuple[str, ...] = ()
    unresolved_question_ids: tuple[str, ...] = ()
    token_measurement_ids: tuple[str, ...] = ()
    total_model_calls: int = 0
    strong_model_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    human_interventions: int = 0
    program_id: str = AUTONOMOUS_META_CONTROLLER_PROGRAM_ID
    authorizes_completion: bool = False

    def __post_init__(self) -> None:
        for name in ("envelope_id", "policy_id", "graph_id", "budget_ledger_id"):
            object.__setattr__(self, name, _id(getattr(self, name), name))
        object.__setattr__(
            self, "terminal_status", _enum(self.terminal_status, TerminalStatus, "terminal_status")
        )
        object.__setattr__(
            self,
            "safety_gate_results",
            _safety_results(self.safety_gate_results, "safety_gate_results"),
        )
        for name in (
            "meta_decision_ids",
            "action_receipt_ids",
            "accepted_criterion_ids",
            "unresolved_question_ids",
            "token_measurement_ids",
        ):
            object.__setattr__(self, name, _strings(getattr(self, name), name, identifiers=True))
        for name in (
            "total_model_calls",
            "strong_model_calls",
            "total_input_tokens",
            "total_output_tokens",
            "human_interventions",
        ):
            object.__setattr__(self, name, _int(getattr(self, name), name))
        object.__setattr__(self, "program_id", _id(self.program_id, "program_id"))
        object.__setattr__(
            self,
            "authorizes_completion",
            _bool(self.authorizes_completion, "authorizes_completion"),
        )
        if self.program_id != AUTONOMOUS_META_CONTROLLER_PROGRAM_ID:
            raise AutonomyContractError("run receipt has the wrong program identity")
        if self.strong_model_calls > self.total_model_calls:
            raise AutonomyContractError("strong-model calls cannot exceed total model calls")
        if (self.total_input_tokens or self.total_output_tokens) and not self.token_measurement_ids:
            raise AutonomyContractError("run token use must be attributed")
        if self.authorizes_completion:
            raise AutonomyContractError("run receipts are evidence and cannot authorize completion")
        if self.terminal_status is TerminalStatus.SUCCEEDED:
            if self.unresolved_question_ids or not all(self.safety_gate_results.values()):
                raise AutonomyContractError(
                    "a successful run cannot have unresolved questions or failed safety gates"
                )
        self._seal()

    @property
    def run_receipt_id(self) -> str:
        return self.content_id

    receipt_id = run_receipt_id


@dataclass(frozen=True)
class AutonomyPromotionReceipt(_AutonomyContract):
    SCHEMA: ClassVar[str] = _schema("promotion-receipt")
    IDENTITY_ALIASES: ClassVar[tuple[str, ...]] = ("receipt_id", "promotion_receipt_id")

    candidate_policy_id: str
    expected_old_policy_id: str
    resulting_policy_id: str
    status: PromotionStatus
    safety_gate_results: Mapping[str, bool]
    held_out_evaluation_ids: tuple[str, ...]
    safety_gate_receipt_ids: tuple[str, ...]
    authorization_id: str
    compare_and_swap_receipt_id: str
    rollback_policy_id: str
    blocker_codes: tuple[str, ...] = ()
    self_authorized: bool = False

    def __post_init__(self) -> None:
        for name in (
            "candidate_policy_id",
            "expected_old_policy_id",
            "resulting_policy_id",
            "authorization_id",
            "compare_and_swap_receipt_id",
            "rollback_policy_id",
        ):
            required = name in {
                "candidate_policy_id",
                "expected_old_policy_id",
                "rollback_policy_id",
            }
            object.__setattr__(self, name, _id(getattr(self, name), name, required=required))
        object.__setattr__(self, "status", _enum(self.status, PromotionStatus, "status"))
        object.__setattr__(
            self,
            "safety_gate_results",
            _safety_results(self.safety_gate_results, "safety_gate_results"),
        )
        object.__setattr__(
            self,
            "held_out_evaluation_ids",
            _strings(
                self.held_out_evaluation_ids,
                "held_out_evaluation_ids",
                identifiers=True,
                required=True,
            ),
        )
        object.__setattr__(
            self,
            "safety_gate_receipt_ids",
            _strings(
                self.safety_gate_receipt_ids,
                "safety_gate_receipt_ids",
                identifiers=True,
                required=True,
            ),
        )
        object.__setattr__(
            self, "blocker_codes", _strings(self.blocker_codes, "blocker_codes", identifiers=True)
        )
        object.__setattr__(self, "self_authorized", _bool(self.self_authorized, "self_authorized"))
        if self.self_authorized:
            raise AutonomyContractError("policy candidates cannot authorize their own promotion")
        if self.status is PromotionStatus.PROMOTED:
            if (
                not self.authorization_id
                or not self.compare_and_swap_receipt_id
                or not self.resulting_policy_id
            ):
                raise AutonomyContractError(
                    "promotion requires external authority and expected-old CAS evidence"
                )
            if not all(self.safety_gate_results.values()):
                raise AutonomyContractError("a policy with failed safety gates cannot be promoted")
            if self.blocker_codes:
                raise AutonomyContractError("a promoted policy cannot carry blockers")
        elif not self.blocker_codes and self.status is PromotionStatus.NON_PROMOTED:
            raise AutonomyContractError("non-promotion requires an exact blocker")
        self._seal()

    @property
    def promotion_receipt_id(self) -> str:
        return self.content_id

    receipt_id = promotion_receipt_id


__all__ = [
    "AUTONOMOUS_META_CONTROLLER_PROGRAM_ID",
    "AUTONOMOUS_META_CONTROLLER_ROOT_OBJECTIVE_ID",
    "AUTONOMOUS_META_CONTROLLER_TASK_PREFIX",
    "AttributionCause",
    "AuthorityClass",
    "AutonomyContractError",
    "AutonomyEnvelope",
    "AutonomyLevel",
    "AutonomyPolicy",
    "AutonomyPromotionReceipt",
    "AutonomyRunReceipt",
    "AutonomousRepairPlan",
    "AutonomousRepairReceipt",
    "BeliefFact",
    "BeliefState",
    "BudgetLedger",
    "BudgetDimension",
    "BudgetExhaustion",
    "BudgetExhaustionReason",
    "BudgetPurpose",
    "BudgetReservation",
    "BudgetReservationStatus",
    "CancellationBehavior",
    "CausalAttribution",
    "CognitiveBudget",
    "DecisionGraph",
    "DecisionQuestion",
    "DecisionQuestionType",
    "DistillationCandidate",
    "DistillationStatus",
    "DistilledDecisionRule",
    "EvidenceFreshness",
    "ExperienceEpisode",
    "HumanEscalationPacket",
    "MAX_CANONICAL_RECORD_BYTES",
    "MemoryClass",
    "MetaAction",
    "MetaDecision",
    "MetaDecisionDisposition",
    "PolicyObservation",
    "PrivacyClass",
    "PromotionStatus",
    "QuestionDisposition",
    "RepairTier",
    "ResolutionAction",
    "ResolutionCandidate",
    "ResolutionEvidenceKind",
    "RiskAssessment",
    "RiskClass",
    "RoutePolicyCandidate",
    "SupervisorSkill",
    "TerminalStatus",
]
