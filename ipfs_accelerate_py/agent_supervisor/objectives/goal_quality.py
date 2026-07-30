"""Typed, canonical goal grammar and deterministic quality admission.

The objective heap is intentionally convenient for operators, but free-form
Markdown is not an authority boundary.  This module projects that format into
closed immutable contracts, records every compatibility assumption as debt,
and provides a fail-closed linter for planner/refiner admission.

Quality scores are diagnostic.  A high score never compensates for an error
and no object in this module grants mutation, merge, or completion authority.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..proof.formal_verification_contracts import canonical_json_bytes, content_identity


GOAL_QUALITY_VERSION: Final[int] = 1
GOAL_GRAMMAR_REQUIREMENT_ID: Final[str] = (
    "173651182692809061287627308742826778950"
)

FROZEN_ROOT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/frozen-goal-root@1"
)
GOAL_SCOPE_SCHEMA: Final[str] = "ipfs_accelerate_py/agent-supervisor/goal-scope@1"
ACCEPTANCE_CRITERION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/goal-acceptance-criterion@1"
)
EVIDENCE_PRODUCER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/goal-evidence-producer@1"
)
VALIDATION_RULE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/goal-validation-rule@1"
)
FRESHNESS_POLICY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/goal-freshness-policy@1"
)
RESOURCE_ENVELOPE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/goal-resource-envelope@1"
)
UNCERTAINTY_ITEM_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/goal-uncertainty@1"
)
UNSUPPORTED_SEMANTIC_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/goal-unsupported-semantic@1"
)
REFINEMENT_BUDGET_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/goal-refinement-budget@1"
)
TYPED_GOAL_SCHEMA: Final[str] = "ipfs_accelerate_py/agent-supervisor/typed-goal@1"
GOAL_DEBT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/goal-quality-debt@1"
)
GOAL_QUALITY_REPORT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/goal-quality-report@1"
)
OBJECTIVE_TYPED_GOALS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/objective-typed-goals@1"
)
LEGACY_EVIDENCE_OUTPUT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/legacy-evidence-ref@1"
)
LEGACY_GOAL_REF_OUTPUT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/objective-goal-ref@1"
)
PYTEST_RECEIPT_OUTPUT_SCHEMA: Final[str] = "schema:pytest-receipt@1"
ARTIFACT_RECEIPT_OUTPUT_SCHEMA: Final[str] = "schema:artifact-receipt@1"
# Durable datasets-contract evidence for lossless typed objective admission
# (DSCON-G055 / DSCON-G732).  Kept explicit so validation-repair tasks can
# rebind the producer without inventing completion-gate authority.
DATASETS_CONTRACT_GOAL_QUALITY_EVIDENCE_PATH: Final[str] = (
    "data/datasets_contract_analysis/agent_supervisor/goal-quality.json"
)
DATASETS_CONTRACT_GOAL_QUALITY_TEST_PATH: Final[str] = (
    "ipfs_accelerate_py/test/api/"
    "test_agent_supervisor_datasets_contract_goal_quality.py"
)
DATASETS_CONTRACT_GOAL_QUALITY_VALIDATION_COMMAND: Final[str] = (
    "python -m pytest -q "
    "ipfs_accelerate_py/test/api/test_agent_supervisor_goal_quality.py "
    "ipfs_accelerate_py/test/api/"
    "test_agent_supervisor_datasets_contract_goal_quality.py"
)
DEFAULT_TYPED_FRESHNESS_SECONDS: Final[int] = 3_600
DEFAULT_TYPED_MAX_ROUNDS: Final[int] = 2
DEFAULT_TYPED_MAX_CHILDREN: Final[int] = 4
DEFAULT_TYPED_MAX_DEPTH: Final[int] = 3
DEFAULT_TYPED_MAX_DEBT_ITEMS: Final[int] = 16
DEFAULT_TYPED_MAX_REFINEMENT_TOKENS: Final[int] = 8_192

MAX_TEXT_BYTES: Final[int] = 16_384
MAX_ITEMS: Final[int] = 1_024
MAX_INTEGER: Final[int] = (1 << 63) - 1
MILLION: Final[int] = 1_000_000


class GoalQualityError(ValueError):
    """A goal contract is malformed or unsafe to interpret."""


class GoalAdmissionError(GoalQualityError):
    """A goal has non-compensable quality debt."""

    def __init__(self, report: "GoalQualityReport") -> None:
        self.report = report
        codes = ", ".join(item.code.value for item in report.debt)
        super().__init__(f"goal rejected by quality lint: {codes}")


class EvidenceAuthority(str, Enum):
    """Closed evidence authority vocabulary; it is deliberately not a rank."""

    DIAGNOSTIC = "diagnostic"
    PROPOSAL = "proposal"
    VALIDATION = "validation"
    PROOF = "proof"
    OPERATOR = "operator"
    COMPLETION_GATE = "completion_gate"


class UncertaintyDisposition(str, Enum):
    OPEN = "open"
    MITIGATED = "mitigated"
    ACCEPTED_RISK = "accepted_risk"
    BLOCKING = "blocking"


class DebtSeverity(str, Enum):
    WARNING = "warning"
    ERROR = "error"


class GoalDebtCode(str, Enum):
    MISSING_OUTCOME = "missing_outcome"
    MISSING_SCOPE = "missing_scope"
    MISSING_ASSUMPTIONS = "missing_assumptions"
    MISSING_NON_GOALS = "missing_non_goals"
    MISSING_ACCEPTANCE = "missing_acceptance"
    MISSING_EVIDENCE_PRODUCER = "missing_evidence_producer"
    MISSING_VALIDATION = "missing_validation"
    MISSING_FRESHNESS = "missing_freshness"
    MISSING_RESOURCE_ENVELOPE = "missing_resource_envelope"
    MISSING_UNCERTAINTY = "missing_uncertainty"
    MISSING_UNSUPPORTED_SEMANTICS = "missing_unsupported_semantics"
    MISSING_REFINEMENT_BUDGET = "missing_refinement_budget"
    UNCERTAINTY_DEBT = "uncertainty_debt"
    UNSUPPORTED_SEMANTICS = "unsupported_semantics"
    CIRCULAR_ACCEPTANCE = "circular_acceptance"
    UNBOUNDED_SCOPE = "unbounded_scope"
    CONFLICTING_SCOPE = "conflicting_scope"
    HIDDEN_AUTHORITY = "hidden_authority"
    UNVERIFIABLE_EVIDENCE = "unverifiable_evidence"
    ORPHAN_DEPENDENCY = "orphan_dependency"
    AMBIGUOUS_COMPLETION = "ambiguous_completion"
    EXCESSIVE_BREADTH = "excessive_breadth"


class RepairKind(str, Enum):
    ADD_FIELD = "add_field"
    BOUND_SCOPE = "bound_scope"
    REMOVE_CONFLICT = "remove_conflict"
    BREAK_CYCLE = "break_cycle"
    DECLARE_AUTHORITY = "declare_authority"
    BIND_EVIDENCE = "bind_evidence"
    RESOLVE_DEPENDENCY = "resolve_dependency"
    DEFINE_COMPLETION = "define_completion"
    SPLIT_GOAL = "split_goal"


def _text(
    value: Any,
    name: str,
    *,
    required: bool = False,
    max_bytes: int = MAX_TEXT_BYTES,
) -> str:
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise GoalQualityError(f"{name} must be a string")
    result = " ".join(value.split())
    if required and not result:
        raise GoalQualityError(f"{name} is required")
    if "\x00" in result:
        raise GoalQualityError(f"{name} must not contain NUL")
    if len(result.encode("utf-8")) > max_bytes:
        raise GoalQualityError(f"{name} exceeds {max_bytes} UTF-8 bytes")
    return result


def _strings(
    value: Iterable[Any] | None,
    name: str,
    *,
    required: bool = False,
) -> tuple[str, ...]:
    if value is None:
        result: tuple[str, ...] = ()
    elif isinstance(value, (str, bytes, bytearray, memoryview)):
        raise GoalQualityError(f"{name} must be a sequence of strings")
    else:
        normalized = {_text(item, name, required=True) for item in value}
        if len(normalized) > MAX_ITEMS:
            raise GoalQualityError(f"{name} exceeds {MAX_ITEMS} items")
        result = tuple(sorted(normalized, key=lambda item: (item.casefold(), item)))
    if required and not result:
        raise GoalQualityError(f"{name} must not be empty")
    return result


def _records(
    value: Iterable[Any] | None,
    record_type: type[Any],
    name: str,
) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray, memoryview, Mapping)):
        raise GoalQualityError(f"{name} must be a sequence")
    records = tuple(value)
    if len(records) > MAX_ITEMS:
        raise GoalQualityError(f"{name} exceeds {MAX_ITEMS} items")
    if any(not isinstance(item, record_type) for item in records):
        raise GoalQualityError(f"{name} must contain {record_type.__name__} values")
    identifiers = [getattr(item, "item_id", getattr(item, "criterion_id", "")) for item in records]
    if len(identifiers) != len(set(identifiers)):
        raise GoalQualityError(f"{name} contains duplicate identities")
    return tuple(
        sorted(
            records,
            key=lambda item: getattr(
                item, "item_id", getattr(item, "criterion_id", "")
            ),
        )
    )


def _integer(
    value: Any,
    name: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise GoalQualityError(f"{name} must be an integer")
    effective_maximum = MAX_INTEGER if maximum is None else min(maximum, MAX_INTEGER)
    if value < minimum or value > effective_maximum:
        bounds = f"between {minimum} and {effective_maximum}"
        raise GoalQualityError(f"{name} must be {bounds}")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise GoalQualityError(f"{name} must be a boolean")
    return value


def _enum(value: Any, enum_type: type[Enum], name: str) -> Any:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value))
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise GoalQualityError(f"{name} must be one of: {allowed}") from exc


def _closed(
    payload: Mapping[str, Any],
    *,
    schema: str,
    fields: Iterable[str],
    noun: str,
) -> None:
    if not isinstance(payload, Mapping):
        raise GoalQualityError(f"{noun} must be an object")
    if payload.get("schema") != schema:
        raise GoalQualityError(f"{noun} has an unsupported schema")
    if payload.get("version") != GOAL_QUALITY_VERSION:
        raise GoalQualityError(f"{noun} has an unsupported version")
    allowed = set(fields) | {"schema", "version", "content_id"}
    if set(payload).difference(allowed):
        raise GoalQualityError(f"{noun} contains unsupported fields")


class _GoalContract:
    SCHEMA: ClassVar[str] = ""

    def _payload(self) -> dict[str, Any]:
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "version": GOAL_QUALITY_VERSION,
            **self._payload(),
        }

    def to_json(self) -> str:
        return canonical_json_bytes(self.to_dict()).decode("utf-8")

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())

    @property
    def identity(self) -> str:
        return self.content_id

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "content_id": self.content_id}

    @classmethod
    def from_json(cls, payload: str) -> Any:
        try:
            value = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise GoalQualityError("goal contract JSON is malformed") from exc
        return cls.from_dict(value)

    @classmethod
    def _verify_claim(cls, payload: Mapping[str, Any], result: "_GoalContract") -> None:
        claimed = payload.get("content_id")
        if claimed not in (None, result.content_id):
            raise GoalQualityError("goal contract content identity does not match")


@dataclass(frozen=True)
class FrozenRootIdentity(_GoalContract):
    """Immutable root identity inherited by every refinement descendant."""

    SCHEMA: ClassVar[str] = FROZEN_ROOT_SCHEMA

    goal_id: str
    revision: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "goal_id", _text(self.goal_id, "goal_id", required=True))
        object.__setattr__(self, "revision", _text(self.revision, "revision", required=True))

    def _payload(self) -> dict[str, Any]:
        return {"goal_id": self.goal_id, "revision": self.revision}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FrozenRootIdentity":
        _closed(payload, schema=cls.SCHEMA, fields=("goal_id", "revision"), noun="frozen root")
        result = cls(goal_id=payload.get("goal_id"), revision=payload.get("revision"))
        cls._verify_claim(payload, result)
        return result


@dataclass(frozen=True)
class GoalScope(_GoalContract):
    """Finite included/excluded subjects and declared external dependencies."""

    SCHEMA: ClassVar[str] = GOAL_SCOPE_SCHEMA

    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    dependency_goal_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("include", "exclude", "dependency_goal_ids"):
            object.__setattr__(self, name, _strings(getattr(self, name), name))

    @property
    def item_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "include": self.include,
            "exclude": self.exclude,
            "dependency_goal_ids": self.dependency_goal_ids,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GoalScope":
        fields = ("include", "exclude", "dependency_goal_ids")
        _closed(payload, schema=cls.SCHEMA, fields=fields, noun="goal scope")
        result = cls(**{name: payload.get(name) or () for name in fields})
        cls._verify_claim(payload, result)
        return result


@dataclass(frozen=True)
class AcceptanceCriterion(_GoalContract):
    SCHEMA: ClassVar[str] = ACCEPTANCE_CRITERION_SCHEMA

    criterion_id: str
    statement: str
    evidence_producer_ids: tuple[str, ...] = ()
    validation_rule_ids: tuple[str, ...] = ()
    depends_on_criterion_ids: tuple[str, ...] = ()
    completion_signal: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "criterion_id", _text(self.criterion_id, "criterion_id", required=True)
        )
        object.__setattr__(self, "statement", _text(self.statement, "statement"))
        object.__setattr__(
            self, "completion_signal", _text(self.completion_signal, "completion_signal")
        )
        for name in (
            "evidence_producer_ids",
            "validation_rule_ids",
            "depends_on_criterion_ids",
        ):
            object.__setattr__(self, name, _strings(getattr(self, name), name))

    @property
    def item_id(self) -> str:
        return self.criterion_id

    def _payload(self) -> dict[str, Any]:
        return {
            "criterion_id": self.criterion_id,
            "statement": self.statement,
            "evidence_producer_ids": self.evidence_producer_ids,
            "validation_rule_ids": self.validation_rule_ids,
            "depends_on_criterion_ids": self.depends_on_criterion_ids,
            "completion_signal": self.completion_signal,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AcceptanceCriterion":
        fields = (
            "criterion_id",
            "statement",
            "evidence_producer_ids",
            "validation_rule_ids",
            "depends_on_criterion_ids",
            "completion_signal",
        )
        _closed(payload, schema=cls.SCHEMA, fields=fields, noun="acceptance criterion")
        result = cls(
            criterion_id=payload.get("criterion_id"),
            statement=payload.get("statement"),
            evidence_producer_ids=payload.get("evidence_producer_ids") or (),
            validation_rule_ids=payload.get("validation_rule_ids") or (),
            depends_on_criterion_ids=payload.get("depends_on_criterion_ids") or (),
            completion_signal=payload.get("completion_signal") or "",
        )
        cls._verify_claim(payload, result)
        return result


@dataclass(frozen=True)
class EvidenceProducer(_GoalContract):
    SCHEMA: ClassVar[str] = EVIDENCE_PRODUCER_SCHEMA

    producer_id: str
    kind: str
    output_schema: str
    authority: EvidenceAuthority = EvidenceAuthority.DIAGNOSTIC
    capability_id: str = ""
    independent: bool = False

    def __post_init__(self) -> None:
        for name in ("producer_id", "kind", "output_schema"):
            object.__setattr__(
                self, name, _text(getattr(self, name), name, required=(name == "producer_id"))
            )
        object.__setattr__(
            self, "capability_id", _text(self.capability_id, "capability_id")
        )
        object.__setattr__(
            self, "authority", _enum(self.authority, EvidenceAuthority, "authority")
        )
        object.__setattr__(self, "independent", _boolean(self.independent, "independent"))

    @property
    def item_id(self) -> str:
        return self.producer_id

    def _payload(self) -> dict[str, Any]:
        return {
            "producer_id": self.producer_id,
            "kind": self.kind,
            "output_schema": self.output_schema,
            "authority": self.authority.value,
            "capability_id": self.capability_id,
            "independent": self.independent,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvidenceProducer":
        fields = (
            "producer_id",
            "kind",
            "output_schema",
            "authority",
            "capability_id",
            "independent",
        )
        _closed(payload, schema=cls.SCHEMA, fields=fields, noun="evidence producer")
        result = cls(
            producer_id=payload.get("producer_id"),
            kind=payload.get("kind") or "",
            output_schema=payload.get("output_schema") or "",
            authority=payload.get("authority") or EvidenceAuthority.DIAGNOSTIC,
            capability_id=payload.get("capability_id") or "",
            independent=payload.get("independent", False),
        )
        cls._verify_claim(payload, result)
        return result


@dataclass(frozen=True)
class ValidationRule(_GoalContract):
    SCHEMA: ClassVar[str] = VALIDATION_RULE_SCHEMA

    rule_id: str
    command: str
    producer_id: str
    criterion_ids: tuple[str, ...] = ()
    hermetic: bool = False

    def __post_init__(self) -> None:
        for name in ("rule_id", "command", "producer_id"):
            object.__setattr__(
                self, name, _text(getattr(self, name), name, required=(name == "rule_id"))
            )
        object.__setattr__(
            self, "criterion_ids", _strings(self.criterion_ids, "criterion_ids")
        )
        object.__setattr__(self, "hermetic", _boolean(self.hermetic, "hermetic"))

    @property
    def item_id(self) -> str:
        return self.rule_id

    def _payload(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "command": self.command,
            "producer_id": self.producer_id,
            "criterion_ids": self.criterion_ids,
            "hermetic": self.hermetic,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ValidationRule":
        fields = ("rule_id", "command", "producer_id", "criterion_ids", "hermetic")
        _closed(payload, schema=cls.SCHEMA, fields=fields, noun="validation rule")
        result = cls(
            rule_id=payload.get("rule_id"),
            command=payload.get("command") or "",
            producer_id=payload.get("producer_id") or "",
            criterion_ids=payload.get("criterion_ids") or (),
            hermetic=payload.get("hermetic", False),
        )
        cls._verify_claim(payload, result)
        return result


@dataclass(frozen=True)
class FreshnessPolicy(_GoalContract):
    SCHEMA: ClassVar[str] = FRESHNESS_POLICY_SCHEMA

    max_age_seconds: int = 0
    require_repository_revision: bool = True
    require_tree_revision: bool = True
    require_semantic_dependencies: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "max_age_seconds", _integer(self.max_age_seconds, "max_age_seconds")
        )
        for name in (
            "require_repository_revision",
            "require_tree_revision",
            "require_semantic_dependencies",
        ):
            object.__setattr__(self, name, _boolean(getattr(self, name), name))

    def _payload(self) -> dict[str, Any]:
        return {
            "max_age_seconds": self.max_age_seconds,
            "require_repository_revision": self.require_repository_revision,
            "require_tree_revision": self.require_tree_revision,
            "require_semantic_dependencies": self.require_semantic_dependencies,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FreshnessPolicy":
        fields = (
            "max_age_seconds",
            "require_repository_revision",
            "require_tree_revision",
            "require_semantic_dependencies",
        )
        _closed(payload, schema=cls.SCHEMA, fields=fields, noun="freshness policy")
        result = cls(
            max_age_seconds=payload.get("max_age_seconds", 0),
            require_repository_revision=payload.get("require_repository_revision", True),
            require_tree_revision=payload.get("require_tree_revision", True),
            require_semantic_dependencies=payload.get(
                "require_semantic_dependencies", True
            ),
        )
        cls._verify_claim(payload, result)
        return result


@dataclass(frozen=True)
class ResourceEnvelope(_GoalContract):
    SCHEMA: ClassVar[str] = RESOURCE_ENVELOPE_SCHEMA

    max_wall_seconds: int = 0
    max_tokens: int = 0
    max_cost_microunits: int = 0
    max_artifacts: int = 0
    max_parallelism: int = 0
    max_scope_items: int = 0

    def __post_init__(self) -> None:
        for name in (
            "max_wall_seconds",
            "max_tokens",
            "max_cost_microunits",
            "max_artifacts",
            "max_parallelism",
            "max_scope_items",
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name))

    @property
    def bounded(self) -> bool:
        return all(
            getattr(self, name) > 0
            for name in (
                "max_wall_seconds",
                "max_tokens",
                "max_artifacts",
                "max_parallelism",
                "max_scope_items",
            )
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "max_wall_seconds": self.max_wall_seconds,
            "max_tokens": self.max_tokens,
            "max_cost_microunits": self.max_cost_microunits,
            "max_artifacts": self.max_artifacts,
            "max_parallelism": self.max_parallelism,
            "max_scope_items": self.max_scope_items,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ResourceEnvelope":
        fields = (
            "max_wall_seconds",
            "max_tokens",
            "max_cost_microunits",
            "max_artifacts",
            "max_parallelism",
            "max_scope_items",
        )
        _closed(payload, schema=cls.SCHEMA, fields=fields, noun="resource envelope")
        result = cls(**{name: payload.get(name, 0) for name in fields})
        cls._verify_claim(payload, result)
        return result


@dataclass(frozen=True)
class UncertaintyItem(_GoalContract):
    SCHEMA: ClassVar[str] = UNCERTAINTY_ITEM_SCHEMA

    uncertainty_id: str
    statement: str
    disposition: UncertaintyDisposition = UncertaintyDisposition.OPEN
    impact: str = ""
    resolution: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "uncertainty_id",
            _text(self.uncertainty_id, "uncertainty_id", required=True),
        )
        for name in ("statement", "impact", "resolution"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, UncertaintyDisposition, "disposition"),
        )

    @property
    def item_id(self) -> str:
        return self.uncertainty_id

    def _payload(self) -> dict[str, Any]:
        return {
            "uncertainty_id": self.uncertainty_id,
            "statement": self.statement,
            "disposition": self.disposition.value,
            "impact": self.impact,
            "resolution": self.resolution,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "UncertaintyItem":
        fields = ("uncertainty_id", "statement", "disposition", "impact", "resolution")
        _closed(payload, schema=cls.SCHEMA, fields=fields, noun="uncertainty item")
        result = cls(
            uncertainty_id=payload.get("uncertainty_id"),
            statement=payload.get("statement") or "",
            disposition=payload.get("disposition") or UncertaintyDisposition.OPEN,
            impact=payload.get("impact") or "",
            resolution=payload.get("resolution") or "",
        )
        cls._verify_claim(payload, result)
        return result


@dataclass(frozen=True)
class UnsupportedSemantic(_GoalContract):
    SCHEMA: ClassVar[str] = UNSUPPORTED_SEMANTIC_SCHEMA

    semantic_id: str
    statement: str
    fallback: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "semantic_id", _text(self.semantic_id, "semantic_id", required=True)
        )
        object.__setattr__(self, "statement", _text(self.statement, "statement"))
        object.__setattr__(self, "fallback", _text(self.fallback, "fallback"))

    @property
    def item_id(self) -> str:
        return self.semantic_id

    def _payload(self) -> dict[str, Any]:
        return {
            "semantic_id": self.semantic_id,
            "statement": self.statement,
            "fallback": self.fallback,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "UnsupportedSemantic":
        fields = ("semantic_id", "statement", "fallback")
        _closed(payload, schema=cls.SCHEMA, fields=fields, noun="unsupported semantic")
        result = cls(
            semantic_id=payload.get("semantic_id"),
            statement=payload.get("statement") or "",
            fallback=payload.get("fallback") or "",
        )
        cls._verify_claim(payload, result)
        return result


@dataclass(frozen=True)
class RefinementBudget(_GoalContract):
    SCHEMA: ClassVar[str] = REFINEMENT_BUDGET_SCHEMA

    max_rounds: int = 0
    max_children: int = 0
    max_depth: int = 0
    max_debt_items: int = 0
    max_tokens: int = 0

    def __post_init__(self) -> None:
        for name in (
            "max_rounds",
            "max_children",
            "max_depth",
            "max_debt_items",
            "max_tokens",
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name))

    @property
    def bounded(self) -> bool:
        return all(getattr(self, name) > 0 for name in self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "max_rounds": self.max_rounds,
            "max_children": self.max_children,
            "max_depth": self.max_depth,
            "max_debt_items": self.max_debt_items,
            "max_tokens": self.max_tokens,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RefinementBudget":
        fields = ("max_rounds", "max_children", "max_depth", "max_debt_items", "max_tokens")
        _closed(payload, schema=cls.SCHEMA, fields=fields, noun="refinement budget")
        result = cls(**{name: payload.get(name, 0) for name in fields})
        cls._verify_claim(payload, result)
        return result


@dataclass(frozen=True)
class TypedGoal(_GoalContract):
    """One complete goal grammar value.

    Empty fields are representable so a compatibility projection can be
    linted and repaired.  Admission is exclusively the responsibility of
    :func:`validate_goal`.
    """

    SCHEMA: ClassVar[str] = TYPED_GOAL_SCHEMA

    goal_id: str
    root: FrozenRootIdentity
    outcome: str = ""
    scope: GoalScope = field(default_factory=GoalScope)
    assumptions: tuple[str, ...] = ()
    non_goals: tuple[str, ...] = ()
    acceptance_criteria: tuple[AcceptanceCriterion, ...] = ()
    evidence_producers: tuple[EvidenceProducer, ...] = ()
    validation_rules: tuple[ValidationRule, ...] = ()
    freshness: FreshnessPolicy = field(default_factory=FreshnessPolicy)
    resources: ResourceEnvelope = field(default_factory=ResourceEnvelope)
    uncertainties: tuple[UncertaintyItem, ...] = ()
    unsupported_semantics: tuple[UnsupportedSemantic, ...] = ()
    refinement_budget: RefinementBudget = field(default_factory=RefinementBudget)
    authorized_completion_producer_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "goal_id", _text(self.goal_id, "goal_id", required=True))
        if not isinstance(self.root, FrozenRootIdentity):
            raise GoalQualityError("root must be a FrozenRootIdentity")
        object.__setattr__(self, "outcome", _text(self.outcome, "outcome"))
        if not isinstance(self.scope, GoalScope):
            raise GoalQualityError("scope must be a GoalScope")
        for name in ("assumptions", "non_goals", "authorized_completion_producer_ids"):
            object.__setattr__(self, name, _strings(getattr(self, name), name))
        for name, record_type in (
            ("acceptance_criteria", AcceptanceCriterion),
            ("evidence_producers", EvidenceProducer),
            ("validation_rules", ValidationRule),
            ("uncertainties", UncertaintyItem),
            ("unsupported_semantics", UnsupportedSemantic),
        ):
            object.__setattr__(
                self, name, _records(getattr(self, name), record_type, name)
            )
        for name, record_type in (
            ("freshness", FreshnessPolicy),
            ("resources", ResourceEnvelope),
            ("refinement_budget", RefinementBudget),
        ):
            if not isinstance(getattr(self, name), record_type):
                raise GoalQualityError(f"{name} must be a {record_type.__name__}")

    @property
    def root_identity(self) -> FrozenRootIdentity:
        return self.root

    @property
    def item_id(self) -> str:
        return self.goal_id

    def _payload(self) -> dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "root": self.root.to_dict(),
            "outcome": self.outcome,
            "scope": self.scope.to_dict(),
            "assumptions": self.assumptions,
            "non_goals": self.non_goals,
            "acceptance_criteria": tuple(item.to_dict() for item in self.acceptance_criteria),
            "evidence_producers": tuple(item.to_dict() for item in self.evidence_producers),
            "validation_rules": tuple(item.to_dict() for item in self.validation_rules),
            "freshness": self.freshness.to_dict(),
            "resources": self.resources.to_dict(),
            "uncertainties": tuple(item.to_dict() for item in self.uncertainties),
            "unsupported_semantics": tuple(
                item.to_dict() for item in self.unsupported_semantics
            ),
            "refinement_budget": self.refinement_budget.to_dict(),
            "authorized_completion_producer_ids": (
                self.authorized_completion_producer_ids
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TypedGoal":
        fields = (
            "goal_id",
            "root",
            "outcome",
            "scope",
            "assumptions",
            "non_goals",
            "acceptance_criteria",
            "evidence_producers",
            "validation_rules",
            "freshness",
            "resources",
            "uncertainties",
            "unsupported_semantics",
            "refinement_budget",
            "authorized_completion_producer_ids",
        )
        _closed(payload, schema=cls.SCHEMA, fields=fields, noun="typed goal")

        def nested(name: str, record_type: type[Any]) -> Any:
            raw = payload.get(name)
            if not isinstance(raw, Mapping):
                raise GoalQualityError(f"typed goal {name} must be an object")
            return record_type.from_dict(raw)

        def nested_many(name: str, record_type: type[Any]) -> tuple[Any, ...]:
            raw = payload.get(name) or ()
            if isinstance(raw, (str, bytes, bytearray, memoryview, Mapping)):
                raise GoalQualityError(f"typed goal {name} must be a sequence")
            return tuple(record_type.from_dict(item) for item in raw)

        result = cls(
            goal_id=payload.get("goal_id"),
            root=nested("root", FrozenRootIdentity),
            outcome=payload.get("outcome") or "",
            scope=nested("scope", GoalScope),
            assumptions=payload.get("assumptions") or (),
            non_goals=payload.get("non_goals") or (),
            acceptance_criteria=nested_many(
                "acceptance_criteria", AcceptanceCriterion
            ),
            evidence_producers=nested_many("evidence_producers", EvidenceProducer),
            validation_rules=nested_many("validation_rules", ValidationRule),
            freshness=nested("freshness", FreshnessPolicy),
            resources=nested("resources", ResourceEnvelope),
            uncertainties=nested_many("uncertainties", UncertaintyItem),
            unsupported_semantics=nested_many(
                "unsupported_semantics", UnsupportedSemantic
            ),
            refinement_budget=nested("refinement_budget", RefinementBudget),
            authorized_completion_producer_ids=payload.get(
                "authorized_completion_producer_ids"
            )
            or (),
        )
        cls._verify_claim(payload, result)
        return result


@dataclass(frozen=True)
class GoalQualityPolicy(_GoalContract):
    SCHEMA: ClassVar[str] = (
        "ipfs_accelerate_py/agent-supervisor/goal-quality-policy@1"
    )

    max_scope_items: int = 32
    max_acceptance_criteria: int = 16
    max_dependencies: int = 16
    max_total_breadth: int = 64
    ambiguous_terms: tuple[str, ...] = (
        "appropriate",
        "as needed",
        "etc",
        "good",
        "high quality",
        "reasonable",
        "sufficient",
    )

    def __post_init__(self) -> None:
        for name in (
            "max_scope_items",
            "max_acceptance_criteria",
            "max_dependencies",
            "max_total_breadth",
        ):
            object.__setattr__(
                self, name, _integer(getattr(self, name), name, minimum=1)
            )
        object.__setattr__(
            self, "ambiguous_terms", _strings(self.ambiguous_terms, "ambiguous_terms")
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "max_scope_items": self.max_scope_items,
            "max_acceptance_criteria": self.max_acceptance_criteria,
            "max_dependencies": self.max_dependencies,
            "max_total_breadth": self.max_total_breadth,
            "ambiguous_terms": self.ambiguous_terms,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GoalQualityPolicy":
        fields = (
            "max_scope_items",
            "max_acceptance_criteria",
            "max_dependencies",
            "max_total_breadth",
            "ambiguous_terms",
        )
        _closed(payload, schema=cls.SCHEMA, fields=fields, noun="goal quality policy")
        result = cls(
            max_scope_items=payload.get("max_scope_items", 32),
            max_acceptance_criteria=payload.get("max_acceptance_criteria", 16),
            max_dependencies=payload.get("max_dependencies", 16),
            max_total_breadth=payload.get("max_total_breadth", 64),
            ambiguous_terms=payload.get("ambiguous_terms") or (),
        )
        cls._verify_claim(payload, result)
        return result


_DEBT_DEFAULTS: Final[Mapping[GoalDebtCode, tuple[DebtSeverity, RepairKind, str]]] = (
    MappingProxyType(
        {
            GoalDebtCode.MISSING_OUTCOME: (
                DebtSeverity.ERROR,
                RepairKind.ADD_FIELD,
                "Declare one observable outcome.",
            ),
            GoalDebtCode.MISSING_SCOPE: (
                DebtSeverity.ERROR,
                RepairKind.BOUND_SCOPE,
                "List the exact included subjects.",
            ),
            GoalDebtCode.MISSING_ASSUMPTIONS: (
                DebtSeverity.ERROR,
                RepairKind.ADD_FIELD,
                "Declare assumptions explicitly, using an explicit none marker if empty.",
            ),
            GoalDebtCode.MISSING_NON_GOALS: (
                DebtSeverity.WARNING,
                RepairKind.ADD_FIELD,
                "Declare explicit non-goals to bound interpretation.",
            ),
            GoalDebtCode.MISSING_ACCEPTANCE: (
                DebtSeverity.ERROR,
                RepairKind.ADD_FIELD,
                "Add observable acceptance criteria.",
            ),
            GoalDebtCode.MISSING_EVIDENCE_PRODUCER: (
                DebtSeverity.ERROR,
                RepairKind.BIND_EVIDENCE,
                "Bind every criterion to a typed evidence producer.",
            ),
            GoalDebtCode.MISSING_VALIDATION: (
                DebtSeverity.ERROR,
                RepairKind.BIND_EVIDENCE,
                "Bind every criterion to an executable validation rule.",
            ),
            GoalDebtCode.MISSING_FRESHNESS: (
                DebtSeverity.ERROR,
                RepairKind.ADD_FIELD,
                "Set a finite freshness horizon and revision bindings.",
            ),
            GoalDebtCode.MISSING_RESOURCE_ENVELOPE: (
                DebtSeverity.ERROR,
                RepairKind.ADD_FIELD,
                "Set finite time, token, artifact, parallelism, and scope limits.",
            ),
            GoalDebtCode.MISSING_UNCERTAINTY: (
                DebtSeverity.WARNING,
                RepairKind.ADD_FIELD,
                "Record uncertainty or an explicit reviewed-none item.",
            ),
            GoalDebtCode.MISSING_UNSUPPORTED_SEMANTICS: (
                DebtSeverity.WARNING,
                RepairKind.ADD_FIELD,
                "Record unsupported semantics or an explicit reviewed-none item.",
            ),
            GoalDebtCode.MISSING_REFINEMENT_BUDGET: (
                DebtSeverity.ERROR,
                RepairKind.ADD_FIELD,
                "Set finite rounds, depth, breadth, debt, and token limits.",
            ),
            GoalDebtCode.UNCERTAINTY_DEBT: (
                DebtSeverity.WARNING,
                RepairKind.ADD_FIELD,
                "Resolve, mitigate, or explicitly accept the uncertainty.",
            ),
            GoalDebtCode.UNSUPPORTED_SEMANTICS: (
                DebtSeverity.ERROR,
                RepairKind.ADD_FIELD,
                "Provide a conservative supported fallback or remove the semantic.",
            ),
            GoalDebtCode.CIRCULAR_ACCEPTANCE: (
                DebtSeverity.ERROR,
                RepairKind.BREAK_CYCLE,
                "Remove the indicated acceptance dependency edge.",
            ),
            GoalDebtCode.UNBOUNDED_SCOPE: (
                DebtSeverity.ERROR,
                RepairKind.BOUND_SCOPE,
                "Replace wildcard or repository-wide scope with finite subjects.",
            ),
            GoalDebtCode.CONFLICTING_SCOPE: (
                DebtSeverity.ERROR,
                RepairKind.REMOVE_CONFLICT,
                "Remove the subject from either include or exclude.",
            ),
            GoalDebtCode.HIDDEN_AUTHORITY: (
                DebtSeverity.ERROR,
                RepairKind.DECLARE_AUTHORITY,
                "Use an explicit reviewed completion gate; producers cannot self-authorize.",
            ),
            GoalDebtCode.UNVERIFIABLE_EVIDENCE: (
                DebtSeverity.ERROR,
                RepairKind.BIND_EVIDENCE,
                "Provide a typed producer, output schema, command, and criterion mapping.",
            ),
            GoalDebtCode.ORPHAN_DEPENDENCY: (
                DebtSeverity.ERROR,
                RepairKind.RESOLVE_DEPENDENCY,
                "Declare the referenced object or remove the reference.",
            ),
            GoalDebtCode.AMBIGUOUS_COMPLETION: (
                DebtSeverity.ERROR,
                RepairKind.DEFINE_COMPLETION,
                "Replace vague language and add an exact completion signal.",
            ),
            GoalDebtCode.EXCESSIVE_BREADTH: (
                DebtSeverity.ERROR,
                RepairKind.SPLIT_GOAL,
                "Split the goal into dependency-linked bounded children.",
            ),
        }
    )
)


@dataclass(frozen=True)
class GoalDebt(_GoalContract):
    """A stable, machine-repairable quality finding."""

    SCHEMA: ClassVar[str] = GOAL_DEBT_SCHEMA

    code: GoalDebtCode
    severity: DebtSeverity
    path: str
    related_ids: tuple[str, ...]
    message: str
    repair_kind: RepairKind
    repair: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _enum(self.code, GoalDebtCode, "code"))
        object.__setattr__(
            self, "severity", _enum(self.severity, DebtSeverity, "severity")
        )
        object.__setattr__(
            self, "repair_kind", _enum(self.repair_kind, RepairKind, "repair_kind")
        )
        for name in ("path", "message", "repair"):
            object.__setattr__(
                self, name, _text(getattr(self, name), name, required=True)
            )
        object.__setattr__(
            self, "related_ids", _strings(self.related_ids, "related_ids")
        )
        default_severity, default_repair_kind, default_repair = _DEBT_DEFAULTS[
            self.code
        ]
        allowed_severities = (
            {DebtSeverity.WARNING, DebtSeverity.ERROR}
            if self.code is GoalDebtCode.UNCERTAINTY_DEBT
            else {default_severity}
        )
        if self.severity not in allowed_severities:
            raise GoalQualityError("goal debt severity does not match its code")
        if (
            self.repair_kind is not default_repair_kind
            or self.repair != default_repair
        ):
            raise GoalQualityError("goal debt repair does not match its code")

    @property
    def item_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "severity": self.severity.value,
            "path": self.path,
            "related_ids": self.related_ids,
            "message": self.message,
            "repair_kind": self.repair_kind.value,
            "repair": self.repair,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GoalDebt":
        fields = (
            "code",
            "severity",
            "path",
            "related_ids",
            "message",
            "repair_kind",
            "repair",
        )
        _closed(payload, schema=cls.SCHEMA, fields=fields, noun="goal debt")
        result = cls(
            code=payload.get("code"),
            severity=payload.get("severity"),
            path=payload.get("path"),
            related_ids=payload.get("related_ids") or (),
            message=payload.get("message"),
            repair_kind=payload.get("repair_kind"),
            repair=payload.get("repair"),
        )
        cls._verify_claim(payload, result)
        return result


@dataclass(frozen=True)
class GoalQualityReport(_GoalContract):
    SCHEMA: ClassVar[str] = GOAL_QUALITY_REPORT_SCHEMA

    goal_id: str
    goal_content_id: str
    policy_id: str
    debt: tuple[GoalDebt, ...]
    score_millionths: int

    def __post_init__(self) -> None:
        for name in ("goal_id", "goal_content_id", "policy_id"):
            object.__setattr__(
                self, name, _text(getattr(self, name), name, required=True)
            )
        object.__setattr__(self, "debt", _records(self.debt, GoalDebt, "debt"))
        object.__setattr__(
            self,
            "score_millionths",
            _integer(self.score_millionths, "score_millionths", maximum=MILLION),
        )
        expected_score = _score_debt(self.debt)
        if self.score_millionths != expected_score:
            raise GoalQualityError(
                "goal quality score does not match deterministic debt scoring"
            )

    @property
    def accepted(self) -> bool:
        return not any(item.severity is DebtSeverity.ERROR for item in self.debt)

    @property
    def valid(self) -> bool:
        return self.accepted

    @property
    def debt_codes(self) -> tuple[GoalDebtCode, ...]:
        return tuple(item.code for item in self.debt)

    def _payload(self) -> dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "goal_content_id": self.goal_content_id,
            "policy_id": self.policy_id,
            "debt": tuple(item.to_dict() for item in self.debt),
            "score_millionths": self.score_millionths,
            "accepted": self.accepted,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GoalQualityReport":
        fields = (
            "goal_id",
            "goal_content_id",
            "policy_id",
            "debt",
            "score_millionths",
            "accepted",
        )
        _closed(payload, schema=cls.SCHEMA, fields=fields, noun="goal quality report")
        raw_debt = payload.get("debt") or ()
        if isinstance(raw_debt, (str, bytes, bytearray, memoryview, Mapping)):
            raise GoalQualityError("goal quality report debt must be a sequence")
        result = cls(
            goal_id=payload.get("goal_id"),
            goal_content_id=payload.get("goal_content_id"),
            policy_id=payload.get("policy_id"),
            debt=tuple(GoalDebt.from_dict(item) for item in raw_debt),
            score_millionths=payload.get("score_millionths"),
        )
        if payload.get("accepted") is not result.accepted:
            raise GoalQualityError("goal quality accepted projection was forged")
        cls._verify_claim(payload, result)
        return result


def _debt(
    code: GoalDebtCode,
    path: str,
    *,
    related_ids: Iterable[str] = (),
    detail: str = "",
    severity: DebtSeverity | None = None,
) -> GoalDebt:
    default_severity, repair_kind, repair = _DEBT_DEFAULTS[code]
    message = code.value.replace("_", " ")
    if detail:
        message = f"{message}: {_text(detail, 'debt detail')}"
    return GoalDebt(
        code=code,
        severity=severity or default_severity,
        path=path,
        related_ids=tuple(related_ids),
        message=message,
        repair_kind=repair_kind,
        repair=repair,
    )


def _score_debt(debt: Iterable[GoalDebt]) -> int:
    penalty = sum(
        100_000 if item.severity is DebtSeverity.ERROR else 25_000
        for item in debt
    )
    return max(0, MILLION - penalty)


def _scope_conflicts(include: Sequence[str], exclude: Sequence[str]) -> tuple[str, ...]:
    conflicts: set[str] = set()
    for included in include:
        left = included.casefold().rstrip("/")
        for excluded in exclude:
            right = excluded.casefold().rstrip("/")
            if left == right or left.startswith(right + "/") or right.startswith(left + "/"):
                conflicts.update((included, excluded))
    return tuple(sorted(conflicts))


def _acceptance_cycles(
    criteria: Sequence[AcceptanceCriterion],
) -> tuple[tuple[str, ...], ...]:
    graph = {
        item.criterion_id: tuple(
            dependency
            for dependency in item.depends_on_criterion_ids
            if dependency in {candidate.criterion_id for candidate in criteria}
        )
        for item in criteria
    }
    cycles: set[tuple[str, ...]] = set()
    visiting: list[str] = []
    visited: set[str] = set()

    def walk(node: str) -> None:
        if node in visiting:
            start = visiting.index(node)
            cycle = visiting[start:] + [node]
            rotations = [
                tuple(cycle[index:-1] + cycle[:index] + [cycle[index]])
                for index in range(len(cycle) - 1)
            ]
            cycles.add(min(rotations))
            return
        if node in visited:
            return
        visiting.append(node)
        for child in graph.get(node, ()):
            walk(child)
        visiting.pop()
        visited.add(node)

    for criterion_id in sorted(graph):
        walk(criterion_id)
    return tuple(sorted(cycles))


def lint_goal(
    goal: TypedGoal,
    *,
    policy: GoalQualityPolicy | None = None,
    known_goal_ids: Iterable[str] | None = None,
) -> GoalQualityReport:
    """Return deterministic repairable debt for ``goal``.

    ``known_goal_ids=None`` means the caller did not provide a complete goal
    population, so external goal dependencies are not guessed to be orphaned.
    Internal criterion/producer/rule references are always checked.
    """

    if not isinstance(goal, TypedGoal):
        raise TypeError("goal must be a TypedGoal")
    selected = policy or GoalQualityPolicy()
    if not isinstance(selected, GoalQualityPolicy):
        raise TypeError("policy must be a GoalQualityPolicy")
    known = None if known_goal_ids is None else set(_strings(known_goal_ids, "known_goal_ids"))
    findings: list[GoalDebt] = []

    missing = (
        (not goal.outcome, GoalDebtCode.MISSING_OUTCOME, "outcome"),
        (not goal.scope.include, GoalDebtCode.MISSING_SCOPE, "scope.include"),
        (not goal.assumptions, GoalDebtCode.MISSING_ASSUMPTIONS, "assumptions"),
        (not goal.non_goals, GoalDebtCode.MISSING_NON_GOALS, "non_goals"),
        (
            not goal.acceptance_criteria,
            GoalDebtCode.MISSING_ACCEPTANCE,
            "acceptance_criteria",
        ),
        (
            not goal.evidence_producers,
            GoalDebtCode.MISSING_EVIDENCE_PRODUCER,
            "evidence_producers",
        ),
        (
            not goal.validation_rules,
            GoalDebtCode.MISSING_VALIDATION,
            "validation_rules",
        ),
        (
            goal.freshness.max_age_seconds <= 0
            or not goal.freshness.require_repository_revision
            or not goal.freshness.require_tree_revision
            or not goal.freshness.require_semantic_dependencies,
            GoalDebtCode.MISSING_FRESHNESS,
            "freshness",
        ),
        (
            not goal.resources.bounded,
            GoalDebtCode.MISSING_RESOURCE_ENVELOPE,
            "resources",
        ),
        (
            not goal.uncertainties,
            GoalDebtCode.MISSING_UNCERTAINTY,
            "uncertainties",
        ),
        (
            not goal.unsupported_semantics,
            GoalDebtCode.MISSING_UNSUPPORTED_SEMANTICS,
            "unsupported_semantics",
        ),
        (
            not goal.refinement_budget.bounded,
            GoalDebtCode.MISSING_REFINEMENT_BUDGET,
            "refinement_budget",
        ),
    )
    findings.extend(_debt(code, path) for condition, code, path in missing if condition)

    unbounded = tuple(
        item
        for item in goal.scope.include
        if item.casefold().strip() in {"*", "**", ".", "/", "all", "repository", "any"}
        or "*" in item
    )
    if unbounded:
        findings.append(
            _debt(
                GoalDebtCode.UNBOUNDED_SCOPE,
                "scope.include",
                related_ids=unbounded,
            )
        )
    conflicts = _scope_conflicts(goal.scope.include, goal.scope.exclude)
    if conflicts:
        findings.append(
            _debt(
                GoalDebtCode.CONFLICTING_SCOPE,
                "scope",
                related_ids=conflicts,
            )
        )

    for uncertainty in goal.uncertainties:
        unresolved = (
            not uncertainty.statement
            or (
                uncertainty.disposition
                in {
                    UncertaintyDisposition.OPEN,
                    UncertaintyDisposition.BLOCKING,
                }
                and not uncertainty.resolution
            )
        )
        if unresolved:
            findings.append(
                _debt(
                    GoalDebtCode.UNCERTAINTY_DEBT,
                    f"uncertainties.{uncertainty.uncertainty_id}",
                    related_ids=(uncertainty.uncertainty_id,),
                    severity=(
                        DebtSeverity.ERROR
                        if uncertainty.disposition
                        is UncertaintyDisposition.BLOCKING
                        else DebtSeverity.WARNING
                    ),
                )
            )
    for semantic in goal.unsupported_semantics:
        if not semantic.statement or not semantic.fallback:
            findings.append(
                _debt(
                    GoalDebtCode.UNSUPPORTED_SEMANTICS,
                    f"unsupported_semantics.{semantic.semantic_id}",
                    related_ids=(semantic.semantic_id,),
                )
            )

    criterion_ids = {item.criterion_id for item in goal.acceptance_criteria}
    producer_by_id = {item.producer_id: item for item in goal.evidence_producers}
    rule_by_id = {item.rule_id: item for item in goal.validation_rules}

    for cycle in _acceptance_cycles(goal.acceptance_criteria):
        findings.append(
            _debt(
                GoalDebtCode.CIRCULAR_ACCEPTANCE,
                "acceptance_criteria",
                related_ids=cycle,
            )
        )

    for criterion in goal.acceptance_criteria:
        orphan_criteria = set(criterion.depends_on_criterion_ids).difference(criterion_ids)
        orphan_producers = set(criterion.evidence_producer_ids).difference(producer_by_id)
        orphan_rules = set(criterion.validation_rule_ids).difference(rule_by_id)
        orphans = tuple(sorted(orphan_criteria | orphan_producers | orphan_rules))
        if orphans:
            findings.append(
                _debt(
                    GoalDebtCode.ORPHAN_DEPENDENCY,
                    f"acceptance_criteria.{criterion.criterion_id}",
                    related_ids=orphans,
                )
            )
        if (
            not criterion.evidence_producer_ids
            or not criterion.validation_rule_ids
            or any(
                not producer_by_id[item].kind
                or not producer_by_id[item].output_schema
                for item in criterion.evidence_producer_ids
                if item in producer_by_id
            )
        ):
            findings.append(
                _debt(
                    GoalDebtCode.UNVERIFIABLE_EVIDENCE,
                    f"acceptance_criteria.{criterion.criterion_id}",
                    related_ids=(criterion.criterion_id,),
                )
            )
        normalized_statement = f" {criterion.statement.casefold()} "
        vague = tuple(
            term
            for term in selected.ambiguous_terms
            if f" {term.casefold()} " in normalized_statement
        )
        if not criterion.statement or not criterion.completion_signal or vague:
            findings.append(
                _debt(
                    GoalDebtCode.AMBIGUOUS_COMPLETION,
                    f"acceptance_criteria.{criterion.criterion_id}",
                    related_ids=(criterion.criterion_id, *vague),
                )
            )

    for rule in goal.validation_rules:
        orphan_criteria = set(rule.criterion_ids).difference(criterion_ids)
        orphan_producer = (
            (rule.producer_id,)
            if rule.producer_id and rule.producer_id not in producer_by_id
            else ()
        )
        if orphan_criteria or orphan_producer:
            findings.append(
                _debt(
                    GoalDebtCode.ORPHAN_DEPENDENCY,
                    f"validation_rules.{rule.rule_id}",
                    related_ids=tuple(orphan_criteria) + orphan_producer,
                )
            )
        if not rule.command or not rule.producer_id or not rule.criterion_ids:
            findings.append(
                _debt(
                    GoalDebtCode.UNVERIFIABLE_EVIDENCE,
                    f"validation_rules.{rule.rule_id}",
                    related_ids=(rule.rule_id,),
                )
            )

    completion_producers = {
        item.producer_id
        for item in goal.evidence_producers
        if item.authority
        in {EvidenceAuthority.OPERATOR, EvidenceAuthority.COMPLETION_GATE}
    }
    authorized = set(goal.authorized_completion_producer_ids)
    undeclared = completion_producers.difference(authorized)
    orphan_authorizations = authorized.difference(completion_producers)
    if undeclared or orphan_authorizations:
        findings.append(
            _debt(
                GoalDebtCode.HIDDEN_AUTHORITY,
                "authorized_completion_producer_ids",
                related_ids=tuple(undeclared | orphan_authorizations),
            )
        )
    for producer in goal.evidence_producers:
        suspicious_kind = producer.kind.casefold() in {
            "llm",
            "model",
            "proposal",
            "generator",
        }
        if suspicious_kind and producer.authority not in {
            EvidenceAuthority.DIAGNOSTIC,
            EvidenceAuthority.PROPOSAL,
        }:
            findings.append(
                _debt(
                    GoalDebtCode.HIDDEN_AUTHORITY,
                    f"evidence_producers.{producer.producer_id}.authority",
                    related_ids=(producer.producer_id,),
                )
            )

    if known is not None:
        orphan_goals = set(goal.scope.dependency_goal_ids).difference(known)
        if orphan_goals:
            findings.append(
                _debt(
                    GoalDebtCode.ORPHAN_DEPENDENCY,
                    "scope.dependency_goal_ids",
                    related_ids=orphan_goals,
                )
            )

    breadth = (
        len(goal.scope.include)
        + len(goal.scope.dependency_goal_ids)
        + len(goal.acceptance_criteria)
        + len(goal.evidence_producers)
        + len(goal.validation_rules)
    )
    breadth_reasons: list[str] = []
    if len(goal.scope.include) > min(
        selected.max_scope_items,
        goal.resources.max_scope_items or selected.max_scope_items,
    ):
        breadth_reasons.append("scope")
    if len(goal.acceptance_criteria) > selected.max_acceptance_criteria:
        breadth_reasons.append("acceptance")
    if len(goal.scope.dependency_goal_ids) > selected.max_dependencies:
        breadth_reasons.append("dependencies")
    if breadth > selected.max_total_breadth:
        breadth_reasons.append("total")
    if breadth_reasons:
        findings.append(
            _debt(
                GoalDebtCode.EXCESSIVE_BREADTH,
                "goal",
                related_ids=breadth_reasons,
                detail=f"measured breadth {breadth}",
            )
        )

    # One finding per stable semantic location/code/related population.
    deduplicated = {
        (item.code.value, item.path, item.related_ids): item for item in findings
    }
    debt = tuple(
        sorted(
            deduplicated.values(),
            key=lambda item: (
                item.severity.value,
                item.code.value,
                item.path,
                item.related_ids,
            ),
        )
    )
    # Fixed integer weights make scoring platform-independent.  The score is
    # diagnostic and admission still fails on any ERROR.
    score = _score_debt(debt)
    return GoalQualityReport(
        goal_id=goal.goal_id,
        goal_content_id=goal.content_id,
        policy_id=selected.content_id,
        debt=debt,
        score_millionths=score,
    )


def score_goal(
    goal: TypedGoal,
    *,
    policy: GoalQualityPolicy | None = None,
    known_goal_ids: Iterable[str] | None = None,
) -> int:
    """Return the deterministic diagnostic score in integer millionths."""

    return lint_goal(
        goal, policy=policy, known_goal_ids=known_goal_ids
    ).score_millionths


def validate_goal(
    goal: TypedGoal,
    *,
    policy: GoalQualityPolicy | None = None,
    known_goal_ids: Iterable[str] | None = None,
) -> GoalQualityReport:
    """Return an accepted report or raise :class:`GoalAdmissionError`."""

    report = lint_goal(goal, policy=policy, known_goal_ids=known_goal_ids)
    if not report.accepted:
        raise GoalAdmissionError(report)
    return report


class GoalQualityLinter:
    """Small policy-bound facade for planner and refinement call sites."""

    def __init__(self, policy: GoalQualityPolicy | None = None) -> None:
        self.policy = policy or GoalQualityPolicy()
        if not isinstance(self.policy, GoalQualityPolicy):
            raise TypeError("policy must be a GoalQualityPolicy")

    def lint(
        self,
        goal: TypedGoal,
        *,
        known_goal_ids: Iterable[str] | None = None,
    ) -> GoalQualityReport:
        return lint_goal(
            goal, policy=self.policy, known_goal_ids=known_goal_ids
        )

    def score(
        self,
        goal: TypedGoal,
        *,
        known_goal_ids: Iterable[str] | None = None,
    ) -> int:
        return self.lint(goal, known_goal_ids=known_goal_ids).score_millionths

    def validate(
        self,
        goal: TypedGoal,
        *,
        known_goal_ids: Iterable[str] | None = None,
    ) -> GoalQualityReport:
        report = self.lint(goal, known_goal_ids=known_goal_ids)
        if not report.accepted:
            raise GoalAdmissionError(report)
        return report


def assert_frozen_root(parent: TypedGoal, refinement: TypedGoal) -> None:
    """Reject a refinement that mutates or substitutes the frozen root."""

    if not isinstance(parent, TypedGoal) or not isinstance(refinement, TypedGoal):
        raise TypeError("parent and refinement must be TypedGoal values")
    if parent.root != refinement.root:
        raise GoalQualityError("refinement must preserve the frozen root identity")


def canonical_goal_bytes(goal: TypedGoal) -> bytes:
    if not isinstance(goal, TypedGoal):
        raise TypeError("goal must be a TypedGoal")
    return goal.canonical_bytes()


def canonical_goal_json(goal: TypedGoal) -> str:
    return canonical_goal_bytes(goal).decode("utf-8")


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                item.strip()
                for item in re.split(r"[,;\n]+", str(value or ""))
                if item.strip()
            }
        )
    )


def _split_acceptance(value: str) -> tuple[str, ...]:
    text = str(value or "").strip()
    if not text:
        return ()
    return tuple(item.strip(" .") for item in re.split(r";\s+|\n+", text) if item.strip(" ."))


def _parse_int(value: Any) -> int:
    match = re.search(r"\d+", str(value or ""))
    return int(match.group(0)) if match else 0


def _parse_json_field(fields: Mapping[str, Any], *names: str) -> Any | None:
    for name in names:
        raw = fields.get(name)
        if raw is None:
            continue
        if isinstance(raw, (dict, list, tuple)):
            return raw
        text = str(raw).strip()
        if not text:
            continue
        try:
            return json.loads(text)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
    return None


def _compat_root_revision(goal: Any) -> str:
    return content_identity(
        {
            "schema": "ipfs_accelerate_py/agent-supervisor/objective-markdown-root@1",
            "goal_id": goal.goal_id,
            "title": goal.title,
            "fields": dict(goal.fields),
        }
    )


def _authority_from_value(value: Any) -> EvidenceAuthority:
    if isinstance(value, EvidenceAuthority):
        return value
    text = str(value or "").strip().lower()
    if not text:
        return EvidenceAuthority.DIAGNOSTIC
    try:
        return EvidenceAuthority(text)
    except ValueError as exc:
        raise GoalQualityError(
            "evidence authority must be one of: "
            + ", ".join(item.value for item in EvidenceAuthority)
        ) from exc


def _producer_from_mapping(payload: Mapping[str, Any], *, index: int) -> EvidenceProducer:
    if not isinstance(payload, Mapping):
        raise GoalQualityError("evidence producer entry must be an object")
    producer_id = _text(
        payload.get("producer_id") or payload.get("id") or f"producer:{index + 1}",
        "producer_id",
        required=True,
    )
    return EvidenceProducer(
        producer_id=producer_id,
        kind=_text(payload.get("kind") or "", "kind"),
        output_schema=_text(payload.get("output_schema") or "", "output_schema"),
        authority=_authority_from_value(payload.get("authority")),
        capability_id=_text(payload.get("capability_id") or "", "capability_id"),
        independent=bool(payload.get("independent", False)),
    )


def _is_goal_reference(value: str) -> bool:
    text = value.strip()
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*-G\d+", text)) or text.startswith(
        "goal:"
    )


def is_datasets_contract_goal_quality_evidence_path(reference: str) -> bool:
    """Return True when ``reference`` is the durable goal-quality evidence path."""

    text = str(reference or "").strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text == DATASETS_CONTRACT_GOAL_QUALITY_EVIDENCE_PATH


def _reviewed_legacy_producer(reference: str) -> EvidenceProducer:
    text = reference.strip()
    lower = text.casefold()
    if lower.endswith(".py") and (
        "/test" in lower or lower.startswith("test") or "test_" in lower
    ):
        return EvidenceProducer(
            producer_id=text,
            kind="test_runner",
            output_schema=PYTEST_RECEIPT_OUTPUT_SCHEMA,
            authority=EvidenceAuthority.VALIDATION,
            capability_id="capability:python-pytest",
            independent=True,
        )
    if is_datasets_contract_goal_quality_evidence_path(text) or lower.endswith(
        (".json", ".md", ".jsonl", ".yaml", ".yml")
    ):
        return EvidenceProducer(
            producer_id=text,
            kind="artifact_receipt",
            output_schema=ARTIFACT_RECEIPT_OUTPUT_SCHEMA,
            authority=EvidenceAuthority.VALIDATION,
            independent=True,
        )
    if _is_goal_reference(text):
        return EvidenceProducer(
            producer_id=text,
            kind="goal_reference",
            output_schema=LEGACY_GOAL_REF_OUTPUT_SCHEMA,
            authority=EvidenceAuthority.DIAGNOSTIC,
        )
    return EvidenceProducer(
        producer_id=text,
        kind="legacy_reference",
        output_schema=LEGACY_EVIDENCE_OUTPUT_SCHEMA,
        authority=EvidenceAuthority.DIAGNOSTIC,
    )


def _producers_from_fields(
    fields: Mapping[str, Any],
    *,
    require_schemas: bool,
) -> tuple[EvidenceProducer, ...]:
    structured = _parse_json_field(
        fields,
        "evidence_producers_json",
        "evidence_producer_records_json",
        "typed_evidence_producers_json",
    )
    if isinstance(structured, list):
        return tuple(
            _producer_from_mapping(item, index=index)
            for index, item in enumerate(structured)
            if isinstance(item, Mapping)
        )
    raw_producers = _split_csv(
        fields.get("evidence_producers")
        or fields.get("evidence")
        or fields.get("producing_task_or_scan")
        or ""
    )
    producers: list[EvidenceProducer] = []
    for item in raw_producers:
        if require_schemas:
            producers.append(_reviewed_legacy_producer(item))
        else:
            producers.append(
                EvidenceProducer(
                    producer_id=item,
                    kind="legacy_reference",
                    output_schema="",
                    authority=EvidenceAuthority.DIAGNOSTIC,
                )
            )
    return tuple(producers)


def _completion_signals_from_fields(
    fields: Mapping[str, Any],
    statements: Sequence[str],
) -> tuple[str, ...]:
    structured = _parse_json_field(
        fields,
        "completion_signals_json",
        "acceptance_completion_signals_json",
    )
    if isinstance(structured, list):
        signals = tuple(_text(item, "completion_signal") for item in structured)
        if len(signals) == len(statements):
            return signals
    if isinstance(structured, Mapping):
        ordered: list[str] = []
        for index, statement in enumerate(statements):
            key_options = (
                f"criterion:{index + 1}",
                str(index + 1),
                statement,
            )
            signal = ""
            for key in key_options:
                if key in structured:
                    signal = _text(structured[key], "completion_signal")
                    break
            ordered.append(signal)
        return tuple(ordered)
    raw = str(fields.get("completion_signals") or fields.get("completion_signal") or "")
    if raw.strip():
        parts = _split_acceptance(raw)
        if len(parts) == len(statements):
            return parts
    return tuple("" for _ in statements)


def _criteria_from_fields(
    fields: Mapping[str, Any],
    *,
    producer_ids: Sequence[str],
    rule_ids: Sequence[str],
    require_completion_signals: bool,
) -> tuple[AcceptanceCriterion, ...]:
    structured = _parse_json_field(
        fields,
        "acceptance_criteria_json",
        "typed_acceptance_criteria_json",
    )
    if isinstance(structured, list) and structured:
        criteria: list[AcceptanceCriterion] = []
        for index, item in enumerate(structured):
            if not isinstance(item, Mapping):
                raise GoalQualityError("acceptance criterion entry must be an object")
            criterion_id = _text(
                item.get("criterion_id") or f"criterion:{index + 1}",
                "criterion_id",
                required=True,
            )
            statement = _text(item.get("statement") or "", "statement")
            signal = _text(item.get("completion_signal") or "", "completion_signal")
            if require_completion_signals and not signal:
                signal = (
                    f"criterion:{index + 1} is satisfied when bound producers emit "
                    f"fresh receipts and validation exits 0"
                )
            criteria.append(
                AcceptanceCriterion(
                    criterion_id=criterion_id,
                    statement=statement,
                    evidence_producer_ids=tuple(
                        item.get("evidence_producer_ids") or producer_ids
                    ),
                    validation_rule_ids=tuple(
                        item.get("validation_rule_ids") or rule_ids
                    ),
                    depends_on_criterion_ids=tuple(
                        item.get("depends_on_criterion_ids") or ()
                    ),
                    completion_signal=signal,
                )
            )
        return tuple(criteria)

    acceptance_texts = _split_acceptance(
        fields.get("acceptance_criteria") or fields.get("acceptance") or ""
    )
    signals = _completion_signals_from_fields(fields, acceptance_texts)
    criteria_out: list[AcceptanceCriterion] = []
    for index, statement in enumerate(acceptance_texts):
        signal = signals[index] if index < len(signals) else ""
        if require_completion_signals and not signal:
            signal = (
                f"criterion:{index + 1} is satisfied when bound producers emit "
                f"fresh receipts and validation exits 0"
            )
        criteria_out.append(
            AcceptanceCriterion(
                criterion_id=f"criterion:{index + 1}",
                statement=statement,
                evidence_producer_ids=tuple(producer_ids),
                validation_rule_ids=tuple(rule_ids),
                completion_signal=signal,
            )
        )
    return tuple(criteria_out)


def _resource_envelope_from_fields(
    fields: Mapping[str, Any],
    *,
    defaults: bool,
) -> ResourceEnvelope:
    structured = _parse_json_field(
        fields, "resource_envelope_json", "resources_json", "typed_resources_json"
    )
    if isinstance(structured, Mapping):
        return ResourceEnvelope(
            max_wall_seconds=_parse_int(
                structured.get("max_wall_seconds") or structured.get("runtime_seconds")
            ),
            max_tokens=_parse_int(
                structured.get("max_tokens") or structured.get("tokens")
            ),
            max_cost_microunits=_parse_int(structured.get("max_cost_microunits")),
            max_artifacts=_parse_int(structured.get("max_artifacts")),
            max_parallelism=_parse_int(structured.get("max_parallelism")),
            max_scope_items=_parse_int(structured.get("max_scope_items")),
        )
    if not defaults:
        return ResourceEnvelope(
            max_wall_seconds=_parse_int(fields.get("max_wall_seconds")),
            max_tokens=_parse_int(fields.get("max_tokens")),
            max_cost_microunits=_parse_int(fields.get("max_cost_microunits")),
            max_artifacts=_parse_int(fields.get("max_artifacts")),
            max_parallelism=_parse_int(fields.get("max_parallelism")),
            max_scope_items=_parse_int(fields.get("max_scope_items")),
        )
    resource_class = str(fields.get("resource_class") or "cpu-medium").strip().lower()
    presets = {
        "cpu-small": (300, 8_192, 1_000_000, 8, 1, 16),
        "cpu-medium": (900, 16_384, 2_000_000, 16, 2, 32),
        "cpu-large": (3_600, 32_768, 5_000_000, 32, 4, 64),
        "io-large": (3_600, 16_384, 5_000_000, 64, 2, 64),
    }
    wall, tokens, cost, artifacts, parallelism, scope_items = presets.get(
        resource_class, presets["cpu-medium"]
    )
    return ResourceEnvelope(
        max_wall_seconds=_parse_int(fields.get("max_wall_seconds")) or wall,
        max_tokens=_parse_int(fields.get("max_tokens")) or tokens,
        max_cost_microunits=_parse_int(fields.get("max_cost_microunits")) or cost,
        max_artifacts=_parse_int(fields.get("max_artifacts")) or artifacts,
        max_parallelism=_parse_int(fields.get("max_parallelism")) or parallelism,
        max_scope_items=_parse_int(fields.get("max_scope_items")) or scope_items,
    )


def _refinement_budget_from_fields(
    fields: Mapping[str, Any],
    *,
    defaults: bool,
) -> RefinementBudget:
    structured = _parse_json_field(
        fields, "refinement_budget_json", "typed_refinement_budget_json"
    )
    if isinstance(structured, Mapping):
        return RefinementBudget(
            max_rounds=_parse_int(structured.get("max_rounds")),
            max_children=_parse_int(
                structured.get("max_children") or structured.get("breadth")
            ),
            max_depth=_parse_int(structured.get("max_depth")),
            max_debt_items=_parse_int(
                structured.get("max_debt_items") or structured.get("max_debt")
            ),
            max_tokens=_parse_int(structured.get("max_tokens")),
        )
    max_children = _parse_int(
        fields.get("max_refinement_children") or fields.get("refinement_breadth_limit")
    )
    max_depth = _parse_int(
        fields.get("max_refinement_depth") or fields.get("refinement_depth_limit")
    )
    max_rounds = _parse_int(fields.get("max_refinement_rounds"))
    max_debt = _parse_int(fields.get("max_refinement_debt"))
    max_tokens = _parse_int(fields.get("max_refinement_tokens"))
    if defaults:
        max_children = max_children or DEFAULT_TYPED_MAX_CHILDREN
        max_depth = max_depth or DEFAULT_TYPED_MAX_DEPTH
        max_rounds = max_rounds or DEFAULT_TYPED_MAX_ROUNDS
        max_debt = max_debt or DEFAULT_TYPED_MAX_DEBT_ITEMS
        max_tokens = max_tokens or DEFAULT_TYPED_MAX_REFINEMENT_TOKENS
    return RefinementBudget(
        max_rounds=max_rounds,
        max_children=max_children,
        max_depth=max_depth,
        max_debt_items=max_debt,
        max_tokens=max_tokens,
    )


def _uncertainties_from_fields(
    fields: Mapping[str, Any],
    *,
    defaults: bool,
) -> tuple[UncertaintyItem, ...]:
    structured = _parse_json_field(
        fields, "uncertainties_json", "uncertainty_json", "typed_uncertainties_json"
    )
    if isinstance(structured, list) and structured:
        items: list[UncertaintyItem] = []
        for index, item in enumerate(structured):
            if isinstance(item, Mapping):
                disposition_raw = item.get("disposition") or UncertaintyDisposition.OPEN
                items.append(
                    UncertaintyItem(
                        uncertainty_id=_text(
                            item.get("uncertainty_id") or f"uncertainty:{index + 1}",
                            "uncertainty_id",
                            required=True,
                        ),
                        statement=_text(item.get("statement") or "", "statement"),
                        disposition=_enum(
                            disposition_raw, UncertaintyDisposition, "disposition"
                        ),
                        impact=_text(item.get("impact") or "", "impact"),
                        resolution=_text(item.get("resolution") or "", "resolution"),
                    )
                )
            else:
                items.append(
                    UncertaintyItem(
                        uncertainty_id=f"uncertainty:{index + 1}",
                        statement=_text(item, "statement"),
                    )
                )
        return tuple(items)
    statements = _split_csv(fields.get("uncertainty") or fields.get("uncertainties") or "")
    if statements:
        return tuple(
            UncertaintyItem(
                uncertainty_id=f"uncertainty:{index + 1}",
                statement=statement,
            )
            for index, statement in enumerate(statements)
        )
    if defaults:
        return (
            UncertaintyItem(
                uncertainty_id="uncertainty:reviewed-none",
                statement="No unresolved uncertainty remains after typed migration review.",
                disposition=UncertaintyDisposition.MITIGATED,
                impact="none",
                resolution="Reopen when a declared dependency or producer binding changes.",
            ),
        )
    return ()


def _unsupported_from_fields(
    fields: Mapping[str, Any],
    *,
    defaults: bool,
) -> tuple[UnsupportedSemantic, ...]:
    structured = _parse_json_field(
        fields,
        "unsupported_semantics_json",
        "typed_unsupported_semantics_json",
    )
    if isinstance(structured, list) and structured:
        items: list[UnsupportedSemantic] = []
        for index, item in enumerate(structured):
            if isinstance(item, Mapping):
                items.append(
                    UnsupportedSemantic(
                        semantic_id=_text(
                            item.get("semantic_id") or f"unsupported:{index + 1}",
                            "semantic_id",
                            required=True,
                        ),
                        statement=_text(item.get("statement") or "", "statement"),
                        fallback=_text(item.get("fallback") or "", "fallback"),
                    )
                )
            else:
                items.append(
                    UnsupportedSemantic(
                        semantic_id=f"unsupported:{index + 1}",
                        statement=_text(item, "statement"),
                    )
                )
        return tuple(items)
    statements = _split_csv(fields.get("unsupported_semantics") or "")
    if statements:
        return tuple(
            UnsupportedSemantic(
                semantic_id=f"unsupported:{index + 1}",
                statement=statement,
            )
            for index, statement in enumerate(statements)
        )
    if defaults:
        return (
            UnsupportedSemantic(
                semantic_id="semantic:reviewed-none",
                statement="No unsupported semantic is used for typed admission.",
                fallback="Fail closed on unknown semantics and report typed quality debt.",
            ),
        )
    return ()


def _project_one_objective_goal(
    raw: Any,
    *,
    root_raw: Any,
    lossless: bool,
    overlay: "TypedGoal | None" = None,
) -> TypedGoal:
    if overlay is not None:
        if overlay.goal_id != raw.goal_id:
            raise GoalQualityError(
                f"typed overlay goal_id {overlay.goal_id!r} does not match {raw.goal_id!r}"
            )
        return overlay

    fields = raw.fields
    scope_values = _split_csv(
        fields.get("scope")
        or fields.get("outputs")
        or fields.get("predicted_files")
        or ""
    )
    if lossless and not scope_values:
        scope_values = (f"objective:{raw.goal_id}",)
    dependencies = _split_csv(fields.get("depends_on") or "")
    producers = _producers_from_fields(fields, require_schemas=lossless)
    if lossless and not producers:
        validation_text = str(fields.get("validation") or "").strip()
        producer_id = (
            f"producer:validation:{raw.goal_id}"
            if validation_text
            else f"producer:diagnostic:{raw.goal_id}"
        )
        producers = (
            EvidenceProducer(
                producer_id=producer_id,
                kind="test_runner" if validation_text else "legacy_reference",
                output_schema=(
                    PYTEST_RECEIPT_OUTPUT_SCHEMA
                    if validation_text
                    else LEGACY_EVIDENCE_OUTPUT_SCHEMA
                ),
                authority=(
                    EvidenceAuthority.VALIDATION
                    if validation_text
                    else EvidenceAuthority.DIAGNOSTIC
                ),
                capability_id="capability:python-pytest" if validation_text else "",
                independent=bool(validation_text),
            ),
        )
    # The current objective format stores the entire shell invocation in
    # one field (often with several pytest paths).  Do not split inside a
    # command or attempt to parse shell authority here.
    validation_text = str(fields.get("validation") or "").strip()
    commands = (validation_text,) if validation_text else ()
    if lossless and not commands:
        commands = (f"true  # no validation declared for {raw.goal_id}",)
    producer_ids = tuple(item.producer_id for item in producers)
    rule_ids = tuple(f"validation:{index + 1}" for index in range(len(commands)))
    criteria = _criteria_from_fields(
        fields,
        producer_ids=producer_ids,
        rule_ids=rule_ids,
        require_completion_signals=lossless,
    )
    if lossless and not criteria:
        criteria = (
            AcceptanceCriterion(
                criterion_id="criterion:1",
                statement=str(
                    fields.get("outcome")
                    or fields.get("goal")
                    or fields.get("objective")
                    or raw.title
                    or raw.goal_id
                ),
                evidence_producer_ids=producer_ids,
                validation_rule_ids=rule_ids,
                completion_signal=(
                    "criterion:1 is satisfied when bound producers emit fresh "
                    "receipts and validation exits 0"
                ),
            ),
        )
    criterion_ids = tuple(item.criterion_id for item in criteria)
    validation_producer_id = producer_ids[0] if producer_ids else ""
    rules = tuple(
        ValidationRule(
            rule_id=f"validation:{index + 1}",
            command=command,
            producer_id=validation_producer_id,
            criterion_ids=criterion_ids,
            hermetic=lossless,
        )
        for index, command in enumerate(commands)
    )
    freshness_seconds = _parse_int(
        fields.get("freshness_horizon_seconds")
        or fields.get("evidence_freshness_seconds")
    )
    if lossless and freshness_seconds <= 0:
        freshness_seconds = DEFAULT_TYPED_FRESHNESS_SECONDS
    assumptions = _split_csv(fields.get("assumptions") or "")
    if lossless and not assumptions:
        if dependencies:
            assumptions = (
                f"Declared dependencies remain admitted: {', '.join(dependencies)}.",
            )
        else:
            assumptions = (
                "No external assumptions beyond the frozen objective root identity.",
            )
    non_goals = _split_csv(fields.get("non_goals") or "")
    if lossless and not non_goals:
        non_goals = (
            "Do not invent completion-gate authority or claim typed admission "
            "from legacy structural parsing alone.",
        )
    authorized = _split_csv(
        fields.get("authorized_completion_producer_ids")
        or fields.get("authorized_completion_producers")
        or ""
    )
    return TypedGoal(
        goal_id=raw.goal_id,
        root=FrozenRootIdentity(
            goal_id=root_raw.goal_id,
            revision=_compat_root_revision(root_raw),
        ),
        outcome=fields.get("outcome")
        or fields.get("goal")
        or fields.get("objective")
        or raw.title,
        scope=GoalScope(
            include=scope_values,
            exclude=_split_csv(fields.get("non_scope") or fields.get("exclude") or ""),
            dependency_goal_ids=dependencies,
        ),
        assumptions=assumptions,
        non_goals=non_goals,
        acceptance_criteria=criteria,
        evidence_producers=producers,
        validation_rules=rules,
        freshness=FreshnessPolicy(max_age_seconds=freshness_seconds),
        resources=_resource_envelope_from_fields(fields, defaults=lossless),
        uncertainties=_uncertainties_from_fields(fields, defaults=lossless),
        unsupported_semantics=_unsupported_from_fields(fields, defaults=lossless),
        refinement_budget=_refinement_budget_from_fields(fields, defaults=lossless),
        authorized_completion_producer_ids=authorized,
    )


def project_objective_markdown(
    markdown: str,
    *,
    goal_id: str | None = None,
    typed_overlay: Mapping[str, TypedGoal] | ObjectiveTypedGoals | None = None,
    lossless: bool = False,
) -> tuple[TypedGoal, ...]:
    """Project current ``ObjectiveGoal`` Markdown without inventing authority.

    By default this is the documented structural legacy path: missing
    freshness/resource/refinement bounds stay at zero and produce repairable
    debt, and bare evidence references stay diagnostic without output schemas
    or completion signals.

    When Markdown carries explicit typed JSON fields (producer records,
    completion signals, resource envelopes, and similar), those values are
    preserved instead of dropped.  A heap-bound :class:`ObjectiveTypedGoals`
    sidecar or ``typed_overlay`` mapping supplies a lossless representation for
    admission without mutating the Markdown heap.  ``lossless=True`` applies the
    reviewed migration defaults used by :func:`migrate_objective_markdown`.
    """

    if not isinstance(markdown, str):
        raise TypeError("markdown must be a string")
    from .objective_graph import parse_goal_heap

    parsed = parse_goal_heap(markdown)
    selected = [item for item in parsed if goal_id is None or item.goal_id == goal_id]
    if goal_id is not None and not selected:
        raise GoalQualityError(f"objective Markdown does not contain {goal_id}")
    by_id = {item.goal_id: item for item in parsed}

    overlay_map: dict[str, TypedGoal] = {}
    if isinstance(typed_overlay, ObjectiveTypedGoals):
        validate_objective_typed_goals(markdown, typed_overlay)
        overlay_map = {item.goal_id: item for item in typed_overlay.goals}
    elif isinstance(typed_overlay, Mapping):
        for key, value in typed_overlay.items():
            if not isinstance(value, TypedGoal):
                raise GoalQualityError("typed_overlay values must be TypedGoal records")
            overlay_map[str(key)] = value

    def root_for(item: Any) -> Any:
        seen: set[str] = set()
        current = item
        while True:
            if current.goal_id in seen:
                raise GoalQualityError("objective Markdown contains a parent cycle")
            seen.add(current.goal_id)
            parent_id = str(current.fields.get("parent") or "").strip()
            if not parent_id:
                return current
            if parent_id not in by_id:
                return current
            current = by_id[parent_id]

    results: list[TypedGoal] = []
    for raw in selected:
        results.append(
            _project_one_objective_goal(
                raw,
                root_raw=root_for(raw),
                lossless=lossless,
                overlay=overlay_map.get(raw.goal_id),
            )
        )
    return tuple(results)


@dataclass(frozen=True)
class ObjectiveTypedGoals(_GoalContract):
    """Canonical typed sidecar bound to one exact objective heap identity."""

    SCHEMA: ClassVar[str] = OBJECTIVE_TYPED_GOALS_SCHEMA

    objective_heap_id: str
    goals: tuple[TypedGoal, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "objective_heap_id",
            _text(self.objective_heap_id, "objective_heap_id", required=True),
        )
        object.__setattr__(self, "goals", _records(self.goals, TypedGoal, "goals"))
        goal_ids = tuple(item.goal_id for item in self.goals)
        if len(goal_ids) != len(set(goal_ids)):
            raise GoalQualityError(
                "objective typed goals contain duplicate goal_id values"
            )
        object.__setattr__(
            self,
            "goals",
            tuple(
                sorted(
                    self.goals,
                    key=lambda item: (item.goal_id.casefold(), item.goal_id),
                )
            ),
        )

    def goal_map(self) -> Mapping[str, TypedGoal]:
        return MappingProxyType({item.goal_id: item for item in self.goals})

    def _payload(self) -> dict[str, Any]:
        return {
            "objective_heap_id": self.objective_heap_id,
            "goals": tuple(item.to_dict() for item in self.goals),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ObjectiveTypedGoals":
        fields = ("objective_heap_id", "goals")
        _closed(
            payload, schema=cls.SCHEMA, fields=fields, noun="objective typed goals"
        )
        raw_goals = payload.get("goals") or ()
        if isinstance(raw_goals, (str, bytes, bytearray, memoryview, Mapping)):
            raise GoalQualityError("objective typed goals must be a sequence")
        result = cls(
            objective_heap_id=payload.get("objective_heap_id"),
            goals=tuple(TypedGoal.from_dict(item) for item in raw_goals),
        )
        cls._verify_claim(payload, result)
        return result


def validate_objective_typed_goals(
    markdown: str,
    document: ObjectiveTypedGoals,
) -> ObjectiveTypedGoals:
    """Reject a typed sidecar that is stale or incomplete for ``markdown``.

    The heap content identity binds the source bytes, but it does not by
    itself prove that a decoded sidecar carries one record for every heap
    goal. Exact coverage is required before a sidecar can feed supervisor
    admission; otherwise a self-consistent partial document could silently
    remove a newly refined goal from the backlog projection.
    """

    if not isinstance(markdown, str):
        raise TypeError("markdown must be a string")
    if not isinstance(document, ObjectiveTypedGoals):
        raise TypeError("document must be an ObjectiveTypedGoals value")

    from .objective_graph import objective_heap_content_id, parse_goal_heap

    current_id = objective_heap_content_id(markdown)
    if document.objective_heap_id != current_id:
        raise GoalQualityError(
            "objective typed goals sidecar is stale for the current heap"
        )

    expected_goal_ids = {goal.goal_id for goal in parse_goal_heap(markdown)}
    document_goal_ids = {goal.goal_id for goal in document.goals}
    if document_goal_ids != expected_goal_ids:
        missing = sorted(expected_goal_ids - document_goal_ids)
        unexpected = sorted(document_goal_ids - expected_goal_ids)
        details: list[str] = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unexpected:
            details.append("unexpected: " + ", ".join(unexpected))
        raise GoalQualityError(
            "objective typed goals goal coverage does not match "
            f"the current heap ({'; '.join(details)})"
        )
    return document


def migrate_objective_markdown(markdown: str) -> ObjectiveTypedGoals:
    """Migrate one Markdown heap into a lossless versioned typed sidecar.

    Reviewed structural defaults fill dimensions the legacy format cannot
    express (producer output schemas, completion signals, finite envelopes).
    Completion-gate authority is never invented: producers remain diagnostic or
    validation unless the Markdown explicitly authorizes a gate.
    """

    if not isinstance(markdown, str):
        raise TypeError("markdown must be a string")
    from .objective_graph import objective_heap_content_id

    goals = project_objective_markdown(markdown, lossless=True)
    return ObjectiveTypedGoals(
        objective_heap_id=objective_heap_content_id(markdown),
        goals=goals,
    )


def lint_objective_typed_goals(
    document: ObjectiveTypedGoals,
    *,
    policy: GoalQualityPolicy | None = None,
) -> tuple[GoalQualityReport, ...]:
    if not isinstance(document, ObjectiveTypedGoals):
        raise TypeError("document must be an ObjectiveTypedGoals value")
    known = tuple(item.goal_id for item in document.goals)
    return tuple(
        lint_goal(item, policy=policy, known_goal_ids=known) for item in document.goals
    )


def goal_from_objective_markdown(
    markdown: str,
    goal_id: str,
    *,
    typed_overlay: Mapping[str, TypedGoal] | ObjectiveTypedGoals | None = None,
    lossless: bool = False,
) -> TypedGoal:
    """Return one compatibility projection by objective identity."""

    return project_objective_markdown(
        markdown,
        goal_id=goal_id,
        typed_overlay=typed_overlay,
        lossless=lossless,
    )[0]


def lint_objective_markdown(
    markdown: str,
    *,
    policy: GoalQualityPolicy | None = None,
    typed_overlay: Mapping[str, TypedGoal] | ObjectiveTypedGoals | None = None,
    lossless: bool = False,
) -> tuple[GoalQualityReport, ...]:
    goals = project_objective_markdown(
        markdown,
        typed_overlay=typed_overlay,
        lossless=lossless,
    )
    known = tuple(item.goal_id for item in goals)
    return tuple(lint_goal(item, policy=policy, known_goal_ids=known) for item in goals)


# Compatibility names for early design notes and downstream planner drafts.
GoalGrammar = TypedGoal
GoalSpecification = TypedGoal
GoalQualityDebt = GoalDebt
GoalDebtKind = GoalDebtCode
QualityDebt = GoalDebt
UncertaintyDebt = GoalDebt
GoalUncertainty = UncertaintyItem
QualityReport = GoalQualityReport
project_current_objective_markdown = project_objective_markdown
parse_objective_markdown = project_objective_markdown
evaluate_goal_quality = lint_goal
GoalLinter = GoalQualityLinter
TypedObjectiveSidecar = ObjectiveTypedGoals


__all__ = [
    "ACCEPTANCE_CRITERION_SCHEMA",
    "ARTIFACT_RECEIPT_OUTPUT_SCHEMA",
    "DATASETS_CONTRACT_GOAL_QUALITY_EVIDENCE_PATH",
    "DATASETS_CONTRACT_GOAL_QUALITY_TEST_PATH",
    "DATASETS_CONTRACT_GOAL_QUALITY_VALIDATION_COMMAND",
    "DEFAULT_TYPED_FRESHNESS_SECONDS",
    "EVIDENCE_PRODUCER_SCHEMA",
    "FRESHNESS_POLICY_SCHEMA",
    "FROZEN_ROOT_SCHEMA",
    "GOAL_DEBT_SCHEMA",
    "GOAL_GRAMMAR_REQUIREMENT_ID",
    "GOAL_QUALITY_REPORT_SCHEMA",
    "GOAL_QUALITY_VERSION",
    "GOAL_SCOPE_SCHEMA",
    "LEGACY_EVIDENCE_OUTPUT_SCHEMA",
    "LEGACY_GOAL_REF_OUTPUT_SCHEMA",
    "OBJECTIVE_TYPED_GOALS_SCHEMA",
    "PYTEST_RECEIPT_OUTPUT_SCHEMA",
    "REFINEMENT_BUDGET_SCHEMA",
    "RESOURCE_ENVELOPE_SCHEMA",
    "TYPED_GOAL_SCHEMA",
    "UNCERTAINTY_ITEM_SCHEMA",
    "UNSUPPORTED_SEMANTIC_SCHEMA",
    "VALIDATION_RULE_SCHEMA",
    "AcceptanceCriterion",
    "DebtSeverity",
    "EvidenceAuthority",
    "EvidenceProducer",
    "FreshnessPolicy",
    "FrozenRootIdentity",
    "GoalAdmissionError",
    "GoalDebt",
    "GoalDebtCode",
    "GoalDebtKind",
    "GoalGrammar",
    "GoalQualityDebt",
    "GoalQualityError",
    "GoalQualityLinter",
    "GoalQualityPolicy",
    "GoalQualityReport",
    "GoalScope",
    "GoalSpecification",
    "GoalUncertainty",
    "GoalLinter",
    "ObjectiveTypedGoals",
    "QualityDebt",
    "QualityReport",
    "RefinementBudget",
    "RepairKind",
    "ResourceEnvelope",
    "TypedGoal",
    "TypedObjectiveSidecar",
    "UncertaintyDebt",
    "UncertaintyDisposition",
    "UncertaintyItem",
    "UnsupportedSemantic",
    "ValidationRule",
    "assert_frozen_root",
    "canonical_goal_bytes",
    "canonical_goal_json",
    "evaluate_goal_quality",
    "goal_from_objective_markdown",
    "is_datasets_contract_goal_quality_evidence_path",
    "lint_goal",
    "lint_objective_markdown",
    "lint_objective_typed_goals",
    "migrate_objective_markdown",
    "parse_objective_markdown",
    "project_current_objective_markdown",
    "project_objective_markdown",
    "score_goal",
    "validate_objective_typed_goals",
    "validate_goal",
]
