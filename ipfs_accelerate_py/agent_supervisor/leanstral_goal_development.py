"""Capability-isolated Leanstral goal-development provider.

This module is intentionally separate from :mod:`leanstral_proof_provider`.
Its only model operation is the versioned ``goal_development.v1`` operation:
it proposes a bounded decomposition using supervisor-owned identifiers.  It
does not expose theorem proving, source mutation, command execution, plan
admission, or kernel checking.

The model sees immutable references rather than canonical source.  Its strict
JSON response can only select identifiers from explicit allowlists.  Every
accepted response is converted into the canonical, unverified contracts in
``goal_development_contracts``.  Expected route failures are represented by a
stable deterministic-fallback result so an optional model cannot stall the
supervisor.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import queue
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final

from .formal_verification_contracts import (
    AssuranceLevel,
    ContractValidationError,
    ResourceBudget,
    canonical_json_bytes,
)
from .formal_verification_provider import (
    CancellationToken,
    ProofProviderError,
    ProviderFailureCode,
)
from .goal_development_contracts import (
    GoalDecompositionDraft,
    GoalDecompositionProposal,
    GoalDevelopmentMode,
    GoalDevelopmentPolicy,
    GoalDevelopmentRequest,
)
from .leanstral_proof_provider import (
    DEFAULT_LEANSTRAL_LLM_PROVIDER,
    DEFAULT_LEANSTRAL_MODEL,
    LeanstralResourceIsolation,
    _default_llm_generate,
)
from .proof_context import estimate_context_tokens


LEANSTRAL_GOAL_DEVELOPMENT_PROVIDER_ID: Final = "leanstral-goal-development"
LEANSTRAL_GOAL_DEVELOPMENT_PROVIDER_VERSION: Final = "1.0.0"
LEANSTRAL_GOAL_DEVELOPMENT_OPERATION: Final = "goal_development.v1"
LEANSTRAL_GOAL_DEVELOPMENT_OPERATION_VERSION: Final = 1
LEANSTRAL_GOAL_DEVELOPMENT_REQUEST_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/leanstral-goal-development-request@1"
)
LEANSTRAL_GOAL_DEVELOPMENT_CONTEXT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/leanstral-goal-development-context@1"
)
LEANSTRAL_GOAL_DEVELOPMENT_OUTPUT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/leanstral-goal-development-output@1"
)
LEANSTRAL_GOAL_DEVELOPMENT_RESULT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/leanstral-goal-development-result@1"
)

DEFAULT_GOAL_DEVELOPMENT_TIMEOUT_SECONDS: Final = 60.0
DEFAULT_GOAL_DEVELOPMENT_MAX_NEW_TOKENS: Final = 2_048
DEFAULT_GOAL_DEVELOPMENT_MAX_CONTEXT_BYTES: Final = 256 * 1024
DEFAULT_GOAL_DEVELOPMENT_MAX_CONTEXT_TOKENS: Final = 32 * 1024
DEFAULT_GOAL_DEVELOPMENT_MAX_OUTPUT_BYTES: Final = 256 * 1024
DEFAULT_GOAL_DEVELOPMENT_MAX_RECORDS_PER_KIND: Final = 64
DEFAULT_GOAL_DEVELOPMENT_MAX_CONCURRENT_REQUESTS: Final = 1

_ROUTER_ALIASES = frozenset({"", "auto", "default", "llm_router", "router"})
_FORBIDDEN_RESPONSE_KEYS = frozenset(
    {
        "canonical_source",
        "canonical_sources",
        "source_code",
        "source_text",
        "root_goal",
        "root_goal_content_id",
        "root_goal_digest",
        "satisfaction_formula",
        "formula",
        "formula_text",
        "assumption",
        "assumptions",
        "command",
        "commands",
        "shell",
        "shell_command",
        "validation_command",
        "validation_commands",
        "kernel_check",
        "kernel_checked",
        "prove",
        "proof",
        "proof_text",
        "verified",
        "admitted",
        "complete",
    }
)

LLMGenerate = Callable[..., str]


def _text(
    value: Any,
    *,
    field_name: str,
    required: bool = True,
    max_bytes: int = 16 * 1024,
) -> str:
    if value is None and not required:
        return ""
    if not isinstance(value, str):
        raise ContractValidationError(f"{field_name} must be a string")
    normalized = value.strip()
    if required and not normalized:
        raise ContractValidationError(f"{field_name} must not be empty")
    if "\x00" in normalized:
        raise ContractValidationError(f"{field_name} must not contain NUL bytes")
    if len(normalized.encode("utf-8")) > max_bytes:
        raise ContractValidationError(f"{field_name} exceeds its byte limit")
    return normalized


def _ids(
    value: Any,
    *,
    field_name: str,
    required: bool = False,
) -> tuple[str, ...]:
    if value is None:
        source: Sequence[Any] = ()
    elif isinstance(value, str):
        source = (value,)
    elif isinstance(value, Sequence) and not isinstance(
        value, (bytes, bytearray, memoryview)
    ):
        source = value
    else:
        raise ContractValidationError(f"{field_name} must be an array of strings")
    result: list[str] = []
    for index, item in enumerate(source):
        normalized = _text(item, field_name=f"{field_name}[{index}]")
        if normalized not in result:
            result.append(normalized)
    if required and not result:
        raise ContractValidationError(f"{field_name} must not be empty")
    return tuple(sorted(result))


def _positive_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractValidationError(f"{field_name} must be a positive integer")
    return value


def _strict_object(text: str) -> dict[str, Any]:
    def object_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON object key")
            result[key] = value
        return result

    try:
        value = json.loads(
            text,
            object_pairs_hook=object_pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"non-finite number {item}")
            ),
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ContractValidationError(
            "Leanstral goal-development output must be strict JSON"
        ) from exc
    if not isinstance(value, dict):
        raise ContractValidationError(
            "Leanstral goal-development output must be a JSON object"
        )
    return value


def _reject_unknown(
    value: Mapping[str, Any],
    allowed: set[str] | frozenset[str],
    *,
    field_name: str,
) -> None:
    if set(value).difference(allowed):
        raise ContractValidationError(f"{field_name} contains unknown fields")


def _resource_budget(value: ResourceBudget | Mapping[str, Any] | None) -> ResourceBudget:
    if value is None:
        return ResourceBudget()
    if isinstance(value, ResourceBudget):
        return value
    if isinstance(value, Mapping):
        return ResourceBudget.from_dict(value)
    raise ContractValidationError("resource_budget must be an object")


@dataclass(frozen=True)
class ImmutableGoalRecord:
    """Reference-only rendering of the frozen root goal."""

    goal_id: str
    content_id: str
    satisfaction_formula_id: str
    title: str = ""

    def __post_init__(self) -> None:
        for name in ("goal_id", "content_id", "satisfaction_formula_id"):
            object.__setattr__(
                self, name, _text(getattr(self, name), field_name=name)
            )
        object.__setattr__(
            self,
            "title",
            _text(self.title, field_name="title", required=False),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "record_kind": "immutable_goal",
            "goal_id": self.goal_id,
            "content_id": self.content_id,
            "satisfaction_formula_id": self.satisfaction_formula_id,
            "title": self.title,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ImmutableGoalRecord":
        if not isinstance(value, Mapping):
            raise ContractValidationError("goal record must be an object")
        _reject_unknown(
            value,
            {
                "record_kind",
                "goal_id",
                "content_id",
                "root_goal_content_id",
                "satisfaction_formula_id",
                "title",
            },
            field_name="goal record",
        )
        if value.get("record_kind", "immutable_goal") != "immutable_goal":
            raise ContractValidationError("goal record has the wrong kind")
        return cls(
            goal_id=value.get("goal_id", ""),
            content_id=value.get(
                "content_id", value.get("root_goal_content_id", "")
            ),
            satisfaction_formula_id=value.get("satisfaction_formula_id", ""),
            title=value.get("title", ""),
        )


@dataclass(frozen=True)
class EvidenceGapRecord:
    gap_id: str
    evidence_requirement_ids: tuple[str, ...]
    reason_code: str = "missing_evidence"

    def __post_init__(self) -> None:
        object.__setattr__(self, "gap_id", _text(self.gap_id, field_name="gap_id"))
        object.__setattr__(
            self,
            "evidence_requirement_ids",
            _ids(
                self.evidence_requirement_ids,
                field_name="evidence_requirement_ids",
                required=True,
            ),
        )
        object.__setattr__(
            self, "reason_code", _text(self.reason_code, field_name="reason_code")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_kind": "evidence_gap",
            "gap_id": self.gap_id,
            "evidence_requirement_ids": list(self.evidence_requirement_ids),
            "reason_code": self.reason_code,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceGapRecord":
        _reject_unknown(
            value,
            {
                "record_kind",
                "gap_id",
                "evidence_requirement_ids",
                "evidence_ids",
                "reason_code",
            },
            field_name="evidence-gap record",
        )
        if value.get("record_kind", "evidence_gap") != "evidence_gap":
            raise ContractValidationError("evidence-gap record has the wrong kind")
        return cls(
            gap_id=value.get("gap_id", ""),
            evidence_requirement_ids=tuple(
                value.get(
                    "evidence_requirement_ids", value.get("evidence_ids", ())
                )
                or ()
            ),
            reason_code=value.get("reason_code", "missing_evidence"),
        )


class CodeReferenceKind(str, Enum):
    AST = "ast"
    GRAPHRAG = "graphrag"


@dataclass(frozen=True)
class ASTGraphRAGReferenceRecord:
    reference_id: str
    reference_kind: CodeReferenceKind | str
    repository_tree_id: str
    scope_ids: tuple[str, ...]
    symbol_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reference_id",
            _text(self.reference_id, field_name="reference_id"),
        )
        try:
            kind = CodeReferenceKind(
                str(getattr(self.reference_kind, "value", self.reference_kind))
            )
        except ValueError as exc:
            raise ContractValidationError(
                "reference_kind must be ast or graphrag"
            ) from exc
        object.__setattr__(self, "reference_kind", kind)
        object.__setattr__(
            self,
            "repository_tree_id",
            _text(self.repository_tree_id, field_name="repository_tree_id"),
        )
        object.__setattr__(
            self, "scope_ids", _ids(self.scope_ids, field_name="scope_ids", required=True)
        )
        object.__setattr__(
            self, "symbol_ids", _ids(self.symbol_ids, field_name="symbol_ids")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_kind": "code_reference",
            "reference_id": self.reference_id,
            "reference_kind": self.reference_kind.value,
            "repository_tree_id": self.repository_tree_id,
            "scope_ids": list(self.scope_ids),
            "symbol_ids": list(self.symbol_ids),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ASTGraphRAGReferenceRecord":
        _reject_unknown(
            value,
            {
                "record_kind",
                "reference_id",
                "reference_kind",
                "repository_tree_id",
                "scope_ids",
                "symbol_ids",
            },
            field_name="AST/GraphRAG reference",
        )
        if value.get("record_kind", "code_reference") != "code_reference":
            raise ContractValidationError("code-reference record has the wrong kind")
        return cls(
            reference_id=value.get("reference_id", ""),
            reference_kind=value.get("reference_kind", ""),
            repository_tree_id=value.get("repository_tree_id", ""),
            scope_ids=tuple(value.get("scope_ids") or ()),
            symbol_ids=tuple(value.get("symbol_ids") or ()),
        )


@dataclass(frozen=True)
class CapabilityRecord:
    capability_id: str
    resource_ids: tuple[str, ...]
    validation_check_ids: tuple[str, ...]
    available: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "capability_id", _text(self.capability_id, field_name="capability_id")
        )
        object.__setattr__(
            self, "resource_ids", _ids(self.resource_ids, field_name="resource_ids")
        )
        object.__setattr__(
            self,
            "validation_check_ids",
            _ids(self.validation_check_ids, field_name="validation_check_ids"),
        )
        if not isinstance(self.available, bool):
            raise ContractValidationError("available must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_kind": "capability",
            "capability_id": self.capability_id,
            "resource_ids": list(self.resource_ids),
            "validation_check_ids": list(self.validation_check_ids),
            "available": self.available,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CapabilityRecord":
        _reject_unknown(
            value,
            {
                "record_kind",
                "capability_id",
                "resource_ids",
                "validation_check_ids",
                "available",
            },
            field_name="capability record",
        )
        if value.get("record_kind", "capability") != "capability":
            raise ContractValidationError("capability record has the wrong kind")
        return cls(
            capability_id=value.get("capability_id", ""),
            resource_ids=tuple(value.get("resource_ids") or ()),
            validation_check_ids=tuple(value.get("validation_check_ids") or ()),
            available=value.get("available", True),
        )


@dataclass(frozen=True)
class PriorCounterexampleRecord:
    counterexample_id: str
    reason_code: str
    proposal_ids: tuple[str, ...] = ()
    validation_check_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "counterexample_id",
            _text(self.counterexample_id, field_name="counterexample_id"),
        )
        object.__setattr__(
            self, "reason_code", _text(self.reason_code, field_name="reason_code")
        )
        object.__setattr__(
            self, "proposal_ids", _ids(self.proposal_ids, field_name="proposal_ids")
        )
        object.__setattr__(
            self,
            "validation_check_ids",
            _ids(self.validation_check_ids, field_name="validation_check_ids"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_kind": "prior_counterexample",
            "counterexample_id": self.counterexample_id,
            "reason_code": self.reason_code,
            "proposal_ids": list(self.proposal_ids),
            "validation_check_ids": list(self.validation_check_ids),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PriorCounterexampleRecord":
        _reject_unknown(
            value,
            {
                "record_kind",
                "counterexample_id",
                "reason_code",
                "proposal_ids",
                "validation_check_ids",
            },
            field_name="prior-counterexample record",
        )
        if value.get("record_kind", "prior_counterexample") != "prior_counterexample":
            raise ContractValidationError(
                "prior-counterexample record has the wrong kind"
            )
        return cls(
            counterexample_id=value.get("counterexample_id", ""),
            reason_code=value.get("reason_code", ""),
            proposal_ids=tuple(value.get("proposal_ids") or ()),
            validation_check_ids=tuple(value.get("validation_check_ids") or ()),
        )


@dataclass(frozen=True)
class ReusableReceiptRecord:
    receipt_id: str
    evidence_requirement_ids: tuple[str, ...]
    assurance_id: str
    scope_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "receipt_id", _text(self.receipt_id, field_name="receipt_id")
        )
        object.__setattr__(
            self,
            "evidence_requirement_ids",
            _ids(
                self.evidence_requirement_ids,
                field_name="evidence_requirement_ids",
                required=True,
            ),
        )
        object.__setattr__(
            self, "assurance_id", _text(self.assurance_id, field_name="assurance_id")
        )
        object.__setattr__(
            self, "scope_ids", _ids(self.scope_ids, field_name="scope_ids", required=True)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_kind": "reusable_receipt",
            "receipt_id": self.receipt_id,
            "evidence_requirement_ids": list(self.evidence_requirement_ids),
            "assurance_id": self.assurance_id,
            "scope_ids": list(self.scope_ids),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReusableReceiptRecord":
        _reject_unknown(
            value,
            {
                "record_kind",
                "receipt_id",
                "evidence_requirement_ids",
                "evidence_ids",
                "assurance_id",
                "scope_ids",
            },
            field_name="reusable-receipt record",
        )
        if value.get("record_kind", "reusable_receipt") != "reusable_receipt":
            raise ContractValidationError("reusable-receipt record has the wrong kind")
        return cls(
            receipt_id=value.get("receipt_id", ""),
            evidence_requirement_ids=tuple(
                value.get(
                    "evidence_requirement_ids", value.get("evidence_ids", ())
                )
                or ()
            ),
            assurance_id=value.get("assurance_id", ""),
            scope_ids=tuple(value.get("scope_ids") or ()),
        )


@dataclass(frozen=True)
class GoalDevelopmentTemplate:
    """Reviewed template binding a model selection to a typed formula ID."""

    template_id: str
    satisfaction_formula_id: str
    evidence_requirement_ids: tuple[str, ...]
    assurance_ids: tuple[str, ...]
    resource_ids: tuple[str, ...]
    scope_ids: tuple[str, ...]
    validation_check_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("template_id", "satisfaction_formula_id"):
            object.__setattr__(
                self, name, _text(getattr(self, name), field_name=name)
            )
        for name in (
            "evidence_requirement_ids",
            "assurance_ids",
            "resource_ids",
            "scope_ids",
            "validation_check_ids",
        ):
            object.__setattr__(
                self,
                name,
                _ids(
                    getattr(self, name),
                    field_name=name,
                    required=True,
                ),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "satisfaction_formula_id": self.satisfaction_formula_id,
            "evidence_requirement_ids": list(self.evidence_requirement_ids),
            "assurance_ids": list(self.assurance_ids),
            "resource_ids": list(self.resource_ids),
            "scope_ids": list(self.scope_ids),
            "validation_check_ids": list(self.validation_check_ids),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GoalDevelopmentTemplate":
        _reject_unknown(
            value,
            {
                "template_id",
                "satisfaction_formula_id",
                "evidence_requirement_ids",
                "evidence_ids",
                "assurance_ids",
                "resource_ids",
                "scope_ids",
                "validation_check_ids",
            },
            field_name="goal template",
        )
        return cls(
            template_id=value.get("template_id", ""),
            satisfaction_formula_id=value.get("satisfaction_formula_id", ""),
            evidence_requirement_ids=tuple(
                value.get(
                    "evidence_requirement_ids", value.get("evidence_ids", ())
                )
                or ()
            ),
            assurance_ids=tuple(value.get("assurance_ids") or ()),
            resource_ids=tuple(value.get("resource_ids") or ()),
            scope_ids=tuple(value.get("scope_ids") or ()),
            validation_check_ids=tuple(value.get("validation_check_ids") or ()),
        )


def _records(
    values: Any,
    record_type: type[Any],
    *,
    field_name: str,
    maximum: int,
) -> tuple[Any, ...]:
    if values is None:
        values = ()
    if not isinstance(values, Sequence) or isinstance(
        values, (str, bytes, bytearray, memoryview)
    ):
        raise ContractValidationError(f"{field_name} must be an array")
    if len(values) > maximum:
        raise ContractValidationError(f"{field_name} exceeds its record limit")
    normalized: list[Any] = []
    for item in values:
        if isinstance(item, record_type):
            normalized.append(item)
        elif isinstance(item, Mapping):
            normalized.append(record_type.from_dict(item))
        else:
            raise ContractValidationError(
                f"{field_name} entries must be objects"
            )
    result = tuple(normalized)
    return tuple(sorted(result, key=lambda item: canonical_json_bytes(item.to_dict())))


@dataclass(frozen=True)
class GoalDevelopmentContext:
    """Bounded, reference-only context offered to the model."""

    goal: ImmutableGoalRecord
    templates: tuple[GoalDevelopmentTemplate, ...]
    evidence_gaps: tuple[EvidenceGapRecord, ...] = ()
    code_references: tuple[ASTGraphRAGReferenceRecord, ...] = ()
    capabilities: tuple[CapabilityRecord, ...] = ()
    prior_counterexamples: tuple[PriorCounterexampleRecord, ...] = ()
    reusable_receipts: tuple[ReusableReceiptRecord, ...] = ()
    max_records_per_kind: int = DEFAULT_GOAL_DEVELOPMENT_MAX_RECORDS_PER_KIND
    schema: str = LEANSTRAL_GOAL_DEVELOPMENT_CONTEXT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != LEANSTRAL_GOAL_DEVELOPMENT_CONTEXT_SCHEMA:
            raise ContractValidationError("unsupported goal-development context schema")
        maximum = _positive_int(
            self.max_records_per_kind, field_name="max_records_per_kind"
        )
        goal = (
            self.goal
            if isinstance(self.goal, ImmutableGoalRecord)
            else ImmutableGoalRecord.from_dict(self.goal)
        )
        object.__setattr__(self, "goal", goal)
        for name, record_type in (
            ("templates", GoalDevelopmentTemplate),
            ("evidence_gaps", EvidenceGapRecord),
            ("code_references", ASTGraphRAGReferenceRecord),
            ("capabilities", CapabilityRecord),
            ("prior_counterexamples", PriorCounterexampleRecord),
            ("reusable_receipts", ReusableReceiptRecord),
        ):
            object.__setattr__(
                self,
                name,
                _records(
                    getattr(self, name),
                    record_type,
                    field_name=name,
                    maximum=maximum,
                ),
            )
        if not self.templates:
            raise ContractValidationError("templates must not be empty")
        template_ids = [item.template_id for item in self.templates]
        if len(template_ids) != len(set(template_ids)):
            raise ContractValidationError("template IDs must be unique")

    @property
    def allowed_template_ids(self) -> tuple[str, ...]:
        return tuple(sorted(item.template_id for item in self.templates))

    @property
    def allowed_evidence_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    item
                    for template in self.templates
                    for item in template.evidence_requirement_ids
                }
            )
        )

    @property
    def allowed_assurance_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    item
                    for template in self.templates
                    for item in template.assurance_ids
                }
            )
        )

    @property
    def allowed_resource_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    item
                    for template in self.templates
                    for item in template.resource_ids
                }
            )
        )

    @property
    def allowed_scope_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    item for template in self.templates for item in template.scope_ids
                }
            )
        )

    @property
    def allowed_validation_check_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    item
                    for template in self.templates
                    for item in template.validation_check_ids
                }
            )
        )

    def validate_request(self, request: GoalDevelopmentRequest) -> None:
        expected_goal = (
            request.root_goal_id,
            request.root_goal_content_id,
            request.satisfaction_formula_id,
        )
        actual_goal = (
            self.goal.goal_id,
            self.goal.content_id,
            self.goal.satisfaction_formula_id,
        )
        if actual_goal != expected_goal:
            raise ContractValidationError(
                "context goal does not match the immutable request root"
            )
        request_evidence = set(request.evidence_requirement_ids)
        request_scopes = set(request.scope_ids)
        if not set(self.allowed_evidence_ids).issubset(request_evidence):
            raise ContractValidationError(
                "context templates escape request evidence requirements"
            )
        if not set(self.allowed_scope_ids).issubset(request_scopes):
            raise ContractValidationError("context templates escape request scope")
        for gap in self.evidence_gaps:
            if not set(gap.evidence_requirement_ids).issubset(request_evidence):
                raise ContractValidationError(
                    "evidence-gap record escapes request evidence requirements"
                )
        for reference in self.code_references:
            if reference.repository_tree_id != request.repository_tree_id:
                raise ContractValidationError(
                    "code reference uses a different repository tree"
                )
            if not set(reference.scope_ids).issubset(request_scopes):
                raise ContractValidationError("code reference escapes request scope")
        allowed_resources = set(self.allowed_resource_ids)
        allowed_checks = set(self.allowed_validation_check_ids)
        for capability in self.capabilities:
            if not set(capability.resource_ids).issubset(allowed_resources):
                raise ContractValidationError(
                    "capability record contains a non-allowlisted resource ID"
                )
            if not set(capability.validation_check_ids).issubset(allowed_checks):
                raise ContractValidationError(
                    "capability record contains a non-allowlisted validation-check ID"
                )
        for counterexample in self.prior_counterexamples:
            if not set(counterexample.validation_check_ids).issubset(
                allowed_checks
            ):
                raise ContractValidationError(
                    "counterexample record contains a non-allowlisted validation-check ID"
                )
        for receipt in self.reusable_receipts:
            if not set(receipt.evidence_requirement_ids).issubset(request_evidence):
                raise ContractValidationError(
                    "reusable receipt escapes request evidence requirements"
                )
            if not set(receipt.scope_ids).issubset(request_scopes):
                raise ContractValidationError("reusable receipt escapes request scope")
            if receipt.assurance_id not in self.allowed_assurance_ids:
                raise ContractValidationError(
                    "reusable receipt assurance ID is not allowlisted"
                )

    @property
    def context_id(self) -> str:
        return "leanstral-goal-context-" + hashlib.sha256(
            canonical_json_bytes(self.to_dict())
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "goal": self.goal.to_dict(),
            "evidence_gaps": [item.to_dict() for item in self.evidence_gaps],
            "code_references": [item.to_dict() for item in self.code_references],
            "capabilities": [item.to_dict() for item in self.capabilities],
            "prior_counterexamples": [
                item.to_dict() for item in self.prior_counterexamples
            ],
            "reusable_receipts": [
                item.to_dict() for item in self.reusable_receipts
            ],
            "templates": [item.to_dict() for item in self.templates],
            "allowlists": {
                "template_ids": list(self.allowed_template_ids),
                "evidence_ids": list(self.allowed_evidence_ids),
                "assurance_ids": list(self.allowed_assurance_ids),
                "resource_ids": list(self.allowed_resource_ids),
                "scope_ids": list(self.allowed_scope_ids),
                "validation_check_ids": list(
                    self.allowed_validation_check_ids
                ),
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GoalDevelopmentContext":
        if not isinstance(value, Mapping):
            raise ContractValidationError("goal-development context must be an object")
        _reject_unknown(
            value,
            {
                "schema",
                "goal",
                "templates",
                "evidence_gaps",
                "code_references",
                "ast_graphrag_references",
                "capabilities",
                "prior_counterexamples",
                "reusable_receipts",
                "max_records_per_kind",
                "allowlists",
            },
            field_name="goal-development context",
        )
        result = cls(
            schema=value.get("schema", LEANSTRAL_GOAL_DEVELOPMENT_CONTEXT_SCHEMA),
            goal=ImmutableGoalRecord.from_dict(value.get("goal") or {}),
            templates=tuple(value.get("templates") or ()),
            evidence_gaps=tuple(value.get("evidence_gaps") or ()),
            code_references=tuple(
                value.get(
                    "code_references", value.get("ast_graphrag_references", ())
                )
                or ()
            ),
            capabilities=tuple(value.get("capabilities") or ()),
            prior_counterexamples=tuple(
                value.get("prior_counterexamples") or ()
            ),
            reusable_receipts=tuple(value.get("reusable_receipts") or ()),
            max_records_per_kind=value.get(
                "max_records_per_kind",
                DEFAULT_GOAL_DEVELOPMENT_MAX_RECORDS_PER_KIND,
            ),
        )
        supplied_allowlists = value.get("allowlists")
        if supplied_allowlists is not None:
            if not isinstance(supplied_allowlists, Mapping):
                raise ContractValidationError("allowlists must be an object")
            expected = result.to_dict()["allowlists"]
            normalized = {
                name: list(_ids(items, field_name=f"allowlists.{name}"))
                for name, items in supplied_allowlists.items()
            }
            if normalized != expected:
                raise ContractValidationError(
                    "context allowlists must be derived from reviewed templates"
                )
        return result


def build_leanstral_goal_development_context(
    request: GoalDevelopmentRequest | Mapping[str, Any],
    *,
    templates: Sequence[GoalDevelopmentTemplate | Mapping[str, Any]],
    goal: ImmutableGoalRecord | Mapping[str, Any] | None = None,
    evidence_gaps: Sequence[EvidenceGapRecord | Mapping[str, Any]] = (),
    code_references: Sequence[
        ASTGraphRAGReferenceRecord | Mapping[str, Any]
    ] = (),
    capabilities: Sequence[CapabilityRecord | Mapping[str, Any]] = (),
    prior_counterexamples: Sequence[
        PriorCounterexampleRecord | Mapping[str, Any]
    ] = (),
    reusable_receipts: Sequence[ReusableReceiptRecord | Mapping[str, Any]] = (),
    max_records_per_kind: int = DEFAULT_GOAL_DEVELOPMENT_MAX_RECORDS_PER_KIND,
) -> GoalDevelopmentContext:
    """Construct and validate a bounded context against one frozen request."""

    frozen_request = (
        request
        if isinstance(request, GoalDevelopmentRequest)
        else GoalDevelopmentRequest.from_dict(request)
    )
    immutable_goal = (
        ImmutableGoalRecord(
            goal_id=frozen_request.root_goal_id,
            content_id=frozen_request.root_goal_content_id,
            satisfaction_formula_id=frozen_request.satisfaction_formula_id,
        )
        if goal is None
        else (
            goal
            if isinstance(goal, ImmutableGoalRecord)
            else ImmutableGoalRecord.from_dict(goal)
        )
    )
    context = GoalDevelopmentContext(
        goal=immutable_goal,
        templates=tuple(templates),
        evidence_gaps=tuple(evidence_gaps),
        code_references=tuple(code_references),
        capabilities=tuple(capabilities),
        prior_counterexamples=tuple(prior_counterexamples),
        reusable_receipts=tuple(reusable_receipts),
        max_records_per_kind=max_records_per_kind,
    )
    context.validate_request(frozen_request)
    return context


@dataclass(frozen=True)
class LeanstralGoalDevelopmentInvocation:
    """Versioned operation envelope independent of the proof protocol."""

    request: GoalDevelopmentRequest
    policy: GoalDevelopmentPolicy
    context: GoalDevelopmentContext
    resource_budget: ResourceBudget = field(default_factory=ResourceBudget)
    network_allowed: bool = False
    deadline_unix_ms: int | None = None
    operation: str = LEANSTRAL_GOAL_DEVELOPMENT_OPERATION
    schema: str = LEANSTRAL_GOAL_DEVELOPMENT_REQUEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != LEANSTRAL_GOAL_DEVELOPMENT_REQUEST_SCHEMA:
            raise ContractValidationError("unsupported goal-development request schema")
        if self.operation != LEANSTRAL_GOAL_DEVELOPMENT_OPERATION:
            raise ContractValidationError(
                "unsupported Leanstral goal-development operation"
            )
        request = (
            self.request
            if isinstance(self.request, GoalDevelopmentRequest)
            else GoalDevelopmentRequest.from_dict(self.request)
        )
        policy = (
            self.policy
            if isinstance(self.policy, GoalDevelopmentPolicy)
            else GoalDevelopmentPolicy.from_dict(self.policy)
        )
        context = (
            self.context
            if isinstance(self.context, GoalDevelopmentContext)
            else GoalDevelopmentContext.from_dict(self.context)
        )
        request.require_policy(policy)
        context.validate_request(request)
        if request.mode is GoalDevelopmentMode.OFF:
            raise ContractValidationError("off mode cannot invoke goal development")
        if not isinstance(self.network_allowed, bool):
            raise ContractValidationError("network_allowed must be a boolean")
        if self.deadline_unix_ms is not None and (
            isinstance(self.deadline_unix_ms, bool)
            or not isinstance(self.deadline_unix_ms, int)
            or self.deadline_unix_ms < 0
        ):
            raise ContractValidationError(
                "deadline_unix_ms must be a non-negative integer or null"
            )
        object.__setattr__(self, "request", request)
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "context", context)
        object.__setattr__(
            self, "resource_budget", _resource_budget(self.resource_budget)
        )

    @property
    def request_id(self) -> str:
        return self.request.request_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "operation": self.operation,
            "request": self.request.to_dict(),
            "policy": self.policy.to_dict(),
            "context": self.context.to_dict(),
            "resource_budget": self.resource_budget.to_dict(),
            "network_allowed": self.network_allowed,
            "deadline_unix_ms": self.deadline_unix_ms,
        }

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "LeanstralGoalDevelopmentInvocation":
        if not isinstance(value, Mapping):
            raise ContractValidationError(
                "goal-development invocation must be an object"
            )
        _reject_unknown(
            value,
            {
                "schema",
                "operation",
                "request",
                "goal_request",
                "policy",
                "context",
                "resource_budget",
                "network_allowed",
                "deadline_unix_ms",
            },
            field_name="goal-development invocation",
        )
        return cls(
            schema=value.get("schema", LEANSTRAL_GOAL_DEVELOPMENT_REQUEST_SCHEMA),
            operation=value.get(
                "operation", LEANSTRAL_GOAL_DEVELOPMENT_OPERATION
            ),
            request=GoalDevelopmentRequest.from_dict(
                value.get("request", value.get("goal_request")) or {}
            ),
            policy=GoalDevelopmentPolicy.from_dict(value.get("policy") or {}),
            context=GoalDevelopmentContext.from_dict(value.get("context") or {}),
            resource_budget=_resource_budget(value.get("resource_budget")),
            network_allowed=value.get("network_allowed", False),
            deadline_unix_ms=value.get("deadline_unix_ms"),
        )


class GoalDevelopmentResultStatus(str, Enum):
    DRAFT = "draft"
    DETERMINISTIC_FALLBACK = "deterministic_fallback"


class GoalDevelopmentFallbackReason(str, Enum):
    UNAVAILABLE = "unavailable"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    MALFORMED_OUTPUT = "malformed_output"
    OVERLOADED = "overloaded"


@dataclass(frozen=True)
class GoalDevelopmentProviderResult(Mapping[str, Any]):
    """Stable result carrying either an unverified draft or fallback signal."""

    request_id: str
    status: GoalDevelopmentResultStatus
    draft: GoalDecompositionDraft | None = None
    fallback_reason: GoalDevelopmentFallbackReason | None = None
    provider_id: str = LEANSTRAL_GOAL_DEVELOPMENT_PROVIDER_ID
    provider_version: str = LEANSTRAL_GOAL_DEVELOPMENT_PROVIDER_VERSION
    operation: str = LEANSTRAL_GOAL_DEVELOPMENT_OPERATION
    schema: str = LEANSTRAL_GOAL_DEVELOPMENT_RESULT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "request_id", _text(self.request_id, field_name="request_id")
        )
        status = (
            self.status
            if isinstance(self.status, GoalDevelopmentResultStatus)
            else GoalDevelopmentResultStatus(str(self.status))
        )
        reason = (
            self.fallback_reason
            if isinstance(self.fallback_reason, GoalDevelopmentFallbackReason)
            else (
                None
                if self.fallback_reason is None
                else GoalDevelopmentFallbackReason(str(self.fallback_reason))
            )
        )
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "fallback_reason", reason)
        if status is GoalDevelopmentResultStatus.DRAFT:
            if not isinstance(self.draft, GoalDecompositionDraft) or reason is not None:
                raise ContractValidationError("draft result has invalid payload")
            self.draft.validate_request(self.draft.request)
            if self.draft.request_id != self.request_id:
                raise ContractValidationError("draft result request ID mismatch")
        elif self.draft is not None or reason is None:
            raise ContractValidationError("fallback result has invalid payload")

    @property
    def used_fallback(self) -> bool:
        return self.status is GoalDevelopmentResultStatus.DETERMINISTIC_FALLBACK

    @property
    def deterministic_fallback(self) -> bool:
        return self.used_fallback

    @property
    def assurance(self) -> AssuranceLevel:
        return AssuranceLevel.UNVERIFIED

    @property
    def result_id(self) -> str:
        return "leanstral-goal-result-" + hashlib.sha256(
            canonical_json_bytes(self.to_dict(include_id=False))
        ).hexdigest()

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "operation": self.operation,
            "provider_id": self.provider_id,
            "provider_version": self.provider_version,
            "request_id": self.request_id,
            "status": self.status.value,
            "deterministic_fallback": self.used_fallback,
            "fallback_reason": (
                None if self.fallback_reason is None else self.fallback_reason.value
            ),
            "failure_code": (
                None if self.fallback_reason is None else self.fallback_reason.value
            ),
            "draft": None if self.draft is None else self.draft.to_dict(),
            "assurance": AssuranceLevel.UNVERIFIED.value,
            "authoritative": False,
            "verified": False,
            "admitted": False,
            "complete": False,
            "kernel_checked": False,
            "can_mutate_root": False,
            "can_mutate_canonical_source": False,
            "can_execute_commands": False,
        }
        if include_id:
            payload["result_id"] = self.result_id
        return payload

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "GoalDevelopmentProviderResult":
        if not isinstance(value, Mapping):
            raise ContractValidationError(
                "goal-development result must be an object"
            )
        _reject_unknown(
            value,
            {
                "schema",
                "operation",
                "provider_id",
                "provider_version",
                "request_id",
                "result_id",
                "status",
                "deterministic_fallback",
                "fallback_reason",
                "failure_code",
                "draft",
                "assurance",
                "authoritative",
                "verified",
                "admitted",
                "complete",
                "kernel_checked",
                "can_mutate_root",
                "can_mutate_canonical_source",
                "can_execute_commands",
            },
            field_name="goal-development result",
        )
        for name in (
            "authoritative",
            "verified",
            "admitted",
            "complete",
            "kernel_checked",
            "can_mutate_root",
            "can_mutate_canonical_source",
            "can_execute_commands",
        ):
            if value.get(name, False) is not False:
                raise ContractValidationError(
                    "goal-development result cannot claim authority"
                )
        if value.get("assurance", AssuranceLevel.UNVERIFIED.value) != (
            AssuranceLevel.UNVERIFIED.value
        ):
            raise ContractValidationError(
                "goal-development result cannot claim proof assurance"
            )
        reason = value.get("fallback_reason", value.get("failure_code"))
        if (
            value.get("failure_code") not in (None, reason)
            or value.get("provider_id", LEANSTRAL_GOAL_DEVELOPMENT_PROVIDER_ID)
            != LEANSTRAL_GOAL_DEVELOPMENT_PROVIDER_ID
            or value.get(
                "provider_version", LEANSTRAL_GOAL_DEVELOPMENT_PROVIDER_VERSION
            )
            != LEANSTRAL_GOAL_DEVELOPMENT_PROVIDER_VERSION
            or value.get("operation", LEANSTRAL_GOAL_DEVELOPMENT_OPERATION)
            != LEANSTRAL_GOAL_DEVELOPMENT_OPERATION
            or value.get("schema", LEANSTRAL_GOAL_DEVELOPMENT_RESULT_SCHEMA)
            != LEANSTRAL_GOAL_DEVELOPMENT_RESULT_SCHEMA
        ):
            raise ContractValidationError(
                "goal-development result bindings are invalid"
            )
        result = cls(
            request_id=value.get("request_id", ""),
            status=value.get("status", ""),
            draft=(
                None
                if value.get("draft") is None
                else GoalDecompositionDraft.from_dict(value["draft"])
            ),
            fallback_reason=reason,
        )
        if value.get("deterministic_fallback", result.used_fallback) is not (
            result.used_fallback
        ):
            raise ContractValidationError(
                "goal-development fallback marker is invalid"
            )
        if value.get("result_id") not in (None, "", result.result_id):
            raise ContractValidationError(
                "goal-development result identity is invalid"
            )
        return result


@dataclass(frozen=True)
class LeanstralGoalDevelopmentCapability:
    provider_id: str = LEANSTRAL_GOAL_DEVELOPMENT_PROVIDER_ID
    provider_version: str = LEANSTRAL_GOAL_DEVELOPMENT_PROVIDER_VERSION
    operation: str = LEANSTRAL_GOAL_DEVELOPMENT_OPERATION
    resource_isolation: LeanstralResourceIsolation = field(
        default_factory=LeanstralResourceIsolation
    )

    def supports(self, operation: str) -> bool:
        return str(operation) == self.operation

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "provider_version": self.provider_version,
            "operations": [self.operation],
            "proof_operations": [],
            "draft_only": True,
            "assurance": AssuranceLevel.UNVERIFIED.value,
            "authoritative": False,
            "kernel_check_supported": False,
            "command_execution_supported": False,
            "canonical_source_access": False,
            "resource_classes": self.resource_isolation.to_dict(),
            "response_schema": LEANSTRAL_GOAL_DEVELOPMENT_OUTPUT_SCHEMA,
        }


@dataclass(frozen=True)
class LeanstralGoalDevelopmentProviderConfig:
    llm_provider: str = DEFAULT_LEANSTRAL_LLM_PROVIDER
    model: str = DEFAULT_LEANSTRAL_MODEL
    timeout_seconds: float = DEFAULT_GOAL_DEVELOPMENT_TIMEOUT_SECONDS
    max_new_tokens: int = DEFAULT_GOAL_DEVELOPMENT_MAX_NEW_TOKENS
    max_context_bytes: int = DEFAULT_GOAL_DEVELOPMENT_MAX_CONTEXT_BYTES
    max_context_tokens: int = DEFAULT_GOAL_DEVELOPMENT_MAX_CONTEXT_TOKENS
    max_output_bytes: int = DEFAULT_GOAL_DEVELOPMENT_MAX_OUTPUT_BYTES
    max_records_per_kind: int = DEFAULT_GOAL_DEVELOPMENT_MAX_RECORDS_PER_KIND
    max_concurrent_requests: int = DEFAULT_GOAL_DEVELOPMENT_MAX_CONCURRENT_REQUESTS
    temperature: float = 0.0
    vibe_agent: str = "lean"
    network_access_required: bool = False
    resource_isolation: LeanstralResourceIsolation = field(
        default_factory=LeanstralResourceIsolation
    )
    provider: str | None = None

    def __post_init__(self) -> None:
        route = self.provider or self.llm_provider
        route = _text(route, field_name="llm_provider")
        if route.casefold() in _ROUTER_ALIASES:
            raise ContractValidationError(
                "llm_provider must identify a concrete llm_router provider"
            )
        if (
            self.provider is not None
            and self.llm_provider != DEFAULT_LEANSTRAL_LLM_PROVIDER
            and self.provider != self.llm_provider
        ):
            raise ContractValidationError(
                "provider and llm_provider cannot select different routes"
            )
        object.__setattr__(self, "llm_provider", route)
        object.__setattr__(self, "provider", route)
        object.__setattr__(self, "model", _text(self.model, field_name="model"))
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(float(self.timeout_seconds))
            or float(self.timeout_seconds) <= 0
        ):
            raise ContractValidationError(
                "timeout_seconds must be finite and positive"
            )
        object.__setattr__(self, "timeout_seconds", float(self.timeout_seconds))
        for name in (
            "max_new_tokens",
            "max_context_bytes",
            "max_context_tokens",
            "max_output_bytes",
            "max_records_per_kind",
            "max_concurrent_requests",
        ):
            _positive_int(getattr(self, name), field_name=name)
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(float(self.temperature))
            or float(self.temperature) < 0
        ):
            raise ContractValidationError(
                "temperature must be finite and non-negative"
            )
        object.__setattr__(self, "temperature", float(self.temperature))
        object.__setattr__(
            self, "vibe_agent", _text(self.vibe_agent, field_name="vibe_agent")
        )
        if not isinstance(self.network_access_required, bool):
            raise ContractValidationError(
                "network_access_required must be a boolean"
            )
        if not isinstance(self.resource_isolation, LeanstralResourceIsolation):
            raise ContractValidationError(
                "resource_isolation must be LeanstralResourceIsolation"
            )

    @property
    def model_resource_class(self) -> str:
        return self.resource_isolation.model_resource_class

    @property
    def kernel_resource_class(self) -> str:
        return self.resource_isolation.kernel_resource_class

    def to_dict(self) -> dict[str, Any]:
        return {
            "llm_provider": self.llm_provider,
            "provider": self.provider,
            "model": self.model,
            "timeout_seconds": self.timeout_seconds,
            "max_new_tokens": self.max_new_tokens,
            "max_context_bytes": self.max_context_bytes,
            "max_context_tokens": self.max_context_tokens,
            "max_output_bytes": self.max_output_bytes,
            "max_records_per_kind": self.max_records_per_kind,
            "max_concurrent_requests": self.max_concurrent_requests,
            "temperature": self.temperature,
            "vibe_agent": self.vibe_agent,
            "network_access_required": self.network_access_required,
            "resource_classes": self.resource_isolation.to_dict(),
        }


LeanstralGoalDevelopmentConfig = LeanstralGoalDevelopmentProviderConfig


def _cancelled(cancellation: Any) -> bool:
    if cancellation is None:
        return False
    value = getattr(cancellation, "cancelled", None)
    if callable(value):
        value = value()
    if value is None:
        checker = getattr(cancellation, "is_cancelled", None)
        value = checker() if callable(checker) else False
    return bool(value)


class LeanstralGoalDevelopmentProvider:
    """Generate bounded goal-decomposition drafts with no proof authority."""

    provider_id = LEANSTRAL_GOAL_DEVELOPMENT_PROVIDER_ID
    provider_version = LEANSTRAL_GOAL_DEVELOPMENT_PROVIDER_VERSION
    operation = LEANSTRAL_GOAL_DEVELOPMENT_OPERATION

    def __init__(
        self,
        config: LeanstralGoalDevelopmentProviderConfig | None = None,
        *,
        llm_generate: LLMGenerate | None = None,
    ) -> None:
        self.config = config or LeanstralGoalDevelopmentProviderConfig()
        if llm_generate is not None and not callable(llm_generate):
            raise ContractValidationError("llm_generate must be callable")
        self._llm_generate = llm_generate
        self._capacity = threading.BoundedSemaphore(
            self.config.max_concurrent_requests
        )

    @property
    def model_resource_class(self) -> str:
        return self.config.resource_isolation.model_resource_class

    @property
    def kernel_resource_class(self) -> str:
        return self.config.resource_isolation.kernel_resource_class

    def capabilities(self) -> LeanstralGoalDevelopmentCapability:
        return LeanstralGoalDevelopmentCapability(
            resource_isolation=self.config.resource_isolation
        )

    def capability(self, _request: Any = None) -> LeanstralGoalDevelopmentCapability:
        return self.capabilities()

    build_context = staticmethod(build_leanstral_goal_development_context)

    def _coerce_invocation(
        self,
        invocation: LeanstralGoalDevelopmentInvocation
        | GoalDevelopmentRequest
        | Mapping[str, Any],
        *,
        policy: GoalDevelopmentPolicy | Mapping[str, Any] | None,
        context: GoalDevelopmentContext | Mapping[str, Any] | None,
        resource_budget: ResourceBudget | Mapping[str, Any] | None,
        network_allowed: bool,
        deadline_unix_ms: int | None,
    ) -> LeanstralGoalDevelopmentInvocation:
        if isinstance(invocation, LeanstralGoalDevelopmentInvocation):
            if any(
                item is not None
                for item in (policy, context, resource_budget, deadline_unix_ms)
            ) or network_allowed:
                raise ContractValidationError(
                    "invocation options cannot be supplied twice"
                )
            return invocation
        if isinstance(invocation, Mapping) and not isinstance(
            invocation, GoalDevelopmentRequest
        ):
            if (
                "request" in invocation
                or "goal_request" in invocation
                or "operation" in invocation
            ):
                return LeanstralGoalDevelopmentInvocation.from_dict(invocation)
            request = GoalDevelopmentRequest.from_dict(invocation)
        elif isinstance(invocation, GoalDevelopmentRequest):
            request = invocation
        else:
            raise ContractValidationError(
                "goal development requires a versioned invocation or request"
            )
        if policy is None or context is None:
            raise ContractValidationError(
                "policy and context are required with a bare goal request"
            )
        return LeanstralGoalDevelopmentInvocation(
            request=request,
            policy=(
                policy
                if isinstance(policy, GoalDevelopmentPolicy)
                else GoalDevelopmentPolicy.from_dict(policy)
            ),
            context=(
                context
                if isinstance(context, GoalDevelopmentContext)
                else GoalDevelopmentContext.from_dict(context)
            ),
            resource_budget=_resource_budget(resource_budget),
            network_allowed=network_allowed,
            deadline_unix_ms=deadline_unix_ms,
        )

    def build_prompt(
        self, invocation: LeanstralGoalDevelopmentInvocation
    ) -> str:
        for name in (
            "templates",
            "evidence_gaps",
            "code_references",
            "capabilities",
            "prior_counterexamples",
            "reusable_receipts",
        ):
            if len(getattr(invocation.context, name)) > (
                self.config.max_records_per_kind
            ):
                raise ContractValidationError(
                    f"goal-development {name} exceed the provider record limit"
                )
        context = invocation.context.to_dict()
        envelope = {
            "operation": LEANSTRAL_GOAL_DEVELOPMENT_OPERATION,
            "request_id": invocation.request.request_id,
            "frozen_request": invocation.request.to_dict(),
            "bounded_context": context,
            "output_contract": {
                "schema": LEANSTRAL_GOAL_DEVELOPMENT_OUTPUT_SCHEMA,
                "required_top_level_fields": [
                    "schema",
                    "operation",
                    "request_id",
                    "proposals",
                ],
                "required_proposal_fields": [
                    "proposal_id",
                    "parent_id",
                    "template_id",
                    "title",
                    "evidence_requirement_ids",
                    "assurance_ids",
                    "resource_ids",
                    "scope_ids",
                    "validation_check_ids",
                    "depends_on",
                ],
                "strict_json": True,
                "unknown_fields_forbidden": True,
                "formula_text_forbidden": True,
                "commands_forbidden": True,
                "kernel_check_forbidden": True,
                "canonical_source_forbidden": True,
            },
        }
        prompt = (
            "Propose an unverified goal decomposition. Select only IDs from the "
            "provided allowlists. Do not prove, verify, execute, mutate, or add "
            "formulas. Return only the strict JSON object described below.\n"
            + json.dumps(
                envelope,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        )
        prompt_bytes = prompt.encode("utf-8")
        if len(prompt_bytes) > self.config.max_context_bytes:
            raise ContractValidationError(
                "goal-development context exceeds the configured byte limit"
            )
        if estimate_context_tokens(prompt) > self.config.max_context_tokens:
            raise ContractValidationError(
                "goal-development context exceeds the configured token limit"
            )
        return prompt

    def _effective_timeout(
        self, invocation: LeanstralGoalDevelopmentInvocation
    ) -> float:
        timeout = self.config.timeout_seconds
        budget = invocation.resource_budget
        if budget.wall_time_ms:
            timeout = min(timeout, budget.wall_time_ms / 1000.0)
        if invocation.deadline_unix_ms is not None:
            timeout = min(
                timeout,
                invocation.deadline_unix_ms / 1000.0 - time.time(),
            )
        return max(0.0, timeout)

    def _effective_tokens(
        self, invocation: LeanstralGoalDevelopmentInvocation
    ) -> int:
        tokens = self.config.max_new_tokens
        if invocation.resource_budget.model_token_limit:
            tokens = min(tokens, invocation.resource_budget.model_token_limit)
        return tokens

    def _fallback(
        self,
        invocation: LeanstralGoalDevelopmentInvocation,
        reason: GoalDevelopmentFallbackReason,
    ) -> GoalDevelopmentProviderResult:
        return GoalDevelopmentProviderResult(
            request_id=invocation.request.request_id,
            status=GoalDevelopmentResultStatus.DETERMINISTIC_FALLBACK,
            fallback_reason=reason,
        )

    def _invoke_bounded(
        self,
        prompt: str,
        *,
        timeout: float,
        token_budget: int,
        cancellation: CancellationToken | Any | None,
    ) -> str:
        """Invoke the route on a daemon thread that owns the capacity lease.

        A timed-out in-process call cannot be killed safely.  Keeping the
        semaphore acquired until that worker actually exits prevents abandoned
        route calls from bypassing the configured concurrency bound.
        """

        generate = self._llm_generate or _default_llm_generate
        responses: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

        def run() -> None:
            try:
                value = generate(
                    prompt,
                    provider=self.config.llm_provider,
                    model_name=self.config.model,
                    timeout=timeout,
                    max_new_tokens=token_budget,
                    allow_local_fallback=False,
                    disable_model_retry=True,
                    temperature=self.config.temperature,
                    mistral_vibe_agent=self.config.vibe_agent,
                )
                responses.put_nowait((True, value))
            except BaseException as exc:  # transport exceptions are classified below
                try:
                    responses.put_nowait((False, exc))
                except queue.Full:
                    pass
            finally:
                self._capacity.release()

        worker = threading.Thread(
            target=run,
            name="leanstral-goal-development",
            daemon=True,
        )
        try:
            worker.start()
        except BaseException:
            self._capacity.release()
            raise
        deadline = time.monotonic() + timeout
        while True:
            if _cancelled(cancellation):
                raise ProofProviderError(
                    ProviderFailureCode.CANCELLED,
                    "goal-development request was cancelled",
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ProofProviderError(
                    ProviderFailureCode.TIMED_OUT,
                    "goal-development model route timed out",
                    retryable=True,
                )
            try:
                ok, value = responses.get(timeout=min(remaining, 0.05))
            except queue.Empty:
                continue
            if ok:
                return value
            raise value

    def _parse_response(
        self,
        output: str,
        invocation: LeanstralGoalDevelopmentInvocation,
        *,
        token_budget: int,
    ) -> GoalDecompositionDraft:
        if not isinstance(output, str) or not output.strip():
            raise ContractValidationError(
                "llm_router returned an empty or non-text response"
            )
        output = output.strip()
        output_limit = self.config.max_output_bytes
        if invocation.resource_budget.max_output_bytes:
            output_limit = min(
                output_limit, invocation.resource_budget.max_output_bytes
            )
        if len(output.encode("utf-8")) > output_limit:
            raise OverflowError("goal-development output exceeds byte budget")
        if estimate_context_tokens(output) > token_budget:
            raise OverflowError("goal-development output exceeds token budget")
        value = _strict_object(output)
        if _FORBIDDEN_RESPONSE_KEYS.intersection(value):
            raise ContractValidationError(
                "goal-development output attempted a forbidden operation"
            )
        _reject_unknown(
            value,
            {"schema", "operation", "request_id", "proposals"},
            field_name="goal-development output",
        )
        if value.get("schema") != LEANSTRAL_GOAL_DEVELOPMENT_OUTPUT_SCHEMA:
            raise ContractValidationError("goal-development output used the wrong schema")
        if value.get("operation") != LEANSTRAL_GOAL_DEVELOPMENT_OPERATION:
            raise ContractValidationError(
                "goal-development output used the wrong operation"
            )
        request = invocation.request
        if value.get("request_id") != request.request_id:
            raise ContractValidationError(
                "goal-development output changed the frozen request identity"
            )
        raw_proposals = value.get("proposals")
        if not isinstance(raw_proposals, list):
            raise ContractValidationError("proposals must be an array")
        if len(raw_proposals) > invocation.policy.max_proposals:
            raise OverflowError("goal-development output exceeds proposal count")
        templates = {
            template.template_id: template
            for template in invocation.context.templates
        }
        proposals: list[GoalDecompositionProposal] = []
        for index, raw in enumerate(raw_proposals):
            if not isinstance(raw, Mapping):
                raise ContractValidationError(
                    f"proposals[{index}] must be an object"
                )
            if _FORBIDDEN_RESPONSE_KEYS.intersection(raw):
                raise ContractValidationError(
                    "goal-development proposal attempted a forbidden operation"
                )
            required = {
                "proposal_id",
                "parent_id",
                "template_id",
                "title",
                "evidence_requirement_ids",
                "assurance_ids",
                "resource_ids",
                "scope_ids",
                "validation_check_ids",
                "depends_on",
            }
            _reject_unknown(raw, required, field_name=f"proposals[{index}]")
            if set(raw) != required:
                raise ContractValidationError(
                    f"proposals[{index}] is missing required fields"
                )
            template_id = _text(
                raw.get("template_id"),
                field_name=f"proposals[{index}].template_id",
            )
            template = templates.get(template_id)
            if template is None:
                raise ContractValidationError("proposal selected an unknown template ID")
            evidence = _ids(
                raw.get("evidence_requirement_ids"),
                field_name=f"proposals[{index}].evidence_requirement_ids",
                required=True,
            )
            assurance = _ids(
                raw.get("assurance_ids"),
                field_name=f"proposals[{index}].assurance_ids",
                required=True,
            )
            resources = _ids(
                raw.get("resource_ids"),
                field_name=f"proposals[{index}].resource_ids",
                required=True,
            )
            scopes = _ids(
                raw.get("scope_ids"),
                field_name=f"proposals[{index}].scope_ids",
                required=True,
            )
            checks = _ids(
                raw.get("validation_check_ids"),
                field_name=f"proposals[{index}].validation_check_ids",
                required=True,
            )
            selections = (
                (evidence, template.evidence_requirement_ids, "evidence"),
                (assurance, template.assurance_ids, "assurance"),
                (resources, template.resource_ids, "resource"),
                (scopes, template.scope_ids, "scope"),
                (checks, template.validation_check_ids, "validation-check"),
            )
            for selected, allowed, kind in selections:
                if not set(selected).issubset(allowed):
                    raise ContractValidationError(
                        f"proposal selected an unknown {kind} ID"
                    )
            proposals.append(
                GoalDecompositionProposal(
                    proposal_id=_text(
                        raw.get("proposal_id"),
                        field_name=f"proposals[{index}].proposal_id",
                    ),
                    parent_id=_text(
                        raw.get("parent_id"),
                        field_name=f"proposals[{index}].parent_id",
                    ),
                    satisfaction_formula_id=template.satisfaction_formula_id,
                    assumption_ids=request.assumption_ids,
                    evidence_requirement_ids=evidence,
                    scope_ids=scopes,
                    depends_on=_ids(
                        raw.get("depends_on"),
                        field_name=f"proposals[{index}].depends_on",
                    ),
                    title=_text(
                        raw.get("title"),
                        field_name=f"proposals[{index}].title",
                        required=False,
                    ),
                )
            )
        return GoalDecompositionDraft(
            request=request,
            policy=invocation.policy,
            proposals=tuple(proposals),
            producer_id=f"provider:{self.provider_id}@{self.provider_version}",
            token_count=estimate_context_tokens(output),
        )

    def develop(
        self,
        invocation: LeanstralGoalDevelopmentInvocation
        | GoalDevelopmentRequest
        | Mapping[str, Any],
        *,
        policy: GoalDevelopmentPolicy | Mapping[str, Any] | None = None,
        context: GoalDevelopmentContext | Mapping[str, Any] | None = None,
        resource_budget: ResourceBudget | Mapping[str, Any] | None = None,
        network_allowed: bool = False,
        deadline_unix_ms: int | None = None,
        cancellation: CancellationToken | Any | None = None,
    ) -> GoalDevelopmentProviderResult:
        """Run one bounded model proposal or return a deterministic fallback."""

        call = self._coerce_invocation(
            invocation,
            policy=policy,
            context=context,
            resource_budget=resource_budget,
            network_allowed=network_allowed,
            deadline_unix_ms=deadline_unix_ms,
        )
        if _cancelled(cancellation):
            return self._fallback(call, GoalDevelopmentFallbackReason.CANCELLED)
        if self.config.network_access_required and not call.network_allowed:
            return self._fallback(call, GoalDevelopmentFallbackReason.UNAVAILABLE)
        timeout = self._effective_timeout(call)
        if timeout <= 0:
            return self._fallback(call, GoalDevelopmentFallbackReason.TIMEOUT)
        token_budget = self._effective_tokens(call)
        if token_budget <= 0:
            return self._fallback(call, GoalDevelopmentFallbackReason.OVERLOADED)
        try:
            prompt = self.build_prompt(call)
        except ContractValidationError:
            return self._fallback(call, GoalDevelopmentFallbackReason.OVERLOADED)
        if not self._capacity.acquire(blocking=False):
            return self._fallback(call, GoalDevelopmentFallbackReason.OVERLOADED)
        worker_started = False
        try:
            try:
                worker_started = True
                output = self._invoke_bounded(
                    prompt,
                    timeout=timeout,
                    token_budget=token_budget,
                    cancellation=cancellation,
                )
                draft = self._parse_response(
                    output, call, token_budget=token_budget
                )
            except OverflowError:
                return self._fallback(
                    call, GoalDevelopmentFallbackReason.OVERLOADED
                )
            except ProofProviderError as exc:
                reason = {
                    ProviderFailureCode.CANCELLED: GoalDevelopmentFallbackReason.CANCELLED,
                    ProviderFailureCode.TIMED_OUT: GoalDevelopmentFallbackReason.TIMEOUT,
                    ProviderFailureCode.RESOURCE_EXHAUSTED: GoalDevelopmentFallbackReason.OVERLOADED,
                    ProviderFailureCode.MALFORMED_RESPONSE: GoalDevelopmentFallbackReason.MALFORMED_OUTPUT,
                }.get(exc.code, GoalDevelopmentFallbackReason.UNAVAILABLE)
                return self._fallback(call, reason)
            except (TimeoutError, asyncio.TimeoutError):
                return self._fallback(call, GoalDevelopmentFallbackReason.TIMEOUT)
            except (asyncio.CancelledError,):
                return self._fallback(call, GoalDevelopmentFallbackReason.CANCELLED)
            except (ContractValidationError, ValueError, TypeError, json.JSONDecodeError):
                return self._fallback(
                    call, GoalDevelopmentFallbackReason.MALFORMED_OUTPUT
                )
            except (ImportError, ModuleNotFoundError):
                return self._fallback(call, GoalDevelopmentFallbackReason.UNAVAILABLE)
            except BaseException as exc:
                message = str(exc).casefold()
                reason = (
                    GoalDevelopmentFallbackReason.OVERLOADED
                    if any(
                        marker in message
                        for marker in ("overload", "capacity", "too many", "429")
                    )
                    else GoalDevelopmentFallbackReason.UNAVAILABLE
                )
                return self._fallback(call, reason)
            return GoalDevelopmentProviderResult(
                request_id=call.request.request_id,
                status=GoalDevelopmentResultStatus.DRAFT,
                draft=draft,
            )
        finally:
            # Once started, the daemon worker owns the capacity lease and
            # releases it only when the underlying route call truly finishes.
            if not worker_started:
                self._capacity.release()

    # Explicit compatibility spellings still preserve the versioned operation
    # envelope; none route through the proof provider's ``prove`` method.
    goal_development = develop
    develop_goals = develop


def create_leanstral_goal_development_provider(
    config: LeanstralGoalDevelopmentProviderConfig | None = None,
    *,
    llm_generate: LLMGenerate | None = None,
) -> LeanstralGoalDevelopmentProvider:
    return LeanstralGoalDevelopmentProvider(config, llm_generate=llm_generate)


__all__ = [
    "ASTGraphRAGReferenceRecord",
    "CapabilityRecord",
    "CodeReferenceKind",
    "DEFAULT_GOAL_DEVELOPMENT_MAX_CONCURRENT_REQUESTS",
    "DEFAULT_GOAL_DEVELOPMENT_MAX_CONTEXT_BYTES",
    "DEFAULT_GOAL_DEVELOPMENT_MAX_CONTEXT_TOKENS",
    "DEFAULT_GOAL_DEVELOPMENT_MAX_NEW_TOKENS",
    "DEFAULT_GOAL_DEVELOPMENT_MAX_OUTPUT_BYTES",
    "DEFAULT_GOAL_DEVELOPMENT_MAX_RECORDS_PER_KIND",
    "DEFAULT_GOAL_DEVELOPMENT_TIMEOUT_SECONDS",
    "EvidenceGapRecord",
    "GoalDevelopmentContext",
    "GoalDevelopmentFallbackReason",
    "GoalDevelopmentProviderResult",
    "GoalDevelopmentResultStatus",
    "GoalDevelopmentTemplate",
    "ImmutableGoalRecord",
    "LEANSTRAL_GOAL_DEVELOPMENT_CONTEXT_SCHEMA",
    "LEANSTRAL_GOAL_DEVELOPMENT_OPERATION",
    "LEANSTRAL_GOAL_DEVELOPMENT_OPERATION_VERSION",
    "LEANSTRAL_GOAL_DEVELOPMENT_OUTPUT_SCHEMA",
    "LEANSTRAL_GOAL_DEVELOPMENT_PROVIDER_ID",
    "LEANSTRAL_GOAL_DEVELOPMENT_PROVIDER_VERSION",
    "LEANSTRAL_GOAL_DEVELOPMENT_REQUEST_SCHEMA",
    "LEANSTRAL_GOAL_DEVELOPMENT_RESULT_SCHEMA",
    "LeanstralGoalDevelopmentCapability",
    "LeanstralGoalDevelopmentConfig",
    "LeanstralGoalDevelopmentInvocation",
    "LeanstralGoalDevelopmentProvider",
    "LeanstralGoalDevelopmentProviderConfig",
    "PriorCounterexampleRecord",
    "ReusableReceiptRecord",
    "build_leanstral_goal_development_context",
    "create_leanstral_goal_development_provider",
]
