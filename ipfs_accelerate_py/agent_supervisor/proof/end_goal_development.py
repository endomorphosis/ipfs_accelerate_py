"""Post-formalization supervisor adapter for Leanstral goal development.

``FormalizedGoalDevelopmentRoute@1`` is the fail-closed gate that sits between
prompt / prose workflows and the capability-isolated Leanstral goal-development
provider.  A call reaches Leanstral only after a caller-confirmed
:class:`~ipfs_datasets_py.logic.software_verification.tactician.contracts.FormalGoal`
is present.  The untrusted provider then receives only immutable selected goal,
formula, assumption, vocabulary, and template identifiers — never formula text,
source, proof claims, commands, admission, or completion authority.

Acceptance invariants (FVT-G025 / FVT-024):

* prose cannot bypass formalization;
* Leanstral cannot create or mutate formulas, source, assumptions, proof,
  commands, admission, or completion through this route; and
* timeout / unavailable / malformed provider responses fall back
  deterministically without stalling the supervisor.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final

from ..objectives.goal_development_contracts import (
    GoalDevelopmentMode,
    GoalDevelopmentPolicy,
    GoalDevelopmentRequest,
)
from .formal_logic_vocabulary import LOGIC_VOCABULARY_VERSION
from .formal_verification_contracts import (
    AssuranceLevel,
    ContractValidationError,
    ResourceBudget,
    canonical_json_bytes,
)
from .formal_verification_provider import CancellationToken
from .leanstral_goal_development import (
    ASTGraphRAGReferenceRecord,
    CapabilityRecord,
    EvidenceGapRecord,
    GoalDevelopmentContext,
    GoalDevelopmentFallbackReason,
    GoalDevelopmentProviderResult,
    GoalDevelopmentResultStatus,
    GoalDevelopmentTemplate,
    ImmutableGoalRecord,
    LEANSTRAL_GOAL_DEVELOPMENT_OPERATION,
    LEANSTRAL_GOAL_DEVELOPMENT_PROVIDER_ID,
    LeanstralGoalDevelopmentInvocation,
    LeanstralGoalDevelopmentProvider,
    LeanstralGoalDevelopmentProviderConfig,
    PriorCounterexampleRecord,
    ReusableReceiptRecord,
    build_leanstral_goal_development_context,
)

try:  # Prefer the datasets-side confirmed FormalGoal when available.
    from ipfs_datasets_py.logic.software_verification.tactician.contracts import (
        AmbiguityStatus,
        FormalGoal,
        TacticianContractError,
    )
except Exception:  # pragma: no cover - optional import surface
    AmbiguityStatus = None  # type: ignore[assignment,misc]
    FormalGoal = None  # type: ignore[assignment,misc]
    TacticianContractError = ValueError  # type: ignore[misc,assignment]


# ---------------------------------------------------------------------------
# Interface / schema constants
# ---------------------------------------------------------------------------

FORMALIZED_GOAL_DEVELOPMENT_ROUTE_INTERFACE: Final = (
    "FormalizedGoalDevelopmentRoute@1"
)
FORMALIZED_GOAL_DEVELOPMENT_ROUTE_VERSION: Final = "1.0.0"
FORMALIZED_GOAL_DEVELOPMENT_ROUTE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/formalized-goal-development-route@1"
)
FORMALIZED_GOAL_DEVELOPMENT_REQUEST_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/formalized-goal-development-request@1"
)
FORMALIZED_GOAL_DEVELOPMENT_RESULT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/formalized-goal-development-result@1"
)
FORMALIZED_GOAL_IDENTIFIERS_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/formalized-goal-identifiers@1"
)

DEFAULT_VOCABULARY_PROFILE_ID: Final = "supervisor-reviewed"
DEFAULT_FORMULA_PREFIX: Final = "formula"

# Keys that mark a payload as raw prose / pre-formalization input rather than
# a confirmed FormalGoal.  Presence of these without formal_goal_id fails closed.
_PROSE_PRIMARY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "prose",
        "prompt",
        "caller_text",
        "natural_language",
        "nl_goal",
        "informal_goal",
        "raw_text",
        "user_text",
    }
)

# Fields the route must never forward into the untrusted provider envelope.
_FORBIDDEN_PROVIDER_PAYLOAD_KEYS: Final[frozenset[str]] = frozenset(
    {
        "prose",
        "prompt",
        "caller_text",
        "natural_language",
        "source_code",
        "canonical_source",
        "source_text",
        "formula",
        "formula_text",
        "satisfaction_formula",
        "proof",
        "proof_text",
        "commands",
        "command",
        "shell",
        "shell_command",
        "validation_command",
        "validation_commands",
        "kernel_check",
        "kernel_checked",
        "admitted",
        "admission_claimed",
        "complete",
        "completion_claimed",
        "verified",
        "authoritative",
        "implementation_conformance_claimed",
    }
)


class EndGoalDevelopmentError(ContractValidationError):
    """Raised when a formalized goal-development route request is invalid."""


class FormalizationGateReason(str, Enum):
    """Why the formalization gate refused to invoke Leanstral."""

    PROSE_BYPASS = "prose_bypass"
    MISSING_FORMAL_GOAL = "missing_formal_goal"
    UNCONFIRMED = "unconfirmed"
    AMBIGUITY_UNRESOLVED = "ambiguity_unresolved"
    INVALID_FORMAL_GOAL = "invalid_formal_goal"
    MISSING_TEMPLATES = "missing_templates"
    AUTHORITY_CLAIM = "authority_claim"
    MISSING_EVIDENCE = "missing_evidence"
    MISSING_SCOPE = "missing_scope"
    MISSING_REPOSITORY_TREE = "missing_repository_tree"


class FormalizedRouteStatus(str, Enum):
    """Route-level outcome independent of the provider draft status."""

    DRAFT = "draft"
    DETERMINISTIC_FALLBACK = "deterministic_fallback"
    REJECTED = "rejected"


def _text(value: Any, *, field_name: str, required: bool = True) -> str:
    if value is None:
        result = ""
    elif not isinstance(value, str):
        raise EndGoalDevelopmentError(f"{field_name} must be a string")
    else:
        result = value.strip()
    if required and not result:
        raise EndGoalDevelopmentError(f"{field_name} is required")
    if "\x00" in result:
        raise EndGoalDevelopmentError(f"{field_name} must not contain NUL bytes")
    return result


def _string_tuple(
    values: Any,
    *,
    field_name: str,
    required: bool = False,
) -> tuple[str, ...]:
    if values is None:
        source: Sequence[Any] = ()
    elif isinstance(values, str):
        source = (values,)
    elif isinstance(values, Sequence) and not isinstance(
        values, (bytes, bytearray, memoryview)
    ):
        source = values
    else:
        raise EndGoalDevelopmentError(
            f"{field_name} must be a sequence of strings"
        )
    result: list[str] = []
    for index, item in enumerate(source):
        text = _text(item, field_name=f"{field_name}[{index}]", required=True)
        if text not in result:
            result.append(text)
    if required and not result:
        raise EndGoalDevelopmentError(f"{field_name} must not be empty")
    return tuple(result)


def _mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EndGoalDevelopmentError(f"{field_name} must be an object")
    return value


def _resource_budget(value: ResourceBudget | Mapping[str, Any] | None) -> ResourceBudget:
    if value is None:
        return ResourceBudget()
    if isinstance(value, ResourceBudget):
        return value
    if isinstance(value, Mapping):
        return ResourceBudget.from_dict(value)
    raise EndGoalDevelopmentError("resource_budget must be a ResourceBudget")


def _reject_authority_claims(payload: Mapping[str, Any], *, artifact: str) -> None:
    for name in (
        "proof_claimed",
        "completion_claimed",
        "admission_claimed",
        "admitted",
        "complete",
        "verified",
        "authoritative",
        "kernel_checked",
        "implementation_conformance_claimed",
        "implementation_conformant",
    ):
        if payload.get(name) not in (None, False):
            raise EndGoalDevelopmentError(
                f"{artifact} cannot claim {name.replace('_', ' ')}"
            )


def _is_formal_goal_instance(value: Any) -> bool:
    if FormalGoal is None:
        return False
    return isinstance(value, FormalGoal)


def _looks_like_formal_goal_payload(value: Mapping[str, Any]) -> bool:
    return bool(
        value.get("formal_goal_id")
        and (value.get("end_goal") is not None or value.get("end_goal_spec") is not None)
    )


def _looks_like_prose_primary(value: Any) -> bool:
    if isinstance(value, str):
        return True
    if not isinstance(value, Mapping):
        return False
    if _looks_like_formal_goal_payload(value):
        return False
    keys = {str(key).casefold() for key in value}
    if keys & {item.casefold() for item in _PROSE_PRIMARY_KEYS}:
        return True
    # Bare EndGoalSpec / draft without confirmation envelope.
    if "caller_text" in value and "formal_goal_id" not in value:
        return True
    if "goal_id" in value and "status" in value and "formal_goal_id" not in value:
        status = str(value.get("status") or "").casefold()
        if status in {"", "draft", "candidate", "prose", "informal"}:
            return True
    return False


def _coerce_formal_goal(value: Any) -> Any:
    """Return a FormalGoal instance or raise EndGoalDevelopmentError."""

    if value is None:
        raise EndGoalDevelopmentError(
            "formal_goal is required; prose cannot bypass formalization"
        )
    if isinstance(value, str):
        raise EndGoalDevelopmentError(
            "prose cannot bypass formalization; supply a confirmed FormalGoal"
        )
    if _is_formal_goal_instance(value):
        return value
    if not isinstance(value, Mapping):
        raise EndGoalDevelopmentError(
            "formal_goal must be a FormalGoal or mapping payload"
        )
    if _looks_like_prose_primary(value):
        raise EndGoalDevelopmentError(
            "prose cannot bypass formalization; supply a confirmed FormalGoal"
        )
    if not _looks_like_formal_goal_payload(value):
        raise EndGoalDevelopmentError(
            "formal_goal must include formal_goal_id and end_goal"
        )
    _reject_authority_claims(value, artifact="formal_goal")
    if FormalGoal is None:
        # Minimal structural validation when the datasets package is unavailable.
        return dict(value)
    try:
        return FormalGoal.from_dict(value)
    except (TacticianContractError, ContractValidationError, TypeError, ValueError) as exc:
        raise EndGoalDevelopmentError(
            f"invalid formal_goal: {exc}"
        ) from exc


def _end_goal_of(formal_goal: Any) -> Any:
    if _is_formal_goal_instance(formal_goal):
        return formal_goal.end_goal
    end_goal = formal_goal.get("end_goal") or formal_goal.get("end_goal_spec")
    if end_goal is None:
        raise EndGoalDevelopmentError("formal_goal.end_goal is required")
    return end_goal


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _require_confirmed_formal_goal(formal_goal: Any) -> FormalizationGateReason | None:
    """Return a gate rejection reason, or None when formalization is confirmed."""

    status = str(_attr(formal_goal, "status", "") or "").casefold()
    if status not in {"confirmed", "selected", "resolved"}:
        return FormalizationGateReason.UNCONFIRMED

    if bool(_attr(formal_goal, "proof_claimed", False)) or bool(
        _attr(formal_goal, "completion_claimed", False)
    ):
        return FormalizationGateReason.AUTHORITY_CLAIM

    end_goal = _end_goal_of(formal_goal)
    ambiguity = _attr(end_goal, "ambiguity_status", None)
    if AmbiguityStatus is not None and ambiguity is AmbiguityStatus.REQUIRES_SELECTION:
        return FormalizationGateReason.AMBIGUITY_UNRESOLVED
    if isinstance(ambiguity, str) and ambiguity.casefold() in {
        "requires_selection",
        "unresolved",
        "ambiguous",
    }:
        return FormalizationGateReason.AMBIGUITY_UNRESOLVED
    if hasattr(ambiguity, "value"):
        raw = str(ambiguity.value).casefold()
        if raw in {"requires_selection", "unresolved", "ambiguous"}:
            return FormalizationGateReason.AMBIGUITY_UNRESOLVED

    end_status = str(_attr(end_goal, "status", "") or "").casefold()
    if end_status in {"draft", "candidate", "prose", "informal"}:
        return FormalizationGateReason.UNCONFIRMED

    if bool(_attr(end_goal, "proof_claimed", False)) or bool(
        _attr(end_goal, "completion_claimed", False)
    ):
        return FormalizationGateReason.AUTHORITY_CLAIM

    return None


def _formula_id_from_compilation(compilation: Any | None, formal_goal: Any) -> str:
    formal_goal_id = _text(
        _attr(formal_goal, "formal_goal_id", ""),
        field_name="formal_goal_id",
    )
    selected = _text(
        _attr(formal_goal, "selected_interpretation_id", ""),
        field_name="selected_interpretation_id",
        required=False,
    )
    if compilation is None:
        suffix = selected or formal_goal_id
        return f"{DEFAULT_FORMULA_PREFIX}:{suffix}"

    if isinstance(compilation, Mapping):
        obligations = compilation.get("root_obligations") or ()
        if obligations:
            first = obligations[0]
            property_id = _attr(first, "property_id", "")
            if property_id:
                return _text(property_id, field_name="property_id")
        ir = compilation.get("ir")
        if isinstance(ir, Mapping):
            properties = ir.get("properties") or ()
            if properties:
                property_id = _attr(properties[0], "property_id", "")
                if property_id:
                    return _text(property_id, field_name="property_id")
        # Fall through to deterministic local id.
        return f"{DEFAULT_FORMULA_PREFIX}:{selected or formal_goal_id}"

    obligations = getattr(compilation, "root_obligations", ()) or ()
    if obligations:
        property_id = getattr(obligations[0], "property_id", "")
        if property_id:
            return _text(property_id, field_name="property_id")
    return f"{DEFAULT_FORMULA_PREFIX}:{selected or formal_goal_id}"


def _assumption_ids(end_goal: Any) -> tuple[str, ...]:
    assumptions = _attr(end_goal, "assumptions", ()) or ()
    result: list[str] = []
    for item in assumptions:
        assumption_id = _attr(item, "assumption_id", "")
        if not assumption_id and isinstance(item, str):
            assumption_id = item
        if assumption_id:
            text = _text(assumption_id, field_name="assumption_id")
            if text not in result:
                result.append(text)
    return tuple(result)


def _evidence_ids(end_goal: Any, formal_goal: Any) -> tuple[str, ...]:
    evidence = list(_string_tuple(_attr(end_goal, "acceptance_evidence", ()), field_name="acceptance_evidence"))
    if evidence:
        return tuple(evidence)
    receipt = _attr(formal_goal, "confirmation_receipt_id", "")
    if receipt:
        return (f"evidence:confirmation:{receipt}",)
    return ("evidence:formal-goal-confirmed",)


def _scope_ids(end_goal: Any) -> tuple[str, ...]:
    source = _attr(end_goal, "source", None)
    if source is None:
        return ("scope:repository",)
    for name in ("ast_scope_ids", "source_ref_ids", "span_ids"):
        values = _string_tuple(_attr(source, name, ()), field_name=name)
        if values:
            return values
    return ("scope:repository",)


def _repository_tree_id(end_goal: Any) -> str:
    source = _attr(end_goal, "source", None)
    tree_id = _attr(source, "tree_id", "") if source is not None else ""
    if not tree_id:
        raise EndGoalDevelopmentError(
            "formal_goal.end_goal.source.tree_id is required"
        )
    return _text(tree_id, field_name="repository_tree_id")


def _content_id_of(formal_goal: Any) -> str:
    content_id = _attr(formal_goal, "content_id", None)
    if callable(content_id):
        content_id = content_id()
    if content_id:
        return _text(content_id, field_name="content_id")
    # Mapping without computed content_id: hash the formal_goal_id + end_goal id.
    formal_goal_id = _text(
        _attr(formal_goal, "formal_goal_id", ""),
        field_name="formal_goal_id",
    )
    end_goal = _end_goal_of(formal_goal)
    end_goal_id = _attr(end_goal, "content_id", None)
    if callable(end_goal_id):
        end_goal_id = end_goal_id()
    if not end_goal_id:
        end_goal_id = _attr(end_goal, "goal_id", formal_goal_id)
    digest = hashlib.sha256(
        f"{formal_goal_id}:{end_goal_id}".encode("utf-8")
    ).hexdigest()
    return f"cid:formal-goal:{digest[:32]}"


@dataclass(frozen=True)
class FormalizedGoalIdentifiers:
    """Immutable identifier set exposed to the untrusted Leanstral provider.

    Intentionally excludes prose, formula text, source code, proof, commands,
    admission, and completion fields.
    """

    schema: str = FORMALIZED_GOAL_IDENTIFIERS_SCHEMA
    formal_goal_id: str = ""
    root_goal_id: str = ""
    root_goal_content_id: str = ""
    satisfaction_formula_id: str = ""
    assumption_ids: tuple[str, ...] = ()
    evidence_requirement_ids: tuple[str, ...] = ()
    vocabulary_profile_id: str = DEFAULT_VOCABULARY_PROFILE_ID
    vocabulary_version: int = LOGIC_VOCABULARY_VERSION
    repository_tree_id: str = ""
    scope_ids: tuple[str, ...] = ()
    template_ids: tuple[str, ...] = ()
    selected_interpretation_id: str = ""
    confirmation_receipt_id: str = ""

    def __post_init__(self) -> None:
        for name in (
            "formal_goal_id",
            "root_goal_id",
            "root_goal_content_id",
            "satisfaction_formula_id",
            "vocabulary_profile_id",
            "repository_tree_id",
        ):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), field_name=name),
            )
        object.__setattr__(
            self,
            "assumption_ids",
            _string_tuple(self.assumption_ids, field_name="assumption_ids"),
        )
        object.__setattr__(
            self,
            "evidence_requirement_ids",
            _string_tuple(
                self.evidence_requirement_ids,
                field_name="evidence_requirement_ids",
                required=True,
            ),
        )
        object.__setattr__(
            self,
            "scope_ids",
            _string_tuple(self.scope_ids, field_name="scope_ids", required=True),
        )
        object.__setattr__(
            self,
            "template_ids",
            _string_tuple(self.template_ids, field_name="template_ids", required=True),
        )
        object.__setattr__(
            self,
            "selected_interpretation_id",
            _text(
                self.selected_interpretation_id,
                field_name="selected_interpretation_id",
                required=False,
            ),
        )
        object.__setattr__(
            self,
            "confirmation_receipt_id",
            _text(
                self.confirmation_receipt_id,
                field_name="confirmation_receipt_id",
                required=False,
            ),
        )
        if (
            not isinstance(self.vocabulary_version, int)
            or isinstance(self.vocabulary_version, bool)
            or self.vocabulary_version <= 0
        ):
            raise EndGoalDevelopmentError(
                "vocabulary_version must be a positive integer"
            )
        if self.schema != FORMALIZED_GOAL_IDENTIFIERS_SCHEMA:
            raise EndGoalDevelopmentError("unsupported formalized identifiers schema")

    @property
    def content_id(self) -> str:
        return "formalized-ids-" + hashlib.sha256(
            canonical_json_bytes(self.to_dict(include_id=False))
        ).hexdigest()

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "formal_goal_id": self.formal_goal_id,
            "root_goal_id": self.root_goal_id,
            "root_goal_content_id": self.root_goal_content_id,
            "satisfaction_formula_id": self.satisfaction_formula_id,
            "assumption_ids": list(self.assumption_ids),
            "evidence_requirement_ids": list(self.evidence_requirement_ids),
            "vocabulary_profile_id": self.vocabulary_profile_id,
            "vocabulary_version": self.vocabulary_version,
            "repository_tree_id": self.repository_tree_id,
            "scope_ids": list(self.scope_ids),
            "template_ids": list(self.template_ids),
            "selected_interpretation_id": self.selected_interpretation_id,
            "confirmation_receipt_id": self.confirmation_receipt_id,
        }
        if include_id:
            payload["content_id"] = self.content_id
        return payload

    def provider_view(self) -> dict[str, Any]:
        """Identifier-only projection safe for the untrusted provider."""

        view = self.to_dict(include_id=False)
        # Explicitly drop confirmation receipt from the model view; it is a
        # supervisor-side binding, not a selectable template field.
        view.pop("confirmation_receipt_id", None)
        for forbidden in _FORBIDDEN_PROVIDER_PAYLOAD_KEYS:
            view.pop(forbidden, None)
        return view

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FormalizedGoalIdentifiers":
        payload = _mapping(value, field_name="formalized goal identifiers")
        result = cls(
            schema=payload.get("schema", FORMALIZED_GOAL_IDENTIFIERS_SCHEMA),
            formal_goal_id=payload.get("formal_goal_id", ""),
            root_goal_id=payload.get("root_goal_id", ""),
            root_goal_content_id=payload.get("root_goal_content_id", ""),
            satisfaction_formula_id=payload.get("satisfaction_formula_id", ""),
            assumption_ids=tuple(payload.get("assumption_ids") or ()),
            evidence_requirement_ids=tuple(
                payload.get("evidence_requirement_ids") or ()
            ),
            vocabulary_profile_id=payload.get(
                "vocabulary_profile_id", DEFAULT_VOCABULARY_PROFILE_ID
            ),
            vocabulary_version=payload.get(
                "vocabulary_version", LOGIC_VOCABULARY_VERSION
            ),
            repository_tree_id=payload.get("repository_tree_id", ""),
            scope_ids=tuple(payload.get("scope_ids") or ()),
            template_ids=tuple(payload.get("template_ids") or ()),
            selected_interpretation_id=payload.get(
                "selected_interpretation_id", ""
            ),
            confirmation_receipt_id=payload.get("confirmation_receipt_id", ""),
        )
        claimed = payload.get("content_id")
        if claimed not in (None, "", result.content_id):
            raise EndGoalDevelopmentError(
                "formalized identifiers content identity does not match payload"
            )
        return result


@dataclass(frozen=True)
class FormalizedGoalDevelopmentRequest:
    """Supervisor-owned request that may invoke Leanstral only after formalization."""

    schema: str = FORMALIZED_GOAL_DEVELOPMENT_REQUEST_SCHEMA
    formal_goal: Any = None
    policy: GoalDevelopmentPolicy | Mapping[str, Any] | None = None
    templates: tuple[GoalDevelopmentTemplate | Mapping[str, Any], ...] = ()
    compilation_result: Any | None = None
    vocabulary_profile_id: str = DEFAULT_VOCABULARY_PROFILE_ID
    vocabulary_version: int = LOGIC_VOCABULARY_VERSION
    evidence_gaps: tuple[EvidenceGapRecord | Mapping[str, Any], ...] = ()
    code_references: tuple[ASTGraphRAGReferenceRecord | Mapping[str, Any], ...] = ()
    capabilities: tuple[CapabilityRecord | Mapping[str, Any], ...] = ()
    prior_counterexamples: tuple[
        PriorCounterexampleRecord | Mapping[str, Any], ...
    ] = ()
    reusable_receipts: tuple[ReusableReceiptRecord | Mapping[str, Any], ...] = ()
    resource_budget: ResourceBudget | Mapping[str, Any] | None = None
    network_allowed: bool = False
    deadline_unix_ms: int | None = None
    # Explicit prose / pre-formalization fields are accepted only so the gate
    # can reject them with a stable reason rather than a TypeError.
    prose: str = ""
    caller_text: str = ""

    def __post_init__(self) -> None:
        if self.schema != FORMALIZED_GOAL_DEVELOPMENT_REQUEST_SCHEMA:
            raise EndGoalDevelopmentError(
                "unsupported formalized goal-development request schema"
            )
        if not isinstance(self.network_allowed, bool):
            raise EndGoalDevelopmentError("network_allowed must be a boolean")
        object.__setattr__(
            self, "prose", _text(self.prose, field_name="prose", required=False)
        )
        object.__setattr__(
            self,
            "caller_text",
            _text(self.caller_text, field_name="caller_text", required=False),
        )
        object.__setattr__(
            self,
            "vocabulary_profile_id",
            _text(
                self.vocabulary_profile_id,
                field_name="vocabulary_profile_id",
            ),
        )
        if (
            not isinstance(self.vocabulary_version, int)
            or isinstance(self.vocabulary_version, bool)
            or self.vocabulary_version <= 0
        ):
            raise EndGoalDevelopmentError(
                "vocabulary_version must be a positive integer"
            )
        templates = tuple(self.templates or ())
        object.__setattr__(self, "templates", templates)
        object.__setattr__(
            self, "resource_budget", _resource_budget(self.resource_budget)
        )
        policy = self.policy
        if policy is None:
            policy = GoalDevelopmentPolicy(mode=GoalDevelopmentMode.SHADOW)
        elif isinstance(policy, Mapping):
            policy = GoalDevelopmentPolicy.from_dict(policy)
        elif not isinstance(policy, GoalDevelopmentPolicy):
            raise EndGoalDevelopmentError("policy must be a GoalDevelopmentPolicy")
        object.__setattr__(self, "policy", policy)

    def to_dict(self) -> dict[str, Any]:
        formal_goal = self.formal_goal
        if formal_goal is not None and hasattr(formal_goal, "to_dict"):
            formal_payload = formal_goal.to_dict()
        elif isinstance(formal_goal, Mapping):
            formal_payload = dict(formal_goal)
        else:
            formal_payload = formal_goal
        return {
            "schema": self.schema,
            "formal_goal": formal_payload,
            "policy": self.policy.to_dict() if self.policy is not None else None,
            "templates": [
                item.to_dict() if hasattr(item, "to_dict") else dict(item)
                for item in self.templates
            ],
            "vocabulary_profile_id": self.vocabulary_profile_id,
            "vocabulary_version": self.vocabulary_version,
            "network_allowed": self.network_allowed,
            "deadline_unix_ms": self.deadline_unix_ms,
            "resource_budget": (
                self.resource_budget.to_dict()
                if self.resource_budget is not None
                else None
            ),
            "prose": self.prose,
            "caller_text": self.caller_text,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FormalizedGoalDevelopmentRequest":
        payload = _mapping(value, field_name="formalized goal-development request")
        return cls(
            schema=payload.get(
                "schema", FORMALIZED_GOAL_DEVELOPMENT_REQUEST_SCHEMA
            ),
            formal_goal=payload.get("formal_goal"),
            policy=payload.get("policy"),
            templates=tuple(payload.get("templates") or ()),
            compilation_result=payload.get("compilation_result"),
            vocabulary_profile_id=payload.get(
                "vocabulary_profile_id", DEFAULT_VOCABULARY_PROFILE_ID
            ),
            vocabulary_version=payload.get(
                "vocabulary_version", LOGIC_VOCABULARY_VERSION
            ),
            evidence_gaps=tuple(payload.get("evidence_gaps") or ()),
            code_references=tuple(payload.get("code_references") or ()),
            capabilities=tuple(payload.get("capabilities") or ()),
            prior_counterexamples=tuple(
                payload.get("prior_counterexamples") or ()
            ),
            reusable_receipts=tuple(payload.get("reusable_receipts") or ()),
            resource_budget=payload.get("resource_budget"),
            network_allowed=payload.get("network_allowed", False),
            deadline_unix_ms=payload.get("deadline_unix_ms"),
            prose=payload.get("prose", ""),
            caller_text=payload.get("caller_text", ""),
        )


@dataclass(frozen=True)
class FormalizedGoalDevelopmentResult(Mapping[str, Any]):
    """Route result: rejected, deterministic fallback, or unverified draft."""

    status: FormalizedRouteStatus
    request_id: str = ""
    gate_reason: FormalizationGateReason | None = None
    identifiers: FormalizedGoalIdentifiers | None = None
    provider_result: GoalDevelopmentProviderResult | None = None
    fallback_reason: GoalDevelopmentFallbackReason | None = None
    schema: str = FORMALIZED_GOAL_DEVELOPMENT_RESULT_SCHEMA
    interface: str = FORMALIZED_GOAL_DEVELOPMENT_ROUTE_INTERFACE
    route_version: str = FORMALIZED_GOAL_DEVELOPMENT_ROUTE_VERSION
    provider_id: str = LEANSTRAL_GOAL_DEVELOPMENT_PROVIDER_ID
    operation: str = LEANSTRAL_GOAL_DEVELOPMENT_OPERATION

    def __post_init__(self) -> None:
        status = (
            self.status
            if isinstance(self.status, FormalizedRouteStatus)
            else FormalizedRouteStatus(str(self.status))
        )
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self,
            "request_id",
            _text(self.request_id, field_name="request_id", required=False),
        )
        if self.gate_reason is not None and not isinstance(
            self.gate_reason, FormalizationGateReason
        ):
            object.__setattr__(
                self,
                "gate_reason",
                FormalizationGateReason(str(self.gate_reason)),
            )
        if self.fallback_reason is not None and not isinstance(
            self.fallback_reason, GoalDevelopmentFallbackReason
        ):
            object.__setattr__(
                self,
                "fallback_reason",
                GoalDevelopmentFallbackReason(str(self.fallback_reason)),
            )
        if status is FormalizedRouteStatus.REJECTED:
            if self.gate_reason is None or self.provider_result is not None:
                raise EndGoalDevelopmentError(
                    "rejected route result requires a gate reason and no provider result"
                )
        elif status is FormalizedRouteStatus.DETERMINISTIC_FALLBACK:
            if self.fallback_reason is None:
                raise EndGoalDevelopmentError(
                    "fallback route result requires a fallback reason"
                )
        elif status is FormalizedRouteStatus.DRAFT:
            if self.provider_result is None or self.provider_result.used_fallback:
                raise EndGoalDevelopmentError(
                    "draft route result requires a successful provider draft"
                )
            if self.gate_reason is not None or self.fallback_reason is not None:
                raise EndGoalDevelopmentError(
                    "draft route result cannot carry gate or fallback reasons"
                )

    @property
    def used_fallback(self) -> bool:
        return self.status is FormalizedRouteStatus.DETERMINISTIC_FALLBACK

    @property
    def rejected(self) -> bool:
        return self.status is FormalizedRouteStatus.REJECTED

    @property
    def formalization_confirmed(self) -> bool:
        return self.gate_reason is None and self.identifiers is not None

    @property
    def draft(self) -> Any:
        if self.provider_result is None:
            return None
        return self.provider_result.draft

    @property
    def assurance(self) -> AssuranceLevel:
        return AssuranceLevel.UNVERIFIED

    @property
    def result_id(self) -> str:
        return "formalized-goal-route-" + hashlib.sha256(
            canonical_json_bytes(self.to_dict(include_id=False))
        ).hexdigest()

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "interface": self.interface,
            "route_version": self.route_version,
            "operation": self.operation,
            "provider_id": self.provider_id,
            "request_id": self.request_id,
            "status": self.status.value,
            "gate_reason": (
                None if self.gate_reason is None else self.gate_reason.value
            ),
            "fallback_reason": (
                None
                if self.fallback_reason is None
                else self.fallback_reason.value
            ),
            "deterministic_fallback": self.used_fallback,
            "rejected": self.rejected,
            "formalization_confirmed": self.formalization_confirmed,
            "identifiers": (
                None if self.identifiers is None else self.identifiers.to_dict()
            ),
            "provider_result": (
                None
                if self.provider_result is None
                else self.provider_result.to_dict()
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
            "can_create_formulas": False,
            "can_mutate_assumptions": False,
            "can_claim_admission": False,
            "can_claim_completion": False,
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
    def from_dict(cls, value: Mapping[str, Any]) -> "FormalizedGoalDevelopmentResult":
        payload = _mapping(value, field_name="formalized goal-development result")
        for name in (
            "authoritative",
            "verified",
            "admitted",
            "complete",
            "kernel_checked",
            "can_mutate_root",
            "can_mutate_canonical_source",
            "can_execute_commands",
            "can_create_formulas",
            "can_mutate_assumptions",
            "can_claim_admission",
            "can_claim_completion",
        ):
            if payload.get(name, False) is not False:
                raise EndGoalDevelopmentError(
                    "formalized route result cannot claim authority"
                )
        identifiers = payload.get("identifiers")
        provider_result = payload.get("provider_result")
        result = cls(
            status=payload.get("status", ""),
            request_id=payload.get("request_id", ""),
            gate_reason=payload.get("gate_reason"),
            identifiers=(
                None
                if identifiers is None
                else FormalizedGoalIdentifiers.from_dict(identifiers)
            ),
            provider_result=(
                None
                if provider_result is None
                else GoalDevelopmentProviderResult.from_dict(provider_result)
            ),
            fallback_reason=payload.get("fallback_reason"),
            schema=payload.get(
                "schema", FORMALIZED_GOAL_DEVELOPMENT_RESULT_SCHEMA
            ),
            interface=payload.get(
                "interface", FORMALIZED_GOAL_DEVELOPMENT_ROUTE_INTERFACE
            ),
            route_version=payload.get(
                "route_version", FORMALIZED_GOAL_DEVELOPMENT_ROUTE_VERSION
            ),
            provider_id=payload.get(
                "provider_id", LEANSTRAL_GOAL_DEVELOPMENT_PROVIDER_ID
            ),
            operation=payload.get(
                "operation", LEANSTRAL_GOAL_DEVELOPMENT_OPERATION
            ),
        )
        if payload.get("result_id") not in (None, "", result.result_id):
            raise EndGoalDevelopmentError(
                "formalized route result identity is invalid"
            )
        return result


def extract_formalized_identifiers(
    request: FormalizedGoalDevelopmentRequest | Mapping[str, Any],
) -> FormalizedGoalIdentifiers:
    """Validate formalization and project only immutable identifiers.

    Raises :class:`EndGoalDevelopmentError` with a stable message when the
    gate refuses the request.  Callers that prefer a structured rejection
    should use :meth:`FormalizedGoalDevelopmentRoute.admit` instead.
    """

    if not isinstance(request, FormalizedGoalDevelopmentRequest):
        request = FormalizedGoalDevelopmentRequest.from_dict(request)

    if request.prose or request.caller_text:
        # Prose is only allowed as diagnostic context when a confirmed
        # FormalGoal is also present; without one it is a bypass attempt.
        if request.formal_goal is None:
            raise EndGoalDevelopmentError(
                "prose cannot bypass formalization; supply a confirmed FormalGoal"
            )

    if request.formal_goal is None:
        raise EndGoalDevelopmentError(
            "formal_goal is required; prose cannot bypass formalization"
        )

    if isinstance(request.formal_goal, str) or _looks_like_prose_primary(
        request.formal_goal
    ):
        raise EndGoalDevelopmentError(
            "prose cannot bypass formalization; supply a confirmed FormalGoal"
        )

    formal_goal = _coerce_formal_goal(request.formal_goal)
    gate_reason = _require_confirmed_formal_goal(formal_goal)
    if gate_reason is FormalizationGateReason.UNCONFIRMED:
        raise EndGoalDevelopmentError(
            "formal_goal must be confirmed before Leanstral goal development"
        )
    if gate_reason is FormalizationGateReason.AMBIGUITY_UNRESOLVED:
        raise EndGoalDevelopmentError(
            "formal_goal ambiguity must be resolved before Leanstral goal development"
        )
    if gate_reason is FormalizationGateReason.AUTHORITY_CLAIM:
        raise EndGoalDevelopmentError(
            "formal_goal cannot claim proof or completion authority"
        )
    if gate_reason is not None:
        raise EndGoalDevelopmentError(
            f"formalization gate rejected request: {gate_reason.value}"
        )

    if not request.templates:
        raise EndGoalDevelopmentError(
            "reviewed templates are required before Leanstral goal development"
        )

    templates: list[GoalDevelopmentTemplate] = []
    for index, item in enumerate(request.templates):
        if isinstance(item, GoalDevelopmentTemplate):
            templates.append(item)
        elif isinstance(item, Mapping):
            templates.append(GoalDevelopmentTemplate.from_dict(item))
        else:
            raise EndGoalDevelopmentError(
                f"templates[{index}] must be a GoalDevelopmentTemplate"
            )

    end_goal = _end_goal_of(formal_goal)
    try:
        repository_tree_id = _repository_tree_id(end_goal)
    except EndGoalDevelopmentError:
        raise
    evidence = _evidence_ids(end_goal, formal_goal)
    scopes = _scope_ids(end_goal)
    formula_id = _formula_id_from_compilation(request.compilation_result, formal_goal)
    root_goal_id = _attr(formal_goal, "root_goal_id", None) or _attr(
        end_goal, "root_goal_id", None
    ) or _attr(end_goal, "goal_id", "")
    root_goal_id = _text(root_goal_id, field_name="root_goal_id")

    return FormalizedGoalIdentifiers(
        formal_goal_id=_text(
            _attr(formal_goal, "formal_goal_id", ""),
            field_name="formal_goal_id",
        ),
        root_goal_id=root_goal_id,
        root_goal_content_id=_content_id_of(formal_goal),
        satisfaction_formula_id=formula_id,
        assumption_ids=_assumption_ids(end_goal),
        evidence_requirement_ids=evidence,
        vocabulary_profile_id=request.vocabulary_profile_id,
        vocabulary_version=request.vocabulary_version,
        repository_tree_id=repository_tree_id,
        scope_ids=scopes,
        template_ids=tuple(item.template_id for item in templates),
        selected_interpretation_id=_text(
            _attr(formal_goal, "selected_interpretation_id", ""),
            field_name="selected_interpretation_id",
            required=False,
        ),
        confirmation_receipt_id=_text(
            _attr(formal_goal, "confirmation_receipt_id", ""),
            field_name="confirmation_receipt_id",
            required=False,
        ),
    )


def build_goal_development_request(
    identifiers: FormalizedGoalIdentifiers,
    policy: GoalDevelopmentPolicy,
) -> GoalDevelopmentRequest:
    """Build the frozen GoalDevelopmentRequest from identifier-only data."""

    return GoalDevelopmentRequest(
        root_goal_id=identifiers.root_goal_id,
        root_goal_content_id=identifiers.root_goal_content_id,
        satisfaction_formula_id=identifiers.satisfaction_formula_id,
        assumption_ids=identifiers.assumption_ids,
        evidence_requirement_ids=identifiers.evidence_requirement_ids,
        vocabulary_profile_id=identifiers.vocabulary_profile_id,
        vocabulary_version=identifiers.vocabulary_version,
        repository_tree_id=identifiers.repository_tree_id,
        scope_ids=identifiers.scope_ids,
        policy_digest=policy.policy_digest,
        mode=policy.mode,
    )


def build_formalized_leanstral_invocation(
    request: FormalizedGoalDevelopmentRequest | Mapping[str, Any],
    *,
    identifiers: FormalizedGoalIdentifiers | None = None,
) -> LeanstralGoalDevelopmentInvocation:
    """Construct a Leanstral invocation from a formalized route request.

    The resulting prompt envelope contains only immutable identifiers and
    reviewed template / evidence / capability records.  Prose and formula text
    are never included.
    """

    if not isinstance(request, FormalizedGoalDevelopmentRequest):
        request = FormalizedGoalDevelopmentRequest.from_dict(request)
    ids = identifiers or extract_formalized_identifiers(request)
    policy = request.policy
    assert isinstance(policy, GoalDevelopmentPolicy)

    templates: list[GoalDevelopmentTemplate] = []
    for item in request.templates:
        if isinstance(item, GoalDevelopmentTemplate):
            templates.append(item)
        else:
            templates.append(GoalDevelopmentTemplate.from_dict(item))

    goal_request = build_goal_development_request(ids, policy)
    context = build_leanstral_goal_development_context(
        goal_request,
        templates=templates,
        goal=ImmutableGoalRecord(
            goal_id=ids.root_goal_id,
            content_id=ids.root_goal_content_id,
            satisfaction_formula_id=ids.satisfaction_formula_id,
        ),
        evidence_gaps=request.evidence_gaps,
        code_references=request.code_references,
        capabilities=request.capabilities,
        prior_counterexamples=request.prior_counterexamples,
        reusable_receipts=request.reusable_receipts,
    )
    # Defense in depth: the context / request dicts must never smuggle prose.
    for payload in (goal_request.to_dict(), context.to_dict()):
        for forbidden in _FORBIDDEN_PROVIDER_PAYLOAD_KEYS:
            if forbidden in payload and payload[forbidden] not in (None, False, "", [], {}):
                raise EndGoalDevelopmentError(
                    f"formalized invocation cannot expose {forbidden} to Leanstral"
                )

    return LeanstralGoalDevelopmentInvocation(
        request=goal_request,
        policy=policy,
        context=context,
        resource_budget=request.resource_budget or ResourceBudget(),
        network_allowed=request.network_allowed,
        deadline_unix_ms=request.deadline_unix_ms,
    )


class FormalizedGoalDevelopmentRoute:
    """``FormalizedGoalDevelopmentRoute@1`` supervisor adapter.

    Owns the post-formalization gate and delegates only admitted, identifier-
    only envelopes to :class:`LeanstralGoalDevelopmentProvider`.  Preserves the
    existing Leanstral capability-isolation boundary and provider modes.
    """

    INTERFACE: Final = FORMALIZED_GOAL_DEVELOPMENT_ROUTE_INTERFACE
    VERSION: Final = FORMALIZED_GOAL_DEVELOPMENT_ROUTE_VERSION
    SCHEMA: Final = FORMALIZED_GOAL_DEVELOPMENT_ROUTE_SCHEMA

    def __init__(
        self,
        provider: LeanstralGoalDevelopmentProvider
        | LeanstralGoalDevelopmentProviderConfig
        | None = None,
        *,
        config: LeanstralGoalDevelopmentProviderConfig | None = None,
        llm_generate: Any | None = None,
    ) -> None:
        # Accept either a ready provider or a provider config as the first
        # positional argument (mirrors LeanstralGoalDevelopmentProvider).
        if isinstance(provider, LeanstralGoalDevelopmentProviderConfig):
            if config is not None:
                raise EndGoalDevelopmentError(
                    "provider config cannot be supplied twice"
                )
            config = provider
            provider = None
        if provider is not None and (
            config is not None or llm_generate is not None
        ):
            raise EndGoalDevelopmentError(
                "provider cannot be combined with config/llm_generate overrides"
            )
        if provider is not None:
            if not isinstance(provider, LeanstralGoalDevelopmentProvider):
                raise EndGoalDevelopmentError(
                    "provider must be a LeanstralGoalDevelopmentProvider"
                )
            self._provider = provider
        else:
            self._provider = LeanstralGoalDevelopmentProvider(
                config, llm_generate=llm_generate
            )

    @property
    def provider(self) -> LeanstralGoalDevelopmentProvider:
        return self._provider

    @property
    def interface(self) -> str:
        return self.INTERFACE

    def capabilities(self) -> dict[str, Any]:
        capability = self._provider.capabilities()
        payload = capability.to_dict() if hasattr(capability, "to_dict") else {}
        return {
            "interface": self.INTERFACE,
            "route_version": self.VERSION,
            "schema": self.SCHEMA,
            "provider_id": LEANSTRAL_GOAL_DEVELOPMENT_PROVIDER_ID,
            "operation": LEANSTRAL_GOAL_DEVELOPMENT_OPERATION,
            "requires_confirmed_formal_goal": True,
            "prose_bypass_allowed": False,
            "exposes_only_immutable_identifiers": True,
            "can_create_formulas": False,
            "can_mutate_assumptions": False,
            "can_mutate_source": False,
            "can_execute_commands": False,
            "can_claim_admission": False,
            "can_claim_completion": False,
            "can_claim_proof": False,
            "provider_capability": payload,
        }

    def admit(
        self, request: FormalizedGoalDevelopmentRequest | Mapping[str, Any]
    ) -> FormalizedGoalDevelopmentResult | FormalizedGoalIdentifiers:
        """Admit a request past the formalization gate or return rejection.

        Returns identifiers on success, or a rejected
        :class:`FormalizedGoalDevelopmentResult` without invoking Leanstral.
        """

        if not isinstance(request, FormalizedGoalDevelopmentRequest):
            try:
                request = FormalizedGoalDevelopmentRequest.from_dict(request)
            except (EndGoalDevelopmentError, ContractValidationError) as exc:
                return FormalizedGoalDevelopmentResult(
                    status=FormalizedRouteStatus.REJECTED,
                    gate_reason=FormalizationGateReason.INVALID_FORMAL_GOAL,
                    request_id="",
                )

        # Prose-only / missing formal goal paths → stable rejection, no model.
        if request.formal_goal is None and (request.prose or request.caller_text):
            return FormalizedGoalDevelopmentResult(
                status=FormalizedRouteStatus.REJECTED,
                gate_reason=FormalizationGateReason.PROSE_BYPASS,
            )
        if request.formal_goal is None:
            return FormalizedGoalDevelopmentResult(
                status=FormalizedRouteStatus.REJECTED,
                gate_reason=FormalizationGateReason.MISSING_FORMAL_GOAL,
            )
        if isinstance(request.formal_goal, str) or _looks_like_prose_primary(
            request.formal_goal
        ):
            return FormalizedGoalDevelopmentResult(
                status=FormalizedRouteStatus.REJECTED,
                gate_reason=FormalizationGateReason.PROSE_BYPASS,
            )

        try:
            formal_goal = _coerce_formal_goal(request.formal_goal)
        except EndGoalDevelopmentError:
            # Distinguish prose-shaped failures already handled above.
            return FormalizedGoalDevelopmentResult(
                status=FormalizedRouteStatus.REJECTED,
                gate_reason=FormalizationGateReason.INVALID_FORMAL_GOAL,
            )

        gate_reason = _require_confirmed_formal_goal(formal_goal)
        if gate_reason is not None:
            return FormalizedGoalDevelopmentResult(
                status=FormalizedRouteStatus.REJECTED,
                gate_reason=gate_reason,
            )

        if not request.templates:
            return FormalizedGoalDevelopmentResult(
                status=FormalizedRouteStatus.REJECTED,
                gate_reason=FormalizationGateReason.MISSING_TEMPLATES,
            )

        try:
            return extract_formalized_identifiers(request)
        except EndGoalDevelopmentError as exc:
            message = str(exc).casefold()
            if "tree_id" in message or "repository" in message:
                reason = FormalizationGateReason.MISSING_REPOSITORY_TREE
            elif "evidence" in message:
                reason = FormalizationGateReason.MISSING_EVIDENCE
            elif "scope" in message:
                reason = FormalizationGateReason.MISSING_SCOPE
            elif "template" in message:
                reason = FormalizationGateReason.MISSING_TEMPLATES
            else:
                reason = FormalizationGateReason.INVALID_FORMAL_GOAL
            return FormalizedGoalDevelopmentResult(
                status=FormalizedRouteStatus.REJECTED,
                gate_reason=reason,
            )

    def build_invocation(
        self, request: FormalizedGoalDevelopmentRequest | Mapping[str, Any]
    ) -> LeanstralGoalDevelopmentInvocation:
        """Build a Leanstral invocation only after formalization admits."""

        admitted = self.admit(request)
        if isinstance(admitted, FormalizedGoalDevelopmentResult):
            reason = (
                admitted.gate_reason.value
                if admitted.gate_reason is not None
                else "rejected"
            )
            raise EndGoalDevelopmentError(
                f"formalization gate rejected request: {reason}"
            )
        if not isinstance(request, FormalizedGoalDevelopmentRequest):
            request = FormalizedGoalDevelopmentRequest.from_dict(request)
        return build_formalized_leanstral_invocation(
            request, identifiers=admitted
        )

    def develop(
        self,
        request: FormalizedGoalDevelopmentRequest | Mapping[str, Any],
        *,
        cancellation: CancellationToken | Any | None = None,
    ) -> FormalizedGoalDevelopmentResult:
        """Route a formalized goal through Leanstral goal development.

        Never stalls the supervisor: gate rejections and provider transport /
        schema failures return immediately as structured route results.
        """

        if not isinstance(request, FormalizedGoalDevelopmentRequest):
            try:
                request = FormalizedGoalDevelopmentRequest.from_dict(request)
            except (EndGoalDevelopmentError, ContractValidationError, TypeError, ValueError):
                return FormalizedGoalDevelopmentResult(
                    status=FormalizedRouteStatus.REJECTED,
                    gate_reason=FormalizationGateReason.INVALID_FORMAL_GOAL,
                )

        admitted = self.admit(request)
        if isinstance(admitted, FormalizedGoalDevelopmentResult):
            return admitted

        try:
            invocation = build_formalized_leanstral_invocation(
                request, identifiers=admitted
            )
        except (EndGoalDevelopmentError, ContractValidationError) as exc:
            return FormalizedGoalDevelopmentResult(
                status=FormalizedRouteStatus.REJECTED,
                gate_reason=FormalizationGateReason.INVALID_FORMAL_GOAL,
                identifiers=admitted,
                request_id="",
            )

        # Provider.develop already maps timeout / unavailable / malformed /
        # cancelled / overloaded into GoalDevelopmentProviderResult fallbacks.
        provider_result = self._provider.develop(
            invocation, cancellation=cancellation
        )

        if provider_result.used_fallback:
            return FormalizedGoalDevelopmentResult(
                status=FormalizedRouteStatus.DETERMINISTIC_FALLBACK,
                request_id=provider_result.request_id,
                identifiers=admitted,
                provider_result=provider_result,
                fallback_reason=provider_result.fallback_reason,
            )

        return FormalizedGoalDevelopmentResult(
            status=FormalizedRouteStatus.DRAFT,
            request_id=provider_result.request_id,
            identifiers=admitted,
            provider_result=provider_result,
        )


def create_formalized_goal_development_route(
    *,
    provider: LeanstralGoalDevelopmentProvider | None = None,
    config: LeanstralGoalDevelopmentProviderConfig | None = None,
    llm_generate: Any | None = None,
) -> FormalizedGoalDevelopmentRoute:
    """Factory for :class:`FormalizedGoalDevelopmentRoute`."""

    return FormalizedGoalDevelopmentRoute(
        provider=provider,
        config=config,
        llm_generate=llm_generate,
    )


__all__ = [
    "DEFAULT_VOCABULARY_PROFILE_ID",
    "FORMALIZED_GOAL_DEVELOPMENT_REQUEST_SCHEMA",
    "FORMALIZED_GOAL_DEVELOPMENT_RESULT_SCHEMA",
    "FORMALIZED_GOAL_DEVELOPMENT_ROUTE_INTERFACE",
    "FORMALIZED_GOAL_DEVELOPMENT_ROUTE_SCHEMA",
    "FORMALIZED_GOAL_DEVELOPMENT_ROUTE_VERSION",
    "FORMALIZED_GOAL_IDENTIFIERS_SCHEMA",
    "EndGoalDevelopmentError",
    "FormalizationGateReason",
    "FormalizedGoalDevelopmentRequest",
    "FormalizedGoalDevelopmentResult",
    "FormalizedGoalDevelopmentRoute",
    "FormalizedGoalIdentifiers",
    "FormalizedRouteStatus",
    "build_formalized_leanstral_invocation",
    "build_goal_development_request",
    "create_formalized_goal_development_route",
    "extract_formalized_identifiers",
]
