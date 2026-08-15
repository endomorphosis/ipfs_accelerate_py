"""Provider-neutral GUI patch proposal interface (VGO-045).

Interfaces owned by this module:

* ``GuiPatchProposer@1`` — accept a caller-selected route and emit a typed
  proposal, human-review request, or fail-closed rejection
* ``DeterministicGuiTransformation@1`` — exact mechanical label, deprecated
  prop, design-token, ARIA, route, and action-binding substitutions
* ``HumanGuiReviewRequest@1`` — escalation record with no fabricated patch

This is a dependency-injected provider interface, not model routing.  It does
not import a router, choose a vendor, or invent a patch when the provider is
absent or raises.  Caller-declared method and tier are recorded as given
(after a closed vocabulary check); vendor tokens are rejected.

Fail-closed invariants:

* mechanical exact transformations are deterministic;
* opaque, ambiguous, policy-bound, security-sensitive, repeatedly failed, or
  constraint-conflicted requests escalate without a patch;
* provider absence on a model route cannot broaden scope or fabricate a
  patch;
* provider output cannot add files, change the route, or name a vendor;
* security authority reason codes apply to host/credential/command leakage.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from difflib import unified_diff
from enum import Enum
from types import MappingProxyType
from typing import Any, Final, Protocol

from .authority import (
    AuthorityReasonCode,
    DEFAULT_ALLOWED_ROOTS,
    FORBIDDEN_BROWSER_COMMAND_FIELDS,
    FORBIDDEN_BROWSER_CREDENTIAL_FIELDS,
    FORBIDDEN_BROWSER_PATH_FIELDS,
    GuiAuthorityError,
    _normalize_repo_path,
    path_has_forbidden_segment,
    path_under_allowed_roots,
)
from .patch_scope import (
    GUI_IMPROVEMENT_PROPOSAL_INTERFACE,
    GUI_IMPROVEMENT_PROPOSAL_SCHEMA,
)

# ---------------------------------------------------------------------------
# Interface / schema identity
# ---------------------------------------------------------------------------

GUI_PATCH_PROPOSER_INTERFACE: Final[str] = "GuiPatchProposer@1"
DETERMINISTIC_GUI_TRANSFORMATION_INTERFACE: Final[str] = (
    "DeterministicGuiTransformation@1"
)
HUMAN_GUI_REVIEW_REQUEST_INTERFACE: Final[str] = "HumanGuiReviewRequest@1"

GUI_PATCH_PROPOSER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/gui-optimizer/patch-proposer@1"
)
DETERMINISTIC_GUI_TRANSFORMATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/gui-optimizer/"
    "deterministic-transformation@1"
)
HUMAN_GUI_REVIEW_REQUEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/gui-optimizer/"
    "human-review-request@1"
)
GUI_PROPOSAL_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/gui-optimizer/proposal-result@1"
)

DEFAULT_MAX_PRIOR_FAILURES: Final[int] = 2
_IDENTIFIER_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/#@-]{0,255}$")

# Caller-selected routes.  These are not vendors and are not chosen here.
class ProposalRoute(str, Enum):
    DETERMINISTIC_TRANSFORM = "deterministic_transform"
    SMALL_LOCAL_MODEL = "small_local_model"
    MEDIUM_MODEL = "medium_model"
    FRONTIER_MODEL = "frontier_model"
    HUMAN_REVIEW = "human_review"


MODEL_ROUTES: Final[frozenset[ProposalRoute]] = frozenset(
    {
        ProposalRoute.SMALL_LOCAL_MODEL,
        ProposalRoute.MEDIUM_MODEL,
        ProposalRoute.FRONTIER_MODEL,
    }
)

ROUTE_DEFAULT_TIER: Final[dict[ProposalRoute, str]] = {
    ProposalRoute.DETERMINISTIC_TRANSFORM: "deterministic",
    ProposalRoute.SMALL_LOCAL_MODEL: "small_local",
    ProposalRoute.MEDIUM_MODEL: "medium",
    ProposalRoute.FRONTIER_MODEL: "frontier",
    ProposalRoute.HUMAN_REVIEW: "human",
}

ALLOWED_TIERS: Final[frozenset[str]] = frozenset(ROUTE_DEFAULT_TIER.values())


class TransformationKind(str, Enum):
    LABEL = "label"
    DEPRECATED_PROP = "deprecated_prop"
    DESIGN_TOKEN = "design_token"
    ARIA_REFERENCE = "aria_reference"
    EXACT_ROUTE = "exact_route"
    EXACT_ACTION_BINDING = "exact_action_binding"


KIND_DEFAULT_METHOD: Final[dict[TransformationKind, str]] = {
    TransformationKind.LABEL: "exact_label_substitution",
    TransformationKind.DEPRECATED_PROP: "deprecated_prop_replacement",
    TransformationKind.DESIGN_TOKEN: "design_token_substitution",
    TransformationKind.ARIA_REFERENCE: "aria_reference_repair",
    TransformationKind.EXACT_ROUTE: "exact_route_migration",
    TransformationKind.EXACT_ACTION_BINDING: "exact_action_binding_migration",
}

POLICY_EXEMPT_KINDS: Final[frozenset[TransformationKind]] = frozenset(
    {TransformationKind.EXACT_ACTION_BINDING}
)


class ProposalDisposition(str, Enum):
    PROPOSE = "propose"
    ESCALATE = "escalate"
    REJECT = "reject"


class EscalationKind(str, Enum):
    OPAQUE = "opaque"
    AMBIGUOUS = "ambiguous"
    POLICY_BOUND = "policy_bound"
    REPEATED_FAILURE = "repeated_failure"
    SECURITY = "security"
    CONSTRAINT_CONFLICT = "constraint_conflict"
    PROVIDER_ABSENT = "provider_absent"
    PROVIDER_EXCEPTION = "provider_exception"
    HUMAN_ROUTE = "human_route"


class ProposalReasonCode(str, Enum):
    PROPOSED = "proposed"
    ESCALATED = "escalated"
    REJECTED = "rejected"
    DETERMINISTIC_TRANSFORM = "deterministic_transform"
    OPAQUE_CONTEXT = "opaque_context"
    AMBIGUOUS_TRANSFORM = "ambiguous_transform"
    POLICY_BOUND = "policy_bound"
    REPEATED_FAILURE = "repeated_failure"
    CONSTRAINT_CONFLICT = "constraint_conflict"
    PROVIDER_ABSENT = "provider_absent"
    PROVIDER_EXCEPTION = "provider_exception"
    HUMAN_REVIEW_REQUIRED = "human_review_required"
    VENDOR_FORBIDDEN = "vendor_forbidden"
    SCOPE_BROADENED = "scope_broadened"
    MISSING_SOURCE = "missing_source"
    INVALID_PROPOSAL_INPUT = "invalid_proposal_input"
    UNKNOWN_FIELD = AuthorityReasonCode.UNKNOWN_FIELD.value
    INVALID_COLLECTION_TYPE = AuthorityReasonCode.INVALID_COLLECTION_TYPE.value
    PATH_OUTSIDE_ALLOWED_ROOTS = (
        AuthorityReasonCode.PATH_OUTSIDE_ALLOWED_ROOTS.value
    )
    PATH_ABSOLUTE_OR_TRAVERSAL = (
        AuthorityReasonCode.PATH_ABSOLUTE_OR_TRAVERSAL.value
    )
    PATH_FORBIDDEN_SEGMENT = AuthorityReasonCode.PATH_FORBIDDEN_SEGMENT.value
    BROWSER_HOST_PATH_FORBIDDEN = (
        AuthorityReasonCode.BROWSER_HOST_PATH_FORBIDDEN.value
    )
    BROWSER_COMMAND_FORBIDDEN = (
        AuthorityReasonCode.BROWSER_COMMAND_FORBIDDEN.value
    )
    BROWSER_CREDENTIAL_FORBIDDEN = (
        AuthorityReasonCode.BROWSER_CREDENTIAL_FORBIDDEN.value
    )
    SECURITY_REGRESSION = AuthorityReasonCode.SECURITY_REGRESSION.value


_VENDOR_TOKEN_RE: Final = re.compile(
    r"(?i)(?<![a-z0-9])("
    r"openai|anthropic|claude|gpt-?[0-9]|gpt4|chatgpt|grok|xai|"
    r"gemini|mistral|llama|copilot|bedrock|vertexai|vertex-ai|"
    r"togetherai|ollama|cohere"
    r")(?![a-z0-9])"
)

_SECURITY_SELECTORS: Final[frozenset[str]] = (
    FORBIDDEN_BROWSER_COMMAND_FIELDS
    | FORBIDDEN_BROWSER_CREDENTIAL_FIELDS
    | FORBIDDEN_BROWSER_PATH_FIELDS
    | frozenset(
        {
            "authorization",
            "backend_credentials",
            "bearer_token",
            "api_key",
            "password",
            "secret",
            "credential",
            "credentials",
        }
    )
)

_REQUEST_KEYS: Final[frozenset[str]] = frozenset(
    {
        "acceptance_criteria",
        "ambiguous",
        "analysis_classification",
        "application_id",
        "constraint_conflict",
        "context_pack",
        "declared_method",
        "declared_tier",
        "escalation_conditions",
        "expected_screenshot_ids",
        "expected_test_ids",
        "intended_component_ids",
        "intended_file_paths",
        "objective",
        "opaque",
        "policy_bound",
        "prior_failure_count",
        "request_id",
        "route_kind",
        "screen_id",
        "security_sensitive",
        "state_effect_ids",
        "transformations",
        "verification_status",
        "visual_effect_summary",
    }
)
_PACK_KEYS: Final[frozenset[str]] = frozenset(
    {
        "acceptance_criteria",
        "analysis_classification",
        "application_id",
        "escalation_conditions",
        "formal_invariant_failures",
        "objective",
        "pack_id",
        "raw_sources",
        "screen_id",
        "verification_status",
    }
)
_SOURCE_KEYS: Final[frozenset[str]] = frozenset(
    {"component_id", "content", "editable", "path"}
)
_TRANSFORM_KEYS: Final[frozenset[str]] = frozenset(
    {
        "expected_count",
        "find",
        "interface",
        "kind",
        "path",
        "replace",
        "schema_version",
    }
)
_PROVIDER_RESULT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "declared_method",
        "declared_tier",
        "patch_text",
        "proposal",
        "reason_codes",
    }
)
_OPAQUE_STATUSES: Final[frozenset[str]] = frozenset(
    {"stale", "invalid", "simulated"}
)


class GuiProposalError(ValueError):
    """Malformed proposal input.  Never yields a fabricated patch."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = ProposalReasonCode.INVALID_PROPOSAL_INPUT.value,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = str(reason_code)
        self.details = dict(details or {})


class GuiProposalProvider(Protocol):
    """Injected proposal backend.  Implementations must not choose vendors."""

    def propose(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return a closed provider result or raise."""


# ---------------------------------------------------------------------------
# Wire helpers
# ---------------------------------------------------------------------------


def _exact_str(value: Any, name: str) -> str:
    if type(value) is not str:
        raise GuiProposalError(
            f"{name} must be a string",
            details={"field": name, "value_type": type(value).__name__},
        )
    return value


def _text(value: Any, name: str, *, required: bool = True) -> str:
    text = _exact_str(value, name)
    if "\x00" in text:
        raise GuiProposalError(f"{name} must not contain NUL", details={"field": name})
    stripped = text.strip()
    if required and not stripped:
        raise GuiProposalError(f"{name} must not be empty", details={"field": name})
    return stripped


def _raw_content(value: Any, name: str) -> str:
    text = _exact_str(value, name)
    if "\x00" in text:
        raise GuiProposalError(f"{name} must not contain NUL", details={"field": name})
    return text


def _identifier(value: Any, name: str) -> str:
    text = _exact_str(value, name)
    if not _IDENTIFIER_RE.fullmatch(text):
        raise GuiProposalError(
            f"{name} is not a stable identifier",
            details={"field": name},
        )
    return text


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise GuiProposalError(
            f"{name} must be a boolean",
            details={"field": name, "value_type": type(value).__name__},
        )
    return value


def _nonneg_int(value: Any, name: str) -> int:
    if type(value) is not int or type(value) is bool:
        raise GuiProposalError(
            f"{name} must be an integer",
            details={"field": name, "value_type": type(value).__name__},
        )
    if value < 0:
        raise GuiProposalError(
            f"{name} must be a non-negative integer",
            details={"field": name, "value": value},
        )
    return value


def _positive_int(value: Any, name: str) -> int:
    number = _nonneg_int(value, name)
    if number < 1:
        raise GuiProposalError(
            f"{name} must be a positive integer",
            details={"field": name, "value": value},
        )
    return number


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise GuiProposalError(
            f"{name} must be a JSON object",
            reason_code=ProposalReasonCode.INVALID_COLLECTION_TYPE.value,
            details={"field": name, "value_type": type(value).__name__},
        )
    for key in value:
        if type(key) is not str:
            raise GuiProposalError(
                f"{name} keys must be strings",
                reason_code=ProposalReasonCode.INVALID_COLLECTION_TYPE.value,
                details={"field": name, "key_type": type(key).__name__},
            )
    return value


def _require_json_array(value: Any, name: str) -> list[Any]:
    if type(value) is not list:
        raise GuiProposalError(
            f"{name} must be a JSON array (list); "
            f"{type(value).__name__} is not a valid collection",
            reason_code=ProposalReasonCode.INVALID_COLLECTION_TYPE.value,
            details={"field": name, "value_type": type(value).__name__},
        )
    return value


def _reject_unknown(
    payload: Mapping[str, Any], allowed: frozenset[str], noun: str
) -> None:
    unknown = sorted(set(payload) - set(allowed))
    if unknown:
        raise GuiProposalError(
            f"{noun} contains unknown fields: {unknown}",
            reason_code=ProposalReasonCode.UNKNOWN_FIELD.value,
            details={"noun": noun, "unknown_fields": unknown},
        )


def _reject_present_null(payload: Mapping[str, Any], key: str) -> None:
    if key in payload and payload[key] is None:
        raise GuiProposalError(
            f"{key} must not be null when present",
            details={"field": key, "value_type": "NoneType"},
        )


def _optional_bool(payload: Mapping[str, Any], key: str, default: bool) -> bool:
    if key not in payload:
        return default
    _reject_present_null(payload, key)
    return _bool(payload[key], key)


def _optional_text(payload: Mapping[str, Any], key: str) -> str:
    if key not in payload:
        return ""
    _reject_present_null(payload, key)
    return _text(payload[key], key, required=False)


def _optional_identifier(payload: Mapping[str, Any], key: str) -> str:
    if key not in payload:
        return ""
    _reject_present_null(payload, key)
    return _identifier(payload[key], key)


def _unique_strings(value: Any, name: str, *, required: bool = False) -> tuple[str, ...]:
    if value is None:
        raise GuiProposalError(
            f"{name} must be a JSON array; null is not a collection",
            reason_code=ProposalReasonCode.INVALID_COLLECTION_TYPE.value,
            details={"field": name, "value_type": "NoneType"},
        )
    sequence = _require_json_array(value, name)
    items: list[str] = []
    seen: set[str] = set()
    for index, raw in enumerate(sequence):
        text = _identifier(raw, f"{name}[{index}]")
        if text in seen:
            raise GuiProposalError(
                f"{name} entries must be unique",
                details={"field": name, "duplicate": text},
            )
        seen.add(text)
        items.append(text)
    if required and not items:
        raise GuiProposalError(f"{name} must not be empty", details={"field": name})
    return tuple(items)


def _unique_texts(value: Any, name: str, *, required: bool = False) -> tuple[str, ...]:
    sequence = _require_json_array(value, name)
    items: list[str] = []
    seen: set[str] = set()
    for index, raw in enumerate(sequence):
        text = _text(raw, f"{name}[{index}]")
        if text in seen:
            raise GuiProposalError(
                f"{name} entries must be unique",
                details={"field": name, "duplicate": text},
            )
        seen.add(text)
        items.append(text)
    if required and not items:
        raise GuiProposalError(f"{name} must not be empty", details={"field": name})
    return tuple(items)


def _unique_paths(value: Any, name: str, *, required: bool = False) -> tuple[str, ...]:
    sequence = _require_json_array(value, name)
    items: list[str] = []
    seen: set[str] = set()
    for index, raw in enumerate(sequence):
        path = _repo_path(raw, f"{name}[{index}]")
        if path in seen:
            raise GuiProposalError(
                f"{name} entries must be unique",
                details={"field": name, "duplicate": path},
            )
        seen.add(path)
        items.append(path)
    if required and not items:
        raise GuiProposalError(f"{name} must not be empty", details={"field": name})
    return tuple(items)


def _repo_path(value: Any, name: str) -> str:
    try:
        path = _normalize_repo_path(value, name)
    except GuiAuthorityError as error:
        raise GuiProposalError(
            str(error),
            reason_code=getattr(
                error,
                "reason_code",
                ProposalReasonCode.PATH_ABSOLUTE_OR_TRAVERSAL.value,
            ),
            details=getattr(error, "details", {"field": name}),
        ) from error
    if path_has_forbidden_segment(path):
        raise GuiProposalError(
            f"{name} touches a forbidden path segment",
            reason_code=ProposalReasonCode.PATH_FORBIDDEN_SEGMENT.value,
            details={"field": name, "path": path},
        )
    if not path_under_allowed_roots(path, allowed_roots=DEFAULT_ALLOWED_ROOTS):
        raise GuiProposalError(
            f"{name} is outside allowed optimizer roots",
            reason_code=ProposalReasonCode.PATH_OUTSIDE_ALLOWED_ROOTS.value,
            details={"field": name, "path": path},
        )
    return path


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _digest_id(prefix: str, payload: Mapping[str, Any] | bytes) -> str:
    digest = hashlib.sha256(
        payload if isinstance(payload, bytes) else _canonical_bytes(payload)
    ).hexdigest()
    return f"{prefix}:{digest[:24]}"


def _contains_vendor(value: str) -> bool:
    return _VENDOR_TOKEN_RE.search(value) is not None


def _reject_vendor(value: str, name: str) -> str:
    if _contains_vendor(value):
        raise GuiProposalError(
            f"{name} must not name a vendor",
            reason_code=ProposalReasonCode.VENDOR_FORBIDDEN.value,
            details={"field": name},
        )
    return value


def _selector_hits(text: str) -> tuple[str, ...]:
    compact = re.sub(r"[^a-z0-9]+", "", text.lower())
    hits: list[str] = []
    for selector in sorted(_SECURITY_SELECTORS):
        token = re.sub(r"[^a-z0-9]+", "", selector.lower())
        if token and token in compact:
            hits.append(selector)
    return tuple(hits)


def _as_route(value: Any, *, wire: bool) -> ProposalRoute:
    if not wire and type(value) is ProposalRoute:
        return value
    if type(value) is not str:
        raise GuiProposalError(
            "route_kind must be a string",
            details={"value_type": type(value).__name__},
        )
    try:
        return ProposalRoute(_text(value, "route_kind"))
    except ValueError as error:
        raise GuiProposalError(
            f"unknown route_kind: {value}",
            details={"route_kind": value},
        ) from error


def _as_kind(value: Any, *, wire: bool) -> TransformationKind:
    if not wire and type(value) is TransformationKind:
        return value
    if type(value) is not str:
        raise GuiProposalError(
            "kind must be a string",
            details={"value_type": type(value).__name__},
        )
    try:
        return TransformationKind(_text(value, "kind"))
    except ValueError as error:
        raise GuiProposalError(
            f"unknown transformation kind: {value}",
            details={"kind": value},
        ) from error


# ---------------------------------------------------------------------------
# Wire / typed records
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DeterministicGuiTransformation:
    """Exact mechanical substitution.  Find/replace is literal, not a regex."""

    kind: TransformationKind
    path: str
    find: str
    replace: str
    expected_count: int = 1
    interface: str = DETERMINISTIC_GUI_TRANSFORMATION_INTERFACE
    schema_version: str = DETERMINISTIC_GUI_TRANSFORMATION_SCHEMA

    def __post_init__(self) -> None:
        if type(self.kind) is not TransformationKind:
            object.__setattr__(self, "kind", _as_kind(self.kind, wire=False))
        object.__setattr__(self, "path", _repo_path(self.path, "path"))
        find = _raw_content(self.find, "find")
        replace = _raw_content(self.replace, "replace")
        if not find:
            raise GuiProposalError("find must be a nonempty exact fragment")
        if find == replace:
            raise GuiProposalError("find and replace must differ")
        object.__setattr__(self, "find", find)
        object.__setattr__(self, "replace", replace)
        object.__setattr__(
            self, "expected_count", _positive_int(self.expected_count, "expected_count")
        )
        interface = _text(self.interface, "interface")
        if interface != DETERMINISTIC_GUI_TRANSFORMATION_INTERFACE:
            raise GuiProposalError(
                "transformation interface must be DeterministicGuiTransformation@1",
                details={"interface": interface},
            )
        schema = _text(self.schema_version, "schema_version")
        if schema != DETERMINISTIC_GUI_TRANSFORMATION_SCHEMA:
            raise GuiProposalError(
                "transformation schema_version mismatch",
                details={"schema_version": schema},
            )
        object.__setattr__(self, "interface", interface)
        object.__setattr__(self, "schema_version", schema)
        for label, blob in (("find", find), ("replace", replace)):
            hits = _selector_hits(blob)
            if hits:
                raise GuiProposalError(
                    f"{label} contains a forbidden host/credential selector",
                    reason_code=ProposalReasonCode.BROWSER_CREDENTIAL_FORBIDDEN.value
                    if any(h in FORBIDDEN_BROWSER_CREDENTIAL_FIELDS for h in hits)
                    else ProposalReasonCode.SECURITY_REGRESSION.value,
                    details={"field": label, "selectors": list(hits)},
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_count": self.expected_count,
            "find": self.find,
            "interface": self.interface,
            "kind": self.kind.value,
            "path": self.path,
            "replace": self.replace,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "DeterministicGuiTransformation":
        payload = _require_mapping(raw, "transformation")
        _reject_unknown(payload, _TRANSFORM_KEYS, "transformation")
        for key in ("kind", "path", "find", "replace"):
            if key not in payload:
                raise GuiProposalError(f"transformation.{key} is required")
            _reject_present_null(payload, key)
        expected = payload.get("expected_count", 1)
        if "expected_count" in payload:
            _reject_present_null(payload, "expected_count")
        return cls(
            kind=_as_kind(payload["kind"], wire=True),
            path=payload["path"],
            find=payload["find"],
            replace=payload["replace"],
            expected_count=expected,
            interface=payload.get(
                "interface", DETERMINISTIC_GUI_TRANSFORMATION_INTERFACE
            ),
            schema_version=payload.get(
                "schema_version", DETERMINISTIC_GUI_TRANSFORMATION_SCHEMA
            ),
        )


@dataclass(frozen=True, slots=True)
class ContextSourceView:
    path: str
    content: str
    component_id: str = ""
    editable: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _repo_path(self.path, "path"))
        object.__setattr__(self, "content", _raw_content(self.content, "content"))
        object.__setattr__(
            self,
            "component_id",
            _identifier(self.component_id, "component_id")
            if self.component_id
            else "",
        )
        object.__setattr__(self, "editable", _bool(self.editable, "editable"))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ContextSourceView":
        payload = _require_mapping(raw, "raw_source")
        _reject_unknown(payload, _SOURCE_KEYS, "raw_source")
        for key in ("path", "content"):
            if key not in payload:
                raise GuiProposalError(f"raw_source.{key} is required")
            _reject_present_null(payload, key)
        editable = True
        if "editable" in payload:
            _reject_present_null(payload, "editable")
            editable = _bool(payload["editable"], "editable")
        component_id = ""
        if "component_id" in payload:
            _reject_present_null(payload, "component_id")
            component_id = _identifier(payload["component_id"], "component_id")
        return cls(
            path=payload["path"],
            content=payload["content"],
            component_id=component_id,
            editable=editable,
        )


@dataclass(frozen=True, slots=True)
class ContextPackView:
    """Compact closed view of ``UiContextPack@1`` consumed by the proposer."""

    pack_id: str
    application_id: str
    screen_id: str
    objective: str
    raw_sources: tuple[ContextSourceView, ...]
    analysis_classification: str = "exact"
    verification_status: str = "unverified"
    escalation_conditions: tuple[str, ...] = ()
    formal_invariant_failures: tuple[str, ...] = ()
    acceptance_criteria: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "pack_id", _identifier(self.pack_id, "pack_id"))
        object.__setattr__(
            self, "application_id", _identifier(self.application_id, "application_id")
        )
        object.__setattr__(self, "screen_id", _identifier(self.screen_id, "screen_id"))
        object.__setattr__(self, "objective", _text(self.objective, "objective"))
        if not self.raw_sources:
            raise GuiProposalError("context_pack.raw_sources must not be empty")
        object.__setattr__(
            self,
            "analysis_classification",
            _text(self.analysis_classification, "analysis_classification"),
        )
        object.__setattr__(
            self,
            "verification_status",
            _text(self.verification_status, "verification_status"),
        )
        paths = [source.path for source in self.raw_sources]
        if len(paths) != len(set(paths)):
            raise GuiProposalError("context_pack.raw_sources paths must be unique")

    def source_map(self) -> dict[str, ContextSourceView]:
        return {source.path: source for source in self.raw_sources}

    @classmethod
    def from_any(cls, value: Any) -> "ContextPackView":
        if type(value) is cls:
            return value
        if type(value) is dict:
            return cls.from_mapping(value)
        if hasattr(value, "pack_id") and hasattr(value, "raw_sources"):
            sources = []
            for item in getattr(value, "raw_sources"):
                if type(item) is ContextSourceView:
                    sources.append(item)
                elif type(item) is dict:
                    sources.append(ContextSourceView.from_mapping(item))
                else:
                    sources.append(
                        ContextSourceView(
                            path=getattr(item, "path"),
                            content=getattr(item, "content"),
                            component_id=getattr(item, "component_id", ""),
                            editable=getattr(item, "editable", True),
                        )
                    )
            failures = getattr(value, "formal_invariant_failures", ())
            failure_ids: list[str] = []
            for item in failures or ():
                if type(item) is str:
                    failure_ids.append(item)
                elif type(item) is dict:
                    ident = item.get("failure_id") or item.get("violation_id")
                    if type(ident) is str:
                        failure_ids.append(ident)
                else:
                    ident = getattr(item, "failure_id", None) or getattr(
                        item, "violation_id", None
                    )
                    if type(ident) is str:
                        failure_ids.append(ident)
            return cls(
                pack_id=getattr(value, "pack_id"),
                application_id=getattr(value, "application_id"),
                screen_id=getattr(value, "screen_id"),
                objective=getattr(value, "objective"),
                raw_sources=tuple(sources),
                analysis_classification=getattr(
                    value, "analysis_classification", "exact"
                )
                if type(getattr(value, "analysis_classification", "exact")) is str
                else getattr(value, "analysis_classification").value,
                verification_status=getattr(value, "verification_status", "unverified")
                if type(getattr(value, "verification_status", "unverified")) is str
                else getattr(value, "verification_status").value,
                escalation_conditions=tuple(
                    str(item) for item in getattr(value, "escalation_conditions", ())
                ),
                formal_invariant_failures=tuple(failure_ids),
                acceptance_criteria=tuple(
                    str(item) for item in getattr(value, "acceptance_criteria", ())
                ),
            )
        raise GuiProposalError(
            "context_pack must be a JSON object",
            reason_code=ProposalReasonCode.INVALID_COLLECTION_TYPE.value,
            details={"value_type": type(value).__name__},
        )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ContextPackView":
        payload = _require_mapping(raw, "context_pack")
        _reject_unknown(payload, _PACK_KEYS, "context_pack")
        for key in (
            "pack_id",
            "application_id",
            "screen_id",
            "objective",
            "raw_sources",
        ):
            if key not in payload:
                raise GuiProposalError(f"context_pack.{key} is required")
            _reject_present_null(payload, key)
        sources = tuple(
            ContextSourceView.from_mapping(item)
            for item in _require_json_array(payload["raw_sources"], "raw_sources")
        )
        conditions = (
            _unique_strings(payload["escalation_conditions"], "escalation_conditions")
            if "escalation_conditions" in payload
            else ()
        )
        failures_raw = payload.get("formal_invariant_failures", [])
        if "formal_invariant_failures" in payload:
            _reject_present_null(payload, "formal_invariant_failures")
        failure_ids: list[str] = []
        for index, item in enumerate(
            _require_json_array(failures_raw, "formal_invariant_failures")
        ):
            if type(item) is str:
                failure_ids.append(_identifier(item, f"formal_invariant_failures[{index}]"))
            elif type(item) is dict:
                ident = item.get("failure_id") or item.get("violation_id")
                failure_ids.append(
                    _identifier(ident, f"formal_invariant_failures[{index}]")
                )
            else:
                raise GuiProposalError(
                    "formal_invariant_failures entries must be strings or objects",
                    reason_code=ProposalReasonCode.INVALID_COLLECTION_TYPE.value,
                )
        criteria = (
            _unique_texts(payload["acceptance_criteria"], "acceptance_criteria")
            if "acceptance_criteria" in payload
            else ()
        )
        return cls(
            pack_id=payload["pack_id"],
            application_id=payload["application_id"],
            screen_id=payload["screen_id"],
            objective=payload["objective"],
            raw_sources=sources,
            analysis_classification=payload.get("analysis_classification", "exact"),
            verification_status=payload.get("verification_status", "unverified"),
            escalation_conditions=conditions,
            formal_invariant_failures=tuple(failure_ids),
            acceptance_criteria=criteria,
        )


@dataclass(frozen=True, slots=True)
class HumanGuiReviewRequest:
    review_id: str
    escalation_kind: EscalationKind
    reason_codes: tuple[str, ...]
    summary: str
    context_pack_id: str
    declared_method: str
    declared_tier: str
    route_kind: ProposalRoute
    interface: str = HUMAN_GUI_REVIEW_REQUEST_INTERFACE
    schema_version: str = HUMAN_GUI_REVIEW_REQUEST_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "review_id", _identifier(self.review_id, "review_id"))
        if type(self.escalation_kind) is not EscalationKind:
            object.__setattr__(
                self, "escalation_kind", EscalationKind(str(self.escalation_kind))
            )
        object.__setattr__(self, "summary", _text(self.summary, "summary"))
        object.__setattr__(
            self,
            "context_pack_id",
            _identifier(self.context_pack_id, "context_pack_id"),
        )
        object.__setattr__(
            self, "declared_method", _text(self.declared_method, "declared_method")
        )
        object.__setattr__(
            self, "declared_tier", _text(self.declared_tier, "declared_tier")
        )
        if type(self.route_kind) is not ProposalRoute:
            object.__setattr__(self, "route_kind", ProposalRoute(str(self.route_kind)))
        codes = tuple(sorted({_text(code, "reason_code") for code in self.reason_codes}))
        if not codes:
            raise GuiProposalError("review reason_codes must not be empty")
        object.__setattr__(self, "reason_codes", codes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_pack_id": self.context_pack_id,
            "declared_method": self.declared_method,
            "declared_tier": self.declared_tier,
            "escalation_kind": self.escalation_kind.value,
            "interface": self.interface,
            "reason_codes": list(self.reason_codes),
            "review_id": self.review_id,
            "route_kind": self.route_kind.value,
            "schema_version": self.schema_version,
            "summary": self.summary,
        }


@dataclass(frozen=True, slots=True)
class GuiProposalResult:
    disposition: ProposalDisposition
    route_kind: ProposalRoute
    declared_method: str
    declared_tier: str
    reason_codes: tuple[str, ...]
    proposal: Mapping[str, Any] | None = None
    patch_text: str = ""
    review_request: HumanGuiReviewRequest | None = None
    vendor: str = ""
    interface: str = GUI_PATCH_PROPOSER_INTERFACE
    schema_version: str = GUI_PROPOSAL_RESULT_SCHEMA

    def __post_init__(self) -> None:
        if type(self.disposition) is not ProposalDisposition:
            object.__setattr__(
                self, "disposition", ProposalDisposition(str(self.disposition))
            )
        if type(self.route_kind) is not ProposalRoute:
            object.__setattr__(self, "route_kind", ProposalRoute(str(self.route_kind)))
        object.__setattr__(
            self, "declared_method", _text(self.declared_method, "declared_method")
        )
        object.__setattr__(
            self, "declared_tier", _text(self.declared_tier, "declared_tier")
        )
        object.__setattr__(self, "vendor", "")
        object.__setattr__(
            self,
            "reason_codes",
            tuple(sorted({_text(code, "reason_code") for code in self.reason_codes})),
        )
        if self.proposal is not None:
            object.__setattr__(self, "proposal", MappingProxyType(dict(self.proposal)))
        if self.disposition is ProposalDisposition.PROPOSE:
            if self.proposal is None:
                raise GuiProposalError("propose disposition requires a proposal")
            if self.review_request is not None:
                raise GuiProposalError("propose disposition cannot carry a review request")
        else:
            if self.patch_text:
                raise GuiProposalError(
                    "escalate/reject cannot carry a fabricated patch",
                    reason_code=ProposalReasonCode.SCOPE_BROADENED.value,
                )
            if self.proposal is not None:
                raise GuiProposalError(
                    "escalate/reject cannot carry a fabricated proposal"
                )

    @property
    def proposed(self) -> bool:
        return self.disposition is ProposalDisposition.PROPOSE

    @property
    def escalated(self) -> bool:
        return self.disposition is ProposalDisposition.ESCALATE

    def to_dict(self) -> dict[str, Any]:
        return {
            "declared_method": self.declared_method,
            "declared_tier": self.declared_tier,
            "disposition": self.disposition.value,
            "interface": self.interface,
            "patch_text": self.patch_text,
            "proposal": None if self.proposal is None else dict(self.proposal),
            "reason_codes": list(self.reason_codes),
            "review_request": None
            if self.review_request is None
            else self.review_request.to_dict(),
            "route_kind": self.route_kind.value,
            "schema_version": self.schema_version,
            "vendor": "",
        }


# ---------------------------------------------------------------------------
# Proposer
# ---------------------------------------------------------------------------


def _unified_file_diff(path: str, before: str, after: str) -> str:
    return "".join(
        unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            lineterm="\n",
        )
    )


def _derive_method(route: ProposalRoute, kinds: Sequence[TransformationKind]) -> str:
    if route is ProposalRoute.HUMAN_REVIEW:
        return "human_review"
    if route in MODEL_ROUTES:
        return "injected_provider"
    if not kinds:
        return "exact_label_substitution"
    if len(set(kinds)) == 1:
        return KIND_DEFAULT_METHOD[kinds[0]]
    return "deterministic_composite"


class GuiPatchProposer:
    """``GuiPatchProposer@1`` — provider-neutral, fail-closed proposal gate."""

    interface: str = GUI_PATCH_PROPOSER_INTERFACE
    schema_version: str = GUI_PATCH_PROPOSER_SCHEMA

    def __init__(
        self,
        provider: GuiProposalProvider | Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
        *,
        max_prior_failures: int = DEFAULT_MAX_PRIOR_FAILURES,
    ) -> None:
        if provider is not None and not callable(getattr(provider, "propose", provider)):
            raise GuiProposalError("provider must expose propose() or be callable")
        self._provider = provider
        self._max_prior_failures = _positive_int(max_prior_failures, "max_prior_failures")

    def propose(self, request: Mapping[str, Any] | Any) -> GuiProposalResult:
        payload = self._parse_request(request)
        method, tier = self._method_and_tier(payload)
        flags = self._escalation_flags(payload)
        if flags:
            return self._escalate(payload, method, tier, flags)
        route: ProposalRoute = payload["route_kind"]
        if route is ProposalRoute.DETERMINISTIC_TRANSFORM:
            return self._deterministic(payload, method, tier)
        if route in MODEL_ROUTES:
            return self._delegate(payload, method, tier)
        return self._escalate(
            payload,
            method,
            tier,
            (EscalationKind.HUMAN_ROUTE,),
        )

    def _parse_request(self, request: Any) -> dict[str, Any]:
        payload = _require_mapping(request, "request")
        _reject_unknown(payload, _REQUEST_KEYS, "request")
        for key in (
            "request_id",
            "route_kind",
            "context_pack",
            "intended_file_paths",
            "intended_component_ids",
            "acceptance_criteria",
            "objective",
            "application_id",
            "screen_id",
        ):
            if key not in payload:
                raise GuiProposalError(f"request.{key} is required")
            _reject_present_null(payload, key)
        pack = ContextPackView.from_any(payload["context_pack"])
        route = _as_route(payload["route_kind"], wire=True)
        files = _unique_paths(
            payload["intended_file_paths"], "intended_file_paths", required=True
        )
        source_paths = {source.path for source in pack.raw_sources}
        for path in files:
            if path not in source_paths:
                raise GuiProposalError(
                    "intended_file_paths must be present in context_pack.raw_sources",
                    reason_code=ProposalReasonCode.MISSING_SOURCE.value,
                    details={"path": path},
                )
        transforms: tuple[DeterministicGuiTransformation, ...] = ()
        if "transformations" in payload:
            _reject_present_null(payload, "transformations")
            transforms = tuple(
                DeterministicGuiTransformation.from_mapping(item)
                if type(item) is dict
                else item
                if type(item) is DeterministicGuiTransformation
                else (_ for _ in ()).throw(
                    GuiProposalError(
                        "transformations entries must be JSON objects",
                        reason_code=ProposalReasonCode.INVALID_COLLECTION_TYPE.value,
                    )
                )
                for item in _require_json_array(
                    payload["transformations"], "transformations"
                )
            )
            for item in transforms:
                if item.path not in files:
                    raise GuiProposalError(
                        "transformation.path is not a declared intended file",
                        reason_code=ProposalReasonCode.SCOPE_BROADENED.value,
                        details={"path": item.path},
                    )
        prior = 0
        if "prior_failure_count" in payload:
            _reject_present_null(payload, "prior_failure_count")
            prior = _nonneg_int(payload["prior_failure_count"], "prior_failure_count")
        conditions = (
            _unique_strings(payload["escalation_conditions"], "escalation_conditions")
            if "escalation_conditions" in payload
            else pack.escalation_conditions
        )
        return {
            "request_id": _identifier(payload["request_id"], "request_id"),
            "route_kind": route,
            "declared_method": _reject_vendor(
                _optional_text(payload, "declared_method"), "declared_method"
            ),
            "declared_tier": _reject_vendor(
                _optional_text(payload, "declared_tier"), "declared_tier"
            ),
            "context_pack": pack,
            "transformations": transforms,
            "intended_file_paths": files,
            "intended_component_ids": _unique_strings(
                payload["intended_component_ids"],
                "intended_component_ids",
                required=True,
            ),
            "acceptance_criteria": _unique_texts(
                payload["acceptance_criteria"],
                "acceptance_criteria",
                required=True,
            ),
            "expected_test_ids": _unique_strings(
                payload["expected_test_ids"], "expected_test_ids"
            )
            if "expected_test_ids" in payload
            else (),
            "expected_screenshot_ids": _unique_strings(
                payload["expected_screenshot_ids"], "expected_screenshot_ids"
            )
            if "expected_screenshot_ids" in payload
            else (),
            "state_effect_ids": _unique_strings(
                payload["state_effect_ids"], "state_effect_ids"
            )
            if "state_effect_ids" in payload
            else (),
            "objective": _text(payload["objective"], "objective"),
            "application_id": _identifier(payload["application_id"], "application_id"),
            "screen_id": _identifier(payload["screen_id"], "screen_id"),
            "analysis_classification": _text(
                payload.get("analysis_classification", pack.analysis_classification),
                "analysis_classification",
            ),
            "verification_status": _text(
                payload.get("verification_status", pack.verification_status),
                "verification_status",
            ),
            "prior_failure_count": prior,
            "escalation_conditions": conditions,
            "policy_bound": _optional_bool(payload, "policy_bound", False),
            "security_sensitive": _optional_bool(payload, "security_sensitive", False),
            "opaque": _optional_bool(payload, "opaque", False),
            "ambiguous": _optional_bool(payload, "ambiguous", False),
            "constraint_conflict": _optional_bool(payload, "constraint_conflict", False),
            "visual_effect_summary": _optional_text(payload, "visual_effect_summary"),
        }

    def _method_and_tier(self, payload: Mapping[str, Any]) -> tuple[str, str]:
        route: ProposalRoute = payload["route_kind"]
        kinds = tuple(item.kind for item in payload["transformations"])
        method = payload["declared_method"] or _derive_method(route, kinds)
        _reject_vendor(method, "declared_method")
        if not _IDENTIFIER_RE.fullmatch(method):
            raise GuiProposalError("declared_method is not a stable identifier")
        expected_tier = ROUTE_DEFAULT_TIER[route]
        tier = payload["declared_tier"] or expected_tier
        _reject_vendor(tier, "declared_tier")
        if tier not in ALLOWED_TIERS:
            raise GuiProposalError(
                "declared_tier is not a closed provider-neutral tier",
                details={"declared_tier": tier},
            )
        if tier != expected_tier:
            raise GuiProposalError(
                "declared_tier must match the caller-selected route",
                details={"declared_tier": tier, "route_kind": route.value},
            )
        return method, tier

    def _escalation_flags(
        self, payload: Mapping[str, Any]
    ) -> tuple[EscalationKind, ...]:
        flags: list[EscalationKind] = []
        pack: ContextPackView = payload["context_pack"]
        classification = payload["analysis_classification"]
        status = payload["verification_status"]
        conditions = {item.lower() for item in payload["escalation_conditions"]}
        if (
            payload["opaque"]
            or classification == "opaque"
            or status in _OPAQUE_STATUSES
            or "opaque" in conditions
        ):
            flags.append(EscalationKind.OPAQUE)
        if payload["ambiguous"] or "ambiguous" in conditions:
            flags.append(EscalationKind.AMBIGUOUS)
        kinds = {item.kind for item in payload["transformations"]}
        policy_exempt = bool(kinds) and kinds <= POLICY_EXEMPT_KINDS
        if (
            payload["policy_bound"] or "policy_bound" in conditions or "policy" in conditions
        ) and not policy_exempt:
            flags.append(EscalationKind.POLICY_BOUND)
        if payload["security_sensitive"] or "security" in conditions:
            flags.append(EscalationKind.SECURITY)
        if (
            payload["prior_failure_count"] >= self._max_prior_failures
            or "repeated_failure" in conditions
        ):
            flags.append(EscalationKind.REPEATED_FAILURE)
        if (
            payload["constraint_conflict"]
            or pack.formal_invariant_failures
            or "constraint_conflict" in conditions
        ):
            flags.append(EscalationKind.CONSTRAINT_CONFLICT)
        if payload["route_kind"] is ProposalRoute.HUMAN_REVIEW:
            flags.append(EscalationKind.HUMAN_ROUTE)
        # Preserve first-seen order while de-duplicating.
        seen: set[EscalationKind] = set()
        ordered: list[EscalationKind] = []
        for flag in flags:
            if flag not in seen:
                seen.add(flag)
                ordered.append(flag)
        return tuple(ordered)

    def _escalate(
        self,
        payload: Mapping[str, Any],
        method: str,
        tier: str,
        flags: Sequence[EscalationKind],
    ) -> GuiProposalResult:
        kind = flags[0]
        codes = [ProposalReasonCode.ESCALATED.value, ProposalReasonCode.HUMAN_REVIEW_REQUIRED.value]
        summaries = {
            EscalationKind.OPAQUE: (ProposalReasonCode.OPAQUE_CONTEXT.value, "opaque or stale context"),
            EscalationKind.AMBIGUOUS: (
                ProposalReasonCode.AMBIGUOUS_TRANSFORM.value,
                "ambiguous or non-unique transformation",
            ),
            EscalationKind.POLICY_BOUND: (
                ProposalReasonCode.POLICY_BOUND.value,
                "policy-bound request requires human review",
            ),
            EscalationKind.REPEATED_FAILURE: (
                ProposalReasonCode.REPEATED_FAILURE.value,
                "repeated proposal failures require escalation",
            ),
            EscalationKind.SECURITY: (
                ProposalReasonCode.SECURITY_REGRESSION.value,
                "security-sensitive request cannot auto-propose",
            ),
            EscalationKind.CONSTRAINT_CONFLICT: (
                ProposalReasonCode.CONSTRAINT_CONFLICT.value,
                "constraint conflict requires human review",
            ),
            EscalationKind.PROVIDER_ABSENT: (
                ProposalReasonCode.PROVIDER_ABSENT.value,
                "provider absence cannot fabricate a patch",
            ),
            EscalationKind.PROVIDER_EXCEPTION: (
                ProposalReasonCode.PROVIDER_EXCEPTION.value,
                "provider exception cannot fabricate a patch",
            ),
            EscalationKind.HUMAN_ROUTE: (
                ProposalReasonCode.HUMAN_REVIEW_REQUIRED.value,
                "caller selected the human-review route",
            ),
        }
        for flag in flags:
            code, _ = summaries[flag]
            codes.append(code)
        _, summary = summaries[kind]
        pack: ContextPackView = payload["context_pack"]
        review = HumanGuiReviewRequest(
            review_id=_digest_id(
                "review",
                {
                    "flags": [flag.value for flag in flags],
                    "pack_id": pack.pack_id,
                    "request_id": payload["request_id"],
                },
            ),
            escalation_kind=kind,
            reason_codes=tuple(codes),
            summary=summary,
            context_pack_id=pack.pack_id,
            declared_method=method,
            declared_tier=tier,
            route_kind=payload["route_kind"],
        )
        return GuiProposalResult(
            disposition=ProposalDisposition.ESCALATE,
            route_kind=payload["route_kind"],
            declared_method=method,
            declared_tier=tier,
            reason_codes=tuple(codes),
            review_request=review,
        )

    def _deterministic(
        self,
        payload: Mapping[str, Any],
        method: str,
        tier: str,
    ) -> GuiProposalResult:
        transforms: tuple[DeterministicGuiTransformation, ...] = payload["transformations"]
        if not transforms:
            raise GuiProposalError(
                "deterministic_transform requires at least one transformation"
            )
        pack: ContextPackView = payload["context_pack"]
        sources = pack.source_map()
        updated: dict[str, str] = {
            path: sources[path].content for path in payload["intended_file_paths"]
        }
        for item in transforms:
            current = updated[item.path]
            count = current.count(item.find)
            if count != item.expected_count:
                return self._escalate(
                    payload,
                    method,
                    tier,
                    (EscalationKind.AMBIGUOUS,),
                )
            if not sources[item.path].editable:
                return self._escalate(
                    payload,
                    method,
                    tier,
                    (EscalationKind.OPAQUE,),
                )
            updated[item.path] = current.replace(item.find, item.replace)
        diffs = [
            _unified_file_diff(path, sources[path].content, updated[path])
            for path in payload["intended_file_paths"]
            if sources[path].content != updated[path]
        ]
        if not diffs:
            return self._escalate(
                payload, method, tier, (EscalationKind.AMBIGUOUS,)
            )
        patch_text = "".join(diffs)
        proposal = self._proposal_dict(payload, method, patch_text)
        return GuiProposalResult(
            disposition=ProposalDisposition.PROPOSE,
            route_kind=ProposalRoute.DETERMINISTIC_TRANSFORM,
            declared_method=method,
            declared_tier=tier,
            reason_codes=(
                ProposalReasonCode.PROPOSED.value,
                ProposalReasonCode.DETERMINISTIC_TRANSFORM.value,
            ),
            proposal=proposal,
            patch_text=patch_text,
        )

    def _delegate(
        self,
        payload: Mapping[str, Any],
        method: str,
        tier: str,
    ) -> GuiProposalResult:
        provider = self._provider
        if provider is None:
            return self._escalate(
                payload, method, tier, (EscalationKind.PROVIDER_ABSENT,)
            )
        call = provider.propose if hasattr(provider, "propose") else provider
        sanitized = {
            "acceptance_criteria": list(payload["acceptance_criteria"]),
            "application_id": payload["application_id"],
            "context_pack_id": payload["context_pack"].pack_id,
            "declared_method": method,
            "declared_tier": tier,
            "intended_component_ids": list(payload["intended_component_ids"]),
            "intended_file_paths": list(payload["intended_file_paths"]),
            "objective": payload["objective"],
            "request_id": payload["request_id"],
            "route_kind": payload["route_kind"].value,
            "screen_id": payload["screen_id"],
        }
        try:
            raw = call(sanitized)
        except Exception:
            return self._escalate(
                payload, method, tier, (EscalationKind.PROVIDER_EXCEPTION,)
            )
        result = _require_mapping(raw, "provider_result")
        _reject_unknown(result, _PROVIDER_RESULT_KEYS, "provider_result")
        if "proposal" not in result or result["proposal"] is None:
            return self._escalate(
                payload, method, tier, (EscalationKind.PROVIDER_ABSENT,)
            )
        proposal_raw = _require_mapping(result["proposal"], "provider_result.proposal")
        declared = set(payload["intended_file_paths"])
        provider_paths = proposal_raw.get("intended_file_paths", [])
        paths = _unique_paths(provider_paths, "provider_result.proposal.intended_file_paths")
        extra = [path for path in paths if path not in declared]
        if extra:
            raise GuiProposalError(
                "provider cannot broaden intended_file_paths",
                reason_code=ProposalReasonCode.SCOPE_BROADENED.value,
                details={"extra_paths": extra},
            )
        patch_text = ""
        if "patch_text" in result:
            _reject_present_null(result, "patch_text")
            patch_text = _raw_content(result["patch_text"], "patch_text")
        proposal = self._proposal_dict(payload, method, patch_text)
        return GuiProposalResult(
            disposition=ProposalDisposition.PROPOSE,
            route_kind=payload["route_kind"],
            declared_method=method,
            declared_tier=tier,
            reason_codes=(ProposalReasonCode.PROPOSED.value,),
            proposal=proposal,
            patch_text=patch_text,
        )

    def _proposal_dict(
        self,
        payload: Mapping[str, Any],
        method: str,
        patch_text: str,
    ) -> dict[str, Any]:
        pack: ContextPackView = payload["context_pack"]
        summary = payload["visual_effect_summary"]
        if not summary:
            kinds = [item.kind.value for item in payload["transformations"]]
            summary = (
                "Deterministic substitutions: " + ", ".join(kinds)
                if kinds
                else "Provider-supplied bounded proposal"
            )
        body = {
            "acceptance_criteria": list(payload["acceptance_criteria"]),
            "analysis_classification": payload["analysis_classification"],
            "application_id": payload["application_id"],
            "context_pack_id": pack.pack_id,
            "decision": "pending",
            "expected_screenshot_ids": list(payload["expected_screenshot_ids"]),
            "expected_test_ids": list(payload["expected_test_ids"]),
            "intended_component_ids": list(payload["intended_component_ids"]),
            "intended_file_paths": list(payload["intended_file_paths"]),
            "interface": GUI_IMPROVEMENT_PROPOSAL_INTERFACE,
            "objective": payload["objective"],
            "route_kind": payload["route_kind"].value,
            "schema_version": GUI_IMPROVEMENT_PROPOSAL_SCHEMA,
            "screen_id": payload["screen_id"],
            "state_effect_ids": list(payload["state_effect_ids"]),
            "verification_status": "unverified",
            "visual_effect_summary": summary,
        }
        body["proposal_id"] = _digest_id(
            "proposal", {**body, "method": method, "patch_digest": hashlib.sha256(patch_text.encode("utf-8")).hexdigest()}
        )
        return body


def default_gui_patch_proposer(
    provider: GuiProposalProvider | Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> GuiPatchProposer:
    return GuiPatchProposer(provider=provider)
