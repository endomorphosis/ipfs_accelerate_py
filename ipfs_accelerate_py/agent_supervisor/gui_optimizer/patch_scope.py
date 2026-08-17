"""Fail-closed patch-scope gate for VerifiedGuiOptimizer (VGO-043).

Interfaces owned by this module:

* ``GuiPatchScopeGate@1`` — admit a proposal only when the observed patch
  stays inside declared files/components/state/visual/test/screenshot
  bounds, configured file/line limits, and repository safety fencing
* ``GuiImprovementProposal@1`` — closed view of the VGO-001 proposal
  contract consumed before execution
* ``GuiPatchScopeDecision@1`` — typed allow / reject / review outcome
  with stable reason codes

This module never applies a patch, never mutates a worktree, and never
treats a scope declaration as host authority.  Callers inject the
proposal, the observed diff, and an explicit invalidation record.  The
gate reuses ``GuiPatchAuthority@1`` for root/segment fencing and
``AuthorityEvidence`` for action-contract bindings.

Fail-closed invariants:

* undeclared, unresolved, generated, or excessive paths reject;
* unrelated applications reject;
* backend authorization and credential mutations require human review;
* disabled security and arbitrary HTML execution reject;
* deleted tests and unverified action-binding edits reject or require
  review;
* computed declaration/kind/limit facts override caller claims;
* a scope declaration alone never verifies a binding.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Final

from .authority import (
    ALWAYS_HUMAN_REVIEW_KINDS,
    AuthorityDecision,
    AuthorityEvidence,
    AuthorityEvidenceKind,
    AuthorityReasonCode,
    AuthorityVerdict,
    ForbiddenChangeKind,
    GuiAuthorityError,
    GuiPatchAuthority,
    HOST_AUTHORIZING_EVIDENCE_KINDS,
    SENSITIVE_CHANGE_KINDS,
    _normalize_repo_path,
    path_has_forbidden_segment,
)

# ---------------------------------------------------------------------------
# Interface / schema identity
# ---------------------------------------------------------------------------

GUI_PATCH_SCOPE_GATE_INTERFACE: Final[str] = "GuiPatchScopeGate@1"
GUI_PATCH_SCOPE_DECISION_INTERFACE: Final[str] = "GuiPatchScopeDecision@1"
GUI_IMPROVEMENT_PROPOSAL_INTERFACE: Final[str] = "GuiImprovementProposal@1"

GUI_PATCH_SCOPE_GATE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/gui-optimizer/patch-scope-gate@1"
)
GUI_PATCH_SCOPE_DECISION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/gui-optimizer/patch-scope-decision@1"
)
GUI_IMPROVEMENT_PROPOSAL_SCHEMA: Final[str] = "gui-improvement-proposal/v1"
GUI_INVALIDATION_PLAN_INTERFACE: Final[str] = "UiInvalidationPlan@1"
GUI_INVALIDATION_PLAN_SCHEMA: Final[str] = "ui-invalidation-plan/v1"

# Conservative defaults.  Construction may only tighten these; values above
# the absolute caps are rejected so callers cannot weaken the fence.
DEFAULT_MAX_FILES: Final[int] = 8
DEFAULT_MAX_CHANGED_LINES: Final[int] = 250
DEFAULT_MAX_HUNKS: Final[int] = 24
DEFAULT_MAX_NEW_FILES: Final[int] = 3
DEFAULT_MAX_DELETED_FILES: Final[int] = 2
DEFAULT_MAX_PATH_CHARS: Final[int] = 512

ABSOLUTE_MAX_FILES: Final[int] = 16
ABSOLUTE_MAX_CHANGED_LINES: Final[int] = 500
ABSOLUTE_MAX_HUNKS: Final[int] = 48
ABSOLUTE_MAX_NEW_FILES: Final[int] = 6
ABSOLUTE_MAX_DELETED_FILES: Final[int] = 4
ABSOLUTE_MAX_PATH_CHARS: Final[int] = 512

_PROPOSAL_KEYS: Final[frozenset[str]] = frozenset(
    {
        "acceptance_criteria",
        "analysis_classification",
        "application_id",
        "context_pack_id",
        "decision",
        "expected_screenshot_ids",
        "expected_test_ids",
        "intended_component_ids",
        "intended_file_paths",
        "interface",
        "objective",
        "proposal_id",
        "route_kind",
        "schema_version",
        "screen_id",
        "state_effect_ids",
        "verification_status",
        "visual_effect_summary",
    }
)
_HUNK_KEYS: Final[frozenset[str]] = frozenset(
    {
        "added_lines",
        "change_kinds",
        "content_markers",
        "deleted_lines",
        "diff_text",
        "end_line",
        "old_path",
        "operation",
        "path",
        "start_line",
    }
)
_OBSERVATION_KEYS: Final[frozenset[str]] = frozenset(
    {
        "action_argument_digest",
        "action_binding_ids",
        "action_contract_evidence",
        "application_ids",
        "diff_text",
        "hunks",
        "touched_component_ids",
        "touched_screenshot_ids",
        "touched_state_effect_ids",
        "touched_test_ids",
        "unresolved_paths",
        "visual_effect_observed",
    }
)
_INVALIDATION_KEYS: Final[frozenset[str]] = frozenset(
    {
        "affected_check_ids",
        "affected_component_ids",
        "affected_scenario_ids",
        "change_set_id",
        "confidence",
        "fallback_explanation",
        "fallback_triggered",
        "interface",
        "plan_id",
        "reasons",
        "schema_version",
    }
)
_EVALUATE_REQUEST_KEYS: Final[frozenset[str]] = frozenset(
    {"invalidation", "observation", "proposal"}
)
_KNOWN_INVALIDATION_REASONS: Final[frozenset[str]] = frozenset(
    {
        "action_changed",
        "component_changed",
        "dependency_changed",
        "extractor_changed",
        "fallback_expansion",
        "localization_changed",
        "missing_edge",
        "opaque_edge",
        "props_changed",
        "schema_changed",
        "stale_edge",
        "state_changed",
        "style_changed",
    }
)
_TEST_PATH_MARKERS: Final[tuple[str, ...]] = (
    "/test/",
    "/tests/",
    "/__tests__/",
    ".test.",
    ".spec.",
    "_test.",
    "_spec.",
    "/test_",
)
_SCREENSHOT_PATH_MARKERS: Final[tuple[str, ...]] = (
    "/screenshot",
    "/screenshots/",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
)
_HTML_MARKERS: Final[tuple[str, ...]] = (
    "innerHTML",
    "dangerouslySetInnerHTML",
    "document.write",
    "insertAdjacentHTML",
    "eval(",
    "new Function",
    "<script",
    "javascript:",
)
_SECURITY_DISABLE_MARKERS: Final[tuple[str, ...]] = (
    "dangerouslydisable",
    "disablesecurity",
    "skipcsp",
    "csp_off",
    "sandbox=false",
    "nodeintegration",
    "allowrunninginsecure",
    "requires_confirmation=false",
    "requires_confirmation: false",
    "noconfirm",
    "insecure=true",
)
_CREDENTIAL_MARKERS: Final[tuple[str, ...]] = (
    "password=",
    "password:",
    "api_key",
    "apikey",
    "authorization",
    "bearer ",
    "client_secret",
    "private_key",
    "secret=",
    "secret:",
)
_BACKEND_MARKERS: Final[tuple[str, ...]] = (
    "bypass_auth",
    "skip_authorization",
    "allow_all_roles",
    "disable_rbac",
    "authorize=true",
)
_TEST_DELETION_MARKERS: Final[tuple[str, ...]] = (
    "def test_",
    "it(",
    "test(",
    "describe(",
    "pytest",
)
_APPS_PREFIX: Final[str] = "swissknife/web/js/apps/"
_UNIFIED_HUNK_RE = re.compile(
    r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@"
)
_CANONICAL_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class GuiPatchScopeError(GuiAuthorityError):
    """Malformed patch-scope input.  Never grants execution."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "invalid_patch_scope_input",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message, reason_code=reason_code, details=details)


class PatchOperation(str, Enum):
    """Closed diff operations the gate understands."""

    CREATE = "create"
    MODIFY = "modify"
    DELETE = "delete"
    RENAME = "rename"


class PatchScopeReasonCode(str, Enum):
    """Stable reason codes for ``GuiPatchScopeDecision@1``."""

    ALLOWED = "allowed"
    UNDECLARED_FILE = "undeclared_file"
    UNDECLARED_PATH = AuthorityReasonCode.UNDECLARED_PATH.value
    UNDECLARED_COMPONENT = "undeclared_component"
    UNDECLARED_STATE_EFFECT = "undeclared_state_effect"
    UNDECLARED_TEST = "undeclared_test"
    UNDECLARED_SCREENSHOT = "undeclared_screenshot"
    UNDECLARED_VISUAL_EFFECT = "undeclared_visual_effect"
    UNRELATED_APPLICATION = ForbiddenChangeKind.UNRELATED_APPLICATION.value
    BACKEND_AUTHORIZATION = ForbiddenChangeKind.BACKEND_AUTHORIZATION.value
    CREDENTIALS = ForbiddenChangeKind.CREDENTIALS.value
    DISABLED_SECURITY_CHECK = ForbiddenChangeKind.DISABLED_SECURITY_CHECK.value
    DELETED_TEST = ForbiddenChangeKind.DELETED_TEST.value
    UNVERIFIED_ACTION_BINDING = ForbiddenChangeKind.UNVERIFIED_ACTION_BINDING.value
    ARBITRARY_HTML_EXECUTION = ForbiddenChangeKind.ARBITRARY_HTML_EXECUTION.value
    FILE_LIMIT_EXCEEDED = "file_limit_exceeded"
    LINE_LIMIT_EXCEEDED = "line_limit_exceeded"
    HUNK_LIMIT_EXCEEDED = "hunk_limit_exceeded"
    GENERATED_PATH = "generated_path"
    UNRESOLVED_PATH = "unresolved_path"
    EXCESSIVE_PATH = "excessive_path"
    DIFF_SEMANTICS_INVALID = "diff_semantics_invalid"
    MISSING_INVALIDATION_RECORD = "missing_invalidation_record"
    MISSING_ACTION_CONTRACT_EVIDENCE = "missing_action_contract_evidence"
    MISSING_PROPOSAL_DECLARATION = "missing_proposal_declaration"
    OUT_OF_SCOPE_PATH = ForbiddenChangeKind.OUT_OF_SCOPE_PATH.value
    FORBIDDEN_MUTATION = "forbidden_mutation"
    PATH_OUTSIDE_ALLOWED_ROOTS = (
        AuthorityReasonCode.PATH_OUTSIDE_ALLOWED_ROOTS.value
    )
    PATH_ABSOLUTE_OR_TRAVERSAL = (
        AuthorityReasonCode.PATH_ABSOLUTE_OR_TRAVERSAL.value
    )
    PATH_FORBIDDEN_SEGMENT = AuthorityReasonCode.PATH_FORBIDDEN_SEGMENT.value
    SENSITIVE_CHANGE_REQUIRES_REVIEW = (
        AuthorityReasonCode.SENSITIVE_CHANGE_REQUIRES_REVIEW.value
    )
    SENSITIVE_CHANGE_REQUIRES_CONTRACT = (
        AuthorityReasonCode.SENSITIVE_CHANGE_REQUIRES_CONTRACT.value
    )
    SCOPE_DECLARATION_NOT_AUTHORITY = (
        AuthorityReasonCode.SCOPE_DECLARATION_NOT_AUTHORITY.value
    )
    INVALID_AUTHORITY_EVIDENCE = (
        AuthorityReasonCode.INVALID_AUTHORITY_EVIDENCE.value
    )
    EVIDENCE_BINDING_MISMATCH = (
        AuthorityReasonCode.EVIDENCE_BINDING_MISMATCH.value
    )
    INVALID_PATCH_SCOPE_INPUT = "invalid_patch_scope_input"
    UNKNOWN_FIELD = AuthorityReasonCode.UNKNOWN_FIELD.value
    INVALID_COLLECTION_TYPE = (
        AuthorityReasonCode.INVALID_COLLECTION_TYPE.value
    )
    INVALIDATION_COVERAGE_GAP = "invalidation_coverage_gap"


# Sensitive kinds that the scope gate will not auto-execute even inside
# declared files.  Backend/credential/deleted-test/unverified-binding escalate
# to review; the rest hard-reject.
_SCOPE_REJECT_KINDS: Final[frozenset[ForbiddenChangeKind]] = frozenset(
    {
        ForbiddenChangeKind.DISABLED_SECURITY_CHECK,
        ForbiddenChangeKind.ARBITRARY_HTML_EXECUTION,
        ForbiddenChangeKind.HOST_BOUNDARY_BYPASS,
        ForbiddenChangeKind.PRODUCTION_TOOL_ACCESS,
        ForbiddenChangeKind.UNRELATED_APPLICATION,
    }
)
_SCOPE_REVIEW_KINDS: Final[frozenset[ForbiddenChangeKind]] = frozenset(
    {
        ForbiddenChangeKind.BACKEND_AUTHORIZATION,
        ForbiddenChangeKind.CREDENTIALS,
        ForbiddenChangeKind.DELETED_TEST,
        ForbiddenChangeKind.UNVERIFIED_ACTION_BINDING,
        ForbiddenChangeKind.CONFIRMATION_WEAKENING,
        ForbiddenChangeKind.POLICY_WEAKENING,
    }
)


# ---------------------------------------------------------------------------
# Closed input helpers
# ---------------------------------------------------------------------------


def _exact_str(value: Any, name: str) -> str:
    if type(value) is not str:
        raise GuiPatchScopeError(
            f"{name} must be a string",
            reason_code=PatchScopeReasonCode.INVALID_PATCH_SCOPE_INPUT.value,
            details={"field": name, "value_type": type(value).__name__},
        )
    return value


def _text(value: Any, name: str, *, required: bool = True) -> str:
    text_value = _exact_str(value, name)
    if "\x00" in text_value:
        raise GuiPatchScopeError(
            f"{name} must not contain NUL",
            reason_code=PatchScopeReasonCode.INVALID_PATCH_SCOPE_INPUT.value,
            details={"field": name},
        )
    text = text_value.strip()
    if required and not text:
        raise GuiPatchScopeError(
            f"{name} must not be empty",
            reason_code=PatchScopeReasonCode.INVALID_PATCH_SCOPE_INPUT.value,
            details={"field": name},
        )
    return text


def _identifier(value: Any, name: str, *, required: bool = True) -> str:
    text_value = _exact_str(value, name)
    if "\x00" in text_value:
        raise GuiPatchScopeError(
            f"{name} must not contain NUL",
            reason_code=PatchScopeReasonCode.INVALID_PATCH_SCOPE_INPUT.value,
            details={"field": name},
        )
    if text_value == "":
        if required:
            raise GuiPatchScopeError(
                f"{name} must be a nonempty string identifier",
                reason_code=PatchScopeReasonCode.INVALID_PATCH_SCOPE_INPUT.value,
                details={"field": name},
            )
        return ""
    if text_value != text_value.strip():
        raise GuiPatchScopeError(
            f"{name} must be a canonical nonempty string identifier",
            reason_code=PatchScopeReasonCode.INVALID_PATCH_SCOPE_INPUT.value,
            details={"field": name},
        )
    return text_value


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise GuiPatchScopeError(
            f"{name} must be a boolean",
            reason_code=PatchScopeReasonCode.INVALID_PATCH_SCOPE_INPUT.value,
            details={"field": name, "value_type": type(value).__name__},
        )
    return value


def _nonneg_int(value: Any, name: str) -> int:
    if type(value) is not int or type(value) is bool:
        raise GuiPatchScopeError(
            f"{name} must be an integer",
            reason_code=PatchScopeReasonCode.INVALID_PATCH_SCOPE_INPUT.value,
            details={"field": name, "value_type": type(value).__name__},
        )
    if value < 0:
        raise GuiPatchScopeError(
            f"{name} must be a non-negative integer",
            reason_code=PatchScopeReasonCode.INVALID_PATCH_SCOPE_INPUT.value,
            details={"field": name, "value": value},
        )
    return value


def _positive_int(value: Any, name: str) -> int:
    number = _nonneg_int(value, name)
    if number < 1:
        raise GuiPatchScopeError(
            f"{name} must be a positive integer",
            reason_code=PatchScopeReasonCode.INVALID_PATCH_SCOPE_INPUT.value,
            details={"field": name, "value": value},
        )
    return number


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise GuiPatchScopeError(
            f"{name} must be a JSON object",
            reason_code=PatchScopeReasonCode.INVALID_COLLECTION_TYPE.value,
            details={"field": name, "value_type": type(value).__name__},
        )
    for key in value:
        if type(key) is not str:
            raise GuiPatchScopeError(
                f"{name} keys must be strings",
                reason_code=PatchScopeReasonCode.INVALID_COLLECTION_TYPE.value,
                details={"field": name, "key_type": type(key).__name__},
            )
    return value


def _require_json_array(value: Any, name: str) -> list[Any]:
    if type(value) is not list:
        raise GuiPatchScopeError(
            f"{name} must be a JSON array (list); "
            f"{type(value).__name__} is not a valid collection",
            reason_code=PatchScopeReasonCode.INVALID_COLLECTION_TYPE.value,
            details={"field": name, "value_type": type(value).__name__},
        )
    return value


def _require_python_sequence(value: Any, name: str) -> Sequence[Any]:
    if type(value) is list or type(value) is tuple:
        return value
    raise GuiPatchScopeError(
        f"{name} must be a JSON array/sequence",
        reason_code=PatchScopeReasonCode.INVALID_COLLECTION_TYPE.value,
        details={"field": name, "value_type": type(value).__name__},
    )


def _optional_json_array(payload: Mapping[str, Any], key: str) -> list[Any] | None:
    if key not in payload:
        return None
    value = payload[key]
    if value is None:
        raise GuiPatchScopeError(
            f"{key} must be a JSON array when present; null is not a collection",
            reason_code=PatchScopeReasonCode.INVALID_COLLECTION_TYPE.value,
            details={"field": key, "value_type": "NoneType"},
        )
    return _require_json_array(value, key)


def _reject_unknown(
    payload: Mapping[str, Any], allowed: frozenset[str], noun: str
) -> None:
    unknown = sorted(set(payload) - set(allowed))
    if unknown:
        raise GuiPatchScopeError(
            f"{noun} contains unknown fields: {unknown}",
            reason_code=PatchScopeReasonCode.UNKNOWN_FIELD.value,
            details={"noun": noun, "unknown_fields": unknown},
        )


def _reject_present_null(payload: Mapping[str, Any], key: str) -> None:
    if key in payload and payload[key] is None:
        raise GuiPatchScopeError(
            f"{key} must not be null when present",
            reason_code=PatchScopeReasonCode.INVALID_PATCH_SCOPE_INPUT.value,
            details={"field": key, "value_type": "NoneType"},
        )


def _optional_bool(payload: Mapping[str, Any], key: str, default: bool) -> bool:
    if key not in payload:
        return default
    _reject_present_null(payload, key)
    return _bool(payload[key], key)


def _optional_identifier(payload: Mapping[str, Any], key: str) -> str:
    if key not in payload:
        return ""
    _reject_present_null(payload, key)
    return _identifier(payload[key], key, required=True)


def _optional_text(payload: Mapping[str, Any], key: str) -> str:
    if key not in payload:
        return ""
    _reject_present_null(payload, key)
    return _text(payload[key], key, required=False)


def _freeze_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    mapping = _require_mapping(value, "details")
    return MappingProxyType(dict(mapping))


def _unique_strings(
    value: Any,
    name: str,
    *,
    wire: bool,
    required: bool = False,
    as_paths: bool = False,
) -> tuple[str, ...]:
    if value is None:
        raise GuiPatchScopeError(
            f"{name} must be a JSON array; null is not a collection",
            reason_code=PatchScopeReasonCode.INVALID_COLLECTION_TYPE.value,
            details={"field": name, "value_type": "NoneType"},
        )
    sequence = (
        _require_json_array(value, name)
        if wire
        else _require_python_sequence(value, name)
    )
    items: list[str] = []
    seen: set[str] = set()
    for index, raw in enumerate(sequence):
        text = (
            _normalize_scope_path(raw, f"{name}[{index}]")
            if as_paths
            else _identifier(raw, f"{name}[{index}]", required=True)
        )
        if text in seen:
            raise GuiPatchScopeError(
                f"{name} entries must be unique",
                reason_code=PatchScopeReasonCode.INVALID_PATCH_SCOPE_INPUT.value,
                details={"field": name, "duplicate": text},
            )
        seen.add(text)
        items.append(text)
    if required and not items:
        raise GuiPatchScopeError(
            f"{name} must not be empty",
            reason_code=PatchScopeReasonCode.MISSING_PROPOSAL_DECLARATION.value,
            details={"field": name},
        )
    return tuple(items)


_WINDOWS_DRIVE_RE = re.compile(r"^[a-zA-Z]:")


def _normalize_scope_path(value: Any, name: str = "path") -> str:
    try:
        normalized = _normalize_repo_path(value, name)
    except GuiAuthorityError as exc:
        raise GuiPatchScopeError(
            str(exc),
            reason_code=exc.reason_code
            or PatchScopeReasonCode.PATH_ABSOLUTE_OR_TRAVERSAL.value,
            details=exc.details,
        ) from exc
    if (
        len(normalized) > ABSOLUTE_MAX_PATH_CHARS
    ):
        raise GuiPatchScopeError(
            f"{name} exceeds the path-length cap",
            reason_code=PatchScopeReasonCode.EXCESSIVE_PATH.value,
            details={"field": name, "length": len(normalized)},
        )
    if _WINDOWS_DRIVE_RE.match(normalized) or normalized.startswith("//"):
        raise GuiPatchScopeError(
            f"{name} must be a normalized repository-relative path",
            reason_code=PatchScopeReasonCode.PATH_ABSOLUTE_OR_TRAVERSAL.value,
            details={"field": name, "value": normalized},
        )
    return normalized


def _as_operation(value: Any, *, wire: bool = False) -> PatchOperation:
    if not wire and type(value) is PatchOperation:
        return value
    if type(value) is not str:
        raise GuiPatchScopeError(
            "operation must be a string",
            reason_code=PatchScopeReasonCode.INVALID_PATCH_SCOPE_INPUT.value,
            details={"value_type": type(value).__name__},
        )
    text = _text(value, "operation")
    try:
        return PatchOperation(text)
    except ValueError as exc:
        raise GuiPatchScopeError(
            f"unknown patch operation: {text}",
            reason_code=PatchScopeReasonCode.DIFF_SEMANTICS_INVALID.value,
            details={"operation": text},
        ) from exc


def _as_change_kind(value: Any, *, wire: bool = False) -> ForbiddenChangeKind:
    if not wire and type(value) is ForbiddenChangeKind:
        return value
    if type(value) is not str:
        raise GuiPatchScopeError(
            "change_kind must be a string",
            reason_code=PatchScopeReasonCode.INVALID_PATCH_SCOPE_INPUT.value,
            details={"value_type": type(value).__name__},
        )
    text = _text(value, "change_kind")
    try:
        return ForbiddenChangeKind(text)
    except ValueError as exc:
        raise GuiPatchScopeError(
            f"unknown change kind: {text}",
            reason_code=PatchScopeReasonCode.INVALID_PATCH_SCOPE_INPUT.value,
            details={"change_kind": text},
        ) from exc


def _coerce_change_kinds(
    value: Any, *, field_name: str = "change_kinds", wire: bool = False
) -> tuple[ForbiddenChangeKind, ...]:
    if value is None:
        raise GuiPatchScopeError(
            f"{field_name} must be a JSON array; null is not a collection",
            reason_code=PatchScopeReasonCode.INVALID_COLLECTION_TYPE.value,
            details={"field": field_name, "value_type": "NoneType"},
        )
    sequence = (
        _require_json_array(value, field_name)
        if wire
        else _require_python_sequence(value, field_name)
    )
    kinds: list[ForbiddenChangeKind] = []
    seen: set[ForbiddenChangeKind] = set()
    for item in sequence:
        kind = _as_change_kind(item, wire=wire)
        if kind not in seen:
            seen.add(kind)
            kinds.append(kind)
    return tuple(kinds)


def _optional_change_kinds(
    payload: Mapping[str, Any], key: str = "change_kinds"
) -> tuple[ForbiddenChangeKind, ...]:
    if key not in payload:
        return ()
    return _coerce_change_kinds(payload[key], field_name=key, wire=True)


def _enum_or_text(value: Any, name: str) -> str:
    if value is None or value == "":
        return ""
    if hasattr(value, "value") and type(value.value) is str:
        return _text(value.value, name, required=False)
    return _text(value, name, required=False)


def _optional_digest(payload: Mapping[str, Any], key: str) -> str:
    if key not in payload:
        return ""
    _reject_present_null(payload, key)
    text = _exact_str(payload[key], key)
    if text == "":
        return ""
    if not _CANONICAL_DIGEST_RE.fullmatch(text):
        raise GuiPatchScopeError(
            f"{key} must be a canonical argument digest matching "
            "sha256:[0-9a-f]{64}",
            reason_code=AuthorityReasonCode.NONCANONICAL_ARGUMENT_DIGEST.value,
            details={"field": key},
        )
    return text


# ---------------------------------------------------------------------------
# Path / application / content classifiers
# ---------------------------------------------------------------------------


def is_test_path(path: str) -> bool:
    """Return True when ``path`` looks like a test or spec file."""
    normalized = path.replace("\\", "/")
    lowered = normalized.lower()
    if any(marker in lowered for marker in _TEST_PATH_MARKERS):
        return True
    name = PurePosixPath(normalized).name.lower()
    return name.startswith("test_") or name.endswith("_test.py")


def is_screenshot_path(path: str) -> bool:
    """Return True when ``path`` looks like a screenshot artifact."""
    lowered = path.replace("\\", "/").lower()
    return any(marker in lowered for marker in _SCREENSHOT_PATH_MARKERS)


def application_slug(application_id: str) -> str:
    """Strip a leading ``app:`` prefix from an application identity."""
    text = _identifier(application_id, "application_id")
    if ":" in text:
        prefix, remainder = text.split(":", 1)
        if prefix == "app" and remainder:
            return remainder
    return text


def path_application_slug(path: str) -> str | None:
    """Return the SwissKnife apps/ slug implied by ``path``, if any."""
    normalized = path.replace("\\", "/")
    if not normalized.startswith(_APPS_PREFIX):
        return None
    rest = normalized[len(_APPS_PREFIX) :]
    if not rest:
        return None
    first = rest.split("/", 1)[0]
    if not first:
        return None
    if "." in first:
        first = first.rsplit(".", 1)[0]
    return first or None


def path_implies_unrelated_application(path: str, application_id: str) -> bool:
    """Return True when ``path`` sits under a different apps/ identity."""
    implied = path_application_slug(path)
    if implied is None:
        return False
    slug = application_slug(application_id)
    if implied == slug:
        return False
    if implied.startswith(f"{slug}.") or implied.startswith(f"{slug}-"):
        return False
    return True


def _scan_text_blob(*parts: str) -> str:
    return "\n".join(part for part in parts if part)


def infer_change_kinds(
    *,
    path: str,
    operation: PatchOperation,
    content_markers: Sequence[str],
    diff_text: str,
) -> tuple[ForbiddenChangeKind, ...]:
    """Derive forbidden change kinds from path, operation, and content."""
    blob = _scan_text_blob(path, diff_text, *content_markers)
    lowered = blob.lower()
    found: list[ForbiddenChangeKind] = []

    def _add(kind: ForbiddenChangeKind) -> None:
        if kind not in found:
            found.append(kind)

    if any(marker in blob for marker in _HTML_MARKERS):
        _add(ForbiddenChangeKind.ARBITRARY_HTML_EXECUTION)
    if any(marker in lowered for marker in _SECURITY_DISABLE_MARKERS):
        _add(ForbiddenChangeKind.DISABLED_SECURITY_CHECK)
    if any(marker in lowered for marker in _CREDENTIAL_MARKERS):
        _add(ForbiddenChangeKind.CREDENTIALS)
    if any(marker in lowered for marker in _BACKEND_MARKERS):
        _add(ForbiddenChangeKind.BACKEND_AUTHORIZATION)
    if is_test_path(path) and (
        operation is PatchOperation.DELETE or _diff_deletes_tests(diff_text)
    ):
        _add(ForbiddenChangeKind.DELETED_TEST)
    if "action_binding" in lowered or "actionbinding" in lowered:
        _add(ForbiddenChangeKind.UNVERIFIED_ACTION_BINDING)
    return tuple(found)


def _diff_deletes_tests(diff_text: str) -> bool:
    if type(diff_text) is not str or not diff_text:
        return False
    removed_test = False
    added_test = False
    for raw_line in diff_text.splitlines():
        if not raw_line:
            continue
        prefix = raw_line[0]
        body = raw_line[1:]
        if prefix == "-" and any(marker in body for marker in _TEST_DELETION_MARKERS):
            removed_test = True
        elif prefix == "+" and any(marker in body for marker in _TEST_DELETION_MARKERS):
            added_test = True
    return removed_test and not added_test


# ---------------------------------------------------------------------------
# Unified-diff parser
# ---------------------------------------------------------------------------


def parse_unified_diff(diff_text: Any) -> tuple["PatchHunk", ...]:
    """Parse a unified diff into typed hunks.  Malformed diffs reject."""
    text = _exact_str(diff_text, "diff_text")
    if "\x00" in text:
        raise GuiPatchScopeError(
            "diff_text must not contain NUL",
            reason_code=PatchScopeReasonCode.DIFF_SEMANTICS_INVALID.value,
        )
    if not text.strip():
        raise GuiPatchScopeError(
            "diff_text must not be empty",
            reason_code=PatchScopeReasonCode.DIFF_SEMANTICS_INVALID.value,
        )
    lines = text.splitlines()
    hunks: list[PatchHunk] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("diff --git "):
            index += 1
            continue
        if line.startswith("index ") or line.startswith("similarity "):
            index += 1
            continue
        if line.startswith("rename from ") or line.startswith("rename to "):
            index += 1
            continue
        if not line.startswith("--- "):
            if line.startswith("+++ ") or line.startswith("@@"):
                raise GuiPatchScopeError(
                    "unified diff hunk is missing a --- header",
                    reason_code=PatchScopeReasonCode.DIFF_SEMANTICS_INVALID.value,
                )
            index += 1
            continue
        if index + 1 >= len(lines) or not lines[index + 1].startswith("+++ "):
            raise GuiPatchScopeError(
                "unified diff file header must be a --- / +++ pair",
                reason_code=PatchScopeReasonCode.DIFF_SEMANTICS_INVALID.value,
            )
        old_raw = _strip_diff_path(line[4:])
        new_raw = _strip_diff_path(lines[index + 1][4:])
        index += 2
        file_added = 0
        file_deleted = 0
        file_diff_lines: list[str] = [f"--- {old_raw}", f"+++ {new_raw}"]
        saw_hunk = False
        start_line = 0
        end_line = 0
        while index < len(lines):
            body = lines[index]
            if body.startswith("--- ") or body.startswith("diff --git "):
                break
            if body.startswith("@@"):
                match = _UNIFIED_HUNK_RE.match(body)
                if match is None:
                    raise GuiPatchScopeError(
                        "unified diff hunk header is malformed",
                        reason_code=PatchScopeReasonCode.DIFF_SEMANTICS_INVALID.value,
                        details={"line": body},
                    )
                saw_hunk = True
                start_line = int(match.group(3))
                plus_count = int(match.group(4) or "1")
                end_line = start_line + max(plus_count - 1, 0)
                file_diff_lines.append(body)
                index += 1
                continue
            if not body:
                file_diff_lines.append(body)
                index += 1
                continue
            prefix = body[0]
            if prefix == "+":
                file_added += 1
            elif prefix == "-":
                file_deleted += 1
            elif prefix in {" ", "\\"}:
                pass
            else:
                raise GuiPatchScopeError(
                    "unified diff line must start with ' ', '+', '-', or '\\\\'",
                    reason_code=PatchScopeReasonCode.DIFF_SEMANTICS_INVALID.value,
                    details={"line": body},
                )
            file_diff_lines.append(body)
            index += 1
        if not saw_hunk:
            raise GuiPatchScopeError(
                "unified diff file is missing hunk headers",
                reason_code=PatchScopeReasonCode.DIFF_SEMANTICS_INVALID.value,
                details={"old_path": old_raw, "new_path": new_raw},
            )
        operation, path, old_path = _classify_diff_paths(old_raw, new_raw)
        hunks.append(
            PatchHunk(
                path=path,
                operation=operation,
                added_lines=file_added,
                deleted_lines=file_deleted,
                old_path=old_path,
                start_line=start_line,
                end_line=end_line,
                diff_text="\n".join(file_diff_lines),
            )
        )
    if not hunks:
        raise GuiPatchScopeError(
            "unified diff did not contain any file hunks",
            reason_code=PatchScopeReasonCode.DIFF_SEMANTICS_INVALID.value,
        )
    return tuple(hunks)


def _strip_diff_path(raw: str) -> str:
    text = raw.strip()
    if "\t" in text:
        text = text.split("\t", 1)[0].strip()
    if text.startswith("a/") or text.startswith("b/"):
        text = text[2:]
    return text


def _classify_diff_paths(
    old_raw: str, new_raw: str
) -> tuple[PatchOperation, str, str]:
    old_is_null = old_raw in {"/dev/null", "dev/null"}
    new_is_null = new_raw in {"/dev/null", "dev/null"}
    if old_is_null and new_is_null:
        raise GuiPatchScopeError(
            "unified diff cannot use /dev/null for both sides",
            reason_code=PatchScopeReasonCode.DIFF_SEMANTICS_INVALID.value,
        )
    if old_is_null:
        return PatchOperation.CREATE, _normalize_scope_path(new_raw, "path"), ""
    if new_is_null:
        return PatchOperation.DELETE, _normalize_scope_path(old_raw, "path"), ""
    old_path = _normalize_scope_path(old_raw, "old_path")
    new_path = _normalize_scope_path(new_raw, "path")
    if old_path != new_path:
        return PatchOperation.RENAME, new_path, old_path
    return PatchOperation.MODIFY, new_path, ""


# ---------------------------------------------------------------------------
# Typed records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PatchScopeLimits:
    """Configured file/line/hunk caps.  Construction may only tighten defaults."""

    max_files: int = DEFAULT_MAX_FILES
    max_changed_lines: int = DEFAULT_MAX_CHANGED_LINES
    max_hunks: int = DEFAULT_MAX_HUNKS
    max_new_files: int = DEFAULT_MAX_NEW_FILES
    max_deleted_files: int = DEFAULT_MAX_DELETED_FILES
    max_path_chars: int = DEFAULT_MAX_PATH_CHARS

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_files", _positive_int(self.max_files, "max_files"))
        object.__setattr__(
            self,
            "max_changed_lines",
            _positive_int(self.max_changed_lines, "max_changed_lines"),
        )
        object.__setattr__(self, "max_hunks", _positive_int(self.max_hunks, "max_hunks"))
        object.__setattr__(
            self, "max_new_files", _positive_int(self.max_new_files, "max_new_files")
        )
        object.__setattr__(
            self,
            "max_deleted_files",
            _positive_int(self.max_deleted_files, "max_deleted_files"),
        )
        object.__setattr__(
            self, "max_path_chars", _positive_int(self.max_path_chars, "max_path_chars")
        )
        caps = (
            ("max_files", self.max_files, ABSOLUTE_MAX_FILES),
            ("max_changed_lines", self.max_changed_lines, ABSOLUTE_MAX_CHANGED_LINES),
            ("max_hunks", self.max_hunks, ABSOLUTE_MAX_HUNKS),
            ("max_new_files", self.max_new_files, ABSOLUTE_MAX_NEW_FILES),
            ("max_deleted_files", self.max_deleted_files, ABSOLUTE_MAX_DELETED_FILES),
            ("max_path_chars", self.max_path_chars, ABSOLUTE_MAX_PATH_CHARS),
        )
        for name, value, cap in caps:
            if value > cap:
                raise GuiPatchScopeError(
                    f"{name} cannot exceed the absolute safety cap of {cap}",
                    reason_code=PatchScopeReasonCode.INVALID_PATCH_SCOPE_INPUT.value,
                    details={"field": name, "value": value, "cap": cap},
                )

    def to_dict(self) -> dict[str, int]:
        return {
            "max_changed_lines": self.max_changed_lines,
            "max_deleted_files": self.max_deleted_files,
            "max_files": self.max_files,
            "max_hunks": self.max_hunks,
            "max_new_files": self.max_new_files,
            "max_path_chars": self.max_path_chars,
        }


@dataclass(frozen=True)
class GuiImprovementProposalView:
    """Closed gate view of ``GuiImprovementProposal@1``."""

    proposal_id: str
    application_id: str
    screen_id: str
    objective: str
    intended_file_paths: tuple[str, ...]
    intended_component_ids: tuple[str, ...]
    acceptance_criteria: tuple[str, ...]
    expected_test_ids: tuple[str, ...] = ()
    expected_screenshot_ids: tuple[str, ...] = ()
    state_effect_ids: tuple[str, ...] = ()
    visual_effect_summary: str = ""
    interface: str = GUI_IMPROVEMENT_PROPOSAL_INTERFACE
    schema_version: str = GUI_IMPROVEMENT_PROPOSAL_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "proposal_id", _identifier(self.proposal_id, "proposal_id")
        )
        object.__setattr__(
            self, "application_id", _identifier(self.application_id, "application_id")
        )
        object.__setattr__(self, "screen_id", _identifier(self.screen_id, "screen_id"))
        object.__setattr__(self, "objective", _text(self.objective, "objective"))
        object.__setattr__(
            self,
            "intended_file_paths",
            _unique_strings(
                self.intended_file_paths,
                "intended_file_paths",
                wire=False,
                required=True,
                as_paths=True,
            ),
        )
        object.__setattr__(
            self,
            "intended_component_ids",
            _unique_strings(
                self.intended_component_ids,
                "intended_component_ids",
                wire=False,
                required=True,
            ),
        )
        object.__setattr__(
            self,
            "acceptance_criteria",
            _unique_strings(
                self.acceptance_criteria,
                "acceptance_criteria",
                wire=False,
                required=True,
            ),
        )
        object.__setattr__(
            self,
            "expected_test_ids",
            _unique_strings(
                self.expected_test_ids, "expected_test_ids", wire=False
            ),
        )
        object.__setattr__(
            self,
            "expected_screenshot_ids",
            _unique_strings(
                self.expected_screenshot_ids,
                "expected_screenshot_ids",
                wire=False,
            ),
        )
        object.__setattr__(
            self,
            "state_effect_ids",
            _unique_strings(
                self.state_effect_ids, "state_effect_ids", wire=False
            ),
        )
        object.__setattr__(
            self,
            "visual_effect_summary",
            _text(self.visual_effect_summary, "visual_effect_summary", required=False),
        )
        interface = _text(self.interface, "interface")
        if interface != GUI_IMPROVEMENT_PROPOSAL_INTERFACE:
            raise GuiPatchScopeError(
                "proposal interface must be GuiImprovementProposal@1",
                reason_code=PatchScopeReasonCode.INVALID_PATCH_SCOPE_INPUT.value,
                details={"interface": interface},
            )
        schema = _text(self.schema_version, "schema_version")
        if schema != GUI_IMPROVEMENT_PROPOSAL_SCHEMA:
            raise GuiPatchScopeError(
                "proposal schema_version must be gui-improvement-proposal/v1",
                reason_code=PatchScopeReasonCode.INVALID_PATCH_SCOPE_INPUT.value,
                details={"schema_version": schema},
            )
        object.__setattr__(self, "interface", interface)
        object.__setattr__(self, "schema_version", schema)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "GuiImprovementProposalView":
        payload = _require_mapping(raw, "proposal")
        _reject_unknown(payload, _PROPOSAL_KEYS, "proposal")
        for key in (
            "proposal_id",
            "application_id",
            "screen_id",
            "objective",
            "intended_file_paths",
            "intended_component_ids",
            "acceptance_criteria",
        ):
            if key not in payload:
                raise GuiPatchScopeError(
                    f"proposal.{key} is required",
                    reason_code=PatchScopeReasonCode.MISSING_PROPOSAL_DECLARATION.value,
                    details={"field": key},
                )
        interface = payload.get("interface", GUI_IMPROVEMENT_PROPOSAL_INTERFACE)
        schema = payload.get("schema_version", GUI_IMPROVEMENT_PROPOSAL_SCHEMA)
        return cls(
            proposal_id=payload["proposal_id"],
            application_id=payload["application_id"],
            screen_id=payload["screen_id"],
            objective=payload["objective"],
            intended_file_paths=_unique_strings(
                payload["intended_file_paths"],
                "intended_file_paths",
                wire=True,
                required=True,
                as_paths=True,
            ),
            intended_component_ids=_unique_strings(
                payload["intended_component_ids"],
                "intended_component_ids",
                wire=True,
                required=True,
            ),
            acceptance_criteria=_unique_strings(
                payload["acceptance_criteria"],
                "acceptance_criteria",
                wire=True,
                required=True,
            ),
            expected_test_ids=_unique_strings(
                payload["expected_test_ids"]
                if "expected_test_ids" in payload
                else [],
                "expected_test_ids",
                wire=True,
            ),
            expected_screenshot_ids=_unique_strings(
                payload["expected_screenshot_ids"]
                if "expected_screenshot_ids" in payload
                else [],
                "expected_screenshot_ids",
                wire=True,
            ),
            state_effect_ids=_unique_strings(
                payload["state_effect_ids"] if "state_effect_ids" in payload else [],
                "state_effect_ids",
                wire=True,
            ),
            visual_effect_summary=_optional_text(payload, "visual_effect_summary"),
            interface=interface,
            schema_version=schema,
        )

    @classmethod
    def from_any(cls, value: Any) -> "GuiImprovementProposalView":
        if type(value) is cls:
            return value
        if type(value) is dict:
            return cls.from_mapping(value)
        if _looks_like_proposal(value):
            return cls(
                proposal_id=getattr(value, "proposal_id"),
                application_id=getattr(value, "application_id"),
                screen_id=getattr(value, "screen_id"),
                objective=getattr(value, "objective"),
                intended_file_paths=getattr(value, "intended_file_paths"),
                intended_component_ids=getattr(value, "intended_component_ids"),
                acceptance_criteria=getattr(value, "acceptance_criteria"),
                expected_test_ids=getattr(value, "expected_test_ids", ()),
                expected_screenshot_ids=getattr(
                    value, "expected_screenshot_ids", ()
                ),
                state_effect_ids=getattr(value, "state_effect_ids", ()),
                visual_effect_summary=getattr(value, "visual_effect_summary", ""),
                interface=getattr(
                    value, "interface", GUI_IMPROVEMENT_PROPOSAL_INTERFACE
                ),
                schema_version=getattr(
                    value, "schema_version", GUI_IMPROVEMENT_PROPOSAL_SCHEMA
                ),
            )
        raise GuiPatchScopeError(
            "proposal must be a GuiImprovementProposal@1 mapping or model",
            reason_code=PatchScopeReasonCode.INVALID_PATCH_SCOPE_INPUT.value,
            details={"value_type": type(value).__name__},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "acceptance_criteria": list(self.acceptance_criteria),
            "application_id": self.application_id,
            "expected_screenshot_ids": list(self.expected_screenshot_ids),
            "expected_test_ids": list(self.expected_test_ids),
            "intended_component_ids": list(self.intended_component_ids),
            "intended_file_paths": list(self.intended_file_paths),
            "interface": self.interface,
            "objective": self.objective,
            "proposal_id": self.proposal_id,
            "schema_version": self.schema_version,
            "screen_id": self.screen_id,
            "state_effect_ids": list(self.state_effect_ids),
            "visual_effect_summary": self.visual_effect_summary,
        }


def _looks_like_proposal(value: Any) -> bool:
    required = (
        "proposal_id",
        "application_id",
        "screen_id",
        "objective",
        "intended_file_paths",
        "intended_component_ids",
        "acceptance_criteria",
    )
    return all(hasattr(value, name) for name in required)


@dataclass(frozen=True)
class PatchHunk:
    """One observed file change presented to the scope gate."""

    path: str
    operation: PatchOperation = PatchOperation.MODIFY
    added_lines: int = 0
    deleted_lines: int = 0
    change_kinds: tuple[ForbiddenChangeKind, ...] = ()
    content_markers: tuple[str, ...] = ()
    diff_text: str = ""
    old_path: str = ""
    start_line: int = 0
    end_line: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _normalize_scope_path(self.path, "path"))
        object.__setattr__(
            self, "operation", _as_operation(self.operation, wire=False)
        )
        object.__setattr__(
            self, "added_lines", _nonneg_int(self.added_lines, "added_lines")
        )
        object.__setattr__(
            self, "deleted_lines", _nonneg_int(self.deleted_lines, "deleted_lines")
        )
        object.__setattr__(
            self,
            "change_kinds",
            _coerce_change_kinds(self.change_kinds, wire=False),
        )
        object.__setattr__(
            self,
            "content_markers",
            _unique_strings(
                self.content_markers, "content_markers", wire=False
            )
            if self.content_markers
            else (),
        )
        object.__setattr__(
            self, "diff_text", _text(self.diff_text, "diff_text", required=False)
        )
        old = self.old_path
        if old:
            object.__setattr__(
                self, "old_path", _normalize_scope_path(old, "old_path")
            )
        else:
            object.__setattr__(self, "old_path", "")
        object.__setattr__(
            self, "start_line", _nonneg_int(self.start_line, "start_line")
        )
        object.__setattr__(self, "end_line", _nonneg_int(self.end_line, "end_line"))
        self._assert_semantics()

    def _assert_semantics(self) -> None:
        if self.operation is PatchOperation.CREATE and self.deleted_lines != 0:
            raise GuiPatchScopeError(
                "create hunks must not delete lines",
                reason_code=PatchScopeReasonCode.DIFF_SEMANTICS_INVALID.value,
                details={"path": self.path},
            )
        if self.operation is PatchOperation.DELETE and self.added_lines != 0:
            raise GuiPatchScopeError(
                "delete hunks must not add lines",
                reason_code=PatchScopeReasonCode.DIFF_SEMANTICS_INVALID.value,
                details={"path": self.path},
            )
        if self.operation is PatchOperation.RENAME and not self.old_path:
            raise GuiPatchScopeError(
                "rename hunks require old_path",
                reason_code=PatchScopeReasonCode.DIFF_SEMANTICS_INVALID.value,
                details={"path": self.path},
            )
        if self.operation is not PatchOperation.RENAME and self.old_path:
            raise GuiPatchScopeError(
                "old_path is only valid on rename hunks",
                reason_code=PatchScopeReasonCode.DIFF_SEMANTICS_INVALID.value,
                details={"path": self.path},
            )
        if (
            self.start_line
            and self.end_line
            and self.end_line < self.start_line
        ):
            raise GuiPatchScopeError(
                "end_line must be >= start_line",
                reason_code=PatchScopeReasonCode.DIFF_SEMANTICS_INVALID.value,
                details={"path": self.path},
            )
        if self.added_lines == 0 and self.deleted_lines == 0:
            raise GuiPatchScopeError(
                "hunks must change at least one line",
                reason_code=PatchScopeReasonCode.DIFF_SEMANTICS_INVALID.value,
                details={"path": self.path},
            )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, index: int = 0) -> "PatchHunk":
        payload = _require_mapping(raw, f"hunks[{index}]")
        _reject_unknown(payload, _HUNK_KEYS, f"hunks[{index}]")
        if "path" not in payload:
            raise GuiPatchScopeError(
                f"hunks[{index}].path is required",
                reason_code=PatchScopeReasonCode.DIFF_SEMANTICS_INVALID.value,
            )
        _reject_present_null(payload, "path")
        markers = _optional_json_array(payload, "content_markers")
        return cls(
            path=payload["path"],
            operation=_as_operation(
                payload["operation"] if "operation" in payload else "modify",
                wire=True,
            ),
            added_lines=_nonneg_int(
                payload["added_lines"] if "added_lines" in payload else 0,
                "added_lines",
            ),
            deleted_lines=_nonneg_int(
                payload["deleted_lines"] if "deleted_lines" in payload else 0,
                "deleted_lines",
            ),
            change_kinds=_optional_change_kinds(payload),
            content_markers=(
                _unique_strings(markers, "content_markers", wire=True)
                if markers is not None
                else ()
            ),
            diff_text=_optional_text(payload, "diff_text"),
            old_path=_optional_text(payload, "old_path"),
            start_line=_nonneg_int(
                payload["start_line"] if "start_line" in payload else 0,
                "start_line",
            ),
            end_line=_nonneg_int(
                payload["end_line"] if "end_line" in payload else 0, "end_line"
            ),
        )

    @property
    def changed_lines(self) -> int:
        return self.added_lines + self.deleted_lines

    @property
    def observed_paths(self) -> tuple[str, ...]:
        if self.old_path:
            return (self.path, self.old_path)
        return (self.path,)

    def inferred_kinds(self) -> tuple[ForbiddenChangeKind, ...]:
        inferred = infer_change_kinds(
            path=self.path,
            operation=self.operation,
            content_markers=self.content_markers,
            diff_text=self.diff_text,
        )
        merged: list[ForbiddenChangeKind] = []
        for kind in (*self.change_kinds, *inferred):
            if kind not in merged:
                merged.append(kind)
        return tuple(merged)


@dataclass(frozen=True)
class PatchScopeInvalidationRecord:
    """Explicit invalidation record required before execution."""

    plan_id: str
    change_set_id: str
    reasons: tuple[str, ...]
    affected_component_ids: tuple[str, ...] = ()
    affected_scenario_ids: tuple[str, ...] = ()
    affected_check_ids: tuple[str, ...] = ()
    fallback_triggered: bool = False
    fallback_explanation: str = ""
    confidence: str = ""
    interface: str = GUI_INVALIDATION_PLAN_INTERFACE
    schema_version: str = GUI_INVALIDATION_PLAN_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan_id", _identifier(self.plan_id, "plan_id"))
        object.__setattr__(
            self, "change_set_id", _identifier(self.change_set_id, "change_set_id")
        )
        reasons = _unique_strings(self.reasons, "reasons", wire=False, required=True)
        unknown = sorted(set(reasons) - _KNOWN_INVALIDATION_REASONS)
        if unknown:
            raise GuiPatchScopeError(
                f"unknown invalidation reasons: {unknown}",
                reason_code=PatchScopeReasonCode.INVALID_PATCH_SCOPE_INPUT.value,
                details={"unknown_reasons": unknown},
            )
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(
            self,
            "affected_component_ids",
            _unique_strings(
                self.affected_component_ids, "affected_component_ids", wire=False
            ),
        )
        object.__setattr__(
            self,
            "affected_scenario_ids",
            _unique_strings(
                self.affected_scenario_ids, "affected_scenario_ids", wire=False
            ),
        )
        object.__setattr__(
            self,
            "affected_check_ids",
            _unique_strings(
                self.affected_check_ids, "affected_check_ids", wire=False
            ),
        )
        object.__setattr__(
            self,
            "fallback_triggered",
            _bool(self.fallback_triggered, "fallback_triggered"),
        )
        object.__setattr__(
            self,
            "fallback_explanation",
            _text(self.fallback_explanation, "fallback_explanation", required=False),
        )
        object.__setattr__(
            self, "confidence", _text(self.confidence, "confidence", required=False)
        )
        object.__setattr__(self, "interface", _text(self.interface, "interface"))
        object.__setattr__(
            self, "schema_version", _text(self.schema_version, "schema_version")
        )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "PatchScopeInvalidationRecord":
        payload = _require_mapping(raw, "invalidation")
        _reject_unknown(payload, _INVALIDATION_KEYS, "invalidation")
        for key in ("plan_id", "change_set_id", "reasons"):
            if key not in payload:
                raise GuiPatchScopeError(
                    f"invalidation.{key} is required",
                    reason_code=PatchScopeReasonCode.MISSING_INVALIDATION_RECORD.value,
                    details={"field": key},
                )
        return cls(
            plan_id=payload["plan_id"],
            change_set_id=payload["change_set_id"],
            reasons=_unique_strings(payload["reasons"], "reasons", wire=True, required=True),
            affected_component_ids=_unique_strings(
                payload["affected_component_ids"]
                if "affected_component_ids" in payload
                else [],
                "affected_component_ids",
                wire=True,
            ),
            affected_scenario_ids=_unique_strings(
                payload["affected_scenario_ids"]
                if "affected_scenario_ids" in payload
                else [],
                "affected_scenario_ids",
                wire=True,
            ),
            affected_check_ids=_unique_strings(
                payload["affected_check_ids"]
                if "affected_check_ids" in payload
                else [],
                "affected_check_ids",
                wire=True,
            ),
            fallback_triggered=_optional_bool(payload, "fallback_triggered", False),
            fallback_explanation=_optional_text(payload, "fallback_explanation"),
            confidence=_optional_text(payload, "confidence"),
            interface=payload.get("interface", GUI_INVALIDATION_PLAN_INTERFACE),
            schema_version=payload.get(
                "schema_version", GUI_INVALIDATION_PLAN_SCHEMA
            ),
        )

    @classmethod
    def from_any(cls, value: Any) -> "PatchScopeInvalidationRecord":
        if type(value) is cls:
            return value
        if type(value) is dict:
            return cls.from_mapping(value)
        if all(
            hasattr(value, name)
            for name in ("plan_id", "change_set_id", "reasons")
        ):
            raw_reasons = getattr(value, "reasons")
            reasons: list[str] = []
            for item in _require_python_sequence(raw_reasons, "reasons"):
                if hasattr(item, "value") and type(item.value) is str:
                    reasons.append(item.value)
                else:
                    reasons.append(_identifier(item, "reasons[]"))
            return cls(
                plan_id=getattr(value, "plan_id"),
                change_set_id=getattr(value, "change_set_id"),
                reasons=tuple(reasons),
                affected_component_ids=getattr(
                    value, "affected_component_ids", ()
                ),
                affected_scenario_ids=getattr(
                    value, "affected_scenario_ids", ()
                ),
                affected_check_ids=getattr(value, "affected_check_ids", ()),
                fallback_triggered=getattr(value, "fallback_triggered", False),
                fallback_explanation=getattr(value, "fallback_explanation", ""),
                confidence=_enum_or_text(
                    getattr(value, "confidence", ""), "confidence"
                ),
                interface=getattr(
                    value, "interface", GUI_INVALIDATION_PLAN_INTERFACE
                ),
                schema_version=getattr(
                    value, "schema_version", GUI_INVALIDATION_PLAN_SCHEMA
                ),
            )
        raise GuiPatchScopeError(
            "invalidation must be a UiInvalidationPlan mapping or model",
            reason_code=PatchScopeReasonCode.MISSING_INVALIDATION_RECORD.value,
            details={"value_type": type(value).__name__},
        )


@dataclass(frozen=True)
class PatchScopeObservation:
    """Observed patch facts.  Caller claims never override computed kinds."""

    hunks: tuple[PatchHunk, ...]
    touched_component_ids: tuple[str, ...] = ()
    touched_state_effect_ids: tuple[str, ...] = ()
    touched_test_ids: tuple[str, ...] = ()
    touched_screenshot_ids: tuple[str, ...] = ()
    application_ids: tuple[str, ...] = ()
    action_binding_ids: tuple[str, ...] = ()
    action_argument_digest: str = ""
    action_contract_evidence: tuple[AuthorityEvidence, ...] = ()
    visual_effect_observed: bool = False
    unresolved_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        hunks = tuple(
            item
            if type(item) is PatchHunk
            else PatchHunk.from_mapping(item, index=index)
            if type(item) is dict
            else (_raise_hunk_type(item, index))
            for index, item in enumerate(
                _require_python_sequence(self.hunks, "hunks")
            )
        )
        if not hunks:
            raise GuiPatchScopeError(
                "observation requires at least one hunk",
                reason_code=PatchScopeReasonCode.DIFF_SEMANTICS_INVALID.value,
            )
        object.__setattr__(self, "hunks", hunks)
        object.__setattr__(
            self,
            "touched_component_ids",
            _unique_strings(
                self.touched_component_ids, "touched_component_ids", wire=False
            ),
        )
        object.__setattr__(
            self,
            "touched_state_effect_ids",
            _unique_strings(
                self.touched_state_effect_ids,
                "touched_state_effect_ids",
                wire=False,
            ),
        )
        object.__setattr__(
            self,
            "touched_test_ids",
            _unique_strings(
                self.touched_test_ids, "touched_test_ids", wire=False
            ),
        )
        object.__setattr__(
            self,
            "touched_screenshot_ids",
            _unique_strings(
                self.touched_screenshot_ids, "touched_screenshot_ids", wire=False
            ),
        )
        object.__setattr__(
            self,
            "application_ids",
            _unique_strings(
                self.application_ids, "application_ids", wire=False
            ),
        )
        object.__setattr__(
            self,
            "action_binding_ids",
            _unique_strings(
                self.action_binding_ids, "action_binding_ids", wire=False
            ),
        )
        digest = self.action_argument_digest
        if digest:
            text = _exact_str(digest, "action_argument_digest")
            if not _CANONICAL_DIGEST_RE.fullmatch(text):
                raise GuiPatchScopeError(
                    "action_argument_digest must be sha256:[0-9a-f]{64}",
                    reason_code=AuthorityReasonCode.NONCANONICAL_ARGUMENT_DIGEST.value,
                    details={"field": "action_argument_digest"},
                )
            object.__setattr__(self, "action_argument_digest", text)
        else:
            object.__setattr__(self, "action_argument_digest", "")
        evidence_items: list[AuthorityEvidence] = []
        for index, item in enumerate(
            _require_python_sequence(
                self.action_contract_evidence, "action_contract_evidence"
            )
        ):
            if type(item) is AuthorityEvidence:
                evidence_items.append(item)
            elif type(item) is dict:
                evidence_items.append(
                    AuthorityEvidence.from_mapping(item, index=index)
                )
            else:
                raise GuiPatchScopeError(
                    "action_contract_evidence items must be AuthorityEvidence "
                    "or exact JSON objects",
                    reason_code=PatchScopeReasonCode.INVALID_AUTHORITY_EVIDENCE.value,
                    details={"value_type": type(item).__name__},
                )
        object.__setattr__(self, "action_contract_evidence", tuple(evidence_items))
        object.__setattr__(
            self,
            "visual_effect_observed",
            _bool(self.visual_effect_observed, "visual_effect_observed"),
        )
        unresolved: list[str] = []
        for index, raw in enumerate(
            _require_python_sequence(self.unresolved_paths, "unresolved_paths")
        ):
            text = _exact_str(raw, f"unresolved_paths[{index}]")
            if not text.strip():
                unresolved.append(text)
            else:
                unresolved.append(text)
        object.__setattr__(self, "unresolved_paths", tuple(unresolved))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "PatchScopeObservation":
        payload = _require_mapping(raw, "observation")
        _reject_unknown(payload, _OBSERVATION_KEYS, "observation")
        hunks_raw = _optional_json_array(payload, "hunks")
        diff_text = _optional_text(payload, "diff_text")
        hunks: tuple[PatchHunk, ...]
        if hunks_raw is None and diff_text:
            hunks = parse_unified_diff(diff_text)
        elif hunks_raw is None:
            raise GuiPatchScopeError(
                "observation.hunks or observation.diff_text is required",
                reason_code=PatchScopeReasonCode.DIFF_SEMANTICS_INVALID.value,
            )
        else:
            hunks = tuple(
                PatchHunk.from_mapping(item, index=index)
                for index, item in enumerate(hunks_raw)
            )
        evidence_raw = _optional_json_array(payload, "action_contract_evidence")
        evidence: tuple[AuthorityEvidence, ...]
        if evidence_raw is None:
            evidence = ()
        else:
            evidence = tuple(
                AuthorityEvidence.from_mapping(item, index=index)
                for index, item in enumerate(evidence_raw)
            )
        unresolved_raw = _optional_json_array(payload, "unresolved_paths")
        return cls(
            hunks=hunks,
            touched_component_ids=_unique_strings(
                payload["touched_component_ids"]
                if "touched_component_ids" in payload
                else [],
                "touched_component_ids",
                wire=True,
            ),
            touched_state_effect_ids=_unique_strings(
                payload["touched_state_effect_ids"]
                if "touched_state_effect_ids" in payload
                else [],
                "touched_state_effect_ids",
                wire=True,
            ),
            touched_test_ids=_unique_strings(
                payload["touched_test_ids"]
                if "touched_test_ids" in payload
                else [],
                "touched_test_ids",
                wire=True,
            ),
            touched_screenshot_ids=_unique_strings(
                payload["touched_screenshot_ids"]
                if "touched_screenshot_ids" in payload
                else [],
                "touched_screenshot_ids",
                wire=True,
            ),
            application_ids=_unique_strings(
                payload["application_ids"]
                if "application_ids" in payload
                else [],
                "application_ids",
                wire=True,
            ),
            action_binding_ids=_unique_strings(
                payload["action_binding_ids"]
                if "action_binding_ids" in payload
                else [],
                "action_binding_ids",
                wire=True,
            ),
            action_argument_digest=_optional_digest(
                payload, "action_argument_digest"
            ),
            action_contract_evidence=evidence,
            visual_effect_observed=_optional_bool(
                payload, "visual_effect_observed", False
            ),
            unresolved_paths=tuple(
                _exact_str(item, f"unresolved_paths[{index}]")
                for index, item in enumerate(unresolved_raw or [])
            ),
        )

    @property
    def observed_paths(self) -> tuple[str, ...]:
        paths: list[str] = []
        seen: set[str] = set()
        for hunk in self.hunks:
            for path in hunk.observed_paths:
                if path not in seen:
                    seen.add(path)
                    paths.append(path)
        return tuple(paths)

    @property
    def changed_lines(self) -> int:
        return sum(hunk.changed_lines for hunk in self.hunks)

    @property
    def computed_change_kinds(self) -> tuple[ForbiddenChangeKind, ...]:
        merged: list[ForbiddenChangeKind] = []
        for hunk in self.hunks:
            for kind in hunk.inferred_kinds():
                if kind not in merged:
                    merged.append(kind)
        return tuple(merged)


def _raise_hunk_type(item: Any, index: int) -> PatchHunk:
    raise GuiPatchScopeError(
        f"hunks[{index}] must be a PatchHunk or JSON object",
        reason_code=PatchScopeReasonCode.INVALID_COLLECTION_TYPE.value,
        details={"value_type": type(item).__name__},
    )


@dataclass(frozen=True)
class GuiPatchScopeDecision:
    """Typed, fail-closed decision for ``GuiPatchScopeDecision@1``."""

    verdict: AuthorityVerdict
    reason_codes: tuple[str, ...]
    interface: str = GUI_PATCH_SCOPE_DECISION_INTERFACE
    schema: str = GUI_PATCH_SCOPE_DECISION_SCHEMA
    message: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)
    declared_paths: tuple[str, ...] = ()
    observed_paths: tuple[str, ...] = ()
    undeclared_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.verdict) is not AuthorityVerdict:
            object.__setattr__(
                self, "verdict", AuthorityVerdict(str(self.verdict))
            )
        codes = tuple(
            sorted({_text(code, "reason_code") for code in (self.reason_codes or ())})
        )
        if not codes:
            codes = (
                (PatchScopeReasonCode.ALLOWED.value,)
                if self.verdict is AuthorityVerdict.ALLOW
                else (PatchScopeReasonCode.MISSING_PROPOSAL_DECLARATION.value,)
            )
        object.__setattr__(self, "reason_codes", codes)
        object.__setattr__(self, "interface", _text(self.interface, "interface"))
        object.__setattr__(self, "schema", _text(self.schema, "schema"))
        object.__setattr__(
            self,
            "message",
            str(self.message or "") if self.message is not None else "",
        )
        object.__setattr__(self, "details", _freeze_mapping(self.details))
        object.__setattr__(
            self,
            "declared_paths",
            tuple(_text(path, "declared_path") for path in self.declared_paths),
        )
        object.__setattr__(
            self,
            "observed_paths",
            tuple(_text(path, "observed_path") for path in self.observed_paths),
        )
        object.__setattr__(
            self,
            "undeclared_paths",
            tuple(_text(path, "undeclared_path") for path in self.undeclared_paths),
        )

    @property
    def allowed(self) -> bool:
        return self.verdict is AuthorityVerdict.ALLOW

    @property
    def rejected(self) -> bool:
        return self.verdict is AuthorityVerdict.REJECT

    @property
    def requires_human_review(self) -> bool:
        return self.verdict is AuthorityVerdict.REQUIRE_HUMAN_REVIEW

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "declared_paths": list(self.declared_paths),
            "details": dict(self.details),
            "interface": self.interface,
            "message": self.message,
            "observed_paths": list(self.observed_paths),
            "reason_codes": list(self.reason_codes),
            "rejected": self.rejected,
            "requires_human_review": self.requires_human_review,
            "schema": self.schema,
            "undeclared_paths": list(self.undeclared_paths),
            "verdict": self.verdict.value,
        }

    def as_authority_decision(self) -> AuthorityDecision:
        return AuthorityDecision(
            verdict=self.verdict,
            reason_codes=self.reason_codes,
            interface=self.interface,
            schema=self.schema,
            message=self.message,
            details=dict(self.details),
        )


def _scope_decision(
    verdict: AuthorityVerdict,
    *reason_codes: PatchScopeReasonCode | AuthorityReasonCode | str,
    message: str = "",
    details: Mapping[str, Any] | None = None,
    declared_paths: Sequence[str] = (),
    observed_paths: Sequence[str] = (),
    undeclared_paths: Sequence[str] = (),
) -> GuiPatchScopeDecision:
    codes = tuple(
        code.value
        if isinstance(code, (PatchScopeReasonCode, AuthorityReasonCode))
        else str(code)
        for code in reason_codes
    )
    return GuiPatchScopeDecision(
        verdict=verdict,
        reason_codes=codes,
        message=message,
        details=details or {},
        declared_paths=tuple(declared_paths),
        observed_paths=tuple(observed_paths),
        undeclared_paths=tuple(undeclared_paths),
    )


# ---------------------------------------------------------------------------
# GuiPatchScopeGate@1
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GuiPatchScopeGate:
    """Admit a bounded patch only when every observed mutation is in scope.

    Interface: ``GuiPatchScopeGate@1``.
    """

    authority: GuiPatchAuthority = field(default_factory=GuiPatchAuthority)
    limits: PatchScopeLimits = field(default_factory=PatchScopeLimits)
    schema: str = GUI_PATCH_SCOPE_GATE_SCHEMA
    interface: str = GUI_PATCH_SCOPE_GATE_INTERFACE
    require_invalidation: bool = True

    def __post_init__(self) -> None:
        if type(self.authority) is not GuiPatchAuthority:
            raise GuiPatchScopeError(
                "authority must be a GuiPatchAuthority",
                reason_code=PatchScopeReasonCode.INVALID_PATCH_SCOPE_INPUT.value,
                details={"value_type": type(self.authority).__name__},
            )
        if type(self.limits) is not PatchScopeLimits:
            raise GuiPatchScopeError(
                "limits must be a PatchScopeLimits",
                reason_code=PatchScopeReasonCode.INVALID_PATCH_SCOPE_INPUT.value,
                details={"value_type": type(self.limits).__name__},
            )
        object.__setattr__(self, "schema", _text(self.schema, "schema"))
        object.__setattr__(self, "interface", _text(self.interface, "interface"))
        if self.require_invalidation is not True:
            raise GuiPatchScopeError(
                "require_invalidation must be literal True; "
                "execution cannot proceed without an invalidation record",
                reason_code=PatchScopeReasonCode.INVALID_PATCH_SCOPE_INPUT.value,
                details={"field": "require_invalidation"},
            )
        object.__setattr__(self, "require_invalidation", True)

    def evaluate(
        self,
        proposal: GuiImprovementProposalView | Mapping[str, Any] | Any,
        observation: PatchScopeObservation | Mapping[str, Any],
        *,
        invalidation: PatchScopeInvalidationRecord | Mapping[str, Any] | Any | None = None,
    ) -> GuiPatchScopeDecision:
        """Evaluate a proposal against the observed patch and invalidation."""
        proposal_view = GuiImprovementProposalView.from_any(proposal)
        observation_view = (
            observation
            if type(observation) is PatchScopeObservation
            else PatchScopeObservation.from_mapping(
                _require_mapping(observation, "observation")
            )
        )
        return self._evaluate_typed(proposal_view, observation_view, invalidation)

    def evaluate_request(
        self, request: Mapping[str, Any]
    ) -> GuiPatchScopeDecision:
        """Evaluate a closed ``{proposal, observation, invalidation}`` mapping."""
        payload = _require_mapping(request, "request")
        _reject_unknown(payload, _EVALUATE_REQUEST_KEYS, "request")
        if "proposal" not in payload:
            raise GuiPatchScopeError(
                "request.proposal is required",
                reason_code=PatchScopeReasonCode.MISSING_PROPOSAL_DECLARATION.value,
            )
        if "observation" not in payload:
            raise GuiPatchScopeError(
                "request.observation is required",
                reason_code=PatchScopeReasonCode.DIFF_SEMANTICS_INVALID.value,
            )
        return self.evaluate(
            payload["proposal"],
            payload["observation"],
            invalidation=payload.get("invalidation"),
        )

    def _evaluate_typed(
        self,
        proposal: GuiImprovementProposalView,
        observation: PatchScopeObservation,
        invalidation: PatchScopeInvalidationRecord | Mapping[str, Any] | Any | None,
    ) -> GuiPatchScopeDecision:
        declared = proposal.intended_file_paths
        observed = observation.observed_paths
        undeclared = tuple(path for path in observed if path not in declared)
        reject_codes: list[str] = []
        review_codes: list[str] = []
        details: dict[str, Any] = {
            "application_id": proposal.application_id,
            "changed_lines": observation.changed_lines,
            "declared_paths": list(declared),
            "file_count": len(observed),
            "hunk_count": len(observation.hunks),
            "limits": self.limits.to_dict(),
            "observed_paths": list(observed),
            "proposal_id": proposal.proposal_id,
            "undeclared_paths": list(undeclared),
        }

        if observation.unresolved_paths:
            reject_codes.append(PatchScopeReasonCode.UNRESOLVED_PATH.value)
            details["unresolved_paths"] = list(observation.unresolved_paths)

        for path in observed:
            if len(path) > self.limits.max_path_chars:
                reject_codes.append(PatchScopeReasonCode.EXCESSIVE_PATH.value)
                details.setdefault("excessive_paths", []).append(path)
            try:
                if path_has_forbidden_segment(path):
                    reject_codes.append(PatchScopeReasonCode.GENERATED_PATH.value)
                    reject_codes.append(
                        PatchScopeReasonCode.PATH_FORBIDDEN_SEGMENT.value
                    )
                    details.setdefault("generated_paths", []).append(path)
            except GuiAuthorityError as exc:
                reject_codes.append(
                    exc.reason_code
                    or PatchScopeReasonCode.PATH_ABSOLUTE_OR_TRAVERSAL.value
                )
            path_decision = self.authority.evaluate_path(path, declared=path in declared)
            if path_decision.rejected:
                reject_codes.extend(path_decision.reason_codes)
                if (
                    AuthorityReasonCode.UNDECLARED_PATH.value
                    in path_decision.reason_codes
                ):
                    reject_codes.append(PatchScopeReasonCode.UNDECLARED_FILE.value)
                details.setdefault("path_decisions", []).append(
                    {"path": path, "reason_codes": list(path_decision.reason_codes)}
                )

        if undeclared:
            reject_codes.append(PatchScopeReasonCode.UNDECLARED_FILE.value)
            reject_codes.append(PatchScopeReasonCode.UNDECLARED_PATH.value)

        if len(observed) > self.limits.max_files:
            reject_codes.append(PatchScopeReasonCode.FILE_LIMIT_EXCEEDED.value)
        if observation.changed_lines > self.limits.max_changed_lines:
            reject_codes.append(PatchScopeReasonCode.LINE_LIMIT_EXCEEDED.value)
        if len(observation.hunks) > self.limits.max_hunks:
            reject_codes.append(PatchScopeReasonCode.HUNK_LIMIT_EXCEEDED.value)
        created = sum(
            1
            for hunk in observation.hunks
            if hunk.operation is PatchOperation.CREATE
        )
        deleted = sum(
            1
            for hunk in observation.hunks
            if hunk.operation is PatchOperation.DELETE
        )
        if created > self.limits.max_new_files:
            reject_codes.append(PatchScopeReasonCode.FILE_LIMIT_EXCEEDED.value)
            details["created_files"] = created
        if deleted > self.limits.max_deleted_files:
            reject_codes.append(PatchScopeReasonCode.FILE_LIMIT_EXCEEDED.value)
            details["deleted_files"] = deleted

        foreign_apps = [
            app_id
            for app_id in observation.application_ids
            if app_id != proposal.application_id
        ]
        unrelated_paths = [
            path
            for path in observed
            if path_implies_unrelated_application(path, proposal.application_id)
        ]
        if foreign_apps or unrelated_paths:
            reject_codes.append(PatchScopeReasonCode.UNRELATED_APPLICATION.value)
            details["foreign_application_ids"] = foreign_apps
            details["unrelated_paths"] = unrelated_paths

        undeclared_components = [
            item
            for item in observation.touched_component_ids
            if item not in proposal.intended_component_ids
        ]
        if undeclared_components:
            reject_codes.append(PatchScopeReasonCode.UNDECLARED_COMPONENT.value)
            details["undeclared_components"] = undeclared_components

        undeclared_state = [
            item
            for item in observation.touched_state_effect_ids
            if item not in proposal.state_effect_ids
        ]
        if undeclared_state:
            reject_codes.append(PatchScopeReasonCode.UNDECLARED_STATE_EFFECT.value)
            details["undeclared_state_effect_ids"] = undeclared_state

        test_file_touched = any(is_test_path(path) for path in observed)
        undeclared_tests = [
            item
            for item in observation.touched_test_ids
            if item not in proposal.expected_test_ids
        ]
        if undeclared_tests or (
            test_file_touched and not proposal.expected_test_ids
        ):
            reject_codes.append(PatchScopeReasonCode.UNDECLARED_TEST.value)
            details["undeclared_test_ids"] = undeclared_tests

        screenshot_touched = any(is_screenshot_path(path) for path in observed)
        undeclared_screenshots = [
            item
            for item in observation.touched_screenshot_ids
            if item not in proposal.expected_screenshot_ids
        ]
        if undeclared_screenshots or (
            screenshot_touched and not proposal.expected_screenshot_ids
        ):
            reject_codes.append(PatchScopeReasonCode.UNDECLARED_SCREENSHOT.value)
            details["undeclared_screenshot_ids"] = undeclared_screenshots

        if (
            observation.visual_effect_observed or screenshot_touched
        ) and not proposal.visual_effect_summary:
            reject_codes.append(PatchScopeReasonCode.UNDECLARED_VISUAL_EFFECT.value)

        kinds = list(observation.computed_change_kinds)
        if undeclared:
            kinds.append(ForbiddenChangeKind.OUT_OF_SCOPE_PATH)
        if foreign_apps or unrelated_paths:
            kinds.append(ForbiddenChangeKind.UNRELATED_APPLICATION)
        unique_kinds: list[ForbiddenChangeKind] = []
        for kind in kinds:
            if kind not in unique_kinds:
                unique_kinds.append(kind)
        details["change_kinds"] = [kind.value for kind in unique_kinds]

        for kind in unique_kinds:
            if kind in _SCOPE_REJECT_KINDS:
                reject_codes.append(kind.value)
                reject_codes.append(PatchScopeReasonCode.FORBIDDEN_MUTATION.value)
            elif kind in _SCOPE_REVIEW_KINDS:
                review_codes.append(kind.value)

        binding_active = bool(observation.action_binding_ids) or any(
            kind is ForbiddenChangeKind.UNVERIFIED_ACTION_BINDING
            for kind in unique_kinds
        )
        if binding_active:
            binding_codes = self._binding_reason_codes(observation)
            if not binding_codes:
                review_codes = [
                    code
                    for code in review_codes
                    if code
                    != PatchScopeReasonCode.UNVERIFIED_ACTION_BINDING.value
                ]
            elif (
                PatchScopeReasonCode.UNVERIFIED_ACTION_BINDING.value in binding_codes
                or PatchScopeReasonCode.MISSING_ACTION_CONTRACT_EVIDENCE.value
                in binding_codes
            ):
                review_codes.extend(binding_codes)
            else:
                reject_codes.extend(binding_codes)

        if invalidation is None:
            reject_codes.append(
                PatchScopeReasonCode.MISSING_INVALIDATION_RECORD.value
            )
        else:
            try:
                record = PatchScopeInvalidationRecord.from_any(invalidation)
            except GuiPatchScopeError as exc:
                reject_codes.append(
                    exc.reason_code
                    or PatchScopeReasonCode.MISSING_INVALIDATION_RECORD.value
                )
                details["invalidation_error"] = str(exc)
            else:
                details["invalidation_plan_id"] = record.plan_id
                covered = set(record.affected_component_ids)
                needed = set(observation.touched_component_ids) | set(
                    proposal.intended_component_ids
                )
                if needed and not record.fallback_triggered and not needed.issubset(
                    covered
                ):
                    reject_codes.append(
                        PatchScopeReasonCode.INVALIDATION_COVERAGE_GAP.value
                    )
                    details["uncovered_component_ids"] = sorted(needed - covered)

        reject_codes = _unique_preserve(reject_codes)
        review_codes = [
            code
            for code in _unique_preserve(review_codes)
            if code not in reject_codes
        ]
        if reject_codes:
            return _scope_decision(
                AuthorityVerdict.REJECT,
                *reject_codes,
                *review_codes,
                message="patch is outside the declared execution scope",
                details=details,
                declared_paths=declared,
                observed_paths=observed,
                undeclared_paths=undeclared,
            )
        if review_codes:
            if any(
                code
                in {
                    PatchScopeReasonCode.BACKEND_AUTHORIZATION.value,
                    PatchScopeReasonCode.CREDENTIALS.value,
                    PatchScopeReasonCode.DISABLED_SECURITY_CHECK.value,
                }
                for code in review_codes
            ):
                review_codes.append(
                    PatchScopeReasonCode.SENSITIVE_CHANGE_REQUIRES_REVIEW.value
                )
            if any(
                code
                in {
                    PatchScopeReasonCode.DELETED_TEST.value,
                    PatchScopeReasonCode.UNVERIFIED_ACTION_BINDING.value,
                }
                for code in review_codes
            ):
                review_codes.append(
                    PatchScopeReasonCode.SENSITIVE_CHANGE_REQUIRES_CONTRACT.value
                )
            return _scope_decision(
                AuthorityVerdict.REQUIRE_HUMAN_REVIEW,
                *review_codes,
                message="patch requires human review before execution",
                details=details,
                declared_paths=declared,
                observed_paths=observed,
                undeclared_paths=undeclared,
            )
        return _scope_decision(
            AuthorityVerdict.ALLOW,
            PatchScopeReasonCode.ALLOWED,
            message="patch is within the declared execution scope",
            details=details,
            declared_paths=declared,
            observed_paths=observed,
            undeclared_paths=undeclared,
        )

    def _binding_reason_codes(
        self, observation: PatchScopeObservation
    ) -> list[str]:
        codes: list[str] = []
        evidence = observation.action_contract_evidence
        if not evidence:
            codes.append(PatchScopeReasonCode.MISSING_ACTION_CONTRACT_EVIDENCE.value)
            codes.append(PatchScopeReasonCode.UNVERIFIED_ACTION_BINDING.value)
            return codes
        authorizing: list[AuthorityEvidence] = []
        for item in evidence:
            if not item.valid:
                codes.append(PatchScopeReasonCode.INVALID_AUTHORITY_EVIDENCE.value)
                continue
            if item.kind is AuthorityEvidenceKind.SCOPE_DECLARATION:
                codes.append(
                    PatchScopeReasonCode.SCOPE_DECLARATION_NOT_AUTHORITY.value
                )
                continue
            if item.kind not in HOST_AUTHORIZING_EVIDENCE_KINDS:
                codes.append(PatchScopeReasonCode.INVALID_AUTHORITY_EVIDENCE.value)
                continue
            authorizing.append(item)
        if not authorizing:
            codes.append(PatchScopeReasonCode.MISSING_ACTION_CONTRACT_EVIDENCE.value)
            codes.append(PatchScopeReasonCode.UNVERIFIED_ACTION_BINDING.value)
            return codes
        expected_actions = set(observation.action_binding_ids)
        bound = False
        for item in authorizing:
            if expected_actions and item.binds_action_id not in expected_actions:
                codes.append(PatchScopeReasonCode.EVIDENCE_BINDING_MISMATCH.value)
                continue
            if observation.action_argument_digest:
                if item.binds_argument_digest != observation.action_argument_digest:
                    codes.append(
                        PatchScopeReasonCode.EVIDENCE_BINDING_MISMATCH.value
                    )
                    continue
            elif item.binds_argument_digest:
                # Evidence that claims a digest must match a provided digest.
                codes.append(PatchScopeReasonCode.EVIDENCE_BINDING_MISMATCH.value)
                continue
            bound = True
        if bound:
            return []
        codes.append(PatchScopeReasonCode.UNVERIFIED_ACTION_BINDING.value)
        if PatchScopeReasonCode.MISSING_ACTION_CONTRACT_EVIDENCE.value not in codes:
            codes.append(PatchScopeReasonCode.MISSING_ACTION_CONTRACT_EVIDENCE.value)
        return _unique_preserve(codes)


def _unique_preserve(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in values:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def default_patch_scope_limits() -> PatchScopeLimits:
    """Return the sealed default file/line/hunk caps."""
    return PatchScopeLimits()


def default_patch_scope_gate() -> GuiPatchScopeGate:
    """Return a fail-closed gate with default authority and limits."""
    return GuiPatchScopeGate()


__all__ = (
    "ABSOLUTE_MAX_CHANGED_LINES",
    "ABSOLUTE_MAX_DELETED_FILES",
    "ABSOLUTE_MAX_FILES",
    "ABSOLUTE_MAX_HUNKS",
    "ABSOLUTE_MAX_NEW_FILES",
    "ABSOLUTE_MAX_PATH_CHARS",
    "ALWAYS_HUMAN_REVIEW_KINDS",
    "DEFAULT_MAX_CHANGED_LINES",
    "DEFAULT_MAX_DELETED_FILES",
    "DEFAULT_MAX_FILES",
    "DEFAULT_MAX_HUNKS",
    "DEFAULT_MAX_NEW_FILES",
    "DEFAULT_MAX_PATH_CHARS",
    "GUI_IMPROVEMENT_PROPOSAL_INTERFACE",
    "GUI_IMPROVEMENT_PROPOSAL_SCHEMA",
    "GUI_PATCH_SCOPE_DECISION_INTERFACE",
    "GUI_PATCH_SCOPE_DECISION_SCHEMA",
    "GUI_PATCH_SCOPE_GATE_INTERFACE",
    "GUI_PATCH_SCOPE_GATE_SCHEMA",
    "GuiImprovementProposalView",
    "GuiPatchScopeDecision",
    "GuiPatchScopeError",
    "GuiPatchScopeGate",
    "PatchHunk",
    "PatchOperation",
    "PatchScopeInvalidationRecord",
    "PatchScopeLimits",
    "PatchScopeObservation",
    "PatchScopeReasonCode",
    "SENSITIVE_CHANGE_KINDS",
    "application_slug",
    "default_patch_scope_gate",
    "default_patch_scope_limits",
    "infer_change_kinds",
    "is_screenshot_path",
    "is_test_path",
    "parse_unified_diff",
    "path_application_slug",
    "path_implies_unrelated_application",
)
